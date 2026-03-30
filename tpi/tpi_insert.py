#!/usr/bin/env python3
"""
tpi_insert.py — Synthesise a Verilog RTL file with Yosys then use the Qwen3
LLM to insert Test Points (control + observation) into the gate-level netlist.

Usage:
    python tpi_insert.py design.v
    python tpi_insert.py design.v -o design_tpi.v --top mymodule
    python tpi_insert.py design.v --no-synth        # input is already gate-level
    python tpi_insert.py design.v --url http://localhost:8000/v1
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI


# --------------------------------------------------------------------------- #
#  Yosys synthesis                                                             #
# --------------------------------------------------------------------------- #

_YOSYS_SCRIPT = """\
read_verilog {src}
{hier}
synth {top_flag}
techmap
opt
clean
write_verilog -noattr {dst}
"""


def synthesize(src: Path, dst: Path, top: str | None) -> None:
    """Run Yosys to convert RTL Verilog to a gate-level netlist."""
    script = _YOSYS_SCRIPT.format(
        src=src,
        dst=dst,
        hier=f"hierarchy -top {top}" if top else "",
        top_flag=f"-top {top}" if top else "",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as f:
        f.write(script)
        ys = f.name
    try:
        r = subprocess.run(["yosys", "-q", ys], capture_output=True, text=True)
    finally:
        os.unlink(ys)
    if r.returncode != 0:
        sys.exit(f"Yosys synthesis failed:\n{r.stderr}")


# --------------------------------------------------------------------------- #
#  LLM prompt + call                                                           #
# --------------------------------------------------------------------------- #

_PROMPT = """\
You are a VLSI Design-for-Testability (DFT) expert specialising in Test Point
Insertion (TPI) for stuck-at fault testing.

## Goal
Modify the gate-level Verilog netlist below to:
  1. Maximise stuck-at fault coverage (detect as many SA0/SA1 faults as possible)
  2. Minimise the number of test patterns required to achieve full coverage

## Test Point Types

**Control Point (CP)** — lets a hard-to-control net be forced to 0 or 1 during test.
Add a shared `TEST_MODE` primary input (only once) and a new primary input `CP_<net>`:
    assign <net> = TEST_MODE ? CP_<net> : <original_driver>;

**Observation Point (OP)** — exposes a hard-to-observe internal net at a new output:
    output OP_<net>;
    assign OP_<net> = <internal_net>;

## Selection Strategy (SCOAP heuristics)
- **Control points**: target nets with high CC0+CC1 (hardest to control).
  Good candidates: deep AND/OR chains, fanout stems feeding many gates,
  flip-flop data inputs, feedback paths.
- **Observation points**: target nets with high CO (hardest to observe).
  Good candidates: internal signals with many fanout branches, nodes far
  from any primary output.
- Insert 4–8 control points and 4–8 observation points.
- Do NOT add test points on existing primary inputs or primary outputs.

## Output Format
Return ONLY the complete, syntactically valid, modified Verilog netlist.
No explanation, no markdown fences, no comments about what changed.

## Netlist
{netlist}
"""


def _strip_fences(text: str) -> str:
    """Remove ```verilog / ``` wrappers if the model added them."""
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$",           "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def insert_test_points(netlist: str, client: OpenAI, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PROMPT.format(netlist=netlist)}],
        temperature=0.1,
        max_tokens=16384,
    )
    return _strip_fences(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input",  metavar="INPUT.v",
                    help="RTL or gate-level Verilog source file")
    ap.add_argument("-o", "--output", metavar="OUTPUT.v",
                    help="Path for TPI output netlist (default: <input>_tpi.v)")
    ap.add_argument("--top",      metavar="MODULE",
                    help="Top-level module name (Yosys auto-detects if omitted)")
    ap.add_argument("--no-synth", action="store_true",
                    help="Skip Yosys synthesis — input is already gate-level")
    ap.add_argument("--url",   default="http://localhost:8000/v1", metavar="URL",
                    help="OpenAI-compatible server base URL  [%(default)s]")
    ap.add_argument("--model", default="Qwen/Qwen3-14B",           metavar="ID",
                    help="Model ID served at --url           [%(default)s]")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Error: {src} not found")
    out = Path(args.output) if args.output else src.with_stem(src.stem + "_tpi")

    # ── Step 1: synthesis ────────────────────────────────────────────────────
    if args.no_synth:
        gate_v = src
        print(f"[1/2] Skipping synthesis — using {gate_v}")
    else:
        gate_v = src.with_stem(src.stem + "_synth")
        print(f"[1/2] Synthesising {src}  →  {gate_v}")
        synthesize(src, gate_v, args.top)
        n_lines = len(gate_v.read_text().splitlines())
        print(f"      Gate-level netlist: {n_lines} lines")

    # ── Step 2: LLM TPI ──────────────────────────────────────────────────────
    netlist_text = gate_v.read_text()
    print(f"[2/2] Sending to {args.url}  (model: {args.model}) …")
    client = OpenAI(base_url=args.url, api_key="unused")
    result = insert_test_points(netlist_text, client, args.model)

    out.write_text(result + "\n")
    print(f"\nTPI netlist written to: {out}")


if __name__ == "__main__":
    main()
