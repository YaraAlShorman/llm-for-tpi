#!/usr/bin/env python3
"""
tpi_insert.py — Synthesise a Verilog RTL file with Yosys then use the Qwen3
LLM to insert Test Points (control + observation) into the gate-level netlist.

The LLM is given a compact circuit summary (nets + fanout/depth stats), not the
full netlist, so the approach works even for large designs.  Test point wiring
is applied programmatically from the LLM's selections.

Usage:
    python tpi_insert.py design.v
    python tpi_insert.py design.v -o design_tpi.v --top mymodule
    python tpi_insert.py design.v --no-synth        # input is already gate-level
    python tpi_insert.py design.v --url http://localhost:8000/v1
    python tpi_insert.py design.v --cp 6 --op 6     # choose number of test points
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
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
#  Netlist parser — extract structure for summary                              #
# --------------------------------------------------------------------------- #

def _strip_comments(src: str) -> str:
    src = re.sub(r"//[^\n]*",  "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def _parse_decl(keyword: str, src: str) -> list[str]:
    results: list[str] = []
    for m in re.finditer(
        rf"\b{keyword}\b(?:\s+\w+)?(?:\s*\[\d+:\d+\])?\s*([\w\s,]+?)\s*;", src
    ):
        for sig in re.split(r"\s*,\s*", m.group(1)):
            sig = sig.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", sig):
                results.append(sig)
    return list(dict.fromkeys(results))


def _parse_regs(src: str) -> list[str]:
    """
    Extract all reg names, including escaped Verilog identifiers like \\d_reg[1].
    These represent flip-flop state elements — their current values are
    depth-0 sources for the combinational cones between clock edges.
    """
    results: list[str] = []
    # Standard:  reg [N:0] name;  or  reg name;
    for m in re.finditer(r"\breg\b(?:\s*\[\d+:\d+\])?\s+(\w+)\s*;", src):
        results.append(m.group(1))
    # Escaped:   reg \name[idx] ;
    for m in re.finditer(r"\breg\b\s+(\\[\w\[\]]+)\s*;", src):
        results.append(m.group(1))
    return list(dict.fromkeys(results))


def _parse_always_edges(src: str) -> list[tuple[str, str, list[str]]]:
    """
    Extract D-input edges from always @(posedge/negedge ...) blocks.

    Yosys emits sequential FFs as:
        always @(posedge clk, posedge reset)
            if (reset) reg <= 0;
            else [if (en)] reg <= net;

    We extract the non-reset (else) assignments to build edges:
        (BUF/MUX, reg_name, [driving_net(s)])

    This lets the combinational depth propagate through flip-flop boundaries.
    """
    edges: list[tuple[str, str, list[str]]] = []
    idx = 0

    # Match full always blocks — both single-statement and begin/end forms
    block_pat = re.compile(
        r"always\s*@\s*\([^)]*(?:posedge|negedge)[^)]*\)\s*(.*?)(?=always\s*@|endmodule|\Z)",
        re.DOTALL,
    )
    for bm in block_pat.finditer(src):
        body = bm.group(1)

        # Find "else [if (...)] <lhs> <= <rhs>;" assignments
        # We skip the reset/init "if" branch and only take the else branches.
        for m in re.finditer(
            r"\belse\b(?:\s*if\s*\([^)]*\))?\s+"
            r"(\\?[\w\[\]]+)\s*<=\s*([^;]+);",
            body,
        ):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()

            # Normalise escaped identifier
            lhs = lhs.lstrip("\\")

            # Strip bit-select from LHS (e.g. IR[0] → IR)
            lhs_base = re.sub(r"\[\d+\]$", "", lhs)

            # Simple net reference
            rhs_name = re.sub(r"\[\d+\]$", "", rhs.strip().lstrip("\\"))
            if re.fullmatch(r"\w+", rhs_name):
                edges.append(("DFF_D", lhs_base, [rhs_name]))
            # Conditional:  cond ? a : b
            elif m2 := re.fullmatch(r"(\w+)\s*\?\s*(\w+)\s*:\s*(\w+)", rhs.strip()):
                edges.append(("DFF_MUX", lhs_base,
                               [m2.group(3), m2.group(2), m2.group(1)]))
            idx += 1

    return edges


_YOSYS_CELLS = {
    "$_AND_":  (["A", "B"],       "Y"),
    "$_OR_":   (["A", "B"],       "Y"),
    "$_NOT_":  (["A"],            "Y"),
    "$_NAND_": (["A", "B"],       "Y"),
    "$_NOR_":  (["A", "B"],       "Y"),
    "$_XOR_":  (["A", "B"],       "Y"),
    "$_XNOR_": (["A", "B"],       "Y"),
    "$_BUF_":  (["A"],            "Y"),
    "$_MUX_":  (["A", "B", "S"], "Y"),
}

_PRIMITIVES = {"and", "or", "not", "buf", "nand", "nor", "xor", "xnor"}


def _named_ports(body: str) -> dict[str, str]:
    return {m.group(1): m.group(2)
            for m in re.finditer(r"\.(\w+)\s*\(\s*(\w+)\s*\)", body)}


class _NetInfo:
    def __init__(self) -> None:
        self.fanout:    int = 0    # number of gates that read this net
        self.depth:     int = 0    # longest combinational path from a PI
        self.driven_by: str = ""   # gate type that drives this net
        # Approximate SCOAP controllability difficulty:
        #   cc0_hard=True  → this net is hard to set to 0 (high CC0)
        #   cc1_hard=True  → this net is hard to set to 1 (high CC1)
        self.cc0_hard:  bool = False
        self.cc1_hard:  bool = False


def analyze_netlist(src: str) -> tuple[list[str], list[str], dict[str, _NetInfo]]:
    """
    Parse a gate-level (or post-synthesis behavioral) Verilog netlist and return:
      (primary_inputs, primary_outputs, {net_name: _NetInfo})

    Handles both:
    - Structural: Yosys $_AND_ / $_DFF_P_ cells and Verilog primitives
    - Behavioral sequential: always @(posedge clk) blocks emitted by Yosys
      for flip-flops, where regs are treated as depth-0 combinational sources
      (equivalent to DFF Q outputs).
    """
    src    = _strip_comments(src)
    pis    = _parse_decl("input",  src)
    pos    = _parse_decl("output", src)
    regs   = _parse_regs(src)          # flip-flop state elements
    pi_set = set(pis)
    po_set = set(pos)
    # Regs are depth-0 sources — they represent DFF Q outputs at the start
    # of each combinational cone (i.e. the "current state" is a given).
    roots  = pi_set | set(regs)

    # Collect (driver_type, output_net, [input_nets]) for every gate/assignment
    edges: list[tuple[str, str, list[str]]] = []

    # Structural Yosys cells
    for cell, (in_ports, out_port) in _YOSYS_CELLS.items():
        for m in re.finditer(rf"{re.escape(cell)}\s+\w+\s*\(([^;]+)\)", src):
            pm  = _named_ports(m.group(1))
            out = pm.get(out_port)
            ins = [pm[p] for p in in_ports if p in pm]
            if out and ins:
                edges.append((cell.strip("$_"), out, ins))

    # Verilog gate primitives (positional)
    prim_pat = "|".join(re.escape(k) for k in _PRIMITIVES)
    for m in re.finditer(rf"\b({prim_pat})\b\s+(?:\w+\s*)?\(([^;]+)\)", src):
        nets_list = [s.strip() for s in m.group(2).split(",")]
        if len(nets_list) >= 2:
            edges.append((m.group(1).upper(), nets_list[0], nets_list[1:]))

    # Combinational assign statements
    for m in re.finditer(r"\bassign\b\s+(\w+)\s*=\s*([^;]+);", src):
        lhs = m.group(1)
        rhs = m.group(2).strip()
        if re.fullmatch(r"\w+", rhs):
            edges.append(("BUF", lhs, [rhs]))
        elif re.fullmatch(r"[~!]\s*\w+", rhs):
            inp = re.search(r"\w+", rhs).group()
            edges.append(("NOT", lhs, [inp]))
        elif m2 := re.fullmatch(r"(\w+)\s*\?\s*(\w+)\s*:\s*(\w+)", rhs):
            edges.append(("MUX", lhs, [m2.group(3), m2.group(2), m2.group(1)]))

    # Sequential always blocks → D-input edges (reg treated as both source and sink)
    edges.extend(_parse_always_edges(src))

    # Build net info
    net_info: dict[str, _NetInfo] = {}
    for r in roots:
        n = _NetInfo(); n.depth = 0
        n.driven_by = "PI" if r in pi_set else "DFF"
        net_info[r] = n
    for gtype, out, ins in edges:
        if out not in net_info:
            net_info[out] = _NetInfo()
        net_info[out].driven_by = gtype
        for inp in ins:
            if inp not in net_info:
                net_info[inp] = _NetInfo()
            net_info[inp].fanout += 1

    # Topological combinational depth from roots
    # DFF_D/DFF_MUX edges feed back into regs which are roots → no cycles
    driven_by_edge: dict[str, list[str]] = {out: ins for _, out, ins in edges}
    memo: dict[str, int] = {r: 0 for r in roots}

    def depth(net: str) -> int:
        if net in memo:
            return memo[net]
        memo[net] = 0   # cycle guard (shouldn't happen in acyclic comb logic)
        ins = driven_by_edge.get(net, [])
        # Don't recurse through roots — they are fixed depth-0 sources
        d = (max((depth(i) for i in ins if i not in roots), default=-1) + 1) if ins else 0
        memo[net] = d
        return d

    for net in list(net_info.keys()):
        net_info[net].depth = depth(net)

    # Approximate CC0/CC1 difficulty from the driving gate type.
    #
    # SCOAP rules (simplified):
    #   AND  output: CC0 = min(inputs CC0)+1  → easy to set 0, HARD to set 1
    #   OR   output: CC1 = min(inputs CC1)+1  → easy to set 1, HARD to set 0
    #   NAND output: CC1 = min(inputs CC0)+1  → hard to set 0 (same as AND CC1 inverted)
    #   NOR  output: CC0 = min(inputs CC1)+1  → hard to set 1 (same as OR CC0 inverted)
    #   NOT/BUF: difficulty transfers from input
    #
    # We use gate type + depth as a proxy:
    #   deep AND/NAND output → hard to set to 1 (cc1_hard)
    #   deep OR/NOR   output → hard to set to 0 (cc0_hard)
    _CC1_HARD_GATES = {"AND", "AND_", "NAND", "NAND_"}
    _CC0_HARD_GATES = {"OR",  "OR_",  "NOR",  "NOR_"}
    depth_threshold = max((info.depth for info in net_info.values()), default=0) * 0.4

    for net, info in net_info.items():
        g = info.driven_by.upper().strip("$_")
        if g in _CC1_HARD_GATES and info.depth >= depth_threshold:
            info.cc1_hard = True
        elif g in _CC0_HARD_GATES and info.depth >= depth_threshold:
            info.cc0_hard = True
        else:
            # Fall back to depth alone: deep nets are harder to control either way
            if info.depth >= depth_threshold:
                info.cc0_hard = True
                info.cc1_hard = True

    # Remove PIs, POs, and regs from candidate set
    # (regs are state elements — test points go on combinational nets)
    for p in list(pi_set | po_set | set(regs)):
        net_info.pop(p, None)

    return pis, pos, net_info


# --------------------------------------------------------------------------- #
#  LLM: ask for net selections                                                 #
# --------------------------------------------------------------------------- #

_SELECTION_PROMPT = """\
You are a VLSI DFT expert. Your task is Test Point Insertion (TPI) for stuck-at
fault testing of a synthesised gate-level netlist.

## Background
- **CP0 (Control-0 Point)**: inserted on a net that is hard to set to logic 0
  (high SCOAP CC0). An AND gate is added so TEST_MODE=1 forces the net to 0.
  Best candidates: outputs of deep OR/NOR chains (require all inputs = 0 to
  propagate 0 through fan-in).
- **CP1 (Control-1 Point)**: inserted on a net that is hard to set to logic 1
  (high SCOAP CC1). An OR gate is added so TEST_MODE=1 forces the net to 1.
  Best candidates: outputs of deep AND/NAND chains (require all inputs = 1).
- **OP (Observation Point)**: exposes a hard-to-observe internal net as a new
  primary output. Best candidates: high-fanout nodes or nodes deep in the cone
  of logic far from any primary output (high SCOAP CO).

## Circuit statistics
- Primary inputs  : {n_pi}
- Primary outputs : {n_po}
- Internal nets   : {n_nets}

## Full gate-level netlist
```verilog
{netlist}
```

## Instructions
- Analyse the netlist above to identify the best test point locations.
- {tp_instruction}
- A net may not appear in more than one category.
- Prefer internal wire/net names that are hard to control or observe based on
  the circuit topology (deep logic cones, high fanout, outputs of AND/OR chains).
- Do NOT select primary inputs or primary outputs.

## Response format
Respond with valid JSON only — no explanation, no markdown fences:
{{
  "cp0": ["net_a", "net_b", ...],
  "cp1": ["net_c", "net_d", ...],
  "observation_points": ["net_e", "net_f", ...]
}}
"""



def select_test_points(
    pis: list[str],
    pos: list[str],
    nets: dict[str, _NetInfo],
    client: OpenAI,
    model: str,
    n_cp: int,
    n_op: int,
    netlist_text: str = "",
    n_tp: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Returns (cp0_nets, cp1_nets, op_nets)."""
    if n_tp is not None:
        tp_instruction = (
            f"Choose exactly {n_tp} test points in total across CP0, CP1, and OP. "
            f"Decide the best split between the three categories based on the circuit."
        )
    else:
        n_cp0 = n_cp // 2
        n_cp1 = n_cp - n_cp0
        tp_instruction = (
            f"Choose exactly {n_cp0} nets for CP0, {n_cp1} nets for CP1, and {n_op} nets for OP."
        )

    prompt = _SELECTION_PROMPT.format(
        n_pi=len(pis),
        n_po=len(pos),
        n_nets=len(nets),
        netlist=netlist_text,
        tp_instruction=tp_instruction,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip Qwen3 thinking block
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown fences
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$",           "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        cp0s = data["cp0"]
        cp1s = data["cp1"]
        ops  = data.get("observation_points") or data.get("observation") or []
    except Exception:
        sys.exit(f"Failed to parse LLM JSON response:\n{raw}")

    valid_nets = set(nets.keys())
    cp0s = [n for n in cp0s if n in valid_nets]
    cp1s = [n for n in cp1s if n in valid_nets and n not in cp0s]
    ops  = [n for n in ops  if n in valid_nets and n not in cp0s and n not in cp1s]
    return cp0s, cp1s, ops


# --------------------------------------------------------------------------- #
#  Apply test points to the netlist text                                       #
# --------------------------------------------------------------------------- #

def apply_test_points(
    netlist: str,
    cp0s: list[str],
    cp1s: list[str],
    ops: list[str],
) -> str:
    """
    Surgically modify the netlist Verilog text to wire in the selected
    control and observation points.

    CP0 wiring (AND-based): forces net to 0 in test mode
        wire _pre_<net> = <original driver>;
        assign <net> = _pre_<net> & ~TEST_MODE;
        // when TEST_MODE=1 → net=0 regardless of logic

    CP1 wiring (OR-based): forces net to 1 in test mode
        wire _pre_<net> = <original driver>;
        assign <net> = _pre_<net> | TEST_MODE;
        // when TEST_MODE=1 → net=1 regardless of logic

    OP wiring: taps internal net to a new primary output
        output OP_<net>;
        assign OP_<net> = <net>;
    """
    mod_match = re.search(r"module\s+\w+\s*(?:\([^)]*\))?\s*;", netlist, re.DOTALL)
    if not mod_match:
        sys.exit("Could not locate module declaration in netlist.")

    new_ports: list[str] = []
    new_decls: list[str] = []
    new_assigns: list[str] = []

    if cp0s or cp1s:
        new_ports.append("    input TEST_MODE")
        new_decls.append("input TEST_MODE;")

    for net in cp0s:
        new_assigns.append(
            f"wire _pre_{net} = {net};\n"
            f"assign {net} = _pre_{net} & ~TEST_MODE;"
            f"  // CP0: TEST_MODE=1 forces {net} to 0"
        )

    for net in cp1s:
        new_assigns.append(
            f"wire _pre_{net} = {net};\n"
            f"assign {net} = _pre_{net} | TEST_MODE;"
            f"  // CP1: TEST_MODE=1 forces {net} to 1"
        )

    for net in ops:
        op_port = f"OP_{net}"
        new_ports.append(f"    output {op_port}")
        new_decls.append(f"output {op_port};")
        new_assigns.append(f"assign {op_port} = {net};  // OP")

    injection = "\n// === TPI: inserted by tpi_insert.py ===\n"
    if new_decls:
        injection += "\n".join(new_decls) + "\n"
    if new_assigns:
        injection += "\n".join(new_assigns) + "\n"
    injection += "// === end TPI ===\n"

    # Build the replacement: modified module header + injection block, all in one
    # substitution so we never compute a stale string offset.
    def _replace_header(m: re.Match) -> str:
        header = m.group(0).rstrip()
        if new_ports and header.endswith(");"):
            header = header[:-2] + ",\n" + ",\n".join(new_ports) + "\n);"
        return header + injection

    return re.sub(
        r"module\s+\w+\s*(?:\([^)]*\))?\s*;",
        _replace_header,
        netlist,
        count=1,
        flags=re.DOTALL,
    )


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
    ap.add_argument("--tp", type=int, default=None, metavar="N",
                    help="Total test points to insert (split evenly between CP and OP)")
    ap.add_argument("--cp", type=int, default=6, metavar="N",
                    help="Number of control points to insert [%(default)s]")
    ap.add_argument("--op", type=int, default=6, metavar="N",
                    help="Number of observation points to insert [%(default)s]")
    args = ap.parse_args()


    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Error: {src} not found")
    out = Path(args.output) if args.output else src.with_stem(src.stem + "_tpi")

    # ── Step 1: synthesis ────────────────────────────────────────────────────
    if args.no_synth:
        gate_v = src
        print(f"[1/3] Skipping synthesis — using {gate_v}")
    else:
        gate_v = src.with_stem(src.stem + "_synth")
        print(f"[1/3] Synthesising {src}  →  {gate_v}")
        synthesize(src, gate_v, args.top)
        n_lines = len(gate_v.read_text().splitlines())
        print(f"      Gate-level netlist: {n_lines} lines")

    # ── Step 2: analyse netlist & ask LLM to pick nets ───────────────────────
    netlist_text = gate_v.read_text()
    print(f"[2/3] Analysing netlist …")
    pis, pos, nets = analyze_netlist(netlist_text)
    print(f"      {len(pis)} inputs, {len(pos)} outputs, {len(nets)} internal nets")

    print(f"[2/3] Asking {args.model} to select test points …")
    client = OpenAI(base_url=args.url, api_key="unused")
    cp0s, cp1s, ops = select_test_points(pis, pos, nets, client, args.model, args.cp, args.op, netlist_text, args.tp)
    print(f"      CP0 (force-to-0)  : {cp0s}")
    print(f"      CP1 (force-to-1)  : {cp1s}")
    print(f"      Observation points: {ops}")

    # ── Step 3: apply test points ────────────────────────────────────────────
    print(f"[3/3] Applying test points …")
    result = apply_test_points(netlist_text, cp0s, cp1s, ops)
    out.write_text(result + "\n")
    print(f"\nTPI netlist written to: {out}")


if __name__ == "__main__":
    main()
