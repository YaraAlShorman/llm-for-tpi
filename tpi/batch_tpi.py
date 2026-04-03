#!/usr/bin/env python3
"""
batch_tpi.py — Run the full TPI pipeline on every Verilog file in a directory
               and print a fault coverage comparison table.

Usage:
    python batch_tpi.py ~/ITC99-RTL-Verilog/
    python batch_tpi.py ~/ITC99-RTL-Verilog/ --url http://localhost:8000/v1
    python batch_tpi.py ~/ITC99-RTL-Verilog/ --patterns 1000 --cp 4 --op 6
    python batch_tpi.py ~/ITC99-RTL-Verilog/ --glob 'b0*.v'
"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

from openai import OpenAI

# Import shared logic from tpi_insert and evaluate
sys.path.insert(0, str(Path(__file__).parent))
from tpi_insert import (
    synthesize,
    analyze_netlist,
    select_test_points,
    apply_test_points,
)
from evaluate import parse_netlist, random_patterns, fault_simulate


# --------------------------------------------------------------------------- #
#  Per-benchmark pipeline                                                      #
# --------------------------------------------------------------------------- #

def run_one(
    src: Path,
    client: OpenAI,
    model: str,
    n_cp: int,
    n_op: int,
    n_patterns: int,
    seed: int,
    out_dir: Path,
    n_tp: int | None = None,
) -> dict:
    """
    Full pipeline for one benchmark.
    Returns a dict with coverage stats (or error info).
    """
    name = src.stem
    result = {"name": name, "status": "ok"}

    # ── 1. Synthesize ────────────────────────────────────────────────────────
    synth_v = out_dir / f"{name}_synth.v"
    try:
        if not synth_v.exists():
            synthesize(src, synth_v, top=None)
        result["synth_lines"] = len(synth_v.read_text().splitlines())
    except SystemExit as e:
        result["status"] = f"synth_failed: {e}"
        return result

    # ── 2. Analyse & select test points ─────────────────────────────────────
    try:
        netlist_text = synth_v.read_text()
        pis, pos, nets = analyze_netlist(netlist_text)
        result["n_pi"]   = len(pis)
        result["n_po"]   = len(pos)
        result["n_nets"] = len(nets)

        if len(nets) < 2:
            result["status"] = "too_small"
            return result

        cp0s, cp1s, ops = select_test_points(
            pis, pos, nets, client, model, n_cp, n_op, netlist_text, n_tp
        )
        result["cp0"] = cp0s
        result["cp1"] = cp1s
        result["ops"] = ops
    except Exception:
        result["status"] = f"select_failed: {traceback.format_exc(limit=1)}"
        return result

    # ── 3. Apply test points ─────────────────────────────────────────────────
    tpi_v = out_dir / f"{name}_tpi.v"
    try:
        modified = apply_test_points(netlist_text, cp0s, cp1s, ops)
        tpi_v.write_text(modified + "\n")
    except Exception as e:
        result["status"] = f"apply_failed: {e}"
        return result

    # ── 4. Evaluate baseline ─────────────────────────────────────────────────
    try:
        base_circ    = parse_netlist(synth_v.read_text())
        base_pats    = random_patterns(base_circ.pis, n_patterns, seed)
        base_det, base_tot = fault_simulate(base_circ, base_pats)
        result["base_coverage"] = 100.0 * base_det / base_tot if base_tot else 0.0
        result["base_det"]      = base_det
        result["base_tot"]      = base_tot
    except Exception as e:
        result["base_coverage"] = None
        result["status"] = f"eval_base_failed: {e}"

    # ── 5. Evaluate TPI ──────────────────────────────────────────────────────
    try:
        tpi_circ  = parse_netlist(tpi_v.read_text())
        tpi_pats  = random_patterns(tpi_circ.pis, n_patterns, seed)
        tpi_det, tpi_tot = fault_simulate(tpi_circ, tpi_pats)
        result["tpi_coverage"] = 100.0 * tpi_det / tpi_tot if tpi_tot else 0.0
        result["tpi_det"]      = tpi_det
        result["tpi_tot"]      = tpi_tot
        if result.get("base_coverage") is not None:
            result["delta"] = result["tpi_coverage"] - result["base_coverage"]
    except Exception as e:
        result["tpi_coverage"] = None
        result["status"] = f"eval_tpi_failed: {e}"

    return result


# --------------------------------------------------------------------------- #
#  Table printer                                                               #
# --------------------------------------------------------------------------- #

def _pct(v) -> str:
    return f"{v:6.2f}%" if v is not None else "    n/a"

def _delta(v) -> str:
    if v is None:
        return "    n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:5.2f}%"

def print_table(results: list[dict]) -> None:
    hdr = (
        f"{'Benchmark':<10} {'PIs':>4} {'POs':>4} {'Nets':>6} "
        f"{'Base cov':>9} {'TPI cov':>9} {'Delta':>8} "
        f"{'CPs':>4} {'OPs':>4} {'Status'}"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        n_cp = len(r.get("cp0", [])) + len(r.get("cp1", []))
        n_op = len(r.get("ops", []))
        print(
            f"{r['name']:<10} "
            f"{r.get('n_pi', '-'):>4} "
            f"{r.get('n_po', '-'):>4} "
            f"{r.get('n_nets', '-'):>6} "
            f"{_pct(r.get('base_coverage')):>9} "
            f"{_pct(r.get('tpi_coverage')):>9} "
            f"{_delta(r.get('delta')):>8} "
            f"{n_cp:>4} "
            f"{n_op:>4} "
            f"{r.get('status', '')}"
        )
    print(sep)

    # Summary row for benchmarks that completed successfully
    ok = [r for r in results if r.get("delta") is not None]
    if ok:
        avg_base  = sum(r["base_coverage"] for r in ok) / len(ok)
        avg_tpi   = sum(r["tpi_coverage"]  for r in ok) / len(ok)
        avg_delta = sum(r["delta"]          for r in ok) / len(ok)
        print(
            f"{'AVERAGE':<10} {'':>4} {'':>4} {'':>6} "
            f"{_pct(avg_base):>9} {_pct(avg_tpi):>9} {_delta(avg_delta):>8}"
        )
    print(sep)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dir", metavar="DIR",
                    help="Directory containing Verilog source files")
    ap.add_argument("--glob",     default="*.v",  metavar="PATTERN",
                    help="Glob pattern for source files  [%(default)s]")
    ap.add_argument("--url",      default="http://localhost:8000/v1", metavar="URL",
                    help="OpenAI-compatible server URL   [%(default)s]")
    ap.add_argument("--model",    default="Qwen/Qwen3-14B",           metavar="ID",
                    help="Model ID                       [%(default)s]")
    ap.add_argument("--tp",       type=int, default=None, metavar="N",
                    help="Total test points per design (split evenly between CP and OP)")
    ap.add_argument("--cp",       type=int, default=6,  metavar="N",
                    help="Control points per design      [%(default)s]")
    ap.add_argument("--op",       type=int, default=6,  metavar="N",
                    help="Observation points per design  [%(default)s]")
    ap.add_argument("--patterns", type=int, default=500, metavar="N",
                    help="Random patterns for evaluation [%(default)s]")
    ap.add_argument("--seed",     type=int, default=42,  metavar="S",
                    help="Random seed                    [%(default)s]")
    ap.add_argument("--out",      metavar="DIR",
                    help="Output directory for synthesized/TPI netlists "
                         "(default: same as input dir)")
    ap.add_argument("--json",     metavar="FILE",
                    help="Also save full results to a JSON file")
    ap.add_argument("--skip-synth-existing", action="store_true",
                    help="Skip synthesis if <name>_synth.v already exists")
    args = ap.parse_args()

    src_dir = Path(args.dir)
    out_dir = Path(args.out) if args.out else src_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(f for f in src_dir.glob(args.glob)
                     if not re.search(r'_(synth|tpi)\.v$', f.name))
    if not sources:
        sys.exit(f"No files matching '{args.glob}' found in {src_dir}")

    client = OpenAI(base_url=args.url, api_key="unused")

    print(f"Running TPI pipeline on {len(sources)} benchmarks …")
    print(f"  Server : {args.url}  ({args.model})")
    print(f"  CPs    : {args.cp}   OPs: {args.op}   Patterns: {args.patterns}\n")

    results = []
    for i, src in enumerate(sources, 1):
        print(f"[{i:2d}/{len(sources)}] {src.stem} … ", end="", flush=True)
        r = run_one(src, client, args.model, args.cp, args.op,
                    args.patterns, args.seed, out_dir, args.tp)
        status = r.get("status", "ok")
        if status == "ok":
            print(
                f"base={_pct(r.get('base_coverage'))}  "
                f"tpi={_pct(r.get('tpi_coverage'))}  "
                f"delta={_delta(r.get('delta'))}"
            )
        else:
            print(f"FAILED — {status}")
        results.append(r)

    print_table(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nFull results saved to: {args.json}")


if __name__ == "__main__":
    main()
