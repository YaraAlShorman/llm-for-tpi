#!/usr/bin/env python3
"""
evaluate.py — Stuck-at fault coverage evaluator for gate-level Verilog netlists.

Parses a Yosys-produced gate-level netlist, runs stuck-at fault simulation
with random (or user-supplied) test patterns, and reports fault coverage and
pattern count. Optionally compares two netlists (e.g. before vs. after TPI).

Usage:
    python evaluate.py netlist.v
    python evaluate.py netlist.v -n 1000
    python evaluate.py netlist_tpi.v --compare netlist_synth.v
    python evaluate.py netlist.v --vectors patterns.txt
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# =========================================================================== #
#  Circuit data model                                                          #
# =========================================================================== #

@dataclass
class Gate:
    gid:    str        # unique instance id
    func:   str        # AND OR NOT NAND NOR XOR XNOR BUF MUX NMUX
    output: str        # name of driven net
    inputs: list[str]  # names of input nets (order matters for MUX)


@dataclass
class Circuit:
    name:  str
    pis:   list[str]   # primary inputs
    pos:   list[str]   # primary outputs
    gates: list[Gate]  # gates in topological (simulation) order
    nets:  set[str]    # all net names in the design


# =========================================================================== #
#  Verilog parser                                                              #
# =========================================================================== #

# Yosys internal cell types: {cell_name: (func, [input_ports], output_port)}
_YOSYS_CELLS: dict[str, tuple[str, list[str], str]] = {
    "$_AND_":  ("AND",  ["A", "B"],       "Y"),
    "$_OR_":   ("OR",   ["A", "B"],       "Y"),
    "$_NOT_":  ("NOT",  ["A"],            "Y"),
    "$_NAND_": ("NAND", ["A", "B"],       "Y"),
    "$_NOR_":  ("NOR",  ["A", "B"],       "Y"),
    "$_XOR_":  ("XOR",  ["A", "B"],       "Y"),
    "$_XNOR_": ("XNOR", ["A", "B"],       "Y"),
    "$_BUF_":  ("BUF",  ["A"],            "Y"),
    "$_MUX_":  ("MUX",  ["A", "B", "S"], "Y"),  # S=0 → A,  S=1 → B
    "$_NMUX_": ("NMUX", ["A", "B", "S"], "Y"),  # inverted MUX
}

# Verilog gate primitives: output is the first port, then inputs (positional)
_PRIMITIVES: dict[str, str] = {
    "and": "AND", "or": "OR", "not": "NOT", "buf": "BUF",
    "nand": "NAND", "nor": "NOR", "xor": "XOR", "xnor": "XNOR",
}


def _strip_comments(src: str) -> str:
    src = re.sub(r"//[^\n]*",   "", src)
    src = re.sub(r"/\*.*?\*/",  "", src, flags=re.DOTALL)
    return src


def _named_ports(body: str) -> dict[str, str]:
    """Parse `.PORT(NET)` connections into {PORT: NET}."""
    return {m.group(1): m.group(2)
            for m in re.finditer(r"\.(\w+)\s*\(\s*(\w+)\s*\)", body)}


def _parse_decl(keyword: str, src: str) -> list[str]:
    """
    Extract all signal names from declarations like:
        input a;
        input a, b, c;
        output reg y;
        input [7:0] data;   (only the identifier, not the width)
    """
    results: list[str] = []
    for m in re.finditer(
        rf"\b{keyword}\b(?:\s+\w+)?(?:\s*\[\d+:\d+\])?\s*([\w\s,]+?)\s*;", src
    ):
        for sig in re.split(r"\s*,\s*", m.group(1)):
            sig = sig.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", sig):
                results.append(sig)
    return list(dict.fromkeys(results))   # deduplicate, preserve order


def _parse_assign(lhs: str, rhs: str, idx: int) -> "Gate | None":
    """Try to recognise simple assign expressions and return a Gate."""
    rhs = rhs.strip()

    # NOT:  ~a  or  !a
    m = re.fullmatch(r"[~!]\s*(\w+)", rhs)
    if m:
        return Gate(f"_a{idx}", "NOT", lhs, [m.group(1)])

    # MUX:  sel ? b : a   →  S=0→A, S=1→B
    m = re.fullmatch(r"(\w+)\s*\?\s*(\w+)\s*:\s*(\w+)", rhs)
    if m:
        return Gate(f"_a{idx}", "MUX", lhs, [m.group(3), m.group(2), m.group(1)])

    # AND:  a & b & …
    if "&" in rhs and not re.search(r"[|^]", rhs):
        ops = [s.strip() for s in rhs.split("&")]
        if all(re.fullmatch(r"\w+", o) for o in ops):
            return Gate(f"_a{idx}", "AND", lhs, ops)

    # OR:   a | b | …
    if "|" in rhs and not re.search(r"[&^]", rhs):
        ops = [s.strip() for s in rhs.split("|")]
        if all(re.fullmatch(r"\w+", o) for o in ops):
            return Gate(f"_a{idx}", "OR", lhs, ops)

    # XOR:  a ^ b ^ …
    if "^" in rhs and not re.search(r"[&|]", rhs):
        ops = [s.strip() for s in rhs.split("^")]
        if all(re.fullmatch(r"\w+", o) for o in ops):
            return Gate(f"_a{idx}", "XOR", lhs, ops)

    # Buffer / direct wire alias
    if re.fullmatch(r"\w+", rhs):
        return Gate(f"_a{idx}", "BUF", lhs, [rhs])

    return None   # expression too complex to parse


def _topo_sort(gates: list[Gate], pi_set: set[str]) -> list[Gate]:
    """Return gates in topological (simulation) order."""
    driven:  dict[str, Gate] = {g.output: g for g in gates}
    visited: set[str]        = set()
    order:   list[Gate]      = []

    def visit(net: str) -> None:
        if net in visited or net in pi_set:
            return
        visited.add(net)
        g = driven.get(net)
        if g:
            for inp in g.inputs:
                visit(inp)
            order.append(g)

    for g in gates:
        visit(g.output)
    return order


def parse_netlist(src: str) -> Circuit:
    src    = _strip_comments(src)
    mod_m  = re.search(r"module\s+(\w+)", src)
    name   = mod_m.group(1) if mod_m else "unknown"
    pis    = _parse_decl("input",  src)
    pos    = _parse_decl("output", src)
    pi_set = set(pis)
    gates: list[Gate] = []
    idx = 0

    # ── Yosys named-port cells ───────────────────────────────────────────────
    for cell, (func, in_ports, out_port) in _YOSYS_CELLS.items():
        for m in re.finditer(rf"{re.escape(cell)}\s+\w+\s*\(([^;]+)\)", src):
            pm      = _named_ports(m.group(1))
            out_net = pm.get(out_port)
            in_nets = [pm[p] for p in in_ports if p in pm]
            if out_net and in_nets:
                gates.append(Gate(f"_y{idx}", func, out_net, in_nets))
                idx += 1

    # ── Verilog gate primitives (positional ports) ───────────────────────────
    prim_pat = "|".join(re.escape(k) for k in _PRIMITIVES)
    for m in re.finditer(rf"\b({prim_pat})\b\s+(?:\w+\s*)?\(([^;]+)\)", src):
        func = _PRIMITIVES[m.group(1)]
        nets = [s.strip() for s in m.group(2).split(",")]
        if len(nets) >= 2:
            gates.append(Gate(f"_p{idx}", func, nets[0], nets[1:]))
            idx += 1

    # ── assign statements ────────────────────────────────────────────────────
    for m in re.finditer(r"\bassign\b\s+(\w+)\s*=\s*([^;]+);", src):
        g = _parse_assign(m.group(1), m.group(2), idx)
        if g:
            gates.append(g)
            idx += 1

    nets: set[str] = set(pis) | set(pos)
    for g in gates:
        nets.add(g.output)
        nets.update(g.inputs)

    return Circuit(name, pis, pos, _topo_sort(gates, pi_set), nets)


# =========================================================================== #
#  Gate evaluation                                                             #
# =========================================================================== #

def _eval(func: str, vals: list[int]) -> int:
    if func == "AND":
        r = 1
        for v in vals: r &= v
        return r
    if func == "OR":
        r = 0
        for v in vals: r |= v
        return r
    if func == "NOT":  return 1 - vals[0]
    if func == "BUF":  return vals[0]
    if func == "NAND":
        r = 1
        for v in vals: r &= v
        return 1 - r
    if func == "NOR":
        r = 0
        for v in vals: r |= v
        return 1 - r
    if func == "XOR":
        r = 0
        for v in vals: r ^= v
        return r
    if func == "XNOR":
        r = 0
        for v in vals: r ^= v
        return 1 - r
    if func in ("MUX", "NMUX"):   # A=vals[0], B=vals[1], S=vals[2]
        v = vals[1] if vals[2] else vals[0]
        return (1 - v) if func == "NMUX" else v
    return 0


# =========================================================================== #
#  Simulation                                                                  #
# =========================================================================== #

def simulate_good(c: Circuit, pattern: dict[str, int]) -> dict[str, int]:
    """Evaluate the fault-free circuit for one input pattern."""
    state = dict(pattern)
    for g in c.gates:
        state[g.output] = _eval(g.func, [state.get(n, 0) for n in g.inputs])
    return state


# =========================================================================== #
#  Parallel fault simulation (bit-vector)                                     #
# =========================================================================== #
#
#  Each bit position i in an integer represents a distinct fault scenario.
#  We pack up to BATCH_SIZE faults and simulate them all in one pass using
#  Python's bitwise operators.  This gives ~64× speedup over per-fault sim.

BATCH_SIZE = 64


def _bv_eval(func: str, ins: list[int], mask: int) -> int:
    """Bit-vector version of _eval.  `mask` has 1s in all active bit positions."""
    if func == "AND":
        r = mask
        for v in ins: r &= v
        return r
    if func == "OR":
        r = 0
        for v in ins: r |= v
        return r & mask
    if func == "NOT":  return (~ins[0]) & mask
    if func == "BUF":  return ins[0] & mask
    if func == "NAND":
        r = mask
        for v in ins: r &= v
        return (~r) & mask
    if func == "NOR":
        r = 0
        for v in ins: r |= v
        return (~r) & mask
    if func == "XOR":
        r = 0
        for v in ins: r ^= v
        return r & mask
    if func == "XNOR":
        r = 0
        for v in ins: r ^= v
        return (~r) & mask
    if func in ("MUX", "NMUX"):
        a, b, s = ins[0], ins[1], ins[2]
        r = (s & b) | ((~s) & a)
        r &= mask
        return ((~r) & mask) if func == "NMUX" else r
    return 0


def fault_simulate(c: Circuit, patterns: list[dict[str, int]]) -> tuple[int, int]:
    """
    Parallel stuck-at fault simulation.

    Returns (detected_faults, total_faults).
    """
    # Full fault list: SA0 and SA1 on every net
    faults: list[tuple[str, int]] = [
        (net, sv) for net in sorted(c.nets) for sv in (0, 1)
    ]
    total = len(faults)

    # Pre-compute good-circuit output values for every pattern
    good_states: list[dict[str, int]] = [simulate_good(c, p) for p in patterns]
    good_outputs: list[dict[str, int]] = [
        {k: gs[k] for k in c.pos} for gs in good_states
    ]

    undetected: set[int] = set(range(total))   # indices into faults[]

    for pat, good_state, good_out in zip(patterns, good_states, good_outputs):
        if not undetected:
            break

        remaining = sorted(undetected)

        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch_indices = remaining[batch_start: batch_start + BATCH_SIZE]
            n = len(batch_indices)
            mask = (1 << n) - 1

            # Build initial bit-vector state: each bit = good value for that net
            bv: dict[str, int] = {}
            for net in c.nets:
                gv = good_state.get(net, 0)
                bv[net] = mask if gv else 0
            for pi in c.pis:
                gv = pat.get(pi, 0)
                bv[pi] = mask if gv else 0

            # Map: net → [(bit_position, stuck_value)] for faults in this batch
            fault_map: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for bit, fi in enumerate(batch_indices):
                net, sv = faults[fi]
                fault_map[net].append((bit, sv))

            # Inject PI faults before simulation
            for net, injections in fault_map.items():
                if net in c.pis:
                    for bit, sv in injections:
                        if sv == 0:
                            bv[net] &= ~(1 << bit)
                        else:
                            bv[net] |=  (1 << bit)

            # Simulate in topological order, re-injecting gate-output faults
            for g in c.gates:
                ins = [bv.get(n, 0) for n in g.inputs]
                bv[g.output] = _bv_eval(g.func, ins, mask)
                # Re-inject any stuck-at fault on this gate's output
                for bit, sv in fault_map.get(g.output, []):
                    if sv == 0:
                        bv[g.output] &= ~(1 << bit)
                    else:
                        bv[g.output] |=  (1 << bit)

            # Check which faults are detected: faulty output ≠ good output
            # Build a "mismatch" mask: bit i set if any PO differs
            mismatch = 0
            for po in c.pos:
                good_bv = mask if good_out.get(po, 0) else 0
                mismatch |= bv.get(po, 0) ^ good_bv
            mismatch &= mask

            # Drop detected faults from undetected set
            for bit, fi in enumerate(batch_indices):
                if mismatch & (1 << bit):
                    undetected.discard(fi)

    detected = total - len(undetected)
    return detected, total


# =========================================================================== #
#  Pattern I/O                                                                 #
# =========================================================================== #

def random_patterns(pis: list[str], n: int, seed: int = 42) -> list[dict[str, int]]:
    rng = random.Random(seed)
    return [{pi: rng.randint(0, 1) for pi in pis} for _ in range(n)]


def load_patterns(path: Path, pis: list[str]) -> list[dict[str, int]]:
    """
    Load patterns from a text file.
    Each line: space-separated 0/1 values in PI declaration order,
               OR a single binary string of length == len(pis).
    Lines starting with '#' are ignored.
    """
    patterns: list[dict[str, int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) == len(pis):
            patterns.append({pi: int(t) for pi, t in zip(pis, toks)})
        elif len(toks) == 1 and len(toks[0]) == len(pis):
            patterns.append({pi: int(b) for pi, b in zip(pis, toks[0])})
    return patterns


# =========================================================================== #
#  Reporting                                                                   #
# =========================================================================== #

def evaluate(
    path: Path,
    n_patterns: int,
    vectors_path: Path | None,
    seed: int,
    label: str = "",
) -> tuple[float, int, int, int]:
    """
    Parse and fault-simulate one netlist.
    Returns (coverage_pct, detected, total, n_patterns_used).
    """
    hdr = f"[{label}] " if label else ""
    circuit = parse_netlist(path.read_text())

    print(f"\n{'─' * 62}")
    print(f"  {hdr}{path.name}")
    print(f"{'─' * 62}")
    print(f"  Module  : {circuit.name}")
    print(f"  Inputs  : {len(circuit.pis)}")
    print(f"  Outputs : {len(circuit.pos)}")
    print(f"  Gates   : {len(circuit.gates)}")
    print(f"  Nets    : {len(circuit.nets)}")

    if vectors_path:
        patterns = load_patterns(vectors_path, circuit.pis)
        print(f"  Patterns: {len(patterns)}  (loaded from {vectors_path.name})")
    else:
        patterns = random_patterns(circuit.pis, n_patterns, seed)
        print(f"  Patterns: {len(patterns)}  (random, seed={seed})")

    if not patterns:
        sys.exit("Error: no patterns available — check --vectors file or -n count.")

    n_faults = 2 * len(circuit.nets)
    print(f"  Faults  : {n_faults}  (SA0 + SA1 on every net)")
    print("  Simulating …", end="", flush=True)

    detected, total = fault_simulate(circuit, patterns)
    coverage = 100.0 * detected / total if total else 0.0

    print(f"\r  Fault coverage : {detected}/{total}  ({coverage:.2f}%)")
    print(f"  Pattern count  : {len(patterns)}")

    return coverage, detected, total, len(patterns)


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("netlist",  metavar="NETLIST.v",
                    help="Gate-level Verilog netlist to evaluate")
    ap.add_argument("-n", "--patterns", type=int, default=500, metavar="N",
                    help="Number of random patterns to generate  [%(default)s]")
    ap.add_argument("--vectors", metavar="FILE",
                    help="Load test patterns from file instead of random generation")
    ap.add_argument("--seed",    type=int, default=42, metavar="S",
                    help="Random seed                            [%(default)s]")
    ap.add_argument("--compare", metavar="BASELINE.v",
                    help="Baseline netlist to compare against (e.g. pre-TPI)")
    args = ap.parse_args()

    vec_path = Path(args.vectors) if args.vectors else None

    tpi_cov, tpi_det, tpi_tot, tpi_pats = evaluate(
        Path(args.netlist),
        args.patterns, vec_path, args.seed,
        label="TPI" if args.compare else "",
    )

    if args.compare:
        base_cov, base_det, base_tot, base_pats = evaluate(
            Path(args.compare),
            args.patterns, vec_path, args.seed,
            label="baseline",
        )
        delta = tpi_cov - base_cov
        print(f"\n{'═' * 62}")
        print("  TPI improvement summary")
        print(f"{'═' * 62}")
        print(f"  Fault coverage : {base_cov:.2f}%  →  {tpi_cov:.2f}%   ({delta:+.2f}%)")
        print(f"  Patterns used  : {base_pats}  →  {tpi_pats}")
        print(f"{'═' * 62}")


if __name__ == "__main__":
    main()
