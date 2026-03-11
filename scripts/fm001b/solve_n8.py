#!/usr/bin/env python3
"""
Solve the FM-001b n=8 SAT instance and verify the witness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

def model_to_edges(model: Sequence[int]) -> set[int]:
    return {lit for lit in model if lit > 0}


def decode_red_matrix(
    model: Sequence[int],
    edges: Dict[Tuple[int, int], int],
    vertex_count: int,
) -> List[List[bool]]:
    positives = model_to_edges(model)
    red = [[False] * vertex_count for _ in range(vertex_count)]
    for (u, v), var in edges.items():
        if var in positives:
            red[u][v] = red[v][u] = True
    return red


def verify_books(red: List[List[bool]], n: int) -> Tuple[bool, bool]:
    red_ok = True
    blue_ok = True
    size = len(red)
    for u in range(size):
        for v in range(u + 1, size):
            # Red book check (base edge must be red)
            if red[u][v]:
                pages = sum(1 for w in range(size) if w not in (u, v) and red[u][w] and red[v][w])
                if pages >= n - 1:
                    red_ok = False
            # Blue book check (base edge must be blue)
            if not red[u][v]:
                pages = sum(1 for w in range(size) if w not in (u, v) and not red[u][w] and not red[v][w])
                if pages >= n:
                    blue_ok = False
    return red_ok, blue_ok


def dump_witness(
    red: List[List[bool]],
    path: Path,
    model: Sequence[int],
    n: int,
) -> None:
    edges = []
    for u in range(len(red)):
        for v in range(u + 1, len(red)):
            if red[u][v]:
                edges.append([u, v])
    data = {
        "n": n,
        "vertex_count": len(red),
        "red_edges": edges,
        "model_size": len(model),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def parse_model_file(path: Path) -> List[int]:
    model: List[int] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v"):
                parts = line.strip().split()
                for lit in parts[1:]:
                    if lit == "0":
                        continue
                    model.append(int(lit))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve/verify FM-001b n=8")
    parser.add_argument("--witness-out", type=Path, required=True, help="Witness JSON path")
    parser.add_argument("--solver", default="glucose4", help="PySAT solver name")
    parser.add_argument(
        "--model-log",
        type=Path,
        help="Optional kissat log with 'v ...' lines to reuse instead of solving",
    )
    args = parser.parse_args()

    import sat_encode_n8
    from pysat.solvers import Solver

    cnf, pool, edges = sat_encode_n8.build_cnf()
    if args.model_log:
        model = parse_model_file(args.model_log)
        print(f"[model] loaded {len(model)} literals from {args.model_log}")
    else:
        with Solver(name=args.solver, bootstrap_with=cnf.clauses) as solver:
            sat = solver.solve()
            print(f"[solve] status={sat}")
            if not sat:
                return
            model = solver.get_model()

    red_matrix = decode_red_matrix(model, edges, sat_encode_n8.VERTEX_COUNT)
    red_ok, blue_ok = verify_books(red_matrix, sat_encode_n8.N)
    print(f"[verify] red-ok={red_ok} blue-ok={blue_ok}")
    if not (red_ok and blue_ok):
        raise SystemExit("Witness failed verification")
    dump_witness(red_matrix, args.witness_out, model, sat_encode_n8.N)
    print(f"[write] witness -> {args.witness_out}")


if __name__ == "__main__":
    main()
