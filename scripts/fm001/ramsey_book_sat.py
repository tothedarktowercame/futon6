#!/usr/bin/env python3
"""
FM-001 helper: encode the book Ramsey constraint as a SAT instance.

Given n, we ask for a 2-colouring of K_{4n-2} where the red graph avoids
B_{n-1} and the blue graph avoids B_n.  If the CNF is UNSAT we have evidence
that R(B_{n-1}, B_n) < 4n-1.

Usage examples
--------------
Build + solve for n=5 (requires python-sat; use .venv/bin/python):

    .venv/bin/python scripts/fm001/ramsey_book_sat.py 5 --witness-out /tmp/fm001-n5.json

Dump the CNF for an external solver:

    .venv/bin/python scripts/fm001/ramsey_book_sat.py 8 --cnf-out tmp/fm001-n8.cnf --no-solve
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

EdgeKey = Tuple[int, int]
Assignment = Dict[EdgeKey, bool]


def ordered_pair(u: int, v: int) -> EdgeKey:
    return (u, v) if u < v else (v, u)


def add_vertex_zero_monotone_edges(cnf: CNF, edges: Dict[EdgeKey, int], vertex_count: int) -> None:
    """Enforce x_{0,1} >= x_{0,2} >= ... >= x_{0,vertex_count-1} to break vertex permutations."""

    # Vertices are labeled 0 .. vertex_count-1. We only care about edges incident to vertex 0.
    if vertex_count <= 2:
        return
    for j in range(2, vertex_count):
        prev_edge = edges[ordered_pair(0, j - 1)]
        curr_edge = edges[ordered_pair(0, j)]
        # If the later edge is red (1) then the previous edge must also be red.
        cnf.append([-curr_edge, prev_edge])


def build_edges(vertex_count: int, pool: IDPool) -> Dict[EdgeKey, int]:
    edges: Dict[EdgeKey, int] = {}
    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            edges[(i, j)] = pool.id(("edge", i, j))
    return edges


def add_helper_equiv(
    cnf: CNF,
    helper: int,
    positive_literals: Iterable[int],
    negative_literals: Iterable[int],
) -> None:
    """Add CNF ensuring helper == (AND of positives) AND (AND of negations of negatives)."""

    positives = tuple(positive_literals)
    negatives = tuple(negative_literals)

    # helper -> literal
    for lit in positives:
        cnf.append([-helper, lit])
    for lit in negatives:
        cnf.append([-helper, -lit])

    # reverse implication: (all positives true and all negatives false) -> helper
    back_clause = [helper]
    back_clause.extend([-lit for lit in positives])
    back_clause.extend(list(negatives))
    cnf.append(back_clause)


def build_instance(n: int, verbose: bool = False) -> Tuple[CNF, Dict[EdgeKey, int], IDPool]:
    vertex_count = 4 * n - 2
    pool = IDPool()
    edges = build_edges(vertex_count, pool)
    cnf = CNF()

    edge_items = list(edges.keys())
    total_pairs = len(edge_items)

    for idx, (u, v) in enumerate(edge_items, start=1):
        if verbose and (idx % max(1, total_pairs // 10) == 0 or idx == total_pairs):
            print(f"[build] processed {idx}/{total_pairs} edge pairs")

        uv = edges[(u, v)]
        red_helpers: List[int] = []
        blue_helpers: List[int] = []

        for w in range(vertex_count):
            if w == u or w == v:
                continue
            uw = edges[ordered_pair(u, w)]
            vw = edges[ordered_pair(v, w)]

            red_helper = pool.id(("r", u, v, w))
            red_helpers.append(red_helper)
            add_helper_equiv(cnf, red_helper, positive_literals=(uv, uw, vw), negative_literals=())

            blue_helper = pool.id(("b", u, v, w))
            blue_helpers.append(blue_helper)
            add_helper_equiv(
                cnf,
                blue_helper,
                positive_literals=(),
                negative_literals=(uv, uw, vw),
            )

        if red_helpers:
            cnf.extend(
                CardEnc.atmost(
                    lits=red_helpers,
                    bound=max(0, n - 2),
                    vpool=pool,
                    encoding=1,
                ).clauses
            )

        if blue_helpers:
            cnf.extend(
                CardEnc.atmost(
                    lits=blue_helpers,
                    bound=max(0, n - 1),
                    vpool=pool,
                    encoding=1,
                ).clauses
            )

    add_vertex_zero_monotone_edges(cnf, edges, vertex_count)
    cnf.nv = pool.top
    return cnf, edges, pool


def decode_model(model: List[int], edges: Dict[EdgeKey, int]) -> Assignment:
    positives = {lit for lit in model if lit > 0}
    return {edge: (var in positives) for edge, var in edges.items()}


def verify_assignment(n: int, vertex_count: int, assignment: Assignment) -> bool:
    for u, v in combinations(range(vertex_count), 2):
        uv = assignment[ordered_pair(u, v)]
        red_support = 0
        blue_support = 0
        for w in range(vertex_count):
            if w == u or w == v:
                continue
            uw = assignment[ordered_pair(u, w)]
            vw = assignment[ordered_pair(v, w)]
            if uv and uw and vw:
                red_support += 1
            if (not uv) and (not uw) and (not vw):
                blue_support += 1
        if red_support >= n - 1 or blue_support >= n:
            return False
    return True


def write_witness(path: Path, assignment: Assignment, n: int) -> None:
    vertex_count = 4 * n - 2
    red_edges = []
    blue_edges = []
    for (u, v), is_red in assignment.items():
        if is_red:
            red_edges.append([u, v])
        else:
            blue_edges.append([u, v])
    payload = {
        "n": n,
        "vertex_count": vertex_count,
        "red_edges": red_edges,
        "blue_edges": blue_edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/solve the FM-001 book Ramsey SAT instance.")
    parser.add_argument("n", type=int, help="Book parameter n (builds K_{4n-2})")
    parser.add_argument("--cnf-out", type=Path, help="Optional DIMACS output path")
    parser.add_argument("--no-solve", action="store_true", help="Skip invoking a SAT solver")
    parser.add_argument("--solver", default="glucose4", help="pysat solver name (default: glucose4)")
    parser.add_argument("--witness-out", type=Path, help="Write SAT witness JSON to this path")
    parser.add_argument("--verbose", action="store_true", help="Print build progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 3:
        raise SystemExit("n must be >= 3 for FM-001.")

    cnf, edges, pool = build_instance(args.n, verbose=args.verbose)

    print(
        f"[build] n={args.n} vertices={4 * args.n - 2} "
        f"vars={cnf.nv} clauses={len(cnf.clauses)}",
        flush=True,
    )

    if args.cnf_out:
        args.cnf_out.parent.mkdir(parents=True, exist_ok=True)
        cnf.to_file(args.cnf_out)
        print(f"[write] CNF -> {args.cnf_out}")

    if args.no_solve:
        return

    with Solver(name=args.solver, bootstrap_with=cnf.clauses) as solver:
        sat = solver.solve()
        status = "SAT" if sat else "UNSAT"
        print(f"[solve] {status} via {args.solver}")

        if not sat:
            return

        model = solver.get_model()
        assignment = decode_model(model, edges)
        valid = verify_assignment(args.n, 4 * args.n - 2, assignment)
        print(f"[solve] witness verified={valid}")
        if not valid:
            return
        if args.witness_out:
            write_witness(args.witness_out, assignment, args.n)
            print(f"[write] witness -> {args.witness_out}")


if __name__ == "__main__":
    main()
