#!/usr/bin/env python3
"""
Lazy SAT search for FM-001b n=8 using CEGAR-style blocking clauses.
"""

from __future__ import annotations

import argparse
from itertools import combinations


def edge_var(edges, u, v):
    if u > v:
        u, v = v, u
    return edges[(u, v)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lazy SAT search for FM-001b n=8 using CEGAR blocking clauses."
    )
    parser.add_argument(
        "--solver",
        default="glucose4",
        help="PySAT solver name (default: glucose4).",
    )
    return parser.parse_args()


def find_violations(assignment, edges, vertex_count: int, n: int):
    positives = {lit for lit in assignment if lit > 0}
    red_matrix = [[False] * vertex_count for _ in range(vertex_count)]
    for (u, v), var in edges.items():
        if var in positives:
            red_matrix[u][v] = red_matrix[v][u] = True

    # Red books (B_{n-1})
    limit_red = n - 1
    for u, v in combinations(range(vertex_count), 2):
        if not red_matrix[u][v]:
            continue
        pages = [w for w in range(vertex_count) if w not in (u, v) and red_matrix[u][w] and red_matrix[v][w]]
        if len(pages) >= limit_red:
            clause = [-edge_var(edges, u, v)]
            for w in pages[:limit_red]:
                clause.append(-edge_var(edges, u, w))
                clause.append(-edge_var(edges, v, w))
            return clause

    # Blue books (B_n)
    limit_blue = n
    for u, v in combinations(range(vertex_count), 2):
        if red_matrix[u][v]:
            continue
        pages = [
            w
            for w in range(vertex_count)
            if w not in (u, v) and not red_matrix[u][w] and not red_matrix[v][w]
        ]
        if len(pages) >= limit_blue:
            clause = [edge_var(edges, u, v)]
            for w in pages[:limit_blue]:
                clause.append(edge_var(edges, u, w))
                clause.append(edge_var(edges, v, w))
            return clause
    return None


def main() -> None:
    args = parse_args()
    from pysat.formula import IDPool
    from pysat.solvers import Solver

    from sat_encode_n8 import VERTEX_COUNT, N, build_edge_pool

    pool = IDPool(start_from=1)
    edges = build_edge_pool(pool)
    solver = Solver(name=args.solver)
    added = 0
    while True:
        sat = solver.solve()
        if not sat:
            print("[cegar] UNSAT")
            break
        model = solver.get_model()
        clause = find_violations(model, edges, VERTEX_COUNT, N)
        if clause is None:
            print(f"[cegar] SAT after {added} blocking clauses")
            break
        solver.add_clause(clause)
        added += 1
        if added % 10 == 0:
            print(f"[cegar] added {added} clauses")


if __name__ == "__main__":
    main()
