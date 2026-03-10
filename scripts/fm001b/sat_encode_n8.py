#!/usr/bin/env python3
"""
FM-001b SAT encoding for the n=8 witness search (30 vertices).

Assign one Boolean per edge (True = red, False = blue) and emit CNF clauses
that forbid a red B_{7} (≤ 6 red triangles sharing a base edge) and forbid a
blue B_{8} (≤ 7 blue triangles).  Uses PySAT for CNF construction.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool


VERTEX_COUNT = 30
N = 8
RED_LIMIT = N - 2  # forbid B_{n-1}
BLUE_LIMIT = N - 1  # forbid complement B_n


def build_edge_pool(pool: IDPool) -> dict[tuple[int, int], int]:
    edges = {}
    for u, v in combinations(range(VERTEX_COUNT), 2):
        edges[(u, v)] = pool.id(("edge", u, v))
    return edges


def edge_var(edges: dict[tuple[int, int], int], u: int, v: int) -> int:
    if u > v:
        u, v = v, u
    return edges[(u, v)]


def build_cnf() -> tuple[CNF, IDPool, dict[tuple[int, int], int]]:
    pool = IDPool(start_from=1)
    edges = build_edge_pool(pool)
    cnf = CNF()

    for u, v in combinations(range(VERTEX_COUNT), 2):
        red_helpers: list[int] = []
        blue_helpers: list[int] = []
        for w in range(VERTEX_COUNT):
            if w == u or w == v:
                continue
            e_uv = edge_var(edges, u, v)
            e_uw = edge_var(edges, u, w)
            e_vw = edge_var(edges, v, w)

            # Red triangle helper
            r_var = pool.id(("red", u, v, w))
            cnf.append([-r_var, e_uv])
            cnf.append([-r_var, e_uw])
            cnf.append([-r_var, e_vw])
            cnf.append([r_var, -e_uv, -e_uw, -e_vw])
            red_helpers.append(r_var)

            # Blue triangle helper
            b_var = pool.id(("blue", u, v, w))
            cnf.append([-b_var, -e_uv])
            cnf.append([-b_var, -e_uw])
            cnf.append([-b_var, -e_vw])
            cnf.append([b_var, e_uv, e_uw, e_vw])
            blue_helpers.append(b_var)

        cnf.extend(
            CardEnc.atmost(lits=red_helpers, bound=RED_LIMIT, vpool=pool, encoding=1).clauses
        )
        cnf.extend(
            CardEnc.atmost(lits=blue_helpers, bound=BLUE_LIMIT, vpool=pool, encoding=1).clauses
        )

    return cnf, pool, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="FM-001b SAT encoder for n=8")
    parser.add_argument(
        "--cnf-out",
        type=Path,
        required=True,
        help="DIMACS path for the generated CNF",
    )
    args = parser.parse_args()

    cnf, pool, edges = build_cnf()
    args.cnf_out.parent.mkdir(parents=True, exist_ok=True)
    cnf.to_file(args.cnf_out)
    print(
        f"[encode] wrote {len(cnf.clauses)} clauses with "
        f"{pool.top} variables ({len(edges)} edge vars)"
    )


if __name__ == "__main__":
    main()
