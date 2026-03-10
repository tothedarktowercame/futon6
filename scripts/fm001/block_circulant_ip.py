#!/usr/bin/env python3
"""
Structured FM-001b IP harness.

Searches for 2-block-circulant red graphs on 4n-2 vertices.  Each block has
q = 2n-1 vertices arranged in a cycle.  Edges are determined by three
circular difference sets:

  * D11 — edges inside block 0 (symmetric under ±d, no self-loops)
  * D22 — edges inside block 1 (symmetric under ±d, no self-loops)
  * D12 — edges from block 0 to block 1 (no symmetry assumption; D21 is the
          negation of D12 to preserve undirected edges)

The red graph is defined entirely by these difference sets; the blue graph is
the edgewise complement on K_{4n-2}.  The ILP will eventually enforce the
book-freeness constraints for both colours, but this initial version only sets
up the combinatorial skeleton.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import pulp


def difference_domain(q: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the canonical positive difference domain and full difference list."""

    # Differences for symmetric (within-block) sets: 1 .. floor((q-1)/2).
    half = tuple(range(1, (q // 2) + 1))
    full = tuple(range(q))  # used for cross-block differences (includes 0)
    return half, full


def add_within_block_vars(
    model: pulp.LpProblem, q: int, prefix: str
) -> Dict[int, pulp.LpVariable]:
    """
    Add binary variables for a symmetric difference set (D11 or D22).

    We keep one variable per positive difference and mirror it to the negative
    difference (q - d) when needed.  This keeps the total variable count small
    while guaranteeing an undirected circulant subgraph.
    """

    half_diffs, _ = difference_domain(q)
    vars_: Dict[int, pulp.LpVariable] = {}
    for d in half_diffs:
        var = pulp.LpVariable(f"{prefix}_{d}", lowBound=0, upBound=1, cat="Binary")
        vars_[d] = var
    return vars_


def add_cross_block_vars(model: pulp.LpProblem, q: int) -> Dict[int, pulp.LpVariable]:
    """
    Add binary variables for the D12 difference set (0 .. q-1).

    Cross-block edges are not required to be symmetric; we model every residue
    class explicitly so that D21 can be derived as the modular negation.
    """

    _, full_diffs = difference_domain(q)
    vars_: Dict[int, pulp.LpVariable] = {}
    for d in full_diffs:
        var = pulp.LpVariable(f"D12_{d}", lowBound=0, upBound=1, cat="Binary")
        vars_[d] = var
    return vars_


def build_model(n: int) -> pulp.LpProblem:
    q = 2 * n - 1
    model = pulp.LpProblem(f"FM001b_n{n}", sense=pulp.LpMinimize)

    d11 = add_within_block_vars(model, q, "D11")
    d22 = add_within_block_vars(model, q, "D22")
    d12 = add_cross_block_vars(model, q)

    # Placeholder objective: minimise total edges; real model will encode
    # feasibility via book-freeness constraints and use a dummy objective.
    objective_terms = []
    objective_terms.extend(d11.values())
    objective_terms.extend(d22.values())
    objective_terms.extend(d12.values())
    model += pulp.lpSum(objective_terms)

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FM-001b 2-block-circulant IP builder")
    parser.add_argument("n", type=int, help="book parameter (n=50 => q=99)")
    parser.add_argument(
        "--write-lp", type=str, help="optional LP/ILP dump path for inspection"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 3:
        raise SystemExit("n must be at least 3 for FM-001.")

    model = build_model(args.n)
    if args.write_lp:
        model.writeLP(args.write_lp)
        print(f"[build] wrote LP skeleton to {args.write_lp}")
        return

    # No solving yet; we just print a summary so callers can verify the shape.
    binary_var_count = len(model.variables())
    print(
        f"[build] n={args.n} q={2 * args.n - 1} "
        f"binary-vars={binary_var_count} constraints={len(model.constraints)}"
    )


if __name__ == "__main__":
    main()
