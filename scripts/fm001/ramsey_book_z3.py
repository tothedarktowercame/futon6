#!/usr/bin/env python3
"""Z3 harness for FM-001 book-Ramsey witnesses.

The solver searches for graphs on V vertices whose edges avoid B_{n-1}
and whose complement avoids B_n. The encoding follows claude-2's spec:
one Boolean per unordered pair and a single cardinality constraint per
pair using Z3's pseudo-Boolean AtMost operator.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


def _load_z3():
    try:
        return importlib.import_module("z3")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "z3 is required for ramsey_book_z3.py. Install z3-solver first."
        ) from exc


@dataclass
class SymmetricBoolMatrix:
    """Helper for symmetrical adjacency access."""

    size: int

    def __post_init__(self) -> None:
        z3 = _load_z3()
        self._vars: List[List[object | None]] = [
            [None] * self.size for _ in range(self.size)
        ]
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self._vars[i][j] = z3.Bool(f"e_{i}_{j}")

    def term(self, i: int, j: int):
        if i == j:
            raise ValueError("Self-loops are not part of the model.")
        if j < i:
            i, j = j, i
        term = self._vars[i][j]
        if term is None:
            raise RuntimeError(f"Missing edge variable ({i}, {j}).")
        return term

    def eval(self, model, i: int, j: int) -> bool:
        z3 = _load_z3()
        if i == j:
            return False
        return z3.is_true(model.evaluate(self.term(i, j), model_completion=True))


def add_book_constraints(
    solver,
    edges: SymmetricBoolMatrix,
    n: int,
    symmetry_break: bool,
) -> None:
    """Impose the B_{n-1}/B_n constraints on the solver."""

    z3 = _load_z3()
    if symmetry_break and edges.size >= 2:
        solver.add(edges.term(0, 1))

    limit_graph = n - 2  # B_{n-1}-freeness => < n-1 common neighbors.
    limit_complement = n - 1  # Complement B_n-freeness => < n common neighbors.

    for i in range(edges.size):
        for j in range(i + 1, edges.size):
            common_neighbors: List[object] = []
            complement_neighbors: List[object] = []
            for k in range(edges.size):
                if k in (i, j):
                    continue
                common_neighbors.append(
                    z3.And(edges.term(i, k), edges.term(j, k))
                )
                complement_neighbors.append(
                    z3.And(z3.Not(edges.term(i, k)), z3.Not(edges.term(j, k)))
                )
            solver.add(
                z3.If(
                    edges.term(i, j),
                    z3.AtMost(*common_neighbors, limit_graph),
                    z3.AtMost(*complement_neighbors, limit_complement),
                )
            )


def adjacency_upper_triangle(model, edges: SymmetricBoolMatrix) -> str:
    """Return the ordered upper-triangle adjacency string."""
    bits: List[str] = []
    for i in range(edges.size):
        for j in range(i + 1, edges.size):
            bits.append("1" if edges.eval(model, i, j) else "0")
    return "".join(bits)


def adjacency_rows(model, edges: SymmetricBoolMatrix) -> Sequence[str]:
    """Return row-wise adjacency (including symmetry) for readability."""
    rows: List[str] = []
    for i in range(edges.size):
        row_bits: List[str] = []
        for j in range(edges.size):
            row_bits.append("1" if edges.eval(model, i, j) else "0")
        rows.append("".join(row_bits))
    return rows


def adjacency_matrix(model, edges: SymmetricBoolMatrix) -> List[List[int]]:
    """Return the 0/1 adjacency matrix."""
    matrix: List[List[int]] = []
    for i in range(edges.size):
        row: List[int] = []
        for j in range(edges.size):
            row.append(1 if edges.eval(model, i, j) else 0)
        matrix.append(row)
    return matrix


def verify_witness(adj: List[List[int]], n: int) -> None:
    """Check both graph and complement book bounds."""

    def max_common(mat: List[List[int]]) -> int:
        max_cn = 0
        size = len(mat)
        for i in range(size):
            for j in range(i + 1, size):
                cn = 0
                for k in range(size):
                    if k in (i, j):
                        continue
                    if mat[i][k] and mat[j][k]:
                        cn += 1
                if cn > max_cn:
                    max_cn = cn
        return max_cn

    size = len(adj)
    complement: List[List[int]] = []
    for i in range(size):
        row: List[int] = []
        for j in range(size):
            row.append(0 if i == j else (1 - adj[i][j]))
        complement.append(row)

    graph_cap = n - 2
    comp_cap = n - 1
    graph_max = max_common(adj)
    comp_max = max_common(complement)
    print(
        f"Verification: max graph CN = {graph_max} (limit {graph_cap}), "
        f"max complement CN = {comp_max} (limit {comp_cap})"
    )
    if graph_max > graph_cap or comp_max > comp_cap:
        raise SystemExit("Witness failed verification.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FM-001 Z3 harness for B_{n-1}/B_n book graphs."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Target B_n parameter (default: 8). Graph CN caps are derived from this.",
    )
    parser.add_argument(
        "--vertices",
        type=int,
        help="Override vertex count (default: 4n-2).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Solver timeout in seconds (0 disables).",
    )
    parser.add_argument(
        "--no-symmetry-break",
        action="store_true",
        help="Disable the default symmetry break e[0][1] = True.",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        help="Optional path to write the adjacency upper-triangle string.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 2:
        raise SystemExit("n must be >= 2.")

    vertex_count = args.vertices if args.vertices else 4 * args.n - 2
    if vertex_count < 2:
        raise SystemExit("Vertex count must be >= 2.")

    z3 = _load_z3()
    solver = z3.Solver()
    if args.timeout:
        solver.set("timeout", args.timeout * 1000)

    edges = SymmetricBoolMatrix(vertex_count)
    add_book_constraints(
        solver,
        edges,
        args.n,
        symmetry_break=not args.no_symmetry_break,
    )

    result = solver.check()
    print(f"Solver status: {result}")
    if result != z3.sat:
        return

    model = solver.model()
    upper_bits = adjacency_upper_triangle(model, edges)
    row_bits = adjacency_rows(model, edges)
    adj_matrix = adjacency_matrix(model, edges)
    num_edges = upper_bits.count("1")
    density = 2 * num_edges / (vertex_count * (vertex_count - 1))

    print(f"n = {args.n}, vertices = {vertex_count}")
    print(f"Edge count = {num_edges} ({density:.4f} density)")
    print(f"Upper-triangle adjacency string length = {len(upper_bits)}")
    print(upper_bits)
    print("Row-wise adjacency (1=edge, 0=non-edge):")
    for idx, row in enumerate(row_bits):
        print(f"{idx:3d}: {row}")

    verify_witness(adj_matrix, args.n)

    if args.dump:
        args.dump.write_text(upper_bits + "\n")
        print(f"Wrote adjacency string to {args.dump}")


if __name__ == "__main__":
    main()
