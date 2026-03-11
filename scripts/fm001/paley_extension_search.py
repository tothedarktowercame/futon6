#!/usr/bin/env python3
"""
Heuristic search for FM-001 Paley-extension witnesses.

Given n, build Paley(q) on q = 4n-3 vertices (q prime, q ≡ 1 mod 4),
then look for an adjacency set S for the extra vertex that keeps every
edge involving the new vertex below the B_{n-1} and B_n common-neighbor
thresholds.  When a valid S is found, emit the full (4n-2)-vertex
adjacency string in column-major lower-triangular order.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


def quadratic_residues(q: int) -> List[int]:
    residues = {pow(i, 2, q) for i in range(1, q)}
    residues.discard(0)
    return sorted(residues)


def build_neighbors(q: int, residues: Sequence[int]) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(q)]
    residue_set = set(residues)
    for v in range(q):
        for d in residue_set:
            u = (v + d) % q
            neighbors[v].append(u)
    return [sorted(set(nbhd)) for nbhd in neighbors]


@dataclass
class Evaluation:
    max_cn_g: int
    max_cn_g_vertex: int
    max_cn_gc: int
    max_cn_gc_vertex: int
    size_s: int
    valid: bool


def evaluate(
    membership: List[bool], neighbors: Sequence[Sequence[int]], limit_edges: int, limit_comp: int
) -> Evaluation:
    q = len(neighbors)
    S_vertices = [v for v, flag in enumerate(membership) if flag]
    notS_vertices = [v for v, flag in enumerate(membership) if not flag]

    max_cn_g = -1
    worst_g = -1
    for v in S_vertices:
        cn = sum(1 for nb in neighbors[v] if membership[nb])
        if cn > max_cn_g:
            max_cn_g = cn
            worst_g = v

    max_cn_gc = -1
    worst_gc = -1
    notS_size = len(notS_vertices)
    for v in notS_vertices:
        neighbor_count = sum(1 for nb in neighbors[v] if not membership[nb])
        cn = max(0, (notS_size - 1) - neighbor_count)
        if cn > max_cn_gc:
            max_cn_gc = cn
            worst_gc = v

    valid = max_cn_g <= limit_edges and max_cn_gc <= limit_comp
    return Evaluation(
        max_cn_g=max_cn_g,
        max_cn_g_vertex=worst_g,
        max_cn_gc=max_cn_gc,
        max_cn_gc_vertex=worst_gc,
        size_s=len(S_vertices),
        valid=valid,
    )


def score(
    evaluation: Evaluation,
    target_size: int,
    size_weight: float,
    limit_edges: int,
    limit_comp: int,
) -> float:
    over_edges = max(0, evaluation.max_cn_g - limit_edges)
    over_comp = max(0, evaluation.max_cn_gc - limit_comp)
    size_penalty = abs(evaluation.size_s - target_size) * size_weight
    return over_edges * 100.0 + over_comp * 100.0 + size_penalty


def anneal(
    neighbors: Sequence[Sequence[int]],
    limit_edges: int,
    limit_comp: int,
    iterations: int,
    seed: int,
    target_size: int,
    size_weight: float,
) -> tuple[list[bool], Evaluation] | None:
    random.seed(seed)
    q = len(neighbors)

    initial = random.sample(range(q), target_size)
    membership = [False] * q
    for v in initial:
        membership[v] = True

    current_eval = evaluate(membership, neighbors, limit_edges, limit_comp)
    current_score = score(current_eval, target_size, size_weight, limit_edges, limit_comp)
    best_eval = current_eval
    best_membership = membership[:]

    S_list = [v for v in range(q) if membership[v]]
    notS_list = [v for v in range(q) if not membership[v]]

    def swap_membership(out_v: int, in_v: int) -> None:
        membership[out_v] = False
        membership[in_v] = True

    def update_lists(out_v: int, in_v: int) -> None:
        S_list.remove(out_v)
        notS_list.append(out_v)
        notS_list.remove(in_v)
        S_list.append(in_v)

    start_temp = 5.0
    for it in range(1, iterations + 1):
        if not S_list or not notS_list:
            break
        temperature = max(0.01, start_temp * (1 - it / iterations))
        remove_v = random.choice(S_list)
        add_v = random.choice(notS_list)
        swap_membership(remove_v, add_v)
        candidate_eval = evaluate(membership, neighbors, limit_edges, limit_comp)
        candidate_score = score(candidate_eval, target_size, size_weight, limit_edges, limit_comp)
        delta = candidate_score - current_score
        accept = delta <= 0 or random.random() < math.exp(-delta / temperature)
        if accept:
            update_lists(remove_v, add_v)
            current_eval = candidate_eval
            current_score = candidate_score
            if candidate_score < score(best_eval, target_size, size_weight, limit_edges, limit_comp):
                best_eval = candidate_eval
                best_membership = membership[:]
            if candidate_eval.valid:
                return best_membership, candidate_eval
        else:
            swap_membership(add_v, remove_v)
    if best_eval.valid:
        return best_membership, best_eval
    return None


def build_extended_adjacency(
    neighbors: Sequence[Sequence[int]], membership: Sequence[bool]
) -> List[List[int]]:
    q = len(neighbors)
    size = q + 1
    adj = [[0] * size for _ in range(size)]
    for v in range(q):
        for u in neighbors[v]:
            adj[v][u] = 1
    extra = q
    for v, flag in enumerate(membership):
        if flag:
            adj[v][extra] = 1
            adj[extra][v] = 1
    return adj


def adjacency_string(adj: Sequence[Sequence[int]]) -> str:
    bits: List[str] = []
    size = len(adj)
    for col in range(size):
        for row in range(col):
            bits.append(str(adj[row][col]))
    return "".join(bits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paley extension witness search")
    parser.add_argument("--n", type=int, default=50, help="Book parameter n")
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Simulated annealing iterations"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--target-size",
        type=int,
        help="Desired size of the adjacency set S (defaults to floor(q/2))",
    )
    parser.add_argument(
        "--size-weight",
        type=float,
        default=0.1,
        help="Weight for |S|-target penalty in the score function",
    )
    parser.add_argument(
        "--adj-out",
        type=Path,
        required=True,
        help="Path to write the adjacency string",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        required=True,
        help="Path to write JSON metadata (S list + evaluation)",
    )
    args = parser.parse_args()

    n = args.n
    q = 4 * n - 3
    if q % 4 != 1:
        raise SystemExit(f"q={q} is not 1 mod 4; Paley construction undefined.")

    residues = quadratic_residues(q)
    neighbors = build_neighbors(q, residues)
    limit_edges = n - 2  # need <= n-2 for B_{n-1}-free edges involving ∞
    limit_comp = n - 1   # need <= n-1 for complement edges involving ∞
    target_size = args.target_size or q // 2

    print(f"[search] n={n} q={q} target-size={target_size}")
    attempt = anneal(
        neighbors=neighbors,
        limit_edges=limit_edges,
        limit_comp=limit_comp,
        iterations=args.iterations,
        seed=args.seed,
        target_size=target_size,
        size_weight=args.size_weight,
    )
    if not attempt:
        raise SystemExit("No valid adjacency set found; try increasing iterations or adjusting seed.")

    membership, evaluation = attempt
    adj = build_extended_adjacency(neighbors, membership)
    adj_str = adjacency_string(adj)

    args.adj_out.parent.mkdir(parents=True, exist_ok=True)
    args.adj_out.write_text(adj_str)
    print(f"[write] adjacency string -> {args.adj_out} ({len(adj_str)} bits)")

    S_vertices = [v for v, flag in enumerate(membership) if flag]
    meta = {
        "n": n,
        "base_q": q,
        "set_size": len(S_vertices),
        "max_cn_g": evaluation.max_cn_g,
        "max_cn_g_vertex": evaluation.max_cn_g_vertex,
        "max_cn_gc": evaluation.max_cn_gc,
        "max_cn_gc_vertex": evaluation.max_cn_gc_vertex,
        "adjacency_set": sorted(S_vertices),
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2))
    print(f"[write] metadata -> {args.meta_out}")


if __name__ == "__main__":
    main()
