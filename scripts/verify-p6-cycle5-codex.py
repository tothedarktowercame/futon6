#!/usr/bin/env python3
"""Problem 6 Cycle 5: relaxation scan verifier.

Implements tasks from `problem6-codex-cycle5-handoff.md`:
- Task 1: assembly with c < 1 via dbar0/x/c_needed trajectory scan.
- Task 3: expanded graph-family scan with per-step metrics.
- Task 4: low-rank alpha behavior + rank-1 formula verification.
- Task 5: random edge-partition relaxation probe.
- Task 6: random vertex-selection probe.

Task 2 (symbolic) is emitted as structured proof notes in the markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def to_jsonable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def complete_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def complete_bipartite(n_left: int, n_right: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = n_left + n_right
    edges = []
    for i in range(n_left):
        for j in range(n_left, n):
            edges.append((i, j))
    return n, edges


def random_bipartite_connected(n_left: int, n_right: int, p: float, seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = n_left + n_right
    rng = np.random.default_rng(seed)
    for _ in range(300):
        edges = []
        for i in range(n_left):
            for j in range(n_left, n):
                if rng.random() < p:
                    edges.append((i, j))
        if is_connected(n, edges):
            return n, edges
    raise RuntimeError("failed to generate connected random bipartite graph")


def path_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def star_graph(n: int) -> List[Tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def prufer_random_tree(n: int, seed: int) -> List[Tuple[int, int]]:
    """Random labeled tree via Prufer sequence."""
    rng = np.random.default_rng(seed)
    if n == 1:
        return []
    prufer = rng.integers(0, n, size=n - 2)
    deg = np.ones(n, dtype=int)
    for x in prufer:
        deg[x] += 1
    leaves = sorted([i for i in range(n) if deg[i] == 1])
    edges: List[Tuple[int, int]] = []
    for x in prufer:
        leaf = leaves[0]
        edges.append(edge_key(leaf, int(x)))
        deg[leaf] -= 1
        deg[x] -= 1
        leaves.pop(0)
        if deg[x] == 1:
            # Insert x in sorted order (n is small enough here).
            inserted = False
            xv = int(x)
            for i, y in enumerate(leaves):
                if xv < y:
                    leaves.insert(i, xv)
                    inserted = True
                    break
            if not inserted:
                leaves.append(xv)
    edges.append(edge_key(leaves[0], leaves[1]))
    return edges


def circulant_regular_graph(n: int, d: int) -> List[Tuple[int, int]]:
    """Simple deterministic d-regular graph on n vertices (n even for odd d)."""
    if d >= n:
        raise ValueError("d must be < n")
    if d % 2 == 1 and n % 2 == 1:
        raise ValueError("odd d requires even n")

    edges = set()
    half = d // 2
    for v in range(n):
        for k in range(1, half + 1):
            u = (v + k) % n
            edges.add(edge_key(v, u))
    if d % 2 == 1:
        # Perfect matching by opposite vertices.
        for v in range(n // 2):
            u = v + n // 2
            edges.add(edge_key(v, u))
    return sorted(edges)


def randomize_regular_graph(
    n: int,
    d: int,
    seed: int,
    num_switches: int = 12000,
) -> List[Tuple[int, int]]:
    """Degree-preserving edge-switch randomization from a regular base graph."""
    rng = np.random.default_rng(seed)
    edges = circulant_regular_graph(n, d)
    edge_set = set(edges)

    def build_adj(eset: set[Tuple[int, int]]) -> List[set[int]]:
        adj = [set() for _ in range(n)]
        for a, b in eset:
            adj[a].add(b)
            adj[b].add(a)
        return adj

    adj = build_adj(edge_set)

    edge_list = list(edge_set)
    m = len(edge_list)
    attempts = 0
    done = 0
    while done < num_switches and attempts < 40 * num_switches:
        attempts += 1
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = edge_list[i]
        c, d0 = edge_list[j]
        if len({a, b, c, d0}) < 4:
            continue

        if rng.random() < 0.5:
            p, q, r, s = a, d0, c, b
        else:
            p, q, r, s = a, c, d0, b

        e1 = edge_key(int(p), int(q))
        e2 = edge_key(int(r), int(s))
        if e1[0] == e1[1] or e2[0] == e2[1]:
            continue
        if e1 in edge_set or e2 in edge_set:
            continue

        # Remove old edges.
        edge_set.remove((a, b))
        edge_set.remove((c, d0))
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d0)
        adj[d0].remove(c)

        # Add new edges.
        edge_set.add(e1)
        edge_set.add(e2)
        adj[e1[0]].add(e1[1])
        adj[e1[1]].add(e1[0])
        adj[e2[0]].add(e2[1])
        adj[e2[1]].add(e2[0])

        # Maintain edge list entries (fast path).
        edge_list[i] = e1
        edge_list[j] = e2
        done += 1

    out = sorted(edge_set)
    if not is_connected(n, out):
        # Retry with deterministic fallback if disconnected after randomization.
        out = circulant_regular_graph(n, d)
    return out


def erdos_renyi(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def connected_er(n: int, p: float, seed: int) -> Tuple[List[Tuple[int, int]], int]:
    rng = np.random.default_rng(seed)
    for rep in range(250):
        edges = erdos_renyi(n, p, rng)
        if is_connected(n, edges):
            return edges, rep
    raise RuntimeError(f"failed to sample connected ER_{n}_p{p}")


def is_connected(n: int, edges: Sequence[Tuple[int, int]]) -> bool:
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return all(seen)


def graph_laplacian(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def pseudo_inv_and_half(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(L)
    n = L.shape[0]
    Lplus = np.zeros((n, n), dtype=float)
    Lph = np.zeros((n, n), dtype=float)
    for i, lam in enumerate(eigvals):
        if lam > 1e-10:
            ui = eigvecs[:, i]
            Lplus += (1.0 / lam) * np.outer(ui, ui)
            Lph += (1.0 / np.sqrt(lam)) * np.outer(ui, ui)
    return Lplus, Lph


def compute_edge_z(
    n: int,
    edges: Sequence[Tuple[int, int]],
    Lph: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return z-matrix (m,n) and tau vector (m,)."""
    m = len(edges)
    zmat = np.zeros((m, n), dtype=float)
    taus = np.zeros(m, dtype=float)
    for idx, (u, v) in enumerate(edges):
        b = np.zeros(n, dtype=float)
        b[u] = 1.0
        b[v] = -1.0
        z = Lph @ b
        zmat[idx] = z
        taus[idx] = float(np.dot(z, z))
    return zmat, taus


def find_i0(n: int, edges: Sequence[Tuple[int, int]], taus: np.ndarray, eps: float) -> List[int]:
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)
    i_set = set()
    for v in sorted(range(n), key=lambda vv: len(heavy_adj[vv])):
        if all(u not in i_set for u in heavy_adj[v]):
            i_set.add(v)
    return sorted(i_set)


@dataclass
class GraphPrep:
    name: str
    n: int
    edges: List[Tuple[int, int]]
    Lplus: np.ndarray | None
    Lph: np.ndarray | None
    zmat: np.ndarray | None
    taus: np.ndarray | None
    is_complete: bool = False


@dataclass
class RunOutcome:
    graph: str
    eps: float
    m0: int
    horizon: int
    steps: List[Dict]
    summary: Dict


class GreedyRunner:
    def __init__(self, gp: GraphPrep, eps: float):
        self.gp = gp
        self.eps = eps

    def run(self) -> RunOutcome:
        if self.gp.is_complete and self.gp.n >= 200:
            return self._run_complete_analytic()
        return self._run_generic()

    def _run_complete_analytic(self) -> RunOutcome:
        """Exact formulas for K_n (used for n=200,500 to keep runtime stable)."""
        n = self.gp.n
        eps = self.eps
        m0 = n
        horizon = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0
        steps = []
        rank_alpha_rows = []

        # Track up to horizon with t = |S|.
        for t in range(horizon + 1):
            r_t = n - t
            if r_t <= 0:
                break

            trF = 2.0 * t * (n - t) / n
            dbar0 = trF / (eps * r_t) if (eps > 0 and r_t > 0) else 0.0

            if t <= 1:
                x = 0.0
                alpha = 0.0
                rho1 = 0.0
            else:
                x = (t / n) / eps
                alpha = (t - 1) / (2.0 * t)
                rho1 = alpha

            if x > 1e-14 and dbar0 < 1.0:
                c_needed = (1.0 - x) * (1.0 / dbar0 - 1.0) / x
            elif x > 1e-14 and dbar0 >= 1.0:
                c_needed = -float("inf")
            else:
                c_needed = float("inf") if dbar0 < 1.0 else -float("inf")

            if x < 1.0 - 1e-12:
                asm_at_rho = dbar0 * (1.0 - x * (1.0 - rho1)) / (1.0 - x)
            else:
                asm_at_rho = float("inf")

            steps.append(
                {
                    "t": t,
                    "r_t": r_t,
                    "rank_M": max(0, t - 1),
                    "tr_F": trF,
                    "dbar0": dbar0,
                    "x": x,
                    "alpha": alpha,
                    "rho1": rho1,
                    "c_needed": c_needed,
                    "relaxed_margin": c_needed - rho1 if np.isfinite(c_needed) else float("inf"),
                    "assembly_at_rho1": asm_at_rho,
                    "cross_edge_count": t * (n - t),
                    "cross_edge_exists": t > 0 and t < n,
                    "model": "analytic_kn",
                }
            )

            rank_alpha_rows.append({"rank": max(0, t - 1), "alpha": alpha, "rho1": rho1, "graph": self.gp.name, "eps": eps, "t": t})

        # Random-selection probe on K_n has same values by symmetry.
        random_probe = {
            "num_trials": 0,
            "alpha_mean": steps[-1]["alpha"] if steps else 0.0,
            "alpha_std": 0.0,
            "rho1_mean": steps[-1]["rho1"] if steps else 0.0,
            "rho1_std": 0.0,
            "note": "K_n symmetry: random subset and greedy give same alpha/rho at fixed t",
        }

        return RunOutcome(
            graph=self.gp.name,
            eps=eps,
            m0=m0,
            horizon=horizon,
            steps=steps,
            summary={
                "mode": "analytic_kn",
                "rank_alpha_rows": rank_alpha_rows,
                "random_probe": random_probe,
            },
        )

    def _run_generic(self) -> RunOutcome:
        gp = self.gp
        n = gp.n
        eps = self.eps
        assert gp.zmat is not None and gp.taus is not None
        zmat = gp.zmat
        taus = gp.taus

        i0 = find_i0(n, gp.edges, taus, eps)
        i0_set = set(i0)
        m0 = len(i0)
        horizon = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

        edge_idx: Dict[Tuple[int, int], int] = {}
        neighbors_idx: Dict[int, List[Tuple[int, int]]] = {v: [] for v in i0}
        for idx, (u, v) in enumerate(gp.edges):
            if u in i0_set and v in i0_set:
                ek = edge_key(u, v)
                edge_idx[ek] = idx
                neighbors_idx[u].append((v, idx))
                neighbors_idx[v].append((u, idx))

        S: List[int] = []
        S_set = set()
        M = np.zeros((n, n), dtype=float)

        steps: List[Dict] = []
        rank_alpha_rows = []
        rank1_formula_checks = []

        for _ in range(horizon + 1):
            R = [v for v in i0 if v not in S_set]
            if not R:
                break
            t = len(S)
            r_t = len(R)

            H = eps * np.eye(n) - M
            eigH = np.linalg.eigvalsh(H)
            min_headroom = float(np.min(eigH))
            if min_headroom < 1e-11:
                break
            B = np.linalg.inv(H)
            Bhalf = np.linalg.cholesky(B + 1e-14 * np.eye(n))

            # Build per-v cross edge lists, scores, and collect cross-edge indices.
            cross_idx_all: List[int] = []
            score_map: Dict[int, float] = {}
            d_v_map: Dict[int, float] = {}
            v_to_idxs: Dict[int, List[int]] = {}

            for v in R:
                idxs = [idx for (u, idx) in neighbors_idx[v] if u in S_set]
                v_to_idxs[v] = idxs
                if not idxs:
                    score_map[v] = 0.0
                    d_v_map[v] = 0.0
                    continue
                cross_idx_all.extend(idxs)
                Zv = zmat[idxs]  # (k,n)
                Pv = Bhalf @ Zv.T  # (n,k)
                gram = Pv.T @ Pv
                evals = np.linalg.eigvalsh(gram)
                score_map[v] = float(np.max(evals)) if len(evals) else 0.0
                d_v_map[v] = float(np.sum(evals[evals > 1e-12]))

            if cross_idx_all:
                Zc = zmat[cross_idx_all]  # (m,n)
                F = Zc.T @ Zc
                trF = float(np.sum(taus[cross_idx_all]))
            else:
                F = np.zeros((n, n), dtype=float)
                trF = 0.0

            dbar0 = trF / (eps * r_t) if (eps > 0 and r_t > 0) else 0.0
            dbar = float(np.mean([d_v_map[v] for v in R])) if R else 0.0

            evals_M, evecs_M = np.linalg.eigh(M)
            pos = evals_M > 1e-10
            rank_M = int(np.sum(pos))
            norm_M = float(np.max(evals_M[pos])) if rank_M > 0 else 0.0
            x = norm_M / eps if eps > 0 else 0.0

            if rank_M > 0:
                U = evecs_M[:, pos]
                PM = U @ U.T
            else:
                PM = np.zeros((n, n), dtype=float)

            trPMF = float(np.trace(PM @ F)) if trF > 0 else 0.0
            alpha = trPMF / trF if trF > 1e-14 else 0.0
            trMF = float(np.trace(M @ F)) if trF > 0 else 0.0
            rho1 = trMF / (norm_M * trF) if (norm_M > 1e-14 and trF > 1e-14) else 0.0

            if x > 1e-14 and dbar0 < 1.0:
                c_needed = (1.0 - x) * (1.0 / dbar0 - 1.0) / x
            elif x > 1e-14 and dbar0 >= 1.0:
                c_needed = -float("inf")
            else:
                c_needed = float("inf") if dbar0 < 1.0 else -float("inf")

            if x < 1.0 - 1e-12:
                asm_at_rho = dbar0 * (1.0 - x * (1.0 - rho1)) / (1.0 - x)
            else:
                asm_at_rho = float("inf")

            row = {
                "t": t,
                "r_t": r_t,
                "rank_M": rank_M,
                "tr_F": trF,
                "dbar0": dbar0,
                "dbar": dbar,
                "x": x,
                "alpha": alpha,
                "rho1": rho1,
                "c_needed": c_needed,
                "relaxed_margin": c_needed - rho1 if np.isfinite(c_needed) else float("inf"),
                "assembly_at_rho1": asm_at_rho,
                "cross_edge_count": len(cross_idx_all),
                "cross_edge_exists": len(cross_idx_all) > 0,
                "barrier_headroom_min_eig": min_headroom,
                "model": "generic",
            }

            # Rank-1 formula check at first nontrivial rank.
            if rank_M == 1 and trF > 1e-14:
                i_top = int(np.argmax(evals_M))
                z0 = evecs_M[:, i_top] * np.sqrt(evals_M[i_top])
                z0_norm_sq = float(np.dot(z0, z0))
                if z0_norm_sq > 1e-14 and cross_idx_all:
                    Zc = zmat[cross_idx_all]
                    numer = float(np.sum((Zc @ z0) ** 2) / z0_norm_sq)
                    alpha_rank1_formula = numer / trF
                else:
                    alpha_rank1_formula = 0.0
                rank1_formula_checks.append(
                    {
                        "graph": gp.name,
                        "eps": eps,
                        "t": t,
                        "alpha": alpha,
                        "alpha_rank1_formula": alpha_rank1_formula,
                        "abs_diff": abs(alpha - alpha_rank1_formula),
                    }
                )
                row["alpha_rank1_formula"] = alpha_rank1_formula
                row["alpha_rank1_formula_abs_diff"] = abs(alpha - alpha_rank1_formula)

            steps.append(row)
            rank_alpha_rows.append(
                {
                    "rank": rank_M,
                    "alpha": alpha,
                    "rho1": rho1,
                    "graph": gp.name,
                    "eps": eps,
                    "t": t,
                }
            )

            if len(S) >= horizon:
                break

            # Greedy update: choose smallest operator score.
            best_v = min(R, key=lambda vv: (score_map[vv], vv))
            S.append(best_v)
            S_set.add(best_v)

            # Add newly formed internal edges with best_v.
            new_idxs = [idx for (u, idx) in neighbors_idx[best_v] if u in S_set and u != best_v]
            if new_idxs:
                Znew = zmat[new_idxs]
                M += Znew.T @ Znew

        # Task 6: random subset probe at final size horizon.
        random_probe = random_subset_probe(gp, eps, i0, horizon)

        return RunOutcome(
            graph=gp.name,
            eps=eps,
            m0=m0,
            horizon=horizon,
            steps=steps,
            summary={
                "mode": "generic",
                "rank_alpha_rows": rank_alpha_rows,
                "rank1_formula_checks": rank1_formula_checks,
                "random_probe": random_probe,
            },
        )


def random_subset_probe(
    gp: GraphPrep,
    eps: float,
    i0: Sequence[int],
    T: int,
    trials: int = 80,
    seed: int = 7,
) -> Dict:
    if T <= 1 or len(i0) <= T:
        return {
            "num_trials": 0,
            "alpha_mean": 0.0,
            "alpha_std": 0.0,
            "rho1_mean": 0.0,
            "rho1_std": 0.0,
        }

    assert gp.zmat is not None and gp.taus is not None
    rng = np.random.default_rng(seed)
    n = gp.n
    zmat = gp.zmat

    i0_set = set(i0)
    edge_idx: Dict[Tuple[int, int], int] = {}
    for idx, (u, v) in enumerate(gp.edges):
        if u in i0_set and v in i0_set:
            edge_idx[edge_key(u, v)] = idx

    alphas = []
    rhos = []

    for _ in range(trials):
        S = list(rng.choice(np.array(i0, dtype=int), size=T, replace=False))
        S_set = set(S)
        R = [v for v in i0 if v not in S_set]

        internal_idx = []
        cross_idx = []
        for u in S:
            for v in S:
                if u < v:
                    idx = edge_idx.get((u, v))
                    if idx is not None:
                        internal_idx.append(idx)
        for u in S:
            for v in R:
                idx = edge_idx.get(edge_key(u, v))
                if idx is not None:
                    cross_idx.append(idx)

        if not cross_idx:
            alphas.append(0.0)
            rhos.append(0.0)
            continue

        M = np.zeros((n, n), dtype=float)
        if internal_idx:
            Zi = zmat[internal_idx]
            M = Zi.T @ Zi

        Zc = zmat[cross_idx]
        F = Zc.T @ Zc
        trF = float(np.sum(gp.taus[cross_idx]))

        evals_M, evecs_M = np.linalg.eigh(M)
        pos = evals_M > 1e-10
        rank_M = int(np.sum(pos))
        norm_M = float(np.max(evals_M[pos])) if rank_M > 0 else 0.0

        if rank_M > 0:
            U = evecs_M[:, pos]
            PM = U @ U.T
            alpha = float(np.trace(PM @ F) / trF)
            trMF = float(np.trace(M @ F))
            rho1 = trMF / (norm_M * trF) if norm_M > 1e-14 else 0.0
        else:
            alpha = 0.0
            rho1 = 0.0

        alphas.append(alpha)
        rhos.append(rho1)

    a = np.array(alphas, dtype=float)
    r = np.array(rhos, dtype=float)
    return {
        "num_trials": int(len(a)),
        "alpha_mean": float(np.mean(a)) if len(a) else 0.0,
        "alpha_std": float(np.std(a)) if len(a) else 0.0,
        "alpha_max": float(np.max(a)) if len(a) else 0.0,
        "rho1_mean": float(np.mean(r)) if len(r) else 0.0,
        "rho1_std": float(np.std(r)) if len(r) else 0.0,
        "rho1_max": float(np.max(r)) if len(r) else 0.0,
    }


def edge_partition_probe(
    gp: GraphPrep,
    probs: Sequence[float],
    trials_per_p: int,
    seed: int,
) -> Dict:
    """Task 5 probe: random edge partitions I/C for fixed graph."""
    if gp.zmat is None or gp.taus is None:
        return {"supported": False, "reason": "analytic graph mode"}

    rng = np.random.default_rng(seed)
    n = gp.n
    zmat = gp.zmat

    rows = []
    alpha_ge_1 = []
    alpha_max = -np.inf
    alpha_max_row = None

    m = len(gp.edges)
    for p in probs:
        for tr in range(trials_per_p):
            mask = rng.random(m) < p
            I_idx = np.where(mask)[0]
            C_idx = np.where(~mask)[0]
            if len(C_idx) == 0:
                continue

            M = np.zeros((n, n), dtype=float)
            if len(I_idx):
                Zi = zmat[I_idx]
                M = Zi.T @ Zi
            Zc = zmat[C_idx]
            F = Zc.T @ Zc
            trF = float(np.sum(gp.taus[C_idx]))

            evals_M, evecs_M = np.linalg.eigh(M)
            pos = evals_M > 1e-10
            rank_M = int(np.sum(pos))
            if rank_M > 0:
                U = evecs_M[:, pos]
                PM = U @ U.T
                alpha = float(np.trace(PM @ F) / trF)
            else:
                alpha = 0.0

            # For graph-derived exact edge partition over all edges: M+F == Pi.
            Pi = np.eye(n) - np.ones((n, n), dtype=float) / n
            loewner_err = float(np.max(np.abs(np.linalg.eigvalsh((M + F) - Pi))))

            row = {
                "p": float(p),
                "trial": tr,
                "num_I": int(len(I_idx)),
                "num_C": int(len(C_idx)),
                "rank_M": rank_M,
                "alpha": alpha,
                "max_abs_eig_MplusF_minus_Pi": loewner_err,
            }
            rows.append(row)
            if alpha >= 1.0 - 1e-9:
                alpha_ge_1.append(row)
            if alpha > alpha_max:
                alpha_max = alpha
                alpha_max_row = row

    return {
        "supported": True,
        "rows": rows,
        "num_rows": len(rows),
        "alpha_max": float(alpha_max) if rows else float("nan"),
        "alpha_max_case": alpha_max_row,
        "alpha_ge_1_cases": alpha_ge_1,
        "alpha_lt_1_all": len(alpha_ge_1) == 0,
    }


def build_suite() -> List[GraphPrep]:
    suite: List[GraphPrep] = []

    # Complete graphs (K_200/K_500 handled analytically in runner).
    for n in [100, 200, 500]:
        edges = complete_graph(n)
        suite.append(GraphPrep(name=f"K_{n}", n=n, edges=edges, Lplus=None, Lph=None, zmat=None, taus=None, is_complete=True))

    # d-regular "randomized" via edge-switch randomization.
    for d, seed in [(3, 3103), (10, 3110), (50, 3150)]:
        edges = randomize_regular_graph(100, d, seed)
        suite.append(GraphPrep(name=f"Reg_100_d{d}", n=100, edges=edges, Lplus=None, Lph=None, zmat=None, taus=None))

    # ER(n,p)
    for p, seed in [(0.1, 9101), (0.3, 9303), (0.5, 9505)]:
        edges, rep = connected_er(100, p, seed=seed)
        suite.append(GraphPrep(name=f"ER_100_p{p}_rep{rep}", n=100, edges=edges, Lplus=None, Lph=None, zmat=None, taus=None))

    # Trees.
    suite.append(GraphPrep(name="Tree_prufer_100", n=100, edges=prufer_random_tree(100, seed=777), Lplus=None, Lph=None, zmat=None, taus=None))
    suite.append(GraphPrep(name="Path_100", n=100, edges=path_graph(100), Lplus=None, Lph=None, zmat=None, taus=None))
    suite.append(GraphPrep(name="Star_100", n=100, edges=star_graph(100), Lplus=None, Lph=None, zmat=None, taus=None))

    # Bipartite.
    n, edges = complete_bipartite(50, 50)
    suite.append(GraphPrep(name="K_50_50", n=n, edges=edges, Lplus=None, Lph=None, zmat=None, taus=None))
    n, edges = random_bipartite_connected(50, 50, p=0.2, seed=4242)
    suite.append(GraphPrep(name="BipRand_50_50_p0.2", n=n, edges=edges, Lplus=None, Lph=None, zmat=None, taus=None))

    # Expander proxy: randomized 6-regular graph.
    edges = randomize_regular_graph(100, 6, seed=6006)
    suite.append(GraphPrep(name="ExpanderProxy_Reg_100_d6", n=100, edges=edges, Lplus=None, Lph=None, zmat=None, taus=None))

    return suite


def prepare_graph(gp: GraphPrep) -> GraphPrep:
    # For large complete graphs we stay analytic.
    if gp.is_complete and gp.n >= 200:
        return gp

    if not is_connected(gp.n, gp.edges):
        raise RuntimeError(f"graph {gp.name} is disconnected")

    L = graph_laplacian(gp.n, gp.edges)
    Lplus, Lph = pseudo_inv_and_half(L)
    zmat, taus = compute_edge_z(gp.n, gp.edges, Lph)

    return GraphPrep(
        name=gp.name,
        n=gp.n,
        edges=gp.edges,
        Lplus=Lplus,
        Lph=Lph,
        zmat=zmat,
        taus=taus,
        is_complete=gp.is_complete,
    )


def aggregate(results: List[RunOutcome], edge_partition: Dict[str, Dict]) -> Dict:
    all_steps = []
    for rr in results:
        for s in rr.steps:
            all_steps.append({"graph": rr.graph, "eps": rr.eps, **s})

    def finite(vals: List[float]) -> List[float]:
        return [x for x in vals if np.isfinite(x)]

    dbar0_vals = [s["dbar0"] for s in all_steps]
    c_needed_vals = finite([s["c_needed"] for s in all_steps])
    rho_vals = [s["rho1"] for s in all_steps]
    alpha_vals = [s["alpha"] for s in all_steps]
    x_vals = [s["x"] for s in all_steps]
    margin_vals = finite([s["relaxed_margin"] for s in all_steps])

    # Uniform c0 window:
    # need max(rho1) < c0 < min(c_needed), with c0 < 1.
    max_rho = float(np.max(rho_vals)) if rho_vals else float("nan")
    min_c_needed = float(np.min(c_needed_vals)) if c_needed_vals else float("nan")
    c0_upper = min(min_c_needed, 1.0)
    c0_exists = bool(np.isfinite(max_rho) and np.isfinite(c0_upper) and max_rho < c0_upper)

    # Find witnesses.
    worst_dbar0 = max(all_steps, key=lambda s: s["dbar0"]) if all_steps else None
    worst_x = max(all_steps, key=lambda s: s["x"]) if all_steps else None
    worst_rho = max(all_steps, key=lambda s: s["rho1"]) if all_steps else None
    worst_alpha = max(all_steps, key=lambda s: s["alpha"]) if all_steps else None
    worst_margin = min(all_steps, key=lambda s: s["relaxed_margin"] if np.isfinite(s["relaxed_margin"]) else float("inf")) if all_steps else None

    dbar0_le_2_3_all = all(s["dbar0"] <= (2.0 / 3.0 + 1e-10) for s in all_steps)
    c_needed_ge_1_all = all((not np.isfinite(s["c_needed"])) or s["c_needed"] >= 1.0 - 1e-10 for s in all_steps)
    rho_lt_1_all = all(s["rho1"] < 1.0 - 1e-12 for s in all_steps)
    alpha_lt_1_all = all(s["alpha"] < 1.0 - 1e-12 for s in all_steps)

    # Rank aggregates.
    rank_to_alpha: Dict[int, List[float]] = {}
    rank1_formula_diffs = []
    for rr in results:
        for s in rr.steps:
            rk = int(s["rank_M"])
            rank_to_alpha.setdefault(rk, []).append(float(s["alpha"]))
            if "alpha_rank1_formula_abs_diff" in s:
                rank1_formula_diffs.append(float(s["alpha_rank1_formula_abs_diff"]))

    rank_summary = []
    for rk in sorted(rank_to_alpha.keys()):
        vals = np.array(rank_to_alpha[rk], dtype=float)
        rank_summary.append(
            {
                "rank": rk,
                "count": int(len(vals)),
                "alpha_mean": float(np.mean(vals)),
                "alpha_max": float(np.max(vals)),
                "alpha_min": float(np.min(vals)),
            }
        )

    # Edge partition summary.
    ep_rows = []
    for gname, out in edge_partition.items():
        if not out.get("supported", False):
            continue
        ep_rows.append(
            {
                "graph": gname,
                "alpha_max": out.get("alpha_max"),
                "alpha_lt_1_all": out.get("alpha_lt_1_all"),
                "num_rows": out.get("num_rows"),
            }
        )

    return {
        "num_runs": len(results),
        "num_steps": len(all_steps),
        "dbar0_le_2_3_all": dbar0_le_2_3_all,
        "c_needed_ge_1_all": c_needed_ge_1_all,
        "rho_lt_1_all": rho_lt_1_all,
        "alpha_lt_1_all": alpha_lt_1_all,
        "max_dbar0": float(np.max(dbar0_vals)) if dbar0_vals else float("nan"),
        "max_x": float(np.max(x_vals)) if x_vals else float("nan"),
        "max_rho1": max_rho,
        "max_alpha": float(np.max(alpha_vals)) if alpha_vals else float("nan"),
        "min_relaxed_margin": float(np.min(margin_vals)) if margin_vals else float("nan"),
        "min_c_needed_finite": min_c_needed,
        "uniform_c0": {
            "exists": c0_exists,
            "interval_open": [max_rho, c0_upper] if c0_exists else None,
            "recommended": float((max_rho + c0_upper) / 2.0) if c0_exists else None,
        },
        "worst_cases": {
            "dbar0": worst_dbar0,
            "x": worst_x,
            "rho1": worst_rho,
            "alpha": worst_alpha,
            "relaxed_margin": worst_margin,
        },
        "rank_summary": rank_summary,
        "rank1_formula_max_abs_diff": float(np.max(rank1_formula_diffs)) if rank1_formula_diffs else float("nan"),
        "edge_partition_summary": ep_rows,
    }


def build_output(seed: int, partition_trials: int, partition_probs: Sequence[float]) -> Dict:
    eps_list = [0.1, 0.2, 0.3, 0.5]

    suite = build_suite()
    prepped = [prepare_graph(g) for g in suite]

    results: List[RunOutcome] = []
    for gp in prepped:
        for eps in eps_list:
            rr = GreedyRunner(gp, eps).run()
            results.append(rr)

    # Task 5: edge partition probe on n=100 graphs only (computationally practical).
    edge_partition = {}
    for gp in prepped:
        if gp.n > 100:
            edge_partition[gp.name] = {
                "supported": False,
                "reason": "skipped for n>100 in partition probe",
            }
            continue
        edge_partition[gp.name] = edge_partition_probe(
            gp,
            probs=partition_probs,
            trials_per_p=partition_trials,
            seed=seed + 101,
        )

    summary = aggregate(results, edge_partition)

    runs_out = []
    for rr in results:
        runs_out.append(
            {
                "graph": rr.graph,
                "eps": rr.eps,
                "m0": rr.m0,
                "horizon": rr.horizon,
                "steps": rr.steps,
                "run_summary": rr.summary,
            }
        )

    return {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "seed": seed,
            "eps_list": eps_list,
            "partition_trials_per_p": partition_trials,
            "partition_probs": list(partition_probs),
            "suite": [g.name for g in prepped],
        },
        "summary": summary,
        "runs": runs_out,
        "edge_partition": edge_partition,
    }


def print_summary(out: Dict):
    s = out["summary"]
    print("=" * 96)
    print("P6 CYCLE 5 CODEX: RELAXATION SCAN")
    print("=" * 96)
    print(f"runs={s['num_runs']} steps={s['num_steps']}")
    print(f"dbar0<=2/3 all: {s['dbar0_le_2_3_all']}  max_dbar0={s['max_dbar0']:.6f}")
    print(f"c_needed>=1 all: {s['c_needed_ge_1_all']}  min_c_needed_finite={s['min_c_needed_finite']:.6f}")
    print(f"rho1<1 all: {s['rho_lt_1_all']}  max_rho1={s['max_rho1']:.6f}")
    print(f"alpha<1 all (vertex-induced): {s['alpha_lt_1_all']}  max_alpha={s['max_alpha']:.6f}")
    print(f"min relaxed margin (c_needed-rho1): {s['min_relaxed_margin']:.6f}")
    print(f"max x: {s['max_x']:.6f}")
    print(f"uniform c0 exists: {s['uniform_c0']['exists']} interval={s['uniform_c0']['interval_open']}")
    print(f"rank1 formula max abs diff: {s['rank1_formula_max_abs_diff']:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition-trials", type=int, default=16)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle5-results.json"),
    )
    args = parser.parse_args()

    probs = [0.1 * k for k in range(1, 10)]
    out = build_output(seed=args.seed, partition_trials=args.partition_trials, partition_probs=probs)
    print_summary(out)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=to_jsonable)

    print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
