#!/usr/bin/env python3
"""Compute alpha = tr(P_M F)/tr(F) and rho_1 at each barrier greedy step.

alpha is the fraction of F-trace in col(M).
rho_1 = tr(MF)/(||M||*tr(F)).
Key claim: rho_k <= rho_1 <= alpha, so alpha < 1/2 => rho_k < 1/2 for all k.

Uses the same greedy and graph suite as the Codex C3 verifier.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def complete_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def barbell_graph(k: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = 2 * k
    edges: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, n):
        for j in range(i + 1, n):
            edges.append((i, j))
    edges.append((k - 1, k))
    return n, edges


def star_graph(n: int) -> List[Tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def grid_graph(rows: int, cols: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = rows * cols
    edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return n, edges


def erdos_renyi(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
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


def connected_er(n: int, p: float, seed: int) -> Tuple[List[Tuple[int, int]], int]:
    rng = np.random.default_rng(seed)
    for rep in range(200):
        edges = erdos_renyi(n, p, rng)
        if is_connected(n, edges):
            return edges, rep
    raise RuntimeError("Failed to get connected ER")


def graph_laplacian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def pseudo_sqrt_inv(L: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(L)
    out = np.zeros_like(L)
    for i, lam in enumerate(eigvals):
        if lam > 1e-10:
            out += (1.0 / np.sqrt(lam)) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return out


def compute_edge_matrices(n, edges, Lph):
    x_edges = []
    taus = []
    for u, v in edges:
        b = np.zeros(n)
        b[u] = 1.0
        b[v] = -1.0
        z = Lph @ b
        x_edges.append(np.outer(z, z))
        taus.append(float(np.dot(z, z)))
    return x_edges, taus


def find_i0(n, edges, taus, eps):
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


def run_greedy_alpha(name, n, edges, eps):
    L = graph_laplacian(n, edges)
    Lph = pseudo_sqrt_inv(L)
    x_edges, taus = compute_edge_matrices(n, edges, Lph)
    Pi = np.eye(n) - np.ones((n, n)) / n

    i0 = find_i0(n, edges, taus, eps)
    i0_set = set(i0)
    m0 = len(i0)
    horizon_T = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        if u in i0_set and v in i0_set:
            edge_idx[edge_key(u, v)] = idx

    S = []
    S_set = set()
    M = np.zeros((n, n))
    rows = []

    for step in range(horizon_T + 1):
        R = [v for v in i0 if v not in S_set]
        r_t = len(R)
        if r_t == 0:
            break

        # Build F (cross-edge matrix)
        F = np.zeros((n, n))
        for u in S:
            for v in R:
                idx = edge_idx.get(edge_key(u, v))
                if idx is not None:
                    F += x_edges[idx]

        tr_F = float(np.trace(F))
        tau = float(np.trace(M))
        norm_M = float(np.linalg.norm(M, ord=2))
        x = norm_M / eps if eps > 0 else 0.0

        # Compute P_M: projection onto col(M) (eigenvectors with eigenvalue > threshold)
        eigvals_M, eigvecs_M = np.linalg.eigh(M)
        rank_M = np.sum(eigvals_M > 1e-10)
        P_M = np.zeros((n, n))
        for i in range(n):
            if eigvals_M[i] > 1e-10:
                P_M += np.outer(eigvecs_M[:, i], eigvecs_M[:, i])

        # alpha = tr(P_M F) / tr(F)
        tr_PM_F = float(np.trace(P_M @ F))
        alpha = tr_PM_F / tr_F if tr_F > 1e-14 else 0.0

        # rho_1 = tr(MF) / (||M|| * tr(F))
        tr_MF = float(np.trace(M @ F))
        rho_1 = tr_MF / (norm_M * tr_F) if (norm_M > 1e-14 and tr_F > 1e-14) else 0.0

        # For K_n: theoretical alpha = (t-1)/(2t)
        t = len(S)
        kn_alpha = (t - 1) / (2 * t) if t > 0 else 0.0

        # dbar
        H = eps * np.eye(n) - M
        eigH = np.linalg.eigvalsh(H)
        if float(np.min(eigH)) < 1e-10:
            break
        B = np.linalg.inv(H)
        dbar = float(np.trace(B @ F) / r_t)

        # Neumann bound check
        dbar_m0 = tr_F / (eps * r_t) if eps > 0 else 0.0
        if x < 1.0 - 1e-12:
            neumann_bound = dbar_m0 * (2 - x) / (2 * (1 - x))
        else:
            neumann_bound = float('inf')

        rows.append({
            "t": t,
            "r_t": r_t,
            "rank_M": int(rank_M),
            "tau": tau,
            "tr_F": tr_F,
            "tr_PM_F": tr_PM_F,
            "alpha": alpha,
            "rho_1": rho_1,
            "kn_alpha_theory": kn_alpha,
            "norm_M": norm_M,
            "x": x,
            "dbar": dbar,
            "dbar_m0": dbar_m0,
            "neumann_bound": neumann_bound,
            "dim_colM": int(rank_M),
            "dim_nullM_in_Pi": int(n - 1 - rank_M),
        })

        if len(S) >= horizon_T:
            break

        # Greedy: pick vertex minimizing barrier norm
        Bsqrt = np.linalg.cholesky(B + 1e-14 * np.eye(n))
        scores = {}
        for v in R:
            C_v = np.zeros((n, n))
            for u in S:
                idx = edge_idx.get(edge_key(u, v))
                if idx is not None:
                    C_v += x_edges[idx]
            Y = Bsqrt @ C_v @ Bsqrt.T
            Y = 0.5 * (Y + Y.T)
            scores[v] = float(np.max(np.linalg.eigvalsh(Y)))

        best_v = min(R, key=lambda v: (scores[v], v))
        S.append(best_v)
        S_set.add(best_v)

        for u in S[:-1]:
            idx = edge_idx.get(edge_key(u, best_v))
            if idx is not None:
                M += x_edges[idx]

    return {"graph": name, "n": n, "eps": eps, "m0": m0, "T": horizon_T, "rows": rows}


def main():
    er_edges, er_rep = connected_er(60, 0.5, seed=42)
    n_b, e_b = barbell_graph(40)
    n_g, e_g = grid_graph(8, 5)

    suite = [
        ("K_40", 40, complete_graph(40)),
        ("K_80", 80, complete_graph(80)),
        (f"ER_60_p0.5_seed42_rep{er_rep}", 60, er_edges),
        ("Barbell_40", n_b, e_b),
        ("Star_40", 40, star_graph(40)),
        ("Grid_8x5", n_g, e_g),
    ]
    eps_list = [0.2, 0.3, 0.5]

    max_alpha = -1.0
    max_alpha_witness = None
    max_rho1 = -1.0
    max_rho1_witness = None
    all_results = []

    for gname, n, edges in suite:
        for eps in eps_list:
            result = run_greedy_alpha(gname, n, edges, eps)
            all_results.append(result)
            for row in result["rows"]:
                if row["t"] == 0:
                    continue  # skip t=0 (no M yet)
                if row["alpha"] > max_alpha:
                    max_alpha = row["alpha"]
                    max_alpha_witness = f"{gname}, eps={eps}, t={row['t']}"
                if row["rho_1"] > max_rho1:
                    max_rho1 = row["rho_1"]
                    max_rho1_witness = f"{gname}, eps={eps}, t={row['t']}"

    print("=" * 80)
    print("ALPHA AND RHO_1 ANALYSIS")
    print("=" * 80)
    print(f"\nMax alpha across all steps: {max_alpha:.6f}")
    print(f"  Witness: {max_alpha_witness}")
    print(f"\nMax rho_1 across all steps: {max_rho1:.6f}")
    print(f"  Witness: {max_rho1_witness}")
    print(f"\nalpha < 0.5? {'YES' if max_alpha < 0.5 else 'NO (VIOLATION!)'}")
    print(f"rho_1 < 0.5? {'YES' if max_rho1 < 0.5 else 'NO (VIOLATION!)'}")

    # Show horizon rows for each run
    print("\n" + "-" * 80)
    print("HORIZON SNAPSHOTS (last step of each run)")
    print("-" * 80)
    print(f"{'Graph':<25} {'eps':>5} {'t':>4} {'rank_M':>6} {'alpha':>8} {'rho_1':>8} "
          f"{'dbar':>8} {'x':>8}")
    for result in all_results:
        if not result["rows"]:
            continue
        row = result["rows"][-1]
        if row["t"] == 0:
            continue
        print(f"{result['graph']:<25} {result['eps']:>5.1f} {row['t']:>4d} "
              f"{row['rank_M']:>6d} {row['alpha']:>8.4f} {row['rho_1']:>8.4f} "
              f"{row['dbar']:>8.4f} {row['x']:>8.4f}")

    # Show steps where alpha is highest (top 10)
    print("\n" + "-" * 80)
    print("TOP 10 ALPHA VALUES")
    print("-" * 80)
    all_rows = []
    for result in all_results:
        for row in result["rows"]:
            if row["t"] > 0:
                all_rows.append((row["alpha"], row["rho_1"], result["graph"],
                                result["eps"], row["t"], row["rank_M"],
                                row["dim_colM"], row["dim_nullM_in_Pi"],
                                row["tr_PM_F"], row["tr_F"]))
    all_rows.sort(reverse=True)
    print(f"{'alpha':>8} {'rho_1':>8} {'Graph':<25} {'eps':>5} {'t':>4} "
          f"{'rank':>5} {'dim_M':>6} {'dim_M^':>6} {'F_in_M':>10} {'tr_F':>10}")
    for a, r1, g, e, t, rk, dm, dn, fpm, tf in all_rows[:10]:
        print(f"{a:>8.4f} {r1:>8.4f} {g:<25} {e:>5.1f} {t:>4d} "
              f"{rk:>5d} {dm:>6d} {dn:>6d} {fpm:>10.4f} {tf:>10.4f}")

    # Dump full results
    with open("/home/joe/code/futon6/data/first-proof/alpha-rho-analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to data/first-proof/alpha-rho-analysis.json")


if __name__ == "__main__":
    main()
