#!/usr/bin/env python3
"""Compute the gap analysis for rho_1 < 1/2.

From the operator inequality F + M <= Pi:
  tr(MF) <= tr(M - M^2) = sum mu_i(1-mu_i)

So rho_1 <= sum mu_i(1-mu_i) / (mu_max * tr(F)).
For rho_1 < 1/2: need tr(F) > 2*sum mu_i(1-mu_i) / mu_max.

This script computes the actual tr(MF), the bound, and the margin.
"""

import json
import numpy as np

# Load the alpha-rho analysis results
with open("/home/joe/code/futon6/data/first-proof/alpha-rho-analysis.json") as f:
    results = json.load(f)

print("=" * 100)
print("OPERATOR BOUND GAP ANALYSIS: rho_1 <= sum mu_i(1-mu_i) / (mu_max * tr(F))")
print("=" * 100)

print(f"\n{'Graph':<25} {'eps':>5} {'t':>4} {'rho_1':>8} {'bound':>8} "
      f"{'trF':>10} {'reqd_trF':>10} {'margin':>8} {'alpha':>8}")
print("-" * 100)

max_bound = -1
max_bound_witness = None
all_data = []

for result in results:
    for row in result["rows"]:
        if row["t"] == 0 or row["rank_M"] == 0:
            continue

        # We need to recompute M eigenvalues - use the stored data
        # Actually, we stored alpha, rho_1, but not sum mu_i(1-mu_i)
        # Let's compute the bound from stored data
        # rho_1 = tr(MF)/(||M||*tr(F))
        # operator bound: sum mu_i(1-mu_i)/(||M||*tr(F))
        # We have rho_1 and we need the operator bound

        # Since we don't have the eigenvalues directly, let's use:
        # operator bound = (tau - ||M||_F^2) / (mu_max * tr(F))
        # But we don't have ||M||_F^2 either.

        # For K_n: operator bound = (t-1)/(2t) = rho_1 (tight)
        # For non-K_n: operator bound > rho_1
        pass

# Need to rerun with eigenvalue computation
print("\nRecomputing with eigenvalue data...")

# Inline the necessary parts
import sys
sys.path.insert(0, "/home/joe/code/futon6/scripts")

# Reuse graph generation from compute-alpha-rho
def edge_key(u, v):
    return (u, v) if u < v else (v, u)

def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def barbell_graph(k):
    n = 2 * k
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, n):
        for j in range(i + 1, n):
            edges.append((i, j))
    edges.append((k - 1, k))
    return n, edges

def star_graph(n):
    return [(0, i) for i in range(1, n)]

def grid_graph(rows, cols):
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return n, edges

def erdos_renyi(n, p, rng):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges

def is_connected(n, edges):
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

def connected_er(n, p, seed):
    rng = np.random.default_rng(seed)
    for rep in range(200):
        edges = erdos_renyi(n, p, rng)
        if is_connected(n, edges):
            return edges, rep
    raise RuntimeError("Failed")

def graph_laplacian(n, edges):
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, u] += 1.0; L[v, v] += 1.0; L[u, v] -= 1.0; L[v, u] -= 1.0
    return L

def pseudo_sqrt_inv(L):
    eigvals, eigvecs = np.linalg.eigh(L)
    out = np.zeros_like(L)
    for i, lam in enumerate(eigvals):
        if lam > 1e-10:
            out += (1.0 / np.sqrt(lam)) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return out

def compute_edge_matrices(n, edges, Lph):
    x_edges = []; taus = []
    for u, v in edges:
        b = np.zeros(n); b[u] = 1.0; b[v] = -1.0
        z = Lph @ b
        x_edges.append(np.outer(z, z))
        taus.append(float(np.dot(z, z)))
    return x_edges, taus

def find_i0(n, edges, taus, eps):
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v); heavy_adj[v].add(u)
    i_set = set()
    for v in sorted(range(n), key=lambda vv: len(heavy_adj[vv])):
        if all(u not in i_set for u in heavy_adj[v]):
            i_set.add(v)
    return sorted(i_set)

# Run with eigenvalue tracking
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

print(f"\n{'Graph':<25} {'eps':>5} {'t':>4} {'rho_1':>8} {'op_bnd':>8} "
      f"{'trF':>10} {'reqd':>10} {'margin':>8} {'alpha':>8}")
print("-" * 100)

max_op_bound = -1
max_op_witness = None

for gname, n, edges in suite:
    L = graph_laplacian(n, edges)
    Lph = pseudo_sqrt_inv(L)
    x_edges, taus = compute_edge_matrices(n, edges, Lph)
    Pi = np.eye(n) - np.ones((n, n)) / n

    for eps in eps_list:
        i0 = find_i0(n, edges, taus, eps)
        i0_set = set(i0)
        m0 = len(i0)
        horizon_T = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

        edge_idx = {}
        for idx, (u, v) in enumerate(edges):
            if u in i0_set and v in i0_set:
                edge_idx[edge_key(u, v)] = idx

        S = []; S_set = set(); M = np.zeros((n, n))

        for step in range(horizon_T + 1):
            R = [v for v in i0 if v not in S_set]
            r_t = len(R)
            if r_t == 0:
                break

            F = np.zeros((n, n))
            for u in S:
                for v in R:
                    idx = edge_idx.get(edge_key(u, v))
                    if idx is not None:
                        F += x_edges[idx]

            t = len(S)
            tr_F = float(np.trace(F))
            tau = float(np.trace(M))
            norm_M = float(np.linalg.norm(M, ord=2))

            if t == 0 or norm_M < 1e-14 or tr_F < 1e-14:
                if t >= horizon_T:
                    break
                # Greedy step
                H = eps * np.eye(n) - M
                eigH = np.linalg.eigvalsh(H)
                if float(np.min(eigH)) < 1e-10:
                    break
                B = np.linalg.inv(H)
                Bsqrt = np.linalg.cholesky(B + 1e-14 * np.eye(n))
                scores = {}
                for v in R:
                    C_v = np.zeros((n, n))
                    for u in S:
                        idx2 = edge_idx.get(edge_key(u, v))
                        if idx2 is not None:
                            C_v += x_edges[idx2]
                    Y = Bsqrt @ C_v @ Bsqrt.T
                    Y = 0.5 * (Y + Y.T)
                    scores[v] = float(np.max(np.linalg.eigvalsh(Y)))
                best_v = min(R, key=lambda v: (scores.get(v, 0), v))
                S.append(best_v); S_set.add(best_v)
                for u in S[:-1]:
                    idx2 = edge_idx.get(edge_key(u, best_v))
                    if idx2 is not None:
                        M += x_edges[idx2]
                continue

            # Compute M eigenvalues
            eigs_M = np.linalg.eigvalsh(M)
            eigs_M = eigs_M[eigs_M > 1e-10]  # nonzero eigenvalues

            sum_mu_1_minus_mu = float(np.sum(eigs_M * (1 - eigs_M)))
            frob_sq = float(np.sum(eigs_M**2))

            # Operator bound for rho_1
            op_bound = sum_mu_1_minus_mu / (norm_M * tr_F)

            # Actual rho_1
            tr_MF = float(np.trace(M @ F))
            rho_1 = tr_MF / (norm_M * tr_F)

            # alpha
            eigvals_M, eigvecs_M = np.linalg.eigh(M)
            P_M = np.zeros((n, n))
            for i in range(n):
                if eigvals_M[i] > 1e-10:
                    P_M += np.outer(eigvecs_M[:, i], eigvecs_M[:, i])
            alpha = float(np.trace(P_M @ F)) / tr_F

            # Required tr(F) for rho_1 < 1/2 via operator bound
            reqd_trF = 2 * sum_mu_1_minus_mu / norm_M
            margin = (tr_F - reqd_trF) / reqd_trF if reqd_trF > 0 else float('inf')

            if op_bound > max_op_bound:
                max_op_bound = op_bound
                max_op_witness = f"{gname}, eps={eps}, t={t}"

            # Print only horizon row and rows where op_bound is high
            if t == horizon_T or t >= horizon_T - 1 or op_bound > 0.4:
                print(f"{gname:<25} {eps:>5.1f} {t:>4d} {rho_1:>8.4f} {op_bound:>8.4f} "
                      f"{tr_F:>10.4f} {reqd_trF:>10.4f} {margin:>8.2%} {alpha:>8.4f}")

            if t >= horizon_T:
                break

            # Greedy step
            H = eps * np.eye(n) - M
            eigH = np.linalg.eigvalsh(H)
            if float(np.min(eigH)) < 1e-10:
                break
            B = np.linalg.inv(H)
            Bsqrt = np.linalg.cholesky(B + 1e-14 * np.eye(n))
            scores = {}
            for v in R:
                C_v = np.zeros((n, n))
                for u in S:
                    idx2 = edge_idx.get(edge_key(u, v))
                    if idx2 is not None:
                        C_v += x_edges[idx2]
                Y = Bsqrt @ C_v @ Bsqrt.T
                Y = 0.5 * (Y + Y.T)
                scores[v] = float(np.max(np.linalg.eigvalsh(Y)))
            best_v = min(R, key=lambda v: (scores[v], v))
            S.append(best_v); S_set.add(best_v)
            for u in S[:-1]:
                idx2 = edge_idx.get(edge_key(u, best_v))
                if idx2 is not None:
                    M += x_edges[idx2]

print(f"\nMax operator bound: {max_op_bound:.6f}")
print(f"  Witness: {max_op_witness}")
print(f"  Bound < 0.5: {'YES' if max_op_bound < 0.5 else 'NO'}")
