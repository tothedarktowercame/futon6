#!/usr/bin/env python3
"""
Direction C probe for GPL-H:
- Build p_v(x) = det(xI - Y_t(v)) for each available vertex at each step.
- Build averaged polynomial pbar_t(x) = (1/r_t) sum_v p_v(x).
- Measure largest real root of pbar_t and compare against min_v ||Y_t(v)||.
- Diagnose common-interlacing plausibility via pairwise interlacing tests.

This is a computational diagnostic, not a proof.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np


# --- Graph generators (kept aligned with verify-p6-gpl-h.py) ---

def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def cycle_graph(n):
    return [(i, (i + 1) % n) for i in range(n)]


def barbell_graph(k):
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, 2 * k):
        for j in range(i + 1, 2 * k):
            edges.append((i, j))
    edges.append((k - 1, k))
    return 2 * k, edges


def erdos_renyi(n, p, rng):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def random_regular(n, d, rng):
    edges_set = set()
    for _ in range(d * n * 5):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges_set.add((min(u, v), max(u, v)))
    adj = [0] * n
    kept = []
    for u, v in sorted(edges_set):
        if adj[u] < d and adj[v] < d:
            kept.append((u, v))
            adj[u] += 1
            adj[v] += 1
    return kept


def dumbbell_graph(k):
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    bridge_start = k
    edges.append((k - 1, bridge_start))
    edges.append((bridge_start, bridge_start + 1))
    for i in range(bridge_start + 1, bridge_start + 1 + k):
        for j in range(i + 1, bridge_start + 1 + k):
            edges.append((i, j))
    return bridge_start + 1 + k, edges


def disjoint_cliques(k, num_cliques):
    n = k * num_cliques
    edges = []
    for c in range(num_cliques):
        base = c * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((base + i, base + j))
    for c in range(num_cliques - 1):
        edges.append((c * k + k - 1, (c + 1) * k))
    return n, edges


# --- Spectral primitives ---

def graph_laplacian(n, edges, weights=None):
    L = np.zeros((n, n))
    if weights is None:
        weights = [1.0] * len(edges)
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def pseudo_sqrt_inv(L):
    eigvals, eigvecs = np.linalg.eigh(L)
    d = len(eigvals)
    Lphalf = np.zeros((d, d))
    for i in range(d):
        if eigvals[i] > 1e-10:
            Lphalf += (1.0 / np.sqrt(eigvals[i])) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return Lphalf


def compute_edge_matrices(n, edges, Lphalf, weights=None):
    if weights is None:
        weights = [1.0] * len(edges)
    X_edges = []
    taus = []
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        b = np.zeros(n)
        b[u] = 1.0
        b[v] = -1.0
        z = Lphalf @ b
        Xe = w * np.outer(z, z)
        tau = w * np.dot(z, z)
        X_edges.append(Xe)
        taus.append(tau)
    return X_edges, taus


@dataclass
class Case2bInstance:
    n: int
    edges: list
    epsilon: float
    I0: list
    X_edges: list
    taus: list
    alpha_I: float
    graph_name: str = ""


def find_case2b_instance(n, edges, epsilon, graph_name=""):
    L = graph_laplacian(n, edges)
    Lphalf = pseudo_sqrt_inv(L)
    X_edges, taus = compute_edge_matrices(n, edges, Lphalf)

    heavy_edges_idx = [i for i, t in enumerate(taus) if t > epsilon]
    adj_heavy = [set() for _ in range(n)]
    for idx in heavy_edges_idx:
        u, v = edges[idx]
        adj_heavy[u].add(v)
        adj_heavy[v].add(u)

    I_set = set()
    vertices = list(range(n))
    np.random.shuffle(vertices)
    for v in vertices:
        if all(u not in I_set for u in adj_heavy[v]):
            I_set.add(v)

    if len(I_set) < epsilon * n / 3 * 0.8:
        return None

    I = sorted(I_set)

    edge_in_I = []
    for idx, (u, v) in enumerate(edges):
        if u in I_set and v in I_set:
            edge_in_I.append(idx)

    if not edge_in_I:
        return None

    M_I = sum(X_edges[idx] for idx in edge_in_I)
    alpha_I = np.linalg.norm(M_I, ord=2)
    if alpha_I <= epsilon:
        return None

    ell = {}
    for v in I:
        ell_v = 0.0
        for idx, (u, w) in enumerate(edges):
            if (u == v and w in I_set) or (w == v and u in I_set):
                ell_v += taus[idx]
        ell[v] = ell_v

    T_I = sum(taus[idx] for idx in edge_in_I)
    D_bound = 4 * T_I / len(I) if len(I) > 0 else 0
    I0 = [v for v in I if ell[v] <= max(D_bound * 2, 12 / epsilon)]
    if len(I0) < len(I) / 3:
        I0 = I

    return Case2bInstance(
        n=n,
        edges=edges,
        epsilon=epsilon,
        I0=I0,
        X_edges=X_edges,
        taus=taus,
        alpha_I=alpha_I,
        graph_name=graph_name,
    )


# --- Direction C diagnostics ---

def largest_real_root(coeffs, imag_tol=1e-7):
    roots = np.roots(coeffs)
    real_like = [r.real for r in roots if abs(r.imag) <= imag_tol]
    if real_like:
        return max(real_like), 0
    return max(r.real for r in roots), sum(1 for r in roots if abs(r.imag) > imag_tol)


def interlace_pair(a, b, tol=1e-8):
    """Check interlacing for two ascending root lists of same length."""
    d = len(a)

    def check(ab_first=True):
        if ab_first:
            # a1 <= b1 <= a2 <= b2 <= ...
            for i in range(d):
                lo = a[i]
                hi = a[i + 1] if i + 1 < d else None
                if b[i] + tol < lo:
                    return False
                if hi is not None and b[i] > hi + tol:
                    return False
            return True
        # b1 <= a1 <= b2 <= a2 <= ...
        for i in range(d):
            lo = b[i]
            hi = b[i + 1] if i + 1 < d else None
            if a[i] + tol < lo:
                return False
            if hi is not None and a[i] > hi + tol:
                return False
        return True

    return check(True) or check(False)


def analyze_instance_direction_c(inst: Case2bInstance, c_step=0.5, max_pairwise_r=18):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    d = n

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    m0 = len(I0)
    T = min(int(c_step * eps * n), m0 - 1)
    T = max(T, 1)

    S_t = []
    S_set = set()
    M_t = np.zeros((d, d))

    step_stats = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        headroom = eps * np.eye(d) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(d))

        polys = []
        eig_roots = []
        scores = []

        best_v = None
        best_score = float("inf")

        for v in R_t:
            C_v = np.zeros((d, d))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            ev = np.linalg.eigvalsh(Y_v)
            ev = np.clip(ev, 0.0, None)
            score = float(ev[-1])

            coeff = np.poly(ev)  # monic coefficients
            polys.append(coeff)
            eig_roots.append(ev)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_v = v

        pbar = np.mean(np.vstack(polys), axis=0)
        pbar_lrr, n_complex = largest_real_root(pbar)

        # Pairwise interlacing diagnostic on the first min(r_t, max_pairwise_r) vertices
        k = min(r_t, max_pairwise_r)
        pairs = list(combinations(range(k), 2))
        n_pairs = len(pairs)
        n_interlacing = 0
        for i, j in pairs:
            if interlace_pair(eig_roots[i], eig_roots[j]):
                n_interlacing += 1

        step_stats.append(
            {
                "t": t,
                "r_t": r_t,
                "min_score": float(min(scores) if scores else 0.0),
                "max_score": float(max(scores) if scores else 0.0),
                "pbar_largest_real_root": float(pbar_lrr),
                "pbar_minus_min": float(pbar_lrr - (min(scores) if scores else 0.0)),
                "pbar_lt_1": bool(pbar_lrr < 1.0),
                "complex_roots_count": int(n_complex),
                "pairwise_checked": n_pairs,
                "pairwise_interlacing_count": n_interlacing,
                "pairwise_interlacing_rate": float(n_interlacing / n_pairs) if n_pairs else 1.0,
            }
        )

        # Greedy update from existing verifier: add best vertex, then add edges to existing S_t
        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return step_stats


def build_suite(nmax, rng):
    suite = []
    for n in range(8, nmax + 1, 4):
        suite.append((f"K_{n}", n, complete_graph(n)))
        suite.append((f"C_{n}", n, cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = barbell_graph(k)
            suite.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = dumbbell_graph(n // 3)
            suite.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = disjoint_cliques(n // 3, 3)
            suite.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            e = erdos_renyi(n, p_er, rng)
            if len(e) > n:
                suite.append((f"ER_{n}_p{p_er}", n, e))
        for d_rr in [4, 6]:
            if d_rr < n:
                e = random_regular(n, d_rr, rng)
                if len(e) > n:
                    suite.append((f"RandReg_{n}_d{d_rr}", n, e))
    return suite


def main():
    ap = argparse.ArgumentParser(description="Direction C probe for GPL-H")
    ap.add_argument("--nmax", type=int, default=24)
    ap.add_argument("--eps", nargs="+", type=float, default=[0.12, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--c-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-instances", type=int, default=0,
                    help="If >0, stop after this many Case-2b instances")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    suite = build_suite(args.nmax, rng)

    all_step_rows = []
    n_case2b = 0

    for graph_name, n, edges in suite:
        for eps in args.eps:
            inst = find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            n_case2b += 1
            rows = analyze_instance_direction_c(inst, c_step=args.c_step)
            for row in rows:
                row["graph"] = graph_name
                row["n"] = n
                row["eps"] = eps
                all_step_rows.append(row)

            if args.limit_instances > 0 and n_case2b >= args.limit_instances:
                break
        if args.limit_instances > 0 and n_case2b >= args.limit_instances:
            break

    print("=" * 78)
    print("DIRECTION C PROBE: averaged characteristic polynomial diagnostics")
    print("=" * 78)
    print(f"Case-2b instances: {n_case2b}")
    print(f"Analyzed steps:    {len(all_step_rows)}")

    if not all_step_rows:
        print("No steps analyzed.")
        return

    pbar_roots = np.array([r["pbar_largest_real_root"] for r in all_step_rows])
    min_scores = np.array([r["min_score"] for r in all_step_rows])
    interlace_rates = np.array([r["pairwise_interlacing_rate"] for r in all_step_rows])

    print("\nGlobal stats:")
    print(f"  max min_score:                 {np.max(min_scores):.6f}")
    print(f"  max pbar largest real root:    {np.max(pbar_roots):.6f}")
    print(f"  median pbar largest real root: {np.median(pbar_roots):.6f}")
    print(f"  max (pbar - min_score):        {np.max(pbar_roots - min_scores):.6f}")
    print(f"  min (pbar - min_score):        {np.min(pbar_roots - min_scores):.6f}")
    print(f"  steps with pbar root < 1:      {int(np.sum(pbar_roots < 1.0))}/{len(pbar_roots)}")
    print(f"  steps with full pairwise interlacing in sample:")
    print(f"                                {int(np.sum(interlace_rates >= 0.999999))}/{len(interlace_rates)}")

    # Top problematic rows by pbar root
    top = sorted(all_step_rows, key=lambda r: r["pbar_largest_real_root"], reverse=True)[:20]
    print("\nTop 20 steps by largest real root of pbar:")
    print(f"{'graph':<20} {'n':>3} {'eps':>5} {'t':>3} {'r_t':>4} {'min':>8} {'pbar':>8} {'dlt':>8} {'int%':>7}")
    print("-" * 80)
    for r in top:
        print(
            f"{r['graph']:<20} {r['n']:>3} {r['eps']:>5.2f} {r['t']:>3} {r['r_t']:>4} "
            f"{r['min_score']:>8.4f} {r['pbar_largest_real_root']:>8.4f} "
            f"{r['pbar_minus_min']:>8.4f} {100.0*r['pairwise_interlacing_rate']:>6.1f}%"
        )


if __name__ == "__main__":
    main()
