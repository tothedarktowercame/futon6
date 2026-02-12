#!/usr/bin/env python3
"""Exact dbar computation for K_k including M_t amplification.

For K_k, the eigenstructure is computable exactly:
- M_t eigenvalues: t/k (multiplicity t-1), 0 (multiplicity k-t+1)
- W eigenvalues in M_t eigenbasis: (k-t)/k in V_S, t/k in V_R, 1 in v_bridge

This gives an exact dbar formula:
  dbar = (t-1)/(kε-t) + (t+1)/(kε)

We verify this formula against numerical computation and test whether
it generalizes: is K_k the extremal graph for dbar under H1-H2'?
"""

import numpy as np
import sys
import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def kk_exact_dbar(k, t, eps):
    """Exact dbar for K_k at step t (all t vertices selected, S_t independent for t=1)."""
    if t == 0:
        return 0
    # At step t, S_t has t vertices forming K_t internally.
    # M_t eigenvalues: t/k with multiplicity t-1, 0 with multiplicity k-t+1
    # W eigenvalues: (k-t)/k in V_S (t-1 dim), t/k in V_R (k-t-1 dim), 1 in v_bridge
    # B_t eigenvalues: 1/(eps - t/k) in V_S, 1/eps elsewhere (M_t=0 directions)

    if eps <= t / k:
        return float('inf')  # barrier violated

    r_t = k - t

    # tr(B_t W):
    # V_S: (t-1) * (k-t)/k * 1/(eps - t/k)
    # V_R: (k-t-1) * t/k * 1/eps
    # v_bridge: 1 * 1 * 1/eps
    trBW = (t - 1) * (k - t) / (k * (eps - t / k)) + \
           ((k - t - 1) * t / k + 1) / eps

    dbar = trBW / r_t
    return dbar


def kk_dbar_formula(k, t, eps):
    """Simplified: dbar = (t-1)/(kε-t) + (t+1)/(kε) for K_k."""
    if eps <= t / k:
        return float('inf')
    return (t - 1) / (k * eps - t) + (t + 1) / (k * eps)


def main():
    np.random.seed(42)

    print("=" * 70)
    print("K_k EXACT dbar WITH BARRIER AMPLIFICATION")
    print("=" * 70)

    # Verify formula against numerical computation
    print("\n--- Formula verification ---\n")
    for k in [12, 20, 32, 48, 60, 96]:
        n = k
        edges = base.complete_graph(n)
        L = base.graph_laplacian(n, edges)
        Lphalf = base.pseudo_sqrt_inv(L)
        X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)
        tau_val = taus[0]

        for eps in [tau_val + 0.001, 0.15, 0.2, 0.25, 0.3]:
            if eps <= tau_val or eps >= 1:
                continue

            T = max(1, min(int(eps * n / 3), n - 1))

            # Run greedy and get dbar at each step
            edge_idx = {}
            for idx, (u, v) in enumerate(edges):
                edge_idx[(u, v)] = idx
                edge_idx[(v, u)] = idx

            S_t = []
            S_set = set()
            M_t = np.zeros((n, n))

            for t in range(T):
                R_t = [v for v in range(n) if v not in S_set]
                r_t = len(R_t)
                if r_t == 0:
                    break

                headroom = eps * np.eye(n) - M_t
                if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
                    break

                B_t = np.linalg.inv(headroom)
                Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

                # Compute dbar numerically
                total_trace = 0
                for v in R_t:
                    C_v = np.zeros((n, n))
                    for u in S_t:
                        key = (min(u, v), max(u, v))
                        if key in edge_idx:
                            C_v += X_edges[edge_idx[key]]
                    Y_v = Bsqrt @ C_v @ Bsqrt.T
                    total_trace += np.trace(Y_v)
                dbar_num = total_trace / r_t if r_t > 0 else 0

                # Exact formula
                dbar_exact = kk_exact_dbar(k, t, eps) if t > 0 else 0
                dbar_simple = kk_dbar_formula(k, t, eps) if t > 0 else 0

                if t > 0 and t <= 5:
                    match = abs(dbar_num - dbar_exact) < 0.01
                    print(f"  K_{k:>2} eps={eps:.3f} t={t}: "
                          f"numerical={dbar_num:.6f} exact={dbar_exact:.6f} "
                          f"simple={dbar_simple:.6f} {'✓' if match else '✗'}")

                # Select min-score vertex
                best_v = None
                best_score = float('inf')
                for v in R_t:
                    C_v = np.zeros((n, n))
                    for u in S_t:
                        key = (min(u, v), max(u, v))
                        if key in edge_idx:
                            C_v += X_edges[edge_idx[key]]
                    Y_v = Bsqrt @ C_v @ Bsqrt.T
                    s = float(np.linalg.norm(Y_v, ord=2))
                    if s < best_score:
                        best_score = s
                        best_v = v

                if best_v is None:
                    break
                S_t.append(best_v)
                S_set.add(best_v)
                for u in S_t[:-1]:
                    key = (min(best_v, u), max(best_v, u))
                    if key in edge_idx:
                        M_t += X_edges[edge_idx[key]]

    # K_k dbar at horizon (t = εk/3)
    print("\n--- K_k dbar at greedy horizon ---\n")
    print(f"  {'k':>4} {'ε':>6} {'t':>3} {'dbar':>8} {'5/6+2/kε':>10} {'<1?':>4}")
    for k in [12, 20, 32, 48, 60, 80, 96, 128, 256, 1000]:
        tau = 2 / k
        for eps in [tau + 0.001, 0.15, 0.2, 0.25, 0.3]:
            if eps <= tau or eps >= 1:
                continue
            t = int(eps * k / 3)
            if t < 1:
                t = 1
            dbar = kk_dbar_formula(k, t, eps)
            approx = 5 / 6 + 2 / (k * eps)
            ok = "✓" if dbar < 1 else "✗"
            print(f"  {k:>4} {eps:>6.3f} {t:>3} {dbar:>8.4f} {approx:>10.4f} {ok:>4}")

    # The universal bound: dbar ≤ 5/6 + O(1/(kε))
    # For kε ≥ 12: dbar ≤ 5/6 + 1/6 = 1.0000 (borderline!)
    # For kε ≥ 13: dbar ≤ 5/6 + 2/13 = 0.987 < 1 ✓
    print("\n--- K_k extremality test ---")
    print("  Testing: is K_k the worst graph for dbar at each step?\n")

    rng = np.random.default_rng(42)
    # Compare K_k dbar with other graphs at matching parameters
    max_excess = 0  # max(dbar_other - dbar_Kk)
    for n in [20, 32, 48, 60]:
        graphs = [
            (f"K_{n}", n, base.complete_graph(n)),
        ]
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            graphs.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            cn, ce = base.disjoint_cliques(n // 3, 3)
            graphs.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                graphs.append((f"ER_{n}_p{p_er}", n, er_edges))

        for eps in [0.15, 0.2, 0.25, 0.3]:
            kk_dbar_at = {}
            other_dbars = {}

            for gname, gn, gedges in graphs:
                L = base.graph_laplacian(gn, gedges)
                Lphalf = base.pseudo_sqrt_inv(L)
                X_edges, taus = base.compute_edge_matrices(gn, gedges, Lphalf)

                if any(t > eps for t in taus) and gname.startswith("K_"):
                    continue  # Skip if K_k doesn't satisfy H1

                edge_idx = {}
                for idx, (u, v) in enumerate(gedges):
                    edge_idx[(u, v)] = idx
                    edge_idx[(v, u)] = idx

                T = max(1, min(int(eps * gn / 3), gn - 1))
                S_t = []
                S_set = set()
                M_t_mat = np.zeros((gn, gn))

                for t in range(T):
                    R_t = [v for v in range(gn) if v not in S_set]
                    r_t = len(R_t)
                    if r_t == 0:
                        break

                    headroom = eps * np.eye(gn) - M_t_mat
                    if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
                        break

                    B_t = np.linalg.inv(headroom)
                    Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(gn))

                    total_trace = 0
                    best_v, best_s = None, float('inf')
                    for v in R_t:
                        C_v = np.zeros((gn, gn))
                        for u in S_t:
                            key = (min(u, v), max(u, v))
                            if key in edge_idx:
                                C_v += X_edges[edge_idx[key]]
                        Y_v = Bsqrt @ C_v @ Bsqrt.T
                        total_trace += float(np.trace(Y_v))
                        s = float(np.linalg.norm(Y_v, ord=2))
                        if s < best_s:
                            best_s = s
                            best_v = v

                    dbar = total_trace / r_t if r_t > 0 else 0

                    if t > 0:
                        if gname.startswith("K_") and not "_" in gname[2:]:
                            kk_dbar_at[t] = dbar
                        else:
                            key = (gname, t)
                            other_dbars[key] = dbar

                    if best_v is None:
                        break
                    S_t.append(best_v)
                    S_set.add(best_v)
                    for u in S_t[:-1]:
                        key = (min(best_v, u), max(best_v, u))
                        if key in edge_idx:
                            M_t_mat += X_edges[edge_idx[key]]

            # Compare
            for (gname, t), d_other in other_dbars.items():
                if t in kk_dbar_at:
                    excess = d_other - kk_dbar_at[t]
                    if excess > max_excess:
                        max_excess = excess
                    if excess > 0.01:
                        print(f"  n={n} eps={eps} t={t}: {gname} dbar={d_other:.4f} > "
                              f"K_{n} dbar={kk_dbar_at[t]:.4f} (excess={excess:.4f})")

    if max_excess <= 0.01:
        print(f"  K_k is extremal (or within 0.01) for all tested cases!")
    else:
        print(f"  Max excess over K_k: {max_excess:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: EXACT K_k dbar FORMULA")
    print(f"{'='*70}")
    print(f"""
For K_k at step t with barrier:

  dbar(K_k, t) = (t-1)/(kε-t) + (t+1)/(kε)

At t = εk/3 (greedy horizon):

  dbar → 5/6 + 2/(kε) as k → ∞

This is < 1 for kε ≥ 12 (which follows from H1: kε ≥ 2 only gives
dbar ≤ 5/6 + 1 = 1.83 at kε = 2).

For kε ≥ 12: the K_k formula gives dbar ≤ 1.
For kε ≥ 14: dbar ≤ 5/6 + 1/7 = 0.976 < 1.

The formula includes FULL barrier amplification (M_t ≠ 0 case).

If K_k is extremal (dbar is maximized by K_k among all H1-H2' graphs),
then GPL-H' follows for kε ≥ 14 (or any fixed lower bound on kε).
""")


if __name__ == "__main__":
    main()
