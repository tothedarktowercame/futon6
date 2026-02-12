#!/usr/bin/env python3
"""Test the double-counting argument for dbar_fresh < 1.

Key identity: for an independent set S_t in I_0 with M_t = 0,

    dbar_fresh = Σ_{u∈S_t} ℓ_u / (ε · |A_t|)

where ℓ_u = leverage degree of u within I_0.

The double-counting argument:
  - Σ_{all v ∈ I_0} ℓ_v = 2 · Σ_{e∈E(I_0)} τ_e  (each edge counted twice)
  - Since S_t is independent: Σ_{u∈S_t} ℓ_u = Σ_{e crossing from S_t} τ_e
  - The ratio (Σ_{u∈S_t} ℓ_u) / (|A_t| · ε) is what we need < 1.

This script checks the KEY RATIO:
    (Σ_{u∈S_t} ℓ_u) / t vs 2 (the global average leverage degree ≈ 2)

If the greedy selects vertices with below-average leverage degree, then
dbar_fresh ≤ 2t/(ε · |A_t|) which is provably < 1 when |A_t| > 2t/ε.
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def build_suite(nmax, rng):
    suite = []
    for n in range(8, nmax + 1, 4):
        suite.append((f"K_{n}", n, base.complete_graph(n)))
        suite.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            suite.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = base.dumbbell_graph(n // 3)
            suite.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = base.disjoint_cliques(n // 3, 3)
            suite.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                suite.append((f"ER_{n}_p{p_er}", n, er_edges))
    return suite


def analyze_double_counting(inst, c_step):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
    m0 = len(I0)

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    # Compute leverage degree within I0 for each vertex
    lev_deg_I0 = {v: 0.0 for v in I0}
    for idx, (u, v) in enumerate(inst.edges):
        if u in I0_set and v in I0_set:
            lev_deg_I0[u] += inst.taus[idx]
            lev_deg_I0[v] += inst.taus[idx]

    # Total internal leverage and average leverage degree
    total_int_lev = sum(inst.taus[idx] for idx, (u, v) in enumerate(inst.edges)
                        if u in I0_set and v in I0_set)
    avg_lev = 2 * total_int_lev / m0 if m0 > 0 else 0

    T = max(1, min(int(c_step * eps * n), m0 - 1))
    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    rows = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        if not R_t:
            break
        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        best_v, best_score = None, float("inf")
        n_active = 0
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            s_v = float(np.linalg.norm(Bsqrt @ C_v @ Bsqrt.T, ord=2))
            if s_v > 1e-10:
                n_active += 1
            if s_v < best_score:
                best_score = s_v
                best_v = v

        if t > 0 and n_active > 0 and best_score > 1e-12:
            # Sum of leverage degrees of selected vertices
            sum_lev_St = sum(lev_deg_I0[u] for u in S_t)
            avg_lev_St = sum_lev_St / len(S_t)
            dbar_est = sum_lev_St / (eps * n_active)

            # The "provable bound" if avg_lev_St ≤ avg_lev:
            dbar_provable = avg_lev * len(S_t) / (eps * n_active)

            # Check: is avg_lev_St ≤ avg_lev?
            lev_ratio = avg_lev_St / avg_lev if avg_lev > 1e-10 else 0

            rows.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "m0": m0, "active": n_active,
                "sum_lev_St": sum_lev_St,
                "avg_lev_St": avg_lev_St,
                "avg_lev_I0": avg_lev,
                "lev_ratio": lev_ratio,  # avg_lev(S_t) / avg_lev(I_0)
                "dbar_est": dbar_est,
                "dbar_provable": dbar_provable,
            })

        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=96)
    ap.add_argument("--c-step", type=float, default=1/3)
    ap.add_argument("--seed", type=int, default=20260212)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    suite = build_suite(args.nmax, rng)
    eps_list = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

    all_rows = []
    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            all_rows.extend(analyze_double_counting(inst, args.c_step))

    if not all_rows:
        print("No rows")
        return

    print(f"Rows: {len(all_rows)}")

    # Key question: does the greedy select below-average leverage vertices?
    ratios = [r["lev_ratio"] for r in all_rows]
    print(f"\nLeverage ratio (avg_lev(S_t) / avg_lev(I_0)):")
    print(f"  min  = {min(ratios):.4f}")
    print(f"  mean = {np.mean(ratios):.4f}")
    print(f"  max  = {max(ratios):.4f}")
    above_avg = sum(1 for r in ratios if r > 1.0)
    print(f"  S_t above average: {above_avg}/{len(ratios)} ({100*above_avg/len(ratios):.1f}%)")

    # dbar estimates
    dbar_ests = [r["dbar_est"] for r in all_rows]
    dbar_provs = [r["dbar_provable"] for r in all_rows]
    print(f"\ndbar_est (actual): max = {max(dbar_ests):.4f}")
    print(f"dbar_provable (using avg_lev): max = {max(dbar_provs):.4f}")
    print(f"dbar_provable ≥ 1: {sum(1 for d in dbar_provs if d >= 1)}/{len(dbar_provs)}")

    # Worst rows by lev_ratio
    worst = sorted(all_rows, key=lambda r: r["lev_ratio"], reverse=True)[:10]
    print("\nWorst leverage ratio rows:")
    for r in worst:
        print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
              f"m0={r['m0']:>3} ratio={r['lev_ratio']:.4f} "
              f"avg_St={r['avg_lev_St']:.3f} avg_I0={r['avg_lev_I0']:.3f} "
              f"dbar={r['dbar_est']:.4f}")


if __name__ == "__main__":
    main()
