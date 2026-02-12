#!/usr/bin/env python3
"""Test whether dbar_fresh < 1 can be PROVED from leverage structure alone.

When M_t = 0: dbar = tr(Q)/(eps*|A_t|) = (Σ_{e crossing} τ_e)/(eps*|A_t|).

For K_k at step t: dbar_fresh = 2t/(k*eps).
At t = k*eps/3: dbar_fresh = 2/3.

Question: does the identity Σ_e τ_e = n-1, combined with the fact that
I_0 only has light edges (τ ≤ eps), give dbar_fresh < 1 universally?

Key ratio: (Σ_{u∈S_t} ℓ_u^{I_0}) / (eps * |A_t|)
where ℓ_u^{I_0} = leverage degree of u restricted to I_0.
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


def analyze(inst, c_step):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
    m0 = len(I0)

    # Build I0-subgraph adjacency
    edge_idx = {}
    I0_adj = {v: [] for v in I0}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        if u in I0_set and v in I0_set:
            I0_adj[u].append((v, inst.taus[idx]))
            I0_adj[v].append((u, inst.taus[idx]))

    # Compute leverage degrees within I0
    lev_deg = {}
    for v in I0:
        lev_deg[v] = sum(tau for _, tau in I0_adj[v])

    # Full leverage degrees (all edges)
    full_lev = {}
    for v in I0:
        s = 0.0
        for idx, (u, w) in enumerate(inst.edges):
            if u == v or w == v:
                s += inst.taus[idx]
        full_lev[v] = s

    # Total internal leverage
    total_internal_lev = sum(inst.taus[idx] for idx in range(len(inst.edges))
                             if inst.edges[idx][0] in I0_set and inst.edges[idx][1] in I0_set)

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

        # Compute scores to find best vertex
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
            # Compute crossing leverage and check S_t independence
            crossing_lev = sum(lev_deg.get(u, 0) for u in S_t)
            s_independent = True
            for i, u1 in enumerate(S_t):
                for u2 in S_t[i+1:]:
                    key = (min(u1, u2), max(u1, u2))
                    if key in edge_idx and inst.edges[edge_idx[key]][0] in I0_set and inst.edges[edge_idx[key]][1] in I0_set:
                        s_independent = False

            dbar_fresh_bound = crossing_lev / (eps * n_active) if n_active > 0 else 0
            rows.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "m0": m0, "r_t": len(R_t), "active": n_active,
                "S_independent": s_independent,
                "crossing_lev": crossing_lev,
                "dbar_fresh_bound": dbar_fresh_bound,
                "total_internal_lev": total_internal_lev,
                "avg_lev_deg": np.mean([lev_deg[v] for v in I0]),
                "avg_full_lev": np.mean([full_lev[v] for v in I0]),
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
    ap.add_argument("--nmax", type=int, default=64)
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
            all_rows.extend(analyze(inst, args.c_step))

    if not all_rows:
        print("No rows")
        return

    print(f"Rows: {len(all_rows)}")
    ind = [r for r in all_rows if r["S_independent"]]
    print(f"S_t independent in I0: {len(ind)}/{len(all_rows)} "
          f"({100*len(ind)/len(all_rows):.1f}%)")

    # dbar_fresh_bound analysis
    bounds = [r["dbar_fresh_bound"] for r in all_rows]
    print(f"\ndbar_fresh_bound (= Σ ℓ_u / (ε·|A_t|)):")
    print(f"  max  = {max(bounds):.6f}")
    print(f"  mean = {np.mean(bounds):.6f}")
    print(f"  ≥ 1: {sum(1 for b in bounds if b >= 1)}/{len(bounds)}")

    # Leverage structure
    avg_int = np.mean([r["total_internal_lev"] for r in all_rows])
    avg_lev = np.mean([r["avg_lev_deg"] for r in all_rows])
    avg_full = np.mean([r["avg_full_lev"] for r in all_rows])
    print(f"\nLeverage structure:")
    print(f"  avg total_internal_lev: {avg_int:.3f}")
    print(f"  avg leverage_degree (in I0): {avg_lev:.3f}")
    print(f"  avg full_leverage_degree: {avg_full:.3f}")

    worst = sorted(all_rows, key=lambda r: r["dbar_fresh_bound"], reverse=True)[:10]
    print("\nWorst dbar_fresh_bound rows:")
    for r in worst:
        print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
              f"m0={r['m0']:>3} active={r['active']:>3} "
              f"Σℓ={r['crossing_lev']:.3f} bound={r['dbar_fresh_bound']:.4f} "
              f"indep={r['S_independent']} int_lev={r['total_internal_lev']:.1f}")


if __name__ == "__main__":
    main()
