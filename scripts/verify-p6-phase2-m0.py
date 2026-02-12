#!/usr/bin/env python3
"""Check whether Phase 2 instances always have m_0 ≈ n.

If Phase 2 is reached (all vertices dominated) only when m_0/n is large,
then the double-counting argument (dbar ≤ 2t/(ε(m_0-t)) for m_0 ≈ n)
closes GPL-H.

Case decomposition:
  - Phase 1 covers horizon → GPL-H trivially (min score = 0)
  - Phase 2 reached → need m_0 large for double-counting

If these two cases are exhaustive, GPL-H is closed.
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


def check_phase2(inst, c_step):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
    m0 = len(I0)

    # I0-subgraph adjacency
    I0_adj = {v: set() for v in I0}
    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        if u in I0_set and v in I0_set:
            I0_adj[u].add(v)
            I0_adj[v].add(u)

    T = max(1, min(int(c_step * eps * n), m0 - 1))
    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    phase2_steps = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        if not R_t:
            break
        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        zero_count = 0
        best_v, best_score = None, float("inf")
        for v in R_t:
            has_edge_to_St = any(u in S_set for u in I0_adj[v])
            if not has_edge_to_St:
                zero_count += 1

            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            s_v = float(np.linalg.norm(Bsqrt @ C_v @ Bsqrt.T, ord=2))
            if s_v < best_score:
                best_score = s_v
                best_v = v

        # Phase 2: zero_count = 0 (all dominated)
        if zero_count == 0 and best_score > 1e-12:
            # Compute the double-counting bound
            total_int_lev = sum(inst.taus[idx] for idx, (u, v) in enumerate(inst.edges)
                                if u in I0_set and v in I0_set)
            r_t = len(R_t)
            n_active = r_t  # Phase 2: all active
            dbar_dc = 2 * total_int_lev * len(S_t) / (m0 * eps * n_active)

            phase2_steps.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "m0": m0, "m0_over_n": m0 / n,
                "total_int_lev": total_int_lev,
                "r_t": r_t,
                "dbar_dc": dbar_dc,
                "max_score": best_score,
            })

        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return phase2_steps


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

    all_p2 = []
    total_instances = 0
    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            total_instances += 1
            all_p2.extend(check_phase2(inst, args.c_step))

    print(f"Total Case-2b instances: {total_instances}")
    print(f"Phase 2 step-rows: {len(all_p2)}")

    if all_p2:
        m0_ratios = [r["m0_over_n"] for r in all_p2]
        print(f"\nm0/n in Phase 2 steps:")
        print(f"  min  = {min(m0_ratios):.4f}")
        print(f"  mean = {np.mean(m0_ratios):.4f}")
        print(f"  max  = {max(m0_ratios):.4f}")

        # The double-counting bound
        dbar_dcs = [r["dbar_dc"] for r in all_p2]
        print(f"\nDouble-counting dbar bound in Phase 2:")
        print(f"  max  = {max(dbar_dcs):.6f}")
        print(f"  mean = {np.mean(dbar_dcs):.6f}")
        print(f"  ≥ 1: {sum(1 for d in dbar_dcs if d >= 1)}/{len(dbar_dcs)}")

        # Graph families in Phase 2
        fams = sorted(set(r["graph"].split("_")[0] for r in all_p2))
        print(f"\nFamilies reaching Phase 2: {fams}")

        # Show all Phase 2 step-rows
        print(f"\nAll Phase 2 steps (worst dbar_dc first):")
        for r in sorted(all_p2, key=lambda x: x["dbar_dc"], reverse=True)[:20]:
            print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
                  f"m0={r['m0']:>3} m0/n={r['m0_over_n']:.3f} "
                  f"int_lev={r['total_int_lev']:.1f} "
                  f"dbar_dc={r['dbar_dc']:.4f}")


if __name__ == "__main__":
    main()
