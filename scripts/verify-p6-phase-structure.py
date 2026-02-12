#!/usr/bin/env python3
"""Analyze the phase structure of the barrier greedy.

Phase 1: greedy selects undominated vertices (score 0), M_t stays 0.
Phase 2: all vertices dominated, scores > 0, M_t may grow.

Key questions:
  - How many Phase 1 steps? (= greedy domination number of I_0-subgraph)
  - What fraction of horizon is Phase 1 vs Phase 2?
  - In Phase 2, what is max score and dbar?
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
        for d_rr in [4, 6]:
            if d_rr < n:
                rr_edges = base.random_regular(n, d_rr, rng)
                if len(rr_edges) > n:
                    suite.append((f"RandReg_{n}_d{d_rr}", n, rr_edges))
    return suite


def analyze_phases(inst, c_step):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    m0 = len(I0)

    # Build adjacency within I0
    I0_set = set(I0)
    adj_I0 = {v: set() for v in I0}
    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        if u in I0_set and v in I0_set:
            adj_I0[u].add(v)
            adj_I0[v].add(u)

    T = max(1, min(int(c_step * eps * n), m0 - 1))
    horizon = c_step * eps * n

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    phase1_end = None  # step where Phase 1 ends

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        if not R_t:
            break
        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        # Check how many vertices have score 0 (no edges to S_t in I0)
        zero_score_count = 0
        best_v, best_score = None, float("inf")
        max_score_active = 0.0
        dbar_active = 0.0
        n_active = 0

        for v in R_t:
            has_edge = any(u in S_set for u in adj_I0.get(v, set()))
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            evals = np.linalg.eigvalsh(Y_v)
            s_v = float(max(evals[-1], 0.0))
            d_v = float(np.sum(evals[evals > 0]))

            if s_v < 1e-10:
                zero_score_count += 1
            else:
                n_active += 1
                max_score_active = max(max_score_active, s_v)
                dbar_active += d_v

            if s_v < best_score:
                best_score = s_v
                best_v = v

        if n_active > 0:
            dbar_active /= n_active

        if zero_score_count == 0 and phase1_end is None:
            phase1_end = t

        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    if phase1_end is None:
        phase1_end = T  # never fully dominated

    return {
        "graph": inst.graph_name,
        "n": n, "eps": eps,
        "m0": m0,
        "horizon": horizon,
        "T_actual": T,
        "phase1_end": phase1_end,
        "phase1_frac": phase1_end / max(T, 1),
    }


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

    results = []
    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            results.append(analyze_phases(inst, args.c_step))

    print(f"Instances: {len(results)}")
    print()

    # Phase 1 fraction distribution
    p1_fracs = [r["phase1_frac"] for r in results]
    always_p1 = sum(1 for f in p1_fracs if f >= 1.0)
    print(f"Phase 1 covers entire horizon:  {always_p1}/{len(results)} "
          f"({100*always_p1/len(results):.1f}%)")
    print(f"Phase 1 fraction: min={min(p1_fracs):.3f} "
          f"mean={np.mean(p1_fracs):.3f} median={np.median(p1_fracs):.3f}")

    # |I0| vs horizon
    ratios = [r["m0"] / max(r["horizon"], 1) for r in results]
    print(f"\n|I0| / horizon: min={min(ratios):.3f} mean={np.mean(ratios):.3f}")
    tight = sum(1 for r in ratios if r < 2.0)
    print(f"|I0| < 2*horizon: {tight}/{len(results)}")

    # Instances with short Phase 1
    short_p1 = sorted([r for r in results if r["phase1_frac"] < 1.0],
                       key=lambda r: r["phase1_frac"])[:10]
    if short_p1:
        print("\nInstances with earliest Phase 2:")
        for r in short_p1:
            print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} "
                  f"|I0|={r['m0']:>3} T={r['T_actual']:>2} "
                  f"p1_end={r['phase1_end']:>2} frac={r['phase1_frac']:.3f}")


if __name__ == "__main__":
    main()
