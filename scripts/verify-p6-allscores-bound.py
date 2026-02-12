#!/usr/bin/env python3
"""Verify that ALL scores stay ≤ 1 within c_step = 1/3 horizon.

Key finding: at c_step = 1/3, empirically max_v ||Y_t(v)|| < 1 for ALL
nontrivial steps across all tested graph families. This is STRONGER than
GPL-H (which only needs min_v < 1) and would close it immediately.

This script extends the check to larger n (up to 96) and reports:
  - Number of nontrivial rows
  - Number of rows with any score > 1
  - Worst max_score across all nontrivial rows
  - The trace-sum bound dbar and whether dbar < 1
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load base module
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


def analyze_step(inst, c_step):
    """Run greedy and collect per-step max_score, dbar, gbar."""
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    m0 = len(I0)

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

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

        scores = []
        traces = []
        best_v, best_score = None, float("inf")

        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            evals = np.linalg.eigvalsh(Y_v)
            s_v = float(max(evals[-1], 0.0))
            scores.append(s_v)
            d_v = float(np.sum(evals[evals > 0]))
            traces.append(d_v)
            if s_v < best_score:
                best_score = s_v
                best_v = v

        s_arr = np.array(scores)
        d_arr = np.array(traces)
        active = s_arr > 1e-10
        n_active = int(np.sum(active))

        if n_active > 0:
            dbar = float(np.mean(d_arr[active]))
            g_arr = d_arr[active] / np.maximum(s_arr[active], 1e-15)
            gbar = float(np.mean(g_arr))
            rows.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "active": n_active, "r_t": len(R_t),
                "min_score": float(np.min(s_arr)),
                "max_score": float(np.max(s_arr[active])),
                "mean_score": float(np.mean(s_arr[active])),
                "dbar": dbar, "gbar": gbar,
                "ratio": dbar / max(gbar, 1e-15),
                "M_norm": float(np.linalg.norm(M_t, ord=2)),
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
    case2b = 0

    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            case2b += 1
            all_rows.extend(analyze_step(inst, args.c_step))

    # Filter nontrivial
    nontrivial = [r for r in all_rows if r["min_score"] > 1e-12]
    above1 = [r for r in nontrivial if r["max_score"] > 1.0]
    dbar_above1 = [r for r in nontrivial if r["dbar"] >= 1.0]

    print("=" * 80)
    print(f"ALL-SCORES BOUND CHECK  (c_step = {args.c_step:.4f}, nmax = {args.nmax})")
    print("=" * 80)
    print(f"Case-2b instances:     {case2b}")
    print(f"Total step rows:       {len(all_rows)}")
    print(f"Nontrivial rows:       {len(nontrivial)}")
    print()

    if nontrivial:
        max_scores = [r["max_score"] for r in nontrivial]
        dbars = [r["dbar"] for r in nontrivial]
        ratios = [r["ratio"] for r in nontrivial]
        print(f"max_score:  worst = {max(max_scores):.6f}  mean = {np.mean(max_scores):.6f}")
        print(f"dbar:       worst = {max(dbars):.6f}  mean = {np.mean(dbars):.6f}")
        print(f"ratio:      worst = {max(ratios):.6f}  mean = {np.mean(ratios):.6f}")
        print()
        print(f"Rows with ANY score > 1:   {len(above1)}")
        print(f"Rows with dbar >= 1:       {len(dbar_above1)}")

    # Show worst rows
    if nontrivial:
        print()
        worst = sorted(nontrivial, key=lambda r: r["max_score"], reverse=True)[:10]
        print("Top 10 worst max_score rows:")
        for r in worst:
            print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
                  f"max_s={r['max_score']:.6f} dbar={r['dbar']:.6f} "
                  f"ratio={r['ratio']:.6f} ||M||={r['M_norm']:.4f}")

    # Per-family summary
    if nontrivial:
        print()
        families = sorted(set(r["graph"].split("_")[0] for r in nontrivial))
        print("Per-family worst max_score:")
        for fam in families:
            fam_rows = [r for r in nontrivial
                        if r["graph"].startswith(fam + "_") or r["graph"] == fam]
            if fam_rows:
                wms = max(r["max_score"] for r in fam_rows)
                wdb = max(r["dbar"] for r in fam_rows)
                print(f"  {fam:<12} rows={len(fam_rows):>4} "
                      f"worst_max_score={wms:.6f} worst_dbar={wdb:.6f}")

    # n-scaling check
    if nontrivial:
        print()
        print("Scaling with n (worst max_score per n):")
        ns = sorted(set(r["n"] for r in nontrivial))
        for n_val in ns:
            n_rows = [r for r in nontrivial if r["n"] == n_val]
            if n_rows:
                wms = max(r["max_score"] for r in n_rows)
                wdb = max(r["dbar"] for r in n_rows)
                print(f"  n={n_val:>3}: rows={len(n_rows):>4} "
                      f"worst_max_score={wms:.6f} worst_dbar={wdb:.6f}")


if __name__ == "__main__":
    main()
