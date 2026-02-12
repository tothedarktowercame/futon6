#!/usr/bin/env python3
"""G1 diagnostic: WHY is m_0/n ≥ 0.95 in Phase 2?

Investigate the heavy-edge structure to find the structural reason.
For each Phase 2 instance, compute:
  (a) |E_H| = number of heavy edges (τ > ε)
  (b) Total heavy leverage = Σ_{e∈E_H} τ_e
  (c) Vertices with no heavy edges (they're guaranteed in I_0)
  (d) Heavy degree distribution
  (e) The "vertex removal" count: how many vertices can heavy edges exclude?
  (f) I_0 construction details: maximal independent set in G_H

Key insight to test: if most vertices have zero heavy degree, then I_0 ≈ V
regardless of the independent-set heuristic.
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


def analyze_heavy_structure(n, edges, eps, taus, I0, graph_name):
    """Analyze the heavy-edge graph structure relative to I_0."""
    I0_set = set(I0)
    m0 = len(I0)

    # Heavy edges
    heavy_idx = [i for i, t in enumerate(taus) if t > eps]
    n_heavy = len(heavy_idx)
    total_heavy_lev = sum(taus[i] for i in heavy_idx)

    # Heavy degree per vertex
    heavy_deg = [0] * n
    for idx in heavy_idx:
        u, v = edges[idx]
        heavy_deg[u] += 1
        heavy_deg[v] += 1

    # Vertices with zero heavy degree — these are in ANY independent set of G_H
    n_zero_heavy = sum(1 for d in heavy_deg if d == 0)

    # Vertices incident to heavy edges
    heavy_vertices = set()
    for idx in heavy_idx:
        u, v = edges[idx]
        heavy_vertices.add(u)
        heavy_vertices.add(v)
    n_heavy_vertices = len(heavy_vertices)

    # How many of I_0 have zero heavy degree?
    n_I0_zero_heavy = sum(1 for v in I0 if heavy_deg[v] == 0)

    # Heavy degree distribution
    max_heavy_deg = max(heavy_deg) if heavy_deg else 0
    avg_heavy_deg = np.mean(heavy_deg) if heavy_deg else 0

    # For vertices in V \ I_0: their heavy connections
    non_I0 = [v for v in range(n) if v not in I0_set]
    n_non_I0 = len(non_I0)

    # Internal leverage (edges within I_0)
    L_int = sum(taus[i] for i, (u, v) in enumerate(edges) if u in I0_set and v in I0_set)
    n_int_edges = sum(1 for i, (u, v) in enumerate(edges) if u in I0_set and v in I0_set)

    return {
        "graph": graph_name, "n": n, "eps": eps,
        "m0": m0, "m0_over_n": m0 / n,
        "n_heavy_edges": n_heavy,
        "total_heavy_lev": total_heavy_lev,
        "heavy_lev_frac": total_heavy_lev / (n - 1) if n > 1 else 0,
        "n_zero_heavy_deg": n_zero_heavy,
        "zero_heavy_frac": n_zero_heavy / n,
        "n_heavy_vertices": n_heavy_vertices,
        "heavy_vert_frac": n_heavy_vertices / n,
        "max_heavy_deg": max_heavy_deg,
        "avg_heavy_deg": avg_heavy_deg,
        "n_non_I0": n_non_I0,
        "n_I0_zero_heavy": n_I0_zero_heavy,
        "L_int": L_int,
        "n_int_edges": n_int_edges,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=96)
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
            row = analyze_heavy_structure(n, edges, eps, inst.taus, inst.I0, graph_name)
            all_rows.append(row)

    if not all_rows:
        print("No Case-2b instances")
        return

    print(f"Case-2b instances: {len(all_rows)}")

    # Key question: what fraction of vertices have ZERO heavy degree?
    zero_fracs = [r["zero_heavy_frac"] for r in all_rows]
    print(f"\nFraction of vertices with ZERO heavy degree:")
    print(f"  min  = {min(zero_fracs):.4f}")
    print(f"  mean = {np.mean(zero_fracs):.4f}")
    print(f"  max  = {max(zero_fracs):.4f}")

    # Fraction of vertices incident to any heavy edge
    heavy_fracs = [r["heavy_vert_frac"] for r in all_rows]
    print(f"\nFraction of vertices incident to heavy edges:")
    print(f"  min  = {min(heavy_fracs):.4f}")
    print(f"  mean = {np.mean(heavy_fracs):.4f}")
    print(f"  max  = {max(heavy_fracs):.4f}")

    # Number of heavy edges
    n_heavy = [r["n_heavy_edges"] for r in all_rows]
    n_vals = [r["n"] for r in all_rows]
    heavy_per_n = [h/n_ for h, n_ in zip(n_heavy, n_vals)]
    print(f"\nHeavy edges / n:")
    print(f"  min  = {min(heavy_per_n):.4f}")
    print(f"  mean = {np.mean(heavy_per_n):.4f}")
    print(f"  max  = {max(heavy_per_n):.4f}")

    # Total heavy leverage as fraction of n-1
    heavy_lev_fracs = [r["heavy_lev_frac"] for r in all_rows]
    print(f"\nHeavy leverage / (n-1):")
    print(f"  min  = {min(heavy_lev_fracs):.4f}")
    print(f"  mean = {np.mean(heavy_lev_fracs):.4f}")
    print(f"  max  = {max(heavy_lev_fracs):.4f}")

    # m0/n vs zero-heavy fraction
    print(f"\nm0/n:")
    m0_ratios = [r["m0_over_n"] for r in all_rows]
    print(f"  min  = {min(m0_ratios):.4f}")
    print(f"  mean = {np.mean(m0_ratios):.4f}")

    # Correlation: zero_heavy_frac vs m0/n
    print(f"\nCorrelation between zero_heavy_frac and m0/n:")
    zf = np.array(zero_fracs)
    m0r = np.array(m0_ratios)
    if np.std(zf) > 0 and np.std(m0r) > 0:
        corr = np.corrcoef(zf, m0r)[0, 1]
        print(f"  Pearson r = {corr:.4f}")

    # Key insight: is m0/n ≥ zero_heavy_frac always?
    gap = [m0 - zf for m0, zf in zip(m0_ratios, zero_fracs)]
    print(f"\nm0/n - zero_heavy_frac:")
    print(f"  min  = {min(gap):.4f}")
    print(f"  mean = {np.mean(gap):.4f}")

    # Worst rows (smallest m0/n)
    worst = sorted(all_rows, key=lambda r: r["m0_over_n"])[:15]
    print(f"\nWorst m0/n instances:")
    for r in worst:
        print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} "
              f"m0/n={r['m0_over_n']:.3f} "
              f"zero_heavy={r['zero_heavy_frac']:.3f} "
              f"|E_H|={r['n_heavy_edges']:>4} "
              f"|E_H|/n={r['n_heavy_edges']/r['n']:.2f} "
              f"heavy_lev_frac={r['heavy_lev_frac']:.3f} "
              f"n_heavy_vert={r['n_heavy_vertices']:>3}")

    # Instances where heavy_vert_frac is large (most vertices have heavy edges)
    dense_heavy = [r for r in all_rows if r["heavy_vert_frac"] > 0.5]
    if dense_heavy:
        print(f"\nInstances where >50% vertices have heavy edges: {len(dense_heavy)}")
        for r in sorted(dense_heavy, key=lambda r: r["heavy_vert_frac"], reverse=True)[:10]:
            print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} "
                  f"heavy_vert_frac={r['heavy_vert_frac']:.3f} "
                  f"m0/n={r['m0_over_n']:.3f} "
                  f"|E_H|/n={r['n_heavy_edges']/r['n']:.2f}")
    else:
        print(f"\nNo instances with >50% heavy vertices")

    # Key derivable bound: n_zero_heavy_deg = vertices with no heavy edges
    # These are in EVERY maximal independent set of G_H.
    # So m_0 >= n_zero_heavy_deg.
    # If n_zero_heavy_deg >= cn, then m_0 >= cn.
    guaranteed_lower = [r["n_zero_heavy_deg"] / r["n"] for r in all_rows]
    print(f"\nGuaranteed I_0 lower bound (zero-heavy-deg / n):")
    print(f"  min  = {min(guaranteed_lower):.4f}")
    print(f"  mean = {np.mean(guaranteed_lower):.4f}")
    print(f"  This is a PROVABLE lower bound on m_0/n")

    # Tight bound: can we prove n_zero_heavy_deg is large?
    # Each heavy edge involves 2 vertices. |E_H| < (n-1)/ε.
    # Number of heavy vertices ≤ 2|E_H| but also ≤ n.
    # Better: n_heavy_vert ≤ min(n, 2|E_H|).
    # So n_zero_heavy ≥ max(0, n - 2|E_H|) ≥ max(0, n - 2(n-1)/ε).
    # For ε < 0.5: 2(n-1)/ε > 2n, so this gives 0. Useless.
    #
    # But: MOST heavy edges share vertices (the heavy graph is concentrated).
    # The actual n_heavy_vert << 2|E_H| in most cases.
    print(f"\nHeavy vertex concentration: n_heavy_vert / (2*|E_H|):")
    ratios = [r["n_heavy_vertices"] / max(1, 2 * r["n_heavy_edges"]) for r in all_rows]
    print(f"  min  = {min(ratios):.4f}")
    print(f"  mean = {np.mean(ratios):.4f}")
    print(f"  max  = {max(ratios):.4f}")
    print(f"  (close to 0 = heavy edges concentrated on few vertices)")


if __name__ == "__main__":
    main()
