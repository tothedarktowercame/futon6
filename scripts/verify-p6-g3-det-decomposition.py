#!/usr/bin/env python3
"""G3 diagnostic: Decompose Σ_v det(I - Y_t(v)) by order.

The determinant expands as:
  det(I - Y) = Σ_k (-1)^k e_k(eigenvalues of Y)

where e_k = k-th elementary symmetric polynomial.

The sum over v gives:
  Σ_v det(I-Y_v) = r_t - L_cross/ε + T_2/ε² - T_3/ε³ + ...

where:
  r_t = |R_t|                (order 0 term)
  L_cross = Σ_{u∈S_t} ℓ_u   (order 1 term)
  T_2 = Σ_v Σ_{j<k} τ_j τ_k sin²θ_{jk}  (order 2 term, always positive)

If the sum is positive, some v has ||Y_t(v)|| < 1 (by average argument).
We check: (a) is the sum always positive? (b) are the even-order corrections
large enough to dominate even if the linear bound barely fails?
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


def analyze_det_sum(inst, c_step):
    """Compute Σ_v det(I - Y_t(v)) and its decomposition at each nontrivial step."""
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
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

        best_v, best_score = None, float("inf")
        det_sum = 0.0
        n_active = 0
        n_positive_det = 0
        n_single_edge = 0
        order0 = 0.0
        order1 = 0.0  # will be negative
        min_det = float("inf")
        max_det = float("-inf")

        for v in R_t:
            C_v = np.zeros((n, n))
            n_edges_to_St = 0
            tau_sum = 0.0
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
                    n_edges_to_St += 1
                    tau_sum += inst.taus[edge_idx[key]]

            Y_v = Bsqrt @ C_v @ Bsqrt.T
            score_v = float(np.linalg.norm(Y_v, ord=2))

            if score_v < best_score:
                best_score = score_v
                best_v = v

            if score_v > 1e-10:
                n_active += 1

                # Compute det(I - Y_v) via eigenvalues
                eigs = np.linalg.eigvalsh(Y_v)
                det_v = float(np.prod(1.0 - eigs))
                det_sum += det_v

                if det_v > 0:
                    n_positive_det += 1
                min_det = min(min_det, det_v)
                max_det = max(max_det, det_v)

                # Order contributions (when M_t = 0)
                order0 += 1.0
                order1 -= tau_sum / eps

                if n_edges_to_St == 1:
                    n_single_edge += 1

        if t > 0 and n_active > 0:
            dbar = -order1 / n_active if n_active > 0 else 0
            # Higher-order = det_sum - order0 - order1
            higher_order = det_sum - order0 - order1

            rows.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "m0": m0, "r_t": len(R_t), "n_active": n_active,
                "det_sum": det_sum,
                "det_sum_per_active": det_sum / n_active if n_active > 0 else 0,
                "order0": order0,
                "order1": order1,
                "linear_approx": order0 + order1,  # = r_t - L_cross/eps
                "higher_order": higher_order,
                "dbar": dbar,
                "n_positive_det": n_positive_det,
                "n_single_edge": n_single_edge,
                "min_det": min_det,
                "max_det": max_det,
                "best_score": best_score,
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
            all_rows.extend(analyze_det_sum(inst, args.c_step))

    if not all_rows:
        print("No rows")
        return

    print(f"Nontrivial step-rows: {len(all_rows)}")

    # Is det_sum always positive?
    det_sums = [r["det_sum"] for r in all_rows]
    print(f"\nΣ_v det(I - Y_t(v)):")
    print(f"  min  = {min(det_sums):.6f}")
    print(f"  mean = {np.mean(det_sums):.3f}")
    print(f"  max  = {max(det_sums):.3f}")
    print(f"  negative: {sum(1 for d in det_sums if d < 0)}/{len(det_sums)}")

    # det_sum per active vertex
    per_active = [r["det_sum_per_active"] for r in all_rows]
    print(f"\ndet_sum / |A_t| (= p_avg(1)):")
    print(f"  min  = {min(per_active):.6f}")
    print(f"  mean = {np.mean(per_active):.6f}")

    # Linear approximation vs actual
    lin = [r["linear_approx"] for r in all_rows]
    print(f"\nLinear approximation (order0 + order1 = r_t - L_cross/ε):")
    print(f"  min  = {min(lin):.4f}")
    print(f"  mean = {np.mean(lin):.4f}")
    print(f"  negative: {sum(1 for l in lin if l < 0)}/{len(lin)}")

    # Higher-order correction
    ho = [r["higher_order"] for r in all_rows]
    print(f"\nHigher-order correction (det_sum - linear):")
    print(f"  min  = {min(ho):.6f}")
    print(f"  mean = {np.mean(ho):.6f}")
    print(f"  always positive: {all(h >= -1e-10 for h in ho)}")

    # dbar distribution
    dbars = [r["dbar"] for r in all_rows]
    print(f"\ndbar:")
    print(f"  min  = {min(dbars):.6f}")
    print(f"  mean = {np.mean(dbars):.6f}")
    print(f"  max  = {max(dbars):.6f}")
    print(f"  ≥ 1: {sum(1 for d in dbars if d >= 1)}/{len(dbars)}")

    # Number of vertices with positive det
    pos_fracs = [r["n_positive_det"] / r["n_active"] for r in all_rows]
    print(f"\nFraction of active vertices with det(I-Y_v) > 0:")
    print(f"  min  = {min(pos_fracs):.4f}")
    print(f"  mean = {np.mean(pos_fracs):.4f}")

    # Single-edge fraction
    single_fracs = [r["n_single_edge"] / r["n_active"] for r in all_rows]
    print(f"\nFraction of active vertices with exactly 1 edge to S_t:")
    print(f"  min  = {min(single_fracs):.4f}")
    print(f"  mean = {np.mean(single_fracs):.4f}")

    # Worst det_sum rows
    worst = sorted(all_rows, key=lambda r: r["det_sum"])[:15]
    print(f"\nSmallest det_sum rows:")
    for r in worst:
        print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
              f"active={r['n_active']:>3} "
              f"Σdet={r['det_sum']:.3f} "
              f"linear={r['linear_approx']:.3f} "
              f"higher={r['higher_order']:.4f} "
              f"dbar={r['dbar']:.4f} "
              f"pos_det={r['n_positive_det']}/{r['n_active']}")

    # Rows where linear_approx is smallest (closest to dbar=1)
    worst_lin = sorted(all_rows, key=lambda r: r["linear_approx"])[:10]
    print(f"\nWorst linear approximation rows (closest to dbar ≥ 1):")
    for r in worst_lin:
        print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
              f"linear={r['linear_approx']:.3f} "
              f"higher={r['higher_order']:.4f} "
              f"det_sum={r['det_sum']:.3f} "
              f"dbar={r['dbar']:.4f}")


if __name__ == "__main__":
    main()
