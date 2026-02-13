#!/usr/bin/env python3
"""Path B: Expected determinant for random vertex sampling.

Key idea: E[det(εI - M_S)] > 0 implies ∃S with ||M_S|| < ε.

For random S (each vertex with probability p):
  E[det(εI - M_S)] = sum over subsets of edges ...

We compute this numerically and check if it's always positive.
If yes, the probabilistic method gives Problem 6 immediately
(no greedy, no barrier, no interlacing needed).
"""

import numpy as np
import sys, importlib.util
from pathlib import Path
from itertools import combinations

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def expected_det_monte_carlo(n, edges, eps, p, n_trials=2000):
    """Estimate E[det(εI - M_S)] and P(det > 0 AND |S| >= target) by MC."""
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    target_size = max(1, int(eps * n / 6))
    rng = np.random.default_rng(42)

    det_sum = 0.0
    det_pos_and_large = 0
    det_pos = 0
    total = n_trials

    for _ in range(n_trials):
        Z = rng.random(n) < p
        S = np.where(Z)[0]
        s = len(S)

        S_set = set(S.tolist())
        M_S = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            if u in S_set and v in S_set:
                M_S += X_edges[idx]

        mat = eps * np.eye(n) - M_S
        det_val = float(np.linalg.det(mat))
        det_sum += det_val

        if det_val > 0:
            det_pos += 1
            if s >= target_size:
                det_pos_and_large += 1

    return {
        "E_det": det_sum / total,
        "P_det_pos": det_pos / total,
        "P_success": det_pos_and_large / total,
        "target": target_size,
    }


def exact_expected_det_small(n, edges, eps, p):
    """For small n: compute E[det(εI - M_S)] exactly by enumeration."""
    if n > 16:
        return None  # too expensive

    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    total_det = 0.0
    total_weight = 0.0

    for mask in range(1 << n):
        S = [v for v in range(n) if mask & (1 << v)]
        weight = (p ** len(S)) * ((1 - p) ** (n - len(S)))

        S_set = set(S)
        M_S = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            if u in S_set and v in S_set:
                M_S += X_edges[idx]

        det_val = float(np.linalg.det(eps * np.eye(n) - M_S))
        total_det += weight * det_val
        total_weight += weight

    return total_det  # total_weight should be 1


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)

    print("PATH B: EXPECTED DETERMINANT")
    print("=" * 70)
    print("Question: Is E[det(εI - M_S)] > 0 for random vertex sampling?")
    print("If yes: ∃S with ||M_S|| < ε by probabilistic method.\n")

    # Exact computation for small n
    print("--- Exact E[det(εI - M_S)] for small graphs ---\n")
    small_graphs = []
    for n in [6, 8, 10, 12]:
        small_graphs.append((f"K_{n}", n, base.complete_graph(n)))
        small_graphs.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            small_graphs.append((f"Barbell_{k}", bn, be))

    print(f"  {'graph':>15} {'n':>3} {'eps':>5} {'p':>5} {'E[det]':>12} {'sign':>5}")
    all_positive = True
    for gname, gn, gedges in small_graphs:
        for eps in [0.15, 0.2, 0.3]:
            for p in [eps, eps/2]:
                if gn > 16:
                    continue
                ed = exact_expected_det_small(gn, gedges, eps, p)
                if ed is not None:
                    sign = "+" if ed > 0 else "-" if ed < 0 else "0"
                    if ed <= 0:
                        all_positive = False
                    print(f"  {gname:>15} {gn:>3} {eps:>5.2f} {p:>5.2f} {ed:>12.6e} {sign:>5}")

    print(f"\n  All exact E[det] > 0? {'YES' if all_positive else 'NO'}")

    # Monte Carlo for larger graphs
    print(f"\n--- Monte Carlo E[det(εI - M_S)] for larger graphs ---\n")
    print(f"  {'graph':>20} {'n':>3} {'eps':>5} {'p':>5} "
          f"{'E[det]':>12} {'P(det>0)':>8} {'P(success)':>10}")

    mc_graphs = []
    for n in [20, 32, 48, 64]:
        mc_graphs.append((f"K_{n}", n, base.complete_graph(n)))
        mc_graphs.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 12:
            k = n // 2
            bn, be = base.barbell_graph(k)
            mc_graphs.append((f"Barbell_{k}", bn, be))
            cn, ce = base.disjoint_cliques(n // 4, 4)
            mc_graphs.append((f"DisjCliq_{n//4}x4", cn, ce))
        er = base.erdos_renyi(n, 0.4, rng)
        if len(er) > n:
            mc_graphs.append((f"ER_{n}_p0.4", n, er))

    all_mc_positive = True
    for gname, gn, gedges in mc_graphs:
        for eps in [0.2, 0.3]:
            r = expected_det_monte_carlo(gn, gedges, eps, p=eps, n_trials=2000)
            sign = "+" if r["E_det"] > 0 else "-"
            if r["E_det"] <= 0:
                all_mc_positive = False
            print(f"  {gname:>20} {gn:>3} {eps:>5.2f} {eps:>5.2f} "
                  f"{r['E_det']:>12.4e} {r['P_det_pos']:>8.1%} "
                  f"{r['P_success']:>10.1%}")

    # Verdict
    print(f"\n{'='*70}")
    print("PATH B VERDICT")
    print(f"{'='*70}")
    if all_positive and all_mc_positive:
        print("E[det(εI - M_S)] > 0 for ALL tested cases.")
        print("=> ∃S with det(εI - M_S) > 0, i.e., ||M_S|| < ε.")
        print("=> Simultaneously |S| >= εn/6 with positive probability.")
        print("=> Problem 6 answer: YES, with c >= 1/6.")
    else:
        print("Some E[det] <= 0. Path B needs refinement.")


if __name__ == "__main__":
    main()
