#!/usr/bin/env python3
"""Test random sampling at larger n with demanding size targets."""

import numpy as np
import sys, importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def test_random(n, edges, eps, c=1/6, n_trials=500):
    """Random sample each vertex with prob p. Return success count."""
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    target = max(2, int(c * eps * n))
    rng = np.random.default_rng(42)
    successes = 0
    best_norm = float('inf')

    for _ in range(n_trials):
        Z = rng.random(n) < eps
        S = np.where(Z)[0]
        s = len(S)
        if s < target:
            continue

        S_set = set(S.tolist())
        M_S = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            if u in S_set and v in S_set:
                M_S += X_edges[idx]

        norm_S = float(np.linalg.norm(M_S, ord=2))
        if norm_S <= eps:
            successes += 1
        if norm_S < best_norm:
            best_norm = norm_S

    return successes, n_trials, target, best_norm


def main():
    np.random.seed(42)
    rng = np.random.default_rng(123)

    print("RANDOM SAMPLING AT LARGER n")
    print("=" * 70)
    print(f"{'graph':>25} {'eps':>5} {'n':>4} {'target':>6} {'succ':>8} {'best_norm':>10}")

    for n in [24, 36, 48, 64, 80]:
        graphs = [
            (f"K_{n}", n, base.complete_graph(n)),
            (f"C_{n}", n, base.cycle_graph(n)),
        ]
        if n >= 12:
            k = n // 2
            bn, be = base.barbell_graph(k)
            graphs.append((f"Barbell_{k}", bn, be))
            cn, ce = base.disjoint_cliques(n // 4, 4)
            graphs.append((f"DisjCliq_{n//4}x4", cn, ce))
        er = base.erdos_renyi(n, 0.4, rng)
        if len(er) > n:
            graphs.append((f"ER_{n}_p0.4", n, er))

        for gname, gn, gedges in graphs:
            for eps in [0.15, 0.2, 0.3]:
                s, t, tgt, bn = test_random(gn, gedges, eps, c=1/6, n_trials=500)
                rate = s/t if t > 0 else 0
                status = f"{s}/{t} ({rate:.0%})" if s > 0 else f"FAIL best={bn:.4f}"
                print(f"  {gname:>23} {eps:>5.2f} {gn:>4} {tgt:>6} {status:>18}")

    # Also test: what's the actual ||M_S||/eps distribution for K_n?
    print("\n--- K_n: distribution of ||M_S||/eps ---")
    for n in [32, 64, 128]:
        for eps in [0.2, 0.3]:
            edges = base.complete_graph(n)
            L = base.graph_laplacian(n, edges)
            Lphalf = base.pseudo_sqrt_inv(L)
            X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)
            rng2 = np.random.default_rng(42)
            norms = []
            sizes = []
            for _ in range(200):
                Z = rng2.random(n) < eps
                S = np.where(Z)[0]
                if len(S) < 2:
                    continue
                S_set = set(S.tolist())
                M_S = np.zeros((n, n))
                for idx, (u, v) in enumerate(edges):
                    if u in S_set and v in S_set:
                        M_S += X_edges[idx]
                norms.append(float(np.linalg.norm(M_S, ord=2)))
                sizes.append(len(S))
            if norms:
                ratios = [nm/eps for nm in norms]
                print(f"  K_{n} eps={eps}: ||M||/eps mean={np.mean(ratios):.3f} "
                      f"max={max(ratios):.3f} |S|/n mean={np.mean(sizes)/n:.3f} "
                      f"P(||M||≤eps)={sum(1 for r in ratios if r<=1)/len(ratios):.2f}")


if __name__ == "__main__":
    main()
