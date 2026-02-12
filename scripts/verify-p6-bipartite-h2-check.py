#!/usr/bin/env python3
"""Check whether complete bipartite K_{t,r} passes H1-H4 after regularization.

Key question: does find_case2b_instance return a valid instance for K_{t,r}?
If not, K_{t,r} is not a counterexample to GPL-H.

The issue: left vertices in K_{t,r} have leverage degree ℓ = r·τ = (t+r-1)/t,
which can be huge. H2 regularization removes them. After removing left vertices,
I_0 = right vertices only, which form an independent set with NO edges → Case 1.
"""

import sys
sys.path.insert(0, ".")
import importlib.util
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def complete_bipartite(t, r):
    n = t + r
    edges = []
    for i in range(t):
        for j in range(t, n):
            edges.append((i, j))
    return n, edges


def main():
    np.random.seed(42)

    print("=== K_{t,r} H2 Check ===\n")

    for t_val in [2, 3, 4, 5, 10]:
        for r_val in [10, 20, 30, 50, 100]:
            if t_val + r_val > 120:
                continue
            n, edges = complete_bipartite(t_val, r_val)

            # Compute leverages
            L = base.graph_laplacian(n, edges)
            Lphalf = base.pseudo_sqrt_inv(L)
            _, taus = base.compute_edge_matrices(n, edges, Lphalf)

            tau_val = taus[0]
            eps_min = tau_val + 1e-10

            # Leverage degrees
            lev_deg = [0.0] * n
            for idx, (u, v) in enumerate(edges):
                lev_deg[u] += taus[idx]
                lev_deg[v] += taus[idx]

            left_lev = max(lev_deg[i] for i in range(t_val))
            right_lev = max(lev_deg[i] for i in range(t_val, n))

            print(f"K_{{{t_val},{r_val}}} (n={n})")
            print(f"  tau = {tau_val:.6f}")
            print(f"  left lev_deg = {left_lev:.4f}, right lev_deg = {right_lev:.4f}")

            # Try to create Case 2b instance
            for eps_mult in [1.0, 1.1, 1.5, 2.0, 3.0]:
                eps = eps_min * eps_mult
                if eps >= 1.0:
                    continue

                inst = base.find_case2b_instance(n, edges, eps, graph_name=f"K_{{{t_val},{r_val}}}")

                if inst is None:
                    print(f"  eps={eps:.4f}: NOT Case 2b (inst=None)")
                else:
                    I0 = inst.I0
                    I0_set = set(I0)
                    # Check which vertices are in I0
                    left_in_I0 = sum(1 for v in range(t_val) if v in I0_set)
                    right_in_I0 = sum(1 for v in range(t_val, n) if v in I0_set)
                    # Internal edges in I0
                    int_edges = sum(1 for u, v in edges if u in I0_set and v in I0_set)

                    print(f"  eps={eps:.4f}: Case 2b! |I0|={len(I0)} "
                          f"(left={left_in_I0}, right={right_in_I0}), "
                          f"int_edges={int_edges}, alpha_I={inst.alpha_I:.4f}")

                    if int_edges == 0:
                        print(f"    → I0 has NO internal edges. Should be Case 1, not 2b!")

                    # H2 check
                    D0_bound = max(8 * sum(taus[idx] for idx, (u,v) in enumerate(edges)
                                           if u in I0_set and v in I0_set) / len(I0)
                                   if len(I0) > 0 else 0,
                                   12 / eps)
                    lev_ok = all(lev_deg[v] <= D0_bound for v in I0)
                    max_I0_lev = max(lev_deg[v] for v in I0) if I0 else 0
                    print(f"    H2: max lev in I0 = {max_I0_lev:.4f}, "
                          f"D0_bound = {D0_bound:.4f}, passes = {lev_ok}")

            print()


if __name__ == "__main__":
    main()
