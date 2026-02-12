#!/usr/bin/env python3
"""Stress-test GPL-H on complete bipartite K_{t,r} graphs.

K_{t,r} is a potential counterexample to the single-neighbor property at Phase 2 entry:
- Phase 1 selects all t left-part vertices (they form an independent set)
- Phase 2 entry: all r right-part vertices have t neighbors in S_t
- If t >= 2, no single-neighbor vertex exists at Phase 2 entry!

Does GPL-H still hold? The score is ||Σ_{j=1}^t X_{a_j,x}/ε|| for each right vertex x.
By symmetry, all right vertices have the same score.
"""

import numpy as np

def complete_bipartite(t, r):
    """Generate K_{t,r} edges. Left part: 0..t-1. Right part: t..t+r-1."""
    n = t + r
    edges = []
    for i in range(t):
        for j in range(t, n):
            edges.append((i, j))
    return n, edges

def graph_laplacian(n, edges):
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, u] += 1
        L[v, v] += 1
        L[u, v] -= 1
        L[v, u] -= 1
    return L

def pseudo_sqrt_inv(L):
    eigvals, eigvecs = np.linalg.eigh(L)
    d = len(eigvals)
    Lphalf = np.zeros((d, d))
    for i in range(d):
        if eigvals[i] > 1e-10:
            Lphalf += (1.0 / np.sqrt(eigvals[i])) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return Lphalf

def compute_edge_matrices(n, edges, Lphalf):
    X_edges = []
    taus = []
    for u, v in edges:
        b = np.zeros(n)
        b[u] = 1.0
        b[v] = -1.0
        z = Lphalf @ b
        Xe = np.outer(z, z)
        tau = np.dot(z, z)
        X_edges.append(Xe)
        taus.append(tau)
    return X_edges, taus


def test_bipartite(t, r, verbose=True):
    n, edges = complete_bipartite(t, r)
    L = graph_laplacian(n, edges)
    Lphalf = pseudo_sqrt_inv(L)
    X_edges, taus = compute_edge_matrices(n, edges, Lphalf)

    # All edges have the same leverage (by symmetry)
    tau_val = taus[0]
    max_tau = max(taus)
    min_tau = min(taus)

    # H1 threshold
    eps_min = max_tau + 1e-10  # minimum eps for H1

    if verbose:
        print(f"\nK_{{{t},{r}}} (n={n})")
        print(f"  tau range: [{min_tau:.6f}, {max_tau:.6f}]")
        print(f"  eps_min for H1: {eps_min:.6f}")

    # Build edge lookup
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    # Left part: 0..t-1. Right part: t..n-1.
    I0 = list(range(n))  # all vertices (all edges light for eps >= max_tau)
    I0_set = set(I0)

    # I0-subgraph adjacency
    I0_adj = {v: set() for v in I0}
    for u, v in edges:
        I0_adj[u].add(v)
        I0_adj[v].add(u)

    results = []
    for eps in [eps_min, eps_min * 1.01, eps_min * 1.1, eps_min * 1.5, eps_min * 2.0, 0.5, 0.9]:
        if eps <= max_tau:
            continue
        if eps >= 1.0:
            continue

        # Check Case 2b: alpha_I > eps
        M_I = sum(X_edges[i] for i in range(len(edges)))
        alpha_I = np.linalg.norm(M_I, ord=2)
        if alpha_I <= eps:
            continue

        T = max(1, min(int(eps * n / 3), n - 1))

        # Run barrier greedy
        S_t = []
        S_set = set()
        M_t = np.zeros((n, n))
        step_results = []

        for step in range(T):
            R_t = [v for v in I0 if v not in S_set]
            if not R_t:
                break
            headroom = eps * np.eye(n) - M_t
            if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
                break

            B_t = np.linalg.inv(headroom)
            Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

            zero_count = 0
            single_count = 0
            best_v, best_score = None, float("inf")

            for v in R_t:
                st_nbrs = sum(1 for u in S_t if u in I0_adj[v])
                if st_nbrs == 0:
                    zero_count += 1
                elif st_nbrs == 1:
                    single_count += 1

                C_v = np.zeros((n, n))
                for u in S_t:
                    key = (min(u, v), max(u, v))
                    if key in edge_idx:
                        C_v += X_edges[edge_idx[key]]
                Y_v = Bsqrt @ C_v @ Bsqrt.T
                sc = float(np.linalg.norm(Y_v, ord=2))
                if sc < best_score:
                    best_score = sc
                    best_v = v

            is_phase2 = (zero_count == 0 and step > 0)
            mt_norm = float(np.linalg.norm(M_t, ord=2))

            step_results.append({
                "step": step, "phase": 2 if is_phase2 else 1,
                "zero": zero_count, "single": single_count,
                "multi": len(R_t) - zero_count - single_count,
                "best_score": best_score, "mt_norm": mt_norm,
                "r_t": len(R_t),
            })

            if best_v is None:
                break
            S_t.append(best_v)
            S_set.add(best_v)
            for u in S_t[:-1]:
                key = (min(best_v, u), max(best_v, u))
                if key in edge_idx:
                    M_t += X_edges[edge_idx[key]]

        # Report Phase 2 entry
        p2_steps = [s for s in step_results if s["phase"] == 2]
        if p2_steps and verbose:
            entry = p2_steps[0]
            print(f"  eps={eps:.4f}: Phase 2 at step {entry['step']}, "
                  f"zero={entry['zero']} single={entry['single']} multi={entry['multi']} "
                  f"best_score={entry['best_score']:.6f} ||M||={entry['mt_norm']:.6f}")
            if entry['single'] == 0:
                print(f"    *** NO SINGLE-NEIGHBOR VERTEX at Phase 2 entry! ***")
                print(f"    score={entry['best_score']:.6f} {'< 1 OK' if entry['best_score'] < 1 else '>= 1 VIOLATION!'}")
        elif verbose:
            print(f"  eps={eps:.4f}: Phase 1 only (T={T})")

        results.append({
            "t": t, "r": r, "eps": eps,
            "p2_steps": p2_steps,
            "max_score": max((s["best_score"] for s in step_results), default=0),
        })

    return results


def main():
    print("=== Complete Bipartite K_{t,r} Stress Test ===")

    # Key test: K_{2,r} and K_{3,r} where Phase 2 has no single-neighbor
    all_violations = 0

    for t_val in [2, 3, 4, 5]:
        for r_val in [10, 20, 30, 50, 80, 100]:
            if t_val + r_val > 120:
                continue
            results = test_bipartite(t_val, r_val, verbose=True)
            for res in results:
                if res["max_score"] >= 1.0 - 1e-10:
                    all_violations += 1
                    print(f"  !!! POTENTIAL VIOLATION: K_{{{t_val},{r_val}}} eps={res['eps']:.4f} "
                          f"max_score={res['max_score']:.6f}")

    # Also test K_{t,t} (balanced bipartite)
    print("\n=== Balanced bipartite K_{t,t} ===")
    for t_val in [5, 10, 15, 20, 30, 40, 50]:
        results = test_bipartite(t_val, t_val, verbose=True)

    print(f"\nTotal violations: {all_violations}")


if __name__ == "__main__":
    main()
