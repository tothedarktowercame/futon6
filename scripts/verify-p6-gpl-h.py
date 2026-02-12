#!/usr/bin/env python3
"""
Verify GPL-H conjecture on small/medium graphs.

Tests the open bridge: H1-H4 => min_v ||Y_t(v)|| <= theta < 1
at every step of the barrier greedy for Case-2b instances.

Reports worst-case (graph, epsilon, step) triples and tracks
eigenvector alignment structure in the dangerous subspace.

Usage:
  python3 scripts/verify-p6-gpl-h.py
  python3 scripts/verify-p6-gpl-h.py --nmax 32 --eps 0.15 0.2 0.25
"""
import argparse
import numpy as np
from itertools import combinations
from dataclasses import dataclass, field


# --- Graph generators ---

def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def cycle_graph(n):
    return [(i, (i + 1) % n) for i in range(n)]

def path_graph(n):
    return [(i, i + 1) for i in range(n - 1)]

def star_graph(n):
    return [(0, i) for i in range(1, n)]

def barbell_graph(k):
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, 2 * k):
        for j in range(i + 1, 2 * k):
            edges.append((i, j))
    edges.append((k - 1, k))
    return 2 * k, edges

def erdos_renyi(n, p, rng):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges

def random_regular(n, d, rng):
    """Approximate d-regular graph via random matching."""
    edges_set = set()
    for _ in range(d * n * 5):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges_set.add((min(u, v), max(u, v)))
    # Keep edges where both endpoints have degree <= d
    adj = [0] * n
    kept = []
    for u, v in sorted(edges_set):
        if adj[u] < d and adj[v] < d:
            kept.append((u, v))
            adj[u] += 1
            adj[v] += 1
    return kept

def dumbbell_graph(k):
    """Two K_k connected by a path of length 2."""
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    bridge_start = k
    edges.append((k - 1, bridge_start))
    edges.append((bridge_start, bridge_start + 1))
    for i in range(bridge_start + 1, bridge_start + 1 + k):
        for j in range(i + 1, bridge_start + 1 + k):
            edges.append((i, j))
    return bridge_start + 1 + k, edges

def disjoint_cliques(k, num_cliques):
    """num_cliques copies of K_k connected by a single edge chain."""
    n = k * num_cliques
    edges = []
    for c in range(num_cliques):
        base = c * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((base + i, base + j))
    # Chain the cliques
    for c in range(num_cliques - 1):
        edges.append((c * k + k - 1, (c + 1) * k))
    return n, edges


# --- Core spectral computations ---

def graph_laplacian(n, edges, weights=None):
    L = np.zeros((n, n))
    if weights is None:
        weights = [1.0] * len(edges)
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L

def pseudo_sqrt_inv(L):
    """Compute L^{+/2}: pseudo-inverse square root of Laplacian."""
    eigvals, eigvecs = np.linalg.eigh(L)
    d = len(eigvals)
    Lphalf = np.zeros((d, d))
    for i in range(d):
        if eigvals[i] > 1e-10:
            Lphalf += (1.0 / np.sqrt(eigvals[i])) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return Lphalf

def compute_edge_matrices(n, edges, Lphalf, weights=None):
    """Compute X_e = w_e * (L^{+/2} b_e)(L^{+/2} b_e)^T and tau_e for each edge."""
    if weights is None:
        weights = [1.0] * len(edges)
    X_edges = []
    taus = []
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        b = np.zeros(n)
        b[u] = 1.0
        b[v] = -1.0
        z = Lphalf @ b
        Xe = w * np.outer(z, z)
        tau = w * np.dot(z, z)  # = w * R_eff(u,v)
        X_edges.append(Xe)
        taus.append(tau)
    return X_edges, taus


@dataclass
class Case2bInstance:
    n: int
    edges: list
    epsilon: float
    I0: list               # regularized core vertices
    X_edges: list          # edge PSD matrices
    taus: list             # leverage scores
    alpha_I: float         # ||sum X_f for f within I0||
    graph_name: str = ""


@dataclass
class TrajectoryResult:
    instance: Case2bInstance
    max_min_score: float    # max over t of (min over v of score_t(v))
    scores_at_worst: list   # all scores at the worst step
    step_of_worst: int
    final_size: int         # |S| achieved
    c_eff: float            # |S| / (epsilon * n)
    # Dangerous subspace diagnostics
    n_dangerous_evals: list  # per step: count of eigenvalues near barrier
    alignment_spread: list   # per step: ||sum_v p_v p_v^T|| / r_t


def find_case2b_instance(n, edges, epsilon, graph_name=""):
    """Check if this graph/epsilon gives a Case-2b instance. Return it or None."""
    L = graph_laplacian(n, edges)
    Lphalf = pseudo_sqrt_inv(L)
    X_edges, taus = compute_edge_matrices(n, edges, Lphalf)

    # Identify heavy subgraph G_H
    heavy_edges_idx = [i for i, t in enumerate(taus) if t > epsilon]
    # Build adjacency for G_H
    adj_heavy = [set() for _ in range(n)]
    for idx in heavy_edges_idx:
        u, v = edges[idx]
        adj_heavy[u].add(v)
        adj_heavy[v].add(u)

    # Find a maximal independent set in G_H (greedy)
    I_set = set()
    vertices = list(range(n))
    np.random.shuffle(vertices)
    for v in vertices:
        if all(u not in I_set for u in adj_heavy[v]):
            I_set.add(v)

    if len(I_set) < epsilon * n / 3 * 0.8:
        return None  # I too small (shouldn't happen by Turan but greedy may be suboptimal)

    I = sorted(I_set)

    # Check Case 2b: compute alpha_I = ||sum X_f for f within I||
    edge_in_I = []
    for idx, (u, v) in enumerate(edges):
        if u in I_set and v in I_set:
            edge_in_I.append(idx)

    if not edge_in_I:
        return None  # Case 1: no internal edges

    M_I = sum(X_edges[idx] for idx in edge_in_I)
    alpha_I = np.linalg.norm(M_I, ord=2)

    if alpha_I <= epsilon:
        return None  # Case 2a

    # Regularize: extract I0 with bounded leverage degree
    # Compute leverage degree for each v in I
    ell = {}
    for v in I:
        ell_v = 0.0
        for idx, (u, w) in enumerate(edges):
            if (u == v and w in I_set) or (w == v and u in I_set):
                ell_v += taus[idx]
        ell[v] = ell_v

    T_I = sum(taus[idx] for idx in edge_in_I)
    D_bound = 4 * T_I / len(I) if len(I) > 0 else 0

    I0 = [v for v in I if ell[v] <= max(D_bound * 2, 12 / epsilon)]  # generous bound
    if len(I0) < len(I) / 3:
        I0 = I  # fallback: use all of I

    return Case2bInstance(
        n=n, edges=edges, epsilon=epsilon, I0=I0,
        X_edges=X_edges, taus=taus, alpha_I=alpha_I,
        graph_name=graph_name,
    )


def run_barrier_greedy(inst: Case2bInstance, c_step=0.5, verbose=False):
    """Run the barrier greedy on a Case-2b instance, tracking GPL-H quantities."""
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
    m0 = len(I0)
    d = n  # dimension of matrices

    # Edge lookup: for each pair (u,v) in I0, find the edge index
    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    T = min(int(c_step * eps * n), m0 - 1)
    if T < 1:
        T = 1

    S_t = []
    S_set = set()
    M_t = np.zeros((d, d))

    max_min_score = 0.0
    scores_at_worst = []
    step_of_worst = 0
    n_dangerous_evals = []
    alignment_spread = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        # Barrier matrix
        headroom = eps * np.eye(d) - M_t
        eigvals_h = np.linalg.eigvalsh(headroom)
        if np.min(eigvals_h) < 1e-12:
            if verbose:
                print(f"  Step {t}: barrier violated, stopping")
            break

        B_t = np.linalg.inv(headroom)

        # Compute C_t(v) and Y_t(v) for each v in R_t
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(d))  # B_t^{1/2}

        scores = []
        best_v = None
        best_score = float('inf')

        for v in R_t:
            # C_t(v) = sum_{u in S_t, u~v} X_{uv}
            C_v = np.zeros((d, d))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]

            Y_v = Bsqrt @ C_v @ Bsqrt.T
            score_v = np.linalg.norm(Y_v, ord=2)
            scores.append(score_v)

            if score_v < best_score:
                best_score = score_v
                best_v = v

        min_score = min(scores) if scores else 0.0

        # Track dangerous eigenvalues
        evals_Mt = np.linalg.eigvalsh(M_t)
        delta_threshold = 1.0 / (r_t * 0.9 + 1) if r_t > 0 else 1.0
        n_danger = sum(1 for ev in evals_Mt if ev > eps - delta_threshold)
        n_dangerous_evals.append(n_danger)

        # Track eigenvector alignment (top eigenvectors of Y_t(v))
        # Simplified: compute ||sum_v p_v p_v^T|| / r_t where p_v = top eigvec of Y_t(v)
        P_sum = np.zeros((d, d))
        n_nonzero = 0
        for i, v in enumerate(R_t):
            C_v = np.zeros((d, d))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            if np.linalg.norm(C_v, ord=2) > 1e-14:
                Y_v = Bsqrt @ C_v @ Bsqrt.T
                _, vecs = np.linalg.eigh(Y_v)
                p_v = vecs[:, -1]  # top eigenvector
                P_sum += np.outer(p_v, p_v)
                n_nonzero += 1

        if n_nonzero > 0:
            alignment = np.linalg.norm(P_sum, ord=2) / n_nonzero
        else:
            alignment = 0.0
        alignment_spread.append(alignment)

        if min_score > max_min_score:
            max_min_score = min_score
            scores_at_worst = sorted(scores)
            step_of_worst = t

        if verbose and t % max(1, T // 10) == 0:
            print(f"  Step {t}/{T}: min_score={min_score:.4f}, r_t={r_t}, "
                  f"n_danger={n_danger}, alignment={alignment:.4f}")

        # Add best vertex
        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)

        # Update M_t
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    final_size = len(S_t)
    c_eff = final_size / (eps * n) if eps * n > 0 else 0

    return TrajectoryResult(
        instance=inst,
        max_min_score=max_min_score,
        scores_at_worst=scores_at_worst,
        step_of_worst=step_of_worst,
        final_size=final_size,
        c_eff=c_eff,
        n_dangerous_evals=n_dangerous_evals,
        alignment_spread=alignment_spread,
    )


def main():
    ap = argparse.ArgumentParser(description="Verify GPL-H on small graphs")
    ap.add_argument("--nmax", type=int, default=24, help="Max graph size")
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.12, 0.15, 0.2, 0.25, 0.3],
                    help="Epsilon values to test")
    ap.add_argument("--c-step", type=float, default=0.5,
                    help="Horizon fraction c_step")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build test suite
    test_graphs = []
    for n in range(8, args.nmax + 1, 4):
        test_graphs.append((f"K_{n}", n, complete_graph(n)))
        test_graphs.append((f"C_{n}", n, cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = barbell_graph(k)
            test_graphs.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = dumbbell_graph(n // 3)
            test_graphs.append((f"Dumbbell_{n//3}", dn, de))
        if n >= 12:
            cn, ce = disjoint_cliques(n // 3, 3)
            test_graphs.append((f"DisjCliq_{n//3}x3", cn, ce))
        # Random graphs
        for p_er in [0.3, 0.5]:
            er_edges = erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                test_graphs.append((f"ER_{n}_p{p_er}", n, er_edges))
        for d_rr in [4, 6]:
            if d_rr < n:
                rr_edges = random_regular(n, d_rr, rng)
                if len(rr_edges) > n:
                    test_graphs.append((f"RandReg_{n}_d{d_rr}", n, rr_edges))

    print("=" * 78)
    print("GPL-H VERIFICATION: barrier greedy trajectory analysis")
    print("=" * 78)

    all_results = []
    n_case2b = 0

    for graph_name, n, edges in test_graphs:
        for eps in args.eps:
            inst = find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            n_case2b += 1

            if args.verbose:
                print(f"\n--- {graph_name}, eps={eps}, |I0|={len(inst.I0)}, "
                      f"alpha_I={inst.alpha_I:.4f} ---")

            result = run_barrier_greedy(inst, c_step=args.c_step,
                                        verbose=args.verbose)
            all_results.append(result)

            if args.verbose:
                print(f"  max_min_score={result.max_min_score:.4f}, "
                      f"|S|={result.final_size}, c_eff={result.c_eff:.4f}")

    # Summary
    print(f"\n{'=' * 78}")
    print(f"SUMMARY: {n_case2b} Case-2b instances found")
    print(f"{'=' * 78}")

    if not all_results:
        print("No Case-2b instances generated. Try smaller epsilon or larger n.")
        return

    # Sort by worst max_min_score
    all_results.sort(key=lambda r: -r.max_min_score)

    print(f"\n{'Graph':<25} {'eps':>5} {'|I0|':>5} {'alpha_I':>8} "
          f"{'max_minscore':>12} {'|S|':>5} {'c_eff':>7} {'worst_step':>10}")
    print("-" * 90)

    for r in all_results[:30]:
        print(f"{r.instance.graph_name:<25} {r.instance.epsilon:>5.2f} "
              f"{len(r.instance.I0):>5} {r.instance.alpha_I:>8.4f} "
              f"{r.max_min_score:>12.6f} {r.final_size:>5} "
              f"{r.c_eff:>7.3f} {r.step_of_worst:>10}")

    # Overall statistics
    max_scores = [r.max_min_score for r in all_results]
    print(f"\nmax_min_score statistics:")
    print(f"  min:    {min(max_scores):.6f}")
    print(f"  median: {np.median(max_scores):.6f}")
    print(f"  90th:   {np.percentile(max_scores, 90):.6f}")
    print(f"  99th:   {np.percentile(max_scores, 99):.6f}")
    print(f"  max:    {max(max_scores):.6f}")

    worst = all_results[0]
    print(f"\nWorst case: {worst.instance.graph_name}, eps={worst.instance.epsilon}")
    print(f"  max_min_score = {worst.max_min_score:.6f} at step {worst.step_of_worst}")
    print(f"  |S| = {worst.final_size}, c_eff = {worst.c_eff:.4f}")

    if worst.alignment_spread:
        print(f"\n  Eigenvector alignment (top eigvec overlap) at worst steps:")
        for t in range(min(5, len(worst.alignment_spread))):
            print(f"    step {t}: alignment={worst.alignment_spread[t]:.4f}, "
                  f"n_danger_evals={worst.n_dangerous_evals[t]}")

    # GPL-H verdict
    theta_bound = max(max_scores)
    if theta_bound < 1.0:
        print(f"\nGPL-H: CONSISTENT with theta = {theta_bound:.4f} < 1")
        print(f"  All {n_case2b} Case-2b instances satisfy min_v score_t(v) < 1 "
              f"at every barrier-valid step.")
    else:
        print(f"\nGPL-H: POTENTIAL VIOLATION — max_min_score = {theta_bound:.4f} >= 1")
        print(f"  Investigate worst case carefully.")

    # c_eff statistics
    c_effs = [r.c_eff for r in all_results]
    print(f"\nc_eff = |S|/(eps*n) statistics:")
    print(f"  min:    {min(c_effs):.4f}")
    print(f"  median: {np.median(c_effs):.4f}")
    print(f"  max:    {max(c_effs):.4f}")


if __name__ == "__main__":
    main()
