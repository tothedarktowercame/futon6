#!/usr/bin/env python3
"""Verify Option C fix for GPL-H: leverage-aware barrier greedy.

The K_{t,r} counterexample shows GPL-H fails when the greedy selects
hub vertices with high leverage degree. Option C restricts the greedy
to select only from vertices with ℓ_v ≤ C_lev (e.g., 2 + δ).

This script:
1. Runs the leverage-aware greedy on K_{t,r} counterexamples
2. Confirms the fix: min score < 1 at every step
3. Also runs on the original test suite to confirm no regressions
4. Reports the achievable independent set size under the constraint
"""

import numpy as np
import sys
import importlib.util
from pathlib import Path

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


def run_leverage_aware_greedy(n, edges, eps, C_lev=3.0, c_step=1/3, verbose=False):
    """Run barrier greedy restricted to vertices with leverage degree ≤ C_lev/eps.

    Returns dict with results including max_min_score, steps completed,
    and whether any violation occurred.
    """
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    # Compute leverage degrees
    lev_deg = [0.0] * n
    for idx, (u, v) in enumerate(edges):
        lev_deg[u] += taus[idx]
        lev_deg[v] += taus[idx]

    # I0: vertices with all incident edges light (τ ≤ eps)
    # and leverage degree ≤ C_lev / eps
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)

    # Maximal independent set in heavy subgraph
    I_set = set()
    vertices = list(range(n))
    np.random.shuffle(vertices)
    for v in vertices:
        if all(u not in I_set for u in heavy_adj[v]):
            I_set.add(v)

    # Regularize: ℓ_v ≤ C_lev / eps (within I)
    # Compute I-restricted leverage degrees
    ell_I = {}
    for v in I_set:
        ell_v = 0.0
        for idx, (u, w) in enumerate(edges):
            if (u == v and w in I_set) or (w == v and u in I_set):
                ell_v += taus[idx]
        ell_I[v] = ell_v

    lev_bound = C_lev / eps
    I0 = sorted(v for v in I_set if ell_I[v] <= lev_bound)

    if len(I0) < eps * n / 3 * 0.5:
        return {"status": "I0_too_small", "I0_size": len(I0), "I_size": len(I_set)}

    # Check internal edges
    I0_set = set(I0)
    internal_edges = [(u, v) for u, v in edges if u in I0_set and v in I0_set]
    if not internal_edges:
        return {"status": "case1", "I0_size": len(I0)}

    M_I = sum(X_edges[idx] for idx, (u, v) in enumerate(edges)
              if u in I0_set and v in I0_set)
    alpha_I = np.linalg.norm(M_I, ord=2)
    if alpha_I <= eps:
        return {"status": "case2a", "I0_size": len(I0), "alpha_I": alpha_I}

    # Build adjacency in I0
    I0_adj = {v: set() for v in I0}
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        if u in I0_set and v in I0_set:
            I0_adj[u].add(v)
            I0_adj[v].add(u)

    m0 = len(I0)
    T = max(1, min(int(c_step * eps * n), m0 - 1))

    # Eligible set: vertices in I0 with leverage degree ≤ C_lev / eps
    # (already enforced by I0 construction, but we also restrict greedy
    # selection to low-leverage vertices within I0)
    eligible = set(v for v in I0 if ell_I[v] <= lev_bound)

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    max_min_score = 0.0
    step_of_worst = 0
    step_data = []

    for t in range(T):
        # Candidate set: eligible vertices not yet selected
        R_t = [v for v in I0 if v not in S_set]
        R_eligible = [v for v in eligible if v not in S_set]
        r_t = len(R_t)
        r_elig = len(R_eligible)

        if r_t == 0:
            break

        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        # Compute scores for ALL R_t vertices (to check GPL-H)
        all_scores = {}
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            all_scores[v] = float(np.linalg.norm(Y_v, ord=2))

        min_score_all = min(all_scores.values()) if all_scores else 0.0

        # Among eligible vertices, find the minimum score
        eligible_scores = {v: all_scores[v] for v in R_eligible if v in all_scores}
        if eligible_scores:
            min_score_elig = min(eligible_scores.values())
            best_v = min(eligible_scores, key=eligible_scores.get)
        elif R_eligible:
            # All eligible vertices have zero score
            best_v = R_eligible[0]
            min_score_elig = 0.0
        else:
            # No eligible vertices left — fall back to any R_t vertex
            best_v = min(all_scores, key=all_scores.get)
            min_score_elig = all_scores[best_v]

        # Count neighbor structure
        zero_count = sum(1 for v in R_t if all_scores.get(v, 0) < 1e-14)
        is_phase2 = (zero_count == 0 and t > 0)

        step_data.append({
            "t": t,
            "r_t": r_t,
            "r_eligible": r_elig,
            "phase": 2 if is_phase2 else 1,
            "min_score_all": min_score_all,
            "min_score_eligible": min_score_elig,
            "best_v": best_v,
            "best_v_lev": ell_I.get(best_v, -1),
            "mt_norm": float(np.linalg.norm(M_t, ord=2)),
        })

        if min_score_elig > max_min_score:
            max_min_score = min_score_elig
            step_of_worst = t

        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    violated = max_min_score >= 1.0 - 1e-10

    return {
        "status": "case2b",
        "I0_size": len(I0),
        "I_size": len(I_set),
        "n_eligible": len(eligible),
        "T": T,
        "steps_completed": len(step_data),
        "max_min_score": max_min_score,
        "step_of_worst": step_of_worst,
        "violated": violated,
        "step_data": step_data,
        "alpha_I": alpha_I,
    }


def main():
    np.random.seed(42)

    print("=" * 70)
    print("LEVERAGE-AWARE GREEDY: GPL-H FIX VERIFICATION")
    print("=" * 70)

    # Test multiple C_lev values
    C_lev_values = [2.0, 2.5, 3.0, 4.0]

    # ---------------------------------------------------------------
    # Part 1: K_{t,r} counterexamples
    # ---------------------------------------------------------------
    print("\n=== Part 1: K_{t,r} counterexamples ===\n")

    bipartite_cases = [
        (2, 10), (2, 20), (2, 50), (2, 100),
        (3, 20), (3, 30), (3, 50),
        (4, 30), (4, 50), (4, 80),
        (5, 30), (5, 50), (5, 80),
    ]

    for C_lev in C_lev_values:
        print(f"\n--- C_lev = {C_lev} ---")
        violations = 0
        cases = 0
        for t_val, r_val in bipartite_cases:
            if t_val + r_val > 120:
                continue
            n, edges = complete_bipartite(t_val, r_val)
            tau_val = (t_val + r_val - 1) / (t_val * r_val)
            eps = tau_val + 1e-10

            if eps >= 1.0:
                continue

            res = run_leverage_aware_greedy(n, edges, eps, C_lev=C_lev)
            cases += 1

            status = res["status"]
            if status == "case2b":
                mark = "VIOLATION!" if res["violated"] else "OK"
                if res["violated"]:
                    violations += 1
                print(f"  K_{{{t_val},{r_val}}} eps={eps:.4f}: {status} "
                      f"|I0|={res['I0_size']} eligible={res['n_eligible']} "
                      f"max_score={res['max_min_score']:.4f} [{mark}]")
                # Show Phase 2 details if violated
                if res["violated"]:
                    for sd in res["step_data"]:
                        if sd["phase"] == 2:
                            print(f"    step {sd['t']}: phase 2, "
                                  f"min_elig={sd['min_score_eligible']:.4f} "
                                  f"lev_v={sd['best_v_lev']:.2f}")
            else:
                print(f"  K_{{{t_val},{r_val}}} eps={eps:.4f}: {status} "
                      f"|I0|={res.get('I0_size', '?')}")

        print(f"\n  C_lev={C_lev}: {violations}/{cases} violations")

    # ---------------------------------------------------------------
    # Part 2: dbar bound verification
    # ---------------------------------------------------------------
    print("\n\n=== Part 2: dbar bound under leverage constraint ===\n")

    # For the leverage-aware greedy with C_lev, at step t we have:
    # dbar ≤ C_lev * t / (eps * r_t)
    # At t = eps*n/3, r_t ≥ m_0 - t ≈ n(1 - eps/3), so:
    # dbar ≤ C_lev / (3 - eps) < 1 when C_lev < 3 - eps

    print("Predicted dbar bounds (C_lev / (3 - eps)):")
    for C_lev in C_lev_values:
        for eps in [0.1, 0.2, 0.3, 0.5, 0.8]:
            bound = C_lev / (3 - eps)
            print(f"  C_lev={C_lev}, eps={eps}: dbar ≤ {bound:.4f} "
                  f"{'< 1 OK' if bound < 1 else '>= 1 FAILS'}")

    # ---------------------------------------------------------------
    # Part 3: Original test suite regression check
    # ---------------------------------------------------------------
    print("\n\n=== Part 3: Original suite regression check (C_lev=3.0) ===\n")

    C_lev = 3.0
    rng = np.random.default_rng(42)
    test_graphs = []
    for n in range(8, 65, 4):
        test_graphs.append((f"K_{n}", n, base.complete_graph(n)))
        test_graphs.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            test_graphs.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = base.dumbbell_graph(n // 3)
            test_graphs.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = base.disjoint_cliques(n // 3, 3)
            test_graphs.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                test_graphs.append((f"ER_{n}_p{p_er}", n, er_edges))

    eps_list = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
    total_case2b = 0
    total_violations = 0
    worst_score = 0.0
    worst_graph = ""

    for graph_name, n, edges in test_graphs:
        for eps in eps_list:
            res = run_leverage_aware_greedy(n, edges, eps, C_lev=C_lev)
            if res["status"] == "case2b":
                total_case2b += 1
                if res["violated"]:
                    total_violations += 1
                    print(f"  VIOLATION: {graph_name} eps={eps} "
                          f"score={res['max_min_score']:.4f}")
                if res["max_min_score"] > worst_score:
                    worst_score = res["max_min_score"]
                    worst_graph = f"{graph_name} eps={eps}"

    print(f"\nOriginal suite: {total_case2b} Case-2b instances, "
          f"{total_violations} violations")
    print(f"Worst score: {worst_score:.6f} ({worst_graph})")

    # ---------------------------------------------------------------
    # Part 4: Low-leverage vertex availability (Markov bound)
    # ---------------------------------------------------------------
    print("\n\n=== Part 4: Low-leverage vertex counts ===\n")

    C_lev = 3.0
    print(f"C_lev = {C_lev}")
    for t_val, r_val in bipartite_cases:
        if t_val + r_val > 120:
            continue
        n, edges = complete_bipartite(t_val, r_val)
        L = base.graph_laplacian(n, edges)
        Lphalf = base.pseudo_sqrt_inv(L)
        _, taus = base.compute_edge_matrices(n, edges, Lphalf)

        lev_deg = [0.0] * n
        for idx, (u, v) in enumerate(edges):
            lev_deg[u] += taus[idx]
            lev_deg[v] += taus[idx]

        tau_val = taus[0]
        eps = tau_val + 1e-10
        lev_bound = C_lev / eps

        low_lev = sum(1 for v in range(n) if lev_deg[v] <= lev_bound)
        # Markov prediction: at most (n-1)*2 / lev_bound vertices exceed bound
        markov_high = min(n, int(np.ceil(2 * (n - 1) / lev_bound)))
        markov_low = n - markov_high

        print(f"  K_{{{t_val},{r_val}}}: eps={eps:.4f} bound={lev_bound:.2f} "
              f"low_lev={low_lev}/{n} "
              f"(Markov predicts ≥{markov_low}) "
              f"left_lev={lev_deg[0]:.2f} right_lev={lev_deg[t_val]:.2f}")


if __name__ == "__main__":
    main()
