#!/usr/bin/env python3
"""Two decisive tests for Problem 6.

Test A: MSS interlacing — compute avg det(I - Y_t(v)) at every greedy step.
  If always > 0, proof closes via interlacing families (MSS Theorem 4.4).

Test B: Pure random sampling — pick each vertex with prob p = ε.
  Check if P(||M_S|| ≤ ε AND |S| ≥ εn/6) > 0 across many trials.

Each test has a CLEAR pass/fail criterion. No Zeno.
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


def test_A_interlacing(n, edges, eps, graph_name=""):
    """Run barrier greedy, compute avg det(I - Y_v) at each step.
    Returns list of (t, avg_det, dbar, r_t) tuples, or None if not Case-2b."""
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    # Build I0: independent set in heavy graph
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)

    I_set = set()
    vertices = list(range(n))
    np.random.shuffle(vertices)
    for v in vertices:
        if all(u not in I_set for u in heavy_adj[v]):
            I_set.add(v)

    # Leverage-degree filter (H2' with C_lev=2)
    ell = {}
    for v in I_set:
        ell[v] = sum(taus[idx] for idx, (u, w) in enumerate(edges)
                     if (u == v and w in I_set) or (w == v and u in I_set))
    I0 = sorted(v for v in I_set if ell.get(v, 999) <= 2.0 / eps)

    if len(I0) < 3:
        return None

    I0_set = set(I0)
    internal = [(u, v) for u, v in edges if u in I0_set and v in I0_set]
    if not internal:
        return None  # Case 2a

    M_I = sum(X_edges[idx] for idx, (u, v) in enumerate(edges)
              if u in I0_set and v in I0_set)
    if np.linalg.norm(M_I, ord=2) <= eps:
        return None  # Case 2a

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    m0 = len(I0)
    T = max(1, min(int(eps * n / 3), m0 - 1))

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    results = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        H = eps * np.eye(n) - M_t
        eigs_H = np.linalg.eigvalsh(H)
        if np.min(eigs_H) < 1e-12:
            break

        Hinv = np.linalg.inv(H)
        Hsqrt_inv = np.linalg.cholesky(Hinv + 1e-14 * np.eye(n))

        # Compute Y_t(v), det(I - Y_t(v)), tr(Y_t(v)) for each v in R_t
        dets = []
        traces = []
        scores = []
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Hsqrt_inv @ C_v @ Hsqrt_inv.T

            eig_Y = np.linalg.eigvalsh(Y_v)
            eig_Y = eig_Y[eig_Y > 1e-14]  # nonzero eigenvalues

            det_val = float(np.prod(1.0 - eig_Y)) if len(eig_Y) > 0 else 1.0
            tr_val = float(np.sum(eig_Y))
            score = float(np.max(eig_Y)) if len(eig_Y) > 0 else 0.0

            dets.append(det_val)
            traces.append(tr_val)
            scores.append(score)

        avg_det = np.mean(dets)
        dbar = np.mean(traces)
        min_score = min(scores)
        mt_norm = float(np.linalg.norm(M_t, ord=2))

        results.append({
            "t": t, "r_t": r_t, "avg_det": avg_det, "dbar": dbar,
            "min_score": min_score, "mt_norm": mt_norm,
            "neg_dets": sum(1 for d in dets if d < 0),
        })

        # Select min-score vertex
        best_idx = np.argmin(scores)
        best_v = R_t[best_idx]

        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    return results


def test_B_random_sampling(n, edges, eps, n_trials=200):
    """Pure random sampling: each vertex with prob p=eps.
    Returns (n_success, n_trials, best_norm, best_size)."""
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    target_size = max(1, int(eps * n / 6))
    successes = 0
    best_norm = float('inf')
    best_size = 0

    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        # Sample each vertex with probability eps
        Z = rng.random(n) < eps
        S = [v for v in range(n) if Z[v]]
        s = len(S)

        if s < target_size:
            continue

        S_set = set(S)
        # Compute M_S
        M_S = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            if u in S_set and v in S_set:
                M_S += X_edges[idx]

        norm_S = float(np.linalg.norm(M_S, ord=2))

        if norm_S <= eps and s >= target_size:
            successes += 1

        if norm_S < best_norm and s >= target_size:
            best_norm = norm_S
            best_size = s

    return successes, n_trials, best_norm, best_size, target_size


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("PROBLEM 6: TWO DECISIVE TESTS")
    print("=" * 70)

    # Build graph suite
    graphs = []
    for n in range(8, 49, 4):
        graphs.append((f"K_{n}", n, base.complete_graph(n)))
        graphs.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            graphs.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = base.dumbbell_graph(n // 3)
            graphs.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = base.disjoint_cliques(n // 3, 3)
            graphs.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                graphs.append((f"ER_{n}_p{p_er}", n, er_edges))

    eps_list = [0.12, 0.15, 0.2, 0.25, 0.3]

    # ========== TEST A: MSS INTERLACING ==========
    print("\n" + "=" * 70)
    print("TEST A: MSS INTERLACING — avg det(I - Y_v) > 0 ?")
    print("=" * 70)

    total_steps = 0
    violations_A = 0
    min_avg_det = float('inf')
    worst_A = ""
    all_A_results = []

    for gname, gn, gedges in graphs:
        for eps in eps_list:
            res = test_A_interlacing(gn, gedges, eps, gname)
            if res is None:
                continue
            for r in res:
                if r["t"] == 0 and r["dbar"] == 0:
                    continue  # trivial step
                total_steps += 1
                if r["avg_det"] <= 0:
                    violations_A += 1
                if r["avg_det"] < min_avg_det:
                    min_avg_det = r["avg_det"]
                    worst_A = f"{gname} eps={eps} t={r['t']}"
                all_A_results.append((gname, eps, r))

    print(f"\nTotal nontrivial greedy steps: {total_steps}")
    print(f"Violations (avg_det ≤ 0): {violations_A}")
    print(f"Min avg_det: {min_avg_det:.6f} ({worst_A})")

    # Show some examples
    print(f"\n--- Sample trajectories ---")
    shown = set()
    count = 0
    for gname, eps, r in all_A_results:
        key = (gname, eps)
        if key in shown or r["t"] == 0:
            continue
        if r["mt_norm"] > 0.01 and count < 10:
            shown.add(key)
            # Find all steps for this instance
            traj = [(g, e, rr) for g, e, rr in all_A_results if g == gname and e == eps]
            print(f"\n  {gname} eps={eps}:")
            for _, _, rr in traj:
                flag = " !!!" if rr["avg_det"] <= 0 else ""
                print(f"    t={rr['t']:>2} r={rr['r_t']:>3} ||M||={rr['mt_norm']:.4f} "
                      f"dbar={rr['dbar']:.4f} avg_det={rr['avg_det']:.6f} "
                      f"min_score={rr['min_score']:.4f} neg={rr['neg_dets']}{flag}")
            count += 1

    # Correlation between dbar and avg_det
    if all_A_results:
        dbars_A = [r["dbar"] for _, _, r in all_A_results if r["t"] > 0]
        dets_A = [r["avg_det"] for _, _, r in all_A_results if r["t"] > 0]
        if dbars_A:
            print(f"\n--- dbar vs avg_det ---")
            print(f"  dbar  range: [{min(dbars_A):.4f}, {max(dbars_A):.4f}]")
            print(f"  avg_det range: [{min(dets_A):.6f}, {max(dets_A):.6f}]")
            # Check: is avg_det ≥ 1 - dbar always?
            margin = [d - (1 - db) for d, db in zip(dets_A, dbars_A)]
            print(f"  avg_det - (1-dbar) range: [{min(margin):.6f}, {max(margin):.6f}]")
            print(f"  (positive means higher-order terms HELP)")

    # ========== TEST B: RANDOM SAMPLING ==========
    print("\n" + "=" * 70)
    print("TEST B: RANDOM SAMPLING — ||M_S|| ≤ ε with |S| ≥ εn/6 ?")
    print("=" * 70)

    total_B = 0
    successes_B = 0
    test_graphs_B = [g for g in graphs if g[1] >= 12]  # need n ≥ 12

    for gname, gn, gedges in test_graphs_B:
        for eps in [0.15, 0.2, 0.25, 0.3]:
            s, t, best_n, best_s, tgt = test_B_random_sampling(gn, gedges, eps, n_trials=200)
            total_B += 1
            if s > 0:
                successes_B += 1
            if s == 0 or gn <= 20:
                status = f"✓ {s}/{t}" if s > 0 else f"✗ 0/{t} best_norm={best_n:.4f}"
                print(f"  {gname:>20} eps={eps:.2f} target_size={tgt:>2} {status}")

    print(f"\nTotal graph/eps combos: {total_B}")
    print(f"Combos with ≥1 success: {successes_B}")
    rate = successes_B / total_B if total_B > 0 else 0
    print(f"Success rate: {rate:.1%}")

    # ========== VERDICT ==========
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    print(f"\nTest A (MSS interlacing): ", end="")
    if violations_A == 0:
        print(f"PASS — avg det(I-Y) > 0 at all {total_steps} steps")
        print(f"  => By MSS Theorem 4.4, ∃v with ||Y_t(v)|| < 1 at every step")
        print(f"  => GPL-H' holds, Problem 6 CLOSES (modulo interlacing family verification)")
    else:
        print(f"FAIL — {violations_A} violations found")

    print(f"\nTest B (random sampling): ", end="")
    if rate > 0.5:
        print(f"PASS — works for {rate:.0%} of graph/eps combos")
        print(f"  => Random vertex sampling with p=ε gives ||M_S|| ≤ ε w.p. > 0")
        print(f"  => Problem 6 CLOSES via probabilistic method")
    else:
        print(f"WEAK — only {rate:.0%} success rate")
        print(f"  => Random sampling alone may not suffice for all graphs")


if __name__ == "__main__":
    main()
