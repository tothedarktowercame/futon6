#!/usr/bin/env python3
"""Cycle 3: Three questions for the next handoff.

Q1: Is the leverage filter necessary?
    Run barrier greedy WITHOUT leverage filtering. Does d̄ < 1 still hold?

Q2: What is the sharp horizon?
    For K_n, T_max = nε(3-√5)/2 ≈ 0.382nε. Does this hold for general graphs?
    Can we push the greedy beyond εm₀/3?

Q3: Can we break the ε² bottleneck?
    The current proof gives |S| = ε²n/9 (Turán gives εn/3, greedy takes ε fraction).
    If we run on the FULL vertex set (no Turán, no filter), does the greedy still
    maintain d̄ < 1? This would give |S| = εn·c for universal c.
"""

import numpy as np
import json
from pathlib import Path


def graph_laplacian(n, edges):
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    return L


def pseudo_sqrt_inv(L):
    eigvals, eigvecs = np.linalg.eigh(L)
    Lphalf = np.zeros_like(L)
    for i in range(len(eigvals)):
        if eigvals[i] > 1e-10:
            Lphalf += (1.0 / np.sqrt(eigvals[i])) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return Lphalf


def compute_edge_matrices(n, edges, Lphalf):
    X_edges = []; taus = []
    for u, v in edges:
        b = np.zeros(n); b[u] = 1; b[v] = -1
        z = Lphalf @ b
        X_edges.append(np.outer(z, z))
        taus.append(np.dot(z, z))
    return X_edges, taus


def find_independent_set(n, edges, taus, eps):
    """Standard Turán independent set in heavy subgraph (no leverage filter)."""
    adj_heavy = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            adj_heavy[u].add(v)
            adj_heavy[v].add(u)
    I_set = set()
    for v in sorted(range(n), key=lambda vv: len(adj_heavy[vv])):
        if all(u not in I_set for u in adj_heavy[v]):
            I_set.add(v)
    return sorted(I_set)


def run_greedy_dbar(n, edges, eps, candidate_set, max_steps=None, name=""):
    """Run barrier greedy on candidate_set, tracking d̄ at each step.

    candidate_set: the vertices to operate on (I₀ with or without filter,
                   or even the full vertex set V).
    """
    L = graph_laplacian(n, edges)
    Lphalf = pseudo_sqrt_inv(L)
    X_edges, taus = compute_edge_matrices(n, edges, Lphalf)

    I0 = sorted(candidate_set)
    I0_set = set(I0)
    m0 = len(I0)
    d = n

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        if u in I0_set and v in I0_set:
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx

    ell = {}
    for v in I0:
        ell[v] = sum(taus[edge_idx[(min(v, u), max(v, u))]]
                      for u in I0 if u != v and (min(v, u), max(v, u)) in edge_idx)

    if max_steps is None:
        max_steps = m0 - 1

    S_t = []; S_set = set(); M_t = np.zeros((d, d))
    results = []
    barrier_broken = False

    for t in range(max_steps):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        H_t = eps * np.eye(d) - M_t
        eigs_H = np.linalg.eigvalsh(H_t)
        if np.min(eigs_H) < 1e-12:
            barrier_broken = True
            break

        B_t = np.linalg.inv(H_t)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(d))
        norm_M = np.linalg.norm(M_t, ord=2)

        # Compute d̄ and scores
        traces_v = {}
        scores = {}
        for v in R_t:
            C_v = np.zeros((d, d))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            traces_v[v] = float(np.trace(Y_v))
            scores[v] = float(np.linalg.norm(Y_v, ord=2))

        dbar = np.mean(list(traces_v.values()))
        min_score = min(scores.values())
        max_ell_remaining = max(ell.get(v, 0) for v in R_t)

        # K_n exact reference
        if t > 0 and (m0 * eps - t) > 1e-14:
            kn_exact = (t - 1) / (m0 * eps - t) + (t + 1) / (m0 * eps)
        else:
            kn_exact = 0.0

        # Theoretical max horizon for K_n: T = m0*eps*(3-sqrt(5))/2
        kn_tmax = m0 * eps * (3 - np.sqrt(5)) / 2

        results.append({
            't': t, 'r_t': r_t, 'm0': m0,
            'dbar': dbar, 'min_score': min_score,
            'norm_M': norm_M,
            'gap_frac': (eps - norm_M) / eps,
            'kn_exact': kn_exact,
            'max_ell': max_ell_remaining,
            'kn_tmax': kn_tmax,
            'frac_of_kn_tmax': t / kn_tmax if kn_tmax > 0 else 0,
        })

        # Greedy selection: min score
        best_v = min(R_t, key=lambda v: scores.get(v, 0))
        S_t.append(best_v); S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    return results, barrier_broken


def main():
    rng = np.random.default_rng(42)

    # Graph suite
    graphs = []
    for k in [20, 40, 60, 80]:
        graphs.append((f"K_{k}", k, [(i, j) for i in range(k) for j in range(i+1, k)]))
    for k in [20, 30]:
        n = 2 * k
        graphs.append((f"Barbell_{k}", n,
                        [(i, j) for i in range(k) for j in range(i+1, k)] +
                        [(i, j) for i in range(k, n) for j in range(i+1, n)] +
                        [(k-1, k)]))
    for nn, p in [(40, 0.3), (60, 0.5), (80, 0.3)]:
        edges = [(i, j) for i in range(nn) for j in range(i+1, nn) if rng.random() < p]
        graphs.append((f"ER_{nn}_p{p}", nn, edges))
    # Star graph (extreme leverage structure)
    for k in [20, 40]:
        graphs.append((f"Star_{k}", k, [(0, i) for i in range(1, k)]))
    # Grid
    graphs.append(("Grid_8x8", 64,
                    [(i*8+j, i*8+j+1) for i in range(8) for j in range(7)] +
                    [(i*8+j, (i+1)*8+j) for i in range(7) for j in range(8)]))

    epsilons = [0.15, 0.2, 0.3, 0.5]

    # ========== Q1: Without leverage filter ==========
    print("=" * 120)
    print("Q1: IS THE LEVERAGE FILTER NECESSARY?")
    print("=" * 120)
    print()
    print("Running barrier greedy on Turán I₀ WITHOUT leverage filtering.")
    print("If d̄ < 1 at all steps, the filter is unnecessary and the proof simplifies.")
    print()

    q1_all_ok = True
    q1_max_dbar = 0.0
    q1_max_ell = 0.0
    q1_worst = ""
    q1_total_steps = 0

    for name, n, edges in graphs:
        for eps in epsilons:
            L = graph_laplacian(n, edges)
            Lphalf = pseudo_sqrt_inv(L)
            _, taus = compute_edge_matrices(n, edges, Lphalf)
            I0 = find_independent_set(n, edges, taus, eps)
            m0 = len(I0)
            if m0 < 3:
                continue

            T = max(1, min(int(eps * m0 / 3), m0 - 1))
            results, broken = run_greedy_dbar(n, edges, eps, set(I0), T, name)
            if not results:
                continue

            nontrivial = [r for r in results if r['t'] > 0]
            q1_total_steps += len(nontrivial)

            for r in nontrivial:
                if r['dbar'] >= 1.0 - 1e-10:
                    q1_all_ok = False
                if r['dbar'] > q1_max_dbar:
                    q1_max_dbar = r['dbar']
                    q1_worst = f"{name} eps={eps} t={r['t']}"
                if r['max_ell'] > q1_max_ell:
                    q1_max_ell = r['max_ell']

            # Show trajectory for interesting cases
            last = nontrivial[-1] if nontrivial else None
            if last and last['dbar'] > 0.5:
                print(f"  {name:<15} eps={eps:.2f} m0={m0:>3} T={T:>3}: "
                      f"max_d̄={max(r['dbar'] for r in nontrivial):.4f} "
                      f"max_ℓ={max(r['max_ell'] for r in nontrivial):.2f} "
                      f"max_‖M‖/ε={max(r['norm_M'] for r in nontrivial)/eps:.3f}"
                      f"{'  BROKEN' if broken else ''}")

    print(f"\n  Q1 SUMMARY:")
    print(f"    Total nontrivial steps: {q1_total_steps}")
    print(f"    d̄ < 1 at all steps: {'YES' if q1_all_ok else 'NO'}")
    print(f"    Max d̄: {q1_max_dbar:.6f} ({q1_worst})")
    print(f"    Max leverage degree in I₀: {q1_max_ell:.4f}")
    if q1_all_ok:
        print(f"    ==> LEVERAGE FILTER IS UNNECESSARY for d̄ < 1.")
        print(f"    Proof simplification: skip Section 5b entirely.")
        print(f"    Constant improves: |I₀| = εn/3 (not εn/12).")

    # ========== Q2: Sharp horizon ==========
    print()
    print("=" * 120)
    print("Q2: SHARP HORIZON — How far can the greedy go?")
    print("=" * 120)
    print()
    print("K_n theory: T_max = m₀ε(3-√5)/2 ≈ 0.382·m₀ε (where d̄_Kn = 1).")
    print("Standard horizon: T = m₀ε/3 ≈ 0.333·m₀ε.")
    print("Can we push to T_max for general graphs?")
    print()

    # Run to the maximum possible horizon and find where d̄ first exceeds thresholds
    q2_results = []
    for name, n, edges in graphs:
        for eps in epsilons:
            L = graph_laplacian(n, edges)
            Lphalf = pseudo_sqrt_inv(L)
            _, taus = compute_edge_matrices(n, edges, Lphalf)
            I0 = find_independent_set(n, edges, taus, eps)
            m0 = len(I0)
            if m0 < 5:
                continue

            # Run to 50% of m₀ε (well past the standard and K_n max horizons)
            T_extended = max(2, min(int(0.5 * eps * m0), m0 - 1))
            results, broken = run_greedy_dbar(n, edges, eps, set(I0), T_extended, name)
            if not results:
                continue

            nontrivial = [r for r in results if r['t'] > 0]
            if not nontrivial:
                continue

            # Find where d̄ first exceeds 0.9, 0.95, 1.0
            t_at_09 = None
            t_at_095 = None
            t_at_10 = None
            max_t_ok = 0  # last t where d̄ < 1
            for r in nontrivial:
                if r['dbar'] < 1.0 - 1e-10:
                    max_t_ok = r['t']
                if t_at_09 is None and r['dbar'] >= 0.9:
                    t_at_09 = r['t']
                if t_at_095 is None and r['dbar'] >= 0.95:
                    t_at_095 = r['t']
                if t_at_10 is None and r['dbar'] >= 1.0 - 1e-10:
                    t_at_10 = r['t']

            kn_tmax = m0 * eps * (3 - np.sqrt(5)) / 2
            standard_T = eps * m0 / 3

            max_dbar_at_standard = max((r['dbar'] for r in nontrivial
                                         if r['t'] <= int(standard_T)), default=0)
            max_dbar_at_kn_tmax = max((r['dbar'] for r in nontrivial
                                        if r['t'] <= int(kn_tmax)), default=0)

            q2_results.append({
                'name': name, 'n': n, 'eps': eps, 'm0': m0,
                'standard_T': standard_T,
                'kn_tmax': kn_tmax,
                'max_t_ok': max_t_ok,
                'max_t_ok_frac': max_t_ok / (m0 * eps) if m0 * eps > 0 else 0,
                't_at_09': t_at_09,
                't_at_10': t_at_10,
                'max_dbar_at_standard': max_dbar_at_standard,
                'max_dbar_at_kn_tmax': max_dbar_at_kn_tmax,
                'broken': broken,
            })

    print(f"  {'Graph':<15} {'eps':>4} {'m0':>4} {'T_std':>5} {'T_Kn':>5} "
          f"{'T_ok':>5} {'T_ok/m0ε':>8} {'d̄@std':>6} {'d̄@Kn':>6} {'t@1.0':>5}")
    for r in q2_results:
        t10_str = f"{r['t_at_10']:>5}" if r['t_at_10'] is not None else "  n/a"
        print(f"  {r['name']:<15} {r['eps']:>4.2f} {r['m0']:>4} "
              f"{r['standard_T']:>5.1f} {r['kn_tmax']:>5.1f} "
              f"{r['max_t_ok']:>5} {r['max_t_ok_frac']:>8.3f} "
              f"{r['max_dbar_at_standard']:>6.3f} {r['max_dbar_at_kn_tmax']:>6.3f} "
              f"{t10_str}")

    # Analysis
    ok_fracs = [r['max_t_ok_frac'] for r in q2_results if r['max_t_ok'] > 0]
    if ok_fracs:
        print(f"\n  Max safe horizon T_ok/(m₀ε):")
        print(f"    min  = {min(ok_fracs):.4f}")
        print(f"    mean = {np.mean(ok_fracs):.4f}")
        print(f"    K_n theoretical max = {(3-np.sqrt(5))/2:.4f}")

    # Check: does d̄ < 1 hold up to the K_n max horizon for ALL graphs?
    all_ok_at_kn = all(r['max_dbar_at_kn_tmax'] < 1.0 - 1e-10 for r in q2_results)
    print(f"\n  d̄ < 1 at K_n max horizon for ALL graphs: {'YES' if all_ok_at_kn else 'NO'}")

    # ========== Q3: Full vertex set (breaking ε²) ==========
    print()
    print("=" * 120)
    print("Q3: CAN WE BREAK THE ε² BOTTLENECK?")
    print("=" * 120)
    print()
    print("If we skip Turán and run on ALL vertices, the greedy might still")
    print("maintain d̄ < 1 for T = εn/3 steps, giving |S| = εn/3 (linear in ε).")
    print("CAVEAT: heavy edges (τ_e > ε) can violate the barrier in one step.")
    print()

    q3_results = []
    for name, n, edges in graphs:
        for eps in epsilons:
            # Use ALL vertices as candidates
            T = max(1, min(int(eps * n / 3), n - 1))
            results, broken = run_greedy_dbar(n, edges, eps, set(range(n)), T, name)
            if not results:
                continue

            nontrivial = [r for r in results if r['t'] > 0]
            max_dbar = max((r['dbar'] for r in nontrivial), default=0)
            any_violation = any(r['dbar'] >= 1.0 - 1e-10 for r in nontrivial)

            q3_results.append({
                'name': name, 'n': n, 'eps': eps,
                'T': T, 'n_steps': len(nontrivial),
                'max_dbar': max_dbar,
                'broken': broken,
                'violated': any_violation or broken,
            })

            if broken or any_violation:
                print(f"  FAIL: {name:<15} eps={eps:.2f} n={n:>3} T={T:>3}: "
                      f"{'BARRIER BROKEN' if broken else f'max_d̄={max_dbar:.4f}'}")

    q3_ok = sum(1 for r in q3_results if not r['violated'])
    q3_fail = sum(1 for r in q3_results if r['violated'])
    print(f"\n  Q3 SUMMARY:")
    print(f"    Total configs: {len(q3_results)}")
    print(f"    d̄ < 1 (no violations): {q3_ok}")
    print(f"    Violations: {q3_fail}")
    if q3_fail > 0:
        print(f"    ==> Cannot skip Turán: heavy edges break the barrier.")
        print(f"    The ε² bottleneck is structural for this proof architecture.")
    else:
        print(f"    ==> BREAKTHROUGH: greedy on full V maintains d̄ < 1!")
        print(f"    |S| = εn/3 for universal c = 1/3!")

    # Save summary
    summary = {
        'q1': {
            'filter_needed': not q1_all_ok,
            'max_dbar': float(q1_max_dbar),
            'total_steps': q1_total_steps,
        },
        'q2': {
            'all_ok_at_kn_tmax': bool(all_ok_at_kn),
            'min_safe_frac': float(min(ok_fracs)) if ok_fracs else None,
            'mean_safe_frac': float(np.mean(ok_fracs)) if ok_fracs else None,
            'kn_tmax_frac': float((3 - np.sqrt(5)) / 2),
        },
        'q3': {
            'can_skip_turan': q3_fail == 0,
            'n_ok': q3_ok,
            'n_fail': q3_fail,
        }
    }
    out_path = Path("data/first-proof/problem6-cycle3-results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
