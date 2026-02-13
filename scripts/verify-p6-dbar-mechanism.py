#!/usr/bin/env python3
"""Diagnose WHY d_bar_all < 1 holds despite effective rank condition failing.

d_bar_t = (1/r_t) * tr(B_t * F_t)

where B_t = (eps*I - M_t)^{-1} and F_t = cross-edge matrix (S_t to R_t).

Four candidate mechanisms:
  M1: Spectral misalignment — F_t has little weight in M_t's top eigenspace
  M2: Small trace — tr(F_t) grows slowly
  M3: Small ||M_t|| — greedy keeps barrier gap wide
  M4: Trade-off — as t grows, F_t's trace shrinks

We decompose d_bar into directional contributions along eigenvectors of M_t.
"""

import numpy as np
from collections import defaultdict


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


def diagnose(n, edges, eps, name=""):
    """Run barrier greedy with full diagnostic decomposition."""
    L = graph_laplacian(n, edges)
    Lphalf = pseudo_sqrt_inv(L)
    X_edges, taus = compute_edge_matrices(n, edges, Lphalf)
    I0 = find_independent_set(n, edges, taus, eps)
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

    T = max(1, min(int(eps * m0 / 3), m0 - 1))
    S_t = []; S_set = set(); M_t = np.zeros((d, d))
    results = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        H_t = eps * np.eye(d) - M_t
        eigs_H = np.linalg.eigvalsh(H_t)
        if np.min(eigs_H) < 1e-12:
            break

        B_t = np.linalg.inv(H_t)

        # Build F_t (cross-edge matrix: edges from S_t to R_t)
        F_t = np.zeros((d, d))
        for u in S_t:
            for v in R_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    F_t += X_edges[edge_idx[key]]

        # ===== The actual d_bar =====
        dbar = np.trace(B_t @ F_t) / r_t

        # ===== Decomposition along eigenvectors of M_t =====
        eigvals_M, eigvecs_M = np.linalg.eigh(M_t)

        # For each eigenvector u_i of M_t with eigenvalue lambda_i:
        #   B_t has eigenvalue 1/(eps - lambda_i) in that direction
        #   Contribution to tr(B_t F_t) = (u_i^T F_t u_i) / (eps - lambda_i)

        contributions = []
        for i in range(d):
            lam_i = eigvals_M[i]
            u_i = eigvecs_M[:, i]
            f_weight = u_i @ F_t @ u_i  # how much F_t projects onto this direction
            amplification = 1.0 / (eps - lam_i) if (eps - lam_i) > 1e-14 else 1e14
            contrib = f_weight * amplification
            contributions.append({
                'lam_M': lam_i,
                'f_weight': f_weight,
                'amplification': amplification,
                'contribution': contrib,
            })

        total_contrib = sum(c['contribution'] for c in contributions)
        # Sanity: total_contrib should equal tr(B_t F_t)
        assert abs(total_contrib - np.trace(B_t @ F_t)) < 1e-8, \
            f"Decomposition mismatch: {total_contrib} vs {np.trace(B_t @ F_t)}"

        # Sort by eigenvalue of M_t (descending) to see top eigenspace
        contributions.sort(key=lambda c: c['lam_M'], reverse=True)

        # ===== Key diagnostic quantities =====
        norm_M = np.linalg.norm(M_t, ord=2)
        tr_M = np.trace(M_t)
        tr_F = np.trace(F_t)
        norm_F = np.linalg.norm(F_t, ord=2)

        # Scalar bound (worst case): ||B_t|| * tr(F_t) / r_t
        scalar_bound = tr_F / ((eps - norm_M) * r_t) if (eps - norm_M) > 1e-14 else float('inf')

        # How much of F_t's trace is in top eigenspace of M_t?
        # "Top" = eigenvalues > median
        n_eigs = len(contributions)
        top_k = max(1, n_eigs // 10)  # top 10% of M_t eigenspace
        f_weight_top = sum(c['f_weight'] for c in contributions[:top_k])
        f_weight_total = sum(c['f_weight'] for c in contributions)
        top_fraction = f_weight_top / f_weight_total if f_weight_total > 1e-14 else 0

        # Contribution from top eigenspace vs rest
        contrib_top = sum(c['contribution'] for c in contributions[:top_k])
        contrib_rest = sum(c['contribution'] for c in contributions[top_k:])

        # Effective amplification: tr(B_t F_t) / tr(F_t)
        # = weighted average of 1/(eps - lambda_i), weighted by F_t's projection
        eff_amp = np.trace(B_t @ F_t) / tr_F if tr_F > 1e-14 else 1.0 / eps

        # If M_t and F_t were perfectly aligned, d_bar would be:
        aligned_dbar = norm_F / ((eps - norm_M) * r_t) if (eps - norm_M) > 1e-14 else float('inf')

        # If uniform (all eigenvalues equal): d_bar would be:
        uniform_dbar = tr_F / (eps * r_t) if r_t > 0 else float('inf')

        # Leverage of cross-edges
        cross_lev_per_selected = tr_F / max(1, t)  # avg leverage per selected vertex

        results.append({
            't': t, 'r_t': r_t, 'm0': m0,
            'dbar': dbar,
            'scalar_bound': scalar_bound,
            'uniform_dbar': uniform_dbar,
            'norm_M': norm_M, 'tr_M': tr_M,
            'norm_F': norm_F, 'tr_F': tr_F,
            'eff_amp': eff_amp,  # actual amplification factor
            'uniform_amp': 1.0 / eps,  # amplification if M_t = 0
            'top_fraction': top_fraction,  # fraction of F_t in top eigenspace of M_t
            'contrib_top': contrib_top / r_t,  # d_bar from top eigenspace
            'contrib_rest': contrib_rest / r_t,  # d_bar from rest
            'cross_lev_per_sel': cross_lev_per_selected,
            'barrier_gap': eps - norm_M,
            'gap_frac': (eps - norm_M) / eps,
            'contributions': contributions,  # full decomposition
        })

        # Select vertex with min score (barrier greedy)
        scores = {}
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(d))
        for v in R_t:
            C_v = np.zeros((d, d))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            scores[v] = float(np.linalg.norm(Y_v, ord=2))

        best_v = min(R_t, key=lambda v: scores.get(v, 0))
        S_t.append(best_v); S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    return results


def main():
    rng = np.random.default_rng(42)

    graphs = []
    # Complete graphs — the proved case
    for k in [20, 40, 60, 80]:
        graphs.append((f"K_{k}", k, [(i, j) for i in range(k) for j in range(i+1, k)]))
    # Barbells — structured, heterogeneous
    for k in [20, 30]:
        n = 2 * k
        graphs.append((f"Barbell_{k}", n,
                        [(i, j) for i in range(k) for j in range(i+1, k)] +
                        [(i, j) for i in range(k, n) for j in range(i+1, n)] +
                        [(k-1, k)]))
    # Erdos-Renyi
    for nn, p in [(40, 0.3), (60, 0.5), (80, 0.3)]:
        edges = [(i, j) for i in range(nn) for j in range(i+1, nn) if rng.random() < p]
        graphs.append((f"ER_{nn}_p{p}", nn, edges))

    epsilons = [0.15, 0.2, 0.3, 0.5]

    print("=" * 130)
    print("DIAGNOSTIC: WHY d_bar < 1 HOLDS")
    print("=" * 130)
    print()
    print("Key quantities per step:")
    print("  d_bar         = actual average trace = tr(B_t F_t) / r_t")
    print("  scalar_bound  = worst-case bound = tr(F_t) / ((eps-||M||)*r_t)")
    print("  uniform_dbar  = d_bar if M_t=0 (no amplification) = tr(F_t)/(eps*r_t)")
    print("  eff_amp       = effective amplification = tr(B_t F_t)/tr(F_t)")
    print("  1/eps         = max amplification if M_t=0")
    print("  top_frac      = fraction of tr(F_t) in top 10% eigenspace of M_t")
    print("  gap_frac      = (eps-||M||)/eps (how much barrier is left)")
    print()

    all_results = []

    for name, n, edges in graphs:
        for eps in epsilons:
            results = diagnose(n, edges, eps, name)
            if not results:
                continue

            # Only show non-trivial steps (t > 0)
            nontrivial = [r for r in results if r['t'] > 0]
            if not nontrivial:
                continue

            print(f"--- {name}, eps={eps}, m0={results[0]['m0']}, T={len(results)} ---")
            print(f"{'t':>3} {'r_t':>4} {'d_bar':>7} {'scalar':>7} {'unif':>7} "
                  f"{'eff_amp':>7} {'1/eps':>5} {'||M||':>7} {'gap%':>5} "
                  f"{'tr(F)':>7} {'top%':>5} {'d_top':>7} {'d_rest':>7}")

            for r in results:
                top_pct = f"{100*r['top_fraction']:>5.1f}" if r['tr_F'] > 1e-14 else "  n/a"
                print(f"{r['t']:>3} {r['r_t']:>4} {r['dbar']:>7.4f} "
                      f"{min(r['scalar_bound'], 99.9):>7.3f} "
                      f"{r['uniform_dbar']:>7.4f} "
                      f"{r['eff_amp']:>7.3f} {1/eps:>5.2f} "
                      f"{r['norm_M']:>7.4f} {100*r['gap_frac']:>5.1f} "
                      f"{r['tr_F']:>7.3f} {top_pct} "
                      f"{r['contrib_top']:>7.4f} {r['contrib_rest']:>7.4f}")

            for r in nontrivial:
                r['name'] = name
                r['eps'] = eps
            all_results.extend(nontrivial)
            print()

    # ========== MECHANISM ANALYSIS ==========
    print()
    print("=" * 130)
    print("MECHANISM ANALYSIS")
    print("=" * 130)

    if not all_results:
        print("No nontrivial results to analyze.")
        return

    # M1: Spectral misalignment
    # If F_t is misaligned with M_t's top eigenspace, top_fraction should be small
    top_fracs = [r['top_fraction'] for r in all_results if r['tr_F'] > 1e-14]
    if top_fracs:
        print(f"\n[M1] SPECTRAL MISALIGNMENT (F_t vs top eigenspace of M_t)")
        print(f"  F_t weight in top 10% of M_t eigenspace:")
        print(f"    mean = {np.mean(top_fracs)*100:.1f}%")
        print(f"    max  = {np.max(top_fracs)*100:.1f}%")
        print(f"    min  = {np.min(top_fracs)*100:.1f}%")
        # Compare actual d_bar to what it would be if uniformly distributed
        ratio = [r['dbar'] / r['scalar_bound'] for r in all_results
                 if r['scalar_bound'] > 0 and r['scalar_bound'] < 1e6]
        if ratio:
            print(f"  Actual d_bar / scalar bound (compression ratio):")
            print(f"    mean = {np.mean(ratio):.4f}")
            print(f"    min  = {np.min(ratio):.4f}")
            print(f"    max  = {np.max(ratio):.4f}")
            print(f"  (Low ratio = strong misalignment benefit)")

    # M2: Small tr(F_t)
    print(f"\n[M2] CROSS-EDGE LEVERAGE tr(F_t)")
    tr_fs = [r['tr_F'] for r in all_results]
    print(f"  tr(F_t) range: [{min(tr_fs):.4f}, {max(tr_fs):.4f}]")
    lev_per_sel = [r['cross_lev_per_sel'] for r in all_results]
    print(f"  tr(F_t)/t (leverage per selected vertex):")
    print(f"    mean = {np.mean(lev_per_sel):.4f}")
    print(f"    max  = {np.max(lev_per_sel):.4f}")
    # Compare to uniform_dbar (d_bar if M_t=0)
    unif_dbars = [r['uniform_dbar'] for r in all_results]
    print(f"  Uniform d_bar (M_t=0 case) = tr(F_t)/(eps*r_t):")
    print(f"    mean = {np.mean(unif_dbars):.4f}")
    print(f"    max  = {np.max(unif_dbars):.4f}")
    print(f"  (If max uniform_dbar < 1, then M_t growth doesn't matter!)")

    # M3: Small ||M_t||
    print(f"\n[M3] BARRIER GAP (eps - ||M_t||)")
    gaps = [r['gap_frac'] for r in all_results]
    print(f"  Gap fraction (eps-||M||)/eps:")
    print(f"    mean = {np.mean(gaps)*100:.1f}%")
    print(f"    min  = {np.min(gaps)*100:.1f}%")
    eff_amps = [r['eff_amp'] for r in all_results]
    unif_amps = [1.0 / r['eps'] for r in all_results]  # wrong, let me fix
    print(f"  Effective amplification tr(B_t F_t)/tr(F_t):")
    print(f"    mean = {np.mean(eff_amps):.4f}")
    print(f"    max  = {np.max(eff_amps):.4f}")
    print(f"  Uniform amplification 1/eps (for comparison):")
    print(f"    mean = {np.mean([1/r['eps'] for r in all_results]):.4f}")
    print(f"  Amplification ratio (eff / (1/eps)):")
    amp_ratios = [r['eff_amp'] * r['eps'] for r in all_results]  # = eff_amp / (1/eps)
    print(f"    mean = {np.mean(amp_ratios):.4f}")
    print(f"    max  = {np.max(amp_ratios):.4f}")
    print(f"  (Ratio > 1 means M_t amplifies beyond the M_t=0 baseline)")

    # M4: Trade-off between t and F_t
    print(f"\n[M4] TRADE-OFF: t grows but tr(F_t)/r_t shrinks?")
    # Group by fractional progress t/T
    for frac_lo, frac_hi, label in [(0, 0.33, "early"), (0.33, 0.67, "mid"), (0.67, 1.01, "late")]:
        bucket = [r for r in all_results
                  if frac_lo <= r['t'] / max(1, r['t'] + r['r_t'] - r['m0'] + len([x for x in all_results if x.get('name') == r.get('name') and x.get('eps') == r.get('eps')])) < frac_hi]
        # Simpler: use t / (m0 * eps / 3) as fractional progress
        bucket = [r for r in all_results
                  if frac_lo <= r['t'] / max(1, r['m0'] * r['eps'] / 3) < frac_hi]
        if bucket:
            dbars = [r['dbar'] for r in bucket]
            tr_f_per_r = [r['tr_F'] / r['r_t'] for r in bucket if r['r_t'] > 0]
            print(f"  {label:>5} (t/T in [{frac_lo:.2f},{frac_hi:.2f})): "
                  f"n={len(bucket):>3}, d_bar mean={np.mean(dbars):.4f} max={np.max(dbars):.4f}, "
                  f"tr(F)/r mean={np.mean(tr_f_per_r):.4f} max={np.max(tr_f_per_r):.4f}")

    # ========== VERDICT ==========
    print()
    print("=" * 130)
    print("VERDICT: Which mechanism dominates?")
    print("=" * 130)

    # Check: does uniform_dbar (M_t=0 case) already stay below 1?
    max_unif = max(r['uniform_dbar'] for r in all_results)
    print(f"\n  Max uniform d_bar (M_t=0 case): {max_unif:.4f}")
    if max_unif < 1.0:
        print("  ==> MECHANISM M2 IS SUFFICIENT: tr(F_t)/(eps*r_t) < 1 always.")
        print("      The amplification from M_t is irrelevant because even without")
        print("      it, the cross-edge leverage is small enough.")
    else:
        # Check if amplification matters
        max_actual = max(r['dbar'] for r in all_results)
        print(f"  Max actual d_bar: {max_actual:.4f}")
        # Cases where uniform exceeds 1 but actual doesn't
        amplified_cases = [r for r in all_results if r['uniform_dbar'] > 0.8]
        if amplified_cases:
            print(f"\n  Cases where uniform d_bar > 0.8 ({len(amplified_cases)} steps):")
            for r in sorted(amplified_cases, key=lambda x: -x['uniform_dbar'])[:10]:
                print(f"    {r.get('name','?'):>12} eps={r['eps']:.2f} t={r['t']:>2}: "
                      f"uniform={r['uniform_dbar']:.4f} actual={r['dbar']:.4f} "
                      f"gap%={100*r['gap_frac']:.0f}% top%={100*r['top_fraction']:.0f}%")

    # Final decomposition: for the step with highest d_bar, show eigenvalue breakdown
    worst = max(all_results, key=lambda r: r['dbar'])
    print(f"\n  WORST CASE: {worst.get('name','?')} eps={worst['eps']} t={worst['t']}")
    print(f"    d_bar = {worst['dbar']:.6f}")
    print(f"    scalar_bound = {worst['scalar_bound']:.6f}")
    print(f"    uniform_dbar = {worst['uniform_dbar']:.6f}")
    print(f"    ||M_t|| = {worst['norm_M']:.6f}, gap = {worst['barrier_gap']:.6f}")
    print(f"    tr(F_t) = {worst['tr_F']:.6f}, ||F_t|| = {worst['norm_F']:.6f}")
    print(f"    eff_amp = {worst['eff_amp']:.6f} vs 1/eps = {1/worst['eps']:.6f}")
    print(f"\n    Top eigenvalue contributions to tr(B_t F_t):")
    contribs = worst['contributions']
    cum = 0
    shown = 0
    for i, c in enumerate(contribs):
        if c['f_weight'] < 1e-14 and shown > 5:
            continue
        cum += c['contribution']
        if shown < 20 or c['contribution'] / (worst['dbar'] * worst['r_t']) > 0.01:
            print(f"      eig[{i:>3}]: lam_M={c['lam_M']:>8.5f}, "
                  f"F_weight={c['f_weight']:>8.5f}, "
                  f"1/(eps-lam)={c['amplification']:>8.3f}, "
                  f"contrib={c['contribution']:>8.5f} "
                  f"(cum%={100*cum/(worst['dbar']*worst['r_t']):>5.1f}%)")
            shown += 1


if __name__ == "__main__":
    main()
