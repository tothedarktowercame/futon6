#!/usr/bin/env python3
"""Cycle 2: Two attack paths on the last P6 gap.

Attack A: Log-determinant potential Φ(t) = log det(εI - M_t).
  Each greedy step decreases Φ. If we can bound the per-step decrease,
  we get a formal bound on how close M_t can approach εI.

Attack D: Direct d̄ identity without separating M_t and F_t.
  d̄_t = (1/r_t) tr(B_t F_t)
  We test whether d̄_t = 2t/(nε) for K_n (exact identity) and
  investigate whether a structural bound d̄_t ≤ f(t,m0,eps) holds
  universally, bypassing the need for a separate ||M_t|| bound.

Key conjecture from Cycle 1: K_n is extremal for both d̄ and ||M_t||/eps.
If true, the proof only needs the K_n formula plus a majorization argument.
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


def run_cycle2(n, edges, eps, name=""):
    """Run barrier greedy with log-det potential and d̄ structure tracking."""
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

        # ===== Attack A: Log-det potential =====
        phi = float(np.sum(np.log(np.maximum(eigs_H, 1e-300))))

        # ===== Core quantities =====
        B_t = np.linalg.inv(H_t)
        norm_M = np.linalg.norm(M_t, ord=2)
        tr_M = np.trace(M_t)

        # Build F_t and compute d̄
        F_t = np.zeros((d, d))
        for u in S_t:
            for v in R_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    F_t += X_edges[edge_idx[key]]

        tr_F = np.trace(F_t)
        dbar = np.trace(B_t @ F_t) / r_t if r_t > 0 else 0.0

        # K_n exact reference (derived from eigenstructure):
        # M_t in K_n has eigenvalue t/n (mult t-1), 0 (mult n-t+1)
        # F_t in K_n has tr(P_S F_t) = (t-1)(n-t)/n on M_t's nonzero eigenspace
        # tr(B_t F_t) = (t-1)(n-t)/(n(ε-t/n)) + (t+1)(n-t)/(nε)
        # d̄ = (t-1)/(nε-t) + (t+1)/(nε)
        if t > 0 and (m0 * eps - t) > 1e-14:
            kn_exact = (t - 1) / (m0 * eps - t) + (t + 1) / (m0 * eps)
        else:
            kn_exact = 0.0
        # Also keep the uniform (M_t=0) reference
        kn_uniform = 2.0 * t / (m0 * eps) if m0 > 0 else 0.0

        # Compute scores for greedy selection
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(d))
        scores = {}
        traces_v = {}
        for v in R_t:
            C_v = np.zeros((d, d))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            scores[v] = float(np.linalg.norm(Y_v, ord=2))
            traces_v[v] = float(np.trace(Y_v))

        # ===== Attack D: structural d̄ identity =====
        # d̄ = (1/r_t) Σ_v tr(Y_v) = (1/r_t) tr(B_t F_t)
        # For K_n: tr(F_t) = 2t(n-t)/n, and B_t has eigenvalues
        # 1/(ε - t/n) [multiplicity t-1] and 1/ε [multiplicity n-t]
        # So tr(B_t F_t) = ... let's compute the K_n prediction

        # K_n prediction for d̄ when M_t ≠ 0:
        # M_t eigenvalues in K_n: {t/n} with mult (t-1), {0} with mult (n-t+1)
        # (the null space of M_t includes the all-1s direction and directions orthog to S_t)
        # But actually for K_n with S_t ⊂ V, M_t is the normalized Laplacian of K_t
        # restricted to the n-dimensional space. It has eigenvalue t/n with mult (t-1)
        # and eigenvalue 0 with mult (n-t+1).
        # F_t: cross-edges from S_t to R_t. For K_n, F_t has a specific structure.
        # tr(B_t F_t) = Σ_i [u_i^T F_t u_i / (ε - λ_i(M_t))]
        # We just measure it directly and compare to the K_n formula.

        # ===== Leverage structure of the greedy step =====
        best_v = min(R_t, key=lambda v: scores.get(v, 0))
        best_score = scores[best_v]
        best_trace = traces_v[best_v]
        best_ell = ell.get(best_v, 0)

        # Leverage of edges added this step
        step_lev = 0.0
        step_edges = 0
        for u in S_t:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                step_lev += taus[edge_idx[key]]
                step_edges += 1

        results.append({
            't': t, 'r_t': r_t, 'm0': m0,
            'phi': phi,
            'norm_M': norm_M, 'tr_M': tr_M,
            'tr_F': tr_F,
            'dbar': dbar,
            'kn_exact': kn_exact,
            'kn_uniform': kn_uniform,
            'dbar_over_kn_exact': dbar / kn_exact if kn_exact > 1e-14 else float('nan'),
            'dbar_over_kn_uniform': dbar / kn_uniform if kn_uniform > 1e-14 else float('nan'),
            'best_score': best_score,
            'best_trace': best_trace,
            'best_ell': best_ell,
            'step_lev': step_lev,
            'step_edges': step_edges,
            'gap_frac': (eps - norm_M) / eps,
        })

        # Greedy step
        S_t.append(best_v); S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    # Compute per-step Φ changes
    for i in range(1, len(results)):
        results[i]['delta_phi'] = results[i]['phi'] - results[i-1]['phi']
    if results:
        results[0]['delta_phi'] = 0.0

    return results


def main():
    rng = np.random.default_rng(42)

    graphs = []
    # Complete graphs — the extremal case
    for k in [20, 40, 60, 80, 100]:
        graphs.append((f"K_{k}", k, [(i, j) for i in range(k) for j in range(i+1, k)]))
    # Barbells — worst non-Kn case from Cycle 1
    for k in [20, 30, 40]:
        n = 2 * k
        graphs.append((f"Barbell_{k}", n,
                        [(i, j) for i in range(k) for j in range(i+1, k)] +
                        [(i, j) for i in range(k, n) for j in range(i+1, n)] +
                        [(k-1, k)]))
    # Erdos-Renyi
    for nn, p in [(40, 0.3), (60, 0.5), (80, 0.3), (80, 0.5)]:
        edges = [(i, j) for i in range(nn) for j in range(i+1, nn) if rng.random() < p]
        graphs.append((f"ER_{nn}_p{p}", nn, edges))

    epsilons = [0.15, 0.2, 0.3, 0.5]

    # ========== ATTACK A: Log-det potential ==========
    print("=" * 120)
    print("ATTACK A: LOG-DETERMINANT POTENTIAL Φ(t) = log det(εI - M_t)")
    print("=" * 120)
    print()
    print("If ΔΦ = Φ(t+1) - Φ(t) is bounded, we bound how fast the barrier shrinks.")
    print("For K_n: ΔΦ = log(1 - τ_e/(ε-||M_t||)) ≈ -τ_e/(ε-||M_t||)")
    print()

    all_results = []
    kn_ratio_max = 0.0

    for name, n, edges in graphs:
        for eps in epsilons:
            results = run_cycle2(n, edges, eps, name)
            if not results:
                continue

            nontrivial = [r for r in results if r['t'] > 0]
            if not nontrivial:
                continue

            for r in nontrivial:
                r['name'] = name
                r['n'] = n
                r['eps'] = eps
            all_results.extend(nontrivial)

            # Show trajectory for Attack A
            print(f"--- {name}, eps={eps}, m0={results[0]['m0']}, T={len(results)} ---")
            print(f"{'t':>3} {'Φ(t)':>9} {'ΔΦ':>8} {'step_lev':>8} "
                  f"{'ΔΦ/slev':>8} {'||M||':>7} {'gap%':>5} "
                  f"{'d̄':>7} {'d̄/Kn':>6} {'sel_ℓ':>6}")

            for r in results:
                delta = r['delta_phi']
                slev = r['step_lev']
                ratio = delta / (-slev / (eps - r['norm_M'])) if slev > 1e-14 and (eps - r['norm_M']) > 1e-12 else float('nan')
                kn_ex_str = f"{r['dbar_over_kn_exact']:>6.3f}" if not np.isnan(r['dbar_over_kn_exact']) else "   n/a"
                print(f"{r['t']:>3} {r['phi']:>9.4f} {delta:>8.5f} {slev:>8.5f} "
                      f"{ratio:>8.4f} {r['norm_M']:>7.4f} {100*r['gap_frac']:>5.1f} "
                      f"{r['dbar']:>7.4f} {kn_ex_str} {r['best_ell']:>6.3f}")
            print()

    if not all_results:
        print("No nontrivial results.")
        return

    # ========== ATTACK A analysis ==========
    print("=" * 120)
    print("ATTACK A ANALYSIS: Per-step Φ decrease")
    print("=" * 120)

    delta_phis = [r['delta_phi'] for r in all_results if r['delta_phi'] != 0]
    step_levs = [r['step_lev'] for r in all_results if r['step_lev'] > 1e-14]
    normalized = []
    for r in all_results:
        if r['step_lev'] > 1e-14 and (r['eps'] - r['norm_M']) > 1e-12:
            # ΔΦ ≈ -Σ_e log(1 - τ_e/(ε-||M_t||))
            # For small τ_e/(ε-||M_t||): ΔΦ ≈ -step_lev/(ε-||M_t||)
            # Ratio should be ≈ 1 if the approximation holds
            predicted = -r['step_lev'] / (r['eps'] - r['norm_M'])
            if abs(predicted) > 1e-14:
                normalized.append(r['delta_phi'] / predicted)

    if delta_phis:
        print(f"\n  Per-step Φ decrease (ΔΦ, negative = barrier shrinks):")
        print(f"    mean = {np.mean(delta_phis):.6f}")
        print(f"    min  = {np.min(delta_phis):.6f} (most decrease)")
        print(f"    max  = {np.max(delta_phis):.6f} (least decrease)")
    if normalized:
        print(f"\n  Normalized ratio ΔΦ / (-step_lev/(ε-||M||)):")
        print(f"    mean = {np.mean(normalized):.6f}")
        print(f"    min  = {np.min(normalized):.6f}")
        print(f"    max  = {np.max(normalized):.6f}")
        print(f"    (should be ≈ 1 if log(1-x) ≈ -x approximation holds)")

    # Total Φ decrease vs budget
    print(f"\n  Log-det budget analysis:")
    print(f"    Initial Φ(0) = n·log(ε). At horizon T:")
    for name in sorted(set(r['name'] for r in all_results)):
        for eps in sorted(set(r['eps'] for r in all_results)):
            traj = [r for r in all_results if r['name'] == name and r['eps'] == eps]
            if len(traj) < 2:
                continue
            phi_0 = traj[0]['phi'] - traj[0]['delta_phi']  # approximate initial Φ
            phi_T = traj[-1]['phi']
            total_decrease = phi_0 - phi_T
            n = traj[0]['n']
            initial_phi = n * np.log(eps)
            # Fraction of budget used
            frac_used = total_decrease / (-initial_phi) if initial_phi < 0 else 0
            print(f"    {name:<15} eps={eps:.2f}: Φ_0={phi_0:.2f} → Φ_T={phi_T:.2f}, "
                  f"used={total_decrease:.3f} ({100*frac_used:.1f}% of budget)")

    # ========== ATTACK D: K_n extremality ==========
    print()
    print("=" * 120)
    print("ATTACK D: IS K_n EXTREMAL?")
    print("=" * 120)
    print()
    print("  EXACT K_n formula: d̄_Kn(t) = (t-1)/(m₀ε-t) + (t+1)/(m₀ε)")
    print("  [Derived from M_t = (t/n)Π_S eigenstructure + bipartite F_t decomposition]")
    print("  At horizon T=εm₀/3: d̄_Kn → 5/6 as m₀→∞")
    print()

    # Test 1: d̄ ≤ d̄_Kn(t,m₀,ε) (exact formula as universal bound)
    exact_ratios = []
    for r in all_results:
        ref = r['kn_exact']
        if ref > 1e-14:
            ratio = r['dbar'] / ref
            exact_ratios.append(ratio)
            r['exact_ratio'] = ratio
        else:
            r['exact_ratio'] = float('nan')

    # Test 2: d̄ ≤ 2t/(m₀ε) (uniform formula — ignores M_t)
    uniform_ratios = []
    for r in all_results:
        ref = r['kn_uniform']
        if ref > 1e-14:
            ratio = r['dbar'] / ref
            uniform_ratios.append(ratio)
            r['uniform_ratio'] = ratio
        else:
            r['uniform_ratio'] = float('nan')

    # Group by graph family
    families = {}
    for r in all_results:
        fname = r['name'].split('_')[0]
        if fname not in families:
            families[fname] = []
        families[fname].append(r)

    print(f"  Test 1: d̄ ≤ d̄_Kn_exact(t,m₀,ε)?")
    print(f"  {'Family':<12} {'n_steps':>7} {'mean':>8} {'max':>8} {'≤1?':>4}")
    for fname in sorted(families.keys()):
        fam = families[fname]
        ratios = [r['exact_ratio'] for r in fam if not np.isnan(r['exact_ratio'])]
        if ratios:
            is_ok = "YES" if max(ratios) <= 1.0 + 1e-10 else "NO"
            print(f"  {fname:<12} {len(ratios):>7} {np.mean(ratios):>8.4f} {max(ratios):>8.4f} {is_ok:>4}")

    print(f"\n  Test 2: d̄ ≤ 2t/(m₀ε)? (uniform bound, ignoring M_t)")
    print(f"  {'Family':<12} {'n_steps':>7} {'mean':>8} {'max':>8} {'≤1?':>4}")
    for fname in sorted(families.keys()):
        fam = families[fname]
        ratios = [r['uniform_ratio'] for r in fam if not np.isnan(r['uniform_ratio'])]
        if ratios:
            is_ok = "YES" if max(ratios) <= 1.0 + 1e-10 else "NO"
            print(f"  {fname:<12} {len(ratios):>7} {np.mean(ratios):>8.4f} {max(ratios):>8.4f} {is_ok:>4}")

    if exact_ratios:
        max_exact = max(exact_ratios)
        max_uniform = max(uniform_ratios) if uniform_ratios else 0
        print(f"\n  GLOBAL max d̄/d̄_Kn_exact = {max_exact:.6f}")
        print(f"  GLOBAL max d̄/(2t/(m₀ε))  = {max_uniform:.6f}")
        if max_exact <= 1.0 + 1e-10:
            print("\n  ==> K_n EXACT IS EXTREMAL: d̄ ≤ (t-1)/(m₀ε-t) + (t+1)/(m₀ε)")
            print("  At T = εm₀/3, this gives d̄ → 5/6 < 1.  ✓")
            print("  PROOF CLOSES: just prove this formula is a universal upper bound.")
        else:
            worst = max(all_results, key=lambda r: r.get('exact_ratio', 0))
            print(f"\n  K_n exact is NOT extremal. Worst case:")
            print(f"      {worst['name']} eps={worst['eps']} t={worst['t']} "
                  f"ratio={worst['exact_ratio']:.6f} d̄={worst['dbar']:.6f}")

    # ========== COMBINED VERDICT ==========
    print()
    print("=" * 120)
    print("CYCLE 2 VERDICT")
    print("=" * 120)

    kn_exact_extremal = max(exact_ratios) <= 1.0 + 1e-10 if exact_ratios else False

    if kn_exact_extremal:
        print("\n  BREAKTHROUGH: d̄ ≤ (t-1)/(m₀ε-t) + (t+1)/(m₀ε) universally!")
        print("  At T = εm₀/3: d̄ → 5/6 < 1.")
        print("  Proof strategy: prove this formula as universal upper bound, then:")
        print("    Pigeonhole + PSD ⟹ ∃v with ||Y_t(v)|| < 1")
        print("    Barrier maintained ⟹ |S| = T ≥ ε²n/9")
        print("  No separate ||M_t|| bound needed!")
    else:
        max_e = max(exact_ratios) if exact_ratios else 0
        print(f"\n  K_n exact not extremal (max ratio = {max_e:.4f}).")
        if max_e < 1.2:
            print(f"  But overshoot is modest ({max_e:.4f}). A 6/5 safety factor")
            print(f"  would close via: d̄ ≤ (6/5)·d̄_Kn. At horizon: (6/5)·(5/6) = 1.")
            print(f"  Need to either prove the 6/5 bound or tighten the horizon.")
        print("  Fall back options: Attack A (log-det), tighter horizon, or two-phase.")

    # Save results
    summary = {
        'kn_exact_extremal': bool(kn_exact_extremal),
        'max_exact_ratio': float(max(exact_ratios)) if exact_ratios else None,
        'max_uniform_ratio': float(max(uniform_ratios)) if uniform_ratios else None,
        'max_dbar': float(max(r['dbar'] for r in all_results)),
        'n_steps': len(all_results),
        'phi_analysis': {
            'mean_normalized_ratio': float(np.mean(normalized)) if normalized else None,
            'max_normalized_ratio': float(np.max(normalized)) if normalized else None,
        }
    }
    out_path = Path("data/first-proof/problem6-cycle2-results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
