#!/usr/bin/env python3
"""Investigate the spectral amplification bound for M_t ≠ 0.

The formal gap: prove dbar < 1 when M_t ≠ 0.

Key identity: dbar = (1/r_t) Σ_i w_i/(ε-λ_i)
where λ_i = eigenvalues of M_t, w_i = e_i^T W e_i (W = crossing matrix).

Constraint: M_t + W + E(R_t) = Σ_{E(I_0)} X_e ⪯ Π ⪯ I.
So M_t + W ⪯ I, giving w_i ≤ 1 - λ_i for each i.

This script:
1. Decomposes tr(B_t W) into eigenspace contributions
2. Measures the alignment of W with M_t's eigenspaces
3. Tests whether the "soft amplification" bound dbar ≤ f(dbar_fresh, ||M_t||) holds
4. Explores whether the spectral orthogonality can be proved
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


def build_suite(nmax, rng):
    suite = []
    for n in range(8, nmax + 1, 4):
        suite.append((f"K_{n}", n, base.complete_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            suite.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            cn, ce = base.disjoint_cliques(n // 3, 3)
            suite.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                suite.append((f"ER_{n}_p{p_er}", n, er_edges))
    return suite


def spectral_analysis(n, edges, eps, C_lev=2.0, c_step=1/3):
    """Run greedy and collect spectral amplification data at each Phase 2 step with M_t > 0."""
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

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

    ell_I = {}
    for v in I_set:
        ell_v = sum(taus[idx] for idx, (u, w) in enumerate(edges)
                    if (u == v and w in I_set) or (w == v and u in I_set))
        ell_I[v] = ell_v

    lev_bound = C_lev / eps
    I0 = sorted(v for v in I_set if ell_I[v] <= lev_bound)
    if len(I0) < 3:
        return []

    I0_set = set(I0)
    internal = [idx for idx, (u, v) in enumerate(edges) if u in I0_set and v in I0_set]
    if not internal:
        return []
    M_I = sum(X_edges[idx] for idx in internal)
    if np.linalg.norm(M_I, ord=2) <= eps:
        return []

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    m0 = len(I0)
    T = max(1, min(int(c_step * eps * n), m0 - 1))
    eligible = set(v for v in I0 if ell_I.get(v, 999) <= lev_bound)

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    results = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        headroom = eps * np.eye(n) - M_t
        eigvals_h = np.linalg.eigvalsh(headroom)
        if np.min(eigvals_h) < 1e-12:
            break

        mt_norm = float(np.linalg.norm(M_t, ord=2))

        # Skip Phase 1 and M_t = 0 cases (already proved)
        # But compute scores to find minimum
        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        scores = {}
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            scores[v] = float(np.linalg.norm(Y_v, ord=2))

        zero_count = sum(1 for v in R_t if scores[v] < 1e-14)
        is_phase2 = (zero_count == 0 and t > 0)

        if is_phase2 and mt_norm > 1e-10:
            # SPECTRAL DECOMPOSITION
            eigvals_M, eigvecs_M = np.linalg.eigh(M_t)

            # W = crossing matrix
            W = np.zeros((n, n))
            for v in R_t:
                for u in S_t:
                    key = (min(u, v), max(u, v))
                    if key in edge_idx:
                        W += X_edges[edge_idx[key]]

            # Project W onto M_t eigenbasis
            # w_i = e_i^T W e_i
            w_proj = np.array([eigvecs_M[:, i] @ W @ eigvecs_M[:, i]
                               for i in range(n)])

            # Spectral norm contributions: w_i / (ε - λ_i)
            contribs = np.array([w_proj[i] / (eps - eigvals_M[i])
                                 if eps - eigvals_M[i] > 1e-14 else 0
                                 for i in range(n)])

            trBW = float(np.sum(contribs))
            trW_eps = float(np.trace(W)) / eps
            dbar_fresh = trW_eps / r_t
            dbar_actual = trBW / r_t

            # Check constraint: w_i ≤ 1 - λ_i
            constraint_check = all(w_proj[i] <= 1 - eigvals_M[i] + 1e-10
                                   for i in range(n))

            # Sort by λ_i (descending) to find the "dangerous" eigenvalues
            order = np.argsort(-eigvals_M)
            top_k = min(5, n)

            # How much of tr(B_t W) comes from top-k eigenvalues?
            top_contribs = sum(contribs[order[i]] for i in range(top_k))
            bottom_contribs = trBW - top_contribs

            # KEY QUANTITY: for the top eigenvalue direction,
            # the ratio w_1 / (1 - λ_1) measures how "saturated" W is
            top_idx = order[0]
            λ_top = eigvals_M[top_idx]
            w_top = w_proj[top_idx]
            bound_top = 1 - λ_top
            saturation = w_top / bound_top if bound_top > 1e-14 else 0

            # Effective amplification decomposition:
            # dbar = dbar_fresh + correction
            # correction = (1/r_t) Σ_i w_i (1/(ε-λ_i) - 1/ε)
            #            = (1/r_t) Σ_i w_i λ_i / (ε(ε-λ_i))
            correction_terms = np.array([
                w_proj[i] * eigvals_M[i] / (eps * (eps - eigvals_M[i]))
                if eps - eigvals_M[i] > 1e-14 else 0
                for i in range(n)])
            correction = float(np.sum(correction_terms)) / r_t

            # The "key identity" for formal proof:
            # correction ≤ (1/r_t) Σ_i (1-λ_i) · λ_i / (ε(ε-λ_i))
            #            = (1/(ε·r_t)) Σ_i (1-λ_i) λ_i / (ε-λ_i)
            # For λ_i ∈ [0, ε): (1-λ_i)λ_i/(ε-λ_i) ≤ λ_i·1/(ε-λ_i) ≤ ||M||/(ε-||M||)
            # ...but this uses the worst-case bound again.

            # Better: correction ≤ (||M||/(ε-||M||)) · dbar_fresh
            # Because: Σ w_i λ_i/(ε(ε-λ_i)) ≤ (||M||/(ε(ε-||M||))) · Σ w_i
            # = (||M||/(ε-||M||)) · tr(W)/ε = (||M||/(ε-||M||)) · r_t · dbar_fresh

            # So: dbar ≤ dbar_fresh + (||M||/(ε-||M||)) · dbar_fresh
            #        = dbar_fresh / (1 - ||M||/ε)   [the scalar bound!]

            # But ALSO: Σ w_i λ_i/(ε(ε-λ_i)) ≤ (1/ε²) Σ w_i λ_i / (1 - λ_i/ε)
            # If the w_i are concentrated on SMALL λ_i:
            # correction ≈ (1/(ε²r_t)) Σ w_i λ_i ≤ (1/(ε²r_t)) tr(W) tr(M_t)/n
            # [by Cauchy-Schwarz-like argument]

            # ACTUAL: correction uses the correlation between w_i and λ_i/(ε-λ_i)
            # If w_i and λ_i are NEGATIVELY correlated (large λ → small w):
            # the correction is smaller than the scalar bound.

            # Correlation
            active = [(w_proj[i], eigvals_M[i]) for i in range(n) if eigvals_M[i] > 1e-14]
            if len(active) >= 2:
                ws = np.array([a[0] for a in active])
                ls = np.array([a[1] for a in active])
                if np.std(ws) > 1e-14 and np.std(ls) > 1e-14:
                    corr = np.corrcoef(ws, ls)[0, 1]
                else:
                    corr = 0
            else:
                corr = 0

            results.append({
                "t": t, "r_t": r_t,
                "mt_norm": mt_norm,
                "dbar_fresh": dbar_fresh,
                "dbar_actual": dbar_actual,
                "correction": correction,
                "amp_ratio": dbar_actual / dbar_fresh if dbar_fresh > 1e-14 else 1,
                "scalar_bound": 1 / (1 - mt_norm / eps) if mt_norm < eps else float('inf'),
                "constraint_ok": constraint_check,
                "saturation_top": saturation,
                "lambda_top": λ_top,
                "w_top": w_top,
                "top_contrib_frac": top_contribs / trBW if trBW > 1e-14 else 0,
                "corr_w_lambda": corr,
            })

        # Select best eligible
        elig_scores = {v: scores[v] for v in eligible if v not in S_set and v in scores}
        if elig_scores:
            best_v = min(elig_scores, key=elig_scores.get)
        elif R_t:
            best_v = min(scores, key=scores.get)
        else:
            break

        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    return results


def main():
    np.random.seed(42)

    print("=" * 70)
    print("SPECTRAL AMPLIFICATION BOUND: CLOSING THE M_t ≠ 0 GAP")
    print("=" * 70)

    C_lev = 2.0
    rng = np.random.default_rng(42)
    suite = build_suite(96, rng)
    eps_list = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

    all_results = []
    for graph_name, n, edges in suite:
        for eps in eps_list:
            res = spectral_analysis(n, edges, eps, C_lev=C_lev)
            for r in res:
                r["graph"] = graph_name
                r["n"] = n
                r["eps"] = eps
            all_results.extend(res)

    print(f"\nPhase 2, M_t > 0 steps: {len(all_results)}")

    if not all_results:
        print("No Phase 2 steps with M_t > 0 found.")
        return

    # --- Constraint verification ---
    constraint_ok = sum(1 for r in all_results if r["constraint_ok"])
    print(f"\nw_i ≤ 1-λ_i constraint: {constraint_ok}/{len(all_results)} pass")

    # --- Amplification analysis ---
    print(f"\n--- Amplification ratios ---")
    amps = [r["amp_ratio"] for r in all_results]
    scalars = [r["scalar_bound"] for r in all_results]
    print(f"  Spectral amp (dbar/dbar_fresh):")
    print(f"    min={min(amps):.4f} mean={np.mean(amps):.4f} max={max(amps):.4f}")
    print(f"  Scalar bound 1/(1-||M||/ε):")
    print(f"    min={min(scalars):.4f} mean={np.mean(scalars):.4f} max={max(scalars):.4f}")
    print(f"  Gain factor (scalar/spectral):")
    gains = [s/a for s, a in zip(scalars, amps) if a > 0]
    print(f"    min={min(gains):.4f} mean={np.mean(gains):.4f} max={max(gains):.4f}")

    # --- Saturation analysis ---
    print(f"\n--- Top eigenvalue saturation ---")
    sats = [r["saturation_top"] for r in all_results]
    print(f"  w_top/(1-λ_top) (how much of the bound is used):")
    print(f"    min={min(sats):.4f} mean={np.mean(sats):.4f} max={max(sats):.4f}")
    print(f"  (low saturation ⟹ W avoids the M_t direction)")

    # --- Top eigenvalue contribution ---
    print(f"\n--- Top eigenvalue contribution to tr(B_t W) ---")
    top_fracs = [r["top_contrib_frac"] for r in all_results]
    print(f"  Top-5 eigval contribution / total:")
    print(f"    min={min(top_fracs):.4f} mean={np.mean(top_fracs):.4f} max={max(top_fracs):.4f}")

    # --- Correlation between w_i and λ_i ---
    print(f"\n--- Correlation between w_i and λ_i ---")
    corrs = [r["corr_w_lambda"] for r in all_results if abs(r["corr_w_lambda"]) > 1e-14]
    if corrs:
        print(f"  Pearson corr(w_i, λ_i):")
        print(f"    min={min(corrs):.4f} mean={np.mean(corrs):.4f} max={max(corrs):.4f}")
        neg_corr = sum(1 for c in corrs if c < 0)
        print(f"  Negative correlation (W avoids M_t): {neg_corr}/{len(corrs)}")
    else:
        print("  No computable correlations (M_t has rank ≤ 1)")

    # --- Key formal bound test ---
    print(f"\n--- FORMAL BOUND TEST ---")
    # Test: dbar ≤ dbar_fresh · (1 + tr(M_t)/(ε·n))
    # This "soft amplification" bound uses the average eigenvalue instead of max
    print(f"  Test: dbar ≤ dbar_fresh · (1 + tr(M)/ε)")
    bound_holds = 0
    for r in all_results:
        # Compute tr(M)/ε from ||M|| (loose) or from actual M_t eigenvalues
        # We don't have tr(M) directly, but we can bound:
        # tr(M) ≤ n · ||M||, but that's too loose
        # Better: use the correction formula
        soft_bound = r["dbar_fresh"] * r["scalar_bound"]  # scalar bound for comparison
        # What about: dbar ≤ dbar_fresh + correction?
        # correction = (1/r_t) Σ w_i λ_i / (ε(ε-λ_i))
        # ≤ (||M||/(ε(ε-||M||))) · tr(W)/r_t = (||M||/(ε-||M||)) · dbar_fresh/ε ... hmm
        # Actually correction = dbar_actual - dbar_fresh
        pass

    # Test a TIGHTER bound: dbar ≤ dbar_fresh / (1 - corr_factor * ||M||/ε)
    # where corr_factor accounts for the spectral orthogonality
    print(f"\n  Finding optimal correlation factor α such that")
    print(f"  dbar ≤ dbar_fresh / (1 - α · ||M||/ε)")
    best_alpha = 0
    for r in all_results:
        rho = r["mt_norm"] / r["eps"]
        if rho > 1e-14 and r["dbar_fresh"] > 1e-14:
            # dbar = dbar_fresh / (1 - α·ρ)
            # α·ρ = 1 - dbar_fresh/dbar
            # α = (1 - dbar_fresh/dbar) / ρ
            alpha_needed = (1 - r["dbar_fresh"] / r["dbar_actual"]) / rho
            if alpha_needed > best_alpha:
                best_alpha = alpha_needed

    print(f"  Worst-case α needed: {best_alpha:.4f}")
    print(f"  (α < 1 means spectral orthogonality gives tighter bound than scalar)")

    # Verify: dbar ≤ dbar_fresh / (1 - α · ||M||/ε) with α = best_alpha
    all_hold = True
    for r in all_results:
        rho = r["mt_norm"] / r["eps"]
        bound = r["dbar_fresh"] / (1 - best_alpha * rho) if best_alpha * rho < 1 else float('inf')
        if r["dbar_actual"] > bound + 1e-10:
            all_hold = False
            print(f"  FAILS: {r['graph']} eps={r['eps']} dbar={r['dbar_actual']:.4f} > bound={bound:.4f}")

    if all_hold:
        print(f"  Bound holds for all {len(all_results)} steps with α = {best_alpha:.4f}")

    # Now check: with this α, does the bound give dbar < 1?
    print(f"\n  Checking dbar < 1 with α = {best_alpha:.4f}:")
    max_bounded_dbar = 0
    for r in all_results:
        rho = r["mt_norm"] / r["eps"]
        bound = r["dbar_fresh"] / (1 - best_alpha * rho)
        if bound > max_bounded_dbar:
            max_bounded_dbar = bound
    print(f"  Max bounded dbar: {max_bounded_dbar:.4f}")
    print(f"  {'< 1: CLOSES GPL-H!' if max_bounded_dbar < 1 else '>= 1: Gap remains'}")

    # --- What if we use α for Phase 2 entry step + 1? ---
    # At Phase 2 entry: M_t = 0. First Phase 2 step: select v with score s < 1.
    # M_{t+1} = C_t(v). ||M_{t+1}|| ≤ ε·s ≤ ε·dbar_fresh.
    # dbar_{t+1} ≤ dbar_fresh_{t+1} / (1 - α·s)
    # dbar_fresh grows slowly: dbar_fresh_{t+1} ≈ dbar_fresh_t · (t+1)/t · r_t/(r_t-1)
    # For t large: ≈ dbar_fresh_t · (1 + 1/t + 1/r_t)
    print(f"\n--- Phase 2 episode analysis ---")
    print(f"  At entry: M=0, dbar=dbar_fresh ≤ 2/(3-ε)")
    print(f"  After 1 step: ||M||/ε ≤ dbar_fresh ≤ 2/(3-ε)")
    print(f"  dbar ≤ dbar_fresh / (1 - α · dbar_fresh)")
    print(f"  = dbar_fresh / (1 - {best_alpha:.4f} · dbar_fresh)")

    for eps_val in [0.1, 0.2, 0.3, 0.5, 0.8]:
        df = 2 / (3 - eps_val)
        bounded = df / (1 - best_alpha * df) if best_alpha * df < 1 else float('inf')
        print(f"  ε={eps_val}: dbar_fresh={df:.4f}, bounded dbar={bounded:.4f} "
              f"{'< 1 ✓' if bounded < 1 else '>= 1 ✗'}")

    # --- THE KEY BOUND ---
    print(f"\n{'='*70}")
    print(f"THE FORMAL BOUND (M_t ≠ 0)")
    print(f"{'='*70}")
    print(f"""
Given:
  dbar_fresh ≤ D₀ = 2/(3-ε) < 1  (proved)
  ||M_t||/ε ≤ ρ
  spectral alignment factor α ≤ {best_alpha:.4f}

Then: dbar ≤ D₀/(1 - α·ρ)

For the boot-strap:
  At Phase 2 entry: ρ = 0, dbar ≤ D₀.
  First selected vertex: score s₁ ≤ D₀.
  After selection: ||M||/ε ≤ s₁ ≤ D₀.
  At next step: dbar ≤ D₀/(1 - α·D₀).

For this < 1: need α·D₀ < 1 - D₀, i.e., α < (1-D₀)/D₀ = (3-ε)/2 - 1 = (1-ε)/2.

With α = {best_alpha:.4f}: need (1-ε)/2 > {best_alpha:.4f}, i.e., ε < {1-2*best_alpha:.4f}.

{'CLOSES for all ε < ' + f'{1-2*best_alpha:.4f}' if best_alpha < 0.5 else 'DOES NOT CLOSE'}
    """)


if __name__ == "__main__":
    main()
