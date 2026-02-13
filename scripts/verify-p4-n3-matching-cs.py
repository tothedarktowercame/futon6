#!/usr/bin/env python3
"""Task B (P4): n=3 matching-average Cauchy-Schwarz analysis.

Goal:
  Examine whether Cauchy-Schwarz on the matching-average score representation
  can recover Stam for n=3, and identify where the route is weak.

For each trial:
  1. Build centered real-rooted cubics p(x)=x^3-sx-u, q(x)=x^3-tx-v.
  2. Build six matching polynomials m_sigma from roots gamma_i + delta_sigma(i).
  3. Form c(x) as average of matching polynomials and verify c == p boxplus_3 q.
  4. At each root zeta_k of c:
       S_k(c) = c''(zeta_k)/(2 c'(zeta_k))
             = sum_sigma w_sigma,k * alpha_sigma,k
       with alpha_sigma,k = m''_sigma(zeta_k)/(2 m'_sigma(zeta_k)),
            w_sigma,k = m'_sigma(zeta_k) / sum_tau m'_tau(zeta_k).
  5. Test CS/Jensen formulations and compare implied bounds against Stam.
"""

from itertools import permutations
import argparse
import math

import numpy as np


def phi_n(roots):
    """Phi_n = sum_i (sum_{j != i} 1/(lambda_i-lambda_j))^2 for distinct real roots."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        force = 0.0
        for j in range(n):
            if j == i:
                continue
            diff = roots[i] - roots[j]
            if abs(diff) < 1e-12:
                return float("inf")
            force += 1.0 / diff
        total += force * force
    return total


def finite_free_conv(p_coeffs, q_coeffs):
    """MSS finite free additive convolution coefficients (descending powers)."""
    n = len(p_coeffs) - 1
    c = np.zeros(n + 1, dtype=float)
    c[0] = 1.0
    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            w = math.factorial(n - i) * math.factorial(n - j)
            w /= math.factorial(n) * math.factorial(n - k)
            s += w * p_coeffs[i] * q_coeffs[j]
        c[k] = s
    return c


def poly_score_at(coeffs, x):
    """Return S = p''(x)/(2 p'(x)), plus p'(x), p''(x)."""
    d1 = np.polyder(coeffs)
    d2 = np.polyder(d1)
    p1 = np.polyval(d1, x)
    p2 = np.polyval(d2, x)
    if abs(p1) < 1e-12:
        return np.nan, np.nan, np.nan
    return p2 / (2.0 * p1), p1, p2


def sample_centered_cubic(rng):
    """Sample centered cubic x^3-sx-u with three distinct real roots."""
    s = rng.uniform(0.5, 5.0)
    umax = 2.0 * (s / 3.0) ** 1.5
    u = rng.uniform(-0.95 * umax, 0.95 * umax)
    coeffs = np.array([1.0, 0.0, -s, -u], dtype=float)
    roots = np.roots(coeffs)
    if np.max(np.abs(roots.imag)) > 1e-9:
        return None, None, None
    roots = np.sort(roots.real)
    if np.min(np.diff(roots)) < 1e-8:
        return None, None, None
    return coeffs, roots, (s, u)


def matching_coeffs(gamma, delta):
    """Return list of coefficients for six matching polynomials."""
    coeffs = []
    for perm in permutations(range(3)):
        r = gamma + delta[list(perm)]
        coeffs.append(np.poly(r))
    return coeffs


def run(n_trials, seed):
    rng = np.random.default_rng(seed)
    perms = list(permutations(range(3)))
    n_perm = len(perms)
    tol = 1e-9

    trials = 0
    rootpoints = 0
    skipped = 0

    max_conv_coeff_err = 0.0
    max_score_identity_err = 0.0
    min_weight = float("inf")
    min_var_gap = float("inf")

    all_nonneg_weight_count = 0
    same_sign_beta_count = 0

    stam_violations = 0

    # Method quality stats:
    # method1: Jensen with signed derivative weights w (valid only when w>=0).
    # method2: ratio-of-expectations CS on (a,b)=(m'',m').
    # method3: absolute-weight CS bound.
    method1_valid_trials = 0
    method1_implies_stam = 0
    method2_implies_stam = 0
    method3_implies_stam = 0

    method1_ratio = []
    method2_ratio = []
    method3_ratio = []

    witness_method1_weak = None
    witness_negative_weights = None

    for _ in range(n_trials):
        p_coeffs, gamma, pu = sample_centered_cubic(rng)
        q_coeffs, delta, qv = sample_centered_cubic(rng)
        if p_coeffs is None or q_coeffs is None:
            skipped += 1
            continue

        # Build c from MSS and from explicit matching average.
        conv_mss = finite_free_conv(p_coeffs, q_coeffs)
        match_list = matching_coeffs(gamma, delta)
        conv_avg = np.mean(np.array(match_list), axis=0)
        conv_err = float(np.max(np.abs(conv_mss - conv_avg)))
        max_conv_coeff_err = max(max_conv_coeff_err, conv_err)

        conv_coeffs = conv_avg
        conv_roots = np.roots(conv_coeffs)
        if np.max(np.abs(conv_roots.imag)) > 1e-8:
            skipped += 1
            continue
        zetas = np.sort(conv_roots.real)
        if np.min(np.diff(zetas)) < 1e-8:
            skipped += 1
            continue

        phi_p = phi_n(gamma)
        phi_q = phi_n(delta)
        phi_c = phi_n(zetas)
        if not np.isfinite(phi_p) or not np.isfinite(phi_q) or not np.isfinite(phi_c):
            skipped += 1
            continue

        rhs_stam = 1.0 / phi_p + 1.0 / phi_q
        lhs_stam = 1.0 / phi_c
        if lhs_stam < rhs_stam - 1e-10:
            stam_violations += 1

        B1 = 0.0
        B2 = 0.0
        B3 = 0.0
        trial_all_nonneg = True

        for zeta in zetas:
            rootpoints += 1

            s_c, c1, _ = poly_score_at(conv_coeffs, zeta)
            if not np.isfinite(s_c):
                trial_all_nonneg = False
                continue

            alphas = np.zeros(n_perm, dtype=float)
            betas = np.zeros(n_perm, dtype=float)
            a_vals = np.zeros(n_perm, dtype=float)  # m'' at zeta

            for i, mc in enumerate(match_list):
                s_i, b_i, a_i = poly_score_at(mc, zeta)
                if not np.isfinite(s_i):
                    trial_all_nonneg = False
                    break
                alphas[i] = float(np.real(s_i))
                betas[i] = float(np.real(b_i))
                a_vals[i] = float(np.real(a_i))
            else:
                sum_beta = np.sum(betas)
                if abs(sum_beta) < 1e-12:
                    trial_all_nonneg = False
                    continue

                weights = betas / sum_beta
                min_weight = min(min_weight, float(np.min(weights)))
                if np.min(weights) < -tol:
                    trial_all_nonneg = False
                    if witness_negative_weights is None:
                        witness_negative_weights = {
                            "zeta": float(zeta),
                            "min_w": float(np.min(weights)),
                            "weights": weights.copy(),
                        }
                else:
                    all_nonneg_weight_count += 1

                if (np.all(betas > 0) or np.all(betas < 0)):
                    same_sign_beta_count += 1

                # Exact score identity at root of c.
                s_from_matching = float(np.dot(weights, alphas))
                max_score_identity_err = max(max_score_identity_err, abs(s_c - s_from_matching))

                # Jensen/variance gap with signed weights.
                var_gap = float(np.dot(weights, alphas * alphas) - s_from_matching * s_from_matching)
                min_var_gap = min(min_var_gap, var_gap)

                # Method 1: weighted second moment bound with derivative weights.
                B1 += float(np.dot(weights, alphas * alphas))

                # Method 2: ratio-of-expectations Cauchy-Schwarz:
                # (E[a])^2 <= E[a^2/b^2] E[b^2], where a=m'', b=m'.
                e_a2_over_b2 = np.mean((a_vals * a_vals) / (betas * betas))
                e_b2 = np.mean(betas * betas)
                e_b = np.mean(betas)
                B2 += 0.25 * e_a2_over_b2 * (e_b2 / (e_b * e_b))

                # Method 3: absolute-weight C-S:
                # |sum beta*alpha|^2 <= (sum |beta| alpha^2)(sum |beta|).
                num = np.sum(np.abs(betas) * alphas * alphas)
                den = sum_beta * sum_beta
                B3 += (num * np.sum(np.abs(betas))) / den
                continue

            # Loop broke (non-finite matching score)
            trial_all_nonneg = False
            break

        trials += 1

        if B1 > 0:
            L1 = 1.0 / B1
            method1_ratio.append(L1 / rhs_stam)
            if trial_all_nonneg:
                method1_valid_trials += 1
                if L1 >= rhs_stam - 1e-10:
                    method1_implies_stam += 1
                elif witness_method1_weak is None:
                    witness_method1_weak = {
                        "p": p_coeffs.copy(),
                        "q": q_coeffs.copy(),
                        "stam_rhs": float(rhs_stam),
                        "bound": float(L1),
                    }

        if B2 > 0:
            L2 = 1.0 / B2
            method2_ratio.append(L2 / rhs_stam)
            if L2 >= rhs_stam - 1e-10:
                method2_implies_stam += 1

        if B3 > 0:
            L3 = 1.0 / B3
            method3_ratio.append(L3 / rhs_stam)
            if L3 >= rhs_stam - 1e-10:
                method3_implies_stam += 1

    def ratio_stats(arr):
        if not arr:
            return (float("nan"), float("nan"), float("nan"))
        vals = np.array(arr, dtype=float)
        return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))

    m1_min, m1_mean, m1_max = ratio_stats(method1_ratio)
    m2_min, m2_mean, m2_max = ratio_stats(method2_ratio)
    m3_min, m3_mean, m3_max = ratio_stats(method3_ratio)

    print("=" * 76)
    print("Task B: n=3 matching-average Cauchy-Schwarz analysis")
    print("=" * 76)
    print(f"trials requested: {n_trials}")
    print(f"trials analyzed:  {trials}")
    print(f"trials skipped:   {skipped}")
    print(f"rootpoints used:  {rootpoints}")
    print()
    print("[Sanity checks]")
    print(f"max ||matching-average coeffs - MSS coeffs||_inf = {max_conv_coeff_err:.3e}")
    print(f"max |S_c - sum w_sigma alpha_sigma| at roots      = {max_score_identity_err:.3e}")
    print()
    print("[Weight geometry at roots zeta_k of c]")
    if rootpoints > 0:
        print(f"fraction with all w_sigma >= 0: {all_nonneg_weight_count}/{rootpoints} "
              f"({100.0 * all_nonneg_weight_count / rootpoints:.1f}%)")
        print(f"fraction with all m_sigma'(zeta_k) same sign: {same_sign_beta_count}/{rootpoints} "
              f"({100.0 * same_sign_beta_count / rootpoints:.1f}%)")
    print(f"min observed weight w_sigma,k = {min_weight:.6f}")
    print(f"min observed variance gap [sum w a^2 - (sum w a)^2] = {min_var_gap:.6e}")
    print()
    print("[Stam check]")
    print(f"Stam violations (numeric): {stam_violations}/{trials}")
    print()
    print("[C-S routes vs Stam]")
    print("ratio = (lower bound from route) / (1/Phi(p)+1/Phi(q)); ratio >= 1 would prove Stam")
    print(f"method1 (derivative-weight Jensen): valid-on-cone trials = {method1_valid_trials}/{trials}, "
          f"implies Stam on {method1_implies_stam}/{method1_valid_trials if method1_valid_trials else 1}")
    print(f"  ratio stats: min={m1_min:.6f}, mean={m1_mean:.6f}, max={m1_max:.6f}")
    print(f"method2 (ratio-of-expectations CS): implies Stam on {method2_implies_stam}/{trials}")
    print(f"  ratio stats: min={m2_min:.6f}, mean={m2_mean:.6f}, max={m2_max:.6f}")
    print(f"method3 (absolute-weight CS): implies Stam on {method3_implies_stam}/{trials}")
    print(f"  ratio stats: min={m3_min:.6f}, mean={m3_mean:.6f}, max={m3_max:.6f}")
    print()

    if witness_method1_weak is not None:
        print("[Witness: method1 valid but too weak for Stam]")
        print(f"stam rhs = {witness_method1_weak['stam_rhs']:.8f}")
        print(f"method1 bound = {witness_method1_weak['bound']:.8f}")
        print(f"p coeffs = {witness_method1_weak['p']}")
        print(f"q coeffs = {witness_method1_weak['q']}")
        print()

    if witness_negative_weights is not None:
        print("[Witness: negative derivative weight at a root of c]")
        print(f"zeta = {witness_negative_weights['zeta']:.8f}")
        print(f"min weight = {witness_negative_weights['min_w']:.8f}")
        print(f"weights = {witness_negative_weights['weights']}")
        print()

    print("Interpretation:")
    print("1. The score identity at roots of c is exact as a ratio/weighted average.")
    print("2. A Jensen/C-S step is available when derivative weights are nonnegative.")
    print("3. Even then, this direct route usually gives a lower bound weaker than Stam.")
    print("4. Generic ratio-of-expectations C-S bounds are much weaker.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260213)
    args = parser.parse_args()
    run(args.trials, args.seed)


if __name__ == "__main__":
    main()
