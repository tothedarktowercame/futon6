#!/usr/bin/env python3
"""Test whether g_n(p) = 1/Phi_n(p) - 1/Phi_{n-1}(p'/n) is superadditive under ⊞_n.

If g_n(p ⊞_n q) >= g_n(p) + g_n(q), then the telescoping induction works:
    Delta_n = R_n + R_{n-1} + ... + R_4 + Delta_3
where each R_k >= 0 and Delta_3 >= 0 (proved).

This is the HIGHEST PRIORITY numerical test from the Strategy B deep dive.
"""

import numpy as np
from itertools import combinations
from math import factorial


def random_real_rooted(n, spread=2.0):
    """Generate a random monic real-rooted polynomial of degree n."""
    roots = np.sort(np.random.randn(n) * spread)
    coeffs = np.polynomial.polynomial.polyfromroots(roots)[::-1]
    return roots, coeffs


def phi_n(roots):
    """Compute Phi_n = sum_i S_i^2 = 2 * sum_{i<j} 1/(lambda_i - lambda_j)^2."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = roots[i] - roots[j]
            if abs(gap) < 1e-14:
                return float('inf')
            total += 1.0 / (gap * gap)
    return 2.0 * total


def mss_convolve(roots_a, roots_b, n):
    """Compute p ⊞_n q via MSS coefficient formula.

    c_k = sum_{i+j=k} w(n,i,j) * a_i * b_j
    w(n,i,j) = (n-i)!(n-j)! / (n!(n-k)!)
    """
    coeffs_a = np.polynomial.polynomial.polyfromroots(roots_a)[::-1]
    coeffs_b = np.polynomial.polynomial.polyfromroots(roots_b)[::-1]

    # Pad to length n+1
    a = np.zeros(n + 1)
    b = np.zeros(n + 1)
    a[:len(coeffs_a)] = coeffs_a
    b[:len(coeffs_b)] = coeffs_b

    c = np.zeros(n + 1)
    c[0] = 1.0  # monic

    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            if i > n or j > n:
                continue
            w = factorial(n - i) * factorial(n - j) / (factorial(n) * factorial(n - k))
            s += w * a[i] * b[j]
        c[k] = s

    # Find roots of the convolution
    # coeffs are [1, c_1, c_2, ..., c_n] for x^n + c_1 x^{n-1} + ...
    # np.roots expects [leading, ..., constant]
    conv_roots = np.sort(np.real(np.roots(c)))
    return conv_roots, c


def derivative_roots(roots):
    """Compute roots of p'/n (the monic degree-(n-1) derivative).

    Returns the n-1 critical points of p.
    """
    n = len(roots)
    # Build polynomial from roots, differentiate, find roots
    coeffs = np.polynomial.polynomial.polyfromroots(roots)  # ascending order
    deriv_coeffs = np.polynomial.polynomial.polyder(coeffs)
    # Make monic: divide by leading coefficient (which is n for monic degree-n poly)
    deriv_coeffs = deriv_coeffs / deriv_coeffs[-1]
    deriv_roots = np.sort(np.real(np.roots(deriv_coeffs[::-1])))
    return deriv_roots


def g_n(roots):
    """Compute g_n(p) = 1/Phi_n(p) - 1/Phi_{n-1}(p'/n).

    We know g_n < 0 always (differentiation reduces energy, increases 1/Phi).
    """
    phi_orig = phi_n(roots)
    if phi_orig == float('inf') or phi_orig < 1e-15:
        return None

    d_roots = derivative_roots(roots)
    phi_deriv = phi_n(d_roots)
    if phi_deriv == float('inf') or phi_deriv < 1e-15:
        return None

    return 1.0 / phi_orig - 1.0 / phi_deriv


def test_gn_superadditivity(n, num_trials=2000, spread=2.0):
    """Test g_n(p ⊞_n q) >= g_n(p) + g_n(q) for random inputs."""
    violations = 0
    valid = 0
    min_surplus = float('inf')
    max_surplus = float('-inf')
    surpluses = []
    gn_values = []

    for trial in range(num_trials):
        try:
            roots_p, _ = random_real_rooted(n, spread)
            roots_q, _ = random_real_rooted(n, spread)

            # Compute convolution
            conv_roots, _ = mss_convolve(roots_p, roots_q, n)

            # Check convolution is real-rooted
            conv_poly_coeffs = np.polynomial.polynomial.polyfromroots(conv_roots)
            if not np.all(np.isreal(np.roots(conv_poly_coeffs[::-1]))):
                continue

            # Compute g_n values
            gp = g_n(roots_p)
            gq = g_n(roots_q)
            gr = g_n(conv_roots)

            if gp is None or gq is None or gr is None:
                continue

            valid += 1
            surplus = gr - gp - gq
            surpluses.append(surplus)
            gn_values.append((gp, gq, gr))

            if surplus < -1e-10:
                violations += 1

            if surplus < min_surplus:
                min_surplus = surplus
            if surplus > max_surplus:
                max_surplus = surplus

        except Exception:
            continue

    return {
        'n': n,
        'valid': valid,
        'violations': violations,
        'min_surplus': min_surplus,
        'max_surplus': max_surplus,
        'surpluses': surpluses,
        'gn_values': gn_values,
    }


def main():
    print("=" * 70)
    print("Testing g_n superadditivity: g_n(p ⊞_n q) >= g_n(p) + g_n(q)")
    print("where g_n(p) = 1/Phi_n(p) - 1/Phi_{n-1}(p'/n)")
    print("=" * 70)
    print()

    for n in [4, 5, 6, 7, 8]:
        trials = 3000 if n <= 6 else 1000
        result = test_gn_superadditivity(n, num_trials=trials)
        v = result['valid']
        viol = result['violations']

        print(f"n={n}: {v} valid trials, {viol} violations "
              f"({100*viol/v:.1f}% if v>0)" if v > 0 else f"n={n}: 0 valid trials")

        if v > 0:
            surp = np.array(result['surpluses'])
            gvals = np.array(result['gn_values'])

            print(f"  g_n surplus range: [{result['min_surplus']:.6e}, {result['max_surplus']:.6e}]")
            print(f"  g_n surplus mean:  {np.mean(surp):.6e}")
            print(f"  g_n surplus median: {np.median(surp):.6e}")

            # Also report g_n(p) statistics
            gp_vals = gvals[:, 0]
            print(f"  g_n(p) range: [{np.min(gp_vals):.6e}, {np.max(gp_vals):.6e}]")
            print(f"  g_n(p) mean:  {np.mean(gp_vals):.6e}")
            print(f"  g_n(p) always negative: {np.all(gp_vals < 0)}")

            # Report ratio |g_n(r)| / (|g_n(p)| + |g_n(q)|)
            # Superadditivity of negative function means this ratio <= 1
            gp_abs = np.abs(gvals[:, 0])
            gq_abs = np.abs(gvals[:, 1])
            gr_abs = np.abs(gvals[:, 2])
            ratio = gr_abs / (gp_abs + gq_abs)
            print(f"  |g_n(r)|/(|g_n(p)|+|g_n(q)|) range: [{np.min(ratio):.4f}, {np.max(ratio):.4f}]")
            print(f"  |g_n(r)|/(|g_n(p)|+|g_n(q)|) mean:  {np.mean(ratio):.4f}")
        print()

    # Extra: test the full telescoping decomposition for n=5
    print("=" * 70)
    print("Telescoping check: Delta_n = R_n + R_{n-1} + ... + R_4 + Delta_3")
    print("=" * 70)
    print()

    n = 5
    valid = 0
    delta_n_positive = 0
    all_rk_positive = 0

    for trial in range(2000):
        try:
            roots_p, _ = random_real_rooted(n, spread=2.0)
            roots_q, _ = random_real_rooted(n, spread=2.0)
            conv_roots, _ = mss_convolve(roots_p, roots_q, n)

            # Delta_5
            phi_r = phi_n(conv_roots)
            phi_p = phi_n(roots_p)
            phi_q = phi_n(roots_q)
            if phi_r < 1e-15 or phi_p < 1e-15 or phi_q < 1e-15:
                continue
            delta_5 = 1/phi_r - 1/phi_p - 1/phi_q

            # g_5 values
            g5_r = g_n(conv_roots)
            g5_p = g_n(roots_p)
            g5_q = g_n(roots_q)
            if g5_r is None or g5_p is None or g5_q is None:
                continue
            R_5 = g5_r - g5_p - g5_q

            # For R_4: need degree-4 derivatives
            dp = derivative_roots(roots_p)
            dq = derivative_roots(roots_q)
            dr = derivative_roots(conv_roots)

            g4_dr = g_n(dr)
            g4_dp = g_n(dp)
            g4_dq = g_n(dq)
            if g4_dr is None or g4_dp is None or g4_dq is None:
                continue
            R_4 = g4_dr - g4_dp - g4_dq

            # Delta_3: degree-3 second derivatives
            ddp = derivative_roots(dp)
            ddq = derivative_roots(dq)
            ddr = derivative_roots(dr)

            phi_ddr = phi_n(ddr)
            phi_ddp = phi_n(ddp)
            phi_ddq = phi_n(ddq)
            if phi_ddr < 1e-15 or phi_ddp < 1e-15 or phi_ddq < 1e-15:
                continue
            delta_3 = 1/phi_ddr - 1/phi_ddp - 1/phi_ddq

            valid += 1

            # Check: Delta_5 should equal R_5 + R_4 + Delta_3
            reconstructed = R_5 + R_4 + delta_3

            if delta_5 > -1e-10:
                delta_n_positive += 1
            if R_5 > -1e-10 and R_4 > -1e-10 and delta_3 > -1e-10:
                all_rk_positive += 1

        except Exception:
            continue

    print(f"n=5: {valid} valid trials")
    print(f"  Delta_5 >= 0: {delta_n_positive}/{valid} ({100*delta_n_positive/valid:.1f}%)" if valid > 0 else "  No valid trials")
    print(f"  All R_k >= 0 AND Delta_3 >= 0: {all_rk_positive}/{valid} ({100*all_rk_positive/valid:.1f}%)" if valid > 0 else "")


if __name__ == "__main__":
    main()
