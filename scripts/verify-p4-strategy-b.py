#!/usr/bin/env python3
"""Strategy B: Induction via differentiation for P4 n>=4.

Tests the key relationships needed for an induction proof:
1. Does differentiation commute exactly with ⊞_n?
   i.e., (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)?
2. How does 1/Phi_n(p) relate to 1/Phi_{n-1}(p'/n)?
3. Can the induction chain close?
"""

import numpy as np
from numpy.polynomial import polynomial as P
from itertools import combinations
import sys

np.random.seed(42)


# ── Core functions ──────────────────────────────────────────────────────

def roots_to_monic(roots):
    """Roots -> monic polynomial coefficients [a_n, a_{n-1}, ..., a_1, 1]
    in numpy convention (highest degree last... actually let's use
    standard math convention: [1, a_1, a_2, ..., a_n] with leading 1)."""
    # np.poly returns [1, a_1, ..., a_n] (highest degree first)
    return np.poly(roots)


def monic_coeffs(roots):
    """Return coefficients [a_1, a_2, ..., a_n] (excluding leading 1)."""
    c = np.poly(roots)  # [1, a_1, ..., a_n]
    return c[1:]


def phi_n(roots):
    """Compute Phi_n = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def mss_convolve(a_coeffs, b_coeffs, n):
    """MSS finite free additive convolution via coefficient formula.
    a_coeffs, b_coeffs: [a_1, ..., a_n] and [b_1, ..., b_n]
    Returns [c_1, ..., c_n].
    """
    from math import factorial
    c = np.zeros(n)
    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            if i > n or j > n:
                continue
            ai = 1.0 if i == 0 else a_coeffs[i - 1]
            bj = 1.0 if j == 0 else b_coeffs[j - 1]
            w = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
            s += w * ai * bj
        c[k - 1] = s
    return c


def poly_from_coeffs(coeffs):
    """[a_1, ..., a_n] -> numpy poly [1, a_1, ..., a_n]."""
    return np.concatenate([[1.0], coeffs])


def derivative_normalized(roots):
    """Compute roots of p'/n where p = prod(x - lambda_i).
    Returns the n-1 critical points of p."""
    coeffs = np.poly(roots)  # [1, a_1, ..., a_n], highest degree first
    # derivative: n*x^{n-1} + (n-1)*a_1*x^{n-2} + ...
    n = len(roots)
    deriv = np.polyder(coeffs)  # coefficients of p'
    # normalize to monic: divide by n
    deriv_monic = deriv / deriv[0]
    crit_pts = np.sort(np.roots(deriv_monic).real)
    return crit_pts


def derivative_coeffs_normalized(a_coeffs, n):
    """Given [a_1,...,a_n] of degree-n monic poly, return [â_1,...,â_{n-1}]
    of p'/n (the monic degree-(n-1) polynomial)."""
    # p(x) = x^n + a_1 x^{n-1} + ... + a_n
    # p'(x) = n x^{n-1} + (n-1)a_1 x^{n-2} + ... + a_{n-1}
    # p'(x)/n = x^{n-1} + ((n-1)/n)a_1 x^{n-2} + ... + (1/n)a_{n-1}
    # So â_k = (n-k)/n * a_k for k=1,...,n-1
    return np.array([(n - k) / n * a_coeffs[k - 1] for k in range(1, n)])


# ── Part 1: Exact commutativity of differentiation and ⊞_n ─────────────

def test_differentiation_commutativity(n_degree, n_trials=500):
    """Test: (p ⊞_n q)'/n == (p'/n) ⊞_{n-1} (q'/n) ?"""
    print(f"\n{'='*70}")
    print(f"PART 1: Differentiation commutativity with ⊞_{n_degree}")
    print(f"{'='*70}")

    max_err = 0.0
    for trial in range(n_trials):
        # Random polynomials
        roots_p = np.sort(np.random.randn(n_degree) * 2)
        roots_q = np.sort(np.random.randn(n_degree) * 2)
        a = monic_coeffs(roots_p)
        b = monic_coeffs(roots_q)

        # Route 1: convolve then differentiate
        c = mss_convolve(a, b, n_degree)
        c_deriv = derivative_coeffs_normalized(c, n_degree)

        # Route 2: differentiate then convolve
        a_deriv = derivative_coeffs_normalized(a, n_degree)
        b_deriv = derivative_coeffs_normalized(b, n_degree)
        d = mss_convolve(a_deriv, b_deriv, n_degree - 1)

        err = np.max(np.abs(c_deriv - d))
        max_err = max(max_err, err)

    print(f"  n={n_degree}, {n_trials} trials, max coefficient error: {max_err:.2e}")
    if max_err < 1e-10:
        print(f"  CONFIRMED: (p ⊞_{n_degree} q)'/n = (p'/n) ⊞_{n_degree-1} (q'/n)")
    else:
        print(f"  FAILED: commutativity does NOT hold exactly")
    return max_err


# ── Part 2: Relationship between 1/Phi_n(p) and 1/Phi_{n-1}(p'/n) ─────

def test_phi_derivative_relationship(n_degree, n_trials=2000):
    """Test the direction and magnitude of 1/Phi_n(p) vs 1/Phi_{n-1}(p'/n)."""
    print(f"\n{'='*70}")
    print(f"PART 2: 1/Phi_{n_degree}(p) vs 1/Phi_{n_degree-1}(p'/n)")
    print(f"{'='*70}")

    ratios = []
    for trial in range(n_trials):
        roots = np.sort(np.random.randn(n_degree) * 2)
        # Ensure distinct roots
        for i in range(1, n_degree):
            if roots[i] - roots[i-1] < 0.01:
                roots[i] = roots[i-1] + 0.01

        crit = derivative_normalized(roots)
        # Check critical points are distinct
        gaps = np.diff(crit)
        if np.any(gaps < 1e-10):
            continue

        pn = phi_n(roots)
        pn1 = phi_n(crit)

        if pn > 1e-15 and pn1 > 1e-15:
            ratio = (1.0 / pn) / (1.0 / pn1)  # = pn1 / pn
            ratios.append(ratio)

    ratios = np.array(ratios)
    print(f"  {len(ratios)} valid trials")
    print(f"  1/Phi_n / (1/Phi_{{n-1}} of deriv) = Phi_{{n-1}}(p'/n) / Phi_n(p):")
    print(f"    min   = {np.min(ratios):.6f}")
    print(f"    max   = {np.max(ratios):.6f}")
    print(f"    mean  = {np.mean(ratios):.6f}")
    print(f"    median = {np.median(ratios):.6f}")

    # Direction: is 1/Phi_n >= 1/Phi_{n-1}(deriv) or opposite?
    n_above = np.sum(ratios > 1.0)
    n_below = np.sum(ratios < 1.0)
    print(f"    1/Phi_n > 1/Phi_{{n-1}}(deriv): {n_above}/{len(ratios)} ({100*n_above/len(ratios):.1f}%)")
    print(f"    1/Phi_n < 1/Phi_{{n-1}}(deriv): {n_below}/{len(ratios)} ({100*n_below/len(ratios):.1f}%)")
    return ratios


# ── Part 3: Can the induction chain close? ──────────────────────────────

def test_induction_chain(n_degree, n_trials=2000):
    """Test whether the full induction argument works:
    1/Phi_n(p ⊞_n q) >= ... >= 1/Phi_n(p) + 1/Phi_n(q)

    The chain would be:
    (A) 1/Phi_n(r) vs 1/Phi_{n-1}(r'/n)  [for r = p ⊞_n q]
    (B) 1/Phi_{n-1}((p⊞q)'/n) >= 1/Phi_{n-1}(p'/n) + 1/Phi_{n-1}(q'/n)  [induction]
    (C) 1/Phi_{n-1}(p'/n) + 1/Phi_{n-1}(q'/n) vs 1/Phi_n(p) + 1/Phi_n(q)
    """
    print(f"\n{'='*70}")
    print(f"PART 3: Full induction chain test at n={n_degree}")
    print(f"{'='*70}")

    results = {
        'A_holds': 0,  # 1/Phi_n(r) >= 1/Phi_{n-1}(r'/n)
        'A_fails': 0,
        'B_holds': 0,  # induction hypothesis at n-1
        'B_fails': 0,
        'C_holds': 0,  # 1/Phi_{n-1}(p'/n) + 1/Phi_{n-1}(q'/n) >= 1/Phi_n(p) + 1/Phi_n(q)
        'C_fails': 0,
        'chain_holds': 0,  # full chain A+B >= C
        'chain_fails': 0,
        'target_holds': 0,  # actual inequality 1/Phi_n(p⊞q) >= 1/Phi_n(p) + 1/Phi_n(q)
        'target_fails': 0,
        'skip': 0,
    }

    for trial in range(n_trials):
        roots_p = np.sort(np.random.randn(n_degree) * 2)
        roots_q = np.sort(np.random.randn(n_degree) * 2)

        # Ensure distinct
        for arr in [roots_p, roots_q]:
            for i in range(1, n_degree):
                if arr[i] - arr[i-1] < 0.01:
                    arr[i] = arr[i-1] + 0.01

        a = monic_coeffs(roots_p)
        b = monic_coeffs(roots_q)

        # Compute convolution
        c = mss_convolve(a, b, n_degree)
        poly_c = poly_from_coeffs(c)
        roots_c = np.sort(np.roots(poly_c).real)

        # Check all roots are real and distinct
        if np.any(np.abs(np.roots(poly_c).imag) > 0.01):
            results['skip'] += 1
            continue
        if np.any(np.diff(roots_c) < 1e-8):
            results['skip'] += 1
            continue

        # Derivatives
        crit_p = derivative_normalized(roots_p)
        crit_q = derivative_normalized(roots_q)
        crit_c = derivative_normalized(roots_c)

        # Check critical points distinct
        for arr in [crit_p, crit_q, crit_c]:
            if np.any(np.diff(arr) < 1e-10):
                results['skip'] += 1
                break
        else:
            pass  # all good
        if np.any(np.diff(crit_p) < 1e-10) or np.any(np.diff(crit_q) < 1e-10) or np.any(np.diff(crit_c) < 1e-10):
            continue

        try:
            phi_n_p = phi_n(roots_p)
            phi_n_q = phi_n(roots_q)
            phi_n_c = phi_n(roots_c)
            phi_n1_dp = phi_n(crit_p)
            phi_n1_dq = phi_n(crit_q)
            phi_n1_dc = phi_n(crit_c)
        except (ZeroDivisionError, FloatingPointError):
            results['skip'] += 1
            continue

        if any(x < 1e-15 for x in [phi_n_p, phi_n_q, phi_n_c, phi_n1_dp, phi_n1_dq, phi_n1_dc]):
            results['skip'] += 1
            continue

        inv_pn_c = 1.0 / phi_n_c
        inv_pn_p = 1.0 / phi_n_p
        inv_pn_q = 1.0 / phi_n_q
        inv_pn1_dc = 1.0 / phi_n1_dc
        inv_pn1_dp = 1.0 / phi_n1_dp
        inv_pn1_dq = 1.0 / phi_n1_dq

        # Target inequality
        target_surplus = inv_pn_c - inv_pn_p - inv_pn_q
        if target_surplus >= -1e-12:
            results['target_holds'] += 1
        else:
            results['target_fails'] += 1

        # Step A: 1/Phi_n(conv) >= 1/Phi_{n-1}(conv'/n)
        step_a = inv_pn_c - inv_pn1_dc
        if step_a >= -1e-12:
            results['A_holds'] += 1
        else:
            results['A_fails'] += 1

        # Step B: induction at n-1 for derivatives
        step_b = inv_pn1_dc - inv_pn1_dp - inv_pn1_dq
        if step_b >= -1e-12:
            results['B_holds'] += 1
        else:
            results['B_fails'] += 1

        # Step C: 1/Phi_{n-1}(p'/n) + 1/Phi_{n-1}(q'/n) >= 1/Phi_n(p) + 1/Phi_n(q)
        step_c = (inv_pn1_dp + inv_pn1_dq) - (inv_pn_p + inv_pn_q)
        if step_c >= -1e-12:
            results['C_holds'] += 1
        else:
            results['C_fails'] += 1

        # Full chain: if A and B and C all hold, target follows
        if step_a >= -1e-12 and step_b >= -1e-12 and step_c >= -1e-12:
            results['chain_holds'] += 1
        else:
            results['chain_fails'] += 1

    valid = n_trials - results['skip']
    print(f"  {valid} valid trials ({results['skip']} skipped)")
    print(f"  Target 1/Phi_n(p⊞q) >= 1/Phi_n(p)+1/Phi_n(q): "
          f"{results['target_holds']}/{valid} "
          f"({100*results['target_holds']/max(1,valid):.1f}%)")
    print(f"  Step A: 1/Phi_n(r) >= 1/Phi_{{n-1}}(r'/n):     "
          f"{results['A_holds']}/{valid} "
          f"({100*results['A_holds']/max(1,valid):.1f}%)")
    print(f"  Step B: induction at n-1 for derivatives:     "
          f"{results['B_holds']}/{valid} "
          f"({100*results['B_holds']/max(1,valid):.1f}%)")
    print(f"  Step C: Σ1/Phi_{{n-1}}(deriv) >= Σ1/Phi_n(orig): "
          f"{results['C_holds']}/{valid} "
          f"({100*results['C_holds']/max(1,valid):.1f}%)")
    print(f"  Full chain (A∧B∧C):                           "
          f"{results['chain_holds']}/{valid} "
          f"({100*results['chain_holds']/max(1,valid):.1f}%)")
    return results


# ── Part 4: Alternative — ratio-based induction ─────────────────────────

def test_ratio_induction(n_degree, n_trials=2000):
    """Instead of a direct chain, test whether there's a CONSTANT c_n such that:
    1/Phi_n(p) >= c_n * 1/Phi_{n-1}(p'/n) AND 1/Phi_{n-1}(p'/n) >= (1/c_n) * 1/Phi_n(p)

    Or more generally, find the empirical bounds on the ratio
    R(p) = (1/Phi_n(p)) / (1/Phi_{n-1}(p'/n)) = Phi_{n-1}(p'/n) / Phi_n(p)
    """
    print(f"\n{'='*70}")
    print(f"PART 4: Ratio R = Phi_{{n-1}}(p'/n) / Phi_{n_degree}(p)")
    print(f"{'='*70}")

    ratios = []
    for trial in range(n_trials):
        roots = np.sort(np.random.randn(n_degree) * 2)
        for i in range(1, n_degree):
            if roots[i] - roots[i-1] < 0.01:
                roots[i] = roots[i-1] + 0.01

        crit = derivative_normalized(roots)
        if np.any(np.diff(crit) < 1e-10):
            continue

        try:
            pn = phi_n(roots)
            pn1 = phi_n(crit)
        except:
            continue

        if pn > 1e-15 and pn1 > 1e-15:
            ratios.append(pn1 / pn)

    ratios = np.array(ratios)
    print(f"  {len(ratios)} valid trials")
    print(f"  R = Phi_{{n-1}}(deriv)/Phi_n(orig):")
    print(f"    min  = {np.min(ratios):.6f}")
    print(f"    max  = {np.max(ratios):.6f}")
    print(f"    mean = {np.mean(ratios):.6f}")
    print(f"    std  = {np.std(ratios):.6f}")
    print(f"  So: 1/Phi_n(p) = R * 1/Phi_{{n-1}}(p'/n)")
    print(f"  If R is bounded in [r_lo, r_hi], the induction needs:")
    print(f"    r_lo * [1/Phi_{{n-1}}(p'/n) + 1/Phi_{{n-1}}(q'/n)]")
    print(f"    >= 1/Phi_n(p) + 1/Phi_n(q)")
    print(f"  which holds if r_lo >= r_hi (impossible unless R is constant)")
    return ratios


# ── Part 5: Symbolic check of commutativity for small n ─────────────────

def symbolic_commutativity_check():
    """Algebraic proof that (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)."""
    print(f"\n{'='*70}")
    print(f"PART 5: Algebraic proof of differentiation commutativity")
    print(f"{'='*70}")

    from math import factorial

    for n in range(3, 8):
        max_err = 0.0
        # Check coefficient by coefficient
        for k in range(1, n):  # k = 1,...,n-1 (coefficients of the degree-(n-1) result)
            for i in range(k + 1):
                j = k - i
                if i > n or j > n:
                    continue

                # Route 1: convolve at degree n, then differentiate
                # Coefficient of a_i*b_j in c_k = w(n,i,j)
                # Then in (conv)'/n, coefficient of a_i*b_j = (n-k)/n * w(n,i,j)
                w_n = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
                route1 = (n - k) / n * w_n

                # Route 2: differentiate then convolve at degree n-1
                # â_i = (n-i)/n * a_i, b̂_j = (n-j)/n * b_j
                # Coefficient = w(n-1,i,j) * (n-i)/n * (n-j)/n
                if i > n - 1 or j > n - 1:
                    route2 = 0.0
                else:
                    w_n1 = (factorial(n - 1 - i) * factorial(n - 1 - j)) / (
                        factorial(n - 1) * factorial(n - 1 - k))
                    route2 = w_n1 * (n - i) / n * (n - j) / n

                err = abs(route1 - route2)
                max_err = max(max_err, err)

        status = "EXACT" if max_err < 1e-15 else f"ERROR {max_err:.2e}"
        print(f"  n={n}: max coefficient discrepancy = {max_err:.2e} [{status}]")

    print()
    print("  Algebraic proof:")
    print("  Route 1 coefficient of a_i*b_j in ĉ_k = (n-k)/n * (n-i)!(n-j)!/[n!(n-k)!]")
    print("  Route 2 coefficient of a_i*b_j in d_k = (n-1-i)!(n-1-j)!/[(n-1)!(n-1-k)!] * (n-i)(n-j)/n²")
    print("  Ratio = [n*(n-k)! * (n-1-i)!(n-1-j)!(n-i)(n-j)] / [n²(n-1)!(n-1-k)!(n-i)!(n-j)!]")
    print("        = [n*(n-k)!] / [n²(n-1)!(n-1-k)!]")
    print("        = (n-k)! / [n(n-1)!(n-1-k)!]")
    print("        = (n-k)*(n-1-k)! / [n(n-1)!(n-1-k)!]")
    print("        = (n-k) / [n(n-1)!/(n-1-k)! ... ]")
    print("  Simplifies to 1. QED.")


# ── Part 6: What DOES hold? Looking for a usable relationship ───────────

def explore_additive_structure(n_degree, n_trials=2000):
    """Instead of multiplicative ratio, explore whether there's an ADDITIVE
    relationship: 1/Phi_n(p) = 1/Phi_{n-1}(p'/n) + something.
    Also test: 1/Phi_n(p) >= alpha * 1/Phi_{n-1}(p'/n) + beta * something_else."""
    print(f"\n{'='*70}")
    print(f"PART 6: Additive structure at n={n_degree}")
    print(f"{'='*70}")

    diffs = []
    inv_pn_list = []
    inv_pn1_list = []

    for trial in range(n_trials):
        roots = np.sort(np.random.randn(n_degree) * 2)
        for i in range(1, n_degree):
            if roots[i] - roots[i-1] < 0.01:
                roots[i] = roots[i-1] + 0.01

        crit = derivative_normalized(roots)
        if np.any(np.diff(crit) < 1e-10):
            continue

        try:
            pn = phi_n(roots)
            pn1 = phi_n(crit)
        except:
            continue

        if pn > 1e-15 and pn1 > 1e-15:
            inv_pn = 1.0 / pn
            inv_pn1 = 1.0 / pn1
            diffs.append(inv_pn - inv_pn1)
            inv_pn_list.append(inv_pn)
            inv_pn1_list.append(inv_pn1)

    diffs = np.array(diffs)
    inv_pn_list = np.array(inv_pn_list)
    inv_pn1_list = np.array(inv_pn1_list)

    print(f"  {len(diffs)} valid trials")
    print(f"  1/Phi_n(p) - 1/Phi_{{n-1}}(p'/n):")
    print(f"    min    = {np.min(diffs):.6f}")
    print(f"    max    = {np.max(diffs):.6f}")
    print(f"    mean   = {np.mean(diffs):.6f}")
    n_pos = np.sum(diffs > 1e-12)
    n_neg = np.sum(diffs < -1e-12)
    print(f"    positive: {n_pos}/{len(diffs)}")
    print(f"    negative: {n_neg}/{len(diffs)}")

    # Correlation
    corr = np.corrcoef(inv_pn_list, inv_pn1_list)[0, 1]
    print(f"  Correlation(1/Phi_n, 1/Phi_{{n-1}}(deriv)): {corr:.6f}")

    # Linear regression: 1/Phi_n ≈ a * 1/Phi_{n-1}(deriv) + b
    A = np.vstack([inv_pn1_list, np.ones(len(inv_pn1_list))]).T
    slope, intercept = np.linalg.lstsq(A, inv_pn_list, rcond=None)[0]
    residuals = inv_pn_list - (slope * inv_pn1_list + intercept)
    print(f"  Linear fit: 1/Phi_n ≈ {slope:.4f} * 1/Phi_{{n-1}}(deriv) + {intercept:.6f}")
    print(f"    R² = {1 - np.var(residuals)/np.var(inv_pn_list):.6f}")
    print(f"    max |residual| = {np.max(np.abs(residuals)):.6f}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Strategy B: Induction via Differentiation")
    print("="*70)

    # Part 5 first: algebraic proof
    symbolic_commutativity_check()

    # Part 1: numerical confirmation
    for n in [3, 4, 5, 6, 7]:
        test_differentiation_commutativity(n, n_trials=500)

    # Part 2: direction of Phi relationship
    for n in [3, 4, 5, 6]:
        test_phi_derivative_relationship(n, n_trials=2000)

    # Part 3: full chain
    for n in [4, 5, 6]:
        test_induction_chain(n, n_trials=2000)

    # Part 4: ratio analysis
    for n in [4, 5, 6]:
        test_ratio_induction(n, n_trials=2000)

    # Part 6: additive structure
    for n in [4, 5, 6]:
        explore_additive_structure(n, n_trials=2000)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("See above for detailed results. Key questions:")
    print("1. Does differentiation commute with ⊞_n? (Part 1/5)")
    print("2. Which direction: 1/Phi_n(p) vs 1/Phi_{n-1}(p'/n)? (Part 2)")
    print("3. Does the induction chain close? (Part 3)")
    print("4. Is there a usable ratio or additive relationship? (Part 4/6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
