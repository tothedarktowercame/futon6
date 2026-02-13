#!/usr/bin/env python3
"""Numerical tests for the six new proof approaches (Problem 4).

Priority tests:
  E: Semigroup monotonicity — is g(t) = 1/Φ((p⊞q)_{2t}) - 1/Φ(p_t) - 1/Φ(q_t) decreasing?
  H: Elementary symmetric multiplicativity — is e_k(p⊞q) = e_k(p)·e_k(q)/C(n,k)?
  F: Resolvent score decomposition — is ⟨S^A, S^B⟩ ≤ 0 at roots of p⊞q?
  G: Gårding concavity — is 1/Φ midpoint-concave ON the real-rooted cone?
"""

import numpy as np
from math import factorial, comb
import sys

np.random.seed(2026)
sys.stdout.reconfigure(line_buffering=True)


# ── Core functions ──────────────────────────────────────────────────

def mss_convolve(a_coeffs, b_coeffs, n):
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


def phi_n(roots):
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def score_field(roots):
    """S_k = sum_{j!=k} 1/(roots[k] - roots[j])"""
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
    return S


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol


def hermite_roots(n):
    """Zeros of the probabilist's Hermite polynomial He_n."""
    from numpy.polynomial.hermite_e import hermeroots
    return np.sort(hermeroots([0] * n + [1]))


def hermite_convolve(p_roots, t, n):
    """Compute p ⊞_n He_t where He_t has roots sqrt(t) * (Hermite zeros)."""
    he_r = hermite_roots(n) * np.sqrt(t)
    he_coeffs = np.poly(he_r)[1:]
    p_coeffs = np.poly(p_roots)[1:]
    conv = mss_convolve(p_coeffs, he_coeffs, n)
    if not is_real_rooted(conv):
        return None
    return coeffs_to_roots(conv)


# ══════════════════════════════════════════════════════════════════════
# APPROACH E: Semigroup Monotonicity
# g(t) = 1/Φ((p⊞q)_{2t}) - 1/Φ(p_t) - 1/Φ(q_t)
# Question: is g(t) monotone decreasing? If so, g(0) ≥ g(∞) = 0 ⟹ Stam.
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("APPROACH E: Semigroup Monotonicity — g(t) test")
print("g(t) = 1/Φ((p⊞q)_{2t}) - 1/Φ(p_t) - 1/Φ(q_t)")
print("Need: g monotone decreasing, g(∞) = 0")
print("=" * 70)

t_values = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    monotone_count = 0
    total_count = 0
    always_nonneg = 0
    total_valid = 0

    for trial in range(50):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        # Compute p⊞q
        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        g_vals = []
        valid = True
        for t in t_values:
            if t == 0:
                inv_phi_pq = 1.0 / phi_n(pq_roots)
                inv_phi_p = 1.0 / phi_n(p)
                inv_phi_q = 1.0 / phi_n(q)
            else:
                # (p⊞q)_{2t}
                r_pq = hermite_convolve(pq_roots, 2 * t, n)
                if r_pq is None:
                    valid = False
                    break
                inv_phi_pq = 1.0 / phi_n(r_pq)

                # p_t
                r_p = hermite_convolve(p, t, n)
                if r_p is None:
                    valid = False
                    break
                inv_phi_p = 1.0 / phi_n(r_p)

                # q_t
                r_q = hermite_convolve(q, t, n)
                if r_q is None:
                    valid = False
                    break
                inv_phi_q = 1.0 / phi_n(r_q)

            g_vals.append(inv_phi_pq - inv_phi_p - inv_phi_q)

        if not valid or len(g_vals) < len(t_values):
            continue

        total_valid += 1
        g_arr = np.array(g_vals)

        # Check monotone decreasing
        is_monotone = all(g_arr[i] >= g_arr[i + 1] - 1e-10 for i in range(len(g_arr) - 1))
        if is_monotone:
            monotone_count += 1
        total_count += 1

        # Check always nonneg
        if np.min(g_arr) >= -1e-10:
            always_nonneg += 1

        if trial < 3:
            print(f"    trial {trial}: g = [", ", ".join(f"{v:.6f}" for v in g_arr[:6]), "...]")
            print(f"      monotone={is_monotone}, min={g_arr.min():.8f}, g(0)={g_arr[0]:.6f}")

    if total_count > 0:
        print(f"    RESULTS: monotone decreasing: {monotone_count}/{total_count}"
              f" ({100*monotone_count/total_count:.1f}%)")
        print(f"    always nonneg (Stam): {always_nonneg}/{total_count}"
              f" ({100*always_nonneg/total_count:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# APPROACH H: Elementary Symmetric Multiplicativity
# Test: e_k(p⊞q) = e_k(p) · e_k(q) / C(n,k)
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("APPROACH H: Elementary Symmetric Multiplicativity")
print("Test: e_k(p⊞q) = e_k(p) · e_k(q) / C(n,k) ?")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    for trial in range(5):
        p = np.sort(np.random.randn(n))
        q = np.sort(np.random.randn(n))

        # e_k = (-1)^k * a_k (coefficient of x^{n-k})
        p_coeffs = np.poly(p)  # [1, a1, a2, ..., an] (full polynomial)
        q_coeffs = np.poly(q)
        e_p = [(-1)**k * p_coeffs[k] for k in range(n + 1)]  # e_0=1, e_1, ..., e_n
        e_q = [(-1)**k * q_coeffs[k] for k in range(n + 1)]

        conv = mss_convolve(p_coeffs[1:], q_coeffs[1:], n)
        pq_coeffs_full = np.concatenate([[1.0], conv])
        e_pq = [(-1)**k * pq_coeffs_full[k] for k in range(n + 1)]

        # Check: e_k(p⊞q) vs e_k(p)*e_k(q)/C(n,k)
        if trial == 0:
            print(f"    k  | e_k(p⊞q) | e_k(p)*e_k(q)/C(n,k) | ratio")
            print(f"    ---+----------+----------------------+------")
        for k in range(1, n + 1):
            predicted = e_p[k] * e_q[k] / comb(n, k)
            actual = e_pq[k]
            ratio = actual / predicted if abs(predicted) > 1e-15 else float('nan')
            if trial == 0:
                print(f"    {k}  | {actual:10.6f} | {predicted:10.6f}            | {ratio:.6f}")

    # Large-scale test
    max_err = 0
    for trial in range(200):
        p = np.sort(np.random.randn(n) * 2)
        q = np.sort(np.random.randn(n) * 2)
        p_c = np.poly(p)
        q_c = np.poly(q)
        e_p = [(-1)**k * p_c[k] for k in range(n + 1)]
        e_q = [(-1)**k * q_c[k] for k in range(n + 1)]
        conv = mss_convolve(p_c[1:], q_c[1:], n)
        pq_full = np.concatenate([[1.0], conv])
        e_pq = [(-1)**k * pq_full[k] for k in range(n + 1)]

        for k in range(1, n + 1):
            predicted = e_p[k] * e_q[k] / comb(n, k)
            actual = e_pq[k]
            if abs(predicted) > 1e-10:
                err = abs(actual - predicted) / abs(predicted)
                max_err = max(max_err, err)

    print(f"    Max relative error (multiplicative model): {max_err:.6e}")
    if max_err < 1e-8:
        print(f"    *** MULTIPLICATIVE IDENTITY HOLDS! ***")
    else:
        print(f"    Multiplicative model FAILS (cross-terms exist)")


# ══════════════════════════════════════════════════════════════════════
# APPROACH F: Resolvent Score Decomposition
# S^A_k = Σ_j 1/(γ_k - λ_j),  S^B_k = Σ_j 1/(γ_k - μ_j)
# Test: ⟨S^A, S^B⟩ ≤ 0?  S^A + S^B ≈ S?
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("APPROACH F: Resolvent Score Decomposition")
print("S^A_k = Σ_j 1/(γ_k - λ_j),  S^B_k = Σ_j 1/(γ_k - μ_j)")
print("Test: S = S^A + S^B?  ⟨S^A, S^B⟩ ≤ 0?")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    decomp_works = 0
    inner_neg = 0
    total = 0

    inner_products = []
    decomp_errors = []

    for trial in range(200):
        p = np.sort(np.random.randn(n) * 2)
        q = np.sort(np.random.randn(n) * 2)

        conv = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(conv):
            continue

        gamma = coeffs_to_roots(conv)
        S = score_field(gamma)

        # A-resolvent score
        S_A = np.array([sum(1.0 / (gamma[k] - p[j]) for j in range(n)) for k in range(n)])
        # B-resolvent score
        S_B = np.array([sum(1.0 / (gamma[k] - q[j]) for j in range(n)) for k in range(n)])

        total += 1

        # Does S = S^A + S^B?
        err = np.max(np.abs(S - (S_A + S_B))) / (np.max(np.abs(S)) + 1e-15)
        decomp_errors.append(err)
        if err < 0.01:
            decomp_works += 1

        # Inner product
        ip = np.dot(S_A, S_B)
        inner_products.append(ip)
        if ip <= 1e-10:
            inner_neg += 1

        if trial < 3:
            print(f"    trial {trial}: S = {S[:3]}")
            print(f"      S^A+S^B = {(S_A+S_B)[:3]}, err={err:.6e}")
            print(f"      ⟨S^A,S^B⟩ = {ip:.6f}, ‖S^A‖²={np.dot(S_A,S_A):.4f},"
                  f" ‖S^B‖²={np.dot(S_B,S_B):.4f}, Φ(p)={phi_n(p):.4f}, Φ(q)={phi_n(q):.4f}")

    if total > 0:
        ips = np.array(inner_products)
        errs = np.array(decomp_errors)
        print(f"    S = S^A + S^B: {decomp_works}/{total} ({100*decomp_works/total:.1f}% exact)")
        print(f"      decomp error: mean={errs.mean():.4e}, max={errs.max():.4e}")
        print(f"    ⟨S^A,S^B⟩ ≤ 0: {inner_neg}/{total} ({100*inner_neg/total:.1f}%)")
        print(f"      inner product: min={ips.min():.4f}, mean={ips.mean():.4f}, max={ips.max():.4f}")


# ══════════════════════════════════════════════════════════════════════
# APPROACH G: Gårding Concavity (midpoint test ON the cone)
# Test: 1/Φ((κ+λ)/2) ≥ (1/Φ(κ) + 1/Φ(λ))/2
#       ONLY when (κ+λ)/2 ∈ real-rooted cone
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("APPROACH G: Midpoint Concavity of 1/Φ ON the Real-Rooted Cone")
print("(Only testing midpoints that remain real-rooted)")
print("=" * 70)

for n in [3, 4, 5]:
    violations = 0
    total = 0

    for trial in range(1000):
        # Generate centered polynomials with moderate spread
        p = np.sort(np.random.randn(n) * 1.5)
        p -= np.mean(p)
        q = np.sort(np.random.randn(n) * 1.5)
        q -= np.mean(q)

        p_coeffs = np.poly(p)[1:]
        q_coeffs = np.poly(q)[1:]

        # Midpoint in COEFFICIENT space (not cumulant — but for centered, close enough)
        mid_coeffs = 0.5 * (p_coeffs + q_coeffs)

        # CRUCIAL: only test if midpoint is real-rooted
        if not is_real_rooted(mid_coeffs):
            continue

        mid_roots = coeffs_to_roots(mid_coeffs)

        try:
            F_mid = 1.0 / phi_n(mid_roots)
            F_p = 1.0 / phi_n(p)
            F_q = 1.0 / phi_n(q)
        except (ValueError, ZeroDivisionError):
            continue

        total += 1
        if F_mid < 0.5 * (F_p + F_q) - 1e-10:
            violations += 1

    if total > 0:
        print(f"  n={n}: {violations}/{total} midpoint violations"
              f" ({100*violations/total:.1f}%)")
        if violations == 0:
            print(f"    *** 1/Φ_n appears CONCAVE on the real-rooted cone! ***")


# ══════════════════════════════════════════════════════════════════════
# APPROACH I: Differentiation tower — ratio Φ_{n-1}(p'/n) / Φ_n(p)
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("APPROACH I: Differentiation tower — Φ_{n-1}(p'/n) / Φ_n(p)")
print("=" * 70)

for n in [3, 4, 5, 6]:
    ratios = []
    for trial in range(200):
        p = np.sort(np.random.randn(n) * 2)

        # Derivative roots
        dp_coeffs = np.polyder(np.poly(p)) / n  # p'/n
        dp_roots = np.sort(np.roots(dp_coeffs).real)

        # Check real-rooted (derivative of real-rooted is always real-rooted)
        if np.max(np.abs(np.roots(dp_coeffs).imag)) > 1e-6:
            continue

        phi_p = phi_n(p)
        phi_dp = phi_n(dp_roots)  # This is Φ_{n-1} evaluated on p'/n roots

        ratios.append(phi_dp / phi_p)

    r = np.array(ratios)
    print(f"  n={n}: Φ_{n-1}(p'/n) / Φ_n(p) ="
          f" mean={r.mean():.6f}, std={r.std():.4f},"
          f" min={r.min():.4f}, max={r.max():.4f}")
    # Check: is this close to a simple function of n?
    for guess_name, guess_val in [("(n-1)/n", (n-1)/n), ("n/(n+1)", n/(n+1)),
                                   ("(n-1)²/n²", (n-1)**2/n**2),
                                   ("(n-2)/(n-1)", (n-2)/(n-1) if n > 2 else 0)]:
        if abs(r.mean() - guess_val) < 0.05:
            print(f"    ≈ {guess_name} = {guess_val:.6f} (off by {abs(r.mean()-guess_val):.6f})")


print(f"\n{'='*70}")
print("ALL TESTS COMPLETE")
print("=" * 70)
