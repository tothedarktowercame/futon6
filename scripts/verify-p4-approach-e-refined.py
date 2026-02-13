#!/usr/bin/env python3
"""Refined tests for Approach E: the semigroup identity p_t ⊞ q_t = (p⊞q)_{2t}.

Key findings from first test:
  - g(t) = 1/Φ((p⊞q)_{2t}) - 1/Φ(p_t) - 1/Φ(q_t) is ALWAYS ≥ 0 (Stam at all t)
  - g(t) is NOT always monotone decreasing (72-82%)
  - g(∞) = 0 is confirmed

This script investigates:
  1. The structure of g near its minimum — where is min(g), how deep?
  2. Normalized surplus g(t)/(1/Φ(p_t) + 1/Φ(q_t)) — is THIS monotone?
  3. The ratio ρ(t) = 1/Φ((p⊞q)_{2t}) / [1/Φ(p_t) + 1/Φ(q_t)] — is ρ monotone → 1?
  4. Log surplus: is log(1/Φ((p⊞q)_{2t})) ≥ log(1/Φ(p_t) + 1/Φ(q_t)) with monotone gap?
  5. The "Stam ratio" R(t) = [1/Φ(p_t) + 1/Φ(q_t)] / 1/Φ(p_t ⊞ q_t) ∈ [0,1] — profile?
"""

import numpy as np
from math import factorial
import sys

np.random.seed(2026)
sys.stdout.reconfigure(line_buffering=True)


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


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol


def hermite_roots(n):
    from numpy.polynomial.hermite_e import hermeroots
    return np.sort(hermeroots([0] * n + [1]))


def hermite_convolve(p_roots, t, n):
    he_r = hermite_roots(n) * np.sqrt(t)
    he_coeffs = np.poly(he_r)[1:]
    p_coeffs = np.poly(p_roots)[1:]
    conv = mss_convolve(p_coeffs, he_coeffs, n)
    if not is_real_rooted(conv):
        return None
    return coeffs_to_roots(conv)


# Fine-grained t values for profile analysis
t_fine = np.concatenate([
    np.linspace(0, 0.1, 20),
    np.linspace(0.1, 1.0, 20),
    np.linspace(1.0, 10.0, 20),
    np.linspace(10.0, 100.0, 15),
])
t_fine = np.unique(t_fine)


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Stam ratio profile R(t) = [1/Φ(p_t) + 1/Φ(q_t)] / 1/Φ((p⊞q)_{2t})
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 1: Stam ratio R(t) = [1/Φ(p_t) + 1/Φ(q_t)] / 1/Φ((p⊞q)_{2t})")
print("R(t) ∈ [0,1] is the Stam inequality. R = 1 means equality.")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")

    R_monotone_count = 0
    R_min_at_start = 0
    total = 0

    for trial in range(30):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue

        pq_roots = coeffs_to_roots(pq_coeffs)

        R_vals = []
        g_vals = []
        valid = True

        for t in t_fine:
            if t == 0:
                inv_pq = 1.0 / phi_n(pq_roots)
                inv_p = 1.0 / phi_n(p)
                inv_q = 1.0 / phi_n(q)
            else:
                r_pq = hermite_convolve(pq_roots, 2 * t, n)
                r_p = hermite_convolve(p, t, n)
                r_q = hermite_convolve(q, t, n)
                if r_pq is None or r_p is None or r_q is None:
                    valid = False
                    break
                inv_pq = 1.0 / phi_n(r_pq)
                inv_p = 1.0 / phi_n(r_p)
                inv_q = 1.0 / phi_n(r_q)

            R = (inv_p + inv_q) / inv_pq
            R_vals.append(R)
            g_vals.append(inv_pq - inv_p - inv_q)

        if not valid or len(R_vals) < len(t_fine):
            continue

        total += 1
        R_arr = np.array(R_vals)
        g_arr = np.array(g_vals)

        # Is R(t) monotone increasing toward 1?
        is_R_monotone = all(R_arr[i] <= R_arr[i + 1] + 1e-10 for i in range(len(R_arr) - 1))
        if is_R_monotone:
            R_monotone_count += 1

        # Is R minimum at t=0?
        if np.argmin(R_arr) == 0 or R_arr[0] <= np.min(R_arr) + 1e-10:
            R_min_at_start += 1

        if trial < 3:
            min_idx = np.argmin(g_arr)
            print(f"    trial {trial}: R(0)={R_arr[0]:.6f}, R(∞)={R_arr[-1]:.6f},"
                  f" min(R)={R_arr.min():.6f} at t={t_fine[np.argmin(R_arr)]:.2f}")
            print(f"      g(0)={g_arr[0]:.6f}, min(g)={g_arr.min():.8f}"
                  f" at t={t_fine[min_idx]:.2f},"
                  f" g monotone={all(g_arr[i] >= g_arr[i+1]-1e-10 for i in range(len(g_arr)-1))}")

    if total > 0:
        print(f"    R(t) monotone increasing: {R_monotone_count}/{total}"
              f" ({100*R_monotone_count/total:.1f}%)")
        print(f"    R minimum at t=0: {R_min_at_start}/{total}"
              f" ({100*R_min_at_start/total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Does h(t) = log(1/Φ((p⊞q)_{2t})) - log(1/Φ(p_t) + 1/Φ(q_t)) behave better?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 2: Log Stam surplus h(t) = log(1/Φ((p⊞q)_{2t})) - log(1/Φ(p_t) + 1/Φ(q_t))")
print("h(t) ≥ 0 iff Stam holds. Is h monotone decreasing?")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    h_monotone = 0
    total = 0

    for trial in range(50):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue

        pq_roots = coeffs_to_roots(pq_coeffs)

        h_vals = []
        valid = True

        for t in t_fine:
            if t == 0:
                inv_pq = 1.0 / phi_n(pq_roots)
                inv_p = 1.0 / phi_n(p)
                inv_q = 1.0 / phi_n(q)
            else:
                r_pq = hermite_convolve(pq_roots, 2 * t, n)
                r_p = hermite_convolve(p, t, n)
                r_q = hermite_convolve(q, t, n)
                if r_pq is None or r_p is None or r_q is None:
                    valid = False
                    break
                inv_pq = 1.0 / phi_n(r_pq)
                inv_p = 1.0 / phi_n(r_p)
                inv_q = 1.0 / phi_n(r_q)

            h = np.log(inv_pq) - np.log(inv_p + inv_q)
            h_vals.append(h)

        if not valid or len(h_vals) < len(t_fine):
            continue

        total += 1
        h_arr = np.array(h_vals)

        is_mono = all(h_arr[i] >= h_arr[i + 1] - 1e-10 for i in range(len(h_arr) - 1))
        if is_mono:
            h_monotone += 1

    if total > 0:
        print(f"    h(t) monotone decreasing: {h_monotone}/{total}"
              f" ({100*h_monotone/total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: The "correct" semigroup comparison
# Since p_t ⊞ q_t = (p⊞q)_{2t}, Stam at time t is:
# 1/Φ(p_t ⊞ q_t) ≥ 1/Φ(p_t) + 1/Φ(q_t)
# ⟺ 1/Φ((p⊞q)_{2t}) ≥ 1/Φ(p_t) + 1/Φ(q_t)
#
# So Stam(p,q) ⟺ g(0) ≥ 0
# And Stam(p_t, q_t) ⟺ g(t) ≥ 0
#
# The "hardest" Stam pair (p_t, q_t) is at the t where g is minimized.
# If we can characterize this t, we reduce to a specific shape.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 3: Where is g(t) minimized? Characterizing the 'hardest' Stam pair.")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    min_t_values = []
    min_g_values = []
    g0_values = []

    for trial in range(100):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue

        pq_roots = coeffs_to_roots(pq_coeffs)

        g_vals = []
        valid = True

        for t in t_fine:
            if t == 0:
                inv_pq = 1.0 / phi_n(pq_roots)
                inv_p = 1.0 / phi_n(p)
                inv_q = 1.0 / phi_n(q)
            else:
                r_pq = hermite_convolve(pq_roots, 2 * t, n)
                r_p = hermite_convolve(p, t, n)
                r_q = hermite_convolve(q, t, n)
                if r_pq is None or r_p is None or r_q is None:
                    valid = False
                    break
                inv_pq = 1.0 / phi_n(r_pq)
                inv_p = 1.0 / phi_n(r_p)
                inv_q = 1.0 / phi_n(r_q)

            g_vals.append(inv_pq - inv_p - inv_q)

        if not valid or len(g_vals) < len(t_fine):
            continue

        g_arr = np.array(g_vals)
        min_idx = np.argmin(g_arr)
        min_t_values.append(t_fine[min_idx])
        min_g_values.append(g_arr.min())
        g0_values.append(g_arr[0])

    if min_t_values:
        mt = np.array(min_t_values)
        mg = np.array(min_g_values)
        g0 = np.array(g0_values)
        print(f"    min(g) achieved at t: mean={mt.mean():.2f}, median={np.median(mt):.2f},"
              f" max={mt.max():.2f}")
        print(f"    min(g) value: mean={mg.mean():.8f}, min={mg.min():.8f}")
        print(f"    g(0) value:   mean={g0.mean():.6f}, min={g0.min():.8f}")
        print(f"    fraction with min at t=0: {np.sum(mt < 0.001)/len(mt):.2f}")
        print(f"    fraction with min at t=∞: {np.sum(mt > 90)/len(mt):.2f}")
        print(f"    ALL min(g) ≥ 0: {np.all(mg >= -1e-10)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Self-convolution special case p ⊞ p
# For identical inputs: g(t) = 1/Φ((p⊞p)_{2t}) - 2/Φ(p_t)
# This is a WEAKER inequality (specific case of Stam with q=p).
# Is g monotone for this case?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 4: Self-convolution p ⊞ p (symmetric case)")
print("g(t) = 1/Φ((p⊞p)_{2t}) - 2/Φ(p_t)")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    mono_count = 0
    total = 0

    for trial in range(50):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        pp_coeffs = mss_convolve(np.poly(p)[1:], np.poly(p)[1:], n)
        if not is_real_rooted(pp_coeffs):
            continue

        pp_roots = coeffs_to_roots(pp_coeffs)

        g_vals = []
        valid = True
        for t in t_fine:
            if t == 0:
                inv_pp = 1.0 / phi_n(pp_roots)
                inv_p = 1.0 / phi_n(p)
            else:
                r_pp = hermite_convolve(pp_roots, 2 * t, n)
                r_p = hermite_convolve(p, t, n)
                if r_pp is None or r_p is None:
                    valid = False
                    break
                inv_pp = 1.0 / phi_n(r_pp)
                inv_p = 1.0 / phi_n(r_p)

            g_vals.append(inv_pp - 2 * inv_p)

        if not valid or len(g_vals) < len(t_fine):
            continue

        total += 1
        g_arr = np.array(g_vals)
        is_mono = all(g_arr[i] >= g_arr[i + 1] - 1e-10 for i in range(len(g_arr) - 1))
        if is_mono:
            mono_count += 1

        if trial < 2:
            print(f"    trial {trial}: g(0)={g_arr[0]:.6f}, min={g_arr.min():.8f},"
                  f" monotone={is_mono}")

    if total > 0:
        print(f"    g monotone: {mono_count}/{total} ({100*mono_count/total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: The "Φ-product" inequality — is Φ(p⊞q) ≤ Φ(p)·Φ(q) ALONG the flow?
# We proved -log Φ is superadditive at t=0. Is it superadditive at all t?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 5: -log Φ superadditivity along the flow")
print("Φ((p⊞q)_{2t}) ≤ Φ(p_t) · Φ(q_t) for all t?")
print("=" * 70)

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    all_pass = 0
    total = 0

    for trial in range(50):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue

        pq_roots = coeffs_to_roots(pq_coeffs)

        violations = 0
        valid = True
        for t in t_fine:
            if t == 0:
                phi_pq = phi_n(pq_roots)
                phi_p = phi_n(p)
                phi_q = phi_n(q)
            else:
                r_pq = hermite_convolve(pq_roots, 2 * t, n)
                r_p = hermite_convolve(p, t, n)
                r_q = hermite_convolve(q, t, n)
                if r_pq is None or r_p is None or r_q is None:
                    valid = False
                    break
                phi_pq = phi_n(r_pq)
                phi_p = phi_n(r_p)
                phi_q = phi_n(r_q)

            if phi_pq > phi_p * phi_q + 1e-8:
                violations += 1

        if not valid:
            continue
        total += 1
        if violations == 0:
            all_pass += 1

    if total > 0:
        print(f"    Φ(conv) ≤ Φ(p)·Φ(q) at all t: {all_pass}/{total}"
              f" ({100*all_pass/total:.1f}%)")


print(f"\n{'='*70}")
print("REFINED TESTS COMPLETE")
print("=" * 70)
