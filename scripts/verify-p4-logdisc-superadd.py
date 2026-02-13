#!/usr/bin/env python3
"""Deep investigation of log-discriminant superadditivity.

H'_n(p ⊞_n q) ≥ H'_n(p) + H'_n(q)

where H'_n = Σ_{i<j} log|γ_i - γ_j|.

This is the last missing condition for the Stam inequality via de Bruijn.

Key questions:
1. How does the surplus H'(p⊞q) - H'(p) - H'(q) relate to known quantities?
2. Does the random matrix interpretation help? p⊞q = E_U[char(A+UBU*)]
3. Does the surplus have a nice form in cumulant coordinates?
4. What makes n=3 special (3% violations)?
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


def log_disc(roots):
    """H' = Σ_{i<j} log|γ_i - γ_j|"""
    n = len(roots)
    H = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(roots[i] - roots[j])
            if gap < 1e-15:
                return -np.inf
            H += np.log(gap)
    return H


def phi_n(roots):
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def coeffs_to_roots(coeffs):
    poly = np.concatenate([[1.0], coeffs])
    return np.sort(np.roots(poly).real)


def is_real_rooted(coeffs, tol=1e-6):
    poly = np.concatenate([[1.0], coeffs])
    r = np.roots(poly)
    return np.max(np.abs(r.imag)) < tol


# ── Test 1: Large-scale superadditivity test ─────────────────────────

print("=" * 70)
print("TEST 1: Log-disc superadditivity — large-scale statistical test")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    violations = 0
    total = 0
    surpluses = []
    violation_surpluses = []

    for trial in range(500):
        p_roots = np.sort(np.random.randn(n) * 2)
        q_roots = np.sort(np.random.randn(n) * 2)
        p_coeffs = np.poly(p_roots)[1:]
        q_coeffs = np.poly(q_roots)[1:]

        conv = mss_convolve(p_coeffs, q_coeffs, n)
        if not is_real_rooted(conv):
            continue

        pq_roots = coeffs_to_roots(conv)

        H_pq = log_disc(pq_roots)
        H_p = log_disc(p_roots)
        H_q = log_disc(q_roots)

        if any(np.isinf([H_pq, H_p, H_q])):
            continue

        surplus = H_pq - H_p - H_q
        surpluses.append(surplus)
        total += 1

        if surplus < -1e-8:
            violations += 1
            violation_surpluses.append(surplus)

    surpluses = np.array(surpluses)
    print(f"\n  n={n}: {violations}/{total} violations "
          f"({100*violations/total:.1f}%)")
    print(f"    surplus: min={surpluses.min():.6f}, "
          f"mean={surpluses.mean():.6f}, max={surpluses.max():.6f}")
    if violation_surpluses:
        print(f"    violation magnitudes: {sorted(violation_surpluses)[:5]}")


# ── Test 2: Structure of violations at n=3 ───────────────────────────

print(f"\n{'='*70}")
print("TEST 2: Anatomy of n=3 violations")
print(f"{'='*70}")

n = 3
violations_data = []
for trial in range(2000):
    p_roots = np.sort(np.random.randn(n) * 2)
    q_roots = np.sort(np.random.randn(n) * 2)
    p_coeffs = np.poly(p_roots)[1:]
    q_coeffs = np.poly(q_roots)[1:]

    conv = mss_convolve(p_coeffs, q_coeffs, n)
    if not is_real_rooted(conv):
        continue

    pq_roots = coeffs_to_roots(conv)
    H_pq = log_disc(pq_roots)
    H_p = log_disc(p_roots)
    H_q = log_disc(q_roots)

    if any(np.isinf([H_pq, H_p, H_q])):
        continue

    surplus = H_pq - H_p - H_q

    if surplus < -1e-8:
        # Analyze this violation
        p_spread = np.max(p_roots) - np.min(p_roots)
        q_spread = np.max(q_roots) - np.min(q_roots)
        pq_spread = np.max(pq_roots) - np.min(pq_roots)
        p_var = np.var(p_roots)
        q_var = np.var(q_roots)
        p_phi = phi_n(p_roots)
        q_phi = phi_n(q_roots)
        pq_phi = phi_n(pq_roots)

        violations_data.append({
            'surplus': surplus,
            'p_spread': p_spread, 'q_spread': q_spread,
            'pq_spread': pq_spread,
            'p_var': p_var, 'q_var': q_var,
            'p_phi': p_phi, 'q_phi': q_phi, 'pq_phi': pq_phi,
            'H_p': H_p, 'H_q': H_q, 'H_pq': H_pq,
        })

print(f"  Found {len(violations_data)} violations in 2000 trials")
if violations_data:
    print(f"\n  Violation characteristics:")
    surp = [v['surplus'] for v in violations_data]
    print(f"    surplus range: [{min(surp):.6f}, {max(surp):.6f}]")

    # What characterizes violations?
    for key in ['p_spread', 'q_spread', 'p_var', 'q_var', 'H_p', 'H_q']:
        vals = [v[key] for v in violations_data]
        print(f"    {key}: mean={np.mean(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # Key insight: are violations associated with one polynomial having very small H'?
    for v in sorted(violations_data, key=lambda x: x['surplus'])[:5]:
        print(f"\n    Worst violation: surplus={v['surplus']:.6f}")
        print(f"      H'(p)={v['H_p']:.4f}, H'(q)={v['H_q']:.4f}, H'(p⊞q)={v['H_pq']:.4f}")
        print(f"      Φ(p)={v['p_phi']:.4f}, Φ(q)={v['q_phi']:.4f}, Φ(p⊞q)={v['pq_phi']:.4f}")
        print(f"      spread(p)={v['p_spread']:.4f}, spread(q)={v['q_spread']:.4f}")


# ── Test 3: Random matrix interpretation ─────────────────────────────

print(f"\n{'='*70}")
print("TEST 3: Random matrix interpretation — sample vs expected log-disc")
print(f"{'='*70}")
print("  H'(E_U[char(A+UBU*)]) vs E_U[H'(char(A+UBU*))]")
print("  Jensen's inequality direction depends on convexity of H'")

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    n_samples = 500

    for trial in range(5):
        p_roots = np.sort(np.random.randn(n) * 2)
        q_roots = np.sort(np.random.randn(n) * 2)
        A = np.diag(p_roots)
        B = np.diag(q_roots)

        # Sample H'(char(A + UBU*)) for random U
        sample_H = []
        for _ in range(n_samples):
            # Random unitary from Haar measure
            Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
            Q, R = np.linalg.qr(Z)
            Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))

            M = A + Q @ B @ Q.conj().T
            eigs = np.sort(np.linalg.eigvalsh(M))
            H = log_disc(eigs)
            if not np.isinf(H):
                sample_H.append(H)

        # Expected log-disc of the MSS convolution
        p_coeffs = np.poly(p_roots)[1:]
        q_coeffs = np.poly(q_roots)[1:]
        conv = mss_convolve(p_coeffs, q_coeffs, n)
        if not is_real_rooted(conv):
            continue
        pq_roots = coeffs_to_roots(conv)
        H_pq = log_disc(pq_roots)
        H_p = log_disc(p_roots)
        H_q = log_disc(q_roots)

        if sample_H:
            E_H = np.mean(sample_H)
            print(f"    trial {trial}: E[H'(A+UBU*)]={E_H:.4f}, "
                  f"H'(p⊞q)={H_pq:.4f}, H'(p)+H'(q)={H_p+H_q:.4f}, "
                  f"Jensen direction: E[H'] {'>' if E_H > H_pq else '<'} H'(E[char])")


# ── Test 4: Is log-disc concave/convex in cumulant coordinates? ──────

print(f"\n{'='*70}")
print("TEST 4: Log-disc behavior in cumulant coordinates")
print(f"{'='*70}")
print("  If H'_n is concave in cumulant coordinates, superadditivity follows")
print("  since ⊞_n = addition in cumulant space")

for n in [3, 4, 5]:
    print(f"\n  n={n}: midpoint concavity test (κ-space)")

    # For centered polynomials, κ_2 = a_2 = -variance, κ_k = a_k for k≤3 at n=3
    # For general n, the moment-cumulant relation is more complex
    # But for centered polys at n=3: κ_2 = a_2, κ_3 = a_3, and ⊞_3 = addition
    # So H'_3(p ⊞ q) = H'_3(κ_p + κ_q) and we need concavity of H'_3 in κ-space

    concave_violations = 0
    total_tests = 0

    for trial in range(500):
        # Generate centered polynomials
        p_roots = np.sort(np.random.randn(n) * 1.5)
        p_roots -= np.mean(p_roots)  # center
        q_roots = np.sort(np.random.randn(n) * 1.5)
        q_roots -= np.mean(q_roots)

        p_coeffs = np.poly(p_roots)[1:]
        q_coeffs = np.poly(q_roots)[1:]

        # Midpoint in coefficient space (which = cumulant space for centered)
        mid_coeffs = 0.5 * (p_coeffs + q_coeffs)

        # But midpoint of centered polynomials may not be real-rooted!
        if not is_real_rooted(mid_coeffs):
            continue

        # MSS convolution (for centered n=3, this is coefficient addition)
        conv = mss_convolve(p_coeffs, q_coeffs, n)
        if not is_real_rooted(conv):
            continue

        mid_roots = coeffs_to_roots(mid_coeffs)
        H_mid = log_disc(mid_roots)
        H_p = log_disc(p_roots)
        H_q = log_disc(q_roots)

        if any(np.isinf([H_mid, H_p, H_q])):
            continue

        total_tests += 1
        # Concavity: H'(midpoint) ≥ (H'(p) + H'(q))/2
        if H_mid < 0.5 * (H_p + H_q) - 1e-8:
            concave_violations += 1

    print(f"    Midpoint concavity violations: {concave_violations}/{total_tests} "
          f"({100*concave_violations/total_tests:.1f}%)")
    if concave_violations == 0:
        print(f"    *** H'_n appears CONCAVE in cumulant coordinates! ***")
    else:
        print(f"    H'_n is NOT concave in cumulant coordinates")


# ── Test 5: Normalized log-disc and scaling behavior ─────────────────

print(f"\n{'='*70}")
print("TEST 5: Normalized H'_n behavior")
print(f"{'='*70}")

# The normalized log-disc is H_n = (2/(n(n-1))) H'_n
# In the free probability limit, H_n → χ (Voiculescu's free entropy)
# and superadditivity of H'_n is equivalent to superadditivity of H_n

# Key question: does the normalized version have better properties?

for n in [3, 4, 5, 6]:
    print(f"\n  n={n}:")
    norm = 2.0 / (n * (n-1))

    surpluses_raw = []
    surpluses_norm = []

    for trial in range(500):
        p_roots = np.sort(np.random.randn(n) * 2)
        q_roots = np.sort(np.random.randn(n) * 2)
        p_coeffs = np.poly(p_roots)[1:]
        q_coeffs = np.poly(q_roots)[1:]

        conv = mss_convolve(p_coeffs, q_coeffs, n)
        if not is_real_rooted(conv):
            continue

        pq_roots = coeffs_to_roots(conv)
        H_pq = log_disc(pq_roots)
        H_p = log_disc(p_roots)
        H_q = log_disc(q_roots)

        if any(np.isinf([H_pq, H_p, H_q])):
            continue

        surpluses_raw.append(H_pq - H_p - H_q)
        surpluses_norm.append(norm * (H_pq - H_p - H_q))

    sarr = np.array(surpluses_raw)
    n_viol = np.sum(sarr < -1e-8)
    print(f"    violations: {n_viol}/{len(sarr)}")
    print(f"    raw surplus: min={sarr.min():.6f}, mean={sarr.mean():.6f}")
    narr = np.array(surpluses_norm)
    print(f"    normalized surplus: min={narr.min():.6f}, mean={narr.mean():.6f}")


if __name__ == '__main__':
    pass  # all tests run at import time
