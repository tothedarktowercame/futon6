#!/usr/bin/env python3
"""Attempt to prove h'(t) ≤ 0 for n ≥ 6.

Step 1: Establish structural results:
  - At Hermite zeros: S_k = γ_k, T_{kj} = 1, Φ = n(n-1), Ψ = n(n-1)/2, r = 1/2
  - The condition h'(t)≤0 is: 2r_c ≤ wavg (equality at Hermite)

Step 2: Express h'(t)≤0 in terms of root statistics:
  - r = Ψ/Φ where Ψ = Σ T_{kj}², Φ = Σ S_k²
  - The condition involves comparing three root systems (p, q, p⊞q)

Step 3: Look for the extremal configurations that MAXIMIZE 2r_c/wavg
  - These are the "hardest" cases for the proof
  - Characterize them and see if there's a clean bound

Step 4: Try to prove 2r_c ≤ wavg via SOS or other algebraic technique
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


def score_field(roots):
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
    return S


def phi_n(roots):
    S = score_field(roots)
    return np.sum(S ** 2)


def psi_n(roots):
    S = score_field(roots)
    n = len(roots)
    total = 0.0
    for k in range(n):
        for j in range(k + 1, n):
            total += (S[k] - S[j]) ** 2 / (roots[k] - roots[j]) ** 2
    return total


def r_ratio(roots):
    return psi_n(roots) / phi_n(roots)


def T_matrix(roots):
    """T_{kj} = (S_k - S_j)/(γ_k - γ_j) for k ≠ j."""
    S = score_field(roots)
    n = len(roots)
    T = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if k != j:
                T[k, j] = (S[k] - S[j]) / (roots[k] - roots[j])
    return T


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol


def hermite_roots(n):
    from numpy.polynomial.hermite_e import hermeroots
    return np.sort(hermeroots([0] * n + [1]))


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Structural results at Hermite zeros
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Hermite structure — S_k = γ_k, T_{kj} = 1, r = 1/2")
print("=" * 70)

for n in range(3, 10):
    γ = hermite_roots(n)
    S = score_field(γ)
    T = T_matrix(γ)

    # Check S_k = γ_k
    s_err = np.max(np.abs(S - γ))

    # Check T_{kj} = 1 for k ≠ j
    T_offdiag = T[~np.eye(n, dtype=bool)]
    t_err = np.max(np.abs(T_offdiag - 1.0))

    phi = phi_n(γ)
    psi = psi_n(γ)
    r = psi / phi

    print(f"  n={n}: S_k=γ_k err={s_err:.2e}, T_{n}kj=1 err={t_err:.2e},"
          f" Φ={phi:.4f} (expect {n*(n-1):.0f}),"
          f" Ψ={psi:.4f} (expect {n*(n-1)/2:.0f}), r={r:.6f} (expect 0.5)")


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Understanding deviations from r = 1/2
# For general roots, define δ_k = S_k - γ_k^{He} (deviation from Hermite)
# r = 1/2 at Hermite, what is r for perturbations?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("STEP 2: Distribution of r for random polynomials")
print("r = Ψ/Φ for random real-rooted polynomials")
print("=" * 70)

for n in [3, 4, 5, 6, 7, 8]:
    r_vals = []
    for trial in range(2000):
        roots = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        r_vals.append(r_ratio(roots))
    r = np.array(r_vals)
    print(f"  n={n}: r: min={r.min():.4f}, mean={r.mean():.4f},"
          f" max={r.max():.4f}, std={r.std():.4f}")


# ═══════════════════════════════════════════════════════════════════
# STEP 3: The 2r_c / wavg ratio — extremal analysis
# Find the configurations that MAXIMIZE this ratio
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("STEP 3: Extremal analysis — what maximizes 2r_c / wavg?")
print("=" * 70)

for n in [4, 5, 6, 7]:
    best_ratio = 0
    best_pair = None

    for trial in range(10000):
        # Try various scales to find extremals
        s1 = np.random.uniform(0.1, 5.0)
        s2 = np.random.uniform(0.1, 5.0)
        p = np.sort(np.random.randn(n) * s1)
        q = np.sort(np.random.randn(n) * s2)

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        rc = r_ratio(pq_roots)
        rp = r_ratio(p)
        rq = r_ratio(q)

        alpha = 1.0 / phi_n(p)
        beta = 1.0 / phi_n(q)
        wavg = (rp * alpha + rq * beta) / (alpha + beta)

        ratio = 2 * rc / wavg if wavg > 1e-15 else 0

        if ratio > best_ratio:
            best_ratio = ratio
            phi_p = phi_n(p)
            phi_q = phi_n(q)
            phi_c = phi_n(pq_roots)
            best_pair = {
                'p': p.copy(), 'q': q.copy(), 'ratio': ratio,
                'r_p': rp, 'r_q': rq, 'r_c': rc,
                'phi_p': phi_p, 'phi_q': phi_q, 'phi_c': phi_c,
                'phi_ratio': max(phi_p, phi_q) / min(phi_p, phi_q),
                'scale_ratio': max(np.std(p), np.std(q)) / min(np.std(p), np.std(q)),
            }

    print(f"\n  n={n}: worst 2r_c/wavg = {best_ratio:.6f}")
    if best_pair:
        bp = best_pair
        print(f"    p = {bp['p']}")
        print(f"    q = {bp['q']}")
        print(f"    r(p)={bp['r_p']:.4f}, r(q)={bp['r_q']:.4f}, r(p⊞q)={bp['r_c']:.4f}")
        print(f"    Φ(p)={bp['phi_p']:.4f}, Φ(q)={bp['phi_q']:.4f}, Φ(p⊞q)={bp['phi_c']:.4f}")
        print(f"    Φ ratio (max/min) = {bp['phi_ratio']:.2f}")
        print(f"    scale ratio = {bp['scale_ratio']:.2f}")

        # Check: is the violation associated with one polynomial having
        # much higher Φ (more clustered roots)?
        if bp['phi_ratio'] > 10:
            print(f"    *** HIGH Φ RATIO — one polynomial has much more clustered roots")
        if bp['scale_ratio'] > 5:
            print(f"    *** HIGH SCALE RATIO — root spreads differ significantly")


# ═══════════════════════════════════════════════════════════════════
# STEP 4: Test simpler sufficient conditions
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("STEP 4: Testing simpler sufficient conditions for n≥6")
print("=" * 70)

# Condition A: r(p⊞q) ≤ max(r(p), r(q))/2
# Condition B: r(p⊞q) ≤ (r(p) + r(q))/4
# Condition C: r(p⊞q) ≤ min(r(p), r(q))
# Condition D: Ψ(p⊞q)/Φ(p⊞q) ≤ 1/2  (r ≤ 1/2 always?)

for n in [3, 4, 5, 6, 7, 8]:
    r_leq_half = 0
    condA = 0
    condB = 0
    condC = 0
    total = 0
    total_conv = 0
    r_half_viol = 0

    for trial in range(3000):
        roots = np.sort(np.random.randn(n) * np.random.uniform(0.3, 5.0))
        r = r_ratio(roots)
        total += 1
        if r <= 0.5 + 1e-10:
            r_leq_half += 1

    for trial in range(3000):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.3, 5.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.3, 5.0))

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        rc = r_ratio(pq_roots)
        rp = r_ratio(p)
        rq = r_ratio(q)
        total_conv += 1

        if rc <= max(rp, rq) / 2 + 1e-10:
            condA += 1
        if rc <= (rp + rq) / 4 + 1e-10:
            condB += 1
        if rc <= min(rp, rq) + 1e-10:
            condC += 1
        if rc > 0.5 + 1e-10:
            r_half_viol += 1

    print(f"  n={n}:")
    print(f"    r ≤ 1/2 (single poly): {r_leq_half}/{total}"
          f" ({100*r_leq_half/total:.1f}%)")
    if total_conv > 0:
        print(f"    r(p⊞q) ≤ 1/2: {total_conv-r_half_viol}/{total_conv}"
              f" ({100*(total_conv-r_half_viol)/total_conv:.1f}%)")
        print(f"    A: r_c ≤ max(r_p,r_q)/2: {condA}/{total_conv}"
              f" ({100*condA/total_conv:.1f}%)")
        print(f"    B: r_c ≤ (r_p+r_q)/4: {condB}/{total_conv}"
              f" ({100*condB/total_conv:.1f}%)")
        print(f"    C: r_c ≤ min(r_p,r_q): {condC}/{total_conv}"
              f" ({100*condC/total_conv:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: Is r ≤ 1/2 universal? (Would give h'≤0 immediately!)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("STEP 5: Is r ≤ 1/2 for all root configs? (r=1/2 at Hermite)")
print("Focus on configurations with r > 1/2")
print("=" * 70)

for n in [3, 4, 5, 6, 7, 8, 9, 10]:
    max_r = 0
    max_roots = None
    total = 0
    above_half = 0

    for trial in range(5000):
        roots = np.sort(np.random.randn(n) * np.random.uniform(0.1, 10.0))
        r = r_ratio(roots)
        total += 1
        if r > 0.5 + 1e-10:
            above_half += 1
        if r > max_r:
            max_r = r
            max_roots = roots.copy()

    print(f"  n={n}: max(r) = {max_r:.6f}, r > 1/2 in {above_half}/{total}"
          f" ({100*above_half/total:.1f}%)")
    if max_r > 0.5 + 0.01 and n <= 6:
        print(f"    max-r roots: {max_roots}")
        # Characterize: gaps
        gaps = np.diff(max_roots)
        print(f"    gaps: {gaps}")
        gap_ratio = gaps.max() / gaps.min()
        print(f"    gap ratio (max/min): {gap_ratio:.2f}")


print(f"\n{'='*70}")
print("STEP 5b: Specifically adversarial configs — clustered + outlier")
print("=" * 70)

for n in [3, 4, 5, 6, 7, 8]:
    max_r = 0
    # Try configurations with one root far away (outlier)
    for trial in range(2000):
        # n-1 roots clustered near 0, one far away
        cluster_scale = np.random.uniform(0.01, 0.5)
        outlier = np.random.uniform(3, 20) * np.random.choice([-1, 1])
        cluster = np.sort(np.random.randn(n - 1) * cluster_scale)
        roots = np.sort(np.append(cluster, outlier))
        r = r_ratio(roots)
        max_r = max(max_r, r)

    # Try configs with two tight clusters
    for trial in range(2000):
        n1 = n // 2
        n2 = n - n1
        c1 = np.sort(np.random.randn(n1) * 0.1)
        c2 = np.sort(np.random.randn(n2) * 0.1 + 5.0)
        roots = np.sort(np.concatenate([c1, c2]))
        r = r_ratio(roots)
        max_r = max(max_r, r)

    print(f"  n={n}: max(r) with adversarial configs = {max_r:.6f}"
          f" {'> 1/2 ✗' if max_r > 0.5 + 1e-3 else '≤ 1/2 ✓'}")


print(f"\n{'='*70}")
print("DONE")
print("=" * 70)
