#!/usr/bin/env python3
"""Explore score decomposition under MSS convolution.

GOAL: Find how S_k(p⊞q) relates to S(p) and S(q).

Key identity: S_k = p''(γ_k) / (2·p'(γ_k))
This connects score to the polynomial directly.

The MSS convolution operates on coefficients. If we can express
S(p⊞q) as a function of S(p), S(q) and root data, we might
get a Cauchy-Schwarz proof of Stam (like the continuous case).

This script works with EXPLICIT small examples to see structure,
not random mass testing.
"""

import numpy as np
from math import factorial
import sys

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


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol


def convolve_roots(p_roots, q_roots):
    n = len(p_roots)
    p_coeffs = np.poly(p_roots)[1:]
    q_coeffs = np.poly(q_roots)[1:]
    c_coeffs = mss_convolve(p_coeffs, q_coeffs, n)
    if not is_real_rooted(c_coeffs):
        return None
    return coeffs_to_roots(c_coeffs)


# ═══════════════════════════════════════════════════════════════════
# PART 1: Verify S_k = p''(γ_k) / (2·p'(γ_k))
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Verify S_k = p''(γ_k) / (2·p'(γ_k))")
print("=" * 70)

for roots in [np.array([0.0, 1.0, 3.0]),
              np.array([-2.0, -1.0, 1.0, 4.0]),
              np.array([-3.0, -1.0, 0.0, 2.0, 5.0])]:
    n = len(roots)
    S = score_field(roots)

    # Compute p'(γ_k) and p''(γ_k) via numpy
    p_poly = np.poly(roots)  # [1, -e1, e2, ..., (-1)^n e_n]
    p_deriv = np.polyder(p_poly)
    p_deriv2 = np.polyder(p_deriv)

    S_from_poly = np.array([
        np.polyval(p_deriv2, r) / (2 * np.polyval(p_deriv, r))
        for r in roots
    ])

    err = np.max(np.abs(S - S_from_poly))
    print(f"  n={n}: S direct vs p''/2p': max error = {err:.2e}")


# ═══════════════════════════════════════════════════════════════════
# PART 2: Explicit n=3 examples — what IS S(p⊞q)?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 2: Explicit n=3 score decomposition")
print("For each (p, q), compute S(c) and compare to various functions of S(p), S(q)")
print("=" * 70)

examples_3 = [
    (np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), "equal"),
    (np.array([-1.0, 0.0, 1.0]), np.array([-2.0, 0.0, 2.0]), "scaled 2x"),
    (np.array([-1.0, 0.0, 1.0]), np.array([-3.0, 0.0, 3.0]), "scaled 3x"),
    (np.array([-1.0, 0.0, 1.0]), np.array([0.0, 1.0, 2.0]), "shifted"),
    (np.array([-2.0, -1.0, 3.0]), np.array([-1.0, 1.0, 4.0]), "asymmetric"),
    (np.array([-1.0, 0.0, 1.0]), np.array([-0.1, 0.0, 0.1]), "clustered"),
]

for p, q, label in examples_3:
    c = convolve_roots(p, q)
    if c is None:
        print(f"  {label}: not real-rooted")
        continue

    Sp = score_field(p)
    Sq = score_field(q)
    Sc = score_field(c)

    Phi_p = phi_n(p)
    Phi_q = phi_n(q)
    Phi_c = phi_n(c)

    # Various candidate relationships
    alpha = 1 / Phi_p
    beta = 1 / Phi_q
    w_p = alpha / (alpha + beta)
    w_q = beta / (alpha + beta)

    print(f"\n  --- {label} ---")
    print(f"  p = {p},  q = {q},  c = p⊞q = {c}")
    print(f"  S(p) = {Sp}")
    print(f"  S(q) = {Sq}")
    print(f"  S(c) = {Sc}")
    print(f"  Φ(p) = {Phi_p:.4f},  Φ(q) = {Phi_q:.4f},  Φ(c) = {Phi_c:.4f}")
    print(f"  w_p = {w_p:.4f},  w_q = {w_q:.4f}  (Φ-weighted)")

    # Test: is S(c) a weighted average of S(p) and S(q)?
    # Problem: S vectors have different lengths and correspond to different roots
    # S(c) has 3 entries at c's roots, S(p) has 3 entries at p's roots.
    # There's no natural 1-1 correspondence.

    # But we CAN compare norms: ||S(c)||² = Φ(c)
    print(f"  ||S(c)||² = {np.sum(Sc**2):.4f}  (= Φ(c))")
    print(f"  ||S(p)||² = {np.sum(Sp**2):.4f}  (= Φ(p))")
    print(f"  ||S(q)||² = {np.sum(Sq**2):.4f}  (= Φ(q))")

    # Stam check
    stam_lhs = 1 / Phi_c
    stam_rhs = 1 / Phi_p + 1 / Phi_q
    print(f"  1/Φ(c) = {stam_lhs:.6f} ≥ {stam_rhs:.6f} = 1/Φ(p) + 1/Φ(q)"
          f"  {'✓' if stam_lhs >= stam_rhs - 1e-10 else '✗'}")


# ═══════════════════════════════════════════════════════════════════
# PART 3: The polynomial-level view
# MSS convolution acts on coefficients. Can we express Φ(c) directly
# in terms of the COEFFICIENTS (not roots) of p and q?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 3: Coefficient-level structure")
print("Express Φ in terms of elementary symmetric polynomials")
print("=" * 70)

# For a monic polynomial x^n - e1·x^{n-1} + e2·x^{n-2} - ... + (-1)^n·en
# with roots γ_1,...,γ_n:
#   Φ = Σ S_k² where S_k = p''(γ_k)/(2p'(γ_k))
#
# Φ can also be written using Newton's identities and discriminant.
# Key: Φ = Σ_k S_k² = Σ_k (Σ_{j≠k} 1/(γ_k-γ_j))²
#
# There's a known formula: Σ_k S_k² = -Σ_k d²/dx² log|p(x)| at x=γ_k
# ... but this is circular.
#
# Let's try: express Φ as a rational function of e1,...,en and discriminant.

# For n=3: p(x) = x³ - e1·x² + e2·x - e3
# Discriminant D = e1²e2² - 4e2³ - 4e1³e3 + 18e1e2e3 - 27e3²
# S_k = Σ_{j≠k} 1/(γ_k - γ_j)
# Φ = Σ S_k²

# Numerical check: compute Φ via roots AND via a coefficient formula

# First, establish the relationship for n=3
print("\n  n=3: Coefficient-level Φ")
print("  Testing Φ = f(e1, e2, e3) via numerical examples")

for trial in range(5):
    roots = np.sort(np.random.randn(3) * 2)
    e1 = np.sum(roots)
    e2 = roots[0]*roots[1] + roots[0]*roots[2] + roots[1]*roots[2]
    e3 = roots[0]*roots[1]*roots[2]
    p2 = np.sum(roots**2)  # power sum
    p3 = np.sum(roots**3)

    Phi_roots = phi_n(roots)

    # Known identity for n=3:
    # Σ_k S_k = 0 (always, for any n, since S_k is antisymmetric under relabeling... no)
    S = score_field(roots)
    S_sum = np.sum(S)

    # Discriminant
    disc = e1**2*e2**2 - 4*e2**3 - 4*e1**3*e3 + 18*e1*e2*e3 - 27*e3**2

    # Φ in terms of discriminant?
    # For centered p (e1=0): disc = -4e2³ - 27e3²
    # Φ should be related to p''/p' evaluated at roots, hence to disc

    print(f"  roots={roots}, Φ={Phi_roots:.4f}, ΣS_k={S_sum:.6f}, disc={disc:.4f}")


# ═══════════════════════════════════════════════════════════════════
# PART 4: The KEY question — matrix model for S
# MSS: p⊞q = E_σ[ Π_k (x - (γ_k + δ_{σ(k)})/scaling) ] over matchings σ
# For each matching, the "matched polynomial" has roots (γ_k + δ_{σ(k)}).
# The score of the AVERAGE should relate to the average of scores.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 4: Matching model — S of average vs average of S")
print("=" * 70)

from itertools import permutations

def matching_polys(p_roots, q_roots):
    """For each permutation σ, compute polynomial with roots derived from matching."""
    n = len(p_roots)
    polys = []
    for sigma in permutations(range(n)):
        # MSS convolution: each matching gives roots that are
        # "p_roots + q_roots[sigma]" in some sense.
        # But MSS isn't literally root addition — it's coefficient-level.
        # Still, let's check.
        matched_roots = p_roots + q_roots[list(sigma)]
        polys.append(matched_roots)
    return polys


# For n=3, there are 6 permutations
p3 = np.array([-1.0, 0.0, 1.0])
q3 = np.array([-2.0, 0.0, 2.0])
c3 = convolve_roots(p3, q3)

print(f"\n  p = {p3}, q = {q3}")
print(f"  p ⊞ q = {c3}")

# Expected characteristic polynomial of the sum
matched = matching_polys(p3, q3)
avg_coeffs = np.zeros(3)
for m in matched:
    avg_coeffs += np.poly(np.sort(m))[1:] / len(matched)

# This gives the average of characteristic polynomials
print(f"\n  Average of matched char polys (coefficients): {avg_coeffs}")
print(f"  MSS convolution coefficients: {np.poly(c3)[1:]}")

# Are they the same? MSS convolution IS the expected char poly?
mss_coeffs = mss_convolve(np.poly(p3)[1:], np.poly(q3)[1:], 3)
print(f"  MSS formula coefficients: {mss_coeffs}")

# Check if matching average == MSS
err = np.max(np.abs(avg_coeffs - mss_coeffs))
print(f"  Matching average vs MSS error: {err:.2e}")

# If MSS IS the average of matchings, then:
# S(p⊞q) relates to the AVERAGE of S over matchings, but nonlinearly
# because roots of the average polynomial ≠ average of roots.

# Key insight: Φ of the average ≤ average of Φ (by convexity??)
print(f"\n  Φ for each matching:")
Phi_matchings = []
for i, m in enumerate(matched):
    m_sorted = np.sort(m)
    Phi_m = phi_n(m_sorted)
    Phi_matchings.append(Phi_m)
    if i < 6:  # print first few
        print(f"    σ={i}: roots={m_sorted}, Φ={Phi_m:.4f}")

avg_Phi = np.mean(Phi_matchings)
print(f"\n  Average Φ over matchings: {avg_Phi:.4f}")
print(f"  Φ of MSS convolution:    {phi_n(c3):.4f}")
print(f"  Φ(avg) vs avg(Φ): ratio = {phi_n(c3)/avg_Phi:.6f}")

# Also check: does 1/Φ(p⊞q) ≥ average of 1/Φ over matchings?
avg_inv_Phi = np.mean([1/p for p in Phi_matchings])
print(f"\n  1/Φ(p⊞q):            {1/phi_n(c3):.6f}")
print(f"  Average 1/Φ(matching):{avg_inv_Phi:.6f}")
print(f"  Ratio: {(1/phi_n(c3))/avg_inv_Phi:.6f}")


# ═══════════════════════════════════════════════════════════════════
# PART 5: Is MSS the average-of-matchings for general n?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 5: MSS = average of matching char polys? Check for n=3,4")
print("=" * 70)

np.random.seed(42)

for n in [3, 4]:
    for trial in range(3):
        p = np.sort(np.random.randn(n) * 2)
        q = np.sort(np.random.randn(n) * 2)

        # MSS
        mss = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)

        # Average over all n! matchings
        avg = np.zeros(n)
        count = 0
        for sigma in permutations(range(n)):
            matched_roots = p + q[list(sigma)]
            avg += np.poly(np.sort(matched_roots))[1:]
            count += 1
        avg /= count

        err = np.max(np.abs(mss - avg))
        print(f"  n={n}, trial {trial}: MSS vs matching-avg error = {err:.2e}")


# ═══════════════════════════════════════════════════════════════════
# PART 6: If MSS = E_σ[char poly], then by Jensen's inequality:
# 1/Φ(MSS) vs E[1/Φ(matching)]
# Since Φ = ||S||², and 1/x is convex, Jensen gives:
# E[1/Φ(matching)] ≥ 1/E[Φ(matching)] ≥ 1/Φ(MSS)?  (if Φ is concave in coeffs)
# Or the reverse?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 6: Jensen analysis — Φ(MSS) vs E[Φ(matching)]")
print("For Stam, we'd want 1/Φ(MSS) ≥ 1/Φ(p) + 1/Φ(q)")
print("Each matching has Φ(γ_k + δ_{σ(k)}). How does this decompose?")
print("=" * 70)

np.random.seed(42)

for n in [3, 4]:
    print(f"\n  n={n}:")
    for trial in range(5):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3))

        c = convolve_roots(p, q)
        if c is None:
            continue

        # Φ for each matching
        Phi_ms = []
        for sigma in permutations(range(n)):
            m = np.sort(p + q[list(sigma)])
            Phi_ms.append(phi_n(m))

        Phi_c = phi_n(c)
        E_Phi = np.mean(Phi_ms)
        E_inv_Phi = np.mean([1/x for x in Phi_ms])

        # For each matching σ, roots are γ_k + δ_{σ(k)}.
        # Score of matching: S_k^σ = Σ_{j≠k} 1/((γ_k+δ_{σ(k)})-(γ_j+δ_{σ(j)}))
        # If σ preserves ordering: S_k^σ = Σ_{j≠k} 1/((γ_k-γ_j)+(δ_{σ(k)}-δ_{σ(j)}))
        # This is a PERTURBATION of S_k(p) when δ is small!

        # For the IDENTITY matching (σ = id), roots = γ_k + δ_k
        id_roots = np.sort(p + q)
        Phi_id = phi_n(id_roots)

        print(f"    trial {trial}: Φ(p⊞q)={Phi_c:.4f}, E[Φ]={E_Phi:.4f},"
              f" Φ(id-match)={Phi_id:.4f},"
              f" Φ(c)/E[Φ]={Phi_c/E_Phi:.4f},"
              f" 1/Φ_c vs 1/Φ_p+1/Φ_q:"
              f" {1/Phi_c:.4f} vs {1/phi_n(p)+1/phi_n(q):.4f}")


# ═══════════════════════════════════════════════════════════════════
# PART 7: Score of a matching — decomposition into p and q parts
# For matching σ with roots r_k = γ_k + δ_{σ(k)}:
#   S_k^σ = Σ_{j≠k} 1/(r_k - r_j) = Σ_{j≠k} 1/((γ_k-γ_j)+(δ_{σ(k)}-δ_{σ(j)}))
# This can be expanded as a series in δ/γ if δ is small.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 7: Score decomposition for a single matching")
print("r_k = γ_k + δ_k (identity matching)")
print("S^r_k = Σ 1/((γ_k-γ_j)+(δ_k-δ_j))")
print("Compare to S^γ_k and S^δ_k")
print("=" * 70)

for p, q, label in [
    (np.array([-3.0, 0.0, 3.0]), np.array([-0.3, 0.0, 0.3]), "p >> q"),
    (np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), "p = q"),
    (np.array([-0.3, 0.0, 0.3]), np.array([-3.0, 0.0, 3.0]), "p << q"),
]:
    r = np.sort(p + q)
    Sp = score_field(p)
    Sq = score_field(q)
    Sr = score_field(r)

    print(f"\n  --- {label} ---")
    print(f"  p={p}, q={q}, r=p+q={r}")
    print(f"  S(p) = {Sp}")
    print(f"  S(q) = {Sq}")
    print(f"  S(r) = {Sr}")
    print(f"  S(p)+S(q) = {Sp+Sq}")
    print(f"  S(r)/(S(p)+S(q)) = {Sr/(Sp+Sq)}")

    # Cauchy-Schwarz bound: ||S(r)||² vs (||S(p)||² + ||S(q)||²)
    print(f"  Φ(r)={np.sum(Sr**2):.4f},"
          f" Φ(p)={np.sum(Sp**2):.4f},"
          f" Φ(q)={np.sum(Sq**2):.4f},"
          f" Φ(p)+Φ(q)={np.sum(Sp**2)+np.sum(Sq**2):.4f}")


# ═══════════════════════════════════════════════════════════════════
# PART 8: THE HYPOTHESIS
# If S(r) = A·S(p) + B·S(q) for some A, B depending on roots,
# then ||S(r)||² = A²||S(p)||² + B²||S(q)||² + 2AB·<S(p),S(q)>
# For Stam, we'd need 1/||S(r)||² ≥ 1/||S(p)||² + 1/||S(q)||²
#
# But for a SINGLE matching, this relates to addition of roots.
# The MSS convolution averages over matchings.
# So: Φ(p⊞q) = Φ(avg of matchings) ≤? f(avg(Φ(matching)))
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PART 8: Least-squares fit of S(r) = A·S(p) + B·S(q)")
print("For the identity matching r = p + q")
print("=" * 70)

np.random.seed(123)
for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    for trial in range(5):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3))
        r = np.sort(p + q)

        Sp = score_field(p)
        Sq = score_field(q)
        Sr = score_field(r)

        # Fit Sr = a*Sp + b*Sq via least squares
        # Note: Sp, Sq are at DIFFERENT points than Sr (different root locations)
        # So this is NOT a valid decomposition in the same vector space.
        # But let's see if the norms work out.

        # Instead: compute the n x n "interaction matrix"
        # For r_k = γ_k + δ_k (identity matching with sorted roots):
        # 1/(r_k - r_j) = 1/((γ_k-γ_j) + (δ_k-δ_j))
        # Let g_kj = γ_k - γ_j, d_kj = δ_k - δ_j
        # 1/(g+d) = (1/g)·1/(1+d/g) = (1/g)·Σ(-d/g)^m  (geometric series)
        # So S^r_k ≈ S^γ_k - Σ_{j≠k} (d_kj)/(g_kj)² + ...
        #          = S^γ_k - Σ_{j≠k} (δ_k-δ_j)/(γ_k-γ_j)²

        # First-order correction to S:
        S_correction = np.zeros(n)
        for k in range(n):
            for j in range(n):
                if j != k:
                    g = p[k] - p[j]
                    d = q[k] - q[j]
                    S_correction[k] -= d / g**2

        Sr_approx1 = Sp + S_correction
        err1 = np.max(np.abs(Sr - Sr_approx1))

        # The correction term involves (δ_k-δ_j)/(γ_k-γ_j)²
        # This is like T_{kj}(q, in p-basis) but not quite.

        # Ratio of correction to Sq
        print(f"    trial {trial}: ||S(r)-S(p)||/||S(p)||={np.linalg.norm(Sr-Sp)/np.linalg.norm(Sp):.4f},"
              f" 1st-order err={err1:.4f},"
              f" ||correction||/||S(p)||={np.linalg.norm(S_correction)/np.linalg.norm(Sp):.4f}")


print(f"\n{'='*70}")
print("DONE")
print("=" * 70)
