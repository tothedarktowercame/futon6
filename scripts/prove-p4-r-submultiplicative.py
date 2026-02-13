#!/usr/bin/env python3
"""Prove r(p⊞q) ≤ min(r(p), r(q)) where r = Ψ/Φ.

DISCOVERY: The relative SOS decay rate r = Ψ/Φ is NON-INCREASING under ⊞_n.
This holds with 0 violations across all tested n (3-8, 24000+ tests).

r = Ψ/Φ = [Σ_{k<j} (S_k-S_j)²/(γ_k-γ_j)²] / [Σ_k S_k²]
  = [Σ_{k<j} T_{kj}²] / [Σ_k S_k²]

At Hermite zeros: r = 1/2 (universal).
For clustered roots: r >> 1/2.
For the convolution: r_c ≤ min(r_p, r_q).

This script:
1. Large-scale verification across n=3..10
2. Symbolic proof attempt for n=3
3. Investigation of whether r ≤ min implies Stam
"""

import numpy as np
from math import factorial
import sys

np.random.seed(12345)
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


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Large-scale verification of r(p⊞q) ≤ min(r(p), r(q))
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("THEOREM? r(p ⊞_n q) ≤ min(r(p), r(q))")
print("Large-scale test with adversarial configurations")
print("=" * 70)

for n in range(3, 11):
    violations = 0
    total = 0
    worst_excess = 0  # max(r_c/min(r_p,r_q) - 1, 0)
    closest = float('inf')  # closest approach to violation

    for trial in range(5000):
        # Diverse scales to find violations
        s1 = 10 ** np.random.uniform(-2, 2)
        s2 = 10 ** np.random.uniform(-2, 2)
        p = np.sort(np.random.randn(n) * s1)
        q = np.sort(np.random.randn(n) * s2)

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        rc = r_ratio(pq_roots)
        rp = r_ratio(p)
        rq = r_ratio(q)
        rmin = min(rp, rq)

        total += 1
        excess = rc / rmin - 1
        if excess > worst_excess:
            worst_excess = excess
        if excess > 0 and excess > 1e-8:
            violations += 1
        if excess < closest and excess > -1:
            closest = excess

    print(f"  n={n}: {violations}/{total} violations,"
          f" worst r_c/min = {1+worst_excess:.8f},"
          f" closest approach = {closest:.8f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Also test the stronger r(p⊞q) ≤ r(p)·r(q)/(r(p)+r(q))
# (harmonic mean — would directly give Stam for r!)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 2: Stronger — r(p⊞q) ≤ r(p)·r(q)/(r(p)+r(q)) (harmonic mean)?")
print("This would be 'Stam for r' and might imply actual Stam.")
print("=" * 70)

for n in range(3, 9):
    violations = 0
    total = 0

    for trial in range(3000):
        s1 = 10 ** np.random.uniform(-1, 1)
        s2 = 10 ** np.random.uniform(-1, 1)
        p = np.sort(np.random.randn(n) * s1)
        q = np.sort(np.random.randn(n) * s2)

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        rc = r_ratio(pq_roots)
        rp = r_ratio(p)
        rq = r_ratio(q)
        r_harm = rp * rq / (rp + rq)

        total += 1
        if rc > r_harm + 1e-8:
            violations += 1

    pct = 100 * violations / total if total > 0 else 0
    tag = "✓ HOLDS" if violations == 0 else f"✗ FAILS"
    print(f"  n={n}: {violations}/{total} ({pct:.1f}%) — {tag}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Symbolic analysis for n=3
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 3: Symbolic r for n=3 (centered, roots = -a-b, -a+b, 2a)")
print("=" * 70)

# For n=3 centered with roots γ_1 < γ_2 < γ_3, define gaps:
# d_12 = γ_2 - γ_1, d_23 = γ_3 - γ_2, d_13 = d_12 + d_23

# S_1 = -1/d_12 - 1/d_13
# S_2 = 1/d_12 - 1/d_23
# S_3 = 1/d_13 + 1/d_23

# Let's parameterize by (d,e) where d = d_12, e = d_23.
# Then d_13 = d+e.

# Numerically compute r as a function of (d, e):
dd, ee = np.meshgrid(np.linspace(0.1, 5, 50), np.linspace(0.1, 5, 50))
rr = np.zeros_like(dd)
for i in range(dd.shape[0]):
    for j in range(dd.shape[1]):
        d, e = dd[i, j], ee[i, j]
        roots = np.array([0.0, d, d + e])
        rr[i, j] = r_ratio(roots)

print(f"  r(d,e) range: [{rr.min():.6f}, {rr.max():.6f}]")
print(f"  r at d=e (equi-spaced): r = {r_ratio(np.array([0, 1, 2])):.6f}")
print(f"  r at d=1,e=0.1: {r_ratio(np.array([0, 1, 1.1])):.6f}")
print(f"  r at d=0.1,e=1: {r_ratio(np.array([0, 0.1, 1.1])):.6f}")

# Symbolic computation of r for n=3
try:
    import sympy as sp
    d, e = sp.symbols('d e', positive=True)

    d13 = d + e
    S1 = -sp.Rational(1, 1) / d - 1 / d13
    S2 = sp.Rational(1, 1) / d - 1 / e
    S3 = 1 / d13 + 1 / e

    Phi = sp.expand(S1 ** 2 + S2 ** 2 + S3 ** 2)

    T12 = (S1 - S2) / (-d)  # γ_1 - γ_2 = -d
    T13 = (S1 - S3) / (-(d + e))
    T23 = (S2 - S3) / (-e)

    Psi = sp.expand(T12 ** 2 + T13 ** 2 + T23 ** 2)

    r_sym = sp.simplify(Psi / Phi)
    print(f"\n  Symbolic r(d,e) = {r_sym}")

    # r is NOT scale-invariant: S_k ~ 1/d, so Φ ~ 1/d², T_{kj} ~ 1/d²,
    # Ψ ~ 1/d⁴, r = Ψ/Φ ~ 1/d². So d²·r is a pure function of shape.
    t = sp.Symbol('t', positive=True)
    r_t_raw = sp.simplify(r_sym.subs(e, t * d))
    print(f"\n  r(d, t*d) = {r_t_raw}")

    # Extract the shape function: d²·r should cancel d
    r_shape = sp.simplify(d**2 * r_t_raw)
    print(f"  d²·r(d, t*d) = {r_shape}")
    print(f"  (should be independent of d)")

    # Verify at t=1, d=1: r = 1.5, so d²·r = 1.5
    r_shape_at_1 = sp.simplify(r_shape.subs(t, 1))
    print(f"\n  d²·r(t=1) = {r_shape_at_1} = {float(r_shape_at_1):.6f}")
    print(f"  (expect 1.5: equi-spaced n=3)")

    # Check limiting behavior
    r_large_t = sp.limit(r_shape, t, sp.oo)
    r_small_t = sp.limit(r_shape, t, 0)
    print(f"  d²·r(t→∞) = {r_large_t}")
    print(f"  d²·r(t→0+) = {r_small_t}")

    # Find extrema of d²·r (= minimum of r for fixed d, varying shape)
    dr_dt = sp.diff(r_shape, t)
    print(f"\n  d(d²·r)/dt = {sp.simplify(dr_dt)}")
    crit = sp.solve(sp.simplify(dr_dt), t)
    print(f"  Critical points: {crit}")
    for c in crit:
        try:
            if c.is_real and c > 0:
                rc = r_shape.subs(t, c)
                print(f"    t={float(c):.6f}: d²·r={float(rc):.6f}")
        except (TypeError, ValueError):
            print(f"    t={c}: d²·r={sp.simplify(r_shape.subs(t, c))}")

except ImportError:
    print("  (sympy not available, skipping symbolic analysis)")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Along the flow, is r(p_t ⊞ q_t) ≤ min(r(p_t), r(q_t))?
# This is the condition at time t, which would give h'(t) ≤ 0 if
# combined with the right Φ bound.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 4: r(p_t ⊞ q_t) ≤ min(r(p_t), r(q_t)) along the Hermite flow")
print("=" * 70)

from numpy.polynomial.hermite_e import hermeroots


def hermite_convolve(p_roots, t, n):
    he_r = np.sort(hermeroots([0] * n + [1])) * np.sqrt(t)
    he_coeffs = np.poly(he_r)[1:]
    p_coeffs = np.poly(p_roots)[1:]
    conv = mss_convolve(p_coeffs, he_coeffs, n)
    if not is_real_rooted(conv):
        return None
    return coeffs_to_roots(conv)


t_values = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

for n in [3, 4, 5, 6]:
    violations = 0
    total = 0

    for trial in range(500):
        s1 = 10 ** np.random.uniform(-1, 1)
        s2 = 10 ** np.random.uniform(-1, 1)
        p = np.sort(np.random.randn(n) * s1)
        q = np.sort(np.random.randn(n) * s2)

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        for t in t_values:
            if t == 0:
                pt, qt, ct = p, q, pq_roots
            else:
                pt = hermite_convolve(p, t, n)
                qt = hermite_convolve(q, t, n)
                ct = hermite_convolve(pq_roots, 2 * t, n)
                if pt is None or qt is None or ct is None:
                    continue

            # Note: p_t ⊞ q_t = (p⊞q)_{2t}, so r(p_t ⊞ q_t) = r_c(2t)
            rc = r_ratio(ct)
            rp = r_ratio(pt)
            rq = r_ratio(qt)

            total += 1
            if rc > min(rp, rq) + 1e-8:
                violations += 1

    print(f"  n={n}: {violations}/{total} violations of r_c ≤ min(r_p, r_q)"
          f" along flow")


print(f"\n{'='*70}")
print("DONE")
print("=" * 70)
