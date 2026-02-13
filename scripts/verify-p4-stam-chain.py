#!/usr/bin/env python3
"""Investigate the proof chain: de Bruijn + Φ monotonicity + B2 → Stam.

The classical continuous proof of Stam (1/J(X+Y) ≥ 1/J(X) + 1/J(Y)):

Method 1 (via EPI):
  - EPI: N(X+Y) ≥ N(X) + N(Y) where N = exp(2H/n)
  - De Bruijn + concavity of log gives Stam

Method 2 (direct, Blachman-Stam):
  - For X independent of Y with X+Y:
    J(X+Y) ≤ (J(X)J(Y))/(J(X)+J(Y)) (harmonic mean)
  - Proved by: score decomposition J(X+Y) = E[(E[score(X)|X+Y])²]
    and Cauchy-Schwarz

For our finite case, the most promising route seems to be:
  - Use the Coulomb flow to express p ⊞_n q as the endpoint of a flow
  - Use Φ_n monotonicity to bound Φ_n at the endpoint

Let me test a specific construction:
  - Given p, q, define p_t = p ⊞_n He_t
  - At t = "appropriate value related to q", p_t should relate to p ⊞_n q
  - If Φ_n(p_t) is decreasing in t, then Φ_n(p ⊞_n q) ≤ Φ_n(p)
  - But we need the harmonic mean bound, not just this...

Actually, let me think about this differently. The key observation might be:

  p ⊞_n q = (p ⊞_n He_s) ⊞_n (q ⊞_n He_t) [NO — ⊞_n is not iterable like this]

Wait, actually MSS convolution IS associative. So:
  (p ⊞_n He_s) ⊞_n (q ⊞_n He_t) = (p ⊞_n q) ⊞_n (He_s ⊞_n He_t)

And He_s ⊞_n He_t = He_{s+t} if we're lucky (additive in the parameter).

Let me check that numerically first.
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


def hermite_coeffs(n, t):
    from numpy.polynomial.hermite_e import hermeroots
    base = sorted(hermeroots([0]*n + [1]))
    roots = np.array(base) * np.sqrt(t)
    return np.poly(roots)[1:]


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


# Test 1: Is He_s ⊞_n He_t = He_{s+t}?
print("=" * 70)
print("TEST 1: Is He_s ⊞_n He_t = He_{s+t}?")
print("=" * 70)

for n in [3, 4, 5, 6]:
    max_err = 0.0
    for s in [0.3, 0.5, 1.0]:
        for t in [0.3, 0.5, 1.0]:
            he_s = hermite_coeffs(n, s)
            he_t = hermite_coeffs(n, t)
            he_st = hermite_coeffs(n, s + t)

            conv = mss_convolve(he_s, he_t, n)
            err = np.max(np.abs(conv - he_st))
            max_err = max(max_err, err)

    status = "✓" if max_err < 1e-10 else f"✗ (err={max_err:.2e})"
    print(f"  n={n}: max error = {max_err:.2e}  {status}")

# Test 2: Investigate the "data processing" inequality for ⊞_n
# Does Φ_n(p ⊞_n q) ≤ min(Φ_n(p), Φ_n(q))?
# Does 1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q)?

print(f"\n{'='*70}")
print("TEST 2: Stam inequality 1/Φ_n(p⊞q) ≥ 1/Φ_n(p) + 1/Φ_n(q)")
print(f"{'='*70}")

for n in [3, 4, 5]:
    n_violations = 0
    n_tests = 200
    surpluses = []

    for test in range(n_tests):
        p_roots = np.sort(np.random.randn(n) * 2)
        q_roots = np.sort(np.random.randn(n) * 2)

        p_coeffs = np.poly(p_roots)[1:]
        q_coeffs = np.poly(q_roots)[1:]

        conv = mss_convolve(p_coeffs, q_coeffs, n)
        if not is_real_rooted(conv):
            continue

        pq_roots = coeffs_to_roots(conv)

        phi_p = phi_n(p_roots)
        phi_q = phi_n(q_roots)
        phi_pq = phi_n(pq_roots)

        surplus = 1.0/phi_pq - 1.0/phi_p - 1.0/phi_q
        surpluses.append(surplus)

        if surplus < -1e-8:
            n_violations += 1

    surpluses = np.array(surpluses)
    print(f"\n  n={n}: {n_violations}/{len(surpluses)} violations")
    print(f"    surplus min={surpluses.min():.8f}, mean={surpluses.mean():.8f}, "
          f"max={surpluses.max():.8f}")


# Test 3: The flow-based proof attempt
# Idea: p ⊞_n q = endpoint of a two-parameter flow
# Define r(s,t) = (p ⊞_n He_s) ⊞_n (q ⊞_n He_t)
#                = (p ⊞_n q) ⊞_n He_{s+t}  [if He_s ⊞_n He_t = He_{s+t}]
# At s=t=0: r(0,0) = p ⊞_n q
# As s→∞: r(s,0) → He_s (semicircular limit) ⊞_n q
# As s,t→∞: r(s,t) → He_{s+t}

# Key: 1/Φ_n(r(s,t)) = 1/Φ_n((p⊞q) ⊞ He_{s+t}) is increasing in s+t
# (by our Φ_n monotonicity along Coulomb flow)

# This gives: 1/Φ_n((p⊞q) ⊞ He_u) is increasing in u.
# At u=0: 1/Φ_n(p ⊞ q)
# As u→∞: 1/Φ_n → 1/Φ_n(He_u equilibrium) → ∞

# But this doesn't directly give the Stam inequality...

print(f"\n{'='*70}")
print("TEST 3: Two-parameter flow r(s,t) = (p⊞He_s) ⊞ (q⊞He_t)")
print(f"{'='*70}")

for n in [3, 4, 5]:
    print(f"\n  n={n}:")

    p_roots = np.sort(np.random.randn(n) * 2)
    q_roots = np.sort(np.random.randn(n) * 2)
    p_coeffs = np.poly(p_roots)[1:]
    q_coeffs = np.poly(q_roots)[1:]

    # Direct convolution
    pq = mss_convolve(p_coeffs, q_coeffs, n)
    if not is_real_rooted(pq):
        print("    p⊞q not real-rooted, skip")
        continue

    pq_roots = coeffs_to_roots(pq)
    phi_pq = phi_n(pq_roots)
    phi_p = phi_n(p_roots)
    phi_q = phi_n(q_roots)

    print(f"    Φ(p) = {phi_p:.6f}, Φ(q) = {phi_q:.6f}, Φ(p⊞q) = {phi_pq:.6f}")
    print(f"    1/Φ(p⊞q) = {1/phi_pq:.6f}")
    print(f"    1/Φ(p) + 1/Φ(q) = {1/phi_p + 1/phi_q:.6f}")
    print(f"    Surplus = {1/phi_pq - 1/phi_p - 1/phi_q:.6f}")

    # Flow: (p⊞q) ⊞ He_u for various u
    print(f"\n    u → Φ((p⊞q)⊞He_u)  1/Φ  [should increase]")
    for u in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        if u == 0.0:
            r = pq
        else:
            he_u = hermite_coeffs(n, u)
            r = mss_convolve(pq, he_u, n)
        if is_real_rooted(r):
            roots = coeffs_to_roots(r)
            phi_val = phi_n(roots)
            print(f"    u={u:.1f}: Φ={phi_val:.6f}, 1/Φ={1/phi_val:.6f}")

    # Also: p ⊞ He_u and q ⊞ He_u separately
    print(f"\n    u → Φ(p⊞He_u)  Φ(q⊞He_u)  1/Φ(p)+1/Φ(q)")
    for u in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        if u == 0.0:
            phi_pu = phi_p
            phi_qu = phi_q
        else:
            he_u = hermite_coeffs(n, u)
            pu = mss_convolve(p_coeffs, he_u, n)
            qu = mss_convolve(q_coeffs, he_u, n)
            if not (is_real_rooted(pu) and is_real_rooted(qu)):
                continue
            phi_pu = phi_n(coeffs_to_roots(pu))
            phi_qu = phi_n(coeffs_to_roots(qu))
        print(f"    u={u:.1f}: Φ(p)={phi_pu:.6f}, Φ(q)={phi_qu:.6f}, "
              f"1/Φ(p)+1/Φ(q)={1/phi_pu + 1/phi_qu:.6f}")


# Test 4: Key identity — does He_s ⊞_n He_t = He_{s+t}?
# If yes, then the semigroup property means:
# (p ⊞ He_s) ⊞ (q ⊞ He_t) = ... we need associativity+commutativity of ⊞_n

# Actually the key interpolation is:
# Define f(α) = 1/Φ_n(p ⊞_n He_{αT}) for some T
# f is increasing (Φ monotonicity)
# f(0) = 1/Φ_n(p)
# f(1) = 1/Φ_n(p ⊞_n He_T)
# Similarly g(α) = 1/Φ_n(q ⊞_n He_{αT}) is increasing

# But we want to relate f(1) + g(1) to 1/Φ_n((p⊞He_T) ⊞ (q⊞He_T))

# Hmm, the problem is that (p⊞He_T) ⊞ (q⊞He_T) ≠ (p⊞q) ⊞ He_{2T}
# unless He_T ⊞ He_T = He_{2T}

print(f"\n{'='*70}")
print("TEST 4: Is He_s ⊞_n He_s = He_{2s}? (Hermite self-convolution)")
print(f"{'='*70}")

for n in [3, 4, 5]:
    for s in [0.5, 1.0, 2.0]:
        he_s = hermite_coeffs(n, s)
        he_2s = hermite_coeffs(n, 2*s)
        conv = mss_convolve(he_s, he_s, n)
        err = np.max(np.abs(conv - he_2s))
        print(f"  n={n}, s={s}: He_s ⊞ He_s vs He_{2*s:.0f}: error = {err:.2e}")


# Test 5: Direct "de Bruijn route" for Stam
# Classical proof:
#   1. de Bruijn: d/dt H(X+√tZ) = (1/2)J(X+√tZ)
#   2. H(X+√tZ) ≥ H(X) + H(√tZ)  [independence/EPI]
#   3. Therefore: integral of J gives H bound
#   4. Optimizing over t gives Stam

# Finite analog:
#   1. de Bruijn: d/dt H'_n(p_t) = Φ_n(p_t)  [PROVED]
#   2. H'_n(p ⊞_n q) ≥ H'_n(p) + H'_n(q)  [B2, holds n≥4]
#   3. Φ_n is decreasing along flow [PROVED]

# Can we get Stam from (1)+(2)+(3)?

# Key: for p_t = p ⊞ He_t and q_u = q ⊞ He_u:
# By semigroup: p_t ⊞ q_u = (p ⊞ q) ⊞ He_{t+u}  [if He_t ⊞ He_u = He_{t+u}]

# Then H'(p_t ⊞ q_u) = H'((p⊞q) ⊞ He_{t+u})
# Using de Bruijn, d/du H'((p⊞q)⊞He_{t+u}) = Φ((p⊞q)⊞He_{t+u})

# Also H'(p_t ⊞ q_u) ≥ H'(p_t) + H'(q_u) by B2

# So: H'((p⊞q)⊞He_{t+u}) ≥ H'(p⊞He_t) + H'(q⊞He_u)

# Differentiating in t at t=u=0:
# d/dt H'((p⊞q)⊞He_t)|_{t=0} ≥ d/dt H'(p⊞He_t)|_{t=0} + 0

# That gives: Φ(p⊞q) ≥ Φ(p)  -- NOT what we want.

# Need a different parametrization...

print(f"\n{'='*70}")
print("TEST 5: Investigating the correct interpolation for Stam via de Bruijn")
print(f"{'='*70}")

# The classical Stam proof actually uses:
# 1. X_t = X + √t Z independent of Y_t = Y + √t Z'
# 2. J(X_t + Y_t) ≤ α²J(X_t) + (1-α)²J(Y_t) for optimal α
#    (by score decomposition and Cauchy-Schwarz)
# 3. This gives J(X+Y+√(2t)Z) ≤ harmonic mean of J(X+√tZ) and J(Y+√tZ)
# 4. Take t → 0

# In our case:
# p_t ⊞ q_t = (p ⊞ q) ⊞ He_{2t}  [by semigroup]
# Φ(p_t ⊞ q_t) ≤ ? ... this is the score decomposition step

# Actually the direct route in the finite case might be:
# Use the FLOW to deform p → Hermite, q → Hermite
# Track how 1/Φ changes
# Use the semigroup property

# Let me test: does 1/Φ((p⊞q)⊞He_t) increase faster than 1/Φ(p⊞He_t) + 1/Φ(q⊞He_t)?

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    p_roots = np.sort(np.random.randn(n) * 2)
    q_roots = np.sort(np.random.randn(n) * 2)
    p_coeffs = np.poly(p_roots)[1:]
    q_coeffs = np.poly(q_roots)[1:]

    pq = mss_convolve(p_coeffs, q_coeffs, n)
    if not is_real_rooted(pq):
        print("    skip")
        continue

    print(f"    {'t':>5} {'1/Φ(pq⊞He_t)':>15} {'1/Φ(p⊞He_t)+1/Φ(q⊞He_t)':>28} {'surplus':>12}")
    for t in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if t == 0.0:
            pq_t = pq
            p_t = p_coeffs
            q_t = q_coeffs
        else:
            he_t = hermite_coeffs(n, t)
            pq_t = mss_convolve(pq, he_t, n)
            p_t = mss_convolve(p_coeffs, he_t, n)
            q_t = mss_convolve(q_coeffs, he_t, n)

        if not (is_real_rooted(pq_t) and is_real_rooted(p_t) and is_real_rooted(q_t)):
            continue

        inv_phi_pq = 1.0/phi_n(coeffs_to_roots(pq_t))
        inv_phi_p = 1.0/phi_n(coeffs_to_roots(p_t))
        inv_phi_q = 1.0/phi_n(coeffs_to_roots(q_t))
        surplus = inv_phi_pq - inv_phi_p - inv_phi_q

        print(f"    {t:5.1f} {inv_phi_pq:15.6f} {inv_phi_p + inv_phi_q:28.6f} {surplus:12.6f}")
