#!/usr/bin/env python3
"""Explore n=4 Stam for SYMMETRIC quartics (e₃ = 0).

For e₃ = 0, the Φ₄·disc identity factors beautifully:
  disc(e₂, 0, e₄) = 16·e₄·(e₂² - 4e₄)²
  P(e₂, 0, e₄) = -8·e₂·(e₂² + 12e₄)·(e₂² - 4e₄)

So: 1/Φ₄(e₂, 0, e₄) = -2·e₄·(e₂² - 4e₄) / [e₂·(e₂² + 12e₄)]

In positive coords (s = -e₂ > 0):
  1/Φ₄ = 2·e₄·(s² - 4e₄) / [s·(s² + 12e₄)]

The centered ⊞₄ with e₃ = 0 gives:
  S = s + t,  A = a + b + st/6

Stam surplus = 1/Φ₄(S,0,A) - 1/Φ₄(s,0,a) - 1/Φ₄(t,0,b)

This is a 4-variable problem: (s, t, a, b) with constraints s,t > 0,
0 < 4a < s², 0 < 4b < t².

If this decomposes via SOS/Titu, we have a proof for symmetric quartics!
"""

import numpy as np
import sympy as sp
from itertools import combinations

# ═══════════════════════════════════════════════════════════════════
# PART 1: Verify the e₃=0 formula numerically
# ═══════════════════════════════════════════════════════════════════

print("=" * 72)
print("PART 1: Verify 1/Φ₄ = 2e₄(s²-4e₄)/[s(s²+12e₄)] for symmetric quartics")
print("=" * 72)


def phi_n_numeric(roots):
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
    return np.sum(S ** 2)


np.random.seed(42)
for trial in range(10):
    # Generate symmetric quartic: roots ±α, ±β with α > β > 0
    alpha = np.random.uniform(1, 5)
    beta = np.random.uniform(0.1, alpha - 0.01)
    roots = np.array([-alpha, -beta, beta, alpha])

    s = alpha ** 2 + beta ** 2  # = -e₂ (since e₂ = -(α²+β²) for centered)
    e4 = alpha ** 2 * beta ** 2  # = e₄

    Phi_actual = phi_n_numeric(roots)
    inv_Phi_formula = 2 * e4 * (s ** 2 - 4 * e4) / (s * (s ** 2 + 12 * e4))
    inv_Phi_actual = 1 / Phi_actual

    err = abs(inv_Phi_formula - inv_Phi_actual)
    print(f"  trial {trial}: s={s:.4f}, e4={e4:.4f},"
          f" 1/Φ formula={inv_Phi_formula:.8f},"
          f" 1/Φ actual={inv_Phi_actual:.8f}, err={err:.2e}")


# ═══════════════════════════════════════════════════════════════════
# PART 2: Symbolic surplus for symmetric quartics
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print("PART 2: Symbolic Stam surplus for symmetric quartics")
print("=" * 72)

s, t, a, b = sp.symbols('s t a b', positive=True)


def inv_phi4_sym(sigma, e4):
    """1/Φ₄ for symmetric quartic with -e₂ = sigma, e₄ = e4."""
    return 2 * e4 * (sigma ** 2 - 4 * e4) / (sigma * (sigma ** 2 + 12 * e4))


S_conv = s + t
A_conv = a + b + s * t / 6

inv_phi_p = inv_phi4_sym(s, a)
inv_phi_q = inv_phi4_sym(t, b)
inv_phi_c = inv_phi4_sym(S_conv, A_conv)

surplus = sp.expand(inv_phi_c - inv_phi_p - inv_phi_q)
surplus_simplified = sp.simplify(surplus)

print(f"  surplus = {surplus_simplified}")

# Get numerator and denominator
surplus_together = sp.together(surplus)
num, den = sp.fraction(surplus_together)
num = sp.expand(num)
den = sp.expand(den)

print(f"\n  numerator degree in (s,t,a,b):")
num_poly = sp.Poly(num, s, t, a, b)
print(f"    total degree = {num_poly.total_degree()}")
print(f"    number of terms = {len(num_poly.as_dict())}")

print(f"\n  denominator:")
den_poly = sp.Poly(den, s, t, a, b)
print(f"    total degree = {den_poly.total_degree()}")

# Check denominator sign: should be positive for s,t>0, a,b>0
print(f"\n  denominator = {sp.factor(den)}")

print(f"\n  numerator = {num}")


# ═══════════════════════════════════════════════════════════════════
# PART 3: Test numerically — does surplus ≥ 0?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print("PART 3: Numerical Stam check for symmetric quartics")
print("=" * 72)

np.random.seed(123)
violations = 0
total = 0
min_surplus = float('inf')

for trial in range(10000):
    s_val = np.random.uniform(0.5, 5)
    t_val = np.random.uniform(0.5, 5)
    # Need 4a < s², 4b < t² for real roots
    a_val = np.random.uniform(0.01, s_val ** 2 / 4 - 0.01)
    b_val = np.random.uniform(0.01, t_val ** 2 / 4 - 0.01)

    A_val = a_val + b_val + s_val * t_val / 6
    S_val = s_val + t_val

    # Check convolution is real-rooted: need 4A < S²
    if 4 * A_val >= S_val ** 2:
        continue

    inv_p = 2 * a_val * (s_val ** 2 - 4 * a_val) / (s_val * (s_val ** 2 + 12 * a_val))
    inv_q = 2 * b_val * (t_val ** 2 - 4 * b_val) / (t_val * (t_val ** 2 + 12 * b_val))
    inv_c = 2 * A_val * (S_val ** 2 - 4 * A_val) / (S_val * (S_val ** 2 + 12 * A_val))

    sur = inv_c - inv_p - inv_q
    total += 1
    if sur < min_surplus:
        min_surplus = sur
    if sur < -1e-10:
        violations += 1

print(f"  {violations}/{total} violations, min surplus = {min_surplus:.8f}")


# ═══════════════════════════════════════════════════════════════════
# PART 4: Try to factor / SOS-decompose the numerator
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print("PART 4: Attempt factoring/SOS of numerator")
print("=" * 72)

# First, try sympy's factor
num_factored = sp.factor(num)
print(f"  factor(numerator) = {num_factored}")

# Check if it's manifestly non-negative
# The denominator is positive for s,t,a,b > 0 (product of positive terms)
# So we need numerator ≥ 0 on the feasible region.

# Try further simplification with substitution a = s²α/4, b = t²β/4
# where 0 < α < 1, 0 < β < 1
alpha, beta = sp.symbols('alpha beta', positive=True)
num_sub = num.subs({a: s**2 * alpha / 4, b: t**2 * beta / 4})
num_sub = sp.expand(num_sub)
print(f"\n  After a = s²α/4, b = t²β/4:")
num_sub_poly = sp.Poly(num_sub, s, t, alpha, beta)
print(f"    total degree = {num_sub_poly.total_degree()}")
print(f"    terms = {len(num_sub_poly.as_dict())}")

# Try the equal-scale case s = t
num_equal = num.subs(t, s)
num_equal = sp.expand(num_equal)
num_equal_factored = sp.factor(num_equal)
print(f"\n  At s = t:")
print(f"    factor(numerator) = {num_equal_factored}")

# Try the equal case s=t, a=b
num_symm = num.subs({t: s, b: a})
num_symm = sp.expand(num_symm)
num_symm_factored = sp.factor(num_symm)
print(f"\n  At s=t, a=b (self-convolution):")
print(f"    factor(numerator) = {num_symm_factored}")


print(f"\n{'='*72}")
print("DONE")
print("=" * 72)
