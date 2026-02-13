#!/usr/bin/env python3
"""Search for scale-invariant entropy functionals that ARE superadditive.

The Stam inequality 1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q) holds universally.
The de Bruijn identity d/dt H'_n = Φ_n is proved.
But H'_n is NOT superadditive (scale-dependent failures).

We need a scale-invariant functional E_n such that:
  E_n(p ⊞_n q) ≥ E_n(p) + E_n(q)  (superadditive)
  d/dt E_n(p_t) = f(Φ_n(p_t))      (de Bruijn-like)

Candidates:
1. H'_n / Φ_n  (ratio of entropy to Fisher info)
2. log(1/Φ_n)  (log inverse Fisher)
3. H'_n - (n(n-1)/4) log Φ_n  (mixture)
4. H'_n normalized by variance: H'_n - (n(n-1)/2) log σ_p
5. The "entropy power" N_n = exp(2H'_n / n(n-1))
6. 1/Φ_n itself (already proved: this IS the Stam functional)

Key insight: 1/Φ_n IS superadditive (that's the Stam inequality!) and
Φ_n is monotone along the flow (proved). But we need the de Bruijn
identity to CONNECT these. The question is whether there's an intermediate
functional E_n such that:
  d/dt E_n = g(Φ_n)  (de Bruijn)
  E_n(p⊞q) ≥ E_n(p) + E_n(q)  (superadditive)
  => 1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q)  (Stam)
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
    return sum(sum(1.0/(roots[i]-roots[j]) for j in range(n) if j!=i)**2 for i in range(n))


def log_disc(roots):
    n = len(roots)
    return sum(np.log(abs(roots[i]-roots[j])) for i in range(n) for j in range(i+1,n))


def variance_roots(roots):
    return np.var(roots)


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < 1e-6


def test_functional_superadditivity(name, func, n_vals=[3, 4, 5], n_tests=500):
    """Test if func(p⊞q) ≥ func(p) + func(q)."""
    print(f"\n{'='*70}")
    print(f"Testing superadditivity of: {name}")
    print(f"{'='*70}")

    for n in n_vals:
        violations = 0
        total = 0
        surpluses = []

        for _ in range(n_tests):
            p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
            q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

            conv = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
            if not is_real_rooted(conv):
                continue

            r = coeffs_to_roots(conv)

            try:
                f_pq = func(r)
                f_p = func(p)
                f_q = func(q)
            except (ValueError, ZeroDivisionError):
                continue

            if any(np.isinf([f_pq, f_p, f_q]) | np.isnan([f_pq, f_p, f_q])):
                continue

            surplus = f_pq - f_p - f_q
            surpluses.append(surplus)
            total += 1
            if surplus < -1e-8:
                violations += 1

        if total > 0:
            s = np.array(surpluses)
            print(f"  n={n}: {violations}/{total} violations ({100*violations/total:.1f}%)")
            print(f"    surplus: min={s.min():.6f}, mean={s.mean():.6f}, max={s.max():.6f}")


# Candidate 1: 1/Φ_n (THE Stam functional)
def inv_phi(roots):
    return 1.0 / phi_n(roots)

# Candidate 2: log(1/Φ_n)
def log_inv_phi(roots):
    return np.log(1.0 / phi_n(roots))

# Candidate 3: Variance-normalized log-disc
# H'_n - n(n-1)/2 · log(σ) where σ² = Var(roots)
def var_normalized_logdisc(roots):
    H = log_disc(roots)
    var = variance_roots(roots)
    if var < 1e-15:
        return -np.inf
    n = len(roots)
    return H - n*(n-1)/2.0 * np.log(np.sqrt(var))

# Candidate 4: "Entropy power" N_n = exp(2H'/(n(n-1)))
def entropy_power(roots):
    H = log_disc(roots)
    n = len(roots)
    return np.exp(2.0 * H / (n * (n-1)))

# Candidate 5: log Φ_n (negative, since Φ>0)
def neg_log_phi(roots):
    return -np.log(phi_n(roots))

# Candidate 6: H'_n / Φ_n^{1/2}
def H_over_sqrt_phi(roots):
    return log_disc(roots) / np.sqrt(phi_n(roots))

# Candidate 7: log det Vandermonde = H'_n (raw, for reference)
def raw_logdisc(roots):
    return log_disc(roots)

# Candidate 8: Normalized by number of pairs
def H_per_pair(roots):
    n = len(roots)
    return log_disc(roots) / (n*(n-1)/2.0)

# Candidate 9: -1/H'_n
def neg_inv_logdisc(roots):
    H = log_disc(roots)
    if H <= 0:
        return -np.inf
    return -1.0 / H

# Candidate 10: Φ_n * exp(-2H'_n/(n(n-1)))  (Fisher/entropy-power ratio)
def phi_over_N(roots):
    n = len(roots)
    H = log_disc(roots)
    return phi_n(roots) * np.exp(-2.0 * H / (n*(n-1)))

# Run tests
test_functional_superadditivity("1/Φ_n (Stam functional)", inv_phi)
test_functional_superadditivity("log(1/Φ_n)", log_inv_phi)
test_functional_superadditivity("Var-normalized H'_n", var_normalized_logdisc)
test_functional_superadditivity("Entropy power N_n = exp(2H'/(n(n-1)))", entropy_power)
test_functional_superadditivity("-log Φ_n", neg_log_phi)
test_functional_superadditivity("H'_n (raw, reference)", raw_logdisc)
test_functional_superadditivity("H'_n per pair", H_per_pair)


# Special test: Is the entropy power CONCAVE?
# In the classical case, EPI says N(X+Y) ≥ N(X) + N(Y)
# If the finite N_n is superadditive under ⊞_n, that's the finite EPI!

print(f"\n{'='*70}")
print("ENTROPY POWER INEQUALITY (EPI) TEST")
print(f"{'='*70}")
print("N_n(p⊞q) ≥ N_n(p) + N_n(q) where N_n = exp(2H'/(n(n-1)))")

for n in [3, 4, 5, 6]:
    violations = 0
    total = 0
    for _ in range(1000):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        conv = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(conv):
            continue

        r = coeffs_to_roots(conv)
        try:
            N_pq = entropy_power(r)
            N_p = entropy_power(p)
            N_q = entropy_power(q)
        except:
            continue

        if any(np.isinf([N_pq, N_p, N_q])):
            continue

        total += 1
        if N_pq < N_p + N_q - 1e-8:
            violations += 1

    print(f"  n={n}: {violations}/{total} violations ({100*violations/total:.1f}%)")
