#!/usr/bin/env python3
"""Test: do roots of p ⊞_n h_t evolve as dγ_k/dt = c · S_k(γ)?

If true, the de Bruijn identity follows immediately:
  dH'/dt = Σ_k γ_k' · S_k = c · Σ_k S_k² = c · Φ_n.

With Hermite kernel: c should be 1 (since dVar/dt = n-1 and constant = 1/(n-1)·(n-1) = 1).
"""

import numpy as np
from math import factorial
import sys

np.random.seed(2026)
sys.stdout.reconfigure(line_buffering=True)


def phi_n(roots):
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def score_field(roots):
    """S_k = Σ_{j≠k} 1/(γ_k - γ_j) for each k."""
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
    return S


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


def roots_to_coeffs(roots):
    return np.poly(roots)[1:]


def coeffs_to_roots(coeffs):
    poly = np.concatenate([[1.0], coeffs])
    return np.sort(np.roots(poly).real)


def random_real_rooted(n, spread=1.0):
    roots = np.sort(np.random.randn(n) * spread)
    return roots_to_coeffs(roots), roots


def hermite_roots(n, t):
    from numpy.polynomial.hermite_e import hermeroots
    base = sorted(hermeroots([0]*n + [1]))
    return np.array(base) * np.sqrt(t)


def equally_spaced_roots(n, t):
    return np.sqrt(t) * np.linspace(-(n-1)/2, (n-1)/2, n)


def test_root_velocity(n, kernel_func, kernel_name, dt=0.001, n_tests=30):
    """Compare numerical dγ_k/dt with S_k(γ)."""
    print(f"\nn={n}, kernel={kernel_name}")
    print(f"{'test':>4} {'max|v - c·S|/|S|':>20} {'c (fitted)':>12} {'max|S|':>10}")

    fitted_c_values = []

    for test in range(n_tests):
        a_coeffs, _ = random_real_rooted(n, spread=2.0)
        t = 1.0

        h_roots_t = kernel_func(n, t)
        h_roots_p = kernel_func(n, t + dt)
        h_roots_m = kernel_func(n, t - dt)

        c_t = mss_convolve(a_coeffs, roots_to_coeffs(h_roots_t), n)
        c_p = mss_convolve(a_coeffs, roots_to_coeffs(h_roots_p), n)
        c_m = mss_convolve(a_coeffs, roots_to_coeffs(h_roots_m), n)

        poly_t = np.concatenate([[1.0], c_t])
        poly_p = np.concatenate([[1.0], c_p])
        poly_m = np.concatenate([[1.0], c_m])

        r_t = np.sort(np.roots(poly_t).real)
        r_p = np.sort(np.roots(poly_p).real)
        r_m = np.sort(np.roots(poly_m).real)

        if np.max(np.abs(np.roots(poly_t).imag)) > 1e-6:
            continue

        # Numerical velocity
        velocity = (r_p - r_m) / (2 * dt)

        # Score field at current roots
        S = score_field(r_t)

        # Fit c: minimize |velocity - c·S|²
        # c = (velocity · S) / (S · S)
        SS = np.dot(S, S)
        if SS < 1e-15:
            continue

        c_fit = np.dot(velocity, S) / SS

        # Residual
        residual = velocity - c_fit * S
        rel_error = np.max(np.abs(residual)) / np.max(np.abs(S))

        fitted_c_values.append(c_fit)

        if test < 10 or rel_error > 0.01:
            print(f"{test:4d} {rel_error:20.8f} {c_fit:12.8f} {np.max(np.abs(S)):10.4f}")

    if fitted_c_values:
        c_arr = np.array(fitted_c_values)
        print(f"\nFitted c: mean={c_arr.mean():.8f}, std={c_arr.std():.8f}, "
              f"CV={c_arr.std()/abs(c_arr.mean()):.6f}")

        # Expected: for Hermite kernel, c = 1.0
        # For equally-spaced: c = dVar/dt / (n-1) = (n²-1)/12 / (n-1) = (n+1)/12 / ... wait
        # The root velocity should be c · S_k where c = overall deBruijn constant = c'_n / Φ_n
        # No: from dH'/dt = Σ γ'_k S_k = c' Φ_n, if γ'_k = c_vel · S_k, then
        # c_vel · Σ S_k² = c' Φ_n, so c_vel = c'.
        # For Hermite: c' = 1.0
        # For equally-spaced: c' = (n+1)/12
        print(f"\nExpected c:")
        if kernel_name == "Hermite":
            print(f"  Hermite: c = 1.0")
        elif kernel_name == "Equally-spaced":
            print(f"  Equally-spaced: c = (n+1)/12 = {(n+1)/12.0:.8f}")


def main():
    for n in [3, 4, 5]:
        test_root_velocity(n, hermite_roots, "Hermite")
        test_root_velocity(n, equally_spaced_roots, "Equally-spaced")


if __name__ == '__main__':
    main()
