#!/usr/bin/env python3
"""Investigate the finite de Bruijn identity for ⊞_n.

Discovery: d/dt [Σ_{i<j} log|γ_i - γ_j|](p_t) = ((n+1)/12) · Φ_n(p_t)
where p_t = p ⊞_n h_t, h_t = equally spaced roots at scale √t.

This script:
1. Verifies with alternative heat kernels (Hermite, Chebyshev, random)
2. Investigates root dynamics dγ_k/dt via implicit differentiation
3. Tests whether the constant depends on the kernel's variance growth rate
"""

import numpy as np
from math import factorial
import sys

np.random.seed(2026)
sys.stdout.reconfigure(line_buffering=True)


# ── Core functions ──────────────────────────────────────────────────────

def phi_n(roots):
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def log_disc_unnorm(roots):
    """H' = Σ_{i<j} log|λ_i - λ_j|"""
    n = len(roots)
    H = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(roots[i] - roots[j])
            if gap < 1e-15:
                return -np.inf
            H += np.log(gap)
    return H


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


def is_real_rooted(coeffs, tol=1e-8):
    poly = np.concatenate([[1.0], coeffs])
    r = np.roots(poly)
    return np.max(np.abs(r.imag)) < tol


# ── Heat kernels ───────────────────────────────────────────────────────

def equally_spaced_roots(n, t):
    """Roots at √t · (-(n-1)/2, ..., (n-1)/2)."""
    return np.sqrt(t) * np.linspace(-(n-1)/2, (n-1)/2, n)


def hermite_roots(n, t):
    """Roots at √t × zeros of probabilist's Hermite polynomial He_n."""
    from numpy.polynomial.hermite_e import hermeroots
    base = sorted(hermeroots([0]*n + [1]))
    return np.array(base) * np.sqrt(t)


def chebyshev_roots(n, t):
    """Roots at √t × zeros of T_n (Chebyshev first kind)."""
    k = np.arange(1, n+1)
    base = np.cos((2*k - 1) * np.pi / (2*n))
    return np.sort(base) * np.sqrt(t)


def variance_of_roots(roots):
    """Variance of empirical root distribution = (1/n)Σ r_i² - ((1/n)Σ r_i)²."""
    return np.var(roots)


# ── De Bruijn test ─────────────────────────────────────────────────────

def test_debruijn_kernel(n, kernel_func, kernel_name, n_points=20, dt=0.005):
    """Test de Bruijn identity with a specific heat kernel."""
    print(f"\n  Kernel: {kernel_name}")

    # Compute variance growth rate
    t0, t1 = 1.0, 1.0 + dt
    var0 = variance_of_roots(kernel_func(n, t0))
    var1 = variance_of_roots(kernel_func(n, t1))
    dvar_dt = (var1 - var0) / dt
    print(f"  dVar/dt = {dvar_dt:.6f}")

    ratios = []
    for point in range(n_points):
        a_coeffs, _ = random_real_rooted(n, spread=2.0)

        for t in [0.3, 0.5, 1.0, 2.0]:
            h_roots = kernel_func(n, t)
            h_roots_p = kernel_func(n, t + dt)
            h_roots_m = kernel_func(n, t - dt)

            h_coeffs = roots_to_coeffs(h_roots)
            h_coeffs_p = roots_to_coeffs(h_roots_p)
            h_coeffs_m = roots_to_coeffs(h_roots_m)

            c = mss_convolve(a_coeffs, h_coeffs, n)
            cp = mss_convolve(a_coeffs, h_coeffs_p, n)
            cm = mss_convolve(a_coeffs, h_coeffs_m, n)

            if not (is_real_rooted(c) and is_real_rooted(cp) and is_real_rooted(cm)):
                continue

            roots_c = coeffs_to_roots(c)
            roots_cp = coeffs_to_roots(cp)
            roots_cm = coeffs_to_roots(cm)

            H = log_disc_unnorm(roots_c)
            Hp = log_disc_unnorm(roots_cp)
            Hm = log_disc_unnorm(roots_cm)

            if any(np.isinf([H, Hp, Hm])):
                continue

            dH_dt = (Hp - Hm) / (2 * dt)
            Phi = phi_n(roots_c)

            if Phi < 1e-10:
                continue

            ratio = dH_dt / Phi
            ratios.append(ratio)

    if ratios:
        ratios = np.array(ratios)
        c_mean = ratios.mean()
        c_std = ratios.std()
        cv = c_std / abs(c_mean) if abs(c_mean) > 1e-10 else np.inf

        # Predicted constant
        predicted = (n + 1) / 12.0
        normalized = c_mean / dvar_dt if abs(dvar_dt) > 1e-10 else np.inf

        print(f"  c' = dH'/dt / Φ_n: mean={c_mean:.8f}, std={c_std:.6f}, CV={cv:.4f}")
        print(f"  Predicted (n+1)/12 = {predicted:.8f}")
        print(f"  Ratio c'/predicted = {c_mean/predicted:.6f}")
        print(f"  Ratio c'/(dVar/dt) = {normalized:.6f}")

        return {
            "kernel": kernel_name, "n": n,
            "c_mean": c_mean, "c_std": c_std, "cv": cv,
            "predicted": predicted,
            "dvar_dt": dvar_dt,
            "c_over_predicted": c_mean / predicted,
            "c_over_dvar": normalized
        }
    else:
        print("  No valid data.")
        return None


def main():
    print("=" * 70)
    print("FINITE DE BRUIJN IDENTITY: KERNEL DEPENDENCE")
    print("=" * 70)

    kernels = [
        (equally_spaced_roots, "Equally-spaced"),
        (hermite_roots, "Hermite"),
        (chebyshev_roots, "Chebyshev"),
    ]

    all_results = []

    for n in [3, 4, 5]:
        print(f"\n{'#'*70}")
        print(f"# n = {n}")
        print(f"{'#'*70}")

        for kernel_func, kernel_name in kernels:
            result = test_debruijn_kernel(n, kernel_func, kernel_name, n_points=15)
            if result:
                all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Is the constant kernel-dependent?")
    print(f"{'='*70}")
    print(f"{'n':>3} {'Kernel':<18} {'c_unnorm':>12} {'(n+1)/12':>10} {'c/pred':>8} {'c/dVar':>8}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['n']:3d} {r['kernel']:<18} {r['c_mean']:12.8f} {r['predicted']:10.6f} "
              f"{r['c_over_predicted']:8.4f} {r['c_over_dvar']:8.4f}")


if __name__ == '__main__':
    main()
