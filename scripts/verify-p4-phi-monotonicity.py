#!/usr/bin/env python3
"""Test: Is 1/Φ_n(p_t) non-decreasing along the Coulomb flow p_t = p ⊞_n He_t?

If yes, combined with:
  - De Bruijn identity: d/dt H'_n = Φ_n  (PROVED)
  - Log-disc superadditivity: H'_n(p ⊞_n q) ≥ H'_n(p) + H'_n(q)  (holds n≥4)
this would give the Stam inequality for n ≥ 4.

Also tests whether Φ_n itself is monotone (decreasing) along the flow,
and computes d/dt Φ_n(p_t) and d/dt[1/Φ_n(p_t)] numerically.
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


def hermite_roots(n, t):
    from numpy.polynomial.hermite_e import hermeroots
    base = sorted(hermeroots([0]*n + [1]))
    return np.array(base) * np.sqrt(t)


def test_phi_monotonicity(n, n_tests=30, n_timepoints=20):
    """Test monotonicity of Φ_n and 1/Φ_n along Coulomb flow."""
    print(f"\n{'='*70}")
    print(f"n = {n}: Monotonicity of Φ_n along p_t = p ⊞_n He_t")
    print(f"{'='*70}")

    phi_increasing = 0  # Φ_n increasing (bad for 1/Φ_n monotonicity)
    phi_decreasing = 0  # Φ_n decreasing (good for 1/Φ_n monotonicity)
    violations = 0
    total_pairs = 0

    all_phi_trajectories = []

    for test in range(n_tests):
        p_roots = np.sort(np.random.randn(n) * 2)
        a_coeffs = np.poly(p_roots)[1:]

        t_values = np.linspace(0.1, 5.0, n_timepoints)
        phi_values = []

        for t in t_values:
            h_roots = hermite_roots(n, t)
            c = mss_convolve(a_coeffs, np.poly(h_roots)[1:], n)
            poly = np.concatenate([[1.0], c])
            r = np.roots(poly)
            if np.max(np.abs(r.imag)) > 1e-6:
                phi_values.append(None)
                continue
            roots = np.sort(r.real)
            phi_values.append(phi_n(roots))

        # Check monotonicity
        valid_phis = [(t, p) for t, p in zip(t_values, phi_values) if p is not None]
        if len(valid_phis) < 2:
            continue

        all_phi_trajectories.append(valid_phis)

        for i in range(len(valid_phis) - 1):
            t1, p1 = valid_phis[i]
            t2, p2 = valid_phis[i + 1]
            total_pairs += 1
            if p2 > p1 * (1 + 1e-8):  # Φ increasing
                phi_increasing += 1
            elif p1 > p2 * (1 + 1e-8):  # Φ decreasing
                phi_decreasing += 1

        # Check if Φ_n is monotonically decreasing for this trajectory
        phis = [p for _, p in valid_phis]
        is_decreasing = all(phis[i] >= phis[i+1] * (1 - 1e-6) for i in range(len(phis)-1))
        is_increasing = all(phis[i+1] >= phis[i] * (1 - 1e-6) for i in range(len(phis)-1))

        if test < 5:
            print(f"\n  Test {test}: Φ_n trajectory (t=0.1 to 5.0)")
            print(f"    Φ(0.1) = {phis[0]:.6f}, Φ(5.0) = {phis[-1]:.6f}, "
                  f"ratio = {phis[-1]/phis[0]:.4f}")
            print(f"    Monotone decreasing: {is_decreasing}")
            print(f"    Monotone increasing: {is_increasing}")

    print(f"\n  SUMMARY over {n_tests} tests, {total_pairs} consecutive pairs:")
    print(f"    Φ_n increasing steps: {phi_increasing}/{total_pairs} "
          f"({100*phi_increasing/total_pairs:.1f}%)")
    print(f"    Φ_n decreasing steps: {phi_decreasing}/{total_pairs} "
          f"({100*phi_decreasing/total_pairs:.1f}%)")
    print(f"    Flat steps: {total_pairs - phi_increasing - phi_decreasing}/{total_pairs}")

    # Check 1/Φ_n monotonicity
    inv_phi_increasing = phi_decreasing  # 1/Φ increases when Φ decreases
    print(f"\n    1/Φ_n increasing steps: {inv_phi_increasing}/{total_pairs} "
          f"({100*inv_phi_increasing/total_pairs:.1f}%)")
    print(f"    1/Φ_n monotonically non-decreasing: "
          f"{'YES' if inv_phi_increasing == total_pairs else 'NO'}")

    return phi_decreasing == total_pairs


def test_dphi_dt_formula(n, n_tests=20):
    """Compute d/dt Φ_n along the Coulomb flow and see if it has a sign."""
    print(f"\n{'='*70}")
    print(f"n = {n}: d/dt Φ_n along Coulomb flow")
    print(f"{'='*70}")

    dt = 0.001
    dphi_values = []

    for test in range(n_tests):
        p_roots = np.sort(np.random.randn(n) * 2)
        a_coeffs = np.poly(p_roots)[1:]

        for t in [0.5, 1.0, 2.0, 3.0]:
            h_t = hermite_roots(n, t)
            h_p = hermite_roots(n, t + dt)
            h_m = hermite_roots(n, t - dt)

            c_t = mss_convolve(a_coeffs, np.poly(h_t)[1:], n)
            c_p = mss_convolve(a_coeffs, np.poly(h_p)[1:], n)
            c_m = mss_convolve(a_coeffs, np.poly(h_m)[1:], n)

            poly_t = np.concatenate([[1.0], c_t])
            poly_p = np.concatenate([[1.0], c_p])
            poly_m = np.concatenate([[1.0], c_m])

            # Check real-rootedness
            r_t = np.roots(poly_t)
            r_p = np.roots(poly_p)
            r_m = np.roots(poly_m)
            if (np.max(np.abs(r_t.imag)) > 1e-6 or
                np.max(np.abs(r_p.imag)) > 1e-6 or
                np.max(np.abs(r_m.imag)) > 1e-6):
                continue

            roots_t = np.sort(r_t.real)
            roots_p = np.sort(r_p.real)
            roots_m = np.sort(r_m.real)

            phi_t = phi_n(roots_t)
            phi_p = phi_n(roots_p)
            phi_m = phi_n(roots_m)

            dphi_dt = (phi_p - phi_m) / (2 * dt)
            dphi_values.append((t, phi_t, dphi_dt))

    if dphi_values:
        dphi_arr = np.array([d for _, _, d in dphi_values])
        phi_arr = np.array([p for _, p, _ in dphi_values])
        ratios = dphi_arr / phi_arr

        n_neg = np.sum(dphi_arr < -1e-8)
        n_pos = np.sum(dphi_arr > 1e-8)
        print(f"  dΦ_n/dt < 0: {n_neg}/{len(dphi_arr)} ({100*n_neg/len(dphi_arr):.1f}%)")
        print(f"  dΦ_n/dt > 0: {n_pos}/{len(dphi_arr)} ({100*n_pos/len(dphi_arr):.1f}%)")
        print(f"  (dΦ/dt)/Φ: mean={ratios.mean():.6f}, min={ratios.min():.6f}, "
              f"max={ratios.max():.6f}")

        # Also check d/dt[1/Φ_n]
        dinv_phi = -dphi_arr / phi_arr**2
        n_inv_pos = np.sum(dinv_phi > 1e-12)
        print(f"\n  d(1/Φ_n)/dt > 0: {n_inv_pos}/{len(dinv_phi)} "
              f"({100*n_inv_pos/len(dinv_phi):.1f}%)")

        if n_inv_pos == len(dinv_phi):
            print("  *** 1/Φ_n is INCREASING along the Coulomb flow! ***")
        elif n_neg == len(dphi_arr):
            print("  *** Φ_n is DECREASING along the Coulomb flow! ***")


def test_phi_analytic_derivative(n, n_tests=20):
    """Compute dΦ_n/dt analytically using the Coulomb flow.

    Since γ_k' = S_k, we can compute:
    dΦ_n/dt = d/dt Σ_k S_k² = 2 Σ_k S_k · dS_k/dt
    where dS_k/dt = Σ_{j≠k} -(γ_k' - γ_j')/(γ_k - γ_j)²
                  = -Σ_{j≠k} (S_k - S_j)/(γ_k - γ_j)²
    """
    print(f"\n{'='*70}")
    print(f"n = {n}: Analytic dΦ_n/dt via Coulomb flow equations")
    print(f"{'='*70}")

    dphi_values = []

    for test in range(n_tests):
        roots = np.sort(np.random.randn(n) * 2)
        while np.min(np.diff(roots)) < 0.3:
            roots = np.sort(np.random.randn(n) * 2)

        S = score_field(roots)

        # dS_k/dt = Σ_{j≠k} -(S_k - S_j)/(γ_k - γ_j)²
        dS = np.zeros(n)
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                gap = roots[k] - roots[j]
                dS[k] -= (S[k] - S[j]) / gap**2

        # dΦ/dt = 2 Σ_k S_k · dS_k/dt
        dphi = 2 * np.dot(S, dS)
        phi_val = np.dot(S, S)
        dphi_values.append((phi_val, dphi, dphi/phi_val))

    if dphi_values:
        dphis = np.array([d for _, d, _ in dphi_values])
        ratios = np.array([r for _, _, r in dphi_values])

        n_neg = np.sum(dphis < -1e-10)
        n_pos = np.sum(dphis > 1e-10)
        print(f"  dΦ_n/dt < 0: {n_neg}/{len(dphis)} ({100*n_neg/len(dphis):.1f}%)")
        print(f"  dΦ_n/dt > 0: {n_pos}/{len(dphis)} ({100*n_pos/len(dphis):.1f}%)")
        print(f"  (dΦ/dt)/Φ: mean={ratios.mean():.6f}, std={ratios.std():.6f}")
        print(f"  (dΦ/dt)/Φ: min={ratios.min():.6f}, max={ratios.max():.6f}")

        if n_neg == len(dphis):
            print("  *** Φ_n is strictly decreasing → 1/Φ_n strictly increasing ***")
        elif n_neg > 0 and n_pos > 0:
            print("  *** Φ_n changes sign — NOT monotone ***")
        elif n_pos == len(dphis):
            print("  *** Φ_n is strictly increasing ***")


def main():
    for n in [3, 4, 5, 6]:
        test_phi_monotonicity(n, n_tests=30, n_timepoints=30)

    print("\n" + "=" * 70)
    print("DETAILED d/dt ANALYSIS")
    print("=" * 70)

    for n in [3, 4, 5, 6]:
        test_dphi_dt_formula(n)

    print("\n" + "=" * 70)
    print("ANALYTIC d/dt VIA COULOMB FLOW EQUATIONS")
    print("=" * 70)

    for n in [3, 4, 5, 6]:
        test_phi_analytic_derivative(n, n_tests=50)


if __name__ == '__main__':
    main()
