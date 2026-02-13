#!/usr/bin/env python3
"""Numerical verification of the Conditional Finite Stam Theorem.

Tests:
1. Jensen condition (A1): Φ_n(p ⊞_n q) ≤ E_U[Φ_n(A+UBU*)]
2. Sample-level Stam (A2): E_U[Φ_n(A+UBU*)] ≤ harmonic mean
3. De Bruijn identity (B1): d/dt H_n(p_t) ≈ -c · Φ_n(p_t)
4. Log-discriminant superadditivity (B2): H_n(p ⊞_n q) ≥ H_n(p) + H_n(q)

Results saved to data/first-proof/problem4-conditional-tests.jsonl
"""

import numpy as np
from scipy.stats import unitary_group
from math import factorial
import json
import sys
import os
import time

np.random.seed(2026)
sys.stdout.reconfigure(line_buffering=True)

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
    "../data/first-proof/problem4-conditional-tests.jsonl")


# ── Core functions ──────────────────────────────────────────────────────

def phi_n(roots):
    """Φ_n = Σ_i (Σ_{j≠i} 1/(λ_i - λ_j))²."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def log_discriminant(roots):
    """H_n = (2/n²) Σ_{i<j} log|λ_i - λ_j|."""
    n = len(roots)
    H = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(roots[i] - roots[j])
            if gap < 1e-15:
                return -np.inf
            H += np.log(gap)
    return 2.0 * H / (n * n)


def mss_convolve(a_coeffs, b_coeffs, n):
    """MSS finite free additive convolution on coefficients.
    a_coeffs, b_coeffs: [a_1, ..., a_n] (excluding leading 1)."""
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


def coeffs_to_roots(coeffs):
    """[a_1, ..., a_n] -> sorted real roots."""
    poly = np.concatenate([[1.0], coeffs])
    r = np.roots(poly)
    return np.sort(r.real)


def roots_to_coeffs(roots):
    """Roots -> [a_1, ..., a_n]."""
    return np.poly(roots)[1:]


def random_real_rooted(n, spread=1.0):
    """Generate a random monic real-rooted degree-n polynomial."""
    roots = np.sort(np.random.randn(n) * spread)
    return roots_to_coeffs(roots), roots


def is_real_rooted(coeffs, tol=1e-8):
    """Check if polynomial has all real roots."""
    poly = np.concatenate([[1.0], coeffs])
    r = np.roots(poly)
    return np.max(np.abs(r.imag)) < tol


# ── Test 1: Jensen condition (A1) ──────────────────────────────────────

def test_jensen(n, n_pairs=50, n_samples=500):
    """Test: Φ_n(p ⊞_n q) ≤ E_U[Φ_n(A + UBU*)]."""
    print(f"\n{'='*70}")
    print(f"TEST 1 (A1): Jensen condition at n={n}")
    print(f"  {n_pairs} polynomial pairs, {n_samples} Haar samples each")
    print(f"{'='*70}")

    ratios = []
    violations = 0

    for pair in range(n_pairs):
        a_coeffs, roots_a = random_real_rooted(n)
        b_coeffs, roots_b = random_real_rooted(n)

        # Convolution
        c_coeffs = mss_convolve(a_coeffs, b_coeffs, n)
        if not is_real_rooted(c_coeffs):
            continue
        roots_c = coeffs_to_roots(c_coeffs)
        phi_conv = phi_n(roots_c)

        # Haar sampling
        A = np.diag(roots_a)
        B = np.diag(roots_b)
        phi_samples = []

        for _ in range(n_samples):
            U = unitary_group.rvs(n)
            C = A + U @ B @ U.conj().T
            eigs = np.sort(np.linalg.eigvalsh(C))
            try:
                phi_samples.append(phi_n(eigs))
            except (ZeroDivisionError, FloatingPointError):
                continue

        if len(phi_samples) < n_samples // 2:
            continue

        E_phi = np.mean(phi_samples)
        ratio = phi_conv / E_phi  # should be ≤ 1

        ratios.append(ratio)
        if ratio > 1.0 + 1e-6:
            violations += 1

        if pair < 5 or ratio > 0.99:
            print(f"  pair {pair:3d}: Φ(conv)={phi_conv:.4f}, E[Φ]={E_phi:.4f}, "
                  f"ratio={ratio:.6f} {'VIOLATION' if ratio > 1.0 + 1e-6 else 'ok'}")

    ratios = np.array(ratios)
    print(f"\n  Summary: {len(ratios)} valid pairs")
    print(f"  ratio: min={ratios.min():.6f}, max={ratios.max():.6f}, "
          f"mean={ratios.mean():.6f}, std={ratios.std():.6f}")
    print(f"  violations: {violations}/{len(ratios)}")

    return {
        "test": "A1_jensen", "n": n, "n_pairs": len(ratios),
        "n_samples": n_samples, "violations": violations,
        "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max()),
        "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std()),
        "verdict": "PASS" if violations == 0 else "FAIL"
    }


# ── Test 2: Sample-level Stam (A2) ────────────────────────────────────

def test_sample_stam(n, n_pairs=50, n_samples=500):
    """Test: E_U[Φ_n(A+UBU*)] ≤ Φ(p)·Φ(q)/(Φ(p)+Φ(q))."""
    print(f"\n{'='*70}")
    print(f"TEST 2 (A2): Sample-level Stam at n={n}")
    print(f"{'='*70}")

    ratios = []
    violations = 0

    for pair in range(n_pairs):
        a_coeffs, roots_a = random_real_rooted(n)
        b_coeffs, roots_b = random_real_rooted(n)

        phi_p = phi_n(roots_a)
        phi_q = phi_n(roots_b)
        harmonic = phi_p * phi_q / (phi_p + phi_q)

        A = np.diag(roots_a)
        B = np.diag(roots_b)
        phi_samples = []

        for _ in range(n_samples):
            U = unitary_group.rvs(n)
            C = A + U @ B @ U.conj().T
            eigs = np.sort(np.linalg.eigvalsh(C))
            try:
                phi_samples.append(phi_n(eigs))
            except (ZeroDivisionError, FloatingPointError):
                continue

        if len(phi_samples) < n_samples // 2:
            continue

        E_phi = np.mean(phi_samples)
        ratio = E_phi / harmonic  # should be ≤ 1

        ratios.append(ratio)
        if ratio > 1.0 + 1e-6:
            violations += 1

        if pair < 5 or ratio > 0.99:
            print(f"  pair {pair:3d}: E[Φ]={E_phi:.4f}, HM={harmonic:.4f}, "
                  f"ratio={ratio:.6f} {'VIOLATION' if ratio > 1.0 + 1e-6 else 'ok'}")

    ratios = np.array(ratios)
    print(f"\n  Summary: {len(ratios)} valid pairs")
    print(f"  ratio: min={ratios.min():.6f}, max={ratios.max():.6f}, "
          f"mean={ratios.mean():.6f}, std={ratios.std():.6f}")
    print(f"  violations: {violations}/{len(ratios)}")

    return {
        "test": "A2_sample_stam", "n": n, "n_pairs": len(ratios),
        "n_samples": n_samples, "violations": violations,
        "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max()),
        "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std()),
        "verdict": "PASS" if violations == 0 else "FAIL"
    }


# ── Test 3: De Bruijn identity (B1) ───────────────────────────────────

def test_de_bruijn(n, n_points=30, dt=0.005):
    """Test: d/dt H_n(p_t) ≈ -c · Φ_n(p_t) where p_t = p ⊞_n h_t.

    h_t = polynomial with equally spaced roots at scale √t.
    """
    print(f"\n{'='*70}")
    print(f"TEST 3 (B1): De Bruijn identity at n={n}")
    print(f"{'='*70}")

    def heat_kernel_coeffs(n, t):
        """h_t with roots √t · (-k, ..., k) for n odd, or similar."""
        if t < 1e-15:
            # h_0 = x^n
            return np.zeros(n)
        spacing = np.sqrt(t)
        roots = spacing * np.linspace(-(n-1)/2, (n-1)/2, n)
        return roots_to_coeffs(roots)

    ratios_by_point = []  # ratio = -(dH/dt) / Φ_n for each (p, t)

    for point in range(n_points):
        a_coeffs, roots_a = random_real_rooted(n, spread=2.0)

        point_ratios = []
        for t in [0.1, 0.3, 0.5, 1.0, 2.0]:
            h_coeffs_t = heat_kernel_coeffs(n, t)
            h_coeffs_tp = heat_kernel_coeffs(n, t + dt)
            h_coeffs_tm = heat_kernel_coeffs(n, t - dt)

            c_t = mss_convolve(a_coeffs, h_coeffs_t, n)
            c_tp = mss_convolve(a_coeffs, h_coeffs_tp, n)
            c_tm = mss_convolve(a_coeffs, h_coeffs_tm, n)

            if not (is_real_rooted(c_t) and is_real_rooted(c_tp) and is_real_rooted(c_tm)):
                continue

            roots_t = coeffs_to_roots(c_t)
            roots_tp = coeffs_to_roots(c_tp)
            roots_tm = coeffs_to_roots(c_tm)

            H_t = log_discriminant(roots_t)
            H_tp = log_discriminant(roots_tp)
            H_tm = log_discriminant(roots_tm)

            if any(np.isinf([H_t, H_tp, H_tm])):
                continue

            dH_dt = (H_tp - H_tm) / (2 * dt)
            Phi_t = phi_n(roots_t)

            if Phi_t < 1e-10:
                continue

            ratio = -dH_dt / Phi_t
            point_ratios.append(ratio)

        if point_ratios:
            ratios_by_point.append(point_ratios)
            if point < 5:
                print(f"  point {point}: ratios across t = {[f'{r:.6f}' for r in point_ratios]}")

    # Check: is ratio approximately constant?
    all_ratios = [r for pr in ratios_by_point for r in pr]
    if all_ratios:
        all_ratios = np.array(all_ratios)
        print(f"\n  All ratios: min={all_ratios.min():.6f}, max={all_ratios.max():.6f}, "
              f"mean={all_ratios.mean():.6f}, std={all_ratios.std():.6f}")
        cv = all_ratios.std() / abs(all_ratios.mean()) if abs(all_ratios.mean()) > 1e-10 else np.inf
        print(f"  CV (coefficient of variation) = {cv:.4f}")
        if cv < 0.1:
            print(f"  CONSISTENT with de Bruijn: c ≈ {all_ratios.mean():.6f}")
        else:
            print(f"  NOT consistent with de Bruijn (ratio varies too much)")

        return {
            "test": "B1_de_bruijn", "n": n, "n_points": len(ratios_by_point),
            "ratio_min": float(all_ratios.min()), "ratio_max": float(all_ratios.max()),
            "ratio_mean": float(all_ratios.mean()), "ratio_std": float(all_ratios.std()),
            "cv": float(cv),
            "verdict": "CONSISTENT" if cv < 0.1 else "INCONSISTENT"
        }
    else:
        print("  No valid data points.")
        return {"test": "B1_de_bruijn", "n": n, "verdict": "NO_DATA"}


# ── Test 4: Log-discriminant superadditivity (B2) ─────────────────────

def test_log_disc_superadditivity(n, n_pairs=200):
    """Test: H_n(p ⊞_n q) ≥ H_n(p) + H_n(q)."""
    print(f"\n{'='*70}")
    print(f"TEST 4 (B2): Log-discriminant superadditivity at n={n}")
    print(f"{'='*70}")

    surpluses = []
    violations = 0

    for pair in range(n_pairs):
        a_coeffs, roots_a = random_real_rooted(n)
        b_coeffs, roots_b = random_real_rooted(n)

        c_coeffs = mss_convolve(a_coeffs, b_coeffs, n)
        if not is_real_rooted(c_coeffs):
            continue
        roots_c = coeffs_to_roots(c_coeffs)

        H_p = log_discriminant(roots_a)
        H_q = log_discriminant(roots_b)
        H_conv = log_discriminant(roots_c)

        if any(np.isinf([H_p, H_q, H_conv])):
            continue

        surplus = H_conv - H_p - H_q
        surpluses.append(surplus)

        if surplus < -1e-8:
            violations += 1
            if violations <= 5:
                print(f"  VIOLATION at pair {pair}: surplus={surplus:.8f}")
                print(f"    roots_a={roots_a}")
                print(f"    roots_b={roots_b}")

    surpluses = np.array(surpluses)
    print(f"\n  Summary: {len(surpluses)} valid pairs")
    print(f"  surplus: min={surpluses.min():.8f}, max={surpluses.max():.8f}, "
          f"mean={surpluses.mean():.6f}")
    print(f"  violations (surplus < 0): {violations}/{len(surpluses)}")

    return {
        "test": "B2_log_disc_superadd", "n": n, "n_pairs": len(surpluses),
        "violations": violations,
        "surplus_min": float(surpluses.min()), "surplus_max": float(surpluses.max()),
        "surplus_mean": float(surpluses.mean()),
        "verdict": "PASS" if violations == 0 else "FAIL"
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    results = []

    for n in [3, 4, 5]:
        print(f"\n{'#'*70}")
        print(f"# n = {n}")
        print(f"{'#'*70}")

        t0 = time.time()
        results.append(test_jensen(n, n_pairs=30, n_samples=300))
        results.append(test_sample_stam(n, n_pairs=30, n_samples=300))
        results.append(test_de_bruijn(n, n_points=20))
        results.append(test_log_disc_superadditivity(n, n_pairs=100))
        print(f"\n  n={n} completed in {time.time()-t0:.1f}s")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        v = r.get("verdict", "?")
        tag = "✓" if v in ("PASS", "CONSISTENT") else "✗" if v in ("FAIL", "INCONSISTENT") else "?"
        print(f"  {tag} {r['test']} (n={r['n']}): {v}")


if __name__ == '__main__':
    main()
