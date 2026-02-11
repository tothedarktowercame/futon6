#!/usr/bin/env python3
"""Problem 4 Lemma 3: Explore Haar/matrix identities for Phi_n.

Since p ⊞_n q = E_Q[chi(A + QBQ*)], we sample the unitary orbit and
look for identities relating Phi_n of individual realizations to
Phi_n of the expected polynomial (= p ⊞_n q).

Key questions:
1. What is E[Phi_n(A+QBQ*)] vs Phi_n(p⊞q)?
2. What is E[1/Phi_n(A+QBQ*)] vs 1/Phi_n(p⊞q)?
3. Is there a decomposition: 1/Phi_n(p⊞q) = g(A,B) that we can bound?
4. What is the relationship between Phi_n and the derivative p''(λ_i)/p'(λ_i)?
"""
import numpy as np
from scipy.stats import unitary_group
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
_mod = __import__("verify-p4-inequality")
phi_n = _mod.phi_n
finite_free_conv = _mod.finite_free_conv
roots_to_coeffs = _mod.roots_to_coeffs
coeffs_to_roots = _mod.coeffs_to_roots


def sample_haar_unitary(n):
    """Sample a Haar-random unitary matrix."""
    return unitary_group.rvs(n)


def eigenvalues_sorted(M):
    """Sorted real eigenvalues of Hermitian matrix M."""
    eigs = np.linalg.eigvalsh(M)
    return np.sort(eigs)


def make_distinct(roots, min_gap=0.02):
    roots = np.sort(roots.real)
    for i in range(1, len(roots)):
        if roots[i] - roots[i-1] < min_gap:
            roots[i] = roots[i-1] + min_gap
    return roots


def phi_n_safe(roots):
    """Compute Phi_n, returning nan on failure."""
    roots = make_distinct(roots)
    val = phi_n(roots)
    if val == float('inf') or val == 0:
        return float('nan')
    return val


def explore_haar_orbit(roots_p, roots_q, n_samples=2000):
    """Sample the unitary orbit A + QBQ* and compute statistics of Phi_n."""
    n = len(roots_p)
    A = np.diag(roots_p.astype(float))
    B = np.diag(roots_q.astype(float))

    # Compute p ⊞_n q
    coeffs_p = roots_to_coeffs(roots_p)
    coeffs_q = roots_to_coeffs(roots_q)
    coeffs_conv = finite_free_conv(coeffs_p, coeffs_q)
    roots_conv = coeffs_to_roots(coeffs_conv)

    phi_conv = phi_n_safe(roots_conv)
    phi_p = phi_n_safe(roots_p)
    phi_q = phi_n_safe(roots_q)

    if np.isnan(phi_conv) or np.isnan(phi_p) or np.isnan(phi_q):
        return None

    # Sample unitary orbit
    phi_samples = []
    inv_phi_samples = []

    for _ in range(n_samples):
        Q = sample_haar_unitary(n)
        M = A + Q @ B @ Q.conj().T
        eigs = eigenvalues_sorted(M)
        phi_val = phi_n_safe(eigs)
        if not np.isnan(phi_val):
            phi_samples.append(phi_val)
            inv_phi_samples.append(1.0 / phi_val)

    if len(phi_samples) < n_samples * 0.5:
        return None

    phi_samples = np.array(phi_samples)
    inv_phi_samples = np.array(inv_phi_samples)

    return {
        'phi_p': phi_p,
        'phi_q': phi_q,
        'phi_conv': phi_conv,
        'inv_phi_conv': 1.0 / phi_conv,
        'inv_phi_p': 1.0 / phi_p,
        'inv_phi_q': 1.0 / phi_q,
        'E_phi': np.mean(phi_samples),
        'E_inv_phi': np.mean(inv_phi_samples),
        'std_phi': np.std(phi_samples),
        'std_inv_phi': np.std(inv_phi_samples),
        'med_phi': np.median(phi_samples),
        'med_inv_phi': np.median(inv_phi_samples),
        'min_phi': np.min(phi_samples),
        'max_phi': np.max(phi_samples),
        'n_valid': len(phi_samples),
    }


def test_jensen_direction(n_roots, n_trials=200, n_samples=500):
    """Test which side of Jensen's inequality holds.

    If 1/Phi is concave on the orbit: E[1/Phi(M)] <= 1/Phi(conv)
    If 1/Phi is convex on the orbit:  E[1/Phi(M)] >= 1/Phi(conv)

    Note: 1/Phi(conv) is NOT E[1/Phi(M)] because roots of E[chi] ≠ E[roots].
    """
    n = n_roots
    E_inv_phi_larger = 0  # E[1/Phi] > 1/Phi(conv)
    E_inv_phi_smaller = 0  # E[1/Phi] < 1/Phi(conv)
    # Also test: does E[1/Phi(M)] >= 1/Phi(p) + 1/Phi(q)?
    orbit_superadd = 0
    total = 0

    ratios = []  # E[1/Phi(M)] / (1/Phi(p) + 1/Phi(q))

    for _ in range(n_trials):
        roots_p = make_distinct(np.random.randn(n) * 2)
        roots_q = make_distinct(np.random.randn(n) * 2)

        result = explore_haar_orbit(roots_p, roots_q, n_samples=n_samples)
        if result is None:
            continue

        total += 1
        if result['E_inv_phi'] > result['inv_phi_conv'] + 1e-10:
            E_inv_phi_larger += 1
        elif result['E_inv_phi'] < result['inv_phi_conv'] - 1e-10:
            E_inv_phi_smaller += 1

        rhs = result['inv_phi_p'] + result['inv_phi_q']
        if result['E_inv_phi'] >= rhs - 1e-10:
            orbit_superadd += 1
        ratios.append(result['E_inv_phi'] / rhs if rhs > 0 else float('nan'))

    ratios = np.array([r for r in ratios if not np.isnan(r)])
    return {
        'total': total,
        'E_inv_phi_larger': E_inv_phi_larger,
        'E_inv_phi_smaller': E_inv_phi_smaller,
        'orbit_superadd': orbit_superadd,
        'ratio_min': np.min(ratios) if len(ratios) > 0 else float('nan'),
        'ratio_mean': np.mean(ratios) if len(ratios) > 0 else float('nan'),
        'ratio_max': np.max(ratios) if len(ratios) > 0 else float('nan'),
    }


def test_harmonic_mean_identity(n_roots, n_trials=200, n_samples=500):
    """Test if 1/Phi_n(p⊞q) relates to the harmonic mean of Phi over the orbit.

    Hypothesis: maybe 1/Phi(conv) = E[something simple] ?

    Test various candidates:
    - E[1/Phi(M)]
    - 1/E[Phi(M)]
    - E[1/Phi(M)] - correction term
    """
    n = n_roots
    records = []

    for _ in range(n_trials):
        roots_p = make_distinct(np.random.randn(n) * 2)
        roots_q = make_distinct(np.random.randn(n) * 2)

        result = explore_haar_orbit(roots_p, roots_q, n_samples=n_samples)
        if result is None:
            continue

        records.append({
            'inv_phi_conv': result['inv_phi_conv'],
            'E_inv_phi': result['E_inv_phi'],
            'inv_E_phi': 1.0 / result['E_phi'],
            'inv_phi_p': result['inv_phi_p'],
            'inv_phi_q': result['inv_phi_q'],
            'phi_conv': result['phi_conv'],
            'E_phi': result['E_phi'],
        })

    if len(records) == 0:
        return None

    # Compute correlations and ratios
    inv_phi_conv = np.array([r['inv_phi_conv'] for r in records])
    E_inv_phi = np.array([r['E_inv_phi'] for r in records])
    inv_E_phi = np.array([r['inv_E_phi'] for r in records])
    inv_phi_p = np.array([r['inv_phi_p'] for r in records])
    inv_phi_q = np.array([r['inv_phi_q'] for r in records])
    sum_inv = inv_phi_p + inv_phi_q

    return {
        'n_trials': len(records),
        # Ratio: 1/Phi(conv) vs E[1/Phi(M)]
        'conv_vs_E_ratio_mean': np.mean(inv_phi_conv / E_inv_phi),
        'conv_vs_E_ratio_std': np.std(inv_phi_conv / E_inv_phi),
        # Ratio: 1/Phi(conv) vs 1/E[Phi(M)]
        'conv_vs_invE_ratio_mean': np.mean(inv_phi_conv / inv_E_phi),
        'conv_vs_invE_ratio_std': np.std(inv_phi_conv / inv_E_phi),
        # Correlation between 1/Phi(conv) and E[1/Phi(M)]
        'corr_conv_E': np.corrcoef(inv_phi_conv, E_inv_phi)[0, 1],
        # Correlation between 1/Phi(conv) and 1/E[Phi(M)]
        'corr_conv_invE': np.corrcoef(inv_phi_conv, inv_E_phi)[0, 1],
        # Does 1/Phi(conv) >= E[1/Phi(M)] always?
        'conv_geq_E_pct': 100 * np.mean(inv_phi_conv >= E_inv_phi - 1e-10),
        # Does E[1/Phi(M)] >= sum(1/Phi) always?
        'E_geq_sum_pct': 100 * np.mean(E_inv_phi >= sum_inv - 1e-10),
        # Does 1/Phi(conv) >= sum(1/Phi) always? (the target inequality)
        'conv_geq_sum_pct': 100 * np.mean(inv_phi_conv >= sum_inv - 1e-10),
    }


def detailed_example(roots_p, roots_q, n_samples=5000):
    """Detailed analysis of one specific case."""
    n = len(roots_p)
    print(f"\n  roots_p = {roots_p}")
    print(f"  roots_q = {roots_q}")

    result = explore_haar_orbit(roots_p, roots_q, n_samples=n_samples)
    if result is None:
        print("  (failed)")
        return

    print(f"  Phi(p) = {result['phi_p']:.6f},  1/Phi(p) = {result['inv_phi_p']:.6f}")
    print(f"  Phi(q) = {result['phi_q']:.6f},  1/Phi(q) = {result['inv_phi_q']:.6f}")
    print(f"  1/Phi(p) + 1/Phi(q) = {result['inv_phi_p'] + result['inv_phi_q']:.6f}")
    print(f"  ")
    print(f"  Phi(p⊞q) = {result['phi_conv']:.6f},  1/Phi(p⊞q) = {result['inv_phi_conv']:.6f}")
    print(f"  ratio = {result['inv_phi_conv'] / (result['inv_phi_p'] + result['inv_phi_q']):.6f}")
    print(f"  ")
    print(f"  E[Phi(A+QBQ*)]   = {result['E_phi']:.6f}  (std={result['std_phi']:.4f})")
    print(f"  E[1/Phi(A+QBQ*)] = {result['E_inv_phi']:.6f}  (std={result['std_inv_phi']:.4f})")
    print(f"  1/E[Phi]         = {1/result['E_phi']:.6f}")
    print(f"  ")
    print(f"  1/Phi(conv) / E[1/Phi] = {result['inv_phi_conv'] / result['E_inv_phi']:.6f}")
    print(f"  1/Phi(conv) / (1/E[Phi]) = {result['inv_phi_conv'] * result['E_phi']:.6f}")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("Problem 4 Lemma 3: Haar orbit exploration for Phi_n identity")
    print("=" * 70)

    print("\n--- Detailed examples ---")

    # n=3 symmetric
    print("\nExample 1: n=3 symmetric")
    detailed_example(
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
    )

    print("\nExample 2: n=3 asymmetric")
    detailed_example(
        np.array([-2.0, 0.0, 1.0]),
        np.array([-1.0, 0.5, 2.0]),
    )

    print("\nExample 3: n=4")
    detailed_example(
        np.array([-2.0, -0.5, 0.5, 2.0]),
        np.array([-1.5, -0.3, 0.8, 1.5]),
    )

    print("\n" + "=" * 70)
    print("Jensen direction test: E[1/Phi(M)] vs 1/Phi(p⊞q)")
    print("=" * 70)

    for n in [3, 4]:
        print(f"\n  n={n}:")
        r = test_jensen_direction(n, n_trials=200, n_samples=500)
        t = r['total']
        print(f"    {t} trials")
        print(f"    E[1/Phi] > 1/Phi(conv): {r['E_inv_phi_larger']} ({100*r['E_inv_phi_larger']/t:.1f}%)")
        print(f"    E[1/Phi] < 1/Phi(conv): {r['E_inv_phi_smaller']} ({100*r['E_inv_phi_smaller']/t:.1f}%)")
        print(f"    E[1/Phi(M)] >= 1/Phi(p)+1/Phi(q): {r['orbit_superadd']}/{t}")
        print(f"    ratio E[1/Phi]/(1/Phi(p)+1/Phi(q)): "
              f"min={r['ratio_min']:.4f} mean={r['ratio_mean']:.4f} max={r['ratio_max']:.4f}")

    print("\n" + "=" * 70)
    print("Identity search: what does 1/Phi(conv) equal?")
    print("=" * 70)

    for n in [3, 4]:
        print(f"\n  n={n}:")
        r = test_harmonic_mean_identity(n, n_trials=200, n_samples=500)
        if r is None:
            print("    (failed)")
            continue
        print(f"    {r['n_trials']} trials")
        print(f"    1/Phi(conv) / E[1/Phi(M)]:  mean={r['conv_vs_E_ratio_mean']:.4f} "
              f"std={r['conv_vs_E_ratio_std']:.4f}")
        print(f"    1/Phi(conv) / (1/E[Phi(M)]): mean={r['conv_vs_invE_ratio_mean']:.4f} "
              f"std={r['conv_vs_invE_ratio_std']:.4f}")
        print(f"    corr(1/Phi(conv), E[1/Phi]):  {r['corr_conv_E']:.4f}")
        print(f"    corr(1/Phi(conv), 1/E[Phi]):  {r['corr_conv_invE']:.4f}")
        print(f"    1/Phi(conv) >= E[1/Phi]:      {r['conv_geq_E_pct']:.1f}%")
        print(f"    E[1/Phi] >= sum(1/Phi):       {r['E_geq_sum_pct']:.1f}%")
        print(f"    1/Phi(conv) >= sum(1/Phi):    {r['conv_geq_sum_pct']:.1f}%  (TARGET)")
