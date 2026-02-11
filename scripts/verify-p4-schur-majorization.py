#!/usr/bin/env python3
"""Problem 4 next checks from proof-strategy-skeleton.md.

Tests:
1. Schur-convexity/concavity of F = 1/Phi_n on root vectors.
2. Interpolation path from lambda(p ⊞ q) to lambda(p)+lambda(q),
   testing F along the path and discrete second differences.
3. Submodularity test for F (diminishing returns under root perturbation).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
_mod = __import__("verify-p4-inequality")
phi_n = _mod.phi_n
finite_free_conv = _mod.finite_free_conv
roots_to_coeffs = _mod.roots_to_coeffs
coeffs_to_roots = _mod.coeffs_to_roots


def F(roots):
    """F = 1/Phi_n, the target functional."""
    phi = phi_n(roots)
    if phi == float('inf') or phi == 0:
        return float('nan')
    return 1.0 / phi


def make_distinct(roots, min_gap=0.01):
    """Ensure roots are distinct with minimum gap."""
    roots = np.sort(roots)
    for i in range(1, len(roots)):
        if roots[i] - roots[i-1] < min_gap:
            roots[i] = roots[i-1] + min_gap
    return roots


def random_doubly_stochastic(n, strength=0.3):
    """Generate a random doubly stochastic matrix near identity.

    D = (1-strength)*I + strength*P where P is a random permutation matrix
    mixed with identity. This gives Dx ≺ x (majorization).
    """
    # Use Birkhoff: convex combination of permutation matrices
    D = np.eye(n) * (1 - strength)
    # Add a random transposition component
    i, j = np.random.choice(n, 2, replace=False)
    P = np.eye(n)
    P[i, i] = P[j, j] = 0
    P[i, j] = P[j, i] = 1
    D += strength * P
    return D


def test_schur_convexity(n_roots, n_trials=2000):
    """Test whether F = 1/Phi_n is Schur-convex or Schur-concave.

    Schur-convex: x ≺ y => F(x) <= F(y)  (preserves majorization)
    Schur-concave: x ≺ y => F(x) >= F(y) (reverses majorization)

    We generate y (root vector), then x = Dy where D is doubly stochastic,
    so x ≺ y. Then check F(x) vs F(y).
    """
    n = n_roots
    schur_convex_violations = 0   # F(x) > F(y) when x ≺ y
    schur_concave_violations = 0  # F(x) < F(y) when x ≺ y
    total = 0

    for _ in range(n_trials):
        y = make_distinct(np.random.randn(n) * 2)
        D = random_doubly_stochastic(n, strength=np.random.uniform(0.1, 0.5))
        x = make_distinct(D @ y)  # x ≺ y

        Fx = F(x)
        Fy = F(y)
        if np.isnan(Fx) or np.isnan(Fy):
            continue

        total += 1
        if Fx > Fy + 1e-10:
            schur_convex_violations += 1
        if Fx < Fy - 1e-10:
            schur_concave_violations += 1

    return schur_convex_violations, schur_concave_violations, total


def test_interpolation_path(n_roots, n_trials=500):
    """Interpolate between lambda(p ⊞ q) and lambda(p) + lambda(q).

    MSS Corollary 1.7: lambda(p ⊞ q) ≺ lambda(p) + lambda(q).
    So the convolution roots are majorized by the componentwise sum.

    We interpolate: lambda(t) = (1-t)*lambda_conv + t*lambda_sum
    and check F along this path. If F is monotone decreasing, the
    majorization direction is consistent with Schur-concavity.

    Also compute discrete second differences to detect convexity/concavity
    of F along the path.
    """
    n = n_roots
    monotone_increasing = 0  # F increases from conv to sum
    monotone_decreasing = 0  # F decreases from conv to sum
    neither = 0
    second_diff_positive = 0  # path-convex
    second_diff_negative = 0  # path-concave
    second_diff_mixed = 0
    total = 0

    for _ in range(n_trials):
        roots_p = make_distinct(np.random.randn(n) * 2)
        roots_q = make_distinct(np.random.randn(n) * 2)

        coeffs_p = roots_to_coeffs(roots_p)
        coeffs_q = roots_to_coeffs(roots_q)
        coeffs_conv = finite_free_conv(coeffs_p, coeffs_q)
        roots_conv = coeffs_to_roots(coeffs_conv)

        if not np.all(np.isreal(np.roots(coeffs_conv))):
            continue

        # Componentwise sum of sorted root vectors
        roots_sum = np.sort(roots_p) + np.sort(roots_q)

        # Evaluate F along interpolation path
        ts = np.linspace(0, 1, 11)
        Fs = []
        valid = True
        for t in ts:
            roots_t = make_distinct((1 - t) * np.sort(roots_conv) + t * roots_sum)
            f = F(roots_t)
            if np.isnan(f):
                valid = False
                break
            Fs.append(f)

        if not valid or len(Fs) < 11:
            continue

        total += 1
        Fs = np.array(Fs)

        # Check monotonicity
        diffs = np.diff(Fs)
        if np.all(diffs >= -1e-10):
            monotone_increasing += 1
        elif np.all(diffs <= 1e-10):
            monotone_decreasing += 1
        else:
            neither += 1

        # Check second differences (convexity along path)
        second_diffs = np.diff(diffs)
        n_pos = np.sum(second_diffs > 1e-10)
        n_neg = np.sum(second_diffs < -1e-10)
        if n_neg == 0 and n_pos > 0:
            second_diff_positive += 1
        elif n_pos == 0 and n_neg > 0:
            second_diff_negative += 1
        else:
            second_diff_mixed += 1

    return {
        'total': total,
        'monotone_increasing': monotone_increasing,
        'monotone_decreasing': monotone_decreasing,
        'neither': neither,
        'path_convex': second_diff_positive,
        'path_concave': second_diff_negative,
        'path_mixed': second_diff_mixed,
    }


def test_submodularity(n_roots, n_trials=1000, epsilon=0.05):
    """Test submodularity / diminishing returns for F = 1/Phi_n.

    Submodularity in root coordinates: for a perturbation delta_i
    (increase root i by epsilon), the marginal gain F(lambda + delta_i) - F(lambda)
    should decrease as lambda "grows" (roots spread apart more).

    Specifically, test: if lambda ≺ mu (lambda majorized by mu, i.e. mu
    has more spread), then
        F(mu + delta_i) - F(mu) <= F(lambda + delta_i) - F(lambda)
    for each coordinate i.
    """
    n = n_roots
    violations = 0
    total = 0

    for _ in range(n_trials):
        mu = make_distinct(np.random.randn(n) * 2)
        D = random_doubly_stochastic(n, strength=np.random.uniform(0.1, 0.5))
        lam = make_distinct(D @ mu)  # lam ≺ mu

        F_lam = F(lam)
        F_mu = F(mu)
        if np.isnan(F_lam) or np.isnan(F_mu):
            continue

        for i in range(n):
            delta = np.zeros(n)
            delta[i] = epsilon

            lam_pert = make_distinct(lam + delta)
            mu_pert = make_distinct(mu + delta)

            F_lam_pert = F(lam_pert)
            F_mu_pert = F(mu_pert)

            if np.isnan(F_lam_pert) or np.isnan(F_mu_pert):
                continue

            total += 1
            marginal_lam = F_lam_pert - F_lam
            marginal_mu = F_mu_pert - F_mu

            # Submodularity: marginal at mu <= marginal at lam
            if marginal_mu > marginal_lam + 1e-10:
                violations += 1

    return violations, total


def symbolic_n3():
    """Explicit Schur-convexity analysis for n=3.

    For n=3 with roots (a, b, c), a < b < c:
    Phi_3 = (1/(a-b) + 1/(a-c))^2 + (1/(b-a) + 1/(b-c))^2 + (1/(c-a) + 1/(c-b))^2

    Test F on a grid of majorization-comparable triples.
    """
    print("\n--- n=3 symbolic/grid Schur test ---")
    print("Testing F on root triples with same sum, varying spread.")
    print("If Schur-concave: more spread => smaller F.")
    print("If Schur-convex: more spread => larger F.")

    # Fix sum = 0 (WLOG by translation invariance of Phi_n)
    # Parameterize: roots = (-s-d, d, s) with s > 0, -s < d < s
    # Spread increases with s.
    # Actually, for fixed sum=0, majorization order is determined by
    # sorted partial sums. Let's use: roots = (a, 0, -a) vs (b, 0, -b)
    # with a > b > 0. Then (a, 0, -a) majorizes (b, 0, -b).

    print("\nFixed-sum-zero symmetric triples (-s, 0, s):")
    print(f"  {'s':>6s}  {'F=1/Phi_3':>12s}  {'Phi_3':>12s}")
    for s in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        roots = np.array([-s, 0, s])
        f = F(roots)
        phi = phi_n(roots)
        print(f"  {s:6.2f}  {f:12.6f}  {phi:12.6f}")

    print("\nAsymmetric triples with sum=0, varying spread:")
    print(f"  {'roots':>30s}  {'F':>12s}  {'sum':>8s}  {'spread':>8s}")
    test_cases = [
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.2, 0.1, 1.1]),
        np.array([-1.5, 0.0, 1.5]),
        np.array([-2.0, 0.5, 1.5]),
        np.array([-2.0, 0.0, 2.0]),
        np.array([-3.0, 1.0, 2.0]),
        np.array([-3.0, 0.0, 3.0]),
    ]
    for roots in test_cases:
        roots = np.sort(roots)
        f = F(roots)
        spread = np.sum((roots - np.mean(roots))**2)
        print(f"  {str(roots):>30s}  {f:12.6f}  {np.sum(roots):8.4f}  {spread:8.4f}")

    # Now test with n=4
    print("\n--- n=4 symmetric pairs ---")
    print("Roots (-s, -t, t, s) with sum=0:")
    print(f"  {'(s,t)':>12s}  {'F=1/Phi_4':>12s}  {'spread':>12s}")
    for s, t in [(2.0, 0.5), (2.0, 1.0), (3.0, 0.5), (3.0, 1.0), (3.0, 2.0)]:
        roots = np.array([-s, -t, t, s])
        f = F(roots)
        spread = np.sum(roots**2)
        print(f"  ({s},{t}){' '*(8-len(f'({s},{t})'))}{f:12.6f}  {spread:12.4f}")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("Test 1: Schur-convexity/concavity of F = 1/Phi_n")
    print("  x ≺ y (x majorized by y, y has more spread)")
    print("  Schur-convex: F(x) <= F(y); Schur-concave: F(x) >= F(y)")
    print("=" * 65)

    for n in [3, 4, 5]:
        scv, sccv, total = test_schur_convexity(n, n_trials=3000)
        if total > 0:
            print(f"\n  n={n}: {total} tests")
            print(f"    Schur-convex violations (F(x)>F(y)): {scv} ({100*scv/total:.1f}%)")
            print(f"    Schur-concave violations (F(x)<F(y)): {sccv} ({100*sccv/total:.1f}%)")
            if scv == 0:
                print(f"    => F appears SCHUR-CONVEX (more spread => larger F)")
            elif sccv == 0:
                print(f"    => F appears SCHUR-CONCAVE (more spread => smaller F)")
            else:
                print(f"    => F is NEITHER Schur-convex nor Schur-concave")

    print("\n" + "=" * 65)
    print("Test 2: Interpolation path lambda_conv -> lambda_sum")
    print("  MSS: lambda(p⊞q) ≺ lambda(p)+lambda(q)")
    print("  Test F monotonicity and second differences along path")
    print("=" * 65)

    for n in [3, 4, 5]:
        r = test_interpolation_path(n, n_trials=1000)
        if r['total'] > 0:
            t = r['total']
            print(f"\n  n={n}: {t} paths")
            print(f"    monotone increasing: {r['monotone_increasing']} ({100*r['monotone_increasing']/t:.1f}%)")
            print(f"    monotone decreasing: {r['monotone_decreasing']} ({100*r['monotone_decreasing']/t:.1f}%)")
            print(f"    neither:             {r['neither']} ({100*r['neither']/t:.1f}%)")
            print(f"    path convex:  {r['path_convex']} ({100*r['path_convex']/t:.1f}%)")
            print(f"    path concave: {r['path_concave']} ({100*r['path_concave']/t:.1f}%)")
            print(f"    path mixed:   {r['path_mixed']} ({100*r['path_mixed']/t:.1f}%)")

    print("\n" + "=" * 65)
    print("Test 3: Submodularity of F in root coordinates")
    print("  lam ≺ mu; test: marginal(mu) <= marginal(lam)?")
    print("=" * 65)

    for n in [3, 4, 5]:
        v, total = test_submodularity(n, n_trials=1000)
        if total > 0:
            print(f"\n  n={n}: {v}/{total} submodularity violations ({100*v/total:.1f}%)")
            if v == 0:
                print(f"    => F appears SUBMODULAR in root coordinates")

    print("\n" + "=" * 65)
    print("Test 4: Symbolic / grid analysis")
    print("=" * 65)
    symbolic_n3()
