#!/usr/bin/env python3
"""Strategy A: Dyson Brownian Motion / Convexity exploration for P4 n>=4.

Key computational questions:
1. Is 1/Phi_n CONVEX in the finite free cumulants? (If yes + f(0)=0 → superadditivity)
2. Is E[1/Phi_n(A + √t·GUE)] monotone in t?
3. Hessian eigenvalues of 1/Phi_n at random points.
"""

import numpy as np
from math import factorial
import sys

np.random.seed(42)


# ── Core functions ──────────────────────────────────────────────────────

def phi_n(roots):
    """Compute Phi_n = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0 / (roots[i] - roots[j]) for j in range(n) if j != i)
        total += s * s
    return total


def inv_phi_n(roots):
    """1/Phi_n(roots). Returns 0 if roots have near-collisions."""
    try:
        p = phi_n(roots)
        if p < 1e-15:
            return 0.0
        return 1.0 / p
    except (ZeroDivisionError, FloatingPointError):
        return 0.0


def mss_convolve(a_coeffs, b_coeffs, n):
    """MSS finite free additive convolution on coefficients."""
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
    """[a_1, ..., a_n] -> sorted real roots of x^n + a_1 x^{n-1} + ... + a_n."""
    poly = np.concatenate([[1.0], coeffs])
    r = np.roots(poly)
    return np.sort(r.real)


def roots_to_coeffs(roots):
    """Roots -> [a_1, ..., a_n] (excluding leading 1)."""
    return np.poly(roots)[1:]


# ── Finite free cumulants ───────────────────────────────────────────────
# The finite free cumulants κ_k are defined so that κ_k(p ⊞_n q) = κ_k(p) + κ_k(q).
# For centered polynomials (a_1 = 0), the first few are:
#   κ_2 = a_2 (same as the coefficient for all n)
#   κ_3 = a_3 (for centered, κ_3 = a_3 when n >= 3)
#   κ_k for k >= 4: depends on lower cumulants and n
#
# The exact relationship: a_k = Σ over non-crossing partitions...
# For simplicity, we use the COEFFICIENT PARAMETERIZATION (a_1,...,a_n)
# and note that ⊞_n is bilinear in this parameterization (with MSS weights).
# Cumulant additivity means: if we parameterize by cumulants,
# ⊞_n becomes ADDITION. So convexity in cumulants = what we want.
#
# For centered degree-n polys:
#   a_2 = κ_2
#   a_3 = κ_3
#   a_4 = κ_4 + (n-2)(n-3)/(n(n-1)) * κ_2^2 / 2  [correction term]
# (The exact formulas come from the moment-cumulant relation for finite free probability.)
#
# For our purposes, we'll use the coefficient parameterization and
# TEST convexity of 1/Phi_n along the CONVOLUTION LINES
# (t*a + (1-t)*b in cumulant space = MSS convolution with scaled inputs).

def cumulants_to_coeffs_n4(kappa):
    """Convert centered finite free cumulants [κ_2, κ_3, κ_4] to
    coefficients [a_1=0, a_2, a_3, a_4] for degree 4.
    Uses the finite free moment-cumulant relation."""
    k2, k3, k4 = kappa
    a2 = k2
    a3 = k3
    # a4 = k4 + correction involving k2^2
    # For n=4: the correction is (4-2)(4-3)/(4*3) * k2^2 / ...
    # Actually, let me derive this from the MSS convolution.
    # If p has cumulants (k2, 0, 0) and q has cumulants (0, 0, k4-correction),
    # then p ⊞_4 q should have cumulants (k2, 0, k4).
    # Easier: use the fact that for the MSS convolution, coefficients are:
    # c_k = Σ w(n,i,j) a_i b_j
    # If all cumulants of q are 0 except κ_2, then q(x) = x^4 + κ_2 x^2.
    # Then p ⊞_4 q: c_2 = a_2 + κ_2, c_3 = a_3, c_4 = a_4 + w(4,2,2)*a_2*κ_2
    # So the "cumulant" picture: κ_4(p ⊞_4 q) = κ_4(p) + κ_4(q)
    # means: a_4 - f(a_2) is the cumulant part, where f(a_2) accounts for the
    # quadratic correction.
    #
    # For n=4 centered: the finite free cumulants are exactly a_2, a_3,
    # and κ_4 = a_4 - (1/6) * a_2^2 (from the MSS cross-term w(4,2,2)=1/6).
    # Because: (p with a_2=s, a_4=t) ⊞_4 (q with a_2=u, a_4=v) has
    # c_4 = t + (1/6)su + v, so the "additive" part is (t - α s^2) + (v - α u^2)
    # = (t+v) + α(-(s^2+u^2)) but c_4 = t + v + su/6, so we need:
    # κ_4 = a_4 - α a_2^2 such that κ_4(c) = κ_4(a) + κ_4(b):
    # c_4 - α c_2^2 = (a_4 - α a_2^2) + (b_4 - α b_2^2)
    # c_4 - α(a_2+b_2)^2 = a_4 + b_4 + (1/6)a_2 b_2 - α a_2^2 - 2α a_2 b_2 - α b_2^2
    # = (a_4 - α a_2^2) + (b_4 - α b_2^2) + (1/6 - 2α) a_2 b_2
    # For this to equal κ_4(a) + κ_4(b), need 1/6 - 2α = 0, so α = 1/12.
    # Therefore: κ_4 = a_4 - (1/12) a_2^2
    a2 = k2
    a3 = k3
    a4 = k4 + (1.0/12.0) * k2**2
    return np.array([0.0, a2, a3, a4])


def coeffs_to_cumulants_n4(coeffs):
    """Convert centered coefficients [0, a_2, a_3, a_4] to cumulants [κ_2, κ_3, κ_4]."""
    a2 = coeffs[1]
    a3 = coeffs[2]
    a4 = coeffs[3]
    k2 = a2
    k3 = a3
    k4 = a4 - (1.0/12.0) * a2**2
    return np.array([k2, k3, k4])


# ── Part 1: Verify cumulant additivity ──────────────────────────────────

def verify_cumulant_additivity(n_trials=500):
    """Verify that κ_k(p ⊞_4 q) = κ_k(p) + κ_k(q) for our cumulant definition."""
    print(f"\n{'='*70}")
    print(f"PART 1: Verify cumulant additivity at n=4")
    print(f"{'='*70}")

    max_err = 0.0
    for trial in range(n_trials):
        # Random centered polynomials
        a = np.zeros(4)
        a[1] = np.random.randn() * 2  # a_2
        a[2] = np.random.randn()       # a_3
        a[3] = np.random.randn()       # a_4

        b = np.zeros(4)
        b[1] = np.random.randn() * 2
        b[2] = np.random.randn()
        b[3] = np.random.randn()

        c = mss_convolve(a, b, 4)

        ka = coeffs_to_cumulants_n4(a)
        kb = coeffs_to_cumulants_n4(b)
        kc = coeffs_to_cumulants_n4(c)

        err = np.max(np.abs(kc - (ka + kb)))
        max_err = max(max_err, err)

    print(f"  {n_trials} trials, max |κ(p⊞q) - κ(p) - κ(q)| = {max_err:.2e}")
    if max_err < 1e-10:
        print(f"  CONFIRMED: cumulants are exactly additive under ⊞_4")
    else:
        print(f"  FAILED: cumulant additivity error too large")
    return max_err


# ── Part 2: Convexity of 1/Phi_n in cumulant space ─────────────────────

def test_convexity_in_cumulants(n_degree=4, n_trials=3000):
    """Test: is 1/Phi_n CONVEX in the cumulant space?
    i.e., 1/Phi_n(t*κ + (1-t)*κ') <= t*1/Phi_n(κ) + (1-t)*1/Phi_n(κ')
    for random κ, κ' and t ∈ (0,1)?
    """
    print(f"\n{'='*70}")
    print(f"PART 2: Convexity of 1/Phi_{n_degree} in cumulant space")
    print(f"{'='*70}")

    convex_violations = 0
    concave_violations = 0
    valid = 0

    for trial in range(n_trials):
        # Random cumulant vectors
        k1 = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])
        k2 = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])
        t = np.random.uniform(0.1, 0.9)

        # Midpoint in cumulant space
        k_mid = t * k1 + (1 - t) * k2

        # Convert to coefficients and find roots
        try:
            a1 = cumulants_to_coeffs_n4(k1)
            a2 = cumulants_to_coeffs_n4(k2)
            a_mid = cumulants_to_coeffs_n4(k_mid)

            r1 = coeffs_to_roots(a1)
            r2 = coeffs_to_roots(a2)
            r_mid = coeffs_to_roots(a_mid)

            # Check all roots are real
            poly1 = np.concatenate([[1.0], a1])
            poly2 = np.concatenate([[1.0], a2])
            poly_mid = np.concatenate([[1.0], a_mid])
            if (np.max(np.abs(np.roots(poly1).imag)) > 0.01 or
                np.max(np.abs(np.roots(poly2).imag)) > 0.01 or
                np.max(np.abs(np.roots(poly_mid).imag)) > 0.01):
                continue

            # Check distinct roots
            if (np.min(np.diff(r1)) < 1e-6 or
                np.min(np.diff(r2)) < 1e-6 or
                np.min(np.diff(r_mid)) < 1e-6):
                continue

            f1 = inv_phi_n(r1)
            f2 = inv_phi_n(r2)
            f_mid = inv_phi_n(r_mid)

            if f1 < 1e-15 or f2 < 1e-15:
                continue

            valid += 1
            convex_bound = t * f1 + (1 - t) * f2  # upper bound if convex
            # Convex: f_mid <= convex_bound
            # Concave: f_mid >= convex_bound

            if f_mid > convex_bound + 1e-12:
                convex_violations += 1
            if f_mid < convex_bound - 1e-12:
                concave_violations += 1

        except Exception:
            continue

    print(f"  {valid} valid trials (of {n_trials})")
    print(f"  Convex violations (f_mid > t*f1+(1-t)*f2): {convex_violations}/{valid} "
          f"({100*convex_violations/max(1,valid):.1f}%)")
    print(f"  Concave violations (f_mid < t*f1+(1-t)*f2): {concave_violations}/{valid} "
          f"({100*concave_violations/max(1,valid):.1f}%)")
    if convex_violations == 0 and concave_violations > 0:
        print(f"  => 1/Phi_n appears CONVEX in cumulant space!")
    elif concave_violations == 0 and convex_violations > 0:
        print(f"  => 1/Phi_n appears CONCAVE in cumulant space")
    elif convex_violations == 0 and concave_violations == 0:
        print(f"  => 1/Phi_n appears LINEAR in cumulant space (?!)")
    else:
        print(f"  => 1/Phi_n is NEITHER convex NOR concave in cumulant space")
    return convex_violations, concave_violations, valid


# ── Part 3: Convexity along convolution lines ──────────────────────────

def test_convexity_along_conv_lines(n_degree=4, n_trials=2000):
    """More targeted test: is f(κ_p + κ_q) >= f(κ_p) + f(κ_q)?
    This is what we actually need (superadditivity), which is
    DIFFERENT from convexity in general."""
    print(f"\n{'='*70}")
    print(f"PART 3: Superadditivity test: f(κ_p+κ_q) >= f(κ_p)+f(κ_q) at n={n_degree}")
    print(f"{'='*70}")

    violations = 0
    valid = 0
    surpluses = []

    for trial in range(n_trials):
        k_p = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])
        k_q = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])
        k_sum = k_p + k_q

        try:
            a_p = cumulants_to_coeffs_n4(k_p)
            a_q = cumulants_to_coeffs_n4(k_q)
            a_sum = cumulants_to_coeffs_n4(k_sum)

            r_p = coeffs_to_roots(a_p)
            r_q = coeffs_to_roots(a_q)
            r_sum = coeffs_to_roots(a_sum)

            # Check real roots
            for arr, coeffs in [(r_p, a_p), (r_q, a_q), (r_sum, a_sum)]:
                poly = np.concatenate([[1.0], coeffs])
                if np.max(np.abs(np.roots(poly).imag)) > 0.01:
                    raise ValueError("complex roots")
                if np.min(np.diff(arr)) < 1e-6:
                    raise ValueError("near-degenerate")

            fp = inv_phi_n(r_p)
            fq = inv_phi_n(r_q)
            f_sum = inv_phi_n(r_sum)

            if fp < 1e-15 or fq < 1e-15:
                continue

            valid += 1
            surplus = f_sum - fp - fq
            surpluses.append(surplus)
            if surplus < -1e-10:
                violations += 1

        except Exception:
            continue

    surpluses = np.array(surpluses)
    print(f"  {valid} valid trials")
    print(f"  Violations (f(κ_p+κ_q) < f(κ_p)+f(κ_q)): {violations}/{valid}")
    if len(surpluses) > 0:
        print(f"  Surplus range: [{np.min(surpluses):.6f}, {np.max(surpluses):.6f}]")
        print(f"  Surplus mean: {np.mean(surpluses):.6f}")

    # NOTE: This is testing the CUMULANT-ADDITION version, not the MSS convolution.
    # They should be the same if cumulants are truly additive.
    # Let's verify by comparing with MSS:
    print(f"\n  Cross-check: compare cumulant addition vs MSS convolution...")
    mismatches = 0
    for trial in range(min(200, valid)):
        k_p = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])
        k_q = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])

        a_p = cumulants_to_coeffs_n4(k_p)
        a_q = cumulants_to_coeffs_n4(k_q)

        # MSS convolution
        c_mss = mss_convolve(a_p, a_q, 4)

        # Cumulant addition
        k_sum = k_p + k_q
        a_sum = cumulants_to_coeffs_n4(k_sum)

        err = np.max(np.abs(c_mss - a_sum))
        if err > 1e-10:
            mismatches += 1

    print(f"  MSS vs cumulant-addition mismatches: {mismatches}/200")
    return violations, valid


# ── Part 4: Dyson Brownian Motion simulation ────────────────────────────

def simulate_dyson_bm(eigenvalues_init, t_final, n_steps=1000, n_paths=200):
    """Simulate Dyson Brownian motion: dλ_i = dB_i/√n + (1/n)Σ_{j≠i} dt/(λ_i-λ_j).
    Track E[1/Phi_n(λ(t))] as a function of t.

    Returns: times, mean_inv_phi (arrays)
    """
    n = len(eigenvalues_init)
    dt = t_final / n_steps
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0, t_final, n_steps + 1)
    inv_phi_paths = np.zeros((n_paths, n_steps + 1))

    for path in range(n_paths):
        lam = eigenvalues_init.copy()
        inv_phi_paths[path, 0] = inv_phi_n(lam)

        for step in range(n_steps):
            # Drift: (1/n) Σ_{j≠i} 1/(λ_i - λ_j)
            drift = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if j != i:
                        gap = lam[i] - lam[j]
                        if abs(gap) < 1e-10:
                            gap = np.sign(gap) * 1e-10 if gap != 0 else 1e-10
                        drift[i] += 1.0 / gap
                drift[i] /= n

            # Diffusion: dB_i / √n
            noise = np.random.randn(n) * sqrt_dt / np.sqrt(n)

            lam = lam + drift * dt + noise
            lam = np.sort(lam)  # maintain ordering

            inv_phi_paths[path, step + 1] = inv_phi_n(lam)

    mean_inv_phi = np.mean(inv_phi_paths, axis=0)
    return times, mean_inv_phi


def test_dyson_monotonicity(n_degree=4):
    """Test: is E[1/Phi_n(A + √t · GUE)] monotone increasing in t?"""
    print(f"\n{'='*70}")
    print(f"PART 4: Dyson BM monotonicity test at n={n_degree}")
    print(f"{'='*70}")

    # Several starting configurations
    configs = [
        ("arithmetic", np.linspace(-2, 2, n_degree)),
        ("clustered", np.array([-0.1, -0.05, 0.05, 0.1]) if n_degree == 4
         else np.linspace(-0.2, 0.2, n_degree)),
        ("spread", np.linspace(-5, 5, n_degree)),
    ]

    for name, init_eigs in configs:
        print(f"\n  Config: {name}, initial eigenvalues = {init_eigs}")
        times, mean_inv_phi = simulate_dyson_bm(
            init_eigs, t_final=2.0, n_steps=500, n_paths=300
        )

        # Check monotonicity
        diffs = np.diff(mean_inv_phi)
        n_decreasing = np.sum(diffs < -1e-12)
        print(f"    E[1/Phi_n] at t=0:   {mean_inv_phi[0]:.6f}")
        print(f"    E[1/Phi_n] at t=2:   {mean_inv_phi[-1]:.6f}")
        print(f"    Monotone? {n_decreasing} decreasing steps out of {len(diffs)}")
        if n_decreasing == 0:
            print(f"    => MONOTONE INCREASING")
        else:
            pct = 100 * n_decreasing / len(diffs)
            print(f"    => NOT monotone ({pct:.1f}% decreasing)")


# ── Part 5: Hessian eigenvalues ─────────────────────────────────────────

def numerical_hessian_cumulants(kappa, eps=1e-5):
    """Compute Hessian of 1/Phi_4 w.r.t. cumulant coordinates at kappa."""
    n = len(kappa)
    H = np.zeros((n, n))

    def f(k):
        try:
            a = cumulants_to_coeffs_n4(k)
            r = coeffs_to_roots(a)
            poly = np.concatenate([[1.0], a])
            if np.max(np.abs(np.roots(poly).imag)) > 0.01:
                return None
            if np.min(np.diff(r)) < 1e-6:
                return None
            return inv_phi_n(r)
        except:
            return None

    f0 = f(kappa)
    if f0 is None:
        return None

    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps

            fpp = f(kappa + ei + ej)
            fpm = f(kappa + ei - ej)
            fmp = f(kappa - ei + ej)
            fmm = f(kappa - ei - ej)

            if any(v is None for v in [fpp, fpm, fmp, fmm]):
                return None

            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H[j, i] = H[i, j]

    return H


def test_hessian_eigenvalues(n_trials=500):
    """Check eigenvalues of the Hessian of 1/Phi_4 in cumulant space."""
    print(f"\n{'='*70}")
    print(f"PART 5: Hessian eigenvalues of 1/Phi_4 in cumulant space")
    print(f"{'='*70}")

    all_min_eig = []
    all_max_eig = []
    indefinite = 0
    psd = 0
    nsd = 0
    valid = 0

    for trial in range(n_trials):
        k = np.array([np.random.randn() * 1.5, np.random.randn() * 0.5, np.random.randn() * 0.3])

        H = numerical_hessian_cumulants(k)
        if H is None:
            continue

        eigs = np.linalg.eigvalsh(H)
        valid += 1
        all_min_eig.append(eigs[0])
        all_max_eig.append(eigs[-1])

        if eigs[0] >= -1e-8:
            psd += 1
        elif eigs[-1] <= 1e-8:
            nsd += 1
        else:
            indefinite += 1

    all_min_eig = np.array(all_min_eig)
    all_max_eig = np.array(all_max_eig)

    print(f"  {valid} valid trials")
    print(f"  Hessian definiteness:")
    print(f"    PSD (convex): {psd}/{valid} ({100*psd/max(1,valid):.1f}%)")
    print(f"    NSD (concave): {nsd}/{valid} ({100*nsd/max(1,valid):.1f}%)")
    print(f"    Indefinite: {indefinite}/{valid} ({100*indefinite/max(1,valid):.1f}%)")
    print(f"  Eigenvalue ranges:")
    print(f"    min eigenvalue: [{np.min(all_min_eig):.6f}, {np.max(all_min_eig):.6f}]")
    print(f"    max eigenvalue: [{np.min(all_max_eig):.6f}, {np.max(all_max_eig):.6f}]")

    if psd == valid:
        print(f"  => 1/Phi_4 is CONVEX in cumulant space (all Hessians PSD)")
    elif nsd == valid:
        print(f"  => 1/Phi_4 is CONCAVE in cumulant space (all Hessians NSD)")
    else:
        print(f"  => 1/Phi_4 is NEITHER convex NOR concave (mixed definiteness)")


# ── Part 6: Superadditivity via convexity + f(0)=0 in 1D ───────────────

def test_1d_superadditivity_mechanism(n_trials=2000):
    """Test the 1D version: along rays in cumulant space, is 1/Phi_n
    superadditive? i.e., f(s+t) >= f(s) + f(t) where s,t >= 0 parameterize
    scaling along a fixed direction."""
    print(f"\n{'='*70}")
    print(f"PART 6: 1D ray superadditivity test")
    print(f"{'='*70}")

    violations = 0
    valid = 0

    for trial in range(n_trials):
        # Random direction in cumulant space
        direction = np.array([np.random.randn() * 2, np.random.randn(), np.random.randn()])
        s = abs(np.random.randn())
        t = abs(np.random.randn())

        try:
            k_s = s * direction
            k_t = t * direction
            k_st = (s + t) * direction

            a_s = cumulants_to_coeffs_n4(k_s)
            a_t = cumulants_to_coeffs_n4(k_t)
            a_st = cumulants_to_coeffs_n4(k_st)

            for coeffs in [a_s, a_t, a_st]:
                poly = np.concatenate([[1.0], coeffs])
                r = np.roots(poly)
                if np.max(np.abs(r.imag)) > 0.01:
                    raise ValueError("complex")
                if np.min(np.diff(np.sort(r.real))) < 1e-6:
                    raise ValueError("degenerate")

            r_s = coeffs_to_roots(a_s)
            r_t = coeffs_to_roots(a_t)
            r_st = coeffs_to_roots(a_st)

            fs = inv_phi_n(r_s)
            ft = inv_phi_n(r_t)
            fst = inv_phi_n(r_st)

            if fs < 1e-15 or ft < 1e-15:
                continue

            valid += 1
            if fst < fs + ft - 1e-10:
                violations += 1

        except Exception:
            continue

    print(f"  {valid} valid trials")
    print(f"  1D ray violations: {violations}/{valid}")
    if violations == 0:
        print(f"  => 1/Phi_4 is superadditive along all tested rays from origin")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Strategy A: Dyson BM / Convexity Exploration")
    print("=" * 70)

    # Part 1: Verify cumulant additivity
    verify_cumulant_additivity()

    # Part 2: Convexity in cumulant space
    test_convexity_in_cumulants()

    # Part 3: Direct superadditivity test via cumulants
    test_convexity_along_conv_lines()

    # Part 4: Dyson BM monotonicity
    test_dyson_monotonicity()

    # Part 5: Hessian eigenvalues
    test_hessian_eigenvalues()

    # Part 6: 1D ray test
    test_1d_superadditivity_mechanism()

    print(f"\n{'='*70}")
    print("STRATEGY A SUMMARY")
    print(f"{'='*70}")
    print("Key results:")
    print("1. Cumulant additivity under ⊞_4 (Part 1)")
    print("2. Convexity/concavity of 1/Phi_4 in cumulant space (Parts 2, 5)")
    print("3. Direct superadditivity in cumulant picture (Part 3)")
    print("4. Dyson BM monotonicity (Part 4)")
    print("5. 1D ray superadditivity (Part 6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
