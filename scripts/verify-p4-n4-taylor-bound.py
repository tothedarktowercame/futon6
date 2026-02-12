#!/usr/bin/env python3
"""Prove -N >= 0 via Taylor expansion + numerical verification.

Strategy:
1. Near x₀ = (0, 1/12, 0, 1/12): -N = Q₂(δ) + R(δ) where Q₂ is the
   positive-definite quadratic part (Hessian eigenvalues 3/4, 21/8, 6, 8)
   and R is the cubic-and-higher remainder. For |δ| ≤ r, Q₂ ≥ |R|.

2. Away from x₀ (|δ| > r, within the domain): verify -N > 0 numerically
   using dense grid evaluation and optimization.

This avoids the SOS infeasibility from interior zeros.
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly, together, fraction, Matrix
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import sys


def build_neg_N():
    """Build -N and domain constraint polynomials."""
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4
                    - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2

    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4
                    - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2

    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4
                    - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)

    surplus_frac = together(-disc_r/(4*f1_r*f2_r)
                            + disc_p/(4*f1_p*f2_p)
                            + disc_q/(4*f1_q*f2_q))
    N_expr, D_expr = fraction(surplus_frac)
    neg_N = expand(-N_expr)

    variables = (a3, a4, b3, b4)
    domain_constraints = {
        'disc_p': disc_p,
        'disc_q': disc_q,
        '-f2_p': expand(-f2_p),
        '-f2_q': expand(-f2_q),
        'f1_p': f1_p,
        'f1_q': f1_q,
    }
    return neg_N, variables, domain_constraints


def taylor_analysis(neg_N, variables):
    """Compute Taylor expansion of -N at the equality point.

    Returns:
        coeffs: dict of {(α₁,α₂,α₃,α₄): rational_coeff} in shifted coords
        hessian_eigenvalues: eigenvalues of H/2 (the quadratic form)
        radius_bound: r such that Q₂ ≥ |R| for |δ| ≤ r
    """
    a3, a4, b3, b4 = variables
    # Equality point
    x0 = {a3: 0, a4: Rational(1, 12), b3: 0, b4: Rational(1, 12)}

    # Shift: δ₁ = a3, δ₂ = a4 - 1/12, δ₃ = b3, δ₄ = b4 - 1/12
    d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')
    shifted = neg_N.subs({a3: d1, a4: d2 + Rational(1, 12),
                          b3: d3, b4: d4 + Rational(1, 12)})
    shifted = expand(shifted)

    # Extract Taylor coefficients
    poly = Poly(shifted, d1, d2, d3, d4)
    coeffs = {}
    for exp, coeff in poly.as_dict().items():
        coeffs[exp] = coeff

    # Verify: constant and linear terms are 0
    const = coeffs.get((0, 0, 0, 0), 0)
    print(f"  Constant term: {const}")
    assert const == 0, f"Constant term should be 0, got {const}"

    linear_terms = {e: c for e, c in coeffs.items() if sum(e) == 1}
    print(f"  Linear terms: {linear_terms}")
    assert all(c == 0 for c in linear_terms.values()), "Linear terms should be 0"

    # Extract Hessian (degree-2 terms)
    H = sp.zeros(4, 4)
    for e, c in coeffs.items():
        if sum(e) == 2:
            # The coefficient of δᵢδⱼ in the Taylor expansion is H[i,j]/2 for i≠j
            # and H[i,i]/2 for i=j
            indices = []
            for k in range(4):
                indices.extend([k] * e[k])
            i, j = indices
            if i == j:
                H[i, j] = 2 * c  # Q₂ = (1/2) δᵀ H δ, coeff of δᵢ² is H[i,i]/2
            else:
                H[i, j] = c  # coeff of δᵢδⱼ is H[i,j] (since 2*(H[i,j]/2) for off-diag)
                H[j, i] = c

    print(f"\n  Hessian matrix H:")
    for i in range(4):
        row = [f"{str(H[i,j]):>10}" for j in range(4)]
        print(f"    [{', '.join(row)}]")

    eigenvalues = sorted([sp.nsimplify(e) for e in H.eigenvals().keys()])
    print(f"\n  Hessian eigenvalues: {eigenvalues}")
    lambda_min = min(float(e) for e in eigenvalues)
    print(f"  λ_min = {lambda_min}")

    # Compute |R(δ)| bound: for |δ| ≤ t, |R(δ)| ≤ P(t)
    # where P(t) = Σ_{k=3}^{10} A_k t^k, A_k = Σ_{|α|=k} |c_α|
    degree_sums = {}
    for e, c in coeffs.items():
        k = sum(e)
        if k >= 3:
            degree_sums.setdefault(k, sp.Rational(0))
            degree_sums[k] += abs(c)

    print(f"\n  Remainder bound coefficients A_k (Σ|c_α| for |α|=k):")
    for k in sorted(degree_sums.keys()):
        print(f"    k={k}: A_{k} = {float(degree_sums[k]):.6f}")

    # Q₂(δ) ≥ (λ_min/2) |δ|²
    # Need: (λ_min/2) t² ≥ P(t) for t ∈ [0, r]
    # i.e., (λ_min/2) ≥ Σ A_k t^{k-2}
    # i.e., Σ A_k t^{k-2} ≤ λ_min/2

    def remainder_bound(t):
        """P(t) / t² = Σ A_k t^{k-2}."""
        return sum(float(degree_sums[k]) * t**(k - 2) for k in sorted(degree_sums.keys()))

    # Find r: remainder_bound(r) = λ_min/2
    threshold = lambda_min / 2
    print(f"\n  Threshold: λ_min/2 = {threshold:.6f}")

    # Binary search for r
    lo, hi = 0, 1
    while remainder_bound(hi) < threshold and hi < 100:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if remainder_bound(mid) < threshold:
            lo = mid
        else:
            hi = mid
    r = lo

    print(f"  Taylor radius r = {r:.6f}")
    print(f"  (for |δ| ≤ r, Q₂(δ) ≥ |R(δ)|, hence -N ≥ 0)")

    # Verify at r
    q2_at_r = (lambda_min / 2) * r**2
    p_at_r = sum(float(degree_sums[k]) * r**k for k in sorted(degree_sums.keys()))
    print(f"  Check: Q₂ lower bound at |δ|=r: {q2_at_r:.8f}")
    print(f"  Check: |R| upper bound at |δ|=r: {p_at_r:.8f}")
    print(f"  Margin: {q2_at_r - p_at_r:.2e}")

    return coeffs, eigenvalues, r, degree_sums


def numerical_verification(neg_N, variables, domain_constraints, radius, n_grid=50):
    """Verify -N > 0 on {|x - x₀| > radius} ∩ domain.

    Uses dense grid + scipy optimization.
    """
    a3, a4, b3, b4 = variables
    x0 = np.array([0.0, 1/12, 0.0, 1/12])

    # Compile -N and constraints to fast numpy functions
    neg_N_func = sp.lambdify((a3, a4, b3, b4), neg_N, 'numpy')

    constraint_funcs = {}
    for name, expr in domain_constraints.items():
        constraint_funcs[name] = sp.lambdify((a3, a4, b3, b4), expr, 'numpy')

    def in_domain(pt):
        """Check if point is in the real-rooted domain."""
        a3v, a4v, b3v, b4v = pt
        disc_p = constraint_funcs['disc_p'](a3v, a4v, b3v, b4v)
        disc_q = constraint_funcs['disc_q'](a3v, a4v, b3v, b4v)
        nf2_p = constraint_funcs['-f2_p'](a3v, a4v, b3v, b4v)
        nf2_q = constraint_funcs['-f2_q'](a3v, a4v, b3v, b4v)
        f1_p = constraint_funcs['f1_p'](a3v, a4v, b3v, b4v)
        f1_q = constraint_funcs['f1_q'](a3v, a4v, b3v, b4v)
        return (disc_p >= -1e-10 and disc_q >= -1e-10 and
                nf2_p >= -1e-10 and nf2_q >= -1e-10 and
                f1_p >= -1e-10 and f1_q >= -1e-10)

    # Domain bounds
    # f1_p >= 0: a4 >= -1/12
    # -f2_p >= 0: 8*a4 <= 2 - 9*a3^2, so a4 <= 1/4 (when a3=0)
    # disc_p >= 0 + above: |a3| <= sqrt(8/27) ≈ 0.544
    a3_range = (-0.55, 0.55)
    a4_range = (-1/12 - 0.001, 0.26)
    b3_range = (-0.55, 0.55)
    b4_range = (-1/12 - 0.001, 0.26)

    # Step 1: Dense grid evaluation
    print(f"\n  Grid evaluation (n={n_grid} per dim)...")
    min_neg_N_outer = np.inf
    min_point = None
    n_domain = 0
    n_outer = 0

    a3_grid = np.linspace(*a3_range, n_grid)
    a4_grid = np.linspace(*a4_range, n_grid)
    b3_grid = np.linspace(*b3_range, n_grid)
    b4_grid = np.linspace(*b4_range, n_grid)

    # For efficiency, evaluate on a coarser grid first
    for a3v in a3_grid:
        for a4v in a4_grid:
            for b3v in b3_grid:
                for b4v in b4_grid:
                    pt = (a3v, a4v, b3v, b4v)
                    if not in_domain(pt):
                        continue
                    n_domain += 1

                    delta = np.array(pt) - x0
                    dist = np.linalg.norm(delta)
                    if dist <= radius:
                        continue
                    n_outer += 1

                    val = float(neg_N_func(*pt))
                    if val < min_neg_N_outer:
                        min_neg_N_outer = val
                        min_point = pt

    print(f"  Domain points: {n_domain}")
    print(f"  Outer region points: {n_outer}")
    print(f"  Min -N in outer region: {min_neg_N_outer:.8e}")
    if min_point:
        delta = np.array(min_point) - x0
        print(f"  At point: {min_point}, |δ| = {np.linalg.norm(delta):.4f}")

    # Step 2: Targeted optimization in outer region
    print(f"\n  Local optimization in outer region...")

    def objective(x):
        """Minimize -N (find its minimum) subject to being in domain and |δ| > r."""
        a3v, a4v, b3v, b4v = x
        # Penalty for leaving domain
        penalty = 0
        for name, func in constraint_funcs.items():
            val = func(a3v, a4v, b3v, b4v)
            if val < 0:
                penalty += 1e6 * val**2

        # Penalty for being inside Taylor ball
        delta = np.array(x) - x0
        dist = np.linalg.norm(delta)
        if dist < radius:
            penalty += 1e6 * (radius - dist)**2

        return float(neg_N_func(a3v, a4v, b3v, b4v)) + penalty

    rng = np.random.default_rng(42)
    min_opt = np.inf
    min_opt_point = None
    n_starts = 2000

    for trial in range(n_starts):
        # Random start in domain
        a3v = rng.uniform(*a3_range)
        a4v = rng.uniform(*a4_range)
        b3v = rng.uniform(*b3_range)
        b4v = rng.uniform(*b4_range)

        x0_start = np.array([a3v, a4v, b3v, b4v])
        delta = x0_start - x0
        if np.linalg.norm(delta) < radius * 0.5:
            continue

        try:
            res = minimize(objective, x0_start,
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if in_domain(res.x) and np.linalg.norm(res.x - x0) > radius * 0.9:
                val = float(neg_N_func(*res.x))
                if val < min_opt:
                    min_opt = val
                    min_opt_point = res.x
        except Exception:
            pass

    print(f"  Local optimizations: {n_starts}")
    print(f"  Min -N found: {min_opt:.8e}")
    if min_opt_point is not None:
        delta = min_opt_point - x0
        print(f"  At point: {min_opt_point}, |δ| = {np.linalg.norm(delta):.4f}")
        print(f"  Domain check: {in_domain(min_opt_point)}")

    # Step 3: Differential evolution (global optimizer)
    print(f"\n  Differential evolution in outer region...")
    bounds = [a3_range, a4_range, b3_range, b4_range]

    def de_objective(x):
        """Minimize -N with penalties."""
        penalty = 0
        for name, func in constraint_funcs.items():
            val = func(*x)
            if val < 0:
                penalty += 1e8 * val**2
        delta = np.array(x) - x0
        dist = np.linalg.norm(delta)
        if dist < radius:
            penalty += 1e8 * (radius - dist)**2
        return float(neg_N_func(*x)) + penalty

    de_result = differential_evolution(de_objective, bounds,
                                        seed=42, maxiter=1000, tol=1e-12,
                                        popsize=30)
    print(f"  DE result: {de_result.fun:.8e}")
    print(f"  At point: {de_result.x}")
    if in_domain(de_result.x):
        val = float(neg_N_func(*de_result.x))
        delta = de_result.x - x0
        print(f"  -N value: {val:.8e}, |δ| = {np.linalg.norm(delta):.4f}")

    return min_neg_N_outer, min_opt


def boundary_analysis(neg_N, variables, domain_constraints):
    """Check -N on each boundary face of the domain.

    Boundary faces: f₁_p = 0, f₁_q = 0, f₂_p = 0, f₂_q = 0, disc_p = 0, disc_q = 0.
    By (p,q) symmetry, only need to check 3 faces.
    """
    a3, a4, b3, b4 = variables

    print("\n  === Boundary Face Analysis ===")

    neg_N_func = sp.lambdify((a3, a4, b3, b4), neg_N, 'numpy')

    # Face 1: f₁_p = 0, i.e., a₄ = -1/12
    print("\n  Face 1: f₁_p = 0 (a₄ = -1/12)")
    face1 = neg_N.subs(a4, Rational(-1, 12))
    face1 = expand(face1)
    if face1 == 0:
        print("    -N ≡ 0 on this face (f₁_p divides -N)")
    else:
        poly1 = Poly(face1, a3, b3, b4)
        print(f"    -N|_face: {len(poly1.as_dict())} terms, degree {poly1.total_degree()}")

        # Check if f₁_p divides -N
        neg_N_poly = Poly(neg_N, a3, a4, b3, b4)
        f1_p_poly = Poly(1 + 12*a4, a3, a4, b3, b4)
        quotient, remainder = sp.div(neg_N, 1 + 12*a4, a4)
        remainder = expand(remainder)
        if remainder == 0:
            print("    f₁_p divides -N exactly!")
        else:
            # Numerically check -N on this face
            print("    f₁_p does NOT divide -N. Checking numerically...")
            face1_func = sp.lambdify((a3, b3, b4), face1, 'numpy')
            rng = np.random.default_rng(123)
            min_val = np.inf
            for _ in range(100000):
                a3v = rng.uniform(-0.55, 0.55)
                b3v = rng.uniform(-0.55, 0.55)
                b4v = rng.uniform(-1/12, 0.25)
                val = float(face1_func(a3v, b3v, b4v))
                if val < min_val:
                    min_val = val
            print(f"    Min -N on face 1: {min_val:.8e}")

    # Face 2: -f₂_p = 0, i.e., 9a₃² + 8a₄ - 2 = 0, a₄ = (2 - 9a₃²)/8
    print("\n  Face 2: f₂_p = 0 (a₄ = (2-9a₃²)/8)")
    face2 = neg_N.subs(a4, (2 - 9*a3**2)/8)
    face2 = expand(face2)
    if face2 == 0:
        print("    -N ≡ 0 on this face")
    else:
        poly2 = Poly(face2, a3, b3, b4)
        print(f"    -N|_face: {len(poly2.as_dict())} terms, degree {poly2.total_degree()}")
        face2_func = sp.lambdify((a3, b3, b4), face2, 'numpy')
        rng = np.random.default_rng(456)
        min_val = np.inf
        for _ in range(100000):
            a3v = rng.uniform(-0.55, 0.55)
            b3v = rng.uniform(-0.55, 0.55)
            b4v = rng.uniform(-1/12, 0.25)
            val = float(face2_func(a3v, b3v, b4v))
            if val < min_val:
                min_val = val
        print(f"    Min -N on face 2: {min_val:.8e}")

    # Face 3: disc_p = 0
    print("\n  Face 3: disc_p = 0 (implicit surface)")
    print("    Cannot substitute directly (degree 4 in a₃, a₄)")
    print("    Checking numerically on disc_p = 0 surface...")

    disc_p_func = sp.lambdify((a3, a4), domain_constraints['disc_p'], 'numpy')
    nf2_p_func = sp.lambdify((a3, a4), domain_constraints['-f2_p'], 'numpy')
    f1_p_func = sp.lambdify((a3, a4), domain_constraints['f1_p'], 'numpy')
    disc_q_func = sp.lambdify((b3, b4), domain_constraints['disc_q'], 'numpy')
    nf2_q_func = sp.lambdify((b3, b4), domain_constraints['-f2_q'], 'numpy')
    f1_q_func = sp.lambdify((b3, b4), domain_constraints['f1_q'], 'numpy')

    rng = np.random.default_rng(789)
    min_val = np.inf
    n_found = 0

    # Sample a₃ and find a₄ on disc_p = 0
    from scipy.optimize import brentq
    for _ in range(10000):
        a3v = rng.uniform(-0.5, 0.5)
        b3v = rng.uniform(-0.55, 0.55)
        b4v = rng.uniform(-1/12, 0.25)

        # Check b constraints
        if disc_q_func(b3v, b4v) < 0 or nf2_q_func(b3v, b4v) < 0 or f1_q_func(b3v, b4v) < 0:
            continue

        # Find a₄ on disc_p = 0
        def disc_p_a4(a4v):
            return float(disc_p_func(a3v, a4v))

        # Search for roots of disc_p in a₄ ∈ [-1/12, 1/4]
        a4_grid = np.linspace(-1/12, 0.25, 50)
        vals = [disc_p_a4(a4v) for a4v in a4_grid]
        for k in range(len(vals) - 1):
            if vals[k] * vals[k + 1] < 0:
                try:
                    a4_root = brentq(disc_p_a4, a4_grid[k], a4_grid[k + 1])
                    # Check other constraints
                    if nf2_p_func(a3v, a4_root) >= -1e-10 and f1_p_func(a3v, a4_root) >= -1e-10:
                        val = float(neg_N_func(a3v, a4_root, b3v, b4v))
                        n_found += 1
                        if val < min_val:
                            min_val = val
                except Exception:
                    pass

    print(f"    Points on disc_p = 0 found: {n_found}")
    print(f"    Min -N on face 3: {min_val:.8e}")


def main():
    print("=" * 70)
    print("Taylor expansion approach to -N >= 0")
    print("=" * 70)
    print()

    t0 = time.time()

    # Build -N
    print("Building -N...")
    neg_N, variables, domain_constraints = build_neg_N()
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 1: Taylor analysis at equality point
    print("\n" + "=" * 60)
    print("STEP 1: Taylor expansion at x₀ = (0, 1/12, 0, 1/12)")
    print("=" * 60)
    coeffs, eigenvalues, radius, degree_sums = taylor_analysis(neg_N, variables)
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 2: Boundary analysis
    print("\n" + "=" * 60)
    print("STEP 2: Boundary analysis")
    print("=" * 60)
    boundary_analysis(neg_N, variables, domain_constraints)
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 3: Numerical verification of outer region
    print("\n" + "=" * 60)
    print("STEP 3: Numerical verification of outer region (|δ| > r)")
    print("=" * 60)
    min_grid, min_opt = numerical_verification(
        neg_N, variables, domain_constraints, radius, n_grid=30)
    print(f"  ({time.time()-t0:.1f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Taylor radius: r = {radius:.6f}")
    print(f"  Min -N in outer region (grid): {min_grid:.8e}")
    print(f"  Min -N in outer region (opt):  {min_opt:.8e}")

    if min_grid > 0 and min_opt > 0:
        print(f"\n  CONCLUSION: -N >= 0 on the entire domain!")
        print(f"    Near x₀ (|δ| ≤ {radius:.4f}): by Taylor bound + PD Hessian")
        print(f"    Away from x₀ (|δ| > {radius:.4f}): min -N ≈ {min(min_grid, min_opt):.4e} > 0")
        print(f"\n  Proof status: NUMERICAL VERIFICATION COMPLETE")
        print(f"  (Rigorous proof would need interval arithmetic for the outer region)")
    else:
        print(f"\n  WARNING: Found points where -N may be negative!")

    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
