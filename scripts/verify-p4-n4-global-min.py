#!/usr/bin/env python3
"""Global minimum analysis for the n=4 surplus.

Three checks:
1. Full 4D Hessian at the equality point (a3=b3=0, a4=b4=1/12)
2. Numerical global optimization over the real-rooted domain
3. If Hessian is PSD and numerical min ≈ 0 at equality point → proof structure identified

The perturbation approach (fixing a4,b4, varying a3,b3) failed because the
2D Hessian is not PSD everywhere. But the GLOBAL minimum might still be at
the equality point when all four variables are optimized jointly.
"""

import sympy as sp
from sympy import symbols, Rational, expand, diff, simplify, factor, Matrix
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def build_surplus_symbolic():
    """Build the surplus as a symbolic rational function."""
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    # p: x^4 - x^2 + a3*x + a4  (centered, a2=-1)
    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4
                    - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2

    # q: x^4 - x^2 + b3*x + b4
    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4
                    - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2

    # r = p ⊞_4 q: c2=-2, c3=a3+b3, c4=a4+1/6+b4
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 128*4*c4**2 - 144*(-2)*c3**2*c4
                    - 27*c3**4 + 16*16*c4 - 4*(-8)*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)

    surplus = (-disc_r / (4 * f1_r * f2_r)
               + disc_p / (4 * f1_p * f2_p)
               + disc_q / (4 * f1_q * f2_q))

    return surplus, (a3, a4, b3, b4)


def surplus_numeric(a3, a4, b3, b4):
    """Compute surplus numerically."""
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    disc_p = 256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2

    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2
    disc_q = 256*b4**3 - 128*b4**2 - 144*b3**2*b4 - 27*b3**4 + 16*b4 + 4*b3**2

    c3 = a3 + b3
    c4 = a4 + 1.0/6.0 + b4
    f1_r = 4 + 12*c4
    f2_r = -16 + 16*c4 + 9*c3**2
    disc_r = 256*c4**3 - 512*c4**2 + 288*c3**2*c4 - 27*c3**4 + 256*c4 + 32*c3**2

    denom_p = 4 * f1_p * f2_p
    denom_q = 4 * f1_q * f2_q
    denom_r = 4 * f1_r * f2_r

    if abs(denom_p) < 1e-15 or abs(denom_q) < 1e-15 or abs(denom_r) < 1e-15:
        return np.inf

    inv_phi_p = -disc_p / denom_p
    inv_phi_q = -disc_q / denom_q
    inv_phi_r = -disc_r / denom_r

    return inv_phi_r - inv_phi_p - inv_phi_q


def is_real_rooted_domain(a3, a4, b3, b4):
    """Check if (a3,a4) and (b3,b4) are in the real-rooted domain."""
    # p constraints
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    disc_p = 256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2
    if f1_p <= 0 or f2_p >= 0 or disc_p < 0:
        return False

    # q constraints
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2
    disc_q = 256*b4**3 - 128*b4**2 - 144*b3**2*b4 - 27*b3**4 + 16*b4 + 4*b3**2
    if f1_q <= 0 or f2_q >= 0 or disc_q < 0:
        return False

    return True


def main():
    print("=" * 70)
    print("Global minimum analysis for n=4 surplus")
    print("=" * 70)
    print()

    # ===== STEP 1: Full 4D Hessian at equality point =====
    print("Step 1: Full 4D Hessian at equality point (a3=b3=0, a4=b4=1/12)")
    print("-" * 60)
    t0 = time.time()

    surplus, (a3, a4, b3, b4) = build_surplus_symbolic()
    vars = [a3, a4, b3, b4]
    eq_point = {a3: 0, b3: 0, a4: Rational(1, 12), b4: Rational(1, 12)}

    # Compute gradient at equality point
    grad = [diff(surplus, v).subs(eq_point) for v in vars]
    grad_simplified = [simplify(g) for g in grad]
    print(f"  Gradient at equality point: {grad_simplified}")
    is_critical = all(g == 0 for g in grad_simplified)
    print(f"  Critical point: {is_critical}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    if not is_critical:
        print("  NOT a critical point! Equality point is not a local extremum.")
        print("  This would be very surprising given the symmetry.")
        return

    # Compute 4x4 Hessian
    print("  Computing 4x4 Hessian...")
    H = sp.zeros(4, 4)
    for i in range(4):
        for j in range(i, 4):
            hij = diff(surplus, vars[i], vars[j])
            hij_at_eq = hij.subs(eq_point)
            hij_simplified = simplify(hij_at_eq)
            H[i, j] = hij_simplified
            H[j, i] = hij_simplified
            print(f"    H[{vars[i]},{vars[j]}] = {hij_simplified}")

    print(f"  ({time.time()-t0:.1f}s)")
    print()

    print("  Hessian matrix:")
    sp.pprint(H)
    print()

    # Eigenvalues
    print("  Computing eigenvalues...")
    eigenvals = H.eigenvals()
    print(f"  Eigenvalues: {eigenvals}")

    all_nonneg = all(simplify(ev) >= 0 for ev in eigenvals.keys())
    print(f"  All eigenvalues >= 0: {all_nonneg}")

    # Numerical eigenvalues for clarity
    eigenvals_numeric = [float(ev) for ev in eigenvals.keys()]
    print(f"  Numerical eigenvalues: {sorted(eigenvals_numeric)}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ===== STEP 2: Numerical global optimization =====
    print("Step 2: Numerical global optimization")
    print("-" * 60)

    # The domain for p (a3, a4): f1_p > 0, f2_p < 0, disc_p >= 0
    # At a3=0: a4 ∈ (-1/12, 1/4), disc >= 0 iff 256a4^3 - 128a4^2 + 16a4 >= 0
    # i.e. a4(256a4^2 - 128a4 + 16) >= 0, i.e. a4 * 16(16a4^2 - 8a4 + 1) >= 0
    # i.e. a4 * 16(4a4 - 1)^2 >= 0, so a4 >= 0 (for a3=0 disc >= 0)
    # But for a3 ≠ 0, a4 can be slightly negative.

    # Safe bounds: a4, b4 in [-0.08, 0.249], a3, b3 in [-0.35, 0.35]
    # We'll filter by domain constraints.

    def neg_surplus_constrained(x):
        """Negative surplus for minimization (want to minimize surplus = maximize -surplus)."""
        a3v, a4v, b3v, b4v = x
        if not is_real_rooted_domain(a3v, a4v, b3v, b4v):
            return 1e6  # penalty
        s = surplus_numeric(a3v, a4v, b3v, b4v)
        if np.isnan(s) or np.isinf(s):
            return 1e6
        return -s  # minimize negative surplus = find minimum surplus

    bounds = [(-0.35, 0.35), (-0.08, 0.249), (-0.35, 0.35), (-0.08, 0.249)]

    print("  Running differential evolution (global optimizer)...")
    t1 = time.time()
    result = differential_evolution(
        neg_surplus_constrained, bounds,
        seed=42, maxiter=1000, tol=1e-12, polish=True,
        popsize=30, mutation=(0.5, 1.5), recombination=0.9
    )
    opt_x = result.x
    opt_surplus = -result.fun
    print(f"  Optimization took {time.time()-t1:.1f}s")
    print(f"  Minimum surplus found: {opt_surplus:.15e}")
    print(f"  At: a3={opt_x[0]:.10f}, a4={opt_x[1]:.10f}, "
          f"b3={opt_x[2]:.10f}, b4={opt_x[3]:.10f}")
    print(f"  In domain: {is_real_rooted_domain(*opt_x)}")
    print()

    # Also try from the known equality point
    s_eq = surplus_numeric(0, 1/12, 0, 1/12)
    print(f"  Surplus at equality point (0, 1/12, 0, 1/12): {s_eq:.15e}")
    print()

    # ===== STEP 3: Scan near boundary to find smallest surplus =====
    print("Step 3: Monte Carlo scan for smallest surplus")
    print("-" * 60)

    min_surplus = np.inf
    min_point = None
    valid = 0
    n_trials = 100000

    rng = np.random.default_rng(123)
    for _ in range(n_trials):
        a3v = rng.uniform(-0.35, 0.35)
        a4v = rng.uniform(-0.08, 0.249)
        b3v = rng.uniform(-0.35, 0.35)
        b4v = rng.uniform(-0.08, 0.249)

        if not is_real_rooted_domain(a3v, a4v, b3v, b4v):
            continue
        valid += 1

        s = surplus_numeric(a3v, a4v, b3v, b4v)
        if np.isnan(s) or np.isinf(s):
            continue
        if s < min_surplus:
            min_surplus = s
            min_point = (a3v, a4v, b3v, b4v)

    print(f"  Valid trials: {valid}/{n_trials}")
    print(f"  Minimum surplus: {min_surplus:.15e}")
    if min_point:
        print(f"  At: a3={min_point[0]:.10f}, a4={min_point[1]:.10f}, "
              f"b3={min_point[2]:.10f}, b4={min_point[3]:.10f}")
    print()

    # ===== STEP 4: Profile along a3 for fixed (a4, b4) where Hessian fails =====
    print("Step 4: Surplus profile along a3 (b3=0) for H11-negative points")
    print("-" * 60)

    # At a4=0.2, b4=0.2 the Hessian is likely not PSD
    a4_fixed = 0.2
    b4_fixed = 0.2
    max_a3 = np.sqrt((2 - 8*a4_fixed) / 9)  # f2 < 0 constraint

    print(f"  Fixed a4={a4_fixed}, b4={b4_fixed}, b3=0")
    print(f"  Max |a3| ≈ {max_a3:.6f}")

    a3_vals = np.linspace(-max_a3 * 0.99, max_a3 * 0.99, 51)
    surpluses = []
    for a3v in a3_vals:
        if is_real_rooted_domain(a3v, a4_fixed, 0, b4_fixed):
            s = surplus_numeric(a3v, a4_fixed, 0, b4_fixed)
            surpluses.append((a3v, s))

    if surpluses:
        a3_arr = np.array([x[0] for x in surpluses])
        s_arr = np.array([x[1] for x in surpluses])
        i_min = np.argmin(s_arr)
        print(f"  Surplus at a3=0: {surplus_numeric(0, a4_fixed, 0, b4_fixed):.10f}")
        print(f"  Min surplus along profile: {s_arr[i_min]:.10f} at a3={a3_arr[i_min]:.6f}")
        print(f"  Max surplus along profile: {s_arr.max():.10f}")
        print(f"  All non-negative: {np.all(s_arr >= -1e-12)}")
    print()

    # ===== STEP 5: Assessment =====
    print("=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    print()

    if all_nonneg:
        print("  The full 4D Hessian at the equality point is PSD.")
        print("  The equality point IS a local minimum of the surplus.")
        if opt_surplus >= -1e-10:
            print("  The global minimum is ≈ 0 (at the equality point).")
            print()
            print("  PROOF STRUCTURE: The surplus has a global minimum of 0 at")
            print("  (a3,b3,a4,b4) = (0,0,1/12,1/12), corresponding to both")
            print("  polynomials being x^4 - x^2 + 1/12 (degree-4 semicircular).")
            print()
            print("  For a RIGOROUS proof, we need:")
            print("  1. Hessian PSD (computed symbolically above)")
            print("  2. Surplus → +∞ or bounded away from 0 on the boundary of the domain")
            print("  3. No other critical points with surplus = 0")
            print("     (or: show the equality point is the UNIQUE global minimum)")
        else:
            print(f"  BUT the global minimum ({opt_surplus:.10f}) is NEGATIVE!")
            print("  This would be a COUNTEREXAMPLE to the inequality!")
    else:
        print("  The full 4D Hessian at the equality point is NOT PSD.")
        print("  The equality point is NOT a local minimum.")
        print("  A more sophisticated proof strategy is needed.")


if __name__ == "__main__":
    main()
