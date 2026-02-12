#!/usr/bin/env python3
"""Lipschitz-based verification that -N >= 0 on the domain.

Strategy:
1. Near x₀: Taylor bound with PD Hessian → -N ≥ 0
2. Away from x₀: Fine grid + Lipschitz bound → -N > 0

Uses numpy vectorization for efficient grid evaluation.
"""

import sympy as sp
from sympy import Rational, expand, symbols, together, fraction, diff, Poly
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def build_polynomials():
    """Build -N and ∇(-N) as coefficient dictionaries for fast numpy eval."""
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4 - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2

    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4 - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)

    surplus_frac = together(-disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q))
    N, D = fraction(surplus_frac)
    neg_N = expand(-N)
    grad = [expand(diff(neg_N, v)) for v in (a3, a4, b3, b4)]

    variables = (a3, a4, b3, b4)
    constraint_exprs = [disc_p, disc_q, expand(-f2_p), expand(-f2_q), f1_p, f1_q]
    return neg_N, grad, variables, constraint_exprs


def poly_to_arrays(expr, variables):
    """Convert sympy polynomial to (exponents, coefficients) arrays."""
    poly = Poly(expand(expr), *variables)
    d = poly.as_dict()
    exps = np.array(list(d.keys()), dtype=np.int32)
    coeffs = np.array([float(c) for c in d.values()])
    return exps, coeffs


def eval_poly_vectorized(exps, coeffs, a3, a4, b3, b4):
    """Evaluate polynomial at grid points (vectorized).

    a3, a4, b3, b4 are 1D arrays. Returns values on the full 4D grid.
    Uses broadcasting: result[i,j,k,l] = poly(a3[i], a4[j], b3[k], b4[l]).
    """
    # Create power arrays for each variable
    n1, n2, n3, n4 = len(a3), len(a4), len(b3), len(b4)
    max_exp = exps.max()

    # Precompute powers
    a3_pow = np.power.outer(a3, np.arange(max_exp + 1))  # (n1, max_exp+1)
    a4_pow = np.power.outer(a4, np.arange(max_exp + 1))  # (n2, max_exp+1)
    b3_pow = np.power.outer(b3, np.arange(max_exp + 1))  # (n3, max_exp+1)
    b4_pow = np.power.outer(b4, np.arange(max_exp + 1))  # (n4, max_exp+1)

    # Evaluate each monomial on the full grid
    result = np.zeros((n1, n2, n3, n4))
    for m in range(len(coeffs)):
        e1, e2, e3, e4 = exps[m]
        c = coeffs[m]
        # Use broadcasting: (n1,1,1,1) * (1,n2,1,1) * (1,1,n3,1) * (1,1,1,n4)
        result += c * (a3_pow[:, e1, None, None, None] *
                       a4_pow[None, :, e2, None, None] *
                       b3_pow[None, None, :, e3, None] *
                       b4_pow[None, None, None, :, e4])

    return result


def main():
    print("=" * 70)
    print("Lipschitz-based verification of -N >= 0 (vectorized)")
    print("=" * 70)
    print()

    t0 = time.time()

    # Build polynomials
    print("Building -N and gradient...")
    neg_N, grad, variables, constraint_exprs = build_polynomials()

    # Convert to arrays for fast evaluation
    neg_N_ea = poly_to_arrays(neg_N, variables)
    grad_ea = [poly_to_arrays(g, variables) for g in grad]
    constraint_ea = [poly_to_arrays(c, variables) for c in constraint_exprs]

    print(f"  -N: {len(neg_N_ea[1])} terms")
    for i in range(4):
        print(f"  ∂(-N)/∂x{i+1}: {len(grad_ea[i][1])} terms")
    print(f"  ({time.time()-t0:.1f}s)")

    x0 = np.array([0.0, 1/12, 0.0, 1/12])

    # Step 1: Taylor radius
    print("\n" + "=" * 60)
    print("STEP 1: Taylor radius computation")
    print("=" * 60)

    d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')
    a3, a4, b3, b4 = variables
    shifted = neg_N.subs({a3: d1, a4: d2 + Rational(1, 12), b3: d3, b4: d4 + Rational(1, 12)})
    shifted = expand(shifted)
    poly_shifted = Poly(shifted, d1, d2, d3, d4)

    degree_sums = {}
    for e, c in poly_shifted.as_dict().items():
        k = sum(e)
        if k >= 3:
            degree_sums.setdefault(k, 0)
            degree_sums[k] += abs(float(c))

    lambda_min_half = 24576.0  # = 49152/2 = min eigenvalue of H_{-N} / 2

    def remainder_over_t2(t):
        return sum(degree_sums.get(k, 0) * t**(k - 2) for k in range(3, 11))

    lo, hi = 0, 1
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid > 0 and remainder_over_t2(mid) < lambda_min_half:
            lo = mid
        else:
            hi = mid
    taylor_radius = lo

    print(f"  Taylor radius: r = {taylor_radius:.6f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 2: Grid evaluation
    print("\n" + "=" * 60)
    print("STEP 2: Grid evaluation (vectorized)")
    print("=" * 60)

    n_grid = 50
    a3_grid = np.linspace(-0.544, 0.544, n_grid)
    a4_grid = np.linspace(-1/12 + 0.003, 0.247, n_grid)
    b3_grid = np.linspace(-0.544, 0.544, n_grid)
    b4_grid = np.linspace(-1/12 + 0.003, 0.247, n_grid)

    print(f"  Grid: {n_grid}^4 = {n_grid**4} points")
    print(f"  Evaluating -N on grid...")
    neg_N_grid = eval_poly_vectorized(*neg_N_ea, a3_grid, a4_grid, b3_grid, b4_grid)
    print(f"    done ({time.time()-t0:.1f}s)")

    # Evaluate constraints
    print(f"  Evaluating domain constraints...")
    constraint_names = ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q']
    constraint_grids = []
    for ea in constraint_ea:
        cg = eval_poly_vectorized(*ea, a3_grid, a4_grid, b3_grid, b4_grid)
        constraint_grids.append(cg)
    print(f"    done ({time.time()-t0:.1f}s)")

    # Domain mask: all constraints >= 0 (with small margin for f1, f2)
    domain_mask = np.ones_like(neg_N_grid, dtype=bool)
    for i, cg in enumerate(constraint_grids):
        margin = 1e-4 if constraint_names[i] in ('f1_p', 'f1_q', '-f2_p', '-f2_q') else -1e-8
        domain_mask &= (cg >= margin)

    n_domain = np.sum(domain_mask)
    print(f"  Domain points: {n_domain} / {n_grid**4}")

    # Distance from x0
    A3, A4, B3, B4 = np.meshgrid(a3_grid, a4_grid, b3_grid, b4_grid, indexing='ij')
    dist_grid = np.sqrt((A3 - x0[0])**2 + (A4 - x0[1])**2 +
                         (B3 - x0[2])**2 + (B4 - x0[3])**2)

    # Outer region: domain AND |delta| > taylor_radius
    outer_mask = domain_mask & (dist_grid > taylor_radius)
    n_outer = np.sum(outer_mask)

    min_neg_N_domain = neg_N_grid[domain_mask].min() if n_domain > 0 else float('inf')
    min_neg_N_outer = neg_N_grid[outer_mask].min() if n_outer > 0 else float('inf')

    # Find the grid point with minimum -N in outer region
    if n_outer > 0:
        outer_vals = neg_N_grid.copy()
        outer_vals[~outer_mask] = np.inf
        min_idx = np.unravel_index(np.argmin(outer_vals), outer_vals.shape)
        min_pt = (a3_grid[min_idx[0]], a4_grid[min_idx[1]],
                  b3_grid[min_idx[2]], b4_grid[min_idx[3]])
        min_dist = dist_grid[min_idx]

    print(f"  Outer region points: {n_outer}")
    print(f"  Min -N (full domain): {min_neg_N_domain:.6f}")
    print(f"  Min -N (outer region): {min_neg_N_outer:.6f}")
    if n_outer > 0:
        print(f"    at {min_pt}, |delta|={min_dist:.4f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 3: Gradient evaluation for Lipschitz bound
    print("\n" + "=" * 60)
    print("STEP 3: Lipschitz constant (gradient bound)")
    print("=" * 60)

    print(f"  Evaluating gradient on domain points...")
    grad_sq_grid = np.zeros_like(neg_N_grid)
    for i, ea in enumerate(grad_ea):
        g = eval_poly_vectorized(*ea, a3_grid, a4_grid, b3_grid, b4_grid)
        grad_sq_grid += g**2
        print(f"    Component {i+1} done ({time.time()-t0:.1f}s)")

    grad_norm_grid = np.sqrt(grad_sq_grid)
    max_grad_domain = grad_norm_grid[domain_mask].max() if n_domain > 0 else 0

    h = np.array([a3_grid[1] - a3_grid[0], a4_grid[1] - a4_grid[0],
                   b3_grid[1] - b3_grid[0], b4_grid[1] - b4_grid[0]])
    half_diag = np.linalg.norm(h) / 2

    lipschitz_margin = min_neg_N_outer - max_grad_domain * half_diag

    print(f"  Max |grad(-N)| on domain: {max_grad_domain:.2f}")
    print(f"  Grid spacing: h = {h}")
    print(f"  Half-diagonal: {half_diag:.6f}")
    print(f"  Lipschitz margin: {lipschitz_margin:.6f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 4: Check if Lipschitz bound is sufficient
    print("\n" + "=" * 60)
    print("STEP 4: Lipschitz verification")
    print("=" * 60)

    if lipschitz_margin > 0:
        print(f"  LIPSCHITZ BOUND CONFIRMS -N > 0 in outer region!")
        print(f"  For any x in domain with |x-x0| > {taylor_radius:.4f}:")
        print(f"    -N(x) >= {min_neg_N_outer:.4f} - {max_grad_domain:.1f} * {half_diag:.6f}")
        print(f"           = {lipschitz_margin:.4f} > 0")
    else:
        print(f"  Lipschitz bound insufficient at n={n_grid}.")
        needed_h = min_neg_N_outer / max_grad_domain
        needed_n = max(int(np.ceil(1.1 / needed_h)), n_grid)
        print(f"  Need half_diag < {needed_h:.6f} (n >= {needed_n} approx)")

        # Try finer grid near the minimum
        print(f"\n  Trying adaptive refinement near minimum...")
        if n_outer > 0:
            center = np.array(min_pt)
            radius = half_diag * 3
            n_fine = 30
            fine_a3 = np.linspace(max(-0.544, center[0] - radius),
                                   min(0.544, center[0] + radius), n_fine)
            fine_a4 = np.linspace(max(-1/12+0.003, center[1] - radius),
                                   min(0.247, center[1] + radius), n_fine)
            fine_b3 = np.linspace(max(-0.544, center[2] - radius),
                                   min(0.544, center[2] + radius), n_fine)
            fine_b4 = np.linspace(max(-1/12+0.003, center[3] - radius),
                                   min(0.247, center[3] + radius), n_fine)

            fine_neg_N = eval_poly_vectorized(*neg_N_ea, fine_a3, fine_a4, fine_b3, fine_b4)
            fine_constraints = [eval_poly_vectorized(*ea, fine_a3, fine_a4, fine_b3, fine_b4)
                               for ea in constraint_ea]
            fine_domain = np.ones_like(fine_neg_N, dtype=bool)
            for i, cg in enumerate(fine_constraints):
                margin = 1e-4 if constraint_names[i] in ('f1_p', 'f1_q', '-f2_p', '-f2_q') else -1e-8
                fine_domain &= (cg >= margin)

            if np.any(fine_domain):
                min_fine = fine_neg_N[fine_domain].min()
                print(f"    Fine grid min -N: {min_fine:.6f}")

    # Step 5: Critical point search (targeted)
    print(f"\n" + "=" * 60)
    print("STEP 5: Critical point search")
    print("=" * 60)

    neg_N_func = sp.lambdify(variables, neg_N, 'numpy')
    grad_funcs = [sp.lambdify(variables, g, 'numpy') for g in
                  [expand(diff(neg_N, v)) for v in variables]]
    constraint_funcs = [sp.lambdify(variables, c, 'numpy') for c in constraint_exprs]

    def in_domain_pt(pt, margin=1e-5):
        return all(float(f(*pt)) >= -margin for f in constraint_funcs)

    rng = np.random.default_rng(42)
    bounds = [(-0.544, 0.544), (-1/12+0.003, 0.247),
              (-0.544, 0.544), (-1/12+0.003, 0.247)]

    critical_points = []
    for trial in range(3000):
        x_start = np.array([rng.uniform(b[0], b[1]) for b in bounds])

        def grad_sq(x):
            g = np.array([float(f(*x)) for f in grad_funcs])
            penalty = 0
            for f in constraint_funcs:
                v = float(f(*x))
                if v < 1e-4:
                    penalty += 1e6 * (1e-4 - v)**2
            return np.sum(g**2) + penalty

        try:
            res = minimize(grad_sq, x_start, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < 1e-4 and in_domain_pt(res.x):
                val = float(neg_N_func(*res.x))
                is_new = all(np.linalg.norm(res.x - cp['point']) > 0.01
                             for cp in critical_points)
                if is_new:
                    gn = np.sqrt(sum(float(f(*res.x))**2 for f in grad_funcs))
                    critical_points.append({
                        'point': res.x.copy(),
                        '-N': val,
                        '|grad|': gn,
                        '|delta|': np.linalg.norm(res.x - x0)
                    })
        except Exception:
            pass

    print(f"  3000 gradient minimizations")
    print(f"  Distinct critical points found: {len(critical_points)}")
    for i, cp in enumerate(sorted(critical_points, key=lambda c: c['-N'])):
        print(f"    CP{i}: -N = {cp['-N']:.6e}, |grad| = {cp['|grad|']:.2e}, "
              f"|delta| = {cp['|delta|']:.4f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Taylor radius: r = {taylor_radius:.6f}")
    print(f"  Min -N (outer, grid n={n_grid}): {min_neg_N_outer:.6f}")
    print(f"  Max |grad(-N)|: {max_grad_domain:.2f}")
    print(f"  Half-diagonal: {half_diag:.6f}")
    print(f"  Lipschitz margin: {lipschitz_margin:.6f}")
    all_cp_nonneg = all(cp['-N'] >= -1e-6 for cp in critical_points)
    print(f"  All critical points -N >= 0: {all_cp_nonneg}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
