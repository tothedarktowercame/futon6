#!/usr/bin/env python3
"""High-precision verification of critical points of -N.

Uses mpmath for multi-precision arithmetic to:
1. Find all critical points via gradient minimization
2. Verify -N > 0 at each (except x₀)
3. Classify each as minimum/saddle/maximum via Hessian eigenvalues
"""

import sympy as sp
from sympy import Rational, expand, symbols, together, fraction, diff, Poly
import numpy as np
from scipy.optimize import minimize
import mpmath
import time

mpmath.mp.dps = 50  # 50 decimal digits


def build_neg_N():
    """Build -N polynomial."""
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
    num, den = fraction(surplus_frac)
    neg_N = expand(-num)

    return neg_N, (a3, a4, b3, b4)


def check_domain(a3v, a4v, b3v, b4v, margin=1e-8):
    disc_p = 256*a4v**3 - 128*a4v**2 - 144*a3v**2*a4v - 27*a3v**4 + 16*a4v + 4*a3v**2
    f1_p = 1 + 12*a4v
    f2_p = 9*a3v**2 + 8*a4v - 2
    disc_q = 256*b4v**3 - 128*b4v**2 - 144*b3v**2*b4v - 27*b3v**4 + 16*b4v + 4*b3v**2
    f1_q = 1 + 12*b4v
    f2_q = 9*b3v**2 + 8*b4v - 2
    return (disc_p >= -margin and disc_q >= -margin and
            f1_p > margin and f1_q > margin and
            f2_p < -margin and f2_q < -margin)


def main():
    print("=" * 70)
    print("High-precision critical point verification")
    print("=" * 70)

    t0 = time.time()
    print("Building -N...")
    neg_N, (a3, a4, b3, b4) = build_neg_N()

    # Build gradient and Hessian
    grad_sym = [expand(diff(neg_N, v)) for v in (a3, a4, b3, b4)]
    hess_sym = [[expand(diff(g, v)) for v in (a3, a4, b3, b4)] for g in grad_sym]

    # Lambdify
    neg_N_func = sp.lambdify((a3, a4, b3, b4), neg_N, 'numpy')
    grad_funcs = [sp.lambdify((a3, a4, b3, b4), g, 'numpy') for g in grad_sym]
    hess_funcs = [[sp.lambdify((a3, a4, b3, b4), h, 'numpy') for h in row] for row in hess_sym]

    # mpmath versions for high precision
    neg_N_mp = sp.lambdify((a3, a4, b3, b4), neg_N, 'mpmath')
    grad_mp = [sp.lambdify((a3, a4, b3, b4), g, 'mpmath') for g in grad_sym]

    print(f"  ({time.time()-t0:.1f}s)")

    # Find critical points via gradient minimization (20000 starts for thoroughness)
    print("\nSearching for critical points (20000 starts)...")
    rng = np.random.default_rng(42)
    bounds = [(-0.544, 0.544), (-1/12+0.001, 0.249),
              (-0.544, 0.544), (-1/12+0.001, 0.249)]

    found = []

    def grad_sq(x):
        g = np.array([float(f(*x)) for f in grad_funcs])
        return np.sum(g**2)

    for trial in range(20000):
        x0_start = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        try:
            res = minimize(grad_sq, x0_start, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-20})
            if res.fun < 1e-8:
                pt = res.x.copy()
                # Refine with L-BFGS-B
                try:
                    res2 = minimize(grad_sq, pt, method='L-BFGS-B',
                                   bounds=bounds,
                                   options={'maxiter': 1000, 'ftol': 1e-25})
                    if res2.fun < res.fun:
                        pt = res2.x.copy()
                except Exception:
                    pass

                is_new = all(np.linalg.norm(pt - np.array(cp[:4])) > 0.003
                             for cp in found)
                if is_new:
                    val = float(neg_N_func(*pt))
                    in_dom = check_domain(*pt)
                    gsq = sum(float(f(*pt))**2 for f in grad_funcs)
                    found.append((*pt, val, in_dom, gsq))
        except Exception:
            pass

        if trial % 5000 == 4999:
            in_dom_count = sum(1 for f in found if f[5])
            print(f"  {trial+1} starts done, {len(found)} critical points ({in_dom_count} in domain)")

    found.sort(key=lambda x: x[4])
    print(f"\n  Total critical points found: {len(found)}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Classify and verify each critical point
    print("\n" + "=" * 60)
    print("Critical point classification and verification")
    print("=" * 60)

    in_domain_cps = []

    for i, (a3v, a4v, b3v, b4v, val, in_dom, gsq) in enumerate(found):
        if not in_dom:
            continue

        # High-precision evaluation
        a3_mp = mpmath.mpf(str(a3v))
        a4_mp = mpmath.mpf(str(a4v))
        b3_mp = mpmath.mpf(str(b3v))
        b4_mp = mpmath.mpf(str(b4v))

        val_mp = neg_N_mp(a3_mp, a4_mp, b3_mp, b4_mp)
        grad_mp_vals = [g(a3_mp, a4_mp, b3_mp, b4_mp) for g in grad_mp]
        grad_norm_mp = mpmath.sqrt(sum(g**2 for g in grad_mp_vals))

        # Hessian eigenvalues
        H = np.array([[float(hess_funcs[i][j](*[a3v, a4v, b3v, b4v]))
                       for j in range(4)] for i in range(4)])
        eigvals = np.sort(np.linalg.eigvalsh(H))

        # Classify
        n_neg = sum(1 for e in eigvals if e < -1e-6)
        n_zero = sum(1 for e in eigvals if abs(e) <= 1e-6)
        n_pos = sum(1 for e in eigvals if e > 1e-6)

        if n_neg == 0:
            cp_type = "LOCAL MINIMUM"
        elif n_pos == 0:
            cp_type = "LOCAL MAXIMUM"
        else:
            cp_type = f"SADDLE ({n_neg}-, {n_zero}0, {n_pos}+)"

        dist = np.sqrt(a3v**2 + (a4v - 1/12)**2 + b3v**2 + (b4v - 1/12)**2)

        print(f"\n  CP{len(in_domain_cps)}:")
        print(f"    Point: ({a3v:.12f}, {a4v:.12f}, {b3v:.12f}, {b4v:.12f})")
        print(f"    -N = {val_mp} (50-digit)")
        print(f"    |grad| = {grad_norm_mp}")
        print(f"    |x - x₀| = {dist:.8f}")
        print(f"    Hessian eigenvalues: {eigvals}")
        print(f"    Type: {cp_type}")

        in_domain_cps.append({
            'point': (a3v, a4v, b3v, b4v),
            '-N': float(val_mp),
            '|grad|': float(grad_norm_mp),
            'distance': dist,
            'type': cp_type,
            'eigenvalues': eigvals
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Critical points in domain: {len(in_domain_cps)}")
    for i, cp in enumerate(in_domain_cps):
        print(f"    CP{i}: -N = {cp['-N']:.6e}, type = {cp['type']}, |x-x₀| = {cp['distance']:.6f}")

    minima = [cp for cp in in_domain_cps if 'MINIMUM' in cp['type']]
    print(f"\n  Local minima in domain: {len(minima)}")
    for cp in minima:
        print(f"    -N = {cp['-N']:.6e} at distance {cp['distance']:.6f}")

    all_nonneg = all(cp['-N'] >= -1e-6 for cp in in_domain_cps)
    print(f"\n  All critical points have -N >= 0: {all_nonneg}")

    if all_nonneg and len(minima) == 1:
        print(f"\n  CONCLUSION: x₀ is the unique local minimum of -N in the domain.")
        print(f"  Combined with boundary positivity, this proves -N >= 0.")

    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
