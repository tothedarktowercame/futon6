#!/usr/bin/env python3
"""Perturbation analysis: does the n=4 surplus INCREASE when a3, b3 move away from 0?

If the surplus is minimized at a3=b3=0 for each fixed (a4, b4), then
the general case reduces to the symmetric case (which is proved).

Two checks:
1. Is a3=b3=0 a critical point of the surplus for fixed (a4, b4)?
2. Is the Hessian in (a3, b3) positive semidefinite at a3=b3=0?

If both hold, the proof is complete.
"""

import sympy as sp
from sympy import symbols, Rational, expand, diff, simplify, factor, Poly
import time


def main():
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    print("=" * 70)
    print("Perturbation analysis: surplus behavior in (a3, b3) at a3=b3=0")
    print("=" * 70)
    print()

    # --- Build the surplus as a rational function ---
    # With a2 = b2 = -1:

    # Discriminant for p (a2=-1):
    disc_p = (256*a4**3 - 128*a4**2 + 144*(-1)*a3**2*a4
              - 27*a3**4 + 16*a4 - 4*(-1)*a3**2)
    disc_p = expand(disc_p)

    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    # On real-rooted cone: f1_p > 0, f2_p < 0
    # 1/Phi_p = -disc_p / (4 * f1_p * f2_p)

    # Same for q (b2=-1):
    disc_q = (256*b4**3 - 128*b4**2 + 144*(-1)*b3**2*b4
              - 27*b3**4 + 16*b4 - 4*(-1)*b3**2)
    disc_q = expand(disc_q)

    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2

    # Convolution: c2=-2, c3=a3+b3, c4=a4+1/6+b4
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4

    disc_r = (256*c4**3 - 128*(-2)**2*c4**2 + 144*(-2)*c3**2*c4
              - 27*c3**4 + 16*(-2)**4*c4 - 4*(-2)**3*c3**2)
    disc_r = expand(disc_r)

    f1_r = (-2)**2 + 12*c4
    f1_r = expand(f1_r)
    f2_r = 2*(-2)**3 - 8*(-2)*c4 + 9*c3**2
    f2_r = expand(f2_r)

    # Surplus = -disc_r/(4*f1_r*f2_r) - (-disc_p/(4*f1_p*f2_p)) - (-disc_q/(4*f1_q*f2_q))
    #         = -disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q)

    # Work with the surplus as a fraction: surplus = N/D where D > 0 on the cone.
    # We want to show surplus >= 0, i.e., N >= 0.

    # Rather than the huge 233-term numerator, let's work directly with the
    # rational function and take derivatives symbolically.

    surplus = (-disc_r / (4 * f1_r * f2_r)
               + disc_p / (4 * f1_p * f2_p)
               + disc_q / (4 * f1_q * f2_q))

    print("Step 1: Check gradient at a3=b3=0")
    print("-" * 40)

    t0 = time.time()

    # d(surplus)/d(a3) at a3=b3=0
    ds_da3 = diff(surplus, a3)
    ds_da3_at_0 = ds_da3.subs([(a3, 0), (b3, 0)])
    ds_da3_at_0 = simplify(ds_da3_at_0)
    print(f"  d(surplus)/d(a3)|_(a3=b3=0) = {ds_da3_at_0}")
    print(f"  ({time.time()-t0:.1f}s)")

    # d(surplus)/d(b3) at a3=b3=0
    ds_db3 = diff(surplus, b3)
    ds_db3_at_0 = ds_db3.subs([(a3, 0), (b3, 0)])
    ds_db3_at_0 = simplify(ds_db3_at_0)
    print(f"  d(surplus)/d(b3)|_(a3=b3=0) = {ds_db3_at_0}")
    print(f"  ({time.time()-t0:.1f}s)")

    is_critical = (ds_da3_at_0 == 0) and (ds_db3_at_0 == 0)
    print(f"  Critical point at a3=b3=0: {is_critical}")
    print()

    if not is_critical:
        print("  NOT a critical point — perturbation approach fails in this form.")
        print("  The surplus is not minimized at a3=b3=0 for general (a4,b4).")
        print()
        # But maybe it's still a minimum due to the reflection symmetry
        # (surplus is even in a3 and b3). If so, the gradient must be 0.
        print("  Note: surplus is invariant under a3->-a3, b3->-b3,")
        print("  so d/d(a3) and d/d(b3) should vanish at a3=b3=0 by symmetry.")
        print("  If they don't, there may be a computation error.")
        return

    print("Step 2: Compute Hessian in (a3, b3) at a3=b3=0")
    print("-" * 40)

    # H = [[d2/da3^2, d2/da3db3], [d2/da3db3, d2/db3^2]]
    d2_da3da3 = diff(ds_da3, a3)
    d2_da3da3_at_0 = d2_da3da3.subs([(a3, 0), (b3, 0)])
    d2_da3da3_at_0 = simplify(d2_da3da3_at_0)
    print(f"  d2(surplus)/d(a3)^2|_0 = {d2_da3da3_at_0}")
    print(f"  ({time.time()-t0:.1f}s)")

    d2_da3db3 = diff(ds_da3, b3)
    d2_da3db3_at_0 = d2_da3db3.subs([(a3, 0), (b3, 0)])
    d2_da3db3_at_0 = simplify(d2_da3db3_at_0)
    print(f"  d2(surplus)/d(a3)d(b3)|_0 = {d2_da3db3_at_0}")
    print(f"  ({time.time()-t0:.1f}s)")

    d2_db3db3 = diff(ds_db3, b3)
    d2_db3db3_at_0 = d2_db3db3.subs([(a3, 0), (b3, 0)])
    d2_db3db3_at_0 = simplify(d2_db3db3_at_0)
    print(f"  d2(surplus)/d(b3)^2|_0 = {d2_db3db3_at_0}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    print("Step 3: Analyze Hessian positive-semidefiniteness")
    print("-" * 40)

    H11 = d2_da3da3_at_0
    H12 = d2_da3db3_at_0
    H22 = d2_db3db3_at_0

    print(f"  H = [[{H11}, {H12}], [{H12}, {H22}]]")
    print()

    # For PSD: need H11 >= 0 and H11*H22 - H12^2 >= 0
    # These are functions of (a4, b4) only.
    print("  For PSD, need:")
    print(f"    H11 = {H11} >= 0")

    det_H = simplify(H11 * H22 - H12**2)
    print(f"    det(H) = {det_H} >= 0")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Factor them
    print("  Factoring H11...")
    H11_factored = factor(H11)
    print(f"    H11 = {H11_factored}")

    print("  Factoring det(H)...")
    det_factored = factor(det_H)
    print(f"    det(H) = {det_factored}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Numerical check
    print("Step 4: Numerical check on real-rooted domain")
    print("-" * 40)

    import numpy as np

    H11_func = sp.lambdify((a4, b4), H11, 'numpy')
    H12_func = sp.lambdify((a4, b4), H12, 'numpy')
    H22_func = sp.lambdify((a4, b4), H22, 'numpy')

    h11_neg = 0
    det_neg = 0
    valid = 0

    for trial in range(10000):
        a4_val = np.random.uniform(0.001, 0.249)
        b4_val = np.random.uniform(0.001, 0.249)

        h11 = float(H11_func(a4_val, b4_val))
        h12 = float(H12_func(a4_val, b4_val))
        h22 = float(H22_func(a4_val, b4_val))

        valid += 1
        if h11 < -1e-10:
            h11_neg += 1
        det = h11 * h22 - h12**2
        if det < -1e-10:
            det_neg += 1

    print(f"  Trials: {valid}")
    print(f"  H11 < 0: {h11_neg}")
    print(f"  det(H) < 0: {det_neg}")
    print(f"  Hessian PSD: {h11_neg == 0 and det_neg == 0}")
    print()

    if h11_neg == 0 and det_neg == 0:
        print("  ** If Hessian is PSD for all (a4,b4) in the real-rooted domain,")
        print("     then the minimum of surplus over (a3,b3) is at a3=b3=0,")
        print("     reducing the general case to the proved symmetric case. **")
    else:
        print("  Hessian is NOT PSD everywhere — perturbation approach")
        print("  does not immediately reduce to the symmetric case.")
        print("  The surplus minimum over (a3,b3) is not always at zero.")


if __name__ == "__main__":
    main()
