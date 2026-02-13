#!/usr/bin/env python3
"""Compute K_T2R unified numerator and analyze in (u,v) coordinates.

K_T2R_num = 2L·f2p·f2q·f2c + r·f1p·f1q·f1c·R_surplus_num
where:
  T2_surplus = 2L / (9r·f1p·f1q·f1c)  [f1p=(1+3x), f1q=r²(1+3y)]
  R_surplus = R_surplus_num / (9·f2p·f2q·f2c)
  T2+R surplus = K_T2R_num / (9r·f1p·f1q·f1c·f2p·f2q·f2c)

This gives us K_T2R_num as a polynomial we can analyze.
"""

import numpy as np
import sympy as sp
from sympy import symbols, expand, Poly, factor, cancel
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    r, x, y, p, q = symbols('r x y p q')
    u, v = symbols('u v')

    pr("Building components...")

    f1p = 1 + 3*x
    f2p = 2*(1-x) - 9*p**2
    f1q = r**2*(1 + 3*y)
    f2q = 2*r**3*(1-y) - 9*q**2

    Sv = 1 + r
    Av12 = 3*x + 3*y*r**2 + 2*r
    C_c = expand(Av12/3 - Sv**2)
    f1c = expand(Sv**2 + Av12)
    f2c = expand(2*Sv**3 - 2*Sv*Av12/3 - 9*(p+q)**2)
    C_p = x - 1
    C_q = r**2*(y-1)
    f1q_bare = 1 + 3*y  # f1q without the r² factor

    L = 9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2
    W = 3*r**2*y - 3*r**2 - 4*r + 3*x - 3

    # R_surplus_num (92 terms, degree 4 in p,q)
    R_num = expand(C_c*f1c*f2p*f2q - C_p*f1p*f2c*f2q - C_q*f1q*f2c*f2p)
    pr(f"  R_surplus_num: {len(Poly(R_num, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    # K_T2R unified numerator:
    # T2_surplus = 4rL / (18·f1p·f1q·f1c)  = 2L / (9·r·f1p_bare·f1q_bare·f1c)
    # where f1p_bare = 1+3x, f1q = r²(1+3y) so f1p·f1q = r²(1+3x)(1+3y)
    # T2_surplus = 4rL / (18·(1+3x)·r²(1+3y)·f1c) = 2L / (9r(1+3x)(1+3y)f1c)
    #
    # R_surplus = R_num / (9·f2p·f2q·f2c)
    #
    # Sum = [2L·f2p·f2q·f2c + r·(1+3x)(1+3y)·f1c·R_num] / [9r·(1+3x)(1+3y)·f1c·f2p·f2q·f2c]

    pr("Computing K_T2R unified numerator...")
    K_num = expand(2*L*f2p*f2q*f2c + r*(1+3*x)*(1+3*y)*f1c*R_num)
    K_poly = Poly(K_num, r, x, y, p, q)
    pr(f"  K_num: {len(K_poly.as_dict())} terms [{time.time()-t0:.1f}s]")

    # Check (p,q) degree structure
    K_pq = Poly(K_num, p, q)
    pr(f"  Degree in (p,q): {K_pq.total_degree()}")

    coeffs = {}
    for monom, coeff in K_pq.as_dict().items():
        deg = monom[0] + monom[1]
        if deg not in coeffs:
            coeffs[deg] = {}
        coeffs[deg][monom] = coeff

    for deg in sorted(coeffs.keys()):
        total_terms = sum(len(Poly(c, r, x, y).as_dict()) for c in coeffs[deg].values())
        pr(f"  (p,q)-degree {deg}: {len(coeffs[deg])} monomial types, {total_terms} terms")

    # Check r divisibility
    K_r0 = K_num.subs(r, 0)
    if expand(K_r0) == 0:
        pr("  K_num divisible by r")
        K_over_r = cancel(K_num / r)
        K_r0_2 = expand(K_over_r.subs(r, 0))
        if K_r0_2 == 0:
            pr("  K_num divisible by r²")
            K_red = expand(cancel(K_num / r**2))
        else:
            K_red = expand(K_over_r)
            pr("  K_num/r NOT divisible by r again")
    else:
        K_red = K_num
        pr("  K_num NOT divisible by r")

    n_red = len(Poly(K_red, r, x, y, p, q).as_dict())
    pr(f"  K_red (after r division): {n_red} terms [{time.time()-t0:.1f}s]")

    # Numerical verification
    pr("\n  Numerical verification of K_red ≥ 0...")
    K_fn = sp.lambdify((r, x, y, p, q), K_red, 'numpy')
    L_fn = sp.lambdify((r, x, y), L, 'numpy')

    rng = np.random.default_rng(789)
    neg_count = 0
    n_test = 0
    min_K = np.inf

    for _ in range(200000):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))

        f2p_max = 2*(1-xv)
        f2q_max = 2*rv**3*(1-yv)
        p_max = np.sqrt(max(f2p_max/9, 0))
        q_max = np.sqrt(max(f2q_max/9, 0))

        pv = float(rng.uniform(-p_max, p_max))
        qv = float(rng.uniform(-q_max, q_max))

        f2c_v = 2*(1+rv)**3 - 2*(1+rv)*(3*xv+3*yv*rv**2+2*rv)/3 - 9*(pv+qv)**2
        if f2c_v <= 0:
            continue

        n_test += 1
        Kv = float(K_fn(rv, xv, yv, pv, qv))
        if Kv < min_K:
            min_K = Kv
        if Kv < -1e-6:
            neg_count += 1

    pr(f"  Tested: {n_test} feasible samples")
    pr(f"  K_red < 0: {neg_count}")
    pr(f"  Min K_red: {min_K:.6e}")

    # Degree 6 part analysis
    pr(f"\n{'='*72}")
    pr("DEGREE 6 PART OF K_red")
    pr('='*72)

    if 6 in coeffs:
        d6_reconstructed = sp.Integer(0)
        for monom, coeff in coeffs[6].items():
            d6_reconstructed += coeff * p**monom[0] * q**monom[1]
        d6_reconstructed = expand(d6_reconstructed)

        # Expected: proportional to -1296rL·p²q²(p+q)² (possibly with extra r factors)
        d6_test = expand(-1296*r*L*p**2*q**2*(p+q)**2)
        # After r² division, this would be -1296L·p²q²(p+q)²/r

        # Check if d6 factored form holds
        for monom, coeff in coeffs[6].items():
            f = factor(coeff)
            pr(f"  p^{monom[0]}q^{monom[1]}: {f}")

    # (u,v) coordinate analysis
    pr(f"\n{'='*72}")
    pr("K_red in (u=p+q, v=p-q) coordinates")
    pr('='*72)

    pr("  Substituting p=(u+v)/2, q=(u-v)/2...")
    K_uv = K_red.subs([(p, (u+v)/2), (q, (u-v)/2)])
    K_uv_exp = expand(K_uv)
    K_uv_poly = Poly(K_uv_exp, u, v)
    pr(f"  Done [{time.time()-t0:.1f}s]")

    pr("\n  Coefficients of u^i·v^j:")
    for monom in sorted(K_uv_poly.as_dict().keys()):
        c = K_uv_poly.as_dict()[monom]
        if c != 0:
            n_t = len(Poly(c, r, x, y).as_dict())
            c_f = factor(c)
            s = str(c_f)
            if len(s) > 150:
                s = s[:150] + "..."
            pr(f"    u^{monom[0]}v^{monom[1]}: {n_t} terms, = {s}")

    # Check which coefficients factor through W or L
    pr(f"\n  Factorization through W = 3r²y-3r²-4r+3x-3:")
    for monom in sorted(K_uv_poly.as_dict().keys()):
        c = K_uv_poly.as_dict()[monom]
        if c == 0:
            continue
        c_exp = expand(c)
        # Check divisibility by W
        q_w, r_w = sp.div(Poly(c_exp, r, x, y), Poly(W, r, x, y))
        has_W = (r_w == Poly(0, r, x, y))
        # Check divisibility by L
        q_L, r_L = sp.div(Poly(c_exp, r, x, y), Poly(L, r, x, y))
        has_L = (r_L == Poly(0, r, x, y))
        pr(f"    u^{monom[0]}v^{monom[1]}: W-div={has_W}, L-div={has_L}")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
