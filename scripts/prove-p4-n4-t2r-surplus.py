#!/usr/bin/env python3
"""Prove T2+R surplus >= 0 symbolically.

PROVEN: 1/Phi4 = T1 + T2 + R where:
  T1 = 3u^2 / [4(s^2+12a)]
  T2+R = [8a(s^2-4a)^2 - su^2(s^2+60a)] / [2(s^2+12a)(2s^3-8sa-9u^2)]

T1_surplus <= 0 by Titu's lemma.
T2+R surplus >= 0 numerically confirmed (100k/100k).

This script: compute the T2+R surplus as a rational function in (r,x,y,p,q),
extract the numerator polynomial, and analyze its structure.
"""

import sympy as sp
from sympy import Rational, symbols, expand, cancel, together, fraction, factor, Poly
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()

    s, t, u, v, a, b = symbols('s t u v a b')
    r, x, y, p, q = symbols('r x y p q')

    pr("=" * 72)
    pr("COMPUTING T2+R SURPLUS POLYNOMIAL")
    pr("=" * 72)

    # T2+R for a single polynomial with parameters (s, u, a):
    # numerator: 8a(s^2-4a)^2 - s*u^2*(s^2+60a)
    # denominator: 2*(s^2+12a)*(2s^3-8sa-9u^2)
    def T2R(ss, uu, aa):
        """Return (numerator, denominator) of T2+R."""
        num = 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)
        den = 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)
        return num, den

    # Verify this equals 1/Phi4 - T1
    pr("\nVerifying T2+R formula...")
    disc = 256*a**3 - 128*a**2*s**2 + 16*a*s**4 - 144*a*s*u**2 + 4*s**3*u**2 - 27*u**4
    phi4d = 4*(s**2 + 12*a)*(2*s**3 - 8*s*a - 9*u**2)
    T1_val = 3*u**2 / (4*(s**2 + 12*a))
    inv_phi4 = disc / phi4d

    diff = cancel(inv_phi4 - T1_val)
    T2R_num, T2R_den = T2R(s, u, a)
    T2R_val = T2R_num / T2R_den
    check = cancel(diff - T2R_val)
    pr(f"  1/Phi4 - T1 = T2+R? {check == 0}")

    # Also verify the numerator identity
    pr(f"  T2+R numerator = 8a(s^2-4a)^2 - su^2(s^2+60a)?")
    num_expanded = expand(8*a*(s**2 - 4*a)**2 - s*u**2*(s**2 + 60*a))
    num_from_disc = expand(disc - 3*u**2*(2*s**3 - 8*s*a - 9*u**2))
    pr(f"    disc - 3u^2*f2 = {num_from_disc}")
    pr(f"    Our formula    = {num_expanded}")
    pr(f"    Match? {expand(num_expanded - num_from_disc) == 0}")

    # T2+R surplus = T2R(conv) - T2R(p) - T2R(q)
    # where conv: S=s+t, U=u+v, A=a+b+st/6
    pr(f"\n  Computing T2+R surplus [{time.time()-t0:.1f}s]...")

    # For p: (s, u, a)
    num_p, den_p = T2R(s, u, a)
    # For q: (t, v, b)
    num_q, den_q = T2R(t, v, b)
    # For conv:
    S = s + t
    U = u + v
    A = a + b + s*t/6
    num_c, den_c = T2R(S, U, A)

    # Surplus = num_c/den_c - num_p/den_p - num_q/den_q
    # = (num_c*den_p*den_q - num_p*den_c*den_q - num_q*den_c*den_p) / (den_c*den_p*den_q)
    pr(f"  Building cross-products [{time.time()-t0:.1f}s]...")

    # Do this carefully to avoid memory explosion
    pr(f"  Expanding num_c*den_p*den_q...", end=" ")
    term1 = expand(num_c * den_p * den_q)
    pr(f"done. [{time.time()-t0:.1f}s]")

    pr(f"  Expanding num_p*den_c*den_q...", end=" ")
    term2 = expand(num_p * den_c * den_q)
    pr(f"done. [{time.time()-t0:.1f}s]")

    pr(f"  Expanding num_q*den_c*den_p...", end=" ")
    term3 = expand(num_q * den_c * den_p)
    pr(f"done. [{time.time()-t0:.1f}s]")

    surplus_num = expand(term1 - term2 - term3)
    surplus_den = expand(den_c * den_p * den_q)

    n_terms_num = len(Poly(surplus_num, s, t, u, v, a, b).as_dict())
    n_terms_den = len(Poly(surplus_den, s, t, u, v, a, b).as_dict())
    pr(f"\n  T2+R surplus numerator: {n_terms_num} terms [{time.time()-t0:.1f}s]")
    pr(f"  T2+R surplus denominator: {n_terms_den} terms")

    # Normalize: t->rs, a->xs^2/4, b->yr^2s^2/4, u->ps^{3/2}, v->qs^{3/2}
    pr(f"\n  Normalizing to (r,x,y,p,q) coordinates [{time.time()-t0:.1f}s]...")

    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3,2), v: q*s**Rational(3,2)}

    pr(f"  Substituting numerator...", end=" ")
    num_sub = expand(surplus_num.subs(subs_norm))
    pr(f"done. [{time.time()-t0:.1f}s]")

    # Factor out s power
    num_poly_s = Poly(num_sub, s)
    s_idx = list(num_poly_s.gens).index(s)
    s_pows = sorted(set(m[s_idx] for m in num_poly_s.as_dict().keys()))
    pr(f"  Numerator s powers: {s_pows}")

    if len(s_pows) == 1:
        s_pow = s_pows[0]
        K_T2R = expand(num_sub / s**s_pow)
        n_K = len(Poly(K_T2R, r, x, y, p, q).as_dict())
        pr(f"  K_T2R = num / s^{s_pow}: {n_K} terms [{time.time()-t0:.1f}s]")
    else:
        pr(f"  Multiple s powers — checking...")
        s_pow = min(s_pows)
        K_T2R = expand(num_sub / s**s_pow)
        pr(f"  Using s^{s_pow}: checking if s-free...")

    pr(f"  Substituting denominator...", end=" ")
    den_sub = expand(surplus_den.subs(subs_norm))
    pr(f"done. [{time.time()-t0:.1f}s]")

    den_poly_s = Poly(den_sub, s)
    s_idx_d = list(den_poly_s.gens).index(s)
    s_pows_d = sorted(set(m[s_idx_d] for m in den_poly_s.as_dict().keys()))
    pr(f"  Denominator s powers: {s_pows_d}")

    if len(s_pows_d) == 1:
        s_pow_d = s_pows_d[0]
        D_T2R = expand(den_sub / s**s_pow_d)
        n_D = len(Poly(D_T2R, r, x, y, p, q).as_dict())
        pr(f"  D_T2R = den / s^{s_pow_d}: {n_D} terms [{time.time()-t0:.1f}s]")

    net_s = s_pow - s_pow_d
    pr(f"  surplus = s^{net_s} * K_T2R / D_T2R")

    # Factor denominator
    pr(f"\n  Factoring D_T2R [{time.time()-t0:.1f}s]...")
    D_factored = factor(D_T2R)
    D_str = str(D_factored)
    if len(D_str) > 300:
        pr(f"  D_T2R factored: ({len(D_str)} chars, long)")
    else:
        pr(f"  D_T2R factored: {D_factored}")

    # Decompose K_T2R by (p,q)-degree
    pr(f"\n  Decomposing K_T2R by (p,q)-degree [{time.time()-t0:.1f}s]...")
    K_pq = Poly(K_T2R, p, q)
    blocks = {}
    for monom, coeff in K_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    for d in sorted(blocks.keys()):
        n_terms = len(Poly(blocks[d], r, x, y, p, q).as_dict())
        pr(f"    degree {d}: {n_terms} terms")

    # Even-odd split
    pr(f"\n  Even-odd (p,q) split...")
    A_T2R = sp.Integer(0)
    B_T2R = sp.Integer(0)
    for monom, coeff in K_pq.as_dict().items():
        i, j = monom
        if i % 2 == 0 and j % 2 == 0:
            A_T2R = expand(A_T2R + coeff * p**i * q**j)
        elif i % 2 == 1 and j % 2 == 1:
            B_T2R = expand(B_T2R + coeff * p**(i-1) * q**(j-1))
        else:
            pr(f"    WARNING: mixed parity p^{i}q^{j}")

    check_eo = expand(A_T2R + p*q*B_T2R - K_T2R)
    pr(f"  K_T2R = A + pq*B? {check_eo == 0}")

    A_terms = len(Poly(A_T2R, r, x, y, p, q).as_dict()) if A_T2R != 0 else 0
    B_terms = len(Poly(B_T2R, r, x, y, p, q).as_dict()) if B_T2R != 0 else 0
    pr(f"  A: {A_terms} terms, B: {B_terms} terms")

    # Quick factoring attempt on K_T2R
    pr(f"\n  Attempting to factor K_T2R [{time.time()-t0:.1f}s]...")
    try:
        K_factored = factor(K_T2R)
        K_str = str(K_factored)
        if len(K_str) > 500:
            pr(f"  K_T2R factored: ({len(K_str)} chars)")
            # Try to identify common factors
            for test_factor_name, test_factor in [
                ("(x-1)", x-1), ("(y-1)", y-1), ("(1+3x)", 1+3*x), ("(1+3y)", 1+3*y),
                ("r", r), ("r^2", r**2), ("(1-x)", 1-x), ("(1-y)", 1-y)]:
                q_div, r_div = sp.div(K_T2R, test_factor, r, x, y, p, q)
                if expand(r_div) == 0:
                    pr(f"    {test_factor_name} divides K_T2R!")
        else:
            pr(f"  K_T2R factored: {K_factored}")
    except Exception as e:
        pr(f"  Factor failed: {e}")

    # Sign of denominator on feasible domain
    pr(f"\n{'='*72}")
    pr("DENOMINATOR SIGN ANALYSIS")
    pr('='*72)
    pr("D_T2R = den_c * den_p * den_q where each den = 2*f1*f2")
    pr("f1 = s^2+12a > 0 always")
    pr("f2 = 2s^3-8sa-9u^2 > 0 on feasible domain")
    pr("So D_T2R > 0 on feasible domain.")
    pr("Therefore: K_T2R >= 0 on feasible domain iff T2+R surplus >= 0.")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")

    return K_T2R, D_T2R, blocks


if __name__ == '__main__':
    main()
