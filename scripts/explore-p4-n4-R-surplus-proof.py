#!/usr/bin/env python3
"""Explore algebraic structure of R_surplus for proof construction.

R(s,u,a) = (4a-s²)(s²+12a) / (9·f₂) where f₂ = 2s³-8sa-9u²

Key identity: R = C·f₁/(9·f₂) where C = 4a-s² < 0 on feasible

R_surplus = R_conv - R_p - R_q

Goal: understand when R_surplus >= 0 and relate to L sign.
"""

import numpy as np
import sympy as sp
from sympy import Rational, symbols, expand, Poly, factor, cancel, together, fraction
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    r, x, y, p, q = symbols('r x y p q')

    # ================================================================
    pr("="*72)
    pr("R_SURPLUS POLYNOMIAL COMPUTATION")
    pr("="*72)

    # R(s,u,a) = (4a-s²)(s²+12a) / (9*(2s³-8sa-9u²))
    # = C·f1 / (9·f2)

    # In normalized coordinates:
    # R_p: s=1, a=x/4
    #   C_p = x-1, f1p = 1+3x, f2p = 2(1-x)-9p²
    #   R_p = (x-1)(1+3x)/(9*(2(1-x)-9p²))

    # R_q: s=r, a=yr²/4
    #   C_q = r²(y-1), f1q = r²(1+3y), f2q = 2r³(1-y)-9q²
    #   R_q = r⁴(y-1)(1+3y)/(9*(2r³(1-y)-9q²))

    # R_conv: S=1+r, A=(3x+3yr²+2r)/12
    #   C_c = 4A-S² = (3x+3yr²+2r)/3 - (1+r)² = -(3(1-x)+3r²(1-y)+4r)/3
    #   f1c = (1+r)²+3x+3yr²+2r = 1+4r+r²+3x+3yr²
    #   f2c = 2(1+r)³-8(1+r)A-9(p+q)² = ... (complex)

    # R_surplus numerator (×9·f2c·f2p·f2q):
    # R_surplus = C_c·f1c·f2p·f2q - C_p·f1p·f2c·f2q - C_q·f1q·f2c·f2p
    #             all divided by 9·f2c·f2p·f2q

    pr("\n  Computing R_surplus numerator symbolically...")

    C_p = x - 1
    f1p = 1 + 3*x
    f2p = 2*(1-x) - 9*p**2

    C_q = r**2*(y - 1)
    f1q = r**2*(1 + 3*y)
    f2q = 2*r**3*(1-y) - 9*q**2

    # For conv
    Sv = 1 + r
    Av_times12 = 3*x + 3*y*r**2 + 2*r
    C_c = expand(Av_times12/3 - Sv**2)  # = (3x+3yr²+2r)/3 - (1+r)²
    f1c = expand(Sv**2 + Av_times12)
    # f2c = 2*S³-8*S*A-9*(p+q)²
    # = 2(1+r)³ - 8(1+r)·(3x+3yr²+2r)/12 - 9(p+q)²
    # = 2(1+r)³ - 2(1+r)(3x+3yr²+2r)/3 - 9(p+q)²
    f2c = expand(2*Sv**3 - 2*Sv*Av_times12/3 - 9*(p+q)**2)

    pr(f"  C_c = {C_c}")
    pr(f"  C_c simplified: {factor(C_c)}")
    pr(f"  f1c = {f1c}")
    pr(f"  f2c = {f2c}")

    # R_surplus_num = C_c·f1c·f2p·f2q - C_p·f1p·f2c·f2q - C_q·f1q·f2c·f2p
    pr(f"\n  Expanding term 1: C_c·f1c·f2p·f2q [{time.time()-t0:.1f}s]...")
    term1 = expand(C_c * f1c * f2p * f2q)
    pr(f"    {len(Poly(term1, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    pr(f"  Expanding term 2: C_p·f1p·f2c·f2q...")
    term2 = expand(C_p * f1p * f2c * f2q)
    pr(f"    {len(Poly(term2, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    pr(f"  Expanding term 3: C_q·f1q·f2c·f2p...")
    term3 = expand(C_q * f1q * f2c * f2p)
    pr(f"    {len(Poly(term3, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    R_num = expand(term1 - term2 - term3)
    n_R = len(Poly(R_num, r, x, y, p, q).as_dict())
    pr(f"\n  R_surplus_num: {n_R} terms [{time.time()-t0:.1f}s]")

    # Check: what (p,q)-degree does R_surplus_num have?
    poly_pq = Poly(R_num, p, q)
    pq_degs = {}
    for monom, coeff in poly_pq.as_dict().items():
        d = monom[0] + monom[1]
        pq_degs[d] = pq_degs.get(d, 0) + 1

    pr(f"  (p,q)-degree distribution:")
    for d in sorted(pq_degs.keys()):
        pr(f"    degree {d}: {pq_degs[d]} monomial types")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("R_SURPLUS AT p=q=0 (R₀ = R_surplus|_{p=q=0})")
    pr('='*72)

    R_num_0 = R_num.subs({p: 0, q: 0})
    R_num_0 = expand(R_num_0)
    R_num_0_factored = factor(R_num_0)
    pr(f"  R_surplus_num(p=q=0) = {R_num_0_factored}")

    # Also factor the denominators at p=q=0
    f2p_0 = f2p.subs(p, 0)
    f2q_0 = f2q.subs(q, 0)
    f2c_0 = f2c.subs({p: 0, q: 0})
    pr(f"  f2p(0) = {f2p_0}")
    pr(f"  f2q(0) = {f2q_0}")
    pr(f"  f2c(0) = {factor(f2c_0)}")

    R0_val = cancel(R_num_0 / (sp.Integer(9) * f2c_0 * f2p_0 * f2q_0))
    R0_factored = factor(R0_val)
    pr(f"  R_surplus(p=q=0) = {R0_factored}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("DECOMPOSE R_SURPLUS_NUM BY (p,q) DEGREE")
    pr('='*72)

    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        d = monom[0] + monom[1]
        blocks[d] = expand(blocks.get(d, sp.Integer(0)) + coeff * p**monom[0] * q**monom[1])

    for d in sorted(blocks.keys()):
        n_terms = len(Poly(blocks[d], r, x, y, p, q).as_dict())
        pr(f"  degree {d}: {n_terms} terms")

        # Try to factor
        if n_terms <= 50:
            try:
                f = factor(blocks[d])
                f_str = str(f)
                if len(f_str) > 300:
                    f_str = f_str[:300] + "..."
                pr(f"    factored: {f_str}")
            except:
                pr(f"    factor failed")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("KEY QUESTION: DOES R_SURPLUS_NUM HAVE FACTOR L?")
    pr('='*72)

    L = -27*r*x*y + 3*r*x + 9*r*y**2 - 3*r*y + 2*r + 9*x**2 - 27*x*y - 3*x + 3*y + 2

    # Check if L divides R_surplus_num
    pr("  Attempting polynomial division R_num / L...")
    try:
        q_div, r_div = sp.div(R_num, L, r, x, y, p, q)
        r_div_exp = expand(r_div)
        if r_div_exp == 0:
            pr("  *** L DIVIDES R_surplus_num! ***")
            q_n = len(Poly(q_div, r, x, y, p, q).as_dict())
            pr(f"  R_surplus_num = L · Q, Q has {q_n} terms")

            # Does 4r also divide?
            q_div2, r_div2 = sp.div(q_div, 4*r, r, x, y, p, q)
            if expand(r_div2) == 0:
                pr(f"  R_surplus_num = 4rL · Q', Q' has {len(Poly(q_div2, r,x,y,p,q).as_dict())} terms")
        else:
            pr(f"  L does NOT divide R_surplus_num (remainder has {len(Poly(r_div_exp, r,x,y,p,q).as_dict())} terms)")
    except Exception as e:
        pr(f"  Division failed: {e}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("R_SURPLUS AS MÖBIUS TRANSFORM")
    pr('='*72)

    # R = C·f1/(9·(F-9w)) where F=2s³-8sa, w=u²
    # This is a Möbius (linear fractional) transform in w
    # R(w) = a₁/(a₂ - a₃·w) where a₁=C·f1, a₂=9F, a₃=81
    # R is concave in w, monotone decreasing (since C<0)

    # For R_surplus in terms of w_p=p², w_q=q², w_m=(p+q)²:
    # R_surplus = C_c·f1c/(9(F_c-9w_m)) - C_p·f1p/(9(F_p-9w_p)) - C_q·f1q/(9(F_q-9w_q))
    # where w_m = w_p + w_q + 2pq

    # Note: w_m = (p+q)² depends on pq cross-term
    # R_surplus at p·q < 0: w_m < w_p + w_q → |R_conv| smaller → R_surplus larger
    # R_surplus at p·q > 0: w_m > w_p + w_q → |R_conv| larger → R_surplus smaller

    pr("  R depends on (p,q) only through w_p=p², w_q=q², w_m=(p+q)²")
    pr("  The 'excess' w_m - w_p - w_q = 2pq is the interaction term")
    pr("  When pq > 0: R_surplus decreases (worse case)")
    pr("  When pq < 0: R_surplus increases (better case)")

    # At p = -q (anti-aligned): w_m = 0 (best case for R_surplus)
    # R_surplus(p,-p) = C_c·f1c/(9·F_c) - C_p·f1p/(9(F_p-9p²)) - C_q·f1q/(9(F_q-9p²))
    # = R_conv(0) + |R_p(p)| + |R_q(p)| (since R_conv(0) < 0, |R_p|,|R_q| > 0)

    # Let me compute R_surplus at q=0 (simplification)
    pr(f"\n  R_surplus at q=0:")
    R_num_q0 = R_num.subs(q, 0)
    R_num_q0 = expand(R_num_q0)
    n_q0 = len(Poly(R_num_q0, r, x, y, p).as_dict())
    pr(f"    R_surplus_num(q=0): {n_q0} terms")

    # Factor
    R_num_q0_f = factor(R_num_q0)
    R_q0_str = str(R_num_q0_f)
    if len(R_q0_str) > 500:
        pr(f"    factored ({len(R_q0_str)} chars): {R_q0_str[:500]}...")
    else:
        pr(f"    factored: {R_num_q0_f}")

    # R_surplus at p=0
    pr(f"\n  R_surplus at p=0:")
    R_num_p0 = R_num.subs(p, 0)
    R_num_p0 = expand(R_num_p0)
    n_p0 = len(Poly(R_num_p0, r, x, y, q).as_dict())
    pr(f"    R_surplus_num(p=0): {n_p0} terms")
    R_num_p0_f = factor(R_num_p0)
    R_p0_str = str(R_num_p0_f)
    if len(R_p0_str) > 500:
        pr(f"    factored ({len(R_p0_str)} chars): {R_p0_str[:500]}...")
    else:
        pr(f"    factored: {R_num_p0_f}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("KEY TEST: R_surplus_num RESTRICTED TO L<=0")
    pr('='*72)

    # When L <= 0, T2_surplus <= 0, and we need R_surplus >= 0
    # L = α + β·r where α = 9x²-27xy-3x+3y+2, β = -27xy+3x+9y²-3y+2
    # L <= 0 ⟺ r >= -α/β (when β > 0) or r <= -α/β (when β < 0)

    # Numerically: test R_surplus specifically where L < 0
    rng = np.random.default_rng(123)
    R_fn = sp.lambdify((r, x, y, p, q), R_num, 'numpy')
    den_fn = sp.lambdify((r, x, y, p, q), 9*f2c*f2p*f2q, 'numpy')
    L_fn = sp.lambdify((r, x, y), L, 'numpy')

    count_L_neg = 0
    count_R_neg_at_Lneg = 0
    min_R_at_Lneg = np.inf

    for _ in range(500000):
        rv = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng.uniform(1e-5, 1-1e-5))
        yv = float(rng.uniform(1e-5, 1-1e-5))

        Lv = L_fn(rv, xv, yv)
        if Lv >= 0:
            continue

        count_L_neg += 1
        pmax2 = 2*(1-xv)/9
        qmax2 = 2*(rv**3)*(1-yv)/9

        for _ in range(5):
            pv = float(rng.uniform(-0.99*np.sqrt(pmax2), 0.99*np.sqrt(pmax2)))
            qv = float(rng.uniform(-0.99*np.sqrt(qmax2), 0.99*np.sqrt(qmax2)))
            Sv = 1+rv; Uv = pv+qv; Av_v = xv/4+yv*rv**2/4+rv/6
            f2cv = 2*Sv**3-8*Sv*Av_v-9*Uv**2
            if f2cv <= 1e-8:
                continue

            try:
                Rv = float(R_fn(rv, xv, yv, pv, qv))
                dv = float(den_fn(rv, xv, yv, pv, qv))
                if np.isfinite(Rv) and np.isfinite(dv) and dv > 0:
                    R_val = Rv / dv
                    if R_val < min_R_at_Lneg:
                        min_R_at_Lneg = R_val
                    if R_val < -1e-12:
                        count_R_neg_at_Lneg += 1
            except:
                pass

    pr(f"  L<0 samples tested: {count_L_neg}")
    pr(f"  R_surplus < 0 when L < 0: {count_R_neg_at_Lneg}")
    pr(f"  R_surplus min when L < 0: {min_R_at_Lneg:.6e}")
    if count_R_neg_at_Lneg == 0:
        pr("  *** R_surplus >= 0 ALWAYS when L <= 0! ***")
        pr("  This is half of the partition proof.")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
