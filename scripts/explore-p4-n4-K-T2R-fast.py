#!/usr/bin/env python3
"""Fast structural analysis of K_T2R/r^2.

Focus: even-odd decomposition Ā+pq·B̄, d6 factoring, AM-GM bounds.
Skip: slow Gram tests, global non-negativity minimization.
"""

import numpy as np
import sympy as sp
from sympy import Rational, symbols, expand, Poly, factor, sqrt
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    s, t, u, v, a, b = symbols('s t u v a b')
    r, x, y, p, q = symbols('r x y p q')
    P, Q = symbols('P Q')  # P=p^2, Q=q^2

    pr("Building K_T2R/r^2...")

    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)

    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)

    S = s + t; U = u + v; A = a + b + s*t/6
    surplus_num = expand(
        T2R_num(S,U,A)*T2R_den(s,u,a)*T2R_den(t,v,b)
        - T2R_num(s,u,a)*T2R_den(S,U,A)*T2R_den(t,v,b)
        - T2R_num(t,v,b)*T2R_den(S,U,A)*T2R_den(s,u,a))
    pr(f"  Surplus: {len(Poly(surplus_num, s,t,u,v,a,b).as_dict())} terms [{time.time()-t0:.1f}s]")

    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    K_T2R = expand(expand(surplus_num.subs(subs_norm)) / s**16)
    K_red = expand(sp.cancel(K_T2R / r**2))
    pr(f"  K_T2R/r^2: {len(Poly(K_red, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    # Block decomposition
    poly_pq = Poly(K_red, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    # ================================================================
    pr(f"\n{'='*72}")
    pr("EVEN-ODD DECOMPOSITION: K_T2R/r^2 = Abar + pq*Bbar")
    pr('='*72)

    A_bar = sp.Integer(0)
    B_bar = sp.Integer(0)
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        if (i + j) % 2 == 0 and i % 2 == 0 and j % 2 == 0:
            A_bar = expand(A_bar + coeff * p**i * q**j)
        elif i % 2 == 1 and j % 2 == 1:
            B_bar = expand(B_bar + coeff * p**(i-1) * q**(j-1))
        else:
            pr(f"  WARNING: unexpected monomial p^{i}q^{j}")

    check = expand(A_bar + p*q*B_bar - K_red)
    pr(f"  K_T2R/r^2 = Abar + pq*Bbar? {check == 0}")

    # Abar in terms of P=p^2, Q=q^2
    A_PQ = A_bar.subs({p**2: P, q**2: Q})
    B_PQ = B_bar.subs({p**2: P, q**2: Q})
    pr(f"  Abar({len(Poly(A_PQ, r,x,y,P,Q).as_dict())} terms), "
       f"Bbar({len(Poly(B_PQ, r,x,y,P,Q).as_dict())} terms)")

    # Extract (P,Q)-monomial coefficients for Abar
    poly_A = Poly(A_PQ, P, Q)
    pr(f"\n  Abar coefficients in (P,Q):")
    A_coeffs = {}
    for monom, coeff in poly_A.as_dict().items():
        A_coeffs[monom] = expand(coeff)
        n_t = len(Poly(coeff, r, x, y).as_dict())
        pr(f"    P^{monom[0]}Q^{monom[1]}: {n_t} terms")

    # Extract (P,Q)-monomial coefficients for Bbar
    poly_B = Poly(B_PQ, P, Q)
    pr(f"\n  Bbar coefficients in (P,Q):")
    B_coeffs = {}
    for monom, coeff in poly_B.as_dict().items():
        B_coeffs[monom] = expand(coeff)
        n_t = len(Poly(coeff, r, x, y).as_dict())
        pr(f"    P^{monom[0]}Q^{monom[1]}: {n_t} terms")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("FACTORING d6 BLOCK COEFFICIENTS")
    pr('='*72)

    poly6 = Poly(blocks[6], p, q)
    pr(f"  d6 monomials and factorizations:")
    d6_coeffs = {}
    for monom, coeff in poly6.as_dict().items():
        d6_coeffs[monom] = expand(coeff)
        try:
            f = factor(coeff)
            f_str = str(f)
            if len(f_str) > 300:
                f_str = f_str[:300] + "..."
            pr(f"    p^{monom[0]}q^{monom[1]}: {f_str}")
        except:
            pr(f"    p^{monom[0]}q^{monom[1]}: factor failed")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("FACTORING KEY Abar and Bbar COEFFICIENTS")
    pr('='*72)

    # Factor the leading (highest P,Q degree) coefficients
    for label, coeffs_dict in [("Abar", A_coeffs), ("Bbar", B_coeffs)]:
        pr(f"\n  {label}:")
        for monom in sorted(coeffs_dict.keys(), key=lambda m: m[0]+m[1], reverse=True):
            c = coeffs_dict[monom]
            try:
                f = factor(c)
                f_str = str(f)
                if len(f_str) > 300:
                    f_str = f_str[:300] + "..."
                pr(f"    P^{monom[0]}Q^{monom[1]}: {f_str}")
            except:
                pr(f"    P^{monom[0]}Q^{monom[1]}: factor failed ({len(str(c))} chars)")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("NUMERICAL: Abar vs sqrt(PQ)|Bbar| ON FEASIBLE")
    pr('='*72)

    A_fn = sp.lambdify((r, x, y, p, q), A_bar, 'numpy')
    B_fn = sp.lambdify((r, x, y, p, q), B_bar, 'numpy')
    K_fn = sp.lambdify((r, x, y, p, q), K_red, 'numpy')

    rng = np.random.default_rng(42)
    samples = []
    tries = 0
    while len(samples) < 100000 and tries < 3000000:
        tries += 1
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(1e-4, 1-1e-4))
        yv = float(rng.uniform(1e-4, 1-1e-4))
        pmax2 = 2*(1-xv)/9
        qmax2 = 2*(rv**3)*(1-yv)/9
        pv = float(rng.uniform(-0.95*np.sqrt(pmax2), 0.95*np.sqrt(pmax2)))
        qv = float(rng.uniform(-0.95*np.sqrt(qmax2), 0.95*np.sqrt(qmax2)))
        S_i = 1 + rv
        U_i = pv + qv
        A_i = xv/4 + yv*rv**2/4 + rv/6
        f2_c = 2*S_i**3 - 8*S_i*A_i - 9*U_i**2
        if f2_c <= 1e-6:
            continue
        samples.append((rv, xv, yv, pv, qv))
    samples = np.array(samples)
    n = len(samples)
    pr(f"  Got {n} feasible samples")

    rv = samples[:, 0]; xv = samples[:, 1]; yv = samples[:, 2]
    pv = samples[:, 3]; qv = samples[:, 4]

    A_vals = A_fn(rv, xv, yv, pv, qv)
    B_vals = B_fn(rv, xv, yv, pv, qv)
    K_vals = K_fn(rv, xv, yv, pv, qv)
    PQ_vals = pv**2 * qv**2
    pqB_vals = pv * qv * B_vals

    pr(f"  Abar: min={np.min(A_vals):.4e}, neg={np.sum(A_vals < -1e-12)}/{n}")
    pr(f"  K_T2R/r^2: min={np.min(K_vals):.4e}, neg={np.sum(K_vals < -1e-12)}/{n}")
    pr(f"  |pq*Bbar|: max={np.max(np.abs(pqB_vals)):.4e}")

    # Ratio Abar / |pq*Bbar|
    mask = np.abs(pqB_vals) > 1e-15
    if np.any(mask):
        ratios = A_vals[mask] / np.abs(pqB_vals[mask])
        pr(f"  Abar/|pqBbar| min: {np.min(ratios):.4f}")
        pr(f"  Abar/|pqBbar| 1st percentile: {np.percentile(ratios, 1):.4f}")
        pr(f"  Abar/|pqBbar| median: {np.median(ratios):.4f}")
        pr(f"  Abar < |pqBbar|: {np.sum(ratios < 1)}/{np.sum(mask)}")

    # Abar^2 vs PQ*Bbar^2
    ratio2 = A_vals**2 / (PQ_vals * B_vals**2 + 1e-30)
    pr(f"  Abar^2/(PQ*Bbar^2) min: {np.min(ratio2[mask]):.4f}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("T2_SURPLUS ANALYSIS (function of r,x,y only)")
    pr('='*72)

    # T2(s,a) = s(s^2+60a)/(18*f1) where f1=s^2+12a
    # In normalized: s=1, a=x/4
    # T2_p = (1+15x)/(18(1+3x))
    # For q: s=r, a=yr^2/4
    # T2_q = r(r^2+15yr^2)/(18(r^2+3yr^2)) = r^3(1+15y)/(18r^2(1+3y)) = r(1+15y)/(18(1+3y))
    # For conv: S=1+r, A=(3x+3yr^2+2r)/12
    # f1c = (1+r)^2 + 3x + 3yr^2 + 2r = 1 + 4r + r^2 + 3x + 3yr^2

    def T2_normalized(rv, xv, yv):
        """T2_surplus in normalized coordinates."""
        # T2 for p-polynomial
        f1p = 1 + 3*xv
        T2p = (1 + 15*xv) / (18*f1p)

        # T2 for q-polynomial
        f1q = 1 + 3*yv  # Note: after r^2 cancels
        T2q = rv * (1 + 15*yv) / (18*f1q)

        # T2 for convolution
        Sv = 1 + rv
        Av_12 = 3*xv + 3*yv*rv**2 + 2*rv  # = 12*A
        f1c = Sv**2 + Av_12
        T2c_num = Sv*(Sv**2 + 5*Av_12)
        T2c = T2c_num / (18*f1c)

        return T2c - T2p - T2q

    T2_surp = T2_normalized(rv, xv, yv)
    pr(f"  T2_surplus: min={np.min(T2_surp):.4e}, max={np.max(T2_surp):.4e}")
    pr(f"  T2_surplus neg: {np.sum(T2_surp < -1e-15)}/{n}")
    pr(f"  T2_surplus pos: {np.sum(T2_surp > 1e-15)}/{n}")

    # Evaluate T2_surplus at equality point r=1, x=y=1/3
    T2_eq = T2_normalized(1.0, 1/3, 1/3)
    pr(f"  T2_surplus(1, 1/3, 1/3) = {T2_eq:.6e}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("R_SURPLUS STRUCTURE")
    pr('='*72)

    # R = [8a(s^2-4a)^2] / [2f1*f2] - T2 = ... actually
    # From T2+R: T2R = [8a(s^2-4a)^2 - su^2(s^2+60a)] / [2f1*f2]
    # T2 = s(s^2+60a)/(18f1)
    # So R = T2R - T2 = ... let's compute numerically
    # R = T2R - T2 = [8a(s²-4a)² - su²(s²+60a)]/(2f₁f₂) - s(s²+60a)/(18f₁)
    # = [9(8a(s²-4a)² - su²(s²+60a)) - s(s²+60a)f₂] / (18f₁f₂)
    # = [72a(s²-4a)² - 9su²(s²+60a) - s(s²+60a)(2s³-8sa-9u²)] / (18f₁f₂)
    # = [72a(s²-4a)² - s(s²+60a)(2s³-8sa)] / (18f₁f₂)     [u² cancels!]
    # = [72a(s²-4a)² - 2s²(s²+60a)(s²-4a)] / (18f₁f₂)
    # = [2(s²-4a)[36a(s²-4a) - s²(s²+60a)]] / (18f₁f₂)
    # = [(s²-4a)[36as² - 144a² - s⁴ - 60as²]] / (9f₁f₂)
    # = [(s²-4a)[-s⁴ - 24as² - 144a²]] / (9f₁f₂)
    # = [-(s²-4a)(s⁴ + 24as² + 144a²)] / (9f₁f₂)
    # = [-(s²-4a)(s² + 12a)²] / (9f₁f₂)
    # = [-(s²-4a)f₁²] / (9f₁f₂)
    # = [-(s²-4a)f₁] / (9f₂)
    # = [(4a-s²)f₁] / (9f₂)  ← agrees with earlier! R = (4a-s²)(s²+12a)/(9f₂)

    # So R is independent of u! Both T2 and R are u-independent.
    # Wait, R = (4a-s²)(s²+12a)/(9(2s³-8sa-9u²)) — this DOES depend on u through f₂!

    pr("  R(s,u,a) = (4a-s²)(s²+12a) / (9*(2s³-8sa-9u²))")
    pr("  R depends on u through f₂ = 2s³-8sa-9u²")
    pr("  In normalized coords: R = (x-1)(1+3x) / (9*(2-2x-9p²))")
    pr("  Note: x < 1 on feasible, so (x-1) < 0, and (1+3x) > 0")
    pr("  So R_p < 0 always (negative contribution to 1/Phi4)")
    pr("  R_p = -(1-x)(1+3x) / (9*(2(1-x)-9p²))")

    # R in normalized form
    def R_norm(rv, xv, yv, pv, qv):
        # For p-polynomial: s=1, a=x/4
        Rp = (xv - 1)*(1 + 3*xv) / (9*(2 - 2*xv - 9*pv**2))
        # For q-polynomial: s=r, a=yr^2/4
        Rq = rv * (yv - 1)*(1 + 3*yv) / (9*(2*(1-yv) - 9*qv**2/rv**3))
        # Actually let me be more careful
        # s=r, u=q (in s^{3/2} units), a=yr^2/4
        # 4a-s^2 = yr^2-r^2 = r^2(y-1)
        # s^2+12a = r^2+3yr^2 = r^2(1+3y)
        # f2 = 2r^3-2r^3y-9q^2 = 2r^3(1-y)-9q^2
        Rq = rv**4 * (yv-1)*(1+3*yv) / (9*(2*rv**3*(1-yv)-9*qv**2))
        # Hmm, let me redo: R(r,q,yr^2/4)
        # R = (4a-s^2)*(s^2+12a)/(9*f2) = r^2(y-1)*r^2(1+3y)/(9*(2r^3(1-y)-9q^2))
        # = r^4(y-1)(1+3y)/(9*(2r^3(1-y)-9q^2))
        Rq = rv**4 * (yv-1)*(1+3*yv) / (9*(2*rv**3*(1-yv)-9*qv**2))

        # For convolution: S=1+r, U=p+q, A=(3x+3yr^2+2r)/12
        Sv = 1 + rv; Uv = pv + qv
        Av = (3*xv + 3*yv*rv**2 + 2*rv)/12
        f1c = Sv**2 + 12*Av
        f2c = 2*Sv**3 - 8*Sv*Av - 9*Uv**2
        Rc = (4*Av - Sv**2)*f1c/(9*f2c)

        return Rc - Rp - Rq

    R_surp = R_norm(rv, xv, yv, pv, qv)
    pr(f"\n  R_surplus: min={np.min(R_surp):.4e}, max={np.max(R_surp):.4e}")
    pr(f"  R_surplus neg: {np.sum(R_surp < -1e-15)}/{n}")

    # Key: T2+R surplus = T2_surplus + R_surplus
    total_surp = T2_surp + R_surp
    pr(f"  T2+R surplus: min={np.min(total_surp):.4e}, neg={np.sum(total_surp < -1e-12)}/{n}")

    # Where T2_surplus < 0, what is R_surplus?
    T2_neg_mask = T2_surp < -1e-15
    if np.any(T2_neg_mask):
        R_at_T2neg = R_surp[T2_neg_mask]
        compensation = R_surp[T2_neg_mask] + T2_surp[T2_neg_mask]
        pr(f"\n  Where T2_surplus < 0 ({np.sum(T2_neg_mask)} samples):")
        pr(f"    R_surplus: min={np.min(R_at_T2neg):.4e}, always pos? {np.all(R_at_T2neg > 0)}")
        pr(f"    R+T2 surplus: min={np.min(compensation):.4e}")
        pr(f"    R/|T2| ratio min: {np.min(R_at_T2neg/np.abs(T2_surp[T2_neg_mask])):.4f}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("d0 FACTORIZATION (constant term = value at p=q=0)")
    pr('='*72)

    d0 = blocks[0]
    try:
        d0_f = factor(d0)
        d0_str = str(d0_f)
        if len(d0_str) > 500:
            pr(f"  d0 factored ({len(d0_str)} chars): {d0_str[:500]}...")
        else:
            pr(f"  d0 factored: {d0_f}")
    except:
        pr(f"  d0: factor failed")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("TITU-LIKE BOUND FOR R")
    pr('='*72)

    # R(s,u,a) = (4a-s²)(s²+12a) / (9*(2s³-8sa-9u²))
    # With 4a < s² on feasible (x<1), numerator < 0
    # f₂ > 0 on feasible, so R < 0 always
    # R depends on u through f₂ = F - 9u², where F = 2s³-8sa
    #
    # R_surplus = R_conv - R_p - R_q
    # = (4A-S²)·F1c/(9·F2c) - (4a-s²)·f1p/(9·f2p) - (4b-t²)·f1q/(9·f2q)
    #
    # Numerator: (4A-S²)·F1c = ((x+yr²+2r/3)·s² - (1+r)²s²)·(terms)
    # In normalized: 4A-S² = (3x+3yr²+2r)/3 - (1+r)² = (3x+3yr²+2r-3(1+r)²)/3
    # = (3x+3yr²+2r-3-6r-3r²)/3 = (3x+3yr²-3r²-4r-3)/3 = (3(x-1)+3r²(y-1)-4r)/3
    # = -(3(1-x)+3r²(1-y)+4r)/3

    pr("  4A-S² = -(3(1-x) + 3r²(1-y) + 4r)/3")
    pr("  This is always < 0 on feasible (since 1-x>0, 1-y>0, r>0)")
    pr("  So R_conv < 0 always")
    pr("  R_p = -(1-x)(1+3x)/(9(2(1-x)-9p²)) < 0")
    pr("  R_q = -r⁴(1-y)(1+3y)/(9(2r³(1-y)-9q²)) < 0")
    pr("  All three terms negative. Surplus = (larger neg) - (two smaller negs)")

    # The R_surplus is positive when |R_conv| < |R_p| + |R_q|
    # i.e., when combining R values makes |R| smaller (sub-additivity of |R|)

    # Check: is R concave? R is a function of several variables.
    # R = C·f1/(9·f2) where C = 4a-s², f1=s²+12a, f2=2s³-8sa-9u²
    # R(u) = C·f1/(9·(F-9u²)) where F=2s³-8sa
    # dR/du² = C·f1·9/(9·(F-9u²)²) = C·f1/(F-9u²)²
    # Since C < 0, f1 > 0: d²R/d(u²)² < 0 — R is concave in u²!

    pr("\n  R as function of u²: R(w) = C·f1/(9(F-9w)), C<0, f1>0")
    pr("  d²R/dw² = -18·C·f1/(F-9w)³ > 0 (since C<0)")
    pr("  Wait: dR/dw = C·f1·9/(9(F-9w)²) = C·f1/(F-9w)²")
    pr("  d²R/dw² = 2·C·f1·9/(F-9w)³ = 18C·f1/(F-9w)³")
    pr("  Since C<0, f1>0, (F-9w)>0: d²R/dw² < 0")
    pr("  So R is CONCAVE in w=u² — which gives R super-additive?")
    pr("  Need to check carefully for multivariate case...")

    # R_conv - R_p - R_q >= 0 is the claim.
    # R(s,u,a) is a function of multiple coupled variables.
    # For fixed s,a: R is a function of u² only, and it's concave in u².
    # But s,a also change between p,q and the convolution.

    # ================================================================
    pr(f"\n{'='*72}")
    pr("APPROACH: WRITE K_T2R/r^2 AS QUADRATIC IN (p+q) AND (p-q)")
    pr('='*72)

    # Let m = p+q (mean), d_var = p-q (diff)
    # p = (m+d)/2, q = (m-d)/2
    m_sym, d_sym = symbols('m d')
    K_md = K_red.subs({p: (m_sym+d_sym)/2, q: (m_sym-d_sym)/2})
    K_md = expand(K_md)

    poly_md = Poly(K_md, m_sym, d_sym)
    blocks_md = {}
    for monom, coeff in poly_md.as_dict().items():
        i, j = monom
        d_total = i + j
        blocks_md[d_total] = expand(blocks_md.get(d_total, sp.Integer(0)) + coeff * m_sym**i * d_sym**j)

    pr("  K_T2R/r^2 in (m=p+q, d=p-q) coordinates:")
    for d_total in sorted(blocks_md.keys()):
        n_t = len(Poly(blocks_md[d_total], r, x, y, m_sym, d_sym).as_dict())
        pr(f"    degree {d_total}: {n_t} terms")

    # Check: which monomials appear?
    pr("  Monomials in (m,d):")
    for monom in sorted(poly_md.as_dict().keys()):
        pr(f"    m^{monom[0]}d^{monom[1]}")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
