#!/usr/bin/env python3
"""Factor K_red using the coefficient identities B01=2A02, B10=2A20, B11=2A12=2A21.

Key insight: these identities mean we can write
  K_red = C1·(p+q)² + C0
where:
  C1 = a20·p² + a02·q² + a12·p²q²     (degree 4 in p,q; controls high-p,q behavior)
  C0 = a00 + b00·pq + a10·p² + a01·q² + delta1·p²q²   (degree 4, simpler)
  delta1 = a11 - a20 - a02

Since (p+q)² ≥ 0, if C1 ≥ 0 then K_red ≥ C0.
For L ≤ 0: a12 = -1296rL ≥ 0, so C1 ≥ 0, and the proof reduces to C0 ≥ 0.
"""

import numpy as np
import sympy as sp
from sympy import symbols, expand, Poly, factor, cancel
import time, json


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def build_exact_k_red():
    """Build K_red from the exact T2+R derivation (same as Codex script)."""
    s, t, u, v, a, b = sp.symbols('s t u v a b')
    r, x, y, p, q = sp.symbols('r x y p q')

    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)

    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)

    S = s + t; U = u + v; A = a + b + s*t/6
    surplus_num = expand(
        T2R_num(S, U, A)*T2R_den(s, u, a)*T2R_den(t, v, b)
        - T2R_num(s, u, a)*T2R_den(S, U, A)*T2R_den(t, v, b)
        - T2R_num(t, v, b)*T2R_den(S, U, A)*T2R_den(s, u, a)
    )

    subs_norm = {
        t: r*s,
        a: x*s**2/4,
        b: y*r**2*s**2/4,
        u: p*s**sp.Rational(3,2),
        v: q*s**sp.Rational(3,2),
    }
    K_exact = expand(surplus_num.subs(subs_norm) / s**16)
    K_red = expand(cancel(K_exact / r**2))
    return K_red


def main():
    t0 = time.time()
    r, x, y, p, q = symbols('r x y p q')
    P, Q = symbols('P Q')

    pr("Building K_red from exact T2+R derivation...")
    K_red = build_exact_k_red()
    pr(f"  K_red: {len(Poly(K_red, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    L = expand(9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2)

    # Extract A(P,Q) and B(P,Q) from even-odd decomposition
    pr("\nExtracting A(P,Q) + pq·B(P,Q) decomposition...")
    poly = Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr = expand(A_expr + c * p**i * q**j)
        elif i % 2 == 1 and j % 2 == 1:
            B_expr = expand(B_expr + c * p**(i-1) * q**(j-1))

    # Get A, B as polynomials in P=p², Q=q²
    A_PQ = expand(A_expr.subs({p**2: P, q**2: Q}))
    B_PQ = expand(B_expr.subs({p**2: P, q**2: Q}))
    A_dict = Poly(A_PQ, P, Q).as_dict()
    B_dict = Poly(B_PQ, P, Q).as_dict()

    pr(f"  A monomials (P^i·Q^j): {sorted(A_dict.keys())}")
    pr(f"  B monomials (P^i·Q^j): {sorted(B_dict.keys())}")

    # Print and factor each coefficient
    pr("\n  A coefficients:")
    for m in sorted(A_dict.keys()):
        f = factor(A_dict[m])
        s = str(f)
        if len(s) > 200:
            s = s[:200] + "..."
        pr(f"    a_{m[0]}{m[1]}: {s}")

    pr("\n  B coefficients:")
    for m in sorted(B_dict.keys()):
        f = factor(B_dict[m])
        s = str(f)
        if len(s) > 200:
            s = s[:200] + "..."
        pr(f"    b_{m[0]}{m[1]}: {s}")

    # Verify coefficient identities
    pr("\n  Coefficient identities:")
    a = A_dict; b = B_dict
    pr(f"    B01 = 2A02: {expand(b[(0,1)] - 2*a[(0,2)]) == 0}")
    pr(f"    B10 = 2A20: {expand(b[(1,0)] - 2*a[(2,0)]) == 0}")
    pr(f"    B11 = 2A12: {expand(b[(1,1)] - 2*a.get((1,2), 0)) == 0}")
    pr(f"    A21 = A12: {expand(a.get((2,1), 0) - a.get((1,2), 0)) == 0}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("(p+q)² FACTORIZATION: K_red = C1·u² + C0")
    pr('='*72)

    a00 = a[(0,0)]; a10 = a[(1,0)]; a01 = a[(0,1)]; a11 = a[(1,1)]
    a20 = a[(2,0)]; a02 = a[(0,2)]
    a12 = a.get((1,2), sp.Integer(0))
    a21 = a.get((2,1), sp.Integer(0))
    b00 = b[(0,0)]

    # C1 = a20·P + a02·Q + a12·PQ  (multiplied by u²=(p+q)²)
    # C0 = a00 + b00·pq + a10·P + a01·Q + (a11-a20-a02)·PQ
    delta1 = expand(a11 - a20 - a02)

    C1_PQ = expand(a20*P + a02*Q + a12*P*Q)
    C0_expr = expand(a00 + b00*p*q + a10*p**2 + a01*q**2 + delta1*p**2*q**2)

    # Verify: K_red = C1(p²,q²)·(p+q)² + C0
    C1_pq = C1_PQ.subs({P: p**2, Q: q**2})
    K_reconstructed = expand(C1_pq*(p+q)**2 + C0_expr)
    check = expand(K_reconstructed - K_red)
    pr(f"  Reconstruction check: K_red = C1·u² + C0? {check == 0}")

    pr(f"\n  C1 = a20·P + a02·Q + a12·PQ")
    pr(f"    a20 = {factor(a20)}")
    pr(f"    a02 = {factor(a02)}")
    pr(f"    a12 = {factor(a12)}")

    # Check: a12 should be from d6 (coefficient of p²q⁴ in K_red)
    pr(f"\n    a12 vs -1296rL: {expand(a12 + 1296*r*L) == 0}")
    # Also check p⁴q² coefficient (a21):
    pr(f"    a21 vs -1296rL: {expand(a21 + 1296*r*L) == 0}")
    pr(f"    → a12 = a21 = -1296rL")

    pr(f"\n  C0 = a00 + b00·pq + a10·p² + a01·q² + delta1·p²q²")
    pr(f"    a00 = {factor(a00)}")
    pr(f"    b00 = {factor(b00)}")
    pr(f"    a10 = {factor(a10)}")
    pr(f"    a01 = {factor(a01)}")
    pr(f"    delta1 = a11-a20-a02 = {factor(delta1)}")
    n_delta1 = len(Poly(delta1, r, x, y).as_dict())
    pr(f"    ({n_delta1} terms)")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("SIGN ANALYSIS ON FEASIBLE DOMAIN")
    pr('='*72)

    # Lambdify everything
    a20_fn = sp.lambdify((r,x,y), a20, 'numpy')
    a02_fn = sp.lambdify((r,x,y), a02, 'numpy')
    a12_fn = sp.lambdify((r,x,y), a12, 'numpy')
    a00_fn = sp.lambdify((r,x,y), a00, 'numpy')
    b00_fn = sp.lambdify((r,x,y), b00, 'numpy')
    a10_fn = sp.lambdify((r,x,y), a10, 'numpy')
    a01_fn = sp.lambdify((r,x,y), a01, 'numpy')
    d1_fn = sp.lambdify((r,x,y), delta1, 'numpy')
    L_fn = sp.lambdify((r,x,y), L, 'numpy')
    K_fn = sp.lambdify((r,x,y,p,q), K_red, 'numpy')

    rng = np.random.default_rng(42)
    n_test = 0; n_Lneg = 0

    # Counters for C1, C0 sign
    C1_neg_all = 0; C1_neg_Lneg = 0
    C0_neg_all = 0; C0_neg_Lneg = 0
    a20_neg = 0; a02_neg = 0; a12_neg = 0; a12_neg_Lneg = 0
    a00_neg = 0; a00_neg_Lneg = 0
    b00_pos = 0; a10_neg = 0; a01_neg = 0; d1_neg = 0; d1_neg_Lneg = 0
    b00_neg_Lneg = 0; a10_neg_Lneg = 0; a01_neg_Lneg = 0

    # Also track discriminant of C0 as quadratic in p
    disc_pos_all = 0; disc_pos_Lneg = 0

    for _ in range(500000):
        rv = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng.uniform(0.001, 0.999))
        yv = float(rng.uniform(0.001, 0.999))
        Lv = L_fn(rv, xv, yv)

        f2p_max = 2*(1-xv)
        f2q_max = 2*rv**3*(1-yv)
        pmax = np.sqrt(max(f2p_max/9, 0))
        qmax = np.sqrt(max(f2q_max/9, 0))

        pv = float(rng.uniform(-pmax, pmax))
        qv = float(rng.uniform(-qmax, qmax))

        f2c_v = 2*(1+rv)**3 - 2*(1+rv)*(3*xv+3*yv*rv**2+2*rv)/3 - 9*(pv+qv)**2
        if f2c_v <= 0:
            continue

        n_test += 1
        is_Lneg = Lv <= 0
        if is_Lneg:
            n_Lneg += 1

        a20v = float(a20_fn(rv, xv, yv))
        a02v = float(a02_fn(rv, xv, yv))
        a12v = float(a12_fn(rv, xv, yv))
        C1v = a20v*pv**2 + a02v*qv**2 + a12v*pv**2*qv**2
        C0v = float(a00_fn(rv,xv,yv)) + float(b00_fn(rv,xv,yv))*pv*qv + float(a10_fn(rv,xv,yv))*pv**2 + float(a01_fn(rv,xv,yv))*qv**2 + float(d1_fn(rv,xv,yv))*pv**2*qv**2

        if C1v < -1e-10: C1_neg_all += 1
        if C0v < -1e-10: C0_neg_all += 1
        if is_Lneg:
            if C1v < -1e-10: C1_neg_Lneg += 1
            if C0v < -1e-10: C0_neg_Lneg += 1

        if a20v < -1e-10: a20_neg += 1
        if a02v < -1e-10: a02_neg += 1
        if a12v < -1e-10:
            a12_neg += 1
            if is_Lneg: a12_neg_Lneg += 1

        a00v = float(a00_fn(rv, xv, yv))
        b00v = float(b00_fn(rv, xv, yv))
        a10v = float(a10_fn(rv, xv, yv))
        a01v = float(a01_fn(rv, xv, yv))
        d1v = float(d1_fn(rv, xv, yv))

        if a00v < -1e-10:
            a00_neg += 1
            if is_Lneg: a00_neg_Lneg += 1
        if b00v > 1e-10: b00_pos += 1
        if b00v < -1e-10 and is_Lneg: b00_neg_Lneg += 1
        if a10v < -1e-10:
            a10_neg += 1
            if is_Lneg: a10_neg_Lneg += 1
        if a01v < -1e-10:
            a01_neg += 1
            if is_Lneg: a01_neg_Lneg += 1
        if d1v < -1e-10:
            d1_neg += 1
            if is_Lneg: d1_neg_Lneg += 1

        # Discriminant of C0 as quadratic in p (for fixed q)
        # C0 = (d1*q²+a10)p² + b00*q*p + (a00+a01*q²)
        # disc = b00²*q² - 4*(d1*q²+a10)*(a00+a01*q²)
        lead = d1v*qv**2 + a10v
        const = a00v + a01v*qv**2
        disc_val = b00v**2*qv**2 - 4*lead*const
        if disc_val > 1e-10:
            disc_pos_all += 1
            if is_Lneg: disc_pos_Lneg += 1

    pr(f"\n  Tested: {n_test} feasible samples, {n_Lneg} with L≤0")

    pr(f"\n  C1 components (should all be ≥ 0 for L≤0):")
    pr(f"    a20 < 0: {a20_neg}/{n_test}")
    pr(f"    a02 < 0: {a02_neg}/{n_test}")
    pr(f"    a12 < 0: {a12_neg}/{n_test}  (a12<0 when L>0: {a12_neg_Lneg} when L≤0)")
    pr(f"    C1 < 0 overall: {C1_neg_all}/{n_test}")
    pr(f"    C1 < 0 when L≤0: {C1_neg_Lneg}/{n_Lneg}")

    pr(f"\n  C0 components:")
    pr(f"    a00 < 0: {a00_neg}/{n_test}  (when L≤0: {a00_neg_Lneg}/{n_Lneg})")
    pr(f"    b00 > 0: {b00_pos}/{n_test}  (when L≤0 b00<0: {b00_neg_Lneg}/{n_Lneg})")
    pr(f"    a10 < 0: {a10_neg}/{n_test}  (when L≤0: {a10_neg_Lneg}/{n_Lneg})")
    pr(f"    a01 < 0: {a01_neg}/{n_test}  (when L≤0: {a01_neg_Lneg}/{n_Lneg})")
    pr(f"    delta1 < 0: {d1_neg}/{n_test}  (when L≤0: {d1_neg_Lneg}/{n_Lneg})")
    pr(f"    C0 < 0 overall: {C0_neg_all}/{n_test}")
    pr(f"    C0 < 0 when L≤0: {C0_neg_Lneg}/{n_Lneg}")
    pr(f"    disc>0 (C0 not PSD in p): {disc_pos_all}/{n_test}  (when L≤0: {disc_pos_Lneg}/{n_Lneg})")

    if C1_neg_Lneg == 0 and C0_neg_Lneg == 0:
        pr("\n  *** FOR L ≤ 0: C1 ≥ 0 AND C0 ≥ 0 → K_red ≥ 0 ***")
        pr("  Proof reduces to: show C1 ≥ 0 and C0 ≥ 0 when L ≤ 0")

    if C1_neg_all == 0:
        pr("\n  *** C1 ≥ 0 ON ALL FEASIBLE! (not just L≤0) ***")

    if C0_neg_all == 0:
        pr("\n  *** C0 ≥ 0 ON ALL FEASIBLE! (not just L≤0) ***")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("L > 0 CASE: C1 MIGHT BE NEGATIVE")
    pr('='*72)

    if C1_neg_all > 0:
        pr(f"  C1 < 0 in {C1_neg_all} cases (all with L>0)")
        pr("  For L>0, need C1·u² + C0 ≥ 0 with C1 possibly < 0")
        pr("  But u² = (p+q)² is bounded by f2c constraint")
        # Test: is K_red still ≥ 0?
        K_neg_count = 0
        for _ in range(200000):
            rv = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
            xv = float(rng.uniform(0.001, 0.999))
            yv = float(rng.uniform(0.001, 0.999))
            if L_fn(rv, xv, yv) <= 0:
                continue
            f2p_max = 2*(1-xv)
            f2q_max = 2*rv**3*(1-yv)
            pmax = np.sqrt(max(f2p_max/9, 0))
            qmax = np.sqrt(max(f2q_max/9, 0))
            pv = float(rng.uniform(-pmax, pmax))
            qv = float(rng.uniform(-qmax, qmax))
            f2c_v = 2*(1+rv)**3 - 2*(1+rv)*(3*xv+3*yv*rv**2+2*rv)/3 - 9*(pv+qv)**2
            if f2c_v <= 0:
                continue
            Kv = float(K_fn(rv, xv, yv, pv, qv))
            if Kv < -1e-8:
                K_neg_count += 1
        pr(f"  K_red < 0 when L > 0: {K_neg_count} (should be 0)")
    else:
        pr("  C1 ≥ 0 everywhere! L>0 case reduces to C0 ≥ 0 as well.")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
