#!/usr/bin/env python3
"""P4 Path2: High-precision check of A and K_red near (Pmax, Qmax) corner.

A(Pmax,Qmax) = 0 exactly (symbolic). The question: is A < 0 near the
corner, and if so, does K_red = A + sqrt(PQ)*B compensate?

Uses mpmath for exact evaluation to avoid float64 cancellation at extreme r.
"""
from __future__ import annotations

import time
import sympy as sp
import mpmath

mpmath.mp.dps = 50  # 50 decimal digits


def pr(*args):
    print(*args, flush=True)


def build_exact_k_red():
    s, t, u, v, a, b = sp.symbols("s t u v a b")
    r, x, y, p, q = sp.symbols("r x y p q")
    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)
    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)
    S = s+t; U = u+v; A_conv = a+b+s*t/6
    surplus_num = sp.expand(
        T2R_num(S,U,A_conv)*T2R_den(s,u,a)*T2R_den(t,v,b)
        - T2R_num(s,u,a)*T2R_den(S,U,A_conv)*T2R_den(t,v,b)
        - T2R_num(t,v,b)*T2R_den(S,U,A_conv)*T2R_den(s,u,a))
    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**sp.Rational(3,2), v: q*s**sp.Rational(3,2)}
    K_exact = sp.expand(surplus_num.subs(subs_norm) / s**16)
    return sp.expand(sp.cancel(K_exact / r**2))


def decompose_AB(K_red):
    r, x, y, p, q = sp.symbols("r x y p q")
    P, Q = sp.symbols("P Q")
    poly = sp.Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr += c * p**i * q**j
        elif i % 2 == 1 and j % 2 == 1:
            B_expr += c * p**(i-1) * q**(j-1)
    A_PQ = sp.expand(A_expr.subs({p**2: P, q**2: Q}))
    B_PQ = sp.expand(B_expr.subs({p**2: P, q**2: Q}))
    return A_PQ, B_PQ


def main():
    t0 = time.time()
    r, x, y, p, q = sp.symbols("r x y p q")
    P, Q = sp.symbols("P Q")

    pr("Building K_red...")
    K_red = build_exact_k_red()
    pr("Decomposing A, B...")
    A_PQ, B_PQ = decompose_AB(K_red)

    # Lambdify with mpmath for high precision
    K_fn = sp.lambdify((r, x, y, p, q), K_red, "mpmath")
    A_fn = sp.lambdify((r, x, y, P, Q), A_PQ, "mpmath")
    B_fn = sp.lambdify((r, x, y, P, Q), B_PQ, "mpmath")

    # ---------------------------------------------------------------
    # Test 1: at the reported problem point (extreme r)
    # ---------------------------------------------------------------
    pr("\n=== Test at extreme r ===")
    rv = mpmath.mpf("973702.105142")
    xv = mpmath.mpf("0.877841")
    yv = mpmath.mpf("0.486663")

    Pm = 2 * (1 - xv) / 9
    Qm = 2 * rv**3 * (1 - yv) / 9

    # Check at several fractions near corner
    for fP, fQ in [(0.95, 0.95), (0.99, 0.99), (0.999, 0.999), (1.0, 1.0)]:
        Pv = mpmath.mpf(fP) * Pm
        Qv = mpmath.mpf(fQ) * Qm
        pv = mpmath.sqrt(Pv)
        qv = mpmath.sqrt(Qv)

        Av = A_fn(rv, xv, yv, Pv, Qv)
        Bv = B_fn(rv, xv, yv, Pv, Qv)
        Kv = K_fn(rv, xv, yv, pv, qv)
        Kv_check = Av + mpmath.sqrt(Pv * Qv) * Bv

        pr(f"\n  fP={fP}, fQ={fQ}:")
        pr(f"  A = {mpmath.nstr(Av, 15)}")
        pr(f"  B = {mpmath.nstr(Bv, 15)}")
        pr(f"  sqrt(PQ)*B = {mpmath.nstr(mpmath.sqrt(Pv*Qv) * Bv, 15)}")
        pr(f"  K_red = {mpmath.nstr(Kv, 15)}")
        pr(f"  A + sqrt(PQ)*B = {mpmath.nstr(Kv_check, 15)}")
        pr(f"  K >= 0: {Kv >= 0}")

    # ---------------------------------------------------------------
    # Test 2: moderate r values near corner
    # ---------------------------------------------------------------
    pr("\n=== Test at moderate r ===")
    for rv_val in ["0.5", "1.0", "2.0", "10.0", "100.0"]:
        rv2 = mpmath.mpf(rv_val)
        xv2 = mpmath.mpf("0.7")
        yv2 = mpmath.mpf("0.7")
        Pm2 = 2 * (1 - xv2) / 9
        Qm2 = 2 * rv2**3 * (1 - yv2) / 9

        for fP, fQ in [(0.5, 0.5), (0.9, 0.9), (0.99, 0.99)]:
            Pv2 = mpmath.mpf(fP) * Pm2
            Qv2 = mpmath.mpf(fQ) * Qm2
            pv2 = mpmath.sqrt(Pv2)
            qv2 = mpmath.sqrt(Qv2)

            Av2 = A_fn(rv2, xv2, yv2, Pv2, Qv2)
            Bv2 = B_fn(rv2, xv2, yv2, Pv2, Qv2)
            Kv2 = K_fn(rv2, xv2, yv2, pv2, qv2)

            pr(f"  r={rv_val}, fP=fQ={fP}: A={mpmath.nstr(Av2, 10)}, B={mpmath.nstr(Bv2, 10)}, K={mpmath.nstr(Kv2, 10)}")

    # ---------------------------------------------------------------
    # Test 3: B at corner
    # ---------------------------------------------------------------
    pr("\n=== B at (Pmax, Qmax) symbolically ===")
    Pmax_sym = sp.Rational(2, 9) * (1 - sp.Symbol("x"))
    Qmax_sym = sp.Rational(2, 9) * sp.Symbol("r")**3 * (1 - sp.Symbol("y"))
    B_corner = sp.expand(B_PQ.subs({P: Pmax_sym, Q: Qmax_sym}))
    B_corner_f = sp.factor(B_corner)
    pr("B(Pmax, Qmax) =", B_corner_f)

    # ---------------------------------------------------------------
    # Test 4: K_red at (sqrt(Pmax), sqrt(Qmax)) symbolically
    # ---------------------------------------------------------------
    pr("\n=== K_red at boundary ===")
    K_boundary = sp.expand(K_red.subs({p: sp.sqrt(Pmax_sym), q: sp.sqrt(Qmax_sym)}))
    K_boundary_f = sp.factor(K_boundary)
    pr("K_red(sqrt(Pmax), sqrt(Qmax)) =", K_boundary_f)

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
