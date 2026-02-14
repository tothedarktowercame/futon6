#!/usr/bin/env python3
"""P4 Path2: Analyze structure of A(P,Q) to find algebraic proof of A >= 0.

A(P,Q) = a00 + a10P + a01Q + a20P^2 + a11PQ + a02Q^2 + a12(P^2Q + PQ^2)
with P in [0,Pmax], Q in [0,Qmax], Pmax=2(1-x)/9, Qmax=2r^3(1-y)/9.

Key known facts:
- a00 >= 0, a20 >= 0, a02 >= 0 (from Cycles 1-2)
- a12 = -1296rL (negative when L > 0)
- delta1 = a11 - a20 - a02 = 24W*(positive) < 0 on feasible

This script:
1) Factors A(Pmax,Qmax) — the corner value
2) Checks the Hessian of A at interior critical points
3) Tries to find a certificate decomposition
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp


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


def decompose_coeffs(K_red):
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
    Ad = sp.Poly(A_PQ, P, Q).as_dict()
    Bd = sp.Poly(B_PQ, P, Q).as_dict()
    return {
        "a00": sp.expand(Ad[(0,0)]), "a01": sp.expand(Ad[(0,1)]),
        "a02": sp.expand(Ad[(0,2)]), "a10": sp.expand(Ad[(1,0)]),
        "a11": sp.expand(Ad[(1,1)]), "a12": sp.expand(Ad[(1,2)]),
        "a20": sp.expand(Ad[(2,0)]), "b00": sp.expand(Bd[(0,0)]),
    }


def main():
    t0 = time.time()
    r, x, y = sp.symbols("r x y")
    P, Q = sp.symbols("P Q")

    pr("Building K_red...")
    K_red = build_exact_k_red()
    pr("Decomposing...")
    c = decompose_coeffs(K_red)

    a00=c["a00"]; a01=c["a01"]; a02=c["a02"]; a10=c["a10"]
    a11=c["a11"]; a12=c["a12"]; a20=c["a20"]; b00=c["b00"]

    A_poly = sp.expand(a00 + a10*P + a01*Q + a20*P**2 + a11*P*Q + a02*Q**2
                       + a12*P**2*Q + a12*P*Q**2)

    Pmax = sp.Rational(2,9)*(1-x)
    Qmax = sp.Rational(2,9)*r**3*(1-y)

    # ---------------------------------------------------------------
    # 1) Corner A(Pmax, Qmax)
    # ---------------------------------------------------------------
    pr("\n=== Corner A(Pmax, Qmax) ===")
    A_corner = sp.expand(A_poly.subs({P: Pmax, Q: Qmax}))
    A_corner_f = sp.factor(A_corner)
    pr("A(Pmax,Qmax) factored:", A_corner_f)

    # ---------------------------------------------------------------
    # 2) Boundary A(Pmax, Q) as polynomial in Q
    # ---------------------------------------------------------------
    pr("\n=== Boundary A(Pmax, Q) ===")
    A_Pmax = sp.expand(A_poly.subs(P, Pmax))
    A_Pmax_poly = sp.Poly(A_Pmax, Q)
    pr("A(Pmax,Q) degree:", A_Pmax_poly.degree())
    pr("A(Pmax,Q) coefficients:")
    for power, coeff in sorted(A_Pmax_poly.as_dict().items()):
        pr(f"  Q^{power[0]}:", sp.factor(coeff))

    # ---------------------------------------------------------------
    # 3) Boundary A(P, Qmax) as polynomial in P
    # ---------------------------------------------------------------
    pr("\n=== Boundary A(P, Qmax) ===")
    A_Qmax = sp.expand(A_poly.subs(Q, Qmax))
    A_Qmax_poly = sp.Poly(A_Qmax, P)
    pr("A(P,Qmax) degree:", A_Qmax_poly.degree())
    pr("A(P,Qmax) coefficients:")
    for power, coeff in sorted(A_Qmax_poly.as_dict().items()):
        pr(f"  P^{power[0]}:", sp.factor(coeff))

    # ---------------------------------------------------------------
    # 4) A(P,Q) concavity: check if A is concave in (P,Q)
    # ---------------------------------------------------------------
    pr("\n=== Hessian of A(P,Q) ===")
    A_PP = sp.diff(A_poly, P, 2)
    A_QQ = sp.diff(A_poly, Q, 2)
    A_PQ_cross = sp.diff(A_poly, P, Q)
    pr("A_PP =", sp.factor(sp.expand(A_PP)))
    pr("A_QQ =", sp.factor(sp.expand(A_QQ)))
    pr("A_PQ =", sp.factor(sp.expand(A_PQ_cross)))

    # For concavity: need A_PP <= 0, A_QQ <= 0, det(H) >= 0
    det_H = sp.expand(A_PP * A_QQ - A_PQ_cross**2)
    pr("det(H) =", sp.factor(det_H))

    # ---------------------------------------------------------------
    # 5) Check if A is concave (then min at boundary)
    # ---------------------------------------------------------------
    pr("\n=== Concavity analysis ===")
    # A_PP = 2*a20 + 2*a12*Q
    # A_QQ = 2*a02 + 2*a12*P
    # Since a20, a02 >= 0 and a12 can be negative (= -1296rL):
    # When L > 0: a12 < 0, so A_PP = 2a20 - 2*1296rL*Q <= 2a20 at Q=0
    # A_PP at Q=Qmax: 2(a20 + a12*Qmax) = 2*72(1-y)*My - 2*1296rL*2r^3(1-y)/9
    # = 144(1-y)*My - 576r^4L(1-y) = 144(1-y)(My - 4r^4L)... hmm that's not My - 4rL
    pr("A_PP =", sp.expand(A_PP), "= 2*(a20 + a12*Q)")
    pr("A_PP at Q=0:", sp.factor(sp.expand(A_PP.subs(Q, 0))))
    pr("A_PP at Q=Qmax:", sp.factor(sp.expand(A_PP.subs(Q, Qmax))))

    pr("A_QQ =", sp.expand(A_QQ), "= 2*(a02 + a12*P)")
    pr("A_QQ at P=0:", sp.factor(sp.expand(A_QQ.subs(P, 0))))
    pr("A_QQ at P=Pmax:", sp.factor(sp.expand(A_QQ.subs(P, Pmax))))

    # ---------------------------------------------------------------
    # 6) Numeric: A at the gradient = 0 critical points
    # ---------------------------------------------------------------
    pr("\n=== Numeric: interior critical points ===")
    A_fn = sp.lambdify((r, x, y, P, Q), A_poly, "numpy")
    AP_fn = sp.lambdify((r, x, y, P, Q), sp.diff(A_poly, P), "numpy")
    AQ_fn = sp.lambdify((r, x, y, P, Q), sp.diff(A_poly, Q), "numpy")

    rng = np.random.default_rng(42)
    n_test = 200000
    n_A_neg = 0
    min_A = float("inf")
    min_A_params = None

    # Also check if A has interior min < boundary min
    n_interior_crit = 0
    min_A_interior = float("inf")

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e6))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        Pm = 2*(1-xv)/9
        Qm = 2*rv**3*(1-yv)/9

        # Check corners
        corners = [(0,0), (Pm,0), (0,Qm), (Pm,Qm)]
        for Pv, Qv in corners:
            Av = float(A_fn(rv, xv, yv, Pv, Qv))
            if Av < min_A:
                min_A = Av
                min_A_params = (rv, xv, yv, Pv, Qv)
            if Av < -1e-12:
                n_A_neg += 1

        # Grid search for interior
        for fP in [0.2, 0.4, 0.6, 0.8, 0.95]:
            for fQ in [0.2, 0.4, 0.6, 0.8, 0.95]:
                Pv = fP * Pm
                Qv = fQ * Qm
                Av = float(A_fn(rv, xv, yv, Pv, Qv))
                if Av < min_A:
                    min_A = Av
                    min_A_params = (rv, xv, yv, Pv, Qv)
                if Av < -1e-12:
                    n_A_neg += 1

    pr(f"Tests: {n_test} params × 29 points = {n_test*29}")
    pr(f"A negative: {n_A_neg}")
    pr(f"min A: {min_A:.6e}")
    if min_A_params:
        pr(f"  at r={min_A_params[0]:.6f}, x={min_A_params[1]:.6f}, "
           f"y={min_A_params[2]:.6f}, P={min_A_params[3]:.6e}, Q={min_A_params[4]:.6e}")

    # ---------------------------------------------------------------
    # 7) Try decomposition A = non-neg parts
    # ---------------------------------------------------------------
    pr("\n=== Decomposition attempts ===")

    # Idea: A = C1*(P+Q)^2 + remainder, where C1 = a20P + a02Q + a12PQ
    # We already know K_red = C1*u^2 + C0. For A: A = K_red at pq=0
    # So A = C1*(p+q)^2 + C0 with pq=0, i.e., q=0 (then A = a00+a10P+a20P^2)
    # or p=0. That's just the edges.

    # Different idea: A = a20*P^2 + a02*Q^2 + a12*PQ*(P+Q) + (a10*P + a01*Q + a11*PQ + a00)
    # = P*(a20*P + a12*PQ) + Q*(a02*Q + a12*PQ) + a10*P + a01*Q + a11*PQ + a00
    # = P*(a20*P + a12*Q*P) + Q*(a02*Q + a12*P*Q) + linear terms + a00

    # Better idea: since a12 = -1296rL and a20, a02 >= 0:
    # a20 + a12*Q >= 0 iff Q <= a20/(1296rL)
    # At Q = Qmax: a20 + a12*Qmax = 72(1-y)*(My - 4rL*... hmm
    # We showed c4 = a02 + a12*P >= 0 for P in [0,Pmax]. Similarly a20 + a12*Q >= 0?

    # Check: a20 + a12*Qmax
    a20_plus_a12Qmax = sp.expand(a20 + a12*Qmax)
    a20_plus_a12Qmax_f = sp.factor(a20_plus_a12Qmax)
    pr("a20 + a12*Qmax:", a20_plus_a12Qmax_f)

    # Check: a02 + a12*Pmax
    a02_plus_a12Pmax = sp.expand(a02 + a12*Pmax)
    a02_plus_a12Pmax_f = sp.factor(a02_plus_a12Pmax)
    pr("a02 + a12*Pmax:", a02_plus_a12Pmax_f)

    # If both are >= 0: then for any (P,Q) in the box:
    # a20 + a12*Q = a20 + a12*Q >= a20 + a12*Qmax >= 0 (since a12 < 0, Q <= Qmax)
    # So a20*P^2 + a12*P^2*Q = P^2*(a20 + a12*Q) >= 0
    # Similarly a02*Q^2 + a12*P*Q^2 = Q^2*(a02 + a12*P) >= 0

    # Then A = P^2*(a20+a12Q) + Q^2*(a02+a12P) + a00 + a10P + a01Q + a11PQ
    # The first two terms are >= 0. The remaining is a bilinear form + constants.
    # Need: a00 + a10P + a01Q + a11PQ >= 0 on [0,Pmax]x[0,Qmax]

    pr("\n=== Bilinear remainder analysis ===")
    R_bilinear = sp.expand(a00 + a10*P + a01*Q + a11*P*Q)
    R_fn = sp.lambdify((r, x, y, P, Q), R_bilinear, "numpy")

    # This is a bilinear form. Min on box is at a corner!
    # Corners: (0,0)=a00, (Pmax,0)=a00+a10*Pmax, (0,Qmax)=a00+a01*Qmax,
    # (Pmax,Qmax) = a00+a10*Pmax+a01*Qmax+a11*Pmax*Qmax
    pr("R(0,0) = a00 (>= 0)")
    R_P0 = sp.expand(a00 + a10*Pmax)
    R_0Q = sp.expand(a00 + a01*Qmax)
    R_PQ = sp.expand(a00 + a10*Pmax + a01*Qmax + a11*Pmax*Qmax)
    pr("R(Pmax,0) factored:", sp.factor(R_P0))
    pr("R(0,Qmax) factored:", sp.factor(R_0Q))
    pr("R(Pmax,Qmax) factored:", sp.factor(R_PQ))

    # Numeric check
    n_R_neg = 0
    min_R = float("inf")
    rng2 = np.random.default_rng(123)
    for _ in range(200000):
        rv = float(np.exp(rng2.uniform(np.log(1e-6), np.log(1e6))))
        xv = float(rng2.uniform(1e-8, 1 - 1e-8))
        yv = float(rng2.uniform(1e-8, 1 - 1e-8))
        Pm = 2*(1-xv)/9; Qm = 2*rv**3*(1-yv)/9
        for Pv, Qv in [(0,0), (Pm,0), (0,Qm), (Pm,Qm)]:
            Rv = float(R_fn(rv, xv, yv, Pv, Qv))
            if Rv < -1e-12:
                n_R_neg += 1
            if Rv < min_R:
                min_R = Rv

    pr(f"R corners negative: {n_R_neg}/800k")
    pr(f"min R at corners: {min_R:.6e}")

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

    out = {
        "meta": {"date": "2026-02-13", "script": "scripts/explore-p4-path2-A-structure.py",
                 "runtime_sec": round(elapsed, 3)},
        "A_corner_factor": str(A_corner_f),
        "a20_plus_a12Qmax_factor": str(a20_plus_a12Qmax_f),
        "a02_plus_a12Pmax_factor": str(a02_plus_a12Pmax_f),
        "R_P0_factor": str(sp.factor(R_P0)),
        "R_0Q_factor": str(sp.factor(R_0Q)),
        "R_PQ_factor": str(sp.factor(R_PQ)),
        "numeric": {
            "A_negative": n_A_neg,
            "min_A": min_A,
            "R_corners_negative": n_R_neg,
            "min_R_corners": min_R,
        }
    }
    out_path = Path("/home/joe/code/futon6/data/first-proof/p4-path2-A-structure.json")
    out_path.write_text(json.dumps(out, indent=2))
    pr(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
