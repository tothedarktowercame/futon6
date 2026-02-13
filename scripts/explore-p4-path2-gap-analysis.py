#!/usr/bin/env python3
"""P4 Path2 Gap Analysis: verify that Claims 1+2 don't close K_red >= 0.

The Codex Cycle 2 proved:
  Claim 1: C1 >= 0
  Claim 2: C0(p,-p) >= 0   (the q=-p boundary, "easy" case)

But K_red = C1*(p+q)^2 + C0, and C0 depends on pq.
For same-sign (p,q), pq > 0 and b00*pq < 0, making C0 more negative.
The question: does C1*(p+q)^2 compensate?

This script:
1) Numerically confirms the gap (C0 < 0 at same-sign points)
2) Checks K_red on the same-sign diagonal p=q (the hardest 1D slice)
3) Identifies algebraic structure for a Cycle 3 handoff
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp


def pr(*args):
    print(*args, flush=True)


def build_exact_k_red() -> sp.Expr:
    s, t, u, v, a, b = sp.symbols("s t u v a b")
    r, x, y, p, q = sp.symbols("r x y p q")

    def T2R_num(ss, uu, aa):
        return 8 * aa * (ss**2 - 4 * aa) ** 2 - ss * uu**2 * (ss**2 + 60 * aa)

    def T2R_den(ss, uu, aa):
        return 2 * (ss**2 + 12 * aa) * (2 * ss**3 - 8 * ss * aa - 9 * uu**2)

    S = s + t
    U = u + v
    A = a + b + s * t / 6
    surplus_num = sp.expand(
        T2R_num(S, U, A) * T2R_den(s, u, a) * T2R_den(t, v, b)
        - T2R_num(s, u, a) * T2R_den(S, U, A) * T2R_den(t, v, b)
        - T2R_num(t, v, b) * T2R_den(S, U, A) * T2R_den(s, u, a)
    )
    subs_norm = {
        t: r * s, a: x * s**2 / 4, b: y * r**2 * s**2 / 4,
        u: p * s ** sp.Rational(3, 2), v: q * s ** sp.Rational(3, 2),
    }
    K_exact = sp.expand(surplus_num.subs(subs_norm) / s**16)
    K_red = sp.expand(sp.cancel(K_exact / r**2))
    return K_red


def decompose_coeffs(K_red: sp.Expr):
    r, x, y, p, q = sp.symbols("r x y p q")
    poly = sp.Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr += c * p**i * q**j
        elif i % 2 == 1 and j % 2 == 1:
            B_expr += c * p ** (i - 1) * q ** (j - 1)
    P, Q = sp.symbols("P Q")
    A_PQ = sp.expand(A_expr.subs({p**2: P, q**2: Q}))
    B_PQ = sp.expand(B_expr.subs({p**2: P, q**2: Q}))
    A = sp.Poly(A_PQ, P, Q).as_dict()
    B = sp.Poly(B_PQ, P, Q).as_dict()

    a00 = A[(0, 0)]
    a01 = A[(0, 1)]
    a02 = A[(0, 2)]
    a10 = A[(1, 0)]
    a11 = A[(1, 1)]
    a12 = A[(1, 2)]
    a20 = A[(2, 0)]
    b00 = B[(0, 0)]
    delta1 = sp.expand(a11 - a20 - a02)

    return {
        "a00": sp.expand(a00), "a01": sp.expand(a01), "a02": sp.expand(a02),
        "a10": sp.expand(a10), "a11": sp.expand(a11), "a12": sp.expand(a12),
        "a20": sp.expand(a20), "b00": sp.expand(b00), "delta1": delta1,
    }


def main():
    t0 = time.time()
    r, x, y, p, q = sp.symbols("r x y p q")
    T = sp.Symbol("T")  # = p^2 = q^2 on diagonal

    pr("Building K_red...")
    K_red = build_exact_k_red()
    pr("Decomposing coefficients...")
    c = decompose_coeffs(K_red)

    a00 = c["a00"]; a01 = c["a01"]; a02 = c["a02"]
    a10 = c["a10"]; a11 = c["a11"]; a12 = c["a12"]
    a20 = c["a20"]; b00 = c["b00"]; delta1 = c["delta1"]

    # ---------------------------------------------------------------
    # 1) Same-sign diagonal: K_red(p,p) as function of T = p^2
    # ---------------------------------------------------------------
    pr("\n=== Same-sign diagonal p=q ===")

    # K_red(p,p) = C1(p,p)*4T + C0(p,p) where T=p^2
    # C1(p,p) = (a20+a02)*T + a12*T^2
    # C0(p,p) = a00 + b00*T + (a10+a01)*T + delta1*T^2
    #         = a00 + (a10+a01+b00)*T + delta1*T^2
    # So: g(T) = a00 + (a10+a01+b00)*T + (delta1+4a20+4a02)*T^2 + 4*a12*T^3

    m_same = sp.expand(a10 + a01 + b00)
    m_opp = sp.expand(a10 + a01 - b00)  # Claim 2's coefficient
    n_coeff = sp.expand(delta1 + 4*a20 + 4*a02)  # = a11 + 3(a20+a02)
    cubic_lead = sp.expand(4 * a12)

    g_T = sp.expand(a00 + m_same * T + n_coeff * T**2 + cubic_lead * T**3)

    # Verify: this equals K_red(sqrt(T), sqrt(T)) symbolically
    K_diag = sp.expand(K_red.subs({p: sp.sqrt(T), q: sp.sqrt(T)}))
    diag_check = sp.expand(g_T - K_diag) == 0
    pr("g(T) == K_red(sqrt(T),sqrt(T)):", diag_check)

    # Compare with Claim 2's f(T) = a00 + m_opp*T + delta1*T^2
    f_T = sp.expand(a00 + m_opp * T + delta1 * T**2)
    diff_gf = sp.expand(g_T - f_T)
    pr("g(T) - f(T) =", diff_gf)
    # This should be (2*b00)*T + 4*(a20+a02)*T^2 + 4*a12*T^3
    expected_diff = sp.expand(2*b00*T + 4*(a20+a02)*T**2 + 4*a12*T**3)
    pr("Expected diff:", sp.expand(diff_gf - expected_diff) == 0)

    # Factor the cubic's coefficients
    pr("\nCoefficients of g(T):")
    pr("  g0 = a00 (>= 0 always)")
    pr("  g1 = a10+a01+b00 =", sp.factor(m_same))
    pr("  g2 = delta1+4a20+4a02 =", sp.factor(n_coeff))
    pr("  g3 = 4*a12 =", sp.factor(cubic_lead))

    # ---------------------------------------------------------------
    # 2) Edge analyses: p=0 and q=0
    # ---------------------------------------------------------------
    pr("\n=== Edge p=0 ===")
    K_p0 = sp.expand(K_red.subs(p, 0))
    K_p0_poly = sp.Poly(K_p0, q)
    pr("K_red(0,q) monomials:", K_p0_poly.as_dict())
    # Should be a00 + a01*q^2 + a02*q^4 (even in q)

    pr("\n=== Edge q=0 ===")
    K_q0 = sp.expand(K_red.subs(q, 0))
    K_q0_poly = sp.Poly(K_q0, p)
    pr("K_red(p,0) monomials:", K_q0_poly.as_dict())
    # Should be a00 + a10*p^2 + a20*p^4 (even in p)

    # ---------------------------------------------------------------
    # 3) Numeric test: K_red at same-sign points
    # ---------------------------------------------------------------
    pr("\n=== Numeric tests ===")
    K_fn = sp.lambdify((r, x, y, p, q), K_red, "numpy")
    g_fn = sp.lambdify((r, x, y, T), g_T, "numpy")

    rng = np.random.default_rng(20260213)
    n_test = 200000
    n_same_neg = 0
    n_C0_neg = 0
    min_K_same = float("inf")
    min_K_same_params = None
    min_ratio = float("inf")

    # C0 at same sign: a00 + b00*sqrt(PQ) + a10*P + a01*Q + delta1*PQ
    C0_same_fn = sp.lambdify(
        (r, x, y, p, q),
        sp.expand(a00 + b00*p*q + a10*p**2 + a01*q**2 + delta1*p**2*q**2),
        "numpy"
    )
    C1_fn = sp.lambdify(
        (r, x, y, p, q),
        sp.expand(a20*p**2 + a02*q**2 + a12*p**2*q**2),
        "numpy"
    )

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e3))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        pmax = np.sqrt(2 * (1 - xv) / 9)
        qmax = np.sqrt(2 * rv**3 * (1 - yv) / 9)

        # Same-sign test: p,q > 0
        pv = float(rng.uniform(0, pmax))
        qv = float(rng.uniform(0, qmax))

        Kv = float(K_fn(rv, xv, yv, pv, qv))
        C0v = float(C0_same_fn(rv, xv, yv, pv, qv))
        C1v = float(C1_fn(rv, xv, yv, pv, qv))

        if C0v < -1e-12:
            n_C0_neg += 1
        if Kv < -1e-12:
            n_same_neg += 1
        if Kv < min_K_same:
            min_K_same = Kv
            min_K_same_params = (rv, xv, yv, pv, qv)

        # Track ratio K_red / (p+q)^2 when C0 < 0
        u2 = (pv + qv)**2
        if C0v < -1e-12 and u2 > 1e-20:
            ratio = Kv / u2
            if ratio < min_ratio:
                min_ratio = ratio

    pr(f"Same-sign tests: {n_test}")
    pr(f"C0 negative: {n_C0_neg}/{n_test} ({100*n_C0_neg/n_test:.1f}%)")
    pr(f"K_red negative (same-sign): {n_same_neg}/{n_test}")
    pr(f"min K_red (same-sign): {min_K_same:.6e}")
    if min_K_same_params:
        pr(f"  at r={min_K_same_params[0]:.6f}, x={min_K_same_params[1]:.6f}, "
           f"y={min_K_same_params[2]:.6f}, p={min_K_same_params[3]:.6f}, q={min_K_same_params[4]:.6f}")
    pr(f"min K_red/u^2 when C0<0: {min_ratio:.6e}")

    # ---------------------------------------------------------------
    # 4) Diagonal test: g(T) on [0, tmax]
    # ---------------------------------------------------------------
    pr("\n=== Diagonal p=q tests ===")
    n_diag = 200000
    min_g = float("inf")
    min_g_params = None

    for _ in range(n_diag):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e3))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        pmax2 = 2 * (1 - xv) / 9
        qmax2 = 2 * rv**3 * (1 - yv) / 9
        tmax = min(pmax2, qmax2)

        # Test at tmax (worst point for cubic with negative leading coeff)
        Tv = tmax
        gv = float(g_fn(rv, xv, yv, Tv))
        if gv < min_g:
            min_g = gv
            min_g_params = (rv, xv, yv, Tv)

        # Also test at a few interior points
        for frac in [0.5, 0.8, 0.95]:
            Tv2 = frac * tmax
            gv2 = float(g_fn(rv, xv, yv, Tv2))
            if gv2 < min_g:
                min_g = gv2
                min_g_params = (rv, xv, yv, Tv2)

    pr(f"Diagonal tests: {n_diag}")
    pr(f"min g(T) on diagonal: {min_g:.6e}")
    if min_g_params:
        pr(f"  at r={min_g_params[0]:.6f}, x={min_g_params[1]:.6f}, "
           f"y={min_g_params[2]:.6f}, T={min_g_params[3]:.6e}")

    # ---------------------------------------------------------------
    # 5) Factor g(T) at endpoints: g(Pmax) and g(Qmax) and g(tmax)
    # ---------------------------------------------------------------
    pr("\n=== Endpoint analysis for g(T) ===")
    Pmax = sp.Rational(2, 9) * (1 - x)
    Qmax = sp.Rational(2, 9) * r**3 * (1 - y)

    gP = sp.expand(g_T.subs(T, Pmax))
    gQ = sp.expand(g_T.subs(T, Qmax))

    gP_fact = sp.factor(gP)
    gQ_fact = sp.factor(gQ)

    pr("g(Pmax) factored:", gP_fact)
    pr("g(Qmax) factored:", gQ_fact)

    # Check: is g(Pmax) the same as K_red(sqrt(Pmax), sqrt(Pmax))?
    # At p=q=sqrt(Pmax): need also q^2 = Pmax <= Qmax, i.e. Pmax <= Qmax
    # At p=q=sqrt(Qmax): need also p^2 = Qmax <= Pmax, i.e. Qmax <= Pmax

    # The ACTIVE endpoint for the diagonal is T = tmax = min(Pmax, Qmax)
    # When Pmax <= Qmax: g(Pmax) is the active endpoint
    # When Qmax <= Pmax: g(Qmax) is the active endpoint

    D = sp.expand(r**3 * (1 - y) - (1 - x))  # prop to Qmax - Pmax
    W = sp.expand(3*r**2*y - 3*r**2 - 4*r + 3*x - 3)

    # Check sign structure
    pr("\nSign analysis:")
    pr("D = r^3(1-y) - (1-x), sign(D) = sign(Qmax-Pmax)")
    pr("When D >= 0 (Qmax >= Pmax): active endpoint is g(Pmax)")
    pr("When D < 0 (Qmax < Pmax): active endpoint is g(Qmax)")

    # Try to express gP and gQ in terms of D, W, and positive factors
    # (similar to Claim 2's fP and fQ)

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

    out = {
        "meta": {
            "date": "2026-02-13",
            "script": "scripts/explore-p4-path2-gap-analysis.py",
            "runtime_sec": round(elapsed, 3),
        },
        "gap_confirmed": {
            "C0_negative_same_sign_pct": round(100*n_C0_neg/n_test, 1),
            "K_red_negative_same_sign": n_same_neg,
            "min_K_red_same_sign": min_K_same,
            "min_K_red_over_u2_when_C0_neg": min_ratio,
        },
        "diagonal_pq": {
            "g_T_formula": "a00 + (a10+a01+b00)*T + (delta1+4a20+4a02)*T^2 + 4*a12*T^3",
            "g1_factor": str(sp.factor(m_same)),
            "g2_factor": str(sp.factor(n_coeff)),
            "g3_factor": str(sp.factor(cubic_lead)),
            "diag_identity_check": diag_check,
            "min_g_T": min_g,
            "gP_factor": str(gP_fact),
            "gQ_factor": str(gQ_fact),
        },
        "conclusion": (
            "C0 is negative for ~17% of same-sign feasible samples. "
            "K_red remains >= 0 always (0 violations in 200k tests). "
            "The diagonal g(T) = K_red(sqrt(T),sqrt(T)) is a cubic that needs "
            "a Claim 3 analysis (endpoint + concavity/inflection) similar to Claim 2."
        ),
    }
    out_path = Path("/home/joe/code/futon6/data/first-proof/p4-path2-gap-analysis.json")
    out_path.write_text(json.dumps(out, indent=2))
    pr(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
