#!/usr/bin/env python3
"""P4 Path2 Gap Analysis part 2: cubic interior behavior.

g(T) = a00 + g1*T + g2*T^2 + g3*T^3 is the same-sign diagonal.
We proved g(0) >= 0 and g(tmax) >= 0 via endpoint factorizations.
Question: does g dip below zero in (0, tmax)?

Key checks:
- Is g'(0) = g1 >= 0? If so, cubic is initially increasing -> nonneg throughout.
- Is g2 <= 0? If so, g'' <= 0 at T=0, and since g3 < 0, g''(T) is decreasing,
  so g is concave throughout -> min at endpoints.
- If neither, characterize the minimum more carefully.
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
    S = s + t; U = u + v; A = a + b + s * t / 6
    surplus_num = sp.expand(
        T2R_num(S, U, A) * T2R_den(s, u, a) * T2R_den(t, v, b)
        - T2R_num(s, u, a) * T2R_den(S, U, A) * T2R_den(t, v, b)
        - T2R_num(t, v, b) * T2R_den(S, U, A) * T2R_den(s, u, a))
    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**sp.Rational(3,2), v: q*s**sp.Rational(3,2)}
    K_exact = sp.expand(surplus_num.subs(subs_norm) / s**16)
    return sp.expand(sp.cancel(K_exact / r**2))


def decompose_coeffs(K_red):
    r, x, y, p, q = sp.symbols("r x y p q")
    poly = sp.Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr += c * p**i * q**j
        elif i % 2 == 1 and j % 2 == 1:
            B_expr += c * p**(i-1) * q**(j-1)
    P, Q = sp.symbols("P Q")
    A_PQ = sp.expand(A_expr.subs({p**2: P, q**2: Q}))
    B_PQ = sp.expand(B_expr.subs({p**2: P, q**2: Q}))
    A = sp.Poly(A_PQ, P, Q).as_dict()
    B = sp.Poly(B_PQ, P, Q).as_dict()
    a00 = A[(0,0)]; a01 = A[(0,1)]; a02 = A[(0,2)]
    a10 = A[(1,0)]; a11 = A[(1,1)]; a12 = A[(1,2)]
    a20 = A[(2,0)]; b00 = B[(0,0)]
    delta1 = sp.expand(a11 - a20 - a02)
    return {k: sp.expand(v) for k, v in {
        "a00": a00, "a01": a01, "a02": a02, "a10": a10,
        "a11": a11, "a12": a12, "a20": a20, "b00": b00, "delta1": delta1,
    }.items()}


def main():
    t0 = time.time()
    r, x, y = sp.symbols("r x y")
    T = sp.Symbol("T")

    pr("Building K_red...")
    K_red = build_exact_k_red()
    pr("Decomposing...")
    c = decompose_coeffs(K_red)

    a00 = c["a00"]; a01 = c["a01"]; a02 = c["a02"]
    a10 = c["a10"]; a12 = c["a12"]
    a20 = c["a20"]; b00 = c["b00"]; delta1 = c["delta1"]

    g0 = a00
    g1 = sp.expand(a10 + a01 + b00)
    g2 = sp.expand(delta1 + 4*a20 + 4*a02)
    g3 = sp.expand(4 * a12)

    Pmax = sp.Rational(2, 9) * (1 - x)
    Qmax = sp.Rational(2, 9) * r**3 * (1 - y)

    # ---------------------------------------------------------------
    # Numeric: check g1 >= 0 and g2 <= 0 on feasible domain
    # ---------------------------------------------------------------
    pr("Lambdifying...")
    g0_fn = sp.lambdify((r, x, y), g0, "numpy")
    g1_fn = sp.lambdify((r, x, y), g1, "numpy")
    g2_fn = sp.lambdify((r, x, y), g2, "numpy")
    g3_fn = sp.lambdify((r, x, y), g3, "numpy")

    rng = np.random.default_rng(20260213)
    n_test = 300000
    n_g1_neg = 0
    n_g2_pos = 0
    n_g2_neg = 0
    n_cubic_has_interior_min = 0  # g'(T*)=0 for T* in (0, tmax)
    n_interior_min_neg = 0  # g(T*) < 0 at that min
    min_g_interior = float("inf")

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e3))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        g0v = float(g0_fn(rv, xv, yv))
        g1v = float(g1_fn(rv, xv, yv))
        g2v = float(g2_fn(rv, xv, yv))
        g3v = float(g3_fn(rv, xv, yv))

        if g1v < -1e-12:
            n_g1_neg += 1
        if g2v > 1e-12:
            n_g2_pos += 1
        if g2v < -1e-12:
            n_g2_neg += 1

        pmax2 = 2 * (1 - xv) / 9
        qmax2 = 2 * rv**3 * (1 - yv) / 9
        tmax = min(pmax2, qmax2)

        # Find local min of g in (0, tmax)
        # g'(T) = g1 + 2*g2*T + 3*g3*T^2
        # Roots: T = (-2g2 ± sqrt(4g2^2 - 12g3*g1)) / (6g3)
        disc = 4*g2v**2 - 12*g3v*g1v
        if disc > 0 and abs(g3v) > 1e-30:
            sq = np.sqrt(disc)
            T1 = (-2*g2v - sq) / (6*g3v)  # smaller root (local min for g3<0)
            T2 = (-2*g2v + sq) / (6*g3v)  # larger root (local max for g3<0)
            # For g3 < 0: T1 is local min, T2 is local max
            if g3v < 0:
                Tmin = T1
            else:
                Tmin = T2
            if 0 < Tmin < tmax:
                n_cubic_has_interior_min += 1
                gmin = g0v + g1v*Tmin + g2v*Tmin**2 + g3v*Tmin**3
                if gmin < min_g_interior:
                    min_g_interior = gmin
                if gmin < -1e-9:
                    n_interior_min_neg += 1

    pr(f"\n=== Coefficient sign analysis ({n_test} samples) ===")
    pr(f"g1 < 0: {n_g1_neg}/{n_test} ({100*n_g1_neg/n_test:.1f}%)")
    pr(f"g2 > 0: {n_g2_pos}/{n_test} ({100*n_g2_pos/n_test:.1f}%)")
    pr(f"g2 < 0: {n_g2_neg}/{n_test} ({100*n_g2_neg/n_test:.1f}%)")
    pr(f"\nCubic has interior min in (0,tmax): {n_cubic_has_interior_min}/{n_test}")
    pr(f"Interior min < 0: {n_interior_min_neg}/{n_test}")
    pr(f"Min interior g value: {min_g_interior:.6e}")

    # ---------------------------------------------------------------
    # Symbolic: try to show g1 >= 0 OR find the right proof strategy
    # ---------------------------------------------------------------
    pr("\n=== Symbolic analysis ===")

    # Check whether g1 factors nicely through W or other known factors
    W = sp.expand(3*r**2*y - 3*r**2 - 4*r + 3*x - 3)
    L = sp.expand(9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2)

    g1_fact = sp.factor(g1)
    pr("g1 factored:", g1_fact)

    # Check if W divides g1
    g1_rem = sp.rem(sp.Poly(g1, r), sp.Poly(W, r))
    pr("g1 mod W (as poly in r):", "zero" if g1_rem.is_zero else "nonzero (W does not divide g1)")

    # Check g2
    g2_fact = sp.factor(g2)
    pr("g2 factored:", g2_fact)

    # ---------------------------------------------------------------
    # Endpoint factorization details
    # ---------------------------------------------------------------
    pr("\n=== Endpoint factorization structure ===")
    gP = sp.expand(g0 + g1*Pmax + g2*Pmax**2 + g3*Pmax**3)
    gQ = sp.expand(g0 + g1*Qmax + g2*Qmax**2 + g3*Qmax**3)
    gP_f = sp.factor(gP)
    gQ_f = sp.factor(gQ)

    # Extract the "new" factors compared to Claim 2
    D = sp.expand(r**3*(1-y) - (1-x))
    pos_core = sp.expand(3*r**2*y + r**2 + 4*r + 3*x + 1)

    # From gap-analysis: g(Pmax) = 32*(x-1)*(3x+1)^2*(3y+1)*D*pos_core*new_P/27
    # Factor: (r+1)*W from Claim 2's fP is replaced by new_P
    # new_P = (r+1)*W + 12(1-x)
    rp1_W = sp.expand((r+1)*W)
    new_P_expected = sp.expand(rp1_W + 12*(1-x))
    new_Q_expected = sp.expand(rp1_W + 12*r**3*(1-y))

    # Verify the factorization
    gP_target = sp.expand(32*(x-1)*(3*x+1)**2*(3*y+1)*D*pos_core*new_P_expected/27)
    gQ_target = sp.expand(-32*r**4*(3*x+1)*(y-1)*(3*y+1)**2*D*pos_core*new_Q_expected/27)

    pr("g(Pmax) identity check:", sp.expand(gP - gP_target) == 0)
    pr("g(Qmax) identity check:", sp.expand(gQ - gQ_target) == 0)

    # Now prove new_P <= 0 when D >= 0:
    # new_P = (r+1)*W + 12(1-x)
    # = -(r+1)(3r^2(1-y) + 4r + 3(1-x)) + 12(1-x)
    # = -3(r+1)r^2(1-y) - 4r(r+1) + (1-x)(12-3(r+1))
    # = -3(r+1)r^2(1-y) - 4r(r+1) + 3(1-x)(3-r)
    # When D >= 0: (1-x) <= r^3(1-y)
    # new_P <= -3(r+1)r^2(1-y) - 4r(r+1) + 3r^3(1-y)(3-r)
    # = -(1-y)*3r^2(r-1)^2 - 4r(r+1)  (derived algebraically)
    # <= 0

    new_P_bound = sp.expand(
        -3*(r+1)*r**2*(1-y) - 4*r*(r+1) + 3*r**3*(1-y)*(3-r)
    )
    new_P_simplified = sp.expand(new_P_bound)
    new_P_factored = sp.factor(new_P_simplified)
    pr("new_P upper bound (when D>=0) factors:", new_P_factored)

    new_Q_bound = sp.expand(
        3*r**2*(1-y)*(3*r - 1) - 4*r*(r+1) - 3*(r+1)*(1-x)
    )
    # When D <= 0: (1-y) <= (1-x)/r^3
    # new_Q <= 3r^2*(1-x)/r^3*(3r-1) - 4r(r+1) - 3(r+1)(1-x)
    # = (1-x)*3(3r-1)/r - 4r(r+1) - 3(r+1)(1-x)
    # = (1-x)[9-3/r-3r-3] - 4r(r+1)
    # = -3(1-x)(r-1)^2/r - 4r(r+1)
    new_Q_bound2 = sp.expand(
        -3*(1-x)*(r-1)**2/r - 4*r*(r+1)
    )
    pr("new_Q upper bound (when D<=0):", sp.factor(new_Q_bound2))

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

    out = {
        "meta": {
            "date": "2026-02-13",
            "script": "scripts/explore-p4-path2-gap-analysis2.py",
            "runtime_sec": round(elapsed, 3),
        },
        "coefficient_signs": {
            "g1_neg_pct": round(100*n_g1_neg/n_test, 1),
            "g2_pos_pct": round(100*n_g2_pos/n_test, 1),
            "g2_neg_pct": round(100*n_g2_neg/n_test, 1),
        },
        "cubic_interior": {
            "has_interior_min_pct": round(100*n_cubic_has_interior_min/n_test, 1),
            "interior_min_negative": n_interior_min_neg,
            "min_interior_g": min_g_interior,
        },
        "g1_factor": str(sp.factor(g1)),
        "g2_factor": str(sp.factor(g2)),
        "endpoint_identities": {
            "gP_check": sp.expand(gP - gP_target) == 0,
            "gQ_check": sp.expand(gQ - gQ_target) == 0,
            "new_P_formula": "new_P = (r+1)*W + 12*(1-x)",
            "new_Q_formula": "new_Q = (r+1)*W + 12*r^3*(1-y)",
            "new_P_bound_when_D_ge_0": str(new_P_factored),
            "new_Q_bound_when_D_le_0": str(sp.factor(new_Q_bound2)),
        },
    }
    out_path = Path("/home/joe/code/futon6/data/first-proof/p4-path2-gap-analysis2.json")
    out_path.write_text(json.dumps(out, indent=2))
    pr(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
