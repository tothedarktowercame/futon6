#!/usr/bin/env python3
"""P4 Path2: Check AM-GM bound for A >= 0 proof.

For L >= 0 case, Ā(u,v) = (Pmax-P)·F + (Qmax-Q)·G + ā11·(Pmax-P)(Qmax-Q)
where F = ā10 + ā20u + ā21uv >= 0, G = ā01 + ā02v + ā12uv >= 0.
By AM-GM: (Pmax-P)F + (Qmax-Q)G >= 2√(FG(Pmax-P)(Qmax-Q)).
So Ā >= √(uv)·[2√(FG) - |ā11|√(uv)] >= 0 iff 4FG >= ā11²uv.

Since FG >= ā10ā01 (at u=v=0, F,G increase with u,v when ā21,ā12>=0):
Need: 4ā10ā01 >= ā11²·Pmax·Qmax (the weakest bound).

Also check if the full K_red = A + √(PQ)B >= 0 can be proved by a similar route.
"""
import json
import time
import numpy as np
import sympy as sp
from pathlib import Path


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
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr += c * p**i * q**j
    A_PQ = sp.expand(A_expr.subs({p**2: P, q**2: Q}))
    Ad = sp.Poly(A_PQ, P, Q).as_dict()
    return {k: sp.expand(v) for k, v in {
        "a00": Ad[(0,0)], "a01": Ad[(0,1)], "a02": Ad[(0,2)],
        "a10": Ad[(1,0)], "a11": Ad[(1,1)], "a12": Ad[(1,2)],
        "a20": Ad[(2,0)],
    }.items()}


def main():
    t0 = time.time()
    r, x, y = sp.symbols("r x y")

    pr("Building K_red & decomposing...")
    K_red = build_exact_k_red()
    c = decompose_coeffs(K_red)
    a00=c["a00"]; a01=c["a01"]; a02=c["a02"]; a10=c["a10"]
    a11=c["a11"]; a12=c["a12"]; a20=c["a20"]

    Pmax = sp.Rational(2,9)*(1-x)
    Qmax = sp.Rational(2,9)*r**3*(1-y)

    # Taylor coefficients
    a10_bar = sp.expand(-(a10 + 2*a20*Pmax + a11*Qmax + 2*a12*Pmax*Qmax + a12*Qmax**2))
    a01_bar = sp.expand(-(a01 + a11*Pmax + 2*a02*Qmax + a12*Pmax**2 + 2*a12*Pmax*Qmax))
    a20_bar = sp.expand(a20 + a12*Qmax)
    a02_bar = sp.expand(a02 + a12*Pmax)
    a11_bar = sp.expand(a11 + 2*a12*(Pmax + Qmax))
    L = sp.expand(9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2)

    # ---------------------------------------------------------------
    # Key inequality: 4*ā10*ā01 >= ā11²*Pmax*Qmax
    # ---------------------------------------------------------------
    pr("Computing AM-GM bound...")
    lhs = sp.expand(4*a10_bar*a01_bar)
    rhs = sp.expand(a11_bar**2 * Pmax * Qmax)
    delta_amgm = sp.expand(lhs - rhs)

    pr("Trying to factor delta_amgm...")
    # This might be very large, try cancellation
    delta_amgm_simplified = sp.cancel(delta_amgm)
    n_terms = len(sp.Add.make_args(delta_amgm))
    pr(f"delta_amgm has {n_terms} terms")

    # Numeric check
    pr("\n=== Numeric AM-GM check ===")
    lhs_fn = sp.lambdify((r, x, y), lhs, "numpy")
    rhs_fn = sp.lambdify((r, x, y), rhs, "numpy")
    L_fn = sp.lambdify((r, x, y), L, "numpy")

    rng = np.random.default_rng(42)
    n_test = 300000
    n_amgm_fail = 0
    n_amgm_fail_Lpos = 0
    min_ratio = float("inf")

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        lv = float(lhs_fn(rv, xv, yv))
        rv2 = float(rhs_fn(rv, xv, yv))
        Lv = float(L_fn(rv, xv, yv))

        if rv2 > 0:
            ratio = lv / rv2
            if ratio < min_ratio:
                min_ratio = ratio
            if lv < rv2 - 1e-6 * abs(rv2):
                n_amgm_fail += 1
                if Lv > 0:
                    n_amgm_fail_Lpos += 1

    pr(f"4ā10ā01 < ā11²PmaxQmax: {n_amgm_fail}/{n_test}")
    pr(f"  ... when L > 0: {n_amgm_fail_Lpos}")
    pr(f"min ratio 4ā10ā01/(ā11²PmaxQmax): {min_ratio:.6f}")

    # ---------------------------------------------------------------
    # Tighter check: at the worst point u*,v* (the AM-GM critical point)
    # Ā >= 0 iff at the worst (u,v): the sum of all terms >= 0
    # The worst point is where the ratio of positive/negative is tightest.
    # ---------------------------------------------------------------
    pr("\n=== Full Ā check at AM-GM critical point ===")
    # At AM-GM critical: u·F = v·G (balanced allocation)
    # With F = ā10 + ā20u + ā21uv, G = ā01 + ā02v + ā12uv
    # At u=v=0: F=ā10, G=ā01. The critical u*,v* are small.
    # For the loosest bound (u=Pmax, v=Qmax):
    # F = ā10 + ā20·Pmax + ā21·Pmax·Qmax, G = ā01 + ā02·Qmax + ā12·Pmax·Qmax
    # 4FG >= ā11²·Pmax·Qmax is a tighter condition

    F_corner = sp.expand(a10_bar + a20_bar*Pmax + 1296*r*L*Pmax*Qmax)
    G_corner = sp.expand(a01_bar + a02_bar*Qmax + 1296*r*L*Pmax*Qmax)
    tight_lhs = sp.expand(4*F_corner*G_corner)
    tight_delta = sp.expand(tight_lhs - rhs)
    tight_delta_terms = len(sp.Add.make_args(tight_delta))
    pr(f"Tight delta has {tight_delta_terms} terms")

    tight_lhs_fn = sp.lambdify((r, x, y), tight_lhs, "numpy")
    n_tight_fail = 0
    n_tight_fail_Lpos = 0
    min_tight = float("inf")

    rng2 = np.random.default_rng(123)
    for _ in range(n_test):
        rv = float(np.exp(rng2.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng2.uniform(1e-8, 1 - 1e-8))
        yv = float(rng2.uniform(1e-8, 1 - 1e-8))

        tlv = float(tight_lhs_fn(rv, xv, yv))
        rv2 = float(rhs_fn(rv, xv, yv))
        Lv = float(L_fn(rv, xv, yv))

        if rv2 > 0:
            ratio = tlv / rv2
            if ratio < min_tight:
                min_tight = ratio
            if tlv < rv2 - 1e-6 * abs(rv2):
                n_tight_fail += 1
                if Lv > 0:
                    n_tight_fail_Lpos += 1

    pr(f"4F_corner·G_corner < ā11²PmaxQmax: {n_tight_fail}/{n_test}")
    pr(f"  ... when L > 0: {n_tight_fail_Lpos}")
    pr(f"min ratio: {min_tight:.6f}")

    # ---------------------------------------------------------------
    # Alternative: direct factorization of delta_amgm
    # ---------------------------------------------------------------
    pr("\n=== Attempting factorization of delta ===")
    # If delta_amgm is not too large, try to factor
    if n_terms < 200:
        delta_f = sp.factor(delta_amgm)
        pr("delta factored:", delta_f)
    else:
        pr(f"delta has {n_terms} terms, too large for direct factoring")
        # Try dividing by known factors
        pos_core = sp.expand(3*r**2*y + r**2 + 4*r + 3*x + 1)
        F_lin = sp.expand(3*r*y - 7*r + 3*x - 7)

        # Check if pos_core² divides delta
        d_by_pc2 = sp.cancel(delta_amgm / pos_core**2)
        try:
            d_test = sp.Poly(sp.expand(d_by_pc2), r, x, y)
            pr("delta / pos_core² is polynomial")
            # Check if F_lin² divides
            d_by_fl2 = sp.cancel(d_by_pc2 / F_lin**2)
            try:
                d_test2 = sp.Poly(sp.expand(d_by_fl2), r, x, y)
                pr("delta / (pos_core²·F_lin²) is polynomial")
                n_terms2 = len(sp.Add.make_args(sp.expand(d_by_fl2)))
                pr(f"  with {n_terms2} terms")
                if n_terms2 < 300:
                    d_f2 = sp.factor(sp.expand(d_by_fl2))
                    pr("  factored:", d_f2)
            except:
                pr("delta / (pos_core²·F_lin²) is NOT polynomial")
        except:
            pr("delta / pos_core² is NOT polynomial")

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

    out = {
        "amgm_weak": {
            "fail_count": n_amgm_fail,
            "fail_Lpos": n_amgm_fail_Lpos,
            "min_ratio": min_ratio,
            "description": "4*a10_bar*a01_bar vs a11_bar^2*Pmax*Qmax",
        },
        "amgm_tight": {
            "fail_count": n_tight_fail,
            "fail_Lpos": n_tight_fail_Lpos,
            "min_ratio": min_tight,
            "description": "4*F_corner*G_corner vs a11_bar^2*Pmax*Qmax",
        },
    }
    Path("/home/joe/code/futon6/data/first-proof/p4-path2-A-amgm.json").write_text(json.dumps(out, indent=2))
    pr("Wrote results")


if __name__ == "__main__":
    main()
