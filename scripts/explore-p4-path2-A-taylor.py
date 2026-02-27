#!/usr/bin/env python3
"""P4 Path2: Taylor expansion of A around (Pmax, Qmax) corner.

Since A(Pmax,Qmax) = 0 exactly, expand A(Pmax-u, Qmax-v) in powers
of u,v >= 0. If all coefficients are >= 0, then A >= 0 on the box.

A(P,Q) is degree 3 in (P,Q), so the expansion has 8 terms (minus constant=0):
  ā₁₀·u + ā₀₁·v + ā₂₀·u² + ā₁₁·uv + ā₀₂·v² + ā₂₁·u²v + ā₁₂·uv² + ā₃₀·u³ + ā₀₃·v³

Actually A has no P³ or Q³ pure terms, so ā₃₀ = ā₀₃ = 0.
"""
from __future__ import annotations

import json
import time
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
    u_var, v_var = sp.symbols("u v", nonnegative=True)

    pr("Building K_red & decomposing A...")
    K_red = build_exact_k_red()
    c = decompose_coeffs(K_red)

    a00=c["a00"]; a01=c["a01"]; a02=c["a02"]; a10=c["a10"]
    a11=c["a11"]; a12=c["a12"]; a20=c["a20"]

    P_sym, Q_sym = sp.symbols("P Q")
    A_poly = sp.expand(a00 + a10*P_sym + a01*Q_sym + a20*P_sym**2 + a11*P_sym*Q_sym
                       + a02*Q_sym**2 + a12*P_sym**2*Q_sym + a12*P_sym*Q_sym**2)

    Pmax = sp.Rational(2,9)*(1-x)
    Qmax = sp.Rational(2,9)*r**3*(1-y)

    # Expand A(Pmax-u, Qmax-v)
    pr("Expanding A around (Pmax, Qmax)...")
    A_expanded = sp.expand(A_poly.subs({P_sym: Pmax - u_var, Q_sym: Qmax - v_var}))

    # Collect as polynomial in u, v
    A_uv = sp.Poly(A_expanded, u_var, v_var)
    coeffs_dict = A_uv.as_dict()

    pr("\nExpansion coefficients:")
    labels = {
        (0,0): "ā00 (constant, should be 0)",
        (1,0): "ā10 (u coeff)",
        (0,1): "ā01 (v coeff)",
        (2,0): "ā20 (u² coeff)",
        (1,1): "ā11 (uv coeff)",
        (0,2): "ā02 (v² coeff)",
        (2,1): "ā21 (u²v coeff)",
        (1,2): "ā12 (uv² coeff)",
        (3,0): "ā30 (u³ coeff)",
        (0,3): "ā03 (v³ coeff)",
    }

    results = {}
    for key in sorted(coeffs_dict.keys()):
        coeff = sp.expand(coeffs_dict[key])
        label = labels.get(key, f"ā{key}")
        factored = sp.factor(coeff)
        pr(f"\n{label}:")
        pr(f"  factored: {factored}")
        results[f"a_bar_{key[0]}{key[1]}"] = str(factored)

    # ---------------------------------------------------------------
    # Numeric check: are all coefficients >= 0 on feasible?
    # ---------------------------------------------------------------
    pr("\n=== Numeric check of expansion coefficients ===")
    coeff_fns = {}
    for key, coeff in coeffs_dict.items():
        coeff_fns[key] = sp.lambdify((r, x, y), coeff, "numpy")

    rng = np.random.default_rng(777)
    n_test = 300000
    sign_counts = {key: {"pos": 0, "neg": 0, "zero": 0} for key in coeffs_dict}

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        for key, fn in coeff_fns.items():
            val = float(fn(rv, xv, yv))
            if val > 1e-15:
                sign_counts[key]["pos"] += 1
            elif val < -1e-15:
                sign_counts[key]["neg"] += 1
            else:
                sign_counts[key]["zero"] += 1

    pr(f"\nResults ({n_test} samples):")
    for key in sorted(sign_counts.keys()):
        label = labels.get(key, f"ā{key}")
        sc = sign_counts[key]
        pr(f"  {label}: pos={sc['pos']}, neg={sc['neg']}, zero={sc['zero']}")

    # ---------------------------------------------------------------
    # Check ā10 and ā01 factorizations in detail
    # ---------------------------------------------------------------
    pr("\n=== ā10 detailed analysis ===")
    a10_bar = coeffs_dict.get((1,0), sp.Integer(0))
    a01_bar = coeffs_dict.get((0,1), sp.Integer(0))

    # ā10 should be -∂A/∂P at (Pmax,Qmax)
    # = -(a10 + 2a20*Pmax + a11*Qmax + 2a12*Pmax*Qmax + a12*Qmax²)
    a10_bar_check = sp.expand(-(a10 + 2*a20*Pmax + a11*Qmax + 2*a12*Pmax*Qmax + a12*Qmax**2))
    pr("ā10 identity check:", sp.expand(a10_bar - a10_bar_check) == 0)

    a01_bar_check = sp.expand(-(a01 + a11*Pmax + 2*a02*Qmax + a12*Pmax**2 + 2*a12*Pmax*Qmax))
    pr("ā01 identity check:", sp.expand(a01_bar - a01_bar_check) == 0)

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

    out = {
        "meta": {"date": "2026-02-13", "runtime_sec": round(elapsed,3)},
        "expansion_coefficients": results,
        "sign_counts": {
            f"a_bar_{k[0]}{k[1]}": v for k, v in sign_counts.items()
        }
    }
    out_path = Path("/home/joe/code/futon6/data/first-proof/p4-path2-A-taylor.json")
    out_path.write_text(json.dumps(out, indent=2))
    pr(f"Wrote {out_path}")


from pathlib import Path
if __name__ == "__main__":
    main()
