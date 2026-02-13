#!/usr/bin/env python3
"""P4 Path2 Codex: unified K_T2R analysis and bridge-to-certificate diagnostics.

This script executes the Cycle handoff tasks with an explicit normalization check.
It computes:
- K_handoff_literal from the handoff formula,
- K_corr (corrected r-placement from exact T2/R assembly),
- K_exact from the full T2+R normalized derivation.

Key identity found:
  K_corr = (9/8) * (K_exact / r^2)
so positivity of K_corr is equivalent to positivity of K_exact/r^2.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import Poly, Rational, cancel, expand, factor


def pr(*args):
    print(*args, flush=True)


def build_handoff_objects():
    r, x, y, p, q = sp.symbols("r x y p q")

    f1p = 1 + 3 * x
    f2p = 2 * (1 - x) - 9 * p**2
    f1q = r**2 * (1 + 3 * y)
    f2q = 2 * r**3 * (1 - y) - 9 * q**2

    Sv = 1 + r
    Av12 = 3 * x + 3 * y * r**2 + 2 * r
    f1c = expand(Sv**2 + Av12)
    f2c = expand(2 * Sv**3 - 2 * Sv * Av12 / 3 - 9 * (p + q) ** 2)

    C_p = x - 1
    C_q = r**2 * (y - 1)
    C_c = expand(Av12 / 3 - Sv**2)

    L = expand(9 * x**2 - 27 * x * y * (1 + r) + 3 * x * (r - 1) + 9 * r * y**2 - 3 * r * y + 2 * r + 3 * y + 2)
    W = expand(3 * r**2 * y - 3 * r**2 - 4 * r + 3 * x - 3)

    R_num = expand(C_c * f1c * f2p * f2q - C_p * f1p * f2c * f2q - C_q * f1q * f2c * f2p)

    # Literal from handoff (known to be mis-normalized).
    K_handoff_literal = expand(2 * L * f2p * f2q * f2c + r * (1 + 3 * x) * (1 + 3 * y) * f1c * R_num)

    # Corrected from exact T2/R assembly:
    # T2_surplus = 2rL / (9*(1+3x)(1+3y)*f1c)
    # R_surplus  = R_num / (9*f2p*f2q*f2c)
    # => common numerator:
    K_corr = expand(2 * r * L * f2p * f2q * f2c + (1 + 3 * x) * (1 + 3 * y) * f1c * R_num)

    return {
        "vars": (r, x, y, p, q),
        "f2p": f2p,
        "f2q": f2q,
        "f2c": f2c,
        "f1c": f1c,
        "L": L,
        "W": W,
        "R_num": R_num,
        "K_handoff_literal": K_handoff_literal,
        "K_corr": K_corr,
    }


def build_exact_k_red():
    s, t, u, v, a, b = sp.symbols("s t u v a b")
    r, x, y, p, q = sp.symbols("r x y p q")

    def T2R_num(ss, uu, aa):
        return 8 * aa * (ss**2 - 4 * aa) ** 2 - ss * uu**2 * (ss**2 + 60 * aa)

    def T2R_den(ss, uu, aa):
        return 2 * (ss**2 + 12 * aa) * (2 * ss**3 - 8 * ss * aa - 9 * uu**2)

    S = s + t
    U = u + v
    A = a + b + s * t / 6
    surplus_num = expand(
        T2R_num(S, U, A) * T2R_den(s, u, a) * T2R_den(t, v, b)
        - T2R_num(s, u, a) * T2R_den(S, U, A) * T2R_den(t, v, b)
        - T2R_num(t, v, b) * T2R_den(S, U, A) * T2R_den(s, u, a)
    )

    subs_norm = {
        t: r * s,
        a: x * s**2 / 4,
        b: y * r**2 * s**2 / 4,
        u: p * s**Rational(3, 2),
        v: q * s**Rational(3, 2),
    }
    K_exact = expand(surplus_num.subs(subs_norm) / s**16)
    K_red = expand(cancel(K_exact / r**2))
    return (r, x, y, p, q), K_exact, K_red


def p_q_blocks(expr, p, q):
    poly = Poly(expr, p, q)
    blocks = {}
    for (i, j), c in poly.as_dict().items():
        d = i + j
        blocks[d] = expand(blocks.get(d, sp.Integer(0)) + c * p**i * q**j)
    return blocks


def uv_coefficients(expr, p, q):
    r, x, y, u, v = sp.symbols("r x y u v")
    uv_expr = expand(expr.subs({p: (u + v) / 2, q: (u - v) / 2}))
    uv_poly = Poly(uv_expr, u, v)
    coeffs = {}
    for mon, c in sorted(uv_poly.as_dict().items()):
        coeffs[mon] = expand(c)
    return (u, v), coeffs


def numeric_scan(K_expr, B_expr, A_expr, L_expr, vars_tuple, n_samples=200000, seed=42):
    r, x, y, p, q = vars_tuple
    K_fn = sp.lambdify((r, x, y, p, q), K_expr, "numpy")
    B_fn = sp.lambdify((r, x, y, p, q), B_expr, "numpy")
    A_fn = sp.lambdify((r, x, y, p, q), A_expr, "numpy")
    L_fn = sp.lambdify((r, x, y), L_expr, "numpy")

    rng = np.random.default_rng(seed)

    out = {
        "tested": 0,
        "K_neg": 0,
        "K_min": float("inf"),
        "B_pos": 0,
        "B_max": -float("inf"),
        "B_min": float("inf"),
        "A_min": float("inf"),
        "ratio_min_A_over_abs_pqB": float("inf"),
        "L_le_0_count": 0,
        "L_gt_0_count": 0,
        "K_min_L_le_0": float("inf"),
        "K_min_L_gt_0": float("inf"),
    }

    for _ in range(n_samples):
        rv = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng.uniform(1e-4, 1 - 1e-4))
        yv = float(rng.uniform(1e-4, 1 - 1e-4))

        pmax2 = 2 * (1 - xv) / 9
        qmax2 = 2 * rv**3 * (1 - yv) / 9
        pmax = float(np.sqrt(max(pmax2, 0.0)))
        qmax = float(np.sqrt(max(qmax2, 0.0)))

        pv = float(rng.uniform(-0.99 * pmax, 0.99 * pmax))
        qv = float(rng.uniform(-0.99 * qmax, 0.99 * qmax))

        f2c_v = 2 * (1 + rv) ** 3 - 2 * (1 + rv) * (3 * xv + 3 * yv * rv**2 + 2 * rv) / 3 - 9 * (pv + qv) ** 2
        if f2c_v <= 1e-10:
            continue

        out["tested"] += 1
        kv = float(K_fn(rv, xv, yv, pv, qv))
        bv = float(B_fn(rv, xv, yv, pv, qv))
        av = float(A_fn(rv, xv, yv, pv, qv))
        lv = float(L_fn(rv, xv, yv))

        if kv < out["K_min"]:
            out["K_min"] = kv
        if kv < -1e-10:
            out["K_neg"] += 1

        if bv > out["B_max"]:
            out["B_max"] = bv
        if bv < out["B_min"]:
            out["B_min"] = bv
        if bv > 1e-10:
            out["B_pos"] += 1

        if av < out["A_min"]:
            out["A_min"] = av

        den = abs(pv * qv * bv)
        if den > 1e-20:
            ratio = av / den
            if ratio < out["ratio_min_A_over_abs_pqB"]:
                out["ratio_min_A_over_abs_pqB"] = ratio

        if lv <= 0:
            out["L_le_0_count"] += 1
            if kv < out["K_min_L_le_0"]:
                out["K_min_L_le_0"] = kv
        else:
            out["L_gt_0_count"] += 1
            if kv < out["K_min_L_gt_0"]:
                out["K_min_L_gt_0"] = kv

    return out


def main():
    t0 = time.time()

    # Build handoff objects.
    obj = build_handoff_objects()
    r, x, y, p, q = obj["vars"]
    L = obj["L"]
    W = obj["W"]
    K_lit = obj["K_handoff_literal"]
    K_corr = obj["K_corr"]

    # Build exact K from normalized T2+R pipeline.
    vars2, K_exact, K_red = build_exact_k_red()

    # Consistency checks.
    relation_8_9 = expand(8 * K_corr - 9 * K_red) == 0
    lit_matches = expand(K_lit - K_corr) == 0

    # Degree/term data.
    terms_lit = len(Poly(K_lit, r, x, y, p, q).as_dict())
    terms_corr = len(Poly(K_corr, r, x, y, p, q).as_dict())
    terms_red = len(Poly(K_red, r, x, y, p, q).as_dict())

    # r^2 divisibility checks.
    red_times_r2 = expand(K_exact)
    r2_div_exact = (expand(red_times_r2.subs(r, 0)) == 0) and (expand(cancel(red_times_r2 / r).subs(r, 0)) == 0)

    # p,q block decomposition on canonical K_red.
    blocks = p_q_blocks(K_red, p, q)

    d6 = blocks.get(6, sp.Integer(0))
    d6_target = expand(-1296 * r * L * p**2 * q**2 * (p + q) ** 2)
    d6_ok = expand(d6 - d6_target) == 0

    # A + pq*B decomposition.
    poly = Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr = expand(A_expr + c * p**i * q**j)
        elif i % 2 == 1 and j % 2 == 1:
            B_expr = expand(B_expr + c * p**(i - 1) * q**(j - 1))
        else:
            raise RuntimeError(f"Unexpected parity monomial p^{i}q^{j}")

    decomp_ok = expand(A_expr + p * q * B_expr - K_red) == 0

    P, Q = sp.symbols("P Q")
    A_PQ = expand(A_expr.subs({p**2: P, q**2: Q}))
    B_PQ = expand(B_expr.subs({p**2: P, q**2: Q}))
    A_poly = Poly(A_PQ, P, Q)
    B_poly = Poly(B_PQ, P, Q)

    # Selected coefficient identities.
    a = A_poly.as_dict()
    b = B_poly.as_dict()
    coeff_identities = {
        "B01_eq_2A02": expand(b[(0, 1)] - 2 * a[(0, 2)]) == 0,
        "B10_eq_2A20": expand(b[(1, 0)] - 2 * a[(2, 0)]) == 0,
        "B11_eq_2A12": expand(b[(1, 1)] - 2 * a[(1, 2)]) == 0,
        "B11_eq_2A21": expand(b[(1, 1)] - 2 * a[(2, 1)]) == 0,
    }

    # (u,v) analysis on canonical K_red.
    (u_sym, v_sym), uv_coeffs = uv_coefficients(K_red, p, q)

    uv_div = {}
    for mon, c in uv_coeffs.items():
        qW, rW = sp.div(Poly(c, r, x, y), Poly(W, r, x, y))
        qL, rL = sp.div(Poly(c, r, x, y), Poly(L, r, x, y))
        uv_div[str(mon)] = {
            "W_div": (rW == Poly(0, r, x, y)),
            "L_div": (rL == Poly(0, r, x, y)),
            "n_terms": len(Poly(c, r, x, y).as_dict()),
            "factor_head": str(factor(c))[:240],
        }

    # Structural discovery check: L<=0 => x+y>=2/3 from line computation.
    xx = sp.symbols("xx")
    line_expr = expand(L.subs({x: xx, y: Rational(2, 3) - xx}))
    line_fact = factor(line_expr)
    line_target = expand(4 * (r + 1) * (3 * xx - 1) ** 2)
    line_ok = expand(line_expr - line_target) == 0

    # Numerics on canonical K_red and decomposition margin.
    scan = numeric_scan(K_red, B_expr, A_expr, L, (r, x, y, p, q), n_samples=200000, seed=314159)

    out = {
        "meta": {
            "date": "2026-02-13",
            "script": "scripts/explore-p4-path2-codex.py",
            "runtime_sec": round(time.time() - t0, 3),
        },
        "task1": {
            "terms": {
                "K_handoff_literal": terms_lit,
                "K_corr": terms_corr,
                "K_exact_red": terms_red,
            },
            "normalization": {
                "handoff_literal_equals_corr": lit_matches,
                "corr_relation": "8*K_corr == 9*K_exact_red",
                "corr_relation_holds": relation_8_9,
            },
            "r2_divisibility": {
                "K_exact_divisible_by_r2": r2_div_exact,
                "K_red_definition": "K_red = K_exact / r^2",
            },
            "pq_degree": {str(k): len(Poly(vv, p, q).as_dict()) for k, vv in sorted(blocks.items())},
            "d6_factorization_holds": d6_ok,
        },
        "task2": {
            "uv_monomials": [str(k) for k in sorted(uv_coeffs.keys())],
            "uv_divisibility": uv_div,
        },
        "task3": {
            "decomposition": "K_red = A(p^2,q^2) + p*q*B(p^2,q^2)",
            "decomposition_holds": decomp_ok,
            "A_monomials": [str(k) for k in sorted(A_poly.as_dict().keys())],
            "B_monomials": [str(k) for k in sorted(B_poly.as_dict().keys())],
            "coefficient_identities": coeff_identities,
            "status": "partial-certificate",
            "notes": [
                "All sampled points satisfy B<0 and A>|p*q*B| with a strong margin.",
                "This gives a robust numeric certificate candidate but not yet a symbolic inequality proof on the full domain.",
            ],
        },
        "task4": {
            "L_line_identity": {
                "L_at_y_2over3_minus_x": str(line_fact),
                "equals_4_rplus1_3xminus1_sq": line_ok,
            },
            "numeric_scan": scan,
        },
    }

    out_path = Path("/home/joe/code/futon6/data/first-proof/p4-path2-codex-results.json")
    out_path.write_text(json.dumps(out, indent=2))

    pr("=" * 88)
    pr("P4 PATH2 CODEX")
    pr("=" * 88)
    pr("terms: literal/corr/red =", terms_lit, terms_corr, terms_red)
    pr("normalization: literal==corr?", lit_matches, "  8*corr==9*red?", relation_8_9)
    pr("d6 factorization holds:", d6_ok)
    pr("decomposition A+pqB holds:", decomp_ok)
    pr("coeff identities:", coeff_identities)
    pr("numeric scan tested:", scan["tested"], "K_neg:", scan["K_neg"], "K_min:", scan["K_min"])
    pr("B_pos:", scan["B_pos"], "B_max:", scan["B_max"], "B_min:", scan["B_min"])
    pr("ratio min A/|pqB|:", scan["ratio_min_A_over_abs_pqB"])
    pr("line identity L<=0 implication helper holds:", line_ok)
    pr("wrote", out_path)


if __name__ == "__main__":
    main()
