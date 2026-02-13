#!/usr/bin/env python3
"""P4 Path2 Cycle 3 (Codex): same-sign gap analysis and candidate closure route.

This script targets Claim 3 from p4-path2-codex-handoff-3.md:
  K_red(p,q) >= 0 on feasible domain for p,q >= 0.

Outputs:
- exact symbolic identities for the A + sqrt(PQ) B reduction,
- exact edge endpoint factorizations and simple sign bounds for endpoint factors,
- large numerical stress checks for same-sign positivity,
- candidate reduction check via A >= 0, B <= 0, and A^2 - P Q B^2 >= 0.
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
        t: r * s,
        a: x * s**2 / 4,
        b: y * r**2 * s**2 / 4,
        u: p * s ** sp.Rational(3, 2),
        v: q * s ** sp.Rational(3, 2),
    }
    K_exact = sp.expand(surplus_num.subs(subs_norm) / s**16)
    K_red = sp.expand(sp.cancel(K_exact / r**2))
    return K_red


def decompose_coeffs(K_red: sp.Expr):
    r, x, y, p, q = sp.symbols("r x y p q")
    P, Q = sp.symbols("P Q")

    poly = sp.Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr += c * p**i * q**j
        elif i % 2 == 1 and j % 2 == 1:
            B_expr += c * p ** (i - 1) * q ** (j - 1)

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

    return {
        "A_expr": sp.expand(A_expr),
        "B_expr": sp.expand(B_expr),
        "a00": sp.expand(a00),
        "a01": sp.expand(a01),
        "a02": sp.expand(a02),
        "a10": sp.expand(a10),
        "a11": sp.expand(a11),
        "a12": sp.expand(a12),
        "a20": sp.expand(a20),
        "b00": sp.expand(b00),
    }


def main():
    t0 = time.time()
    r, x, y, p, q = sp.symbols("r x y p q")
    P, Q = sp.symbols("P Q")

    pr("Building K_red...")
    K_red = build_exact_k_red()
    pr("Decomposing coefficients...")
    c = decompose_coeffs(K_red)

    a00 = c["a00"]
    a01 = c["a01"]
    a02 = c["a02"]
    a10 = c["a10"]
    a11 = c["a11"]
    a12 = c["a12"]
    a20 = c["a20"]
    b00 = c["b00"]

    # -----------------------------------------------------------------
    # Same-sign reduction: K(P,Q) = A(P,Q) + sqrt(PQ) B(P,Q)
    # -----------------------------------------------------------------
    A = sp.expand(a00 + a10 * P + a01 * Q + a20 * P**2 + a11 * P * Q + a02 * Q**2 + a12 * (P**2 * Q + P * Q**2))
    B = sp.expand(b00 + 2 * a20 * P + 2 * a02 * Q + 2 * a12 * P * Q)

    K_recon = sp.expand(A.subs({P: p**2, Q: q**2}) + p * q * B.subs({P: p**2, Q: q**2}))
    same_sign_identity = sp.expand(K_recon - K_red) == 0

    S = sp.expand(A**2 - P * Q * B**2)

    # -----------------------------------------------------------------
    # Edge 3a exact endpoint formulas and sign lemmas.
    # -----------------------------------------------------------------
    Pmax = sp.expand(2 * (1 - x) / 9)
    Qmax = sp.expand(2 * r**3 * (1 - y) / 9)

    edge_q = sp.expand(a00 + a01 * Q + a02 * Q**2)
    edge_p = sp.expand(a00 + a10 * P + a20 * P**2)

    edge_q_Qmax = sp.expand(edge_q.subs(Q, Qmax))
    edge_p_Pmax = sp.expand(edge_p.subs(P, Pmax))

    edge_q_factor = sp.factor(edge_q_Qmax)
    edge_p_factor = sp.factor(edge_p_Pmax)

    # Remaining linear factors in endpoint factorizations:
    # Fq = 3 r^2 y - 7 r^2 + 3 r x - 7 r + 3 x - 3
    # Fp = 3 r^2 y - 3 r^2 + 3 r y - 7 r + 3 x - 7
    Fq = sp.expand(3 * r**2 * y - 7 * r**2 + 3 * r * x - 7 * r + 3 * x - 3)
    Fp = sp.expand(3 * r**2 * y - 3 * r**2 + 3 * r * y - 7 * r + 3 * x - 7)

    # Upper bounds showing strict negativity on feasible interior x,y in (0,1), r>0.
    Fq_upper = sp.expand(-4 * r**2 - 4 * r)
    Fp_upper = sp.expand(-4 * r - 4)

    Fq_upper_check = sp.expand(Fq - Fq_upper - (3 * r**2 * (y - 1) + 3 * (r + 1) * (x - 1))) == 0
    Fp_upper_check = sp.expand(Fp - Fp_upper - (3 * r**2 * (y - 1) + 3 * r * (y - 1) + 3 * (x - 1))) == 0

    # -----------------------------------------------------------------
    # Numerical stress.
    # -----------------------------------------------------------------
    rng = np.random.default_rng(20260213)

    K_fn = sp.lambdify((r, x, y, p, q), K_red, "numpy")
    A_fn = sp.lambdify((r, x, y, P, Q), A, "numpy")
    B_fn = sp.lambdify((r, x, y, P, Q), B, "numpy")

    a00_fn = sp.lambdify((r, x, y), a00, "numpy")
    a01_fn = sp.lambdify((r, x, y), a01, "numpy")
    a02_fn = sp.lambdify((r, x, y), a02, "numpy")
    a10_fn = sp.lambdify((r, x, y), a10, "numpy")
    a20_fn = sp.lambdify((r, x, y), a20, "numpy")

    n_same = 120000
    # Stress ranges include very small and very large r.
    rv = np.exp(rng.uniform(np.log(1e-8), np.log(1e8), n_same))
    xv = rng.uniform(1e-12, 1 - 1e-12, n_same)
    yv = rng.uniform(1e-12, 1 - 1e-12, n_same)
    Pmax_v = 2 * (1 - xv) / 9
    Qmax_v = 2 * rv**3 * (1 - yv) / 9
    pv = rng.uniform(0, 1.0, n_same) * np.sqrt(np.maximum(Pmax_v, 0.0))
    qv = rng.uniform(0, 1.0, n_same) * np.sqrt(np.maximum(Qmax_v, 0.0))
    Pv = pv * pv
    Qv = qv * qv

    Kvals = np.asarray(K_fn(rv, xv, yv, pv, qv), dtype=float)
    Avals = np.asarray(A_fn(rv, xv, yv, Pv, Qv), dtype=float)
    Bvals = np.asarray(B_fn(rv, xv, yv, Pv, Qv), dtype=float)
    # Numerical stability: evaluate S from K and its sign-flipped companion
    # instead of directly expanding A^2 - P Q B^2 at extreme scales.
    Svals = Kvals * (Avals - np.sqrt(Pv * Qv) * Bvals)

    neg_K_same = int(np.sum(Kvals < -1e-9))
    neg_A = int(np.sum(Avals < -1e-9))
    pos_B = int(np.sum(Bvals > 1e-9))
    S_scale = np.maximum(1.0, np.abs(Avals * Avals) + np.abs(Pv * Qv * Bvals * Bvals))
    S_scaled = Svals / S_scale
    neg_S = int(np.sum(S_scaled < -1e-10))

    iK = int(np.argmin(Kvals))
    iS = int(np.argmin(Svals))
    min_K_same = float(Kvals[iK])
    min_A = float(np.min(Avals))
    max_B = float(np.max(Bvals))
    min_S = float(Svals[iS])
    min_S_scaled = float(S_scaled[iS])
    arg_min_K = (float(rv[iK]), float(xv[iK]), float(yv[iK]), float(pv[iK]), float(qv[iK]))
    arg_min_S = (float(rv[iS]), float(xv[iS]), float(yv[iS]), float(Pv[iS]), float(Qv[iS]))

    # Edge minima checks on [0,Pmax], [0,Qmax] for convex quadratics.
    n_edges = 120000
    r2 = np.exp(rng.uniform(np.log(1e-8), np.log(1e8), n_edges))
    x2 = rng.uniform(1e-12, 1 - 1e-12, n_edges)
    y2 = rng.uniform(1e-12, 1 - 1e-12, n_edges)
    Pmax2 = 2 * (1 - x2) / 9
    Qmax2 = 2 * r2**3 * (1 - y2) / 9

    a00v = np.asarray(a00_fn(r2, x2, y2), dtype=float)
    a01v = np.asarray(a01_fn(r2, x2, y2), dtype=float)
    a02v = np.asarray(a02_fn(r2, x2, y2), dtype=float)
    a10v = np.asarray(a10_fn(r2, x2, y2), dtype=float)
    a20v = np.asarray(a20_fn(r2, x2, y2), dtype=float)

    qstar = np.where(a02v > 0, -a01v / (2 * a02v), 0.0)
    pstar = np.where(a20v > 0, -a10v / (2 * a20v), 0.0)

    fq0 = a00v
    fq1 = a00v + a01v * Qmax2 + a02v * Qmax2 * Qmax2
    fqv = np.where((qstar > 0) & (qstar < Qmax2), a00v + a01v * qstar + a02v * qstar * qstar, np.inf)
    fq = np.minimum(np.minimum(fq0, fq1), fqv)

    fp0 = a00v
    fp1 = a00v + a10v * Pmax2 + a20v * Pmax2 * Pmax2
    fpv = np.where((pstar > 0) & (pstar < Pmax2), a00v + a10v * pstar + a20v * pstar * pstar, np.inf)
    fp = np.minimum(np.minimum(fp0, fp1), fpv)

    neg_edge_q = int(np.sum(fq < -1e-9))
    neg_edge_p = int(np.sum(fp < -1e-9))
    min_edge_q = float(np.min(fq))
    min_edge_p = float(np.min(fp))

    out = {
        "meta": {
            "date": "2026-02-13",
            "script": "scripts/explore-p4-path2-codex-cycle3.py",
            "runtime_sec": round(time.time() - t0, 3),
        },
        "identities": {
            "K_equals_A_plus_pqB": same_sign_identity,
            "edge_q_Qmax_factor": str(edge_q_factor),
            "edge_p_Pmax_factor": str(edge_p_factor),
            "Fq_upper_identity": Fq_upper_check,
            "Fp_upper_identity": Fp_upper_check,
            "Fq_formula": str(Fq),
            "Fp_formula": str(Fp),
            "Fq_upper": str(Fq_upper),
            "Fp_upper": str(Fp_upper),
        },
        "same_sign_stress": {
            "tested": n_same,
            "K_negative": neg_K_same,
            "A_negative": neg_A,
            "B_positive": pos_B,
            "S_negative": neg_S,
            "min_K": min_K_same,
            "min_A": min_A,
            "max_B": max_B,
            "min_S": min_S,
            "min_S_scaled": min_S_scaled,
            "arg_min_K": arg_min_K,
            "arg_min_S": arg_min_S,
        },
        "edge_stress": {
            "tested": n_edges,
            "q_edge_negative": neg_edge_q,
            "p_edge_negative": neg_edge_p,
            "min_q_edge": min_edge_q,
            "min_p_edge": min_edge_p,
        },
        "candidate_route": {
            "statement": "For same-sign p,q, prove A>=0 and A^2-PQ*B^2>=0 (with B<=0) to conclude K=A+sqrt(PQ)B>=0.",
            "status": "numerically supported, not yet symbolic-closed",
        },
    }

    out_path = Path("/home/joe/code/futon6/data/first-proof/p4-path2-codex-cycle3-results.json")
    out_path.write_text(json.dumps(out, indent=2))

    pr("=" * 88)
    pr("P4 PATH2 CYCLE3 CODEX")
    pr("=" * 88)
    pr("identity K=A+pqB:", same_sign_identity)
    pr("Fq/Fp upper-bound identity checks:", Fq_upper_check, Fp_upper_check)
    pr("same-sign tested:", n_same)
    pr("neg K / neg A / pos B / neg S:", neg_K_same, neg_A, pos_B, neg_S)
    pr("min K / min A / max B / min S (raw/scaled):", min_K_same, min_A, max_B, min_S, min_S_scaled)
    pr("edge tested:", n_edges)
    pr("edge negatives q/p:", neg_edge_q, neg_edge_p)
    pr("edge minima q/p:", min_edge_q, min_edge_p)
    pr("wrote", out_path)


if __name__ == "__main__":
    main()
