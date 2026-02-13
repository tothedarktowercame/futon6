#!/usr/bin/env python3
"""Task B: Smale alpha bounds for the 4 Case 3c critical points.

This script certifies *known* Case 3c roots from PHC output using a conservative
alpha-theory bound:

  beta(x) = ||J(x)^(-1) F(x)||_2
  gamma(x) <= mu(F, x) * D^(3/2) / (2 ||x||_1)
  mu(F, x) = max(1, ||F||_BW * ||J(x)^(-1) Delta(x)||_2)

with ||x||_1 = sqrt(1 + ||x||_2^2), D = max degree, and
Delta(x) = diag(sqrt(d_i) * ||x||_1^(d_i - 1)).

Implementation detail:
  we upper-bound ||.||_2 by Frobenius norm for J^(-1) Delta, so alpha values are
  conservative (possibly larger than necessary).

Important caveat:
  This certifies the listed roots are genuine simple roots with quadratic Newton
  basins. It does not by itself prove no additional real in-domain roots exist.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import mpmath as mp
import sympy as sp


def build_negN_and_grad():
    a3, a4, b3, b4 = sp.symbols("a3 a4 b3 b4")

    disc_p = sp.expand(
        256 * a4**3 - 128 * a4**2 - 144 * a3**2 * a4
        - 27 * a3**4 + 16 * a4 + 4 * a3**2
    )
    f1_p = 1 + 12 * a4
    f2_p = 9 * a3**2 + 8 * a4 - 2

    disc_q = sp.expand(
        256 * b4**3 - 128 * b4**2 - 144 * b3**2 * b4
        - 27 * b3**4 + 16 * b4 + 4 * b3**2
    )
    f1_q = 1 + 12 * b4
    f2_q = 9 * b3**2 + 8 * b4 - 2

    c3 = a3 + b3
    c4 = a4 + sp.Rational(1, 6) + b4
    disc_r = sp.expand(
        256 * c4**3 - 512 * c4**2 + 288 * c3**2 * c4
        - 27 * c3**4 + 256 * c4 + 32 * c3**2
    )
    f1_r = sp.expand(4 + 12 * c4)
    f2_r = sp.expand(-16 + 16 * c4 + 9 * c3**2)

    surplus = sp.together(
        -disc_r / (4 * f1_r * f2_r)
        + disc_p / (4 * f1_p * f2_p)
        + disc_q / (4 * f1_q * f2_q)
    )
    num, _ = sp.fraction(surplus)
    negN = sp.expand(-num)

    vars4 = (a3, a4, b3, b4)
    grad = [sp.expand(sp.diff(negN, v)) for v in vars4]
    jac = [[sp.expand(sp.diff(g, v)) for v in vars4] for g in grad]
    degs = [sp.Poly(g, *vars4).total_degree() for g in grad]

    helpers = {
        "negN": negN,
        "disc_p": disc_p,
        "disc_q": disc_q,
        "f1_p": sp.expand(f1_p),
        "f1_q": sp.expand(f1_q),
        "f2_p": sp.expand(f2_p),
        "f2_q": sp.expand(f2_q),
    }
    return vars4, grad, jac, degs, helpers


def mp_frobenius_norm(M: mp.matrix) -> mp.mpf:
    s = mp.mpf("0")
    for i in range(M.rows):
        for j in range(M.cols):
            s += abs(M[i, j]) ** 2
    return mp.sqrt(s)


def mp_vec_norm(v: mp.matrix) -> mp.mpf:
    return mp.sqrt(sum(abs(v[i]) ** 2 for i in range(v.rows)))


def factorial(n: int) -> int:
    return math.factorial(n)


def bombieri_weyl_norm(poly_expr: sp.Expr, vars4: tuple[sp.Symbol, ...], d: int) -> mp.mpf:
    p = sp.Poly(poly_expr, *vars4)
    d_fact = mp.mpf(factorial(d))
    accum = mp.mpf("0")
    for mon, coeff in p.terms():
        deg = sum(mon)
        w_num = mp.mpf(factorial(d - deg))
        for e in mon:
            w_num *= mp.mpf(factorial(e))
        weight = w_num / d_fact
        c = mp.mpf(str(sp.N(coeff, 80)))
        accum += (c * c) * weight
    return mp.sqrt(accum)


def main() -> int:
    ap = argparse.ArgumentParser(description="Task B: Smale alpha certification for Case 3c roots")
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/first-proof/problem4-case3c-phc-certified.json",
        help="Input JSON containing case3c_points",
    )
    ap.add_argument(
        "--out",
        default="data/first-proof/problem4-case3c-alpha-certification.json",
        help="Output JSON path",
    )
    ap.add_argument("--dps", type=int, default=120, help="mpmath decimal precision")
    ap.add_argument("--max-newton-iters", type=int, default=20)
    ap.add_argument("--newton-step-tol", type=str, default="1e-90")
    ap.add_argument("--alpha-threshold", type=str, default="0.157671")
    ap.add_argument("--domain-tol", type=str, default="1e-30")
    args = ap.parse_args()

    mp.mp.dps = args.dps
    t0 = time.time()

    print("=" * 70)
    print("P4 n=4 Case 3c: Task B alpha-theory certification")
    print("=" * 70)
    print(f"input={args.inp}")
    print(f"dps={args.dps}")

    with open(args.inp, "r", encoding="utf-8") as f:
        src = json.load(f)
    seeds = src.get("case3c_points", [])
    if not seeds:
        raise RuntimeError(f"No case3c_points found in {args.inp}")

    vars4, grad, jac, degs, helpers = build_negN_and_grad()
    a3, a4, b3, b4 = vars4
    D = max(degs)

    # Lambdify in mpmath for high precision evaluation.
    grad_fn = [sp.lambdify(vars4, g, "mpmath") for g in grad]
    jac_fn = [[sp.lambdify(vars4, jij, "mpmath") for jij in row] for row in jac]
    negN_fn = sp.lambdify(vars4, helpers["negN"], "mpmath")
    disc_p_fn = sp.lambdify(vars4, helpers["disc_p"], "mpmath")
    disc_q_fn = sp.lambdify(vars4, helpers["disc_q"], "mpmath")
    f1_p_fn = sp.lambdify(vars4, helpers["f1_p"], "mpmath")
    f1_q_fn = sp.lambdify(vars4, helpers["f1_q"], "mpmath")
    f2_p_fn = sp.lambdify(vars4, helpers["f2_p"], "mpmath")
    f2_q_fn = sp.lambdify(vars4, helpers["f2_q"], "mpmath")

    def eval_F(x: mp.matrix) -> mp.matrix:
        return mp.matrix([f(*x) for f in grad_fn])

    def eval_J(x: mp.matrix) -> mp.matrix:
        return mp.matrix([[f(*x) for f in row] for row in jac_fn])

    print("Computing Bombieri-Weyl norm of gradient system...")
    bw_parts = [bombieri_weyl_norm(g, vars4, d) for g, d in zip(grad, degs)]
    f_norm_bw = mp.sqrt(sum(v * v for v in bw_parts))
    print(f"  ||F||_BW = {f_norm_bw}")

    alpha_threshold = mp.mpf(args.alpha_threshold)
    newton_step_tol = mp.mpf(args.newton_step_tol)
    domain_tol = mp.mpf(args.domain_tol)

    root_rows: list[dict[str, Any]] = []
    alpha_pass_count = 0
    all_in_domain = True
    all_nonneg = True

    for idx, seed in enumerate(seeds, start=1):
        x = mp.matrix(
            [
                mp.mpf(str(seed["a3"])),
                mp.mpf(str(seed["a4"])),
                mp.mpf(str(seed["b3"])),
                mp.mpf(str(seed["b4"])),
            ]
        )

        # High-precision Newton refinement from PHC seed.
        for _ in range(args.max_newton_iters):
            F = eval_F(x)
            J = eval_J(x)
            step = mp.lu_solve(J, -F)
            x = x + step
            if mp_vec_norm(step) < newton_step_tol:
                break

        F = eval_F(x)
        J = eval_J(x)
        res_norm = mp_vec_norm(F)

        # beta = ||J^{-1} F||
        beta_vec = mp.lu_solve(J, F)
        beta = mp_vec_norm(beta_vec)

        # Conservative gamma upper bound (Frobenius for operator norm).
        x_norm = mp.sqrt(sum(abs(x[i]) ** 2 for i in range(4)))
        x1 = mp.sqrt(1 + x_norm**2)
        Jinv = J**-1
        delta_diag = [mp.sqrt(degs[i]) * (x1 ** (degs[i] - 1)) for i in range(4)]
        jinv_delta = mp.matrix(
            [[Jinv[r, c] * delta_diag[c] for c in range(4)] for r in range(4)]
        )
        op_upper = mp_frobenius_norm(jinv_delta)
        mu_upper = max(mp.mpf("1"), f_norm_bw * op_upper)
        gamma_upper = mu_upper * (mp.mpf(D) ** mp.mpf("1.5")) / (2 * x1)
        alpha_upper = beta * gamma_upper
        alpha_pass = alpha_upper < alpha_threshold
        if alpha_pass:
            alpha_pass_count += 1

        negN_val = negN_fn(*x)
        dp = disc_p_fn(*x)
        dq = disc_q_fn(*x)
        f1p = f1_p_fn(*x)
        f1q = f1_q_fn(*x)
        f2p = f2_p_fn(*x)
        f2q = f2_q_fn(*x)
        in_domain = (
            dp >= -domain_tol
            and dq >= -domain_tol
            and f1p > domain_tol
            and f1q > domain_tol
            and f2p < -domain_tol
            and f2q < -domain_tol
        )
        if not in_domain:
            all_in_domain = False
        if negN_val < -domain_tol:
            all_nonneg = False

        root_rows.append(
            {
                "index": idx,
                "seed": {
                    "a3": float(seed["a3"]),
                    "a4": float(seed["a4"]),
                    "b3": float(seed["b3"]),
                    "b4": float(seed["b4"]),
                },
                "refined": {
                    "a3": mp.nstr(x[0], 40),
                    "a4": mp.nstr(x[1], 40),
                    "b3": mp.nstr(x[2], 40),
                    "b4": mp.nstr(x[3], 40),
                },
                "residual_norm": mp.nstr(res_norm, 20),
                "beta": mp.nstr(beta, 20),
                "gamma_upper": mp.nstr(gamma_upper, 20),
                "alpha_upper": mp.nstr(alpha_upper, 20),
                "alpha_pass": bool(alpha_pass),
                "alpha_threshold": mp.nstr(alpha_threshold, 20),
                "negN": mp.nstr(negN_val, 30),
                "in_domain": bool(in_domain),
                "domain_values": {
                    "disc_p": mp.nstr(dp, 20),
                    "disc_q": mp.nstr(dq, 20),
                    "f1_p": mp.nstr(f1p, 20),
                    "f1_q": mp.nstr(f1q, 20),
                    "f2_p": mp.nstr(f2p, 20),
                    "f2_q": mp.nstr(f2q, 20),
                },
            }
        )

    out = {
        "method": "smale_alpha_theory_upper_bound",
        "certification_type": "alpha_theory",
        "timestamp": time.time(),
        "runtime_sec": time.time() - t0,
        "input_file": args.inp,
        "failed_paths": src.get("failed"),
        "precision_dps": args.dps,
        "degrees": degs,
        "max_degree_D": D,
        "f_norm_bw": mp.nstr(f_norm_bw, 30),
        "alpha_threshold": mp.nstr(alpha_threshold, 20),
        "roots_total": len(root_rows),
        "roots_alpha_pass": alpha_pass_count,
        "all_alpha_below_threshold": alpha_pass_count == len(root_rows),
        "all_real_in_domain_nonneg": bool(all_in_domain and all_nonneg),
        "all_in_domain": bool(all_in_domain),
        "all_negN_nonneg": bool(all_nonneg),
        "case3c_certified": alpha_pass_count == len(root_rows),
        "case3c_certified_roots": alpha_pass_count == len(root_rows),
        "caveat": (
            "Alpha-theory here certifies listed roots; it does not by itself prove "
            "global exhaustiveness of all real in-domain roots."
        ),
        "roots": root_rows,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote: {args.out}")
    print(f"  roots_alpha_pass: {alpha_pass_count}/{len(root_rows)}")
    print(f"  all_in_domain: {all_in_domain}")
    print(f"  all_negN_nonneg: {all_nonneg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
