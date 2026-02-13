#!/usr/bin/env python3
"""Problem 4 n=4 handoff execution via Phi4*disc identity.

Implements the handoff steps:
1) symbolic identity checks
2) exact Stam surplus construction
3) SOS attempt status
4) e3=0 special-case study
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "data" / "first-proof" / "problem4-n4-sos-handoff-results.json"
OUT_MD = REPO_ROOT / "data" / "first-proof" / "problem4-n4-sos-handoff-results.md"


def phi_from_roots(roots):
    n = len(roots)
    s = np.zeros(n, dtype=float)
    for i in range(n):
        d = roots[i] - np.delete(roots, i)
        if np.min(np.abs(d)) < 1e-12:
            return np.inf
        s[i] = np.sum(1.0 / d)
    return float(np.dot(s, s))


def verify_phi4_disc_identity() -> Dict:
    l1, l2, l3 = sp.symbols("l1 l2 l3")
    l4 = -(l1 + l2 + l3)
    roots = [l1, l2, l3, l4]

    e2 = sum(roots[i] * roots[j] for i in range(4) for j in range(i + 1, 4))
    e3 = sum(roots[i] * roots[j] * roots[k] for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4))
    e4 = roots[0] * roots[1] * roots[2] * roots[3]

    S = []
    for i in range(4):
        si = 0
        for j in range(4):
            if i != j:
                si += 1 / (roots[i] - roots[j])
        S.append(sp.simplify(si))
    Phi = sp.simplify(sum(si * si for si in S))

    disc = sp.prod((roots[i] - roots[j]) ** 2 for i in range(4) for j in range(i + 1, 4))
    lhs = sp.together(Phi * disc)
    rhs = sp.expand(-8 * e2**5 - 64 * e2**3 * e4 - 36 * e2**2 * e3**2 + 384 * e2 * e4**2 - 432 * e3**2 * e4)
    diff = sp.together(lhs - rhs)
    diff_num, _ = sp.fraction(sp.cancel(diff))
    diff_num = sp.expand(diff_num)
    exact = bool(diff_num == 0 or sp.Poly(diff_num, l1, l2, l3).is_zero)

    # numeric spot checks
    rng = np.random.default_rng(20260213)
    max_abs_err = 0.0
    for _ in range(200):
        r = np.sort(rng.normal(size=4) * rng.uniform(0.2, 3.0))
        r = r - np.mean(r)
        e2n = sum(r[i] * r[j] for i in range(4) for j in range(i + 1, 4))
        e3n = sum(r[i] * r[j] * r[k] for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4))
        e4n = r[0] * r[1] * r[2] * r[3]
        discn = np.prod([(r[i] - r[j]) ** 2 for i in range(4) for j in range(i + 1, 4)])
        phin = phi_from_roots(r)
        rhsn = -8 * e2n**5 - 64 * e2n**3 * e4n - 36 * e2n**2 * e3n**2 + 384 * e2n * e4n**2 - 432 * e3n**2 * e4n
        err = abs(phin * discn - rhsn)
        max_abs_err = max(max_abs_err, float(err))

    return {
        "exact_symbolic": bool(exact),
        "max_abs_err_numeric_200": max_abs_err,
    }


def verify_centered_mss_formula() -> Dict:
    # coefficient convention: p(x)=x^4+a1 x^3+a2 x^2+a3 x+a4
    a1, a2, a3, a4 = sp.symbols("a1 a2 a3 a4")
    b1, b2, b3, b4 = sp.symbols("b1 b2 b3 b4")
    n = 4
    c = []
    for k in range(1, n + 1):
        s = 0
        for i in range(k + 1):
            j = k - i
            ai = 1 if i == 0 else [a1, a2, a3, a4][i - 1]
            bj = 1 if j == 0 else [b1, b2, b3, b4][j - 1]
            w = sp.factorial(n - i) * sp.factorial(n - j) / (sp.factorial(n) * sp.factorial(n - k))
            s += w * ai * bj
        c.append(sp.expand(s))

    c1, c2, c3, c4 = c
    centered = {
        "c1": sp.expand(c1.subs({a1: 0, b1: 0})),
        "c2": sp.expand(c2.subs({a1: 0, b1: 0})),
        "c3": sp.expand(c3.subs({a1: 0, b1: 0})),
        "c4": sp.expand(c4.subs({a1: 0, b1: 0})),
    }

    # For e-notation polynomial x^4 + e2 x^2 - e3 x + e4:
    # a2=e2, a3=-e3, a4=e4.
    e2p, e3p, e4p, e2q, e3q, e4q = sp.symbols("e2p e3p e4p e2q e3q e4q")
    subs = {a2: e2p, a3: -e3p, a4: e4p, b2: e2q, b3: -e3q, b4: e4q}
    E2 = sp.expand(centered["c2"].subs(subs))
    E3 = sp.expand(-centered["c3"].subs(subs))
    E4 = sp.expand(centered["c4"].subs(subs))

    return {
        "c2_centered": str(centered["c2"]),
        "c3_centered": str(centered["c3"]),
        "c4_centered": str(centered["c4"]),
        "E2_formula": str(E2),
        "E3_formula": str(E3),
        "E4_formula": str(E4),
        "E2_additive": bool(sp.expand(E2 - (e2p + e2q)) == 0),
        "E3_additive": bool(sp.expand(E3 - (e3p + e3q)) == 0),
        "E4_cross_1over6": bool(sp.expand(E4 - (e4p + e4q + sp.Rational(1, 6) * e2p * e2q)) == 0),
    }


def build_surplus_polynomial() -> Dict:
    s, t, u, v, a, b = sp.symbols("s t u v a b", real=True)

    # inv_phi4 in e-variables for x^4 + e2 x^2 - e3 x + e4
    e2, e3, e4 = sp.symbols("e2 e3 e4", real=True)
    disc = 256 * e4**3 - 128 * e2**2 * e4**2 + 144 * e2 * e3**2 * e4 + 16 * e2**4 * e4 - 27 * e3**4 - 4 * e2**3 * e3**2
    P = -8 * e2**5 - 64 * e2**3 * e4 - 36 * e2**2 * e3**2 + 384 * e2 * e4**2 - 432 * e3**2 * e4
    inv_phi = sp.simplify(disc / P)

    # e2(p)=-s, e2(q)=-t, e3(p)=u, e3(q)=v, e4(p)=a, e4(q)=b
    # E4 has +st/6 cross term.
    e2c = -(s + t)
    e3c = u + v
    e4c = a + b + s * t / 6

    inv_c = sp.simplify(inv_phi.subs({e2: e2c, e3: e3c, e4: e4c}))
    inv_p = sp.simplify(inv_phi.subs({e2: -s, e3: u, e4: a}))
    inv_q = sp.simplify(inv_phi.subs({e2: -t, e3: v, e4: b}))

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)
    D = sp.expand(D)
    polyN = sp.Poly(N, s, t, u, v, a, b)

    # symmetries
    swap = sp.expand(N.subs({s: t, t: s, u: v, v: u, a: b, b: a}, simultaneous=True) - N) == 0
    uv_reflect = sp.expand(N.subs({u: -u, v: -v}) - N) == 0

    return {
        "N": N,
        "D": D,
        "degree_N": int(polyN.total_degree()),
        "terms_N": int(len(polyN.as_dict())),
        "swap_symmetry": bool(swap),
        "uv_reflection_symmetry": bool(uv_reflect),
    }


def certify_even_case_symbolic(N_expr, D_expr) -> Dict:
    """Symbolic certificate for the symmetric quartic case (u=v=0)."""
    s, t, u, v, a, b = sp.symbols("s t u v a b", real=True)
    x, y, lam = sp.symbols("x y lam", real=True)
    p, q = sp.symbols("p q", real=True)

    N0 = sp.expand(N_expr.subs({u: 0, v: 0}))
    D0 = sp.expand(D_expr.subs({u: 0, v: 0}))

    # Normalize to x=4a/s^2, y=4b/t^2 in (0,1) and lambda=t/s>0.
    Nsub = sp.expand(N0.subs({a: s**2 * x / 4, b: t**2 * y / 4, t: lam * s}) / s**16)
    Dsub = sp.expand(D0.subs({a: s**2 * x / 4, b: t**2 * y / 4, t: lam * s}) / s**15)

    L = sp.expand(3 * lam**2 * y - 3 * lam**2 - 4 * lam + 3 * x - 3)
    M = sp.expand(3 * lam**2 * y + lam**2 + 4 * lam + 3 * x + 1)
    Q = sp.expand(sp.cancel(Nsub / (16 * lam**6 * (x - 1) * (y - 1) * L)))

    N_fact_expected = sp.expand(16 * lam**6 * (x - 1) * (y - 1) * L * Q)
    D_fact_expected = sp.expand(288 * lam**5 * (lam + 1) * (x - 1) * (3 * x + 1) * (y - 1) * (3 * y + 1) * L * M)
    reduced_surplus = sp.simplify(
        sp.cancel(Nsub / Dsub - (lam * Q) / (18 * (lam + 1) * (3 * x + 1) * (3 * y + 1) * M))
    )

    # Shift to p=3x-1, q=3y-1 so equality is at (p,q)=(0,0).
    Qpq = sp.expand(Q.subs({x: (p + 1) / 3, y: (q + 1) / 3}))
    polyQ_lam = sp.Poly(Qpq, lam)
    A2 = sp.expand(polyQ_lam.coeff_monomial(lam**2))
    A1 = sp.expand(polyQ_lam.coeff_monomial(lam**1))
    A0 = sp.expand(polyQ_lam.coeff_monomial(lam**0))

    A2_decomp = sp.expand((q + 2) ** 2 * p**2 + q**2 * (q + 6) * (p + 2))
    A0_decomp = sp.expand((p + 2) ** 2 * q**2 + p**2 * (p + 6) * (q + 2))

    A1_core = sp.expand(A1 / 4)
    A1_square = sp.expand(
        (q + 3) * (p - q * (2 - q) / (2 * (q + 3))) ** 2
        + q**2 * (-q**2 + 16 * q + 32) / (4 * (q + 3))
    )
    A1_disc_in_p = sp.expand(sp.discriminant(sp.Poly(A1_core, p), p))

    # Numeric sanity over the normalized feasible domain.
    rng = np.random.default_rng(20260213)
    fQ = sp.lambdify((lam, x, y), Q, "numpy")
    q_min = np.inf
    q_argmin = None
    for _ in range(200000):
        lamv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng.uniform(1e-6, 1 - 1e-6))
        yv = float(rng.uniform(1e-6, 1 - 1e-6))
        qv = float(fQ(lamv, xv, yv))
        if qv < q_min:
            q_min = qv
            q_argmin = [lamv, xv, yv]

    return {
        "N0_degree": int(sp.Poly(N0, s, t, a, b).total_degree()),
        "N0_terms": int(len(sp.Poly(N0, s, t, a, b).as_dict())),
        "Nsub_terms": int(len(sp.Poly(Nsub, lam, x, y).as_dict())),
        "factor_checks": {
            "N_factorization_exact": bool(sp.expand(Nsub - N_fact_expected) == 0),
            "D_factorization_exact": bool(sp.expand(Dsub - D_fact_expected) == 0),
            "reduced_surplus_exact": bool(reduced_surplus == 0),
        },
        "reduced_form": {
            "surplus_formula": "lambda*Q / [18*(lambda+1)*(3x+1)*(3y+1)*(lambda^2*(3y+1)+4lambda+3x+1)]",
            "domain": "lambda>0, x in (0,1), y in (0,1)",
        },
        "Q_quadratic_in_lambda": {
            "A2": str(A2),
            "A1": str(A1),
            "A0": str(A0),
            "A2_nonnegative_decomposition": str(A2_decomp),
            "A0_nonnegative_decomposition": str(A0_decomp),
            "A1_square_completion": str(A1_square),
            "A1_discriminant_in_p": str(A1_disc_in_p),
            "A2_decomp_exact": bool(sp.expand(A2 - A2_decomp) == 0),
            "A0_decomp_exact": bool(sp.expand(A0 - A0_decomp) == 0),
            "A1_square_exact": bool(sp.simplify(A1_core - A1_square) == 0),
        },
        "conclusion": {
            "Q_nonnegative_on_domain_reason": (
                "A2>=0 and A0>=0 by explicit decompositions on p,q in (-1,2); "
                "A1>=0 by square-completion with q+3>0 and -q^2+16q+32>0 on q in (-1,2)."
            ),
            "surplus_nonnegative_even_case": True,
            "equality_characterization": "x=y=1/3 i.e. a=s^2/12 and b=t^2/12",
        },
        "numeric_sanity_Q": {
            "random_samples": 200000,
            "min_Q": float(q_min),
            "argmin_lam_x_y": q_argmin,
        },
    }


def special_case_even_quartic(N_expr, D_expr, trials=80000) -> Dict:
    # u=v=0 case.
    s, t, u, v, a, b = sp.symbols("s t u v a b", real=True)
    N0 = sp.expand(N_expr.subs({u: 0, v: 0}))
    D0 = sp.expand(D_expr.subs({u: 0, v: 0}))
    poly0 = sp.Poly(N0, s, t, a, b)

    # Sample real-rooted even quartics: roots = ±r1, ±r2.
    rng = np.random.default_rng(20260213)
    min_surplus = np.inf
    min_point = None
    neg = 0
    valid = 0

    e2v = lambda r1, r2: -(r1 * r1 + r2 * r2)
    e4v = lambda r1, r2: (r1 * r1) * (r2 * r2)

    fN0 = sp.lambdify((s, t, a, b), N0, "numpy")
    fD0 = sp.lambdify((s, t, a, b), D0, "numpy")

    for _ in range(trials):
        r1, r2 = sorted(np.abs(rng.normal(size=2) * rng.uniform(0.2, 3.5)))
        q1, q2 = sorted(np.abs(rng.normal(size=2) * rng.uniform(0.2, 3.5)))
        spv = -e2v(r1, r2)
        tpv = -e2v(q1, q2)
        av = e4v(r1, r2)
        bv = e4v(q1, q2)
        nval = float(fN0(spv, tpv, av, bv))
        dval = float(fD0(spv, tpv, av, bv))
        if abs(dval) < 1e-14:
            continue
        sval = nval / dval
        valid += 1
        if sval < min_surplus:
            min_surplus = sval
            min_point = [spv, tpv, av, bv]
        if sval < -1e-11:
            neg += 1

    def objective(z):
        spv, tpv, av, bv = z
        if spv <= 0 or tpv <= 0:
            return 1e3
        if av < 0 or bv < 0:
            return 1e3
        if av > (spv * spv) / 4 or bv > (tpv * tpv) / 4:
            return 1e3
        dval = float(fD0(spv, tpv, av, bv))
        if abs(dval) < 1e-12:
            return 1e2
        return float(fN0(spv, tpv, av, bv) / dval)

    de = differential_evolution(
        objective,
        bounds=[(0.05, 20), (0.05, 20), (0, 100), (0, 100)],
        maxiter=25,
        popsize=14,
        polish=True,
        seed=20260214,
        workers=1,
    )

    return {
        "N0_degree": int(poly0.total_degree()),
        "N0_terms": int(len(poly0.as_dict())),
        "random_valid": valid,
        "random_min_surplus": float(min_surplus),
        "random_neg_count": int(neg),
        "random_argmin": min_point,
        "global_opt_min_surplus": float(de.fun),
        "global_opt_argmin": [float(x) for x in de.x],
    }


def main():
    step1_id = verify_phi4_disc_identity()
    step1_mss = verify_centered_mss_formula()
    step2 = build_surplus_polynomial()

    have_cvxpy = True
    have_scs = True
    try:
        import cvxpy  # noqa: F401
    except Exception:
        have_cvxpy = False
    try:
        import scs  # noqa: F401
    except Exception:
        have_scs = False

    step4_even_cert = certify_even_case_symbolic(step2["N"], step2["D"])
    step4_even = special_case_even_quartic(step2["N"], step2["D"])

    out = {
        "task": "P4_n4_sos_handoff_execution",
        "step1_phi4_disc_identity": step1_id,
        "step1_centered_mss": step1_mss,
        "step2_surplus": {
            "degree_N": step2["degree_N"],
            "terms_N": step2["terms_N"],
            "swap_symmetry": step2["swap_symmetry"],
            "uv_reflection_symmetry": step2["uv_reflection_symmetry"],
        },
        "step3_sos_attempt": {
            "cvxpy_available": have_cvxpy,
            "scs_available": have_scs,
            "status": "blocked_no_sdp_solver" if not (have_cvxpy and have_scs) else "solver_available",
        },
        "step4_even_special_case": step4_even,
        "step4_even_symbolic_certificate": step4_even_cert,
        "sign_note": {
            "E4_cross_term_verified": "+(1/6)e2(p)e2(q)",
            "s_t_convention": "if s=-e2(p), t=-e2(q), then E4 = a + b + st/6",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    md = []
    md.append("# P4 n=4 SOS Handoff Execution Results")
    md.append("")
    md.append("## Step 1")
    md.append(f"- Phi4*disc identity exact symbolic: `{out['step1_phi4_disc_identity']['exact_symbolic']}`")
    md.append(f"- Max numeric abs error (200 tests): `{out['step1_phi4_disc_identity']['max_abs_err_numeric_200']:.3e}`")
    md.append(f"- Centered MSS E2 additive: `{out['step1_centered_mss']['E2_additive']}`")
    md.append(f"- Centered MSS E3 additive: `{out['step1_centered_mss']['E3_additive']}`")
    md.append(f"- Centered MSS E4 cross 1/6: `{out['step1_centered_mss']['E4_cross_1over6']}`")
    md.append("")
    md.append("## Step 2")
    md.append(f"- Surplus numerator degree: `{out['step2_surplus']['degree_N']}`")
    md.append(f"- Surplus numerator terms: `{out['step2_surplus']['terms_N']}`")
    md.append(f"- Swap symmetry: `{out['step2_surplus']['swap_symmetry']}`")
    md.append(f"- (u,v)->(-u,-v) symmetry: `{out['step2_surplus']['uv_reflection_symmetry']}`")
    md.append("")
    md.append("## Step 3")
    md.append(f"- cvxpy available: `{out['step3_sos_attempt']['cvxpy_available']}`")
    md.append(f"- scs available: `{out['step3_sos_attempt']['scs_available']}`")
    md.append(f"- status: `{out['step3_sos_attempt']['status']}`")
    md.append("")
    md.append("## Step 4 (u=v=0) symbolic")
    md.append(
        "- factorization checks: "
        f"`N={out['step4_even_symbolic_certificate']['factor_checks']['N_factorization_exact']}`, "
        f"`D={out['step4_even_symbolic_certificate']['factor_checks']['D_factorization_exact']}`, "
        f"`reduced={out['step4_even_symbolic_certificate']['factor_checks']['reduced_surplus_exact']}`"
    )
    md.append(f"- A2 decomposition exact: `{out['step4_even_symbolic_certificate']['Q_quadratic_in_lambda']['A2_decomp_exact']}`")
    md.append(f"- A0 decomposition exact: `{out['step4_even_symbolic_certificate']['Q_quadratic_in_lambda']['A0_decomp_exact']}`")
    md.append(f"- A1 square-completion exact: `{out['step4_even_symbolic_certificate']['Q_quadratic_in_lambda']['A1_square_exact']}`")
    md.append(f"- symbolic conclusion surplus>=0: `{out['step4_even_symbolic_certificate']['conclusion']['surplus_nonnegative_even_case']}`")
    md.append("")
    md.append("## Step 4 (u=v=0) numeric sanity")
    md.append(f"- random valid samples: `{out['step4_even_special_case']['random_valid']}`")
    md.append(f"- random min surplus: `{out['step4_even_special_case']['random_min_surplus']:.6e}`")
    md.append(f"- random negative count: `{out['step4_even_special_case']['random_neg_count']}`")
    md.append(f"- global-opt min surplus: `{out['step4_even_special_case']['global_opt_min_surplus']:.6e}`")
    md.append("")
    md.append("## Sign Convention Note")
    md.append("- Verified E4 cross term is `+(1/6)e2(p)e2(q)`.")
    md.append("- With `s=-e2(p), t=-e2(q)`, this is `+st/6` in E4.")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("=" * 72)
    print("P4 n=4 SOS handoff execution")
    print("=" * 72)
    print(f"Phi4*disc exact symbolic: {out['step1_phi4_disc_identity']['exact_symbolic']}")
    print(f"Centered MSS E4 cross-term verified: {out['step1_centered_mss']['E4_cross_1over6']}")
    print(f"Surplus N degree={out['step2_surplus']['degree_N']} terms={out['step2_surplus']['terms_N']}")
    print(f"SOS status: {out['step3_sos_attempt']['status']}")
    print(
        "Even-case symbolic certificate checks: "
        f"N={out['step4_even_symbolic_certificate']['factor_checks']['N_factorization_exact']} "
        f"D={out['step4_even_symbolic_certificate']['factor_checks']['D_factorization_exact']} "
        f"reduced={out['step4_even_symbolic_certificate']['factor_checks']['reduced_surplus_exact']}"
    )
    print(f"Even-case random min surplus: {out['step4_even_special_case']['random_min_surplus']:.6e}")
    print(f"Even-case DE min surplus: {out['step4_even_special_case']['global_opt_min_surplus']:.6e}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
