#!/usr/bin/env python3
"""Task 4: n=4 SOS/brute-force algebraic analysis.

Given solver constraints (no cvxpy/scs in this environment), this script:
1. Builds the symbolic n=4 Stam surplus numerator in normalized coordinates.
2. Reports algebraic complexity (degree, monomials, symmetries).
3. Runs brute-force and global optimization searches for counterexamples.
4. Evaluates Stam-for-r numerically on n=4 to support/alarm the route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "data" / "first-proof" / "problem4-n4-sos-bruteforce-results.json"


def mss_convolve(a_coeffs, b_coeffs, n):
    from math import factorial

    c = np.zeros(n, dtype=float)
    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            ai = 1.0 if i == 0 else float(a_coeffs[i - 1])
            bj = 1.0 if j == 0 else float(b_coeffs[j - 1])
            w = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
            s += w * ai * bj
        c[k - 1] = s
    return c


def roots_from_coeffs(coeffs, tol=1e-8):
    r = np.roots(np.concatenate([[1.0], np.asarray(coeffs, dtype=float)]))
    if np.max(np.abs(r.imag)) > tol:
        return None
    rr = np.sort(r.real.astype(float))
    if np.min(np.diff(rr)) < 1e-10:
        return None
    return rr


def coeffs_from_roots(roots):
    p = np.poly(np.asarray(roots, dtype=float))
    return p[1:].astype(float)


def score(roots):
    roots = np.asarray(roots, dtype=float)
    n = len(roots)
    s = np.zeros(n, dtype=float)
    for i in range(n):
        d = roots[i] - np.delete(roots, i)
        if np.min(np.abs(d)) < 1e-12:
            return None
        s[i] = np.sum(1.0 / d)
    return s


def phi(roots):
    s = score(roots)
    if s is None:
        return np.inf
    return float(np.dot(s, s))


def psi(roots):
    s = score(roots)
    if s is None:
        return np.inf
    n = len(roots)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += ((s[i] - s[j]) / (roots[i] - roots[j])) ** 2
    return float(total)


def r_ratio(roots):
    return psi(roots) / phi(roots)


def normalize_a2_minus1(roots):
    roots = np.asarray(roots, dtype=float)
    roots = roots - np.mean(roots)
    a2 = float(np.poly(roots)[2])
    if a2 >= -1e-12:
        return None
    scale = np.sqrt((-1.0) / a2)
    return np.sort(roots * scale)


def build_symbolic_surplus():
    a3, a4, b3, b4 = sp.symbols("a3 a4 b3 b4", real=True)
    disc_p = sp.expand(256 * a4**3 - 128 * a4**2 - 144 * a3**2 * a4 - 27 * a3**4 + 16 * a4 + 4 * a3**2)
    f1_p = 1 + 12 * a4
    f2_p = 9 * a3**2 + 8 * a4 - 2

    disc_q = sp.expand(256 * b4**3 - 128 * b4**2 - 144 * b3**2 * b4 - 27 * b3**4 + 16 * b4 + 4 * b3**2)
    f1_q = 1 + 12 * b4
    f2_q = 9 * b3**2 + 8 * b4 - 2

    c3 = a3 + b3
    c4 = a4 + sp.Rational(1, 6) + b4
    disc_r = sp.expand(256 * c4**3 - 512 * c4**2 + 288 * c3**2 * c4 - 27 * c3**4 + 256 * c4 + 32 * c3**2)
    f1_r = sp.expand(4 + 12 * c4)
    f2_r = sp.expand(-16 + 16 * c4 + 9 * c3**2)

    surplus = sp.together(-disc_r / (4 * f1_r * f2_r) + disc_p / (4 * f1_p * f2_p) + disc_q / (4 * f1_q * f2_q))
    N, D = sp.fraction(surplus)
    N = sp.expand(N)
    D = sp.expand(D)
    polyN = sp.Poly(N, a3, a4, b3, b4)
    return {
        "N": N,
        "D": D,
        "polyN": polyN,
        "vars": (a3, a4, b3, b4),
    }


def random_search(seed=20260213, trials=120000):
    rng = np.random.default_rng(seed)
    min_stam = np.inf
    min_r = np.inf
    best_stam = None
    best_r = None
    stam_viol = 0
    r_viol = 0
    valid = 0

    for i in range(trials):
        p = normalize_a2_minus1(np.sort(rng.normal(size=4) * rng.uniform(0.3, 3.0)))
        q = normalize_a2_minus1(np.sort(rng.normal(size=4) * rng.uniform(0.3, 3.0)))
        if p is None or q is None:
            continue
        cp = mss_convolve(coeffs_from_roots(p), coeffs_from_roots(q), 4)
        c = roots_from_coeffs(cp)
        if c is None:
            continue

        phi_p = phi(p)
        phi_q = phi(q)
        phi_c = phi(c)
        if not np.isfinite(phi_p) or not np.isfinite(phi_q) or not np.isfinite(phi_c):
            continue

        r_p = r_ratio(p)
        r_q = r_ratio(q)
        r_c = r_ratio(c)
        if not np.isfinite(r_p) or not np.isfinite(r_q) or not np.isfinite(r_c):
            continue

        valid += 1
        stam = 1.0 / phi_c - 1.0 / phi_p - 1.0 / phi_q
        r_harm = (r_p * r_q) / (r_p + r_q)
        r_gap = r_harm - r_c

        if stam < min_stam:
            min_stam = stam
            best_stam = (p.tolist(), q.tolist(), c.tolist(), float(stam))
        if r_gap < min_r:
            min_r = r_gap
            best_r = (p.tolist(), q.tolist(), c.tolist(), float(r_gap))

        if stam < -1e-12:
            stam_viol += 1
        if r_gap < -1e-12:
            r_viol += 1
        if (i + 1) % max(1, trials // 6) == 0:
            print(f"[random] {i + 1}/{trials}")

    return {
        "trials_requested": trials,
        "valid_samples": valid,
        "stam_min_surplus": float(min_stam),
        "stam_violations": stam_viol,
        "stam_best_example": best_stam,
        "stam_for_r_min_gap": float(min_r),
        "stam_for_r_violations": r_viol,
        "stam_for_r_best_example": best_r,
    }


def coeff_objective_stam(v):
    a3, a4, b3, b4 = v
    p = roots_from_coeffs(np.array([0.0, -1.0, a3, a4]))
    q = roots_from_coeffs(np.array([0.0, -1.0, b3, b4]))
    if p is None or q is None:
        return 1e3
    c = roots_from_coeffs(mss_convolve(np.array([0.0, -1.0, a3, a4]), np.array([0.0, -1.0, b3, b4]), 4))
    if c is None:
        return 1e3
    val = 1.0 / phi(c) - 1.0 / phi(p) - 1.0 / phi(q)
    return float(val)


def global_opt_search(seed=20260213, maxiter=50, popsize=18):
    # Real-rooted region is narrow in a4,b4 so keep conservative bounds.
    bounds = [(-0.58, 0.58), (-0.0832, 0.249), (-0.58, 0.58), (-0.0832, 0.249)]
    de = differential_evolution(
        coeff_objective_stam,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        tol=1e-8,
        workers=1,
    )
    return {"min_value": float(de.fun), "argmin": [float(x) for x in de.x]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260213)
    ap.add_argument("--trials", type=int, default=120000)
    ap.add_argument("--de-maxiter", type=int, default=50)
    ap.add_argument("--de-popsize", type=int, default=18)
    args = ap.parse_args()

    sym = build_symbolic_surplus()
    polyN = sym["polyN"]
    a3, a4, b3, b4 = sym["vars"]
    swap_ok = sp.expand(sym["N"].subs({a3: b3, a4: b4, b3: a3, b4: a4}, simultaneous=True) - sym["N"]) == 0
    refl_ok = sp.expand(sym["N"].subs({a3: -a3, b3: -b3}) - sym["N"]) == 0

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

    rnd = random_search(seed=args.seed, trials=args.trials)
    opt = global_opt_search(seed=args.seed + 1, maxiter=args.de_maxiter, popsize=args.de_popsize)

    out = {
        "task": "Task4_n4_sos_bruteforce",
        "symbolic": {
            "surplus_num_degree": int(polyN.total_degree()),
            "surplus_num_terms": int(len(polyN.as_dict())),
            "swap_symmetry": bool(swap_ok),
            "reflection_symmetry_a3_b3": bool(refl_ok),
        },
        "solver_availability": {"cvxpy": have_cvxpy, "scs": have_scs},
        "random_search": rnd,
        "global_optimization": opt,
        "status": "solver_blocked_for_full_sos" if not (have_cvxpy and have_scs) else "solver_available",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 70)
    print("Task 4: n=4 SOS / brute-force")
    print("=" * 70)
    print(f"Symbolic N degree={out['symbolic']['surplus_num_degree']} terms={out['symbolic']['surplus_num_terms']}")
    print(f"symmetries: swap={swap_ok} reflection={refl_ok}")
    print(f"solver availability: cvxpy={have_cvxpy}, scs={have_scs}")
    print(f"random search min Stam surplus: {rnd['stam_min_surplus']:.6e} (viol={rnd['stam_violations']})")
    print(f"random search min Stam-for-r gap: {rnd['stam_for_r_min_gap']:.6e} (viol={rnd['stam_for_r_violations']})")
    print(f"global-opt min Stam surplus: {opt['min_value']:.6e}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
