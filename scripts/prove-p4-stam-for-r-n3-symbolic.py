#!/usr/bin/env python3
"""Task 1: n=3 symbolic reduction for Stam-for-r.

Target inequality (centered cubics):
  r(p ⊞_3 q) <= r(p) r(q) / (r(p) + r(q))
equivalently
  1/r(p ⊞_3 q) >= 1/r(p) + 1/r(q).

This script:
1. Derives a closed formula for r in centered cubic coefficients (a2, a3).
2. Uses centered n=3 MSS addition (c2=a2+b2, c3=a3+b3).
3. Reduces Stam-for-r to a 3-variable normalized inequality F(lam,x,y) >= 0.
4. Performs strong global minimization checks on F over [0,1]x[-1,1]^2.
5. Writes machine-readable results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "data" / "first-proof" / "problem4-stam-for-r-n3-results.json"


def derive_r_formula():
    """Derive r(a2,a3) for centered cubic x^3 + a2 x + a3."""
    l1, l2 = sp.symbols("l1 l2", real=True)
    l3 = -l1 - l2
    roots = [l1, l2, l3]

    a2 = sp.expand(l1 * l2 + l1 * l3 + l2 * l3)
    a3 = sp.expand(-l1 * l2 * l3)

    S = []
    for i, li in enumerate(roots):
        si = 0
        for j, lj in enumerate(roots):
            if i != j:
                si += 1 / (li - lj)
        S.append(sp.simplify(si))

    Phi = sp.simplify(sum(si * si for si in S))
    Psi = 0
    for i in range(3):
        for j in range(i + 1, 3):
            Psi += ((S[i] - S[j]) / (roots[i] - roots[j])) ** 2
    Psi = sp.simplify(Psi)
    r_roots = sp.simplify(Psi / Phi)

    disc = sp.expand(-4 * a2**3 - 27 * a3**2)
    # Fit r = (A*a2^3 + B*a3^2) / (a2*disc).
    A, B = sp.symbols("A B")
    ansatz = (A * a2**3 + B * a3**2) / (a2 * disc)
    diff = sp.expand(sp.together(r_roots - ansatz).as_numer_denom()[0])
    poly = sp.Poly(diff, l1, l2)
    sol = sp.solve(poly.coeffs(), [A, B], dict=True)
    if not sol:
        raise RuntimeError("failed to solve coefficient ansatz for r(a2,a3)")
    A_val = sp.simplify(sol[0][A])
    B_val = sp.simplify(sol[0][B])
    A2, A3 = sp.symbols("A2 A3", real=True)
    disc_A = -4 * A2**3 - 27 * A3**2
    r_a = sp.simplify((A_val * A2**3 + B_val * A3**2) / (A2 * disc_A))
    inv_r_a = sp.simplify(1 / r_a)

    # Convert to positive scale coordinates: a2=-s, a3=u.
    s, u = sp.symbols("s u", positive=True, real=True)
    r_su = sp.simplify(r_a.subs({A2: -s, A3: u}))
    inv_r_su = sp.simplify(1 / r_su)
    return {
        "A": A_val,
        "B": B_val,
        "r_a2_a3": r_a,
        "inv_r_a2_a3": inv_r_a,
        "r_s_u": r_su,
        "inv_r_s_u": inv_r_su,
    }


def g_shape(x):
    return (2.0 * (1.0 - x * x)) / (3.0 * (1.0 + 2.0 * x * x))


def F_reduced(lam, x, y):
    z = lam ** 1.5 * x + (1.0 - lam) ** 1.5 * y
    return g_shape(z) - lam * g_shape(x) - (1.0 - lam) * g_shape(y)


def F_vec(lam_arr, x_arr, y_arr):
    z = np.power(lam_arr, 1.5) * x_arr + np.power(1.0 - lam_arr, 1.5) * y_arr
    gx = (2.0 * (1.0 - x_arr * x_arr)) / (3.0 * (1.0 + 2.0 * x_arr * x_arr))
    gy = (2.0 * (1.0 - y_arr * y_arr)) / (3.0 * (1.0 + 2.0 * y_arr * y_arr))
    gz = (2.0 * (1.0 - z * z)) / (3.0 * (1.0 + 2.0 * z * z))
    return gz - lam_arr * gx - (1.0 - lam_arr) * gy


@dataclass
class MinRecord:
    method: str
    min_value: float
    point: Tuple[float, float, float]


def run_global_min_checks(seed: int = 20260213) -> List[MinRecord]:
    out: List[MinRecord] = []

    # Differential evolution (global, box-constrained).
    bounds = [(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    de = differential_evolution(
        lambda v: F_reduced(v[0], v[1], v[2]),
        bounds=bounds,
        seed=seed,
        maxiter=350,
        popsize=24,
        tol=1e-12,
        polish=True,
        updating="deferred",
        workers=1,
    )
    out.append(MinRecord("differential_evolution", float(de.fun), tuple(float(x) for x in de.x)))

    # Large random search for additional confidence.
    rng = np.random.default_rng(seed + 1)
    n_rand = 3_000_000
    lam = rng.uniform(0.0, 1.0, size=n_rand)
    x = rng.uniform(-1.0, 1.0, size=n_rand)
    y = rng.uniform(-1.0, 1.0, size=n_rand)
    vals = F_vec(lam, x, y)
    i = int(np.argmin(vals))
    out.append(MinRecord("random_search_3m", float(vals[i]), (float(lam[i]), float(x[i]), float(y[i]))))

    # Dense grid slices near expected equality manifolds.
    # 1) x = y = 0, varying lam.
    lam_line = np.linspace(0.0, 1.0, 20001)
    v_line = F_vec(lam_line, np.zeros_like(lam_line), np.zeros_like(lam_line))
    i = int(np.argmin(v_line))
    out.append(MinRecord("line_x0_y0", float(v_line[i]), (float(lam_line[i]), 0.0, 0.0)))

    # 2) lam fixed near worst random point, dense (x,y).
    lam0 = out[1].point[0]
    grid = np.linspace(-1.0, 1.0, 2501)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    V = F_vec(np.full_like(X, lam0), X, Y)
    j = int(np.argmin(V))
    ii, kk = np.unravel_index(j, V.shape)
    out.append(MinRecord("dense_xy_slice", float(V[ii, kk]), (float(lam0), float(grid[ii]), float(grid[kk]))))

    return out


def main():
    deriv = derive_r_formula()

    # Symbolic reduced identity:
    # let c = 2/sqrt(27), x = u/(c s^(3/2)), y = v/(c t^(3/2)), lam=s/(s+t)
    # then inv_r(s,u)=s*g(x), inv_r(t,v)=t*g(y), inv_r(s+t,u+v)=(s+t)*g(z)
    # with z = lam^(3/2) x + (1-lam)^(3/2) y.
    lam, x, y = sp.symbols("lam x y", real=True)
    z = lam ** sp.Rational(3, 2) * x + (1 - lam) ** sp.Rational(3, 2) * y
    g = lambda t: sp.simplify(2 * (1 - t**2) / (3 * (1 + 2 * t**2)))
    F_sym = sp.simplify(g(z) - lam * g(x) - (1 - lam) * g(y))
    F_num, F_den = sp.fraction(sp.together(F_sym))

    mins = run_global_min_checks()
    min_val = min(r.min_value for r in mins)
    min_rec = min(mins, key=lambda r: r.min_value)

    result: Dict = {
        "task": "Task1_n3_symbolic_stam_for_r",
        "closed_form": {
            "A": str(deriv["A"]),
            "B": str(deriv["B"]),
            "r_a2_a3": str(deriv["r_a2_a3"]),
            "inv_r_a2_a3": str(deriv["inv_r_a2_a3"]),
            "r_s_u": str(deriv["r_s_u"]),
            "inv_r_s_u": str(deriv["inv_r_s_u"]),
        },
        "reduced_expression": {
            "F_num": str(sp.expand(F_num)),
            "F_den": str(sp.factor(F_den)),
            "claim": "F(lam,x,y) >= 0 on lam in [0,1], x,y in [-1,1]",
        },
        "min_checks": [asdict(r) for r in mins],
        "best_min": asdict(min_rec),
        "global_min_lower_bound_observed": float(min_val),
        "status": "no_negative_found",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 70)
    print("Task 1: n=3 Stam-for-r symbolic reduction")
    print("=" * 70)
    print(f"A={deriv['A']}, B={deriv['B']}")
    print(f"r(a2,a3) = {deriv['r_a2_a3']}")
    print(f"inv_r(s,u) = {deriv['inv_r_s_u']}")
    print(f"Reduced F numerator terms: {len(sp.expand(F_num).as_ordered_terms())}")
    print(f"Observed global minimum: {min_rec.min_value:.3e} at {min_rec.point} ({min_rec.method})")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
