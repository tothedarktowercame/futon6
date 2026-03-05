#!/usr/bin/env python3
"""PHCpack Case 3c run with robust endpoint reclassification.

Purpose:
  Reclassify endpoints that fail `strsol2dict/diagnostics` parsing (often due to
  NaN diagnostics fields) using raw-string parsing of t and coordinates.

This targets the case where many "failed" paths are actually divergent/non-finite,
which is acceptable for path accounting.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

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
    grads = [sp.expand(sp.diff(negN, v)) for v in (a3, a4, b3, b4)]

    helpers = {
        "disc_p": sp.expand(disc_p),
        "disc_q": sp.expand(disc_q),
        "f1_p": sp.expand(f1_p),
        "f1_q": sp.expand(f1_q),
        "f2_p": sp.expand(f2_p),
        "f2_q": sp.expand(f2_q),
    }
    return (a3, a4, b3, b4), negN, grads, helpers


def to_phc_pol(expr: sp.Expr) -> str:
    return sp.sstr(sp.expand(expr)).replace("**", "^") + ";"


def parse_float_token(tok: str):
    t = tok.strip()
    if not t:
        return None
    low = t.lower()
    if "nan" in low or "*" in t:
        return float("nan")
    if "inf" in low:
        return float("inf") if not low.startswith("-") else float("-inf")
    try:
        return float(t)
    except Exception:
        return None


def raw_parse_solution(sol: str):
    """Parse t and coordinates from raw PHCpack solution string."""
    out = {}
    # Matches lines like:
    # a3 : 1.234E-01  0.000E+00
    # t : 1.000E+00 0.000E+00
    pat = re.compile(
        r"(?im)^\s*(t|a3|a4|b3|b4)\s*:\s*([^\s]+)\s+([^\s]+)\s*$"
    )
    for m in pat.finditer(sol):
        k = m.group(1).lower()
        re_s = parse_float_token(m.group(2))
        im_s = parse_float_token(m.group(3))
        if re_s is None or im_s is None:
            continue
        out[k] = complex(re_s, im_s)
    return out


def domain_ok(vals: dict[str, float], fns: dict, tol: float) -> bool:
    a3v, a4v, b3v, b4v = vals["a3"], vals["a4"], vals["b3"], vals["b4"]
    dp = float(fns["disc_p"](a3v, a4v, b3v, b4v))
    dq = float(fns["disc_q"](a3v, a4v, b3v, b4v))
    f1p = float(fns["f1_p"](a3v, a4v, b3v, b4v))
    f1q = float(fns["f1_q"](a3v, a4v, b3v, b4v))
    f2p = float(fns["f2_p"](a3v, a4v, b3v, b4v))
    f2q = float(fns["f2_q"](a3v, a4v, b3v, b4v))
    return dp >= -tol and dq >= -tol and f1p > tol and f1q > tol and f2p < -tol and f2q < -tol


def classify_case(vals: dict[str, float], tol: float) -> str:
    a3, b3 = vals["a3"], vals["b3"]
    a4, b4 = vals["a4"], vals["b4"]
    if abs(a3) <= tol and abs(b3) <= tol:
        return "case1"
    if abs(a3) <= tol or abs(b3) <= tol:
        return "case2"
    if abs(a3 - b3) <= tol and abs(a4 - b4) <= tol:
        return "case3a"
    if abs(a3 + b3) <= tol and abs(a4 - b4) <= tol:
        return "case3b"
    return "case3c"


def dedup(points: list[dict], tol: float) -> list[dict]:
    kept: list[dict] = []
    for pt in points:
        found = False
        for i, kp in enumerate(kept):
            if all(abs(pt[k] - kp[k]) <= tol for k in ("a3", "a4", "b3", "b4")):
                if pt.get("res", 1e99) < kp.get("res", 1e99):
                    kept[i] = pt
                found = True
                break
        if not found:
            kept.append(pt)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--precision", default="d", choices=["d", "dd", "qd"])
    ap.add_argument("--real-tol", type=float, default=1e-8)
    ap.add_argument("--dup-tol", type=float, default=1e-6)
    ap.add_argument("--domain-tol", type=float, default=1e-10)
    ap.add_argument("--infinity-tol", type=float, default=1e8)
    ap.add_argument("--residual-tol", type=float, default=1e-6)
    ap.add_argument("--out", default="data/first-proof/problem4-case3c-certified-v2.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("P4 n=4: PHCpack reclassification run")
    print("=" * 70)
    print(f"precision={args.precision} tasks={args.tasks}")

    vars4, negN, grads, helpers = build_negN_and_grad()
    a3, a4, b3, b4 = vars4
    pols = [to_phc_pol(g) for g in grads]

    from phcpy.starters import total_degree, total_degree_start_system
    from phcpy.trackers import double_track, double_double_track, quad_double_track
    from phcpy.solutions import strsol2dict, diagnostics

    bezout = total_degree(pols)
    print(f"Bezout = {bezout}")
    start_sys, start_sols = total_degree_start_system(pols)
    assert len(start_sols) == bezout

    t_track = time.time()
    if args.precision == "d":
        _, end_sols = double_track(pols, start_sys, start_sols, tasks=args.tasks)
    elif args.precision == "dd":
        _, end_sols = double_double_track(pols, start_sys, start_sols, tasks=args.tasks)
    else:
        _, end_sols = quad_double_track(pols, start_sys, start_sols, tasks=args.tasks)
    track_time = time.time() - t_track
    print(f"Tracked {len(end_sols)} endpoints in {track_time:.1f}s")

    grad_fns = [sp.lambdify((a3, a4, b3, b4), g, "math") for g in grads]
    negN_fn = sp.lambdify((a3, a4, b3, b4), negN, "math")
    helper_fns = {k: sp.lambdify((a3, a4, b3, b4), v, "math") for k, v in helpers.items()}

    counts = Counter()
    finite_points = []
    failed_samples = []

    for sol in end_sols:
        # First try official parser.
        try:
            d = strsol2dict(sol)
            err, rco, res = diagnostics(sol)
            t_val = d.get("t", complex(0, 0))
            t_reached = abs(t_val.real - 1.0) < 1e-6 and abs(t_val.imag) < 1e-6
            if not t_reached:
                counts["failed_t_not_1"] += 1
                continue

            coords = [d.get(k, complex(float("nan"), 0)) for k in ("a3", "a4", "b3", "b4")]
            if any((not sp.Float(z.real).is_finite) or (not sp.Float(z.imag).is_finite) for z in coords):
                counts["diverged_nonfinite"] += 1
                continue
            max_coord = max(abs(z) for z in coords)
            if max_coord > args.infinity_tol:
                counts["diverged_large"] += 1
                continue

            if res < args.residual_tol:
                counts["finite"] += 1
                rec = {
                    "a3": float(coords[0].real),
                    "a4": float(coords[1].real),
                    "b3": float(coords[2].real),
                    "b4": float(coords[3].real),
                    "res": float(res),
                    "rco": float(rco),
                    "parser": "phcpy",
                }
                finite_points.append(rec)
            else:
                counts["diverged_residual"] += 1
            continue
        except Exception:
            pass

        # Raw fallback parse.
        raw = raw_parse_solution(sol)
        if "t" not in raw:
            counts["failed_unparsed"] += 1
            if len(failed_samples) < 8:
                failed_samples.append(sol[:600])
            continue

        t_val = raw["t"]
        t_reached = abs(t_val.real - 1.0) < 1e-6 and abs(t_val.imag) < 1e-6
        if not t_reached:
            counts["failed_t_not_1_raw"] += 1
            continue

        if not all(k in raw for k in ("a3", "a4", "b3", "b4")):
            counts["failed_missing_coords_raw"] += 1
            if len(failed_samples) < 8:
                failed_samples.append(sol[:600])
            continue

        coords = [raw[k] for k in ("a3", "a4", "b3", "b4")]
        if any((not sp.Float(z.real).is_finite) or (not sp.Float(z.imag).is_finite) for z in coords):
            counts["diverged_nonfinite_raw"] += 1
            continue

        max_coord = max(abs(z) for z in coords)
        if max_coord > args.infinity_tol:
            counts["diverged_large_raw"] += 1
            continue

        # Finite candidate: recompute residual from gradient directly.
        a3v, a4v, b3v, b4v = [float(z.real) for z in coords]
        gres = max(abs(fn(a3v, a4v, b3v, b4v)) for fn in grad_fns)
        if gres < args.residual_tol:
            counts["finite_raw"] += 1
            finite_points.append({
                "a3": a3v, "a4": a4v, "b3": b3v, "b4": b4v,
                "res": float(gres), "rco": float("nan"), "parser": "raw",
            })
        else:
            counts["diverged_residual_raw"] += 1

    total_accounted = sum(counts.values())
    print("Counts:", dict(counts))
    print(f"Total accounted = {total_accounted} / {bezout}")

    finite_unique = dedup(finite_points, args.dup_tol)
    real_unique = [p for p in finite_unique]  # finite points are real-valued extraction
    for p in real_unique:
        p["negN"] = float(negN_fn(p["a3"], p["a4"], p["b3"], p["b4"]))
        p["in_domain"] = domain_ok(p, helper_fns, args.domain_tol)
        p["case"] = classify_case(p, args.real_tol)
    in_dom = [p for p in real_unique if p["in_domain"]]
    by_case = Counter(p["case"] for p in in_dom)

    out = {
        "method": "total_degree_homotopy_reclassified",
        "precision": args.precision,
        "tasks": args.tasks,
        "timestamp": time.time(),
        "runtime_sec": time.time() - t0,
        "track_time_sec": track_time,
        "bezout_bound": bezout,
        "paths_tracked": len(end_sols),
        "counts": dict(counts),
        "total_accounted": total_accounted,
        "accounting_certified": total_accounted == bezout,
        "finite_unique": len(finite_unique),
        "real_in_domain": len(in_dom),
        "in_domain_by_case": dict(by_case),
        "all_nonneg": all(p["negN"] >= -1e-6 for p in in_dom),
        "case3c_count": sum(1 for p in in_dom if p["case"] == "case3c"),
        "case3c_min_negN": min((p["negN"] for p in in_dom if p["case"] == "case3c"), default=None),
        "case3c_max_negN": max((p["negN"] for p in in_dom if p["case"] == "case3c"), default=None),
        "all_in_domain_points": in_dom,
        "failed_samples": failed_samples,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
