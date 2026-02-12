#!/usr/bin/env python3
"""PHCpack certification pass for P4 n=4 gradient critical points.

Builds the exact 4D gradient system for -N(a3,a4,b3,b4), solves with PHCpack,
then filters real solutions in the domain and classifies by symmetry cases.

Primary target: Case 3c (a3!=0, b3!=0, a3!=+-b3) and checking -N >= 0 there.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import sympy as sp
from phcpy.solver import solve, stable_mixed_volume
from phcpy.solutions import strsol2dict, verify


def build_negN_and_grad():
    a3, a4, b3, b4 = sp.symbols("a3 a4 b3 b4")

    disc_p = sp.expand(
        256 * a4**3 - 128 * a4**2 - 144 * a3**2 * a4 - 27 * a3**4 + 16 * a4 + 4 * a3**2
    )
    f1_p = 1 + 12 * a4
    f2_p = 9 * a3**2 + 8 * a4 - 2

    disc_q = sp.expand(
        256 * b4**3 - 128 * b4**2 - 144 * b3**2 * b4 - 27 * b3**4 + 16 * b4 + 4 * b3**2
    )
    f1_q = 1 + 12 * b4
    f2_q = 9 * b3**2 + 8 * b4 - 2

    c3 = a3 + b3
    c4 = a4 + sp.Rational(1, 6) + b4
    disc_r = sp.expand(
        256 * c4**3 - 512 * c4**2 + 288 * c3**2 * c4 - 27 * c3**4 + 256 * c4 + 32 * c3**2
    )
    f1_r = sp.expand(4 + 12 * c4)
    f2_r = sp.expand(-16 + 16 * c4 + 9 * c3**2)

    surplus = sp.together(
        -disc_r / (4 * f1_r * f2_r)
        + disc_p / (4 * f1_p * f2_p)
        + disc_q / (4 * f1_q * f2_q)
    )
    num, den = sp.fraction(surplus)
    negN = sp.expand(-num)

    grads = [sp.expand(sp.diff(negN, v)) for v in (a3, a4, b3, b4)]

    helpers = {
        "disc_p": sp.expand(disc_p),
        "disc_q": sp.expand(disc_q),
        "f1_p": sp.expand(f1_p),
        "f1_q": sp.expand(f1_q),
        "f2_p": sp.expand(f2_p),
        "f2_q": sp.expand(f2_q),
        "den": sp.expand(den),
    }
    return (a3, a4, b3, b4), negN, grads, helpers


def to_phc_pol(expr: sp.Expr) -> str:
    # PHCpack expects '^' power syntax and terminating ';'.
    s = sp.sstr(sp.expand(expr))
    s = s.replace("**", "^")
    return s + ";"


def domain_ok(vals: dict[str, float], fns: dict[str, object], tol: float) -> bool:
    dp = float(fns["disc_p"](vals["a3"], vals["a4"], vals["b3"], vals["b4"]))
    dq = float(fns["disc_q"](vals["a3"], vals["a4"], vals["b3"], vals["b4"]))
    f1p = float(fns["f1_p"](vals["a3"], vals["a4"], vals["b3"], vals["b4"]))
    f1q = float(fns["f1_q"](vals["a3"], vals["a4"], vals["b3"], vals["b4"]))
    f2p = float(fns["f2_p"](vals["a3"], vals["a4"], vals["b3"], vals["b4"]))
    f2q = float(fns["f2_q"](vals["a3"], vals["a4"], vals["b3"], vals["b4"]))

    return (
        dp >= -tol
        and dq >= -tol
        and f1p > tol
        and f1q > tol
        and f2p < -tol
        and f2q < -tol
    )


def classify(vals: dict[str, float], tol: float) -> str:
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
    kept = []

    def close(p, q):
        return all(abs(p[k] - q[k]) <= tol for k in ("a3", "a4", "b3", "b4"))

    for pt in points:
        found = False
        for i, kp in enumerate(kept):
            if close(pt, kp):
                # keep lower residual representative
                if pt.get("res", 1e99) < kp.get("res", 1e99):
                    kept[i] = pt
                found = True
                break
        if not found:
            kept.append(pt)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="d", choices=["d", "dd", "qd"])
    ap.add_argument("--tasks", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--real-tol", type=float, default=1e-8)
    ap.add_argument("--dup-tol", type=float, default=1e-9)
    ap.add_argument("--domain-tol", type=float, default=1e-10)
    ap.add_argument(
        "--out",
        default="data/first-proof/problem4-case3c-phc-results.json",
        help="JSON output path",
    )
    args = ap.parse_args()

    t0 = time.time()
    print("=== P4 n=4 Case 3c PHCpack pass ===")
    print(f"precision={args.precision} tasks={args.tasks}")

    vars4, negN, grads, helpers = build_negN_and_grad()
    a3, a4, b3, b4 = vars4

    print("Building polynomial strings...")
    pols = [to_phc_pol(g) for g in grads]
    for i, p in enumerate(pols, start=1):
        print(f"  g{i} length: {len(p)}")

    print("Computing stable mixed volume...")
    tmv = time.time()
    try:
        mv, smv = stable_mixed_volume(pols)
    except Exception as exc:  # noqa: BLE001
        mv, smv = None, None
        print(f"  mixed volume failed: {exc}")
    print(f"  mv={mv} smv={smv} ({time.time()-tmv:.2f}s)")

    print("Solving system with PHCpack...")
    tsolve = time.time()
    sols = solve(pols, tasks=args.tasks, precision=args.precision, checkin=True)
    print(f"  total isolated solutions returned: {len(sols)} ({time.time()-tsolve:.2f}s)")

    vres = verify(pols, sols)
    print(f"  aggregate residual from verify(): {vres:.3e}")

    # numeric evaluators
    negN_fn = sp.lambdify((a3, a4, b3, b4), negN, "math")
    fns = {
        k: sp.lambdify((a3, a4, b3, b4), v, "math") for k, v in helpers.items()
    }

    parsed = []
    for s in sols:
        d = strsol2dict(s, precision=args.precision)
        # expected variable names: a3, a4, b3, b4
        coords = {k: d[k] for k in ("a3", "a4", "b3", "b4") if k in d}
        if len(coords) != 4:
            continue
        parsed.append(
            {
                "a3": coords["a3"],
                "a4": coords["a4"],
                "b3": coords["b3"],
                "b4": coords["b4"],
                "res": float(d.get("res", math.nan)),
                "rco": float(d.get("rco", math.nan)),
                "m": int(d.get("m", 1)),
            }
        )

    print(f"Parsed solutions with named coordinates: {len(parsed)}")

    real_raw = []
    for d in parsed:
        vals = {}
        is_real = True
        for k in ("a3", "a4", "b3", "b4"):
            z = d[k]
            if abs(z.imag) > args.real_tol:
                is_real = False
                break
            vals[k] = float(z.real)
        if not is_real:
            continue
        vals["res"] = d["res"]
        vals["rco"] = d["rco"]
        vals["m"] = d["m"]
        vals["negN"] = float(negN_fn(vals["a3"], vals["a4"], vals["b3"], vals["b4"]))
        vals["in_domain"] = domain_ok(vals, fns, args.domain_tol)
        vals["case"] = classify(vals, args.real_tol)
        real_raw.append(vals)

    print(f"Real solutions (raw): {len(real_raw)}")

    real_unique = dedup(real_raw, args.dup_tol)
    print(f"Real solutions (dedup): {len(real_unique)}")

    in_dom = [r for r in real_unique if r["in_domain"]]
    print(f"Real solutions in domain: {len(in_dom)}")

    by_case = Counter(r["case"] for r in in_dom)
    print("In-domain counts by case:", dict(by_case))

    if in_dom:
        min_negN = min(r["negN"] for r in in_dom)
        max_negN = max(r["negN"] for r in in_dom)
        print(f"-N range on in-domain real solutions: [{min_negN:.12g}, {max_negN:.12g}]")

    c3c = [r for r in in_dom if r["case"] == "case3c"]
    print(f"Case 3c in-domain real solutions: {len(c3c)}")
    if c3c:
        c3c_min = min(r["negN"] for r in c3c)
        c3c_max = max(r["negN"] for r in c3c)
        print(f"Case 3c -N range: [{c3c_min:.12g}, {c3c_max:.12g}]")

    out = {
        "timestamp": time.time(),
        "runtime_sec": time.time() - t0,
        "precision": args.precision,
        "tasks": args.tasks,
        "mixed_volume": mv,
        "stable_mixed_volume": smv,
        "total_solutions_returned": len(sols),
        "aggregate_verify_residual": vres,
        "real_solutions_raw": len(real_raw),
        "real_solutions_dedup": len(real_unique),
        "real_in_domain": len(in_dom),
        "in_domain_by_case": dict(by_case),
        "in_domain_min_negN": (min(r["negN"] for r in in_dom) if in_dom else None),
        "in_domain_max_negN": (max(r["negN"] for r in in_dom) if in_dom else None),
        "case3c_count": len(c3c),
        "case3c_min_negN": (min(r["negN"] for r in c3c) if c3c else None),
        "case3c_max_negN": (max(r["negN"] for r in c3c) if c3c else None),
        "case3c_points": c3c,
        "all_in_domain_points": in_dom,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"Total runtime: {time.time()-t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
