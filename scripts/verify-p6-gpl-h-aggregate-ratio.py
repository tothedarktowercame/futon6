#!/usr/bin/env python3
"""Diagnostics for the aggregate-ratio bridge in GPL-H.

Per step (active vertices only):
  s_v = ||Y_t(v)||,
  d_v = tr(Y_t(v)),
  g_v = d_v/s_v.

Define:
  dbar = avg_v d_v,
  gbar = avg_v g_v,
  ratio = dbar/gbar.

Deterministic certificate:
  min_v s_v <= ratio.
Hence ratio < 1 certifies a good step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def load_base_module(repo_root: Path):
    base_path = repo_root / "scripts" / "verify-p6-gpl-h.py"
    spec = importlib.util.spec_from_file_location("p6base", base_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def family_name(graph_name: str) -> str:
    if graph_name.startswith("ER_"):
        return "ER"
    if graph_name.startswith("RandReg_"):
        return "RandReg"
    if graph_name.startswith("DisjCliq_"):
        return "DisjCliq"
    if graph_name.startswith("Dumbbell_"):
        return "Dumbbell"
    if graph_name.startswith("Barbell_"):
        return "Barbell"
    if graph_name.startswith("K_"):
        return "K"
    if graph_name.startswith("C_"):
        return "C"
    return graph_name.split("_", 1)[0]


def build_suite(base, nmax: int, rng):
    suite = []
    for n in range(8, nmax + 1, 4):
        suite.append((f"K_{n}", n, base.complete_graph(n)))
        suite.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            suite.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = base.dumbbell_graph(n // 3)
            suite.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = base.disjoint_cliques(n // 3, 3)
            suite.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                suite.append((f"ER_{n}_p{p_er}", n, er_edges))
        for d_rr in [4, 6]:
            if d_rr < n:
                rr_edges = base.random_regular(n, d_rr, rng)
                if len(rr_edges) > n:
                    suite.append((f"RandReg_{n}_d{d_rr}", n, rr_edges))
    return suite


def analyze_instance(inst, c_step: float):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    m0 = len(I0)

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    T = max(1, min(int(c_step * eps * n), m0 - 1))

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))

    rows = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        if not R_t:
            break

        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        s_all = []
        s_list = []
        d_list = []
        g_list = []

        best_v = None
        best_score = float("inf")

        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]

            Y_v = Bsqrt @ C_v @ Bsqrt.T
            evals = np.linalg.eigvalsh(Y_v)
            s_v = float(max(evals[-1], 0.0))
            s_all.append(s_v)

            if s_v > 1e-10:
                d_v = float(np.sum(evals[evals > 0]))
                g_v = d_v / s_v
                s_list.append(s_v)
                d_list.append(d_v)
                g_list.append(g_v)

            if s_v < best_score:
                best_score = s_v
                best_v = v

        if s_list:
            s = np.array(s_list, dtype=float)
            d = np.array(d_list, dtype=float)
            g = np.array(g_list, dtype=float)

            dbar = float(np.mean(d))
            gbar = float(np.mean(g))
            ratio = dbar / max(gbar, 1e-15)

            above = s > 1.0
            below = s <= 1.0

            sum_d = float(np.sum(d))
            rho_plus = float(np.sum(d[above]) / sum_d) if np.any(above) else 0.0

            pos_mass = float(np.sum(d[below] * (1.0 / s[below] - 1.0))) if np.any(below) else 0.0
            neg_mass = float(np.sum(d[above] * (1.0 - 1.0 / s[above]))) if np.any(above) else 0.0
            gap_margin = pos_mass - neg_mass  # |A|*(gbar-dbar)

            m_minus = float(np.min(s[below])) if np.any(below) else None
            M_plus = float(np.max(s[above])) if np.any(above) else None

            rho_crit = None
            rho_crit_lower = None
            rho_margin = None
            if m_minus is not None:
                if M_plus is None:
                    rho_crit = 1.0
                else:
                    a = (1.0 / m_minus) - 1.0
                    b = 1.0 - (1.0 / M_plus)
                    rho_crit = a / (a + b) if (a + b) > 0 else 1.0
                rho_crit_lower = 1.0 - m_minus
                rho_margin = rho_crit - rho_plus

            rows.append({
                "graph": inst.graph_name,
                "family": family_name(inst.graph_name),
                "n": int(inst.n),
                "eps": float(inst.epsilon),
                "t": int(t),
                "active": int(len(s)),
                "min_score": float(np.min(np.array(s_all, dtype=float))) if s_all else 0.0,
                "min_score_active": float(np.min(s)),
                "max_score": float(np.max(s)),
                "mean_score": float(np.mean(s)),
                "dbar": dbar,
                "gbar": gbar,
                "ratio": ratio,
                "ratio_margin": float(gbar - dbar),
                "above1_frac": float(np.mean(above)),
                "above1_drift_frac": rho_plus,
                "above1_max_score": float(np.max(s[above])) if np.any(above) else 0.0,
                "pos_mass": pos_mass,
                "neg_mass": neg_mass,
                "gap_margin": gap_margin,
                "m_minus": m_minus,
                "M_plus": M_plus,
                "rho_crit": rho_crit,
                "rho_crit_lower": rho_crit_lower,
                "rho_margin": rho_margin,
            })

        if best_v is None:
            break

        S_t.append(best_v)
        S_set.add(best_v)

        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate ratio diagnostics for GPL-H")
    ap.add_argument("--nmax", type=int, default=48)
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--c-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260212)
    ap.add_argument("--output", default="data/first-proof/problem6-aggregate-ratio-results.json")
    args = ap.parse_args()

    np.random.seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    base = load_base_module(repo_root)

    rng = np.random.default_rng(args.seed)
    suite = build_suite(base, args.nmax, rng)

    case2b = 0
    rows = []

    for graph_name, n, edges in suite:
        for eps in args.eps:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            case2b += 1
            rows.extend(analyze_instance(inst, c_step=args.c_step))

    if not rows:
        print("No rows collected.")
        return

    ratio = np.array([r["ratio"] for r in rows], dtype=float)
    min_score = np.array([r["min_score"] for r in rows], dtype=float)
    nontrivial = min_score > 1e-12

    print("=" * 84)
    print("Aggregate Ratio Diagnostics")
    print("=" * 84)
    print(f"Case-2b instances: {case2b}")
    print(f"Step rows:         {len(rows)}")
    print(f"Nontrivial rows:   {int(np.sum(nontrivial))}")
    print()

    print("Ratio summary:")
    print(f"  all rows:       mean={np.mean(ratio):.6f} p95={np.quantile(ratio,0.95):.6f} max={np.max(ratio):.6f}")
    if np.any(nontrivial):
        rn = ratio[nontrivial]
        print(f"  nontrivial:     mean={np.mean(rn):.6f} p95={np.quantile(rn,0.95):.6f} max={np.max(rn):.6f}")
        print(f"  nontrivial fail ratio>=1: {int(np.sum(rn >= 1.0 - 1e-12))}")

    above = np.array([r["above1_frac"] for r in rows], dtype=float)
    above_d = np.array([r["above1_drift_frac"] for r in rows], dtype=float)
    print("\nAbove-1 mass:")
    print(f"  above1_frac:    mean={np.mean(above):.6f} p95={np.quantile(above,0.95):.6f} max={np.max(above):.6f}")
    print(f"  above1_drift:   mean={np.mean(above_d):.6f} p95={np.quantile(above_d,0.95):.6f} max={np.max(above_d):.6f}")

    rho_margin_all = np.array([r["rho_margin"] for r in rows if r["rho_margin"] is not None], dtype=float)
    rho_margin_non = np.array([r["rho_margin"] for r in rows if r["rho_margin"] is not None and r["min_score"] > 1e-12], dtype=float)
    if rho_margin_all.size:
        print("\nMass-threshold margin:")
        print(f"  rho_margin all: mean={np.mean(rho_margin_all):.6f} p05={np.quantile(rho_margin_all,0.05):.6f} min={np.min(rho_margin_all):.6f}")
    if rho_margin_non.size:
        print(f"  rho_margin nontrivial: mean={np.mean(rho_margin_non):.6f} p05={np.quantile(rho_margin_non,0.05):.6f} min={np.min(rho_margin_non):.6f}")

    hard = sorted(rows, key=lambda r: r["ratio"], reverse=True)[:15]
    print("\nTop ratio rows:")
    for r in hard:
        print(
            f"  {r['graph']:<16} n={r['n']:>2} eps={r['eps']:>4.2f} t={r['t']:>2} "
            f"min={r['min_score']:.6f} max={r['max_score']:.6f} ratio={r['ratio']:.6f} "
            f"above1_frac={r['above1_frac']:.3f} above1_drift={r['above1_drift_frac']:.3f}"
        )

    by_family = {}
    fams = sorted(set(r["family"] for r in rows))
    for fam in fams:
        rr = [r for r in rows if r["family"] == fam]
        rf = np.array([r["ratio"] for r in rr], dtype=float)
        nf = np.array([r["min_score"] > 1e-12 for r in rr], dtype=bool)
        rho_m = np.array([r["rho_margin"] for r in rr if r["rho_margin"] is not None and r["min_score"] > 1e-12], dtype=float)
        by_family[fam] = {
            "rows": len(rr),
            "ratio_mean": float(np.mean(rf)),
            "ratio_p95": float(np.quantile(rf, 0.95)),
            "ratio_max": float(np.max(rf)),
            "nontrivial_rows": int(np.sum(nf)),
            "nontrivial_ratio_max": float(np.max(rf[nf])) if np.any(nf) else None,
            "above1_drift_p95": float(np.quantile(np.array([r["above1_drift_frac"] for r in rr]), 0.95)),
            "rho_margin_min": float(np.min(rho_m)) if rho_m.size else None,
        }

    out = {
        "case2b_instances": case2b,
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "nontrivial_rows": int(np.sum(nontrivial)),
            "ratio_mean": float(np.mean(ratio)),
            "ratio_p95": float(np.quantile(ratio, 0.95)),
            "ratio_max": float(np.max(ratio)),
            "nontrivial_ratio_max": float(np.max(ratio[nontrivial])) if np.any(nontrivial) else None,
            "nontrivial_ratio_fail_rows": int(np.sum(ratio[nontrivial] >= 1.0 - 1e-12)) if np.any(nontrivial) else 0,
            "above1_frac_mean": float(np.mean(above)),
            "above1_drift_mean": float(np.mean(above_d)),
            "rho_margin_min": float(np.min(rho_margin_non)) if rho_margin_non.size else None,
        },
        "by_family": by_family,
        "top_ratio_rows": hard,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
