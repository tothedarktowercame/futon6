#!/usr/bin/env python3
"""Direction E operationalization probe for Problem 6.

Builds row-wise dataset along Case-2b greedy trajectories with:
- Direction A UST transfer constant proxy (kappa_row), and
- candidate graph/spectral parameters for Direction E.

Produces:
- JSON rows
- Markdown summary with correlation/threshold diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def key_row(graph: str, n: int, eps: float, t: int) -> Tuple[str, int, int, int]:
    return (graph, n, int(round(eps * 1000000)), t)


def algebraic_connectivity(num_nodes: int, edges: List[Tuple[int, int]]) -> Tuple[float, float]:
    if num_nodes <= 1:
        return 0.0, 0.0
    L = np.zeros((num_nodes, num_nodes), dtype=float)
    for u, v in edges:
        if u == v:
            continue
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    lmax = float(evals[-1]) if len(evals) else 0.0
    if len(evals) < 2:
        return 0.0, lmax
    l2 = float(max(evals[1], 0.0))
    return l2, lmax


def effective_resistance_diameter(dir_a, n: int, edges: List[Tuple[int, int]], nodes: List[int]) -> float:
    if len(nodes) <= 1:
        return 0.0
    L = dir_a.graph_laplacian(n, edges)
    Lplus = np.linalg.pinv(L)
    dmax = 0.0
    for i in range(len(nodes)):
        u = nodes[i]
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            r = float(Lplus[u, u] + Lplus[v, v] - 2.0 * Lplus[u, v])
            if r > dmax:
                dmax = r
    return dmax


def analyze_features_on_instance(dir_a, inst, c_step: float) -> List[Dict[str, float]]:
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    m0 = len(I0)

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    t_horizon = max(1, min(int(c_step * eps * n), m0 - 1))

    # Instance-level graph parameter candidate for E
    reff_diam_i0 = effective_resistance_diameter(dir_a, n, inst.edges, I0)

    s_t: List[int] = []
    s_set = set()
    m_t = np.zeros((n, n))

    rows = []

    for t in range(t_horizon):
        r_t = [v for v in I0 if v not in s_set]
        if not r_t:
            break

        headroom = eps * np.eye(n) - m_t
        evals_h = np.linalg.eigvalsh(headroom)
        h_min = float(np.min(evals_h))
        if h_min < 1e-12:
            break

        b_t = np.linalg.inv(headroom)
        bsqrt = np.linalg.cholesky(b_t + 1e-14 * np.eye(n))

        full_scores: Dict[int, float] = {}
        d_vals: Dict[int, float] = {}
        cross_pairs: List[Tuple[int, int]] = []
        deg_r = {v: 0 for v in r_t}

        for v in r_t:
            c_v = np.zeros((n, n))
            for u in s_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    gidx = edge_idx[key]
                    c_v += inst.X_edges[gidx]
                    deg_r[v] += 1
                    cross_pairs.append((u, v))

            y_v = bsqrt @ c_v @ bsqrt.T
            eigs = np.linalg.eigvalsh(y_v)
            s_v = float(max(eigs[-1], 0.0))
            d_v = float(np.sum(eigs[eigs > 0])) if s_v > 1e-10 else 0.0
            full_scores[v] = s_v
            d_vals[v] = d_v

        active_vs = [v for v in r_t if full_scores[v] > 1e-10]
        if len(active_vs) >= 2 and len(s_t) >= 1:
            # Build local cross graph on S_t union active R_t
            s_nodes = sorted(set(s_t))
            r_nodes = sorted(active_vs)
            local_nodes = s_nodes + r_nodes
            nid = {x: i for i, x in enumerate(local_nodes)}

            cross_local = []
            m_cross = 0
            for u in s_nodes:
                for v in r_nodes:
                    key = (min(u, v), max(u, v))
                    if key in edge_idx:
                        cross_local.append((nid[u], nid[v]))
                        m_cross += 1

            l2, lmax = algebraic_connectivity(len(local_nodes), cross_local)

            degs_r_active = [deg_r[v] for v in active_vs]
            mean_deg_r = float(np.mean(degs_r_active)) if degs_r_active else 0.0
            max_deg_r = float(np.max(degs_r_active)) if degs_r_active else 0.0
            min_deg_r = float(np.min(degs_r_active)) if degs_r_active else 0.0
            std_deg_r = float(np.std(degs_r_active)) if degs_r_active else 0.0
            cv_deg_r = std_deg_r / max(mean_deg_r, 1e-12)

            denom_bip = max(len(s_nodes) * len(r_nodes), 1)
            density_cross = m_cross / denom_bip

            # AR/GL related quantities to connect E to F
            s_arr = np.array([full_scores[v] for v in active_vs], dtype=float)
            d_arr = np.array([d_vals[v] for v in active_vs], dtype=float)
            above = s_arr > 1.0
            below = s_arr <= 1.0

            dsum = float(np.sum(d_arr))
            if dsum > 1e-15:
                w = d_arr / dsum
                rho_plus = float(np.sum(w[above])) if np.any(above) else 0.0
            else:
                rho_plus = 0.0

            pos_mass = float(np.sum(d_arr[below] * (1.0 / s_arr[below] - 1.0))) if np.any(below) else 0.0
            neg_mass = float(np.sum(d_arr[above] * (1.0 - 1.0 / s_arr[above]))) if np.any(above) else 0.0
            m_minus = float(np.max(s_arr[below])) if np.any(below) else 0.0

            mt_norm = float(np.linalg.norm(m_t, ord=2))
            x_headroom = mt_norm / eps
            psi_trace = float(np.trace(b_t))
            sign, logdet_h = np.linalg.slogdet(headroom)
            psi_logdet = float(-logdet_h) if sign > 0 else float("inf")

            best_v = min(r_t, key=lambda v: full_scores[v])

            rows.append(
                {
                    "graph": inst.graph_name,
                    "n": int(n),
                    "eps": float(eps),
                    "t": int(t),
                    "active": int(len(active_vs)),
                    "s_size": int(len(s_nodes)),
                    "r_size": int(len(r_nodes)),
                    "min_score": float(np.min(s_arr)),
                    "max_score": float(np.max(s_arr)),
                    "mean_score": float(np.mean(s_arr)),
                    "m_cross": int(m_cross),
                    "density_cross": float(density_cross),
                    "deg_r_mean": mean_deg_r,
                    "deg_r_max": max_deg_r,
                    "deg_r_min": min_deg_r,
                    "deg_r_cv": float(cv_deg_r),
                    "lambda2_cross": float(l2),
                    "lambda_max_cross": float(lmax),
                    "lambda2_over_lmax": float(l2 / max(lmax, 1e-12)),
                    "x_headroom": float(x_headroom),
                    "psi_trace": psi_trace,
                    "psi_logdet": psi_logdet,
                    "rho_plus": rho_plus,
                    "gl_gain": pos_mass,
                    "gl_penalty": neg_mass,
                    "gl_margin": pos_mass - neg_mass,
                    "m_minus": m_minus,
                    "reff_diam_i0": float(reff_diam_i0),
                    "best_v": int(best_v),
                }
            )

        # Greedy update (same trajectory as Direction A)
        if not r_t:
            break
        best_v = min(r_t, key=lambda v: full_scores[v])
        s_t.append(best_v)
        s_set.add(best_v)
        for u in s_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                m_t += inst.X_edges[edge_idx[key]]

    return rows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    xs = x - np.mean(x)
    ys = y - np.mean(y)
    denom = np.sqrt(np.sum(xs * xs) * np.sum(ys * ys))
    if denom <= 1e-15:
        return 0.0
    return float(np.sum(xs * ys) / denom)


def threshold_sweep(values: np.ndarray, hard: np.ndarray):
    if len(values) < 10:
        return None
    best = None
    qs = np.unique(np.percentile(values, np.linspace(5, 95, 37)))
    for thr in qs:
        for direction in ("<=", ">="):
            pred = values <= thr if direction == "<=" else values >= thr
            tp = int(np.sum(pred & hard))
            fp = int(np.sum(pred & (~hard)))
            fn = int(np.sum((~pred) & hard))
            if tp == 0:
                continue
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            cand = {
                "thr": float(thr),
                "dir": direction,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "prec": float(prec),
                "rec": float(rec),
                "f1": float(f1),
            }
            if best is None or cand["f1"] > best["f1"]:
                best = cand
    return best


def main():
    ap = argparse.ArgumentParser(description="Direction E feature operationalization")
    ap.add_argument("--nmax", type=int, default=32)
    ap.add_argument("--eps", nargs="+", type=float, default=[0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--c-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--kappa-hard", type=float, default=2.2)
    ap.add_argument("--out-json", type=Path, default=Path("data/first-proof/problem6-direction-e-feature-results.json"))
    ap.add_argument("--out-md", type=Path, default=Path("data/first-proof/problem6-direction-e-feature-report.md"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dir_a = load_module(repo / "scripts" / "verify-p6-gpl-h-direction-a.py", "p6_dir_a")

    rng = np.random.default_rng(args.seed)
    suite = dir_a.build_suite(args.nmax, rng)

    rows_all = []
    n_case2b = 0

    for graph_name, n, edges in suite:
        for eps in args.eps:
            inst = dir_a.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue

            n_case2b += 1
            seed_i = int(rng.integers(0, 2**31 - 1))

            # UST transfer rows (Direction A outcome target)
            st = dir_a.analyze_direction_a_on_instance(
                inst,
                samples=args.samples,
                c_step=args.c_step,
                seed=seed_i,
            )
            ust = [x for x in st if x.measure == "ust_forest"]
            kappa_map = {}
            for x in ust:
                kappa_row = max(x.ratio_fullmin_to_meanmin, x.max_ratio_full_to_mean_vertex)
                kappa_map[key_row(x.graph, x.n, x.eps, x.t)] = float(kappa_row)

            # Candidate E features on same deterministic trajectory
            feats = analyze_features_on_instance(dir_a, inst, args.c_step)
            for fr in feats:
                k = key_row(fr["graph"], fr["n"], fr["eps"], fr["t"])
                if k not in kappa_map:
                    continue
                fr["kappa_row"] = float(kappa_map[k])
                fr["hard"] = int(fr["kappa_row"] >= args.kappa_hard)
                rows_all.append(fr)

    # Aggregate diagnostics
    result = {
        "seed": args.seed,
        "nmax": args.nmax,
        "eps": args.eps,
        "samples": args.samples,
        "c_step": args.c_step,
        "kappa_hard": args.kappa_hard,
        "case2b_instances": n_case2b,
        "rows": rows_all,
    }

    numeric_features = [
        "active",
        "s_size",
        "r_size",
        "min_score",
        "max_score",
        "mean_score",
        "m_cross",
        "density_cross",
        "deg_r_mean",
        "deg_r_max",
        "deg_r_min",
        "deg_r_cv",
        "lambda2_cross",
        "lambda_max_cross",
        "lambda2_over_lmax",
        "x_headroom",
        "psi_trace",
        "psi_logdet",
        "rho_plus",
        "gl_gain",
        "gl_penalty",
        "gl_margin",
        "m_minus",
        "reff_diam_i0",
    ]

    if rows_all:
        kappa = np.array([r["kappa_row"] for r in rows_all], dtype=float)
        hard = np.array([bool(r["hard"]) for r in rows_all], dtype=bool)

        feature_stats = []
        for f in numeric_features:
            vals = np.array([r[f] for r in rows_all], dtype=float)
            corr = pearson(vals, kappa)
            thr = threshold_sweep(vals, hard)
            st = {
                "feature": f,
                "pearson_with_kappa": corr,
                "mean": float(np.mean(vals)),
                "p90": float(np.percentile(vals, 90)),
                "max": float(np.max(vals)),
                "best_threshold": thr,
            }
            feature_stats.append(st)

        feature_stats_sorted = sorted(feature_stats, key=lambda x: abs(x["pearson_with_kappa"]), reverse=True)
        result["feature_stats"] = feature_stats_sorted
        result["hard_rows"] = int(np.sum(hard))
        result["total_rows"] = int(len(rows_all))
    else:
        result["feature_stats"] = []
        result["hard_rows"] = 0
        result["total_rows"] = 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Problem 6: Direction E Feature Operationalization Report")
    lines.append("")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- nmax: `{args.nmax}`")
    lines.append(f"- eps: `{args.eps}`")
    lines.append(f"- samples (UST): `{args.samples}`")
    lines.append(f"- c_step: `{args.c_step}`")
    lines.append(f"- hard threshold on kappa_row: `{args.kappa_hard}`")
    lines.append(f"- Case-2b instances: `{n_case2b}`")
    lines.append(f"- Matched row count (with UST kappa + features): `{result['total_rows']}`")
    lines.append(f"- Hard rows: `{result['hard_rows']}`")
    lines.append("")

    lines.append("## Top Feature Correlations")
    lines.append("")
    lines.append("| feature | corr(kappa) | mean | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for st in result["feature_stats"][:12]:
        lines.append(
            f"| `{st['feature']}` | {st['pearson_with_kappa']:.3f} | {st['mean']:.3f} | {st['p90']:.3f} | {st['max']:.3f} |"
        )
    lines.append("")

    lines.append("## Best Single-Feature Hard-Row Classifiers")
    lines.append("")
    lines.append("| feature | rule | precision | recall | f1 | tp/fp/fn |")
    lines.append("|---|---|---:|---:|---:|---|")
    by_f1 = [x for x in result["feature_stats"] if x["best_threshold"]]
    by_f1.sort(key=lambda x: x["best_threshold"]["f1"], reverse=True)
    for st in by_f1[:12]:
        bt = st["best_threshold"]
        rule = f"{st['feature']} {bt['dir']} {bt['thr']:.4f}"
        lines.append(
            f"| `{st['feature']}` | `{rule}` | {bt['prec']:.3f} | {bt['rec']:.3f} | {bt['f1']:.3f} | {bt['tp']}/{bt['fp']}/{bt['fn']} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("Use this table to choose a concrete Direction-E parameter `P_t` and threshold for a theorem attempt.")
    lines.append("A useful candidate should separate hard rows (high recall) with manageable false positives.")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
