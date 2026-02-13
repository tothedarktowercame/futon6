#!/usr/bin/env python3
"""Problem 4 deep-dive task runner.

Implements computational tasks from:
  data/first-proof/CODEX-P4-DEEPDIVE-HANDOFF.md

Supported tasks:
  - d1: Direction D1, extended numerical verification of phi(p conv q) <= phi(p) phi(q)
  - d2: Direction D2, flow-ratio monotonicity probes
  - d3: Direction D3, n=3 algebraic warm-up (symbolic + numeric validation)
  - a2: Direction A2, score decomposition / orthogonality experiments
  - c1: Direction C1, Hessian signature analysis in n=4 cumulant coordinates
  - b1: Direction B1, n=4 cumulant-space surplus reformulation complexity

All tasks append machine-readable records to JSONL output.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from numpy.polynomial.hermite_e import hermeroots
from scipy.stats import unitary_group


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "data" / "first-proof" / "problem4-deepdive-results.jsonl"
EPS = 1e-12


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResultLogger:
    path: Path
    run_id: str
    script: str

    def append(self, payload: Dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts_utc": utc_now_iso(),
            "run_id": self.run_id,
            "script": self.script,
        }
        rec.update(payload)
        line = json.dumps(rec, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def poly_desc_from_root_coeffs(coeffs_n: Sequence[float]) -> np.ndarray:
    return np.concatenate(([1.0], np.asarray(coeffs_n, dtype=float)))


def coeffs_from_roots(roots: Sequence[float]) -> np.ndarray:
    return np.poly(np.asarray(roots, dtype=float))[1:].astype(float)


def roots_from_coeffs(coeffs_n: Sequence[float], tol: float = 1e-7) -> Optional[np.ndarray]:
    roots = np.roots(poly_desc_from_root_coeffs(coeffs_n))
    if np.max(np.abs(roots.imag)) > tol:
        return None
    rr = np.sort(roots.real.astype(float))
    if len(rr) > 1 and np.min(np.diff(rr)) < 1e-10:
        return None
    return rr


def is_real_rooted(coeffs_n: Sequence[float], tol: float = 1e-7) -> bool:
    roots = np.roots(poly_desc_from_root_coeffs(coeffs_n))
    return np.max(np.abs(roots.imag)) <= tol


def make_distinct_sorted(roots: Sequence[float], min_gap: float = 1e-5) -> np.ndarray:
    r = np.sort(np.asarray(roots, dtype=float))
    for i in range(1, len(r)):
        if r[i] - r[i - 1] < min_gap:
            r[i] = r[i - 1] + min_gap
    return r


def sample_roots_regime(n: int, rng: np.random.Generator, regime: str) -> Tuple[np.ndarray, np.ndarray]:
    if regime == "random":
        p = make_distinct_sorted(rng.normal(size=n) * rng.uniform(0.1, 10.0))
        q = make_distinct_sorted(rng.normal(size=n) * rng.uniform(0.1, 10.0))
        return p, q

    if regime == "scale_separated":
        p = make_distinct_sorted(rng.normal(size=n) * rng.uniform(0.3, 2.0))
        q = make_distinct_sorted(rng.normal(size=n) * rng.uniform(30.0, 300.0))
        return p, q

    if regime == "near_degenerate":
        p = np.sort(rng.normal(size=n) * rng.uniform(0.2, 4.0))
        q = np.sort(rng.normal(size=n) * rng.uniform(0.2, 4.0))
        idx_p = int(rng.integers(0, n - 1))
        idx_q = int(rng.integers(0, n - 1))
        eps_p = float(10.0 ** rng.uniform(-8.0, -4.5))
        eps_q = float(10.0 ** rng.uniform(-8.0, -4.5))
        p[idx_p + 1] = p[idx_p] + eps_p
        q[idx_q + 1] = q[idx_q] + eps_q
        return make_distinct_sorted(p, min_gap=1e-10), make_distinct_sorted(q, min_gap=1e-10)

    if regime == "hermite_random":
        t1 = float(rng.uniform(0.05, 5.0))
        p = make_distinct_sorted(np.sort(hermeroots([0] * n + [1])) * math.sqrt(t1))
        q = make_distinct_sorted(rng.normal(size=n) * rng.uniform(0.1, 8.0))
        return p, q

    raise ValueError(f"unknown regime: {regime}")


def normalize_roots_a2_minus1(roots: Sequence[float]) -> Optional[np.ndarray]:
    r = np.asarray(roots, dtype=float)
    r = r - np.mean(r)
    coeff = np.poly(r)
    a2 = float(coeff[2])
    if a2 >= -1e-14:
        return None
    scale = 1.0 / math.sqrt(-a2)
    rn = r * scale
    rn = make_distinct_sorted(rn, min_gap=1e-8)
    return rn


def mss_convolve(a_coeffs: Sequence[float], b_coeffs: Sequence[float], n: int) -> np.ndarray:
    c = np.zeros(n, dtype=float)
    fact_n = math.factorial(n)
    for k in range(1, n + 1):
        s = 0.0
        fact_nk = math.factorial(n - k)
        for i in range(k + 1):
            j = k - i
            ai = 1.0 if i == 0 else float(a_coeffs[i - 1])
            bj = 1.0 if j == 0 else float(b_coeffs[j - 1])
            w = (math.factorial(n - i) * math.factorial(n - j)) / (fact_n * fact_nk)
            s += w * ai * bj
        c[k - 1] = s
    return c


def score_field(roots: Sequence[float]) -> np.ndarray:
    r = np.asarray(roots, dtype=float)
    n = len(r)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        d = r[i] - np.delete(r, i)
        if np.min(np.abs(d)) < EPS:
            return np.full(n, np.nan)
        out[i] = np.sum(1.0 / d)
    return out


def phi_n(roots: Sequence[float]) -> float:
    s = score_field(roots)
    if np.any(np.isnan(s)):
        return float("inf")
    return float(np.dot(s, s))


def hermite_coeffs(n: int, t: float) -> np.ndarray:
    roots = np.sort(hermeroots([0] * n + [1])) * math.sqrt(max(t, 0.0))
    return coeffs_from_roots(roots)


def run_d1(args: argparse.Namespace, logger: ResultLogger) -> None:
    rng = np.random.default_rng(args.seed)
    regimes = ["random", "scale_separated", "near_degenerate", "hermite_random"]

    for n in range(args.n_min, args.n_max + 1):
        for regime in regimes:
            total = 0
            valid = 0
            real_root_fail = 0
            phi_fail = 0
            viol = 0
            ratio_min = float("inf")
            ratio_max = -float("inf")
            ratio_sum = 0.0

            self_valid = 0
            self_viol = 0
            self_ratio_min = float("inf")
            self_ratio_max = -float("inf")
            self_ratio_sum = 0.0

            norm_valid = 0
            norm_viol = 0
            norm_ratio_min = float("inf")
            norm_ratio_max = -float("inf")
            norm_ratio_sum = 0.0
            norm_fail = 0

            for _ in range(args.trials):
                total += 1
                p_roots, q_roots = sample_roots_regime(n, rng, regime)
                p_coeff = coeffs_from_roots(p_roots)
                q_coeff = coeffs_from_roots(q_roots)

                conv = mss_convolve(p_coeff, q_coeff, n)
                r_roots = roots_from_coeffs(conv, tol=args.real_tol)
                if r_roots is None:
                    real_root_fail += 1
                    continue

                phi_p = phi_n(p_roots)
                phi_q = phi_n(q_roots)
                phi_r = phi_n(r_roots)
                if not np.isfinite(phi_p) or not np.isfinite(phi_q) or not np.isfinite(phi_r):
                    phi_fail += 1
                    continue

                valid += 1
                ratio = phi_r / (phi_p * phi_q)
                ratio_min = min(ratio_min, ratio)
                ratio_max = max(ratio_max, ratio)
                ratio_sum += ratio
                if ratio > 1.0 + args.ratio_tol:
                    viol += 1

                p_norm = normalize_roots_a2_minus1(p_roots)
                q_norm = normalize_roots_a2_minus1(q_roots)
                if p_norm is None or q_norm is None:
                    norm_fail += 1
                else:
                    pnc = coeffs_from_roots(p_norm)
                    qnc = coeffs_from_roots(q_norm)
                    cn = mss_convolve(pnc, qnc, n)
                    rn = roots_from_coeffs(cn, tol=args.real_tol)
                    if rn is None:
                        norm_fail += 1
                    else:
                        phi_pn = phi_n(p_norm)
                        phi_qn = phi_n(q_norm)
                        phi_rn = phi_n(rn)
                        if (
                            np.isfinite(phi_pn)
                            and np.isfinite(phi_qn)
                            and np.isfinite(phi_rn)
                        ):
                            norm_valid += 1
                            nr = phi_rn / (phi_pn * phi_qn)
                            norm_ratio_min = min(norm_ratio_min, nr)
                            norm_ratio_max = max(norm_ratio_max, nr)
                            norm_ratio_sum += nr
                            if nr > 1.0 + args.ratio_tol:
                                norm_viol += 1
                        else:
                            norm_fail += 1

                conv_self = mss_convolve(p_coeff, p_coeff, n)
                r_self = roots_from_coeffs(conv_self, tol=args.real_tol)
                if r_self is not None:
                    phi_self = phi_n(r_self)
                    if np.isfinite(phi_self):
                        self_valid += 1
                        sr = phi_self / (phi_p * phi_p)
                        self_ratio_min = min(self_ratio_min, sr)
                        self_ratio_max = max(self_ratio_max, sr)
                        self_ratio_sum += sr
                        if sr > 1.0 + args.ratio_tol:
                            self_viol += 1

            rec = {
                "task": "D1",
                "result_type": "counterexample" if viol > 0 else "conjecture",
                "n": n,
                "regime": regime,
                "trials_total": total,
                "trials_valid": valid,
                "real_root_fail": real_root_fail,
                "phi_fail": phi_fail,
                "raw_violations": viol,
                "raw_violation_rate": (viol / valid) if valid else None,
                "raw_ratio_min": ratio_min if valid else None,
                "raw_ratio_mean": (ratio_sum / valid) if valid else None,
                "raw_ratio_max": ratio_max if valid else None,
                "self_trials_valid": self_valid,
                "self_violations": self_viol,
                "self_violation_rate": (self_viol / self_valid) if self_valid else None,
                "self_ratio_min": self_ratio_min if self_valid else None,
                "self_ratio_mean": (self_ratio_sum / self_valid) if self_valid else None,
                "self_ratio_max": self_ratio_max if self_valid else None,
                "normalized_trials_valid": norm_valid,
                "normalized_fail": norm_fail,
                "normalized_violations": norm_viol,
                "normalized_violation_rate": (norm_viol / norm_valid) if norm_valid else None,
                "normalized_ratio_min": norm_ratio_min if norm_valid else None,
                "normalized_ratio_mean": (norm_ratio_sum / norm_valid) if norm_valid else None,
                "normalized_ratio_max": norm_ratio_max if norm_valid else None,
                "claim": "phi(conv) <= phi(p)*phi(q)",
            }
            logger.append(rec)
            print(
                f"[D1] n={n} regime={regime} valid={valid}/{total} "
                f"raw_viol={viol} raw_ratio_max={rec['raw_ratio_max']} "
                f"norm_viol={norm_viol} norm_ratio_max={rec['normalized_ratio_max']}"
            )


def run_d2(args: argparse.Namespace, logger: ResultLogger) -> None:
    rng = np.random.default_rng(args.seed)
    t_grid = [float(x) for x in args.t_grid.split(",") if x.strip()]

    for n in range(args.n_min, args.n_max + 1):
        he_cache = {t: hermite_coeffs(n, t) for t in t_grid if t > 0.0}
        valid = 0
        monotone = 0
        nonmonotone = 0
        max_upward_step = 0.0
        ratio0_list = []
        ratioT_list = []
        slope_sum = 0.0
        slope_count = 0

        for _ in range(args.trials):
            p_roots, q_roots = sample_roots_regime(n, rng, "random")
            p_coeff = coeffs_from_roots(p_roots)
            q_coeff = coeffs_from_roots(q_roots)
            pq_coeff = mss_convolve(p_coeff, q_coeff, n)
            if not is_real_rooted(pq_coeff, tol=args.real_tol):
                continue

            ratios = []
            ok = True
            for t in t_grid:
                if t <= 0:
                    p_t = p_coeff
                    q_t = q_coeff
                    pq_t = pq_coeff
                else:
                    h = he_cache[t]
                    p_t = mss_convolve(p_coeff, h, n)
                    q_t = mss_convolve(q_coeff, h, n)
                    pq_t = mss_convolve(pq_coeff, h, n)
                rp = roots_from_coeffs(p_t, tol=args.real_tol)
                rq = roots_from_coeffs(q_t, tol=args.real_tol)
                rr = roots_from_coeffs(pq_t, tol=args.real_tol)
                if rp is None or rq is None or rr is None:
                    ok = False
                    break

                phi_p = phi_n(rp)
                phi_q = phi_n(rq)
                phi_r = phi_n(rr)
                if not np.isfinite(phi_p) or not np.isfinite(phi_q) or not np.isfinite(phi_r):
                    ok = False
                    break
                ratios.append(phi_r / (phi_p * phi_q))

            if not ok:
                continue

            valid += 1
            ratio0_list.append(ratios[0])
            ratioT_list.append(ratios[-1])

            local_mono = True
            for i in range(len(ratios) - 1):
                d = ratios[i + 1] - ratios[i]
                dt = t_grid[i + 1] - t_grid[i]
                slope_sum += d / dt
                slope_count += 1
                if d > args.monotone_tol:
                    local_mono = False
                    max_upward_step = max(max_upward_step, d)

            if local_mono:
                monotone += 1
            else:
                nonmonotone += 1

        rec = {
            "task": "D2",
            "result_type": "conjecture",
            "n": n,
            "t_grid": t_grid,
            "trials_requested": args.trials,
            "trials_valid": valid,
            "monotone_nonincreasing_count": monotone,
            "nonmonotone_count": nonmonotone,
            "monotone_rate": (monotone / valid) if valid else None,
            "max_upward_step": max_upward_step,
            "ratio_t0_mean": float(np.mean(ratio0_list)) if ratio0_list else None,
            "ratio_tmax_mean": float(np.mean(ratioT_list)) if ratioT_list else None,
            "mean_dR_dt": (slope_sum / slope_count) if slope_count else None,
            "ratio_definition": "R(t)=phi((p conv q)_t)/(phi(p_t)*phi(q_t))",
        }
        logger.append(rec)
        print(
            f"[D2] n={n} valid={valid} mono={monotone} nonmono={nonmonotone} "
            f"max_up={max_upward_step:.3e}"
        )


def run_d3(args: argparse.Namespace, logger: ResultLogger) -> None:
    s, t, u, v = sp.symbols("s t u v", positive=True, real=True)

    phi_p = 18 * s**2 / (4 * s**3 - 27 * u**2)
    phi_q = 18 * t**2 / (4 * t**3 - 27 * v**2)
    phi_c = 18 * (s + t) ** 2 / (4 * (s + t) ** 3 - 27 * (u + v) ** 2)

    expr = sp.together(phi_p * phi_q - phi_c)
    num, den = sp.fraction(expr)
    p_num = sp.Poly(sp.expand(num), s, t, u, v)

    f_expr = sp.lambdify((s, t, u, v), expr, modules="numpy")
    rng = np.random.default_rng(args.seed)
    valids = 0
    violations = 0
    min_val = float("inf")

    remaining = args.numeric_samples
    batch = args.batch_size
    done = 0
    while remaining > 0:
        m = min(batch, remaining)
        sv = rng.uniform(0.1, 4.0, size=m)
        tv = rng.uniform(0.1, 4.0, size=m)
        umax = np.sqrt((4.0 * sv**3) / 27.0) * 0.99
        vmax = np.sqrt((4.0 * tv**3) / 27.0) * 0.99
        uv = rng.uniform(-umax, umax)
        vv = rng.uniform(-vmax, vmax)

        den_p = 4.0 * sv**3 - 27.0 * uv**2
        den_q = 4.0 * tv**3 - 27.0 * vv**2
        den_c = 4.0 * (sv + tv) ** 3 - 27.0 * (uv + vv) ** 2
        mask = (den_p > 0.0) & (den_q > 0.0) & (den_c > 0.0)

        if np.any(mask):
            vals = np.asarray(f_expr(sv[mask], tv[mask], uv[mask], vv[mask]), dtype=float)
            valids += int(vals.size)
            min_val = min(min_val, float(np.min(vals)))
            violations += int(np.sum(vals < -args.numeric_tol))

        done += m
        remaining -= m
        if done % max(batch, args.numeric_samples // 10) == 0:
            print(f"[D3] progress {done}/{args.numeric_samples}")

    rec = {
        "task": "D3",
        "result_type": "conjecture",
        "symbolic_num_degree": p_num.total_degree(),
        "symbolic_num_terms": len(p_num.as_dict()),
        "symbolic_den_factorized": str(sp.factor(den)),
        "numeric_samples_valid": valids,
        "numeric_violations": violations,
        "numeric_min_value": min_val if valids else None,
        "inequality_tested": "phi_3(p conv q) <= phi_3(p)*phi_3(q)",
    }
    logger.append(rec)
    print(
        f"[D3] valid={valids} violations={violations} "
        f"min(phi_p*phi_q-phi_conv)={rec['numeric_min_value']}"
    )


def run_a2(args: argparse.Namespace, logger: ResultLogger) -> None:
    rng = np.random.default_rng(args.seed)
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]

    for n in n_values:
        ip_naive = []
        ip_weighted = []
        residual_naive = []
        residual_weighted = []
        rel_residual_naive = []
        rel_residual_weighted = []
        valid = 0

        for trial in range(args.trials):
            lam = make_distinct_sorted(rng.normal(size=n) * rng.uniform(0.2, 3.5), min_gap=1e-6)
            mu = make_distinct_sorted(rng.normal(size=n) * rng.uniform(0.2, 3.5), min_gap=1e-6)
            A = np.diag(lam)
            B = np.diag(mu)

            for _ in range(args.samples):
                U = unitary_group.rvs(n, random_state=rng)
                C = A + U @ B @ U.conj().T
                gamma, V = np.linalg.eigh(C)
                gamma = gamma.real

                S = score_field(gamma)
                if np.any(np.isnan(S)):
                    continue

                SA = np.array([np.sum(1.0 / (gamma[k] - lam)) for k in range(n)], dtype=float)
                SB = np.array([np.sum(1.0 / (gamma[k] - mu)) for k in range(n)], dtype=float)

                # Weighted decomposition candidate:
                # A-part in A basis, B-part in B basis after undoing U.
                weights_A = np.abs(V) ** 2
                SA_w = np.array(
                    [np.sum(weights_A[:, k] / (gamma[k] - lam)) for k in range(n)],
                    dtype=float,
                )

                W = U.conj().T @ V
                weights_B = np.abs(W) ** 2
                SB_w = np.array(
                    [np.sum(weights_B[:, k] / (gamma[k] - mu)) for k in range(n)],
                    dtype=float,
                )

                ip_naive.append(float(np.dot(SA, SB)))
                ip_weighted.append(float(np.dot(SA_w, SB_w)))

                rn = float(np.linalg.norm(S - (SA + SB)))
                rw = float(np.linalg.norm(S - (SA_w + SB_w)))
                s_norm = float(np.linalg.norm(S))
                residual_naive.append(rn)
                residual_weighted.append(rw)
                if s_norm > 0:
                    rel_residual_naive.append(rn / s_norm)
                    rel_residual_weighted.append(rw / s_norm)
                valid += 1

            if (trial + 1) % max(1, args.trials // 10) == 0:
                print(f"[A2] n={n} progress trial={trial + 1}/{args.trials}")

        rec = {
            "task": "A2",
            "result_type": "conjecture",
            "n": n,
            "trials": args.trials,
            "samples_per_trial": args.samples,
            "valid_samples": valid,
            "mean_inner_naive": float(np.mean(ip_naive)) if ip_naive else None,
            "std_inner_naive": float(np.std(ip_naive)) if ip_naive else None,
            "mean_inner_weighted": float(np.mean(ip_weighted)) if ip_weighted else None,
            "std_inner_weighted": float(np.std(ip_weighted)) if ip_weighted else None,
            "mean_residual_naive_l2": float(np.mean(residual_naive)) if residual_naive else None,
            "mean_residual_weighted_l2": float(np.mean(residual_weighted)) if residual_weighted else None,
            "mean_rel_residual_naive": float(np.mean(rel_residual_naive)) if rel_residual_naive else None,
            "mean_rel_residual_weighted": float(np.mean(rel_residual_weighted)) if rel_residual_weighted else None,
            "decomposition_note": "S compared against SA+SB with naive and eigenvector-weighted candidates",
        }
        logger.append(rec)
        print(
            f"[A2] n={n} valid={valid} mean_ip_weighted={rec['mean_inner_weighted']} "
            f"mean_rel_res_w={rec['mean_rel_residual_weighted']}"
        )


def quartic_kappa_from_roots(roots: Sequence[float]) -> np.ndarray:
    roots = np.asarray(roots, dtype=float)
    roots = roots - np.mean(roots)
    coeff = np.poly(roots)
    a2 = float(coeff[2])
    a3 = float(coeff[3])
    a4 = float(coeff[4])
    k2 = a2
    k3 = a3
    k4 = a4 + (a2 * a2) / 12.0
    return np.array([k2, k3, k4], dtype=float)


def inv_phi_from_kappa_n4(kappa: Sequence[float], tol: float = 1e-7) -> Optional[float]:
    k2, k3, k4 = [float(x) for x in kappa]
    a2 = k2
    a3 = k3
    a4 = k4 - (k2 * k2) / 12.0
    roots = np.roots(np.array([1.0, 0.0, a2, a3, a4], dtype=float))
    if np.max(np.abs(roots.imag)) > tol:
        return None
    rr = np.sort(roots.real)
    if np.min(np.diff(rr)) < 1e-8:
        return None
    phi = phi_n(rr)
    if not np.isfinite(phi) or phi <= 0.0:
        return None
    return 1.0 / phi


def numerical_hessian_3d(f, x: np.ndarray, dx: float) -> Optional[np.ndarray]:
    n = 3
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = dx
            ej[j] = dx
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            if fpp is None or fpm is None or fmp is None or fmm is None:
                return None
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * dx * dx)
            H[j, i] = H[i, j]
    return H


def sample_interior_kappa_n4(rng: np.random.Generator) -> np.ndarray:
    roots = make_distinct_sorted(rng.normal(size=4) * rng.uniform(0.3, 3.0), min_gap=0.05)
    roots = roots - np.mean(roots)
    return quartic_kappa_from_roots(roots)


def sample_boundary_kappa_n4(rng: np.random.Generator) -> np.ndarray:
    a = float(rng.uniform(0.5, 2.5))
    eps = float(10.0 ** rng.uniform(-5.5, -2.8))
    roots = np.array([-a, -eps, eps, a], dtype=float)
    roots += rng.normal(scale=0.01, size=4)
    roots = roots - np.mean(roots)
    roots = make_distinct_sorted(roots, min_gap=1e-7)
    return quartic_kappa_from_roots(roots)


def eig_signature(vals: np.ndarray, eps: float = 1e-8) -> Tuple[int, int, int]:
    pos = int(np.sum(vals > eps))
    neg = int(np.sum(vals < -eps))
    zero = int(len(vals) - pos - neg)
    return pos, neg, zero


def run_c1(args: argparse.Namespace, logger: ResultLogger) -> None:
    rng = np.random.default_rng(args.seed)
    dx = args.hessian_dx

    signatures_interior: List[Tuple[int, int, int]] = []
    signatures_boundary: List[Tuple[int, int, int]] = []
    eigvals_interior: List[List[float]] = []
    eigvals_boundary: List[List[float]] = []

    attempts = 0
    while len(signatures_interior) < args.interior_points and attempts < args.max_attempts:
        attempts += 1
        x = sample_interior_kappa_n4(rng)
        H = numerical_hessian_3d(inv_phi_from_kappa_n4, x, dx)
        if H is None:
            continue
        ev = np.linalg.eigvalsh(H)
        sig = eig_signature(ev)
        signatures_interior.append(sig)
        eigvals_interior.append([float(v) for v in ev])
        if len(signatures_interior) % max(1, args.interior_points // 10) == 0:
            print(f"[C1] interior progress {len(signatures_interior)}/{args.interior_points}")

    attempts = 0
    while len(signatures_boundary) < args.boundary_points and attempts < args.max_attempts:
        attempts += 1
        x = sample_boundary_kappa_n4(rng)
        H = numerical_hessian_3d(inv_phi_from_kappa_n4, x, dx)
        if H is None:
            continue
        ev = np.linalg.eigvalsh(H)
        sig = eig_signature(ev)
        signatures_boundary.append(sig)
        eigvals_boundary.append([float(v) for v in ev])
        if len(signatures_boundary) % max(1, args.boundary_points // 10) == 0:
            print(f"[C1] boundary progress {len(signatures_boundary)}/{args.boundary_points}")

    def sig_hist(sig_list: List[Tuple[int, int, int]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in sig_list:
            key = f"{s[0]}+/{s[1]}-/{s[2]}0"
            out[key] = out.get(key, 0) + 1
        return out

    rec = {
        "task": "C1",
        "result_type": "conjecture",
        "n": 4,
        "hessian_dx": dx,
        "interior_points": len(signatures_interior),
        "boundary_points": len(signatures_boundary),
        "signature_hist_interior": sig_hist(signatures_interior),
        "signature_hist_boundary": sig_hist(signatures_boundary),
        "interior_eig_min": float(np.min(eigvals_interior)) if eigvals_interior else None,
        "interior_eig_max": float(np.max(eigvals_interior)) if eigvals_interior else None,
        "boundary_eig_min": float(np.min(eigvals_boundary)) if eigvals_boundary else None,
        "boundary_eig_max": float(np.max(eigvals_boundary)) if eigvals_boundary else None,
        "note": "Hessian of inv_phi_4 in (k2,k3,k4) cumulant coordinates",
    }
    logger.append(rec)
    print(
        f"[C1] interior={len(signatures_interior)} boundary={len(signatures_boundary)} "
        f"hist_interior={rec['signature_hist_interior']}"
    )


def disc4(a2, a3, a4):
    return 256 * a4**3 - 128 * a2**2 * a4**2 + 144 * a2 * a3**2 * a4 - 27 * a3**4 + 16 * a2**4 * a4 - 4 * a2**3 * a3**2


def inv_phi4_expr(a2, a3, a4):
    disc = disc4(a2, a3, a4)
    f1 = a2**2 + 12 * a4
    f2 = 2 * a2**3 - 8 * a2 * a4 + 9 * a3**2
    return -disc / (4 * f1 * f2)


def run_b1(args: argparse.Namespace, logger: ResultLogger) -> None:
    a3, a4, b3, b4 = sp.symbols("a3 a4 b3 b4", real=True)
    inv_p_coeff = sp.together(inv_phi4_expr(-1, a3, a4))
    inv_q_coeff = sp.together(inv_phi4_expr(-1, b3, b4))
    inv_r_coeff = sp.together(inv_phi4_expr(-2, a3 + b3, a4 + b4 + sp.Rational(1, 6)))
    surplus_coeff = sp.together(inv_r_coeff - inv_p_coeff - inv_q_coeff)
    num_coeff, den_coeff = sp.fraction(surplus_coeff)
    poly_coeff = sp.Poly(sp.expand(num_coeff), a3, a4, b3, b4)

    k3, k4, l3, l4 = sp.symbols("k3 k4 l3 l4", real=True)
    a2k = -1
    a3k = k3
    a4k = k4 - sp.Rational(1, 12)
    b2k = -1
    b3k = l3
    b4k = l4 - sp.Rational(1, 12)
    c2k = -2
    c3k = k3 + l3
    c4k = (k4 + l4) - sp.Rational(1, 3)

    inv_pk = sp.together(inv_phi4_expr(a2k, a3k, a4k))
    inv_qk = sp.together(inv_phi4_expr(b2k, b3k, b4k))
    inv_rk = sp.together(inv_phi4_expr(c2k, c3k, c4k))
    surplus_k = sp.together(inv_rk - inv_pk - inv_qk)
    num_k, den_k = sp.fraction(surplus_k)
    poly_k = sp.Poly(sp.expand(num_k), k3, k4, l3, l4)

    sym_swap = (
        sp.expand(
            num_k.subs({k3: l3, k4: l4, l3: k3, l4: k4}, simultaneous=True) - num_k
        )
        == 0
    )
    sym_reflect = sp.expand(num_k.subs({k3: -k3, l3: -l3}) - num_k) == 0

    rec = {
        "task": "B1",
        "result_type": "symbolic_expression",
        "coefficient_num_degree": poly_coeff.total_degree(),
        "coefficient_num_terms": len(poly_coeff.as_dict()),
        "coefficient_den_degree": sp.Poly(sp.expand(den_coeff), a3, a4, b3, b4).total_degree(),
        "cumulant_num_degree": poly_k.total_degree(),
        "cumulant_num_terms": len(poly_k.as_dict()),
        "cumulant_den_degree": sp.Poly(sp.expand(den_k), k3, k4, l3, l4).total_degree(),
        "cumulant_swap_symmetry": sym_swap,
        "cumulant_reflection_symmetry": sym_reflect,
        "claim": "compare complexity of n=4 surplus numerator in coefficient vs cumulant coordinates",
    }
    logger.append(rec)
    print(
        f"[B1] coeff terms={rec['coefficient_num_terms']} "
        f"cumulant terms={rec['cumulant_num_terms']} "
        f"swap_sym={sym_swap} reflect_sym={sym_reflect}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Problem 4 deep-dive tasks.")
    p.add_argument("--results", default=str(DEFAULT_RESULTS), help="JSONL output path")
    p.add_argument("--run-id", default=None, help="Optional run identifier")
    p.add_argument("--seed", type=int, default=20260213, help="RNG seed")
    sub = p.add_subparsers(dest="task", required=True)

    p_d1 = sub.add_parser("d1", help="Direction D1: extended log-subadd verification")
    p_d1.add_argument("--n-min", type=int, default=3)
    p_d1.add_argument("--n-max", type=int, default=8)
    p_d1.add_argument("--trials", type=int, default=10000)
    p_d1.add_argument("--real-tol", type=float, default=1e-7)
    p_d1.add_argument("--ratio-tol", type=float, default=1e-10)

    p_d2 = sub.add_parser("d2", help="Direction D2: flow-ratio monotonicity")
    p_d2.add_argument("--n-min", type=int, default=3)
    p_d2.add_argument("--n-max", type=int, default=7)
    p_d2.add_argument("--trials", type=int, default=4000)
    p_d2.add_argument("--real-tol", type=float, default=1e-7)
    p_d2.add_argument("--monotone-tol", type=float, default=1e-10)
    p_d2.add_argument("--t-grid", default="0,0.05,0.1,0.2,0.5,1,2,5")

    p_d3 = sub.add_parser("d3", help="Direction D3: n=3 symbolic warm-up")
    p_d3.add_argument("--numeric-samples", type=int, default=200000)
    p_d3.add_argument("--batch-size", type=int, default=20000)
    p_d3.add_argument("--numeric-tol", type=float, default=1e-12)

    p_a2 = sub.add_parser("a2", help="Direction A2: score decomposition")
    p_a2.add_argument("--n-values", default="4,5,6")
    p_a2.add_argument("--trials", type=int, default=180)
    p_a2.add_argument("--samples", type=int, default=160)

    p_c1 = sub.add_parser("c1", help="Direction C1: Hessian signature")
    p_c1.add_argument("--interior-points", type=int, default=260)
    p_c1.add_argument("--boundary-points", type=int, default=260)
    p_c1.add_argument("--hessian-dx", type=float, default=2e-4)
    p_c1.add_argument("--max-attempts", type=int, default=200000)

    sub.add_parser("b1", help="Direction B1: cumulant symbolic reformulation")
    return p


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        # Helpful for nohup logs.
        import sys

        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args()
    run_id = args.run_id or f"p4deep-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{args.task}"
    logger = ResultLogger(Path(args.results), run_id=run_id, script=Path(__file__).name)

    logger.append(
        {
            "task": args.task,
            "event": "start",
            "argv": " ".join(os.sys.argv),
            "seed": args.seed,
        }
    )
    print(f"[START] task={args.task} run_id={run_id} results={args.results}")

    if args.task == "d1":
        run_d1(args, logger)
    elif args.task == "d2":
        run_d2(args, logger)
    elif args.task == "d3":
        run_d3(args, logger)
    elif args.task == "a2":
        run_a2(args, logger)
    elif args.task == "c1":
        run_c1(args, logger)
    elif args.task == "b1":
        run_b1(args, logger)
    else:
        raise ValueError(f"unsupported task {args.task}")

    logger.append({"task": args.task, "event": "done"})
    print(f"[DONE] task={args.task} run_id={run_id}")


if __name__ == "__main__":
    main()
