#!/usr/bin/env python3
"""Task 2: score decomposition / Cauchy-Schwarz route diagnostics.

Numerically test whether the score vector S(p⊞q) can be represented as
an affine/linear projection of S(p), S(q) (sorted-root coordinates).
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "data" / "first-proof" / "problem4-score-projection-results.json"


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


def coeffs_from_roots(roots):
    return np.poly(np.asarray(roots, dtype=float))[1:].astype(float)


def roots_from_coeffs(coeffs, tol=1e-8):
    r = np.roots(np.concatenate([[1.0], np.asarray(coeffs, dtype=float)]))
    if np.max(np.abs(r.imag)) > tol:
        return None
    rr = np.sort(r.real.astype(float))
    if len(rr) > 1 and np.min(np.diff(rr)) < 1e-10:
        return None
    return rr


def score(roots):
    r = np.asarray(roots, dtype=float)
    n = len(r)
    s = np.zeros(n, dtype=float)
    for i in range(n):
        d = r[i] - np.delete(r, i)
        if np.min(np.abs(d)) < 1e-12:
            return None
        s[i] = np.sum(1.0 / d)
    return s


def build_dataset(n: int, trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    S_p, S_q, S_c = [], [], []
    for _ in range(trials):
        p = np.sort(rng.normal(size=n) * rng.uniform(0.2, 4.0))
        q = np.sort(rng.normal(size=n) * rng.uniform(0.2, 4.0))
        for i in range(1, n):
            if p[i] - p[i - 1] < 1e-5:
                p[i] = p[i - 1] + 1e-5
            if q[i] - q[i - 1] < 1e-5:
                q[i] = q[i - 1] + 1e-5

        c_coeff = mss_convolve(coeffs_from_roots(p), coeffs_from_roots(q), n)
        c = roots_from_coeffs(c_coeff)
        if c is None:
            continue

        spv = score(p)
        sqv = score(q)
        scv = score(c)
        if spv is None or sqv is None or scv is None:
            continue

        S_p.append(spv)
        S_q.append(sqv)
        S_c.append(scv)
    return np.asarray(S_p), np.asarray(S_q), np.asarray(S_c)


def fit_global_affine(S_p: np.ndarray, S_q: np.ndarray, S_c: np.ndarray) -> Dict:
    """Fit Sc = W @ [Sp;Sq;1]."""
    n = S_p.shape[1]
    X = np.concatenate([S_p, S_q, np.ones((S_p.shape[0], 1))], axis=1)  # (m,2n+1)
    Y = S_c  # (m,n)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (2n+1,n)
    pred = X @ W
    err = pred - Y
    rel = np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(Y, axis=1), 1e-12)
    ss_res = float(np.sum(err**2))
    y_center = Y - np.mean(Y, axis=0, keepdims=True)
    ss_tot = float(np.sum(y_center**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    A = W[:n, :].T
    B = W[n : 2 * n, :].T
    c = W[-1, :]
    return {
        "samples": int(S_p.shape[0]),
        "r2": float(r2),
        "rel_err_mean": float(np.mean(rel)),
        "rel_err_p95": float(np.percentile(rel, 95)),
        "rel_err_max": float(np.max(rel)),
        "fro_A_plus_B_minus_I": float(np.linalg.norm(A + B - np.eye(n), ord="fro")),
        "bias_norm": float(np.linalg.norm(c)),
    }


def fit_diag_projection(S_p: np.ndarray, S_q: np.ndarray, S_c: np.ndarray) -> Dict:
    """Coordinate-wise Sc_i = alpha_i Sp_i + (1-alpha_i)Sq_i + beta_i."""
    m, n = S_p.shape
    alpha = np.zeros(n)
    beta = np.zeros(n)
    pred = np.zeros_like(S_c)

    for i in range(n):
        # target: y = a*(sp-sq) + sq + b
        X = np.column_stack([S_p[:, i] - S_q[:, i], np.ones(m)])
        y = S_c[:, i] - S_q[:, i]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha[i] = float(coef[0])
        beta[i] = float(coef[1])
        pred[:, i] = alpha[i] * S_p[:, i] + (1.0 - alpha[i]) * S_q[:, i] + beta[i]

    err = pred - S_c
    rel = np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(S_c, axis=1), 1e-12)
    return {
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "beta_norm": float(np.linalg.norm(beta)),
        "rel_err_mean": float(np.mean(rel)),
        "rel_err_p95": float(np.percentile(rel, 95)),
        "rel_err_max": float(np.max(rel)),
    }


def fit_best_scalar_per_sample(S_p: np.ndarray, S_q: np.ndarray, S_c: np.ndarray) -> Dict:
    """Per-sample best scalar blend Sc ~ a Sp + (1-a)Sq."""
    D = S_p - S_q
    R = S_c - S_q
    num = np.sum(D * R, axis=1)
    den = np.sum(D * D, axis=1)
    alpha = np.where(den > 1e-14, num / den, 0.5)
    pred = alpha[:, None] * S_p + (1.0 - alpha[:, None]) * S_q
    err = pred - S_c
    rel = np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(S_c, axis=1), 1e-12)
    return {
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_p05": float(np.percentile(alpha, 5)),
        "alpha_p95": float(np.percentile(alpha, 95)),
        "rel_err_mean": float(np.mean(rel)),
        "rel_err_p95": float(np.percentile(rel, 95)),
        "rel_err_max": float(np.max(rel)),
    }


def run(n_values=(3, 4, 5, 6), trials=120000, seed=20260213):
    all_results: Dict[str, Dict] = {}
    for n in n_values:
        S_p, S_q, S_c = build_dataset(n=n, trials=trials, seed=seed + n)
        # Split train/test for fit diagnostics.
        m = S_p.shape[0]
        split = int(0.8 * m)
        tr = slice(0, split)
        te = slice(split, m)

        fit_train = fit_global_affine(S_p[tr], S_q[tr], S_c[tr])
        fit_test = fit_global_affine(S_p[te], S_q[te], S_c[te])
        diag_test = fit_diag_projection(S_p[te], S_q[te], S_c[te])
        scalar_test = fit_best_scalar_per_sample(S_p[te], S_q[te], S_c[te])

        all_results[str(n)] = {
            "samples_total": int(m),
            "samples_train": int(split),
            "samples_test": int(m - split),
            "global_affine_train": fit_train,
            "global_affine_test": fit_test,
            "diag_projection_test": diag_test,
            "best_scalar_per_sample_test": scalar_test,
        }
        print(
            f"[n={n}] m={m} affine_test_rel_mean={fit_test['rel_err_mean']:.3e} "
            f"diag_rel_mean={diag_test['rel_err_mean']:.3e} "
            f"scalar_rel_mean={scalar_test['rel_err_mean']:.3e}"
        )

    out = {
        "task": "Task2_score_decomposition",
        "trials_requested": trials,
        "results_by_n": all_results,
        "conclusion_hint": "Low error would support linear projection route; high error falsifies simple linear/affine models.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", default="3,4,5,6")
    ap.add_argument("--trials", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=20260213)
    args = ap.parse_args()
    nvals = tuple(int(x) for x in args.n_values.split(",") if x.strip())
    run(n_values=nvals, trials=args.trials, seed=args.seed)
