# Problem 6 Cycle 9 Codex Verification

Date: 2026-02-13
Agent: Codex
Base: `data/first-proof/problem6-bridge-b-formalization.md` (Cycle 9)

Artifacts:
- Script: `scripts/verify-p6-cycle9-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle9-results.json`

## Executive Summary

- Total runs: 148 (963 steps)
- All completed: True
- Lemma 8 (Pi<=I): True
- Lemma 8 (F<=I-M): True
- Lemma 8 (f_j<=1-lam_j violations): 0
- Lemma 8 (||C_t(v)||<=1 violations): 0
- Lemma 11 (rank violations): 0
- Lemma 12 (pigeonhole violations): 0
- Sparse Dichotomy applies: 63/148
- **Strong Dichotomy holds: True** (violations: 0)

## Lemma 8: Rayleigh-Monotonicity Matrix Bound

Pi_{I_0} <= I at all steps: **True**
F_t <= I - M_t at all steps: **True**
f_j <= 1-lambda_j violations: **0**
||C_t(v)|| <= 1 violations: **0**

## Lemma 9: Cross-Degree Bound

verified in C8 (78619 rows, 0 violations). Not re-run here.

## Lemma 10: Isolation

trivial (deg_S=0 => normY=0). Verified via isolation counts.

## Lemma 11: Rank of Barrier Contribution

rank(Y_t(v)) = deg_S(v) violations: **0**

## Lemma 12: Projection Pigeonhole

min_v u_j^T C_t(v) u_j <= (1-lam_j)/r_t violations: **0**

## Sparse Dichotomy

Applies (Delta < 3/eps - 1): 63/148 runs (42.6%)

## Strong Dichotomy

At every step, either isolation (deg_S=0 exists) or dbar < 1: **True**
Counterexamples: 0

## Dense-Case Probes

Steps where isolation fails: 134
Max dbar at non-isolation steps: 0.720000
dbar quantiles at non-isolation steps: {'q50': 0.4500027323613361, 'q90': 0.6600365751371359, 'q95': 0.7024974358864097, 'q99': 0.717730042016805}
Dense probes (all dominated, eigenspace decomposed): 134
All min-dangerous-vertex feasible: True

## Bottom Line

All five lemmas (8-12) verified with **0 violations**.
**Strong Dichotomy holds on entire test suite** — no step has both isolation failure AND dbar >= 1.
At the 134 non-isolation steps, max dbar = 0.720000 < 1.
