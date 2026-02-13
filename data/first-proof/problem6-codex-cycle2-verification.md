# Problem 6 Cycle 2 Codex Verification

Date: 2026-02-13
Agent: Codex
Status: Completed

## Scope

This pass implements and verifies all requested Cycle 2 tasks:

1. Verify the claimed `K_n` eigenstructure derivation chain.
2. Stress-test `K_n` extremality at larger `n` with ER/Barbell comparisons.
3. Investigate the `ER_60_p0.5, eps=0.5, t=6` outlier and leverage non-uniformity.
4. Explore the Schur/majorization path with at least one partial result.

## Artifacts

- Script: `scripts/verify-p6-cycle2-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle2-results.json`

## Command Run

```bash
python3 scripts/verify-p6-cycle2-codex.py \
  --out-json data/first-proof/problem6-codex-cycle2-results.json
```

## Method Notes

For large `n`, the script uses a low-rank formulation of the same greedy score/traces
instead of dense `n x n` matrix inversion at every candidate. This is still exact for the
simulated steps (no approximation in the linear algebra identity), and is cross-checked
against `scripts/verify-p6-cycle2-logdet-dbar.py` on `K_40` and the `ER_60` outlier.

Cross-check result:
- max `|dbar_lowrank - dbar_dense| = 2.776e-16`
- max `|ratio_lowrank - ratio_dense| = 7.772e-16`

## Task 1: Verify `K_n` Derivation

Checked all four claims numerically over:
- `n in {20, 40, 80, 120}`
- `eps in {0.2, 0.3, 0.5}`
- `t` up to `floor(eps*n/3)` (with valid barrier)

Results:
- `M_t = (1/n)L_{K_t}` eigenstructure matched exactly to numerical precision.
  - max nonzero-eigen error vs `t/n`: `3.608e-16`
  - max zero-eigen error: `2.498e-16`
  - multiplicity mismatches: `0`
- `F_t = (1/n)L_{K_{t,n-t}}` projection traces matched:
  - max `|tr(P_nz F_t) - (t-1)(n-t)/n| = 3.553e-15`
  - max `|tr(P_rest F_t) - (t+1)(n-t)/n| = 7.105e-15`
- `tr(B_t F_t)` formula matched:
  - max error: `7.105e-14`
- final closed form
  - `dbar = (t-1)/(n eps - t) + (t+1)/(n eps)`
  - max numerical error: `5.551e-16`

Algebraic simplification was also checked exactly over rational test cases
(`eps = 1/5, 3/10, 1/2`) with exact `Fraction` arithmetic: PASS.

## Task 2: Large-`n` Stress Test (`n=200,500,1000`)

Setup:
- `eps in {0.2, 0.3, 0.5}`
- Graphs per `n`: `K_n` (analytic exact baseline), `ER_{n,p=0.5}` (2 reps), `Barbell_{n/2}`
- Greedy horizon window for non-`K_n`: first `t <= 8` steps

Summary (max over tested graphs/eps):

- `n=200`
  - all `t>=1`: max ratio `dbar/dbar_Kn = 1.018590` (1.859% overshoot), at `ER_200_p0.5_r0, eps=0.3, t=1`
  - `t>=2`: max ratio `1.002290` (0.229% overshoot)
- `n=500`
  - all `t>=1`: max ratio `1.011138` (1.114% overshoot), at `ER_500_p0.5_r1, eps=0.5, t=5`
  - `t>=2`: same `1.011138`
- `n=1000`
  - all `t>=1`: max ratio `1.020240` (2.024% overshoot), at `ER_1000_p0.5_r1, eps=0.2, t=1`
  - `t>=2`: max ratio `1.000605` (0.060% overshoot), at `ER_1000_p0.5_r0, eps=0.5, t=5`

Answer to “does the 0.5% overshoot shrink with n?”
- Not monotonically if you include `t=1` (single-step fluctuations dominate).
- In the more relevant `t>=2` window (where the original `t=6` outlier lives), overshoot is small,
  and at `n=1000` it drops to `0.060%` in this sample.

## Task 3: ER Outlier Investigation

Reproduced the exact outlier instance from Claude's RNG order:
- `ER_60_p0.5`, `eps=0.5`, `t=6`
- observed ratio: `1.004657`
- `dbar=0.443723`, `dbar_Kn=0.441667`

Leverage non-uniformity comparison (`I0` leverage-degree stats):
- Outlier ER:
  - `m0=60`, mean `ell=1.9667`, var `0.019887`, CV `0.071706`
- `K_60`:
  - `m0=60`, mean `ell=1.9667`, var ~`0`, CV ~`0`

Correlation check across the Cycle-2 suite (`K`, `Barbell`, `ER` families, eps grid):
- corr(`max(0, ratio-1)`, `ell_cv`) = `0.1145`
- corr(`max(0, ratio-1)`, `ell_var`) = `0.0425`

Interpretation: leverage non-uniformity is consistent with the outlier direction
(ER is visibly less uniform than `K_n`), but by itself is only a weak global predictor.

## Task 4: Schur/Majorization Probe

Partial result (formal and verified numerically):
- If `M_t = 0`, then `B_t = (1/eps)I`, so
  `dbar = tr(B_t F_t)/r_t = tr(F_t)/(eps r_t)`.
- Therefore `dbar` is independent of how leverage mass is distributed across eigendirections;
  in this regime, a Schur-convex “uniform maximizes dbar” statement is degenerate/flat.

Numeric check (`200` random trials with fixed `tr(F_t), eps, r_t`):
- max absolute difference between two different weight profiles: `8.882e-16`.

Additional sanity check for nonzero spectrum:
- With fixed eigenvalues and only a fixed total mass constraint, objective
  `sum_i w_i/(eps-lambda_i)` is linear in `w`, and extreme concentration can exceed uniform.
- Example in script:
  - extreme-top: `4.166667`
  - uniform: `2.815598`
  - extreme-bottom: `2.000000`

So a naive Schur-convex uniform-max argument does not hold without stronger structural constraints.

## Bottom Line

- The `K_n` derivation chain is verified algebraically and numerically to machine precision.
- The `ER_60` outlier is reproduced exactly.
- Large-`n` stress tests do not show a simple monotone trend in all steps, but in `t>=2` the
  overshoot is small and near-zero at `n=1000` in this sampled set.
- For Schur/majorization, the `M_t=0` case is closed (flat objective), and nonzero-
  spectrum behavior indicates additional constraints are needed for a uniform-extremal theorem.
