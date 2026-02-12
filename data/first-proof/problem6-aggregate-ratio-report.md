# Problem 6: Aggregate-Ratio Deep Dive

Date: 2026-02-12
Author: Codex

## Goal

Press on the post-D target by studying the aggregate ratio certificate directly:

- `dbar_t := avg_v tr(Y_t(v))`
- `gbar_t := avg_v tr(Y_t(v))/||Y_t(v)||`
- `ratio_t := dbar_t / gbar_t`

with deterministic implication:

`min_v ||Y_t(v)|| <= ratio_t`.

So `ratio_t < 1` is a sufficient step certificate.

## Script

- `scripts/verify-p6-gpl-h-aggregate-ratio.py`

Command run:

`python3 scripts/verify-p6-gpl-h-aggregate-ratio.py --nmax 48 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --c-step 0.5 --seed 20260212 --output data/first-proof/problem6-aggregate-ratio-results.json`

## Main findings

From `data/first-proof/problem6-aggregate-ratio-results.json`:

- Case-2b instances: `216`
- Step rows: `454`
- Nontrivial rows (`min_score>0`): `178`

Ratio behavior:

- all rows: `ratio_max = 1.428571`
- nontrivial rows: `ratio_max = 0.740741`
- nontrivial ratio failures (`ratio>=1`): `0`

Crucial empirical pattern:

> Every observed row with `ratio>=1` has `min_score=0` (trivial good step).

Equivalently in the tested range:

> `min_score>0  =>  ratio<1`.

## Correct bridge lemmas tracked by the script

For active vertices, with `s_v=||Y_t(v)||`, `d_v=tr(Y_t(v))`, `w_v=d_v/sum d`:

- exact criterion: `dbar<gbar  <=>  sum_v w_v/s_v > 1`
- exact threshold form (when both `A_-={s<=1}` and `A_+={s>1}` are nonempty):
  - `rho_+ < rho_exact := (alpha_- - 1)/(alpha_- - beta_+)`
  - `alpha_- = E_{w|A_-}[1/s]`, `beta_+ = E_{w|A_+}[1/s]`
- valid extremal sufficient condition:
  - `rho_+ < rho_safe := ((1/M_-) - 1)/(((1/M_-) - 1) + (1 - 1/m_+))`
  - `M_- = max_{A_-} s`, `m_+ = min_{A_+} s`
- stricter one-parameter sufficient condition:
  - `rho_+ < 1 - M_-`

## Margin results (`n<=48`)

On nontrivial rows:

- `rho_exact - rho_+` min: `0.839219` (failures: `0`)
- `rho_safe - rho_+` min: `0.036920` (failures: `0`)
- `(1-M_-) - rho_+` min: `-0.041177` (failures: `1`)

So the corrected extremal bridge (`rho_safe`) is empirically consistent on all
sampled nontrivial rows. The one-parameter shortcut (`rho_+ < 1-M_-`) is not
universally valid empirically (one counterexample row), so it should be treated
as a stronger optional target, not the primary bridge.

## Implication for the proof program

The current theorem-level bridge is now best framed as proving either:

1. exact threshold control (`rho_+ < rho_exact`), or
2. extremal sufficient control (`rho_+ < rho_safe`).

Either one implies AR-NT and closes the nontrivial step condition via the ratio
certificate.

## Files

- `scripts/verify-p6-gpl-h-aggregate-ratio.py`
- `data/first-proof/problem6-aggregate-ratio-results.json`
- `data/first-proof/problem6-aggregate-ratio-report.md`
