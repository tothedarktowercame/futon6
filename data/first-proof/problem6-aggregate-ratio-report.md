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

## Structural detail

Rows with nontrivial steps are overwhelmingly below the score-1 threshold:

- nontrivial `max_score` p95: `0.83496`
- only 3 nontrivial rows have `max_score>1`, all in dense ER late steps,
  but still `ratio<1`.

Above-1 drift mass is tiny on nontrivial rows:

- mean `above1_drift_frac`: `8.38e-4`
- max `above1_drift_frac`: `0.0650`

## Implication for the proof program

This suggests a sharper open theorem candidate:

### AR-NT (aggregate ratio, nontrivial form) — open

Under H1-H4, if `min_v ||Y_t(v)|| > 0`, then

`dbar_t / gbar_t < 1`.

Combined with the proved ratio certificate, AR-NT implies a good step on every
nontrivial row; trivial rows are already solved (`min=0`).

So GPL-H is reduced to proving AR-NT (or an equivalent inequality forcing it).

## Files

- `scripts/verify-p6-gpl-h-aggregate-ratio.py`
- `data/first-proof/problem6-aggregate-ratio-results.json`
- `data/first-proof/problem6-aggregate-ratio-report.md`

## Conditional bridge inequality (mass-above-1 form)

For nontrivial rows, splitting active vertices by `s_v=||Y_t(v)||` into
`A_-={s_v<=1}` and `A_+={s_v>1}`, we get

`gbar - dbar = (1/|A|)[ sum_{A_-} d_v(1/s_v-1) - sum_{A_+} d_v(1-1/s_v) ]`.

So `dbar<gbar` follows if

`(1/m_- - 1)(1-rho_+) > (1 - 1/M_+)rho_+`,

where
- `m_- = min_{A_-} s_v`,
- `M_+ = max_{A_+} s_v`,
- `rho_+ = (sum_{A_+} d_v)/(sum_A d_v)`.

Equivalent threshold form:

`rho_+ < rho_crit = ((1/m_-)-1)/(((1/m_-)-1)+(1-1/M_+))`.

### Empirical margin (`n<=48`)

From `data/first-proof/problem6-aggregate-ratio-results.json` on nontrivial rows:

- `rho_+` mean `8.38e-4`, max `0.0650`
- `rho_crit` min `0.9655`
- margin `rho_crit - rho_+` min `0.9005`
- violations of threshold condition: `0`

So in tested data, the bridge condition holds with very large slack.

This suggests a plausible path to closure: prove a universal H1-H4 bound forcing
`rho_+` far below `rho_crit`.
