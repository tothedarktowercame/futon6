# Problem 6: Post-D Ratio Certificate Report

Date: 2026-02-12
Author: Codex

## What was done

1. Upgraded `scripts/verify-p6-gpl-h-direction-d.py` to record `mean_drift` per step.
2. Re-ran Direction D full sweep:
   - `python3 scripts/verify-p6-gpl-h-direction-d.py --nmax 48 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --c-step 0.5 --json-out data/first-proof/problem6-direction-d-results.json`
3. Reworked post-D runner to include a theorem-grade ratio branch:
   - `scripts/verify-p6-gpl-h-post-d-regime-split.py`
4. Regenerated post-D wiring with ratio branch reflected:
   - `scripts/proof6-post-direction-d-wiring-diagram.py`

## Proved identity (deterministic)

On active vertices at a fixed step (`s_v=||Y_t(v)||`, `d_v=tr(Y_t(v))`,
`g_v=d_v/s_v`), with

- `m_t = min_v s_v`
- `dbar_t = avg_v d_v`
- `gbar_t = avg_v g_v`

we have:

`m_t <= dbar_t / gbar_t`.

Hence `dbar_t/gbar_t < 1` certifies a good step (`m_t<1`).

## Empirical result on tested range

From `data/first-proof/problem6-direction-d-results.json` (`n<=48`):

- Step rows: `455`
- Nontrivial rows (`min_score>0`): `177`
- Ratio certificate holds on all nontrivial rows:
  - `nontrivial_ratio_fail_rows = 0`
  - `nontrivial_ratio_max = 0.740741`
- `ratio_or_trivial` coverage: `455/455 = 1.000`

## Implication for proof state

This does **not** yet fully close GPL-H unconditionally.

But it sharpens the missing bridge to:

> prove `dbar_t/gbar_t < 1` (or a sufficient disjunction implying it)
> from H1-H4 on nontrivial steps.

So the target moved from direct min-eigenvalue control of grouped atoms to an
aggregate ratio inequality with a proved endpoint-to-goal map.

## Files

- `scripts/verify-p6-gpl-h-direction-d.py`
- `scripts/verify-p6-gpl-h-post-d-regime-split.py`
- `scripts/proof6-post-direction-d-wiring-diagram.py`
- `data/first-proof/problem6-direction-d-results.json`
- `data/first-proof/problem6-post-d-regime-split-results.json`
- `data/first-proof/problem6-post-direction-d-wiring.json`
- `data/first-proof/problem6-post-d-ratio-report.md`
