# Problem 6: Amplification Exploration Notes (Codex)

Date: 2026-02-13

## Goal
Explore the remaining Sub-gap 2 (amplification when `M_t != 0`) in
`problem6-proof-draft.md` by testing candidate inequalities for

- `dbar_t = tr((eps I - M_t)^{-1} A_t)`
- `dbar_t^{M=0} = tr(A_t)/eps`
- `amp_t := dbar_t / dbar_t^{M=0}`
- `x_t := ||M_t||/eps`

## New Script
- `scripts/verify-p6-amplification-candidates.py`

It runs the min-`||Y||` greedy on Case-2b-like instances and reports violations for:

1. crude: `amp <= 1/(1-x)`
2. candidate A: `amp <= 1 + x`
3. candidate B: `amp <= 1 + x/(2(1-x))`

It also reports

- `rho1 := tr(M_t A_t)/(||M_t|| tr(A_t))`.

## Current empirical result (seed=7, nmax=80)

From `python3 scripts/verify-p6-amplification-candidates.py --seed 7 --nmax 80`:

- rows: `1472`
- max `amp`: `1.176471` (at `K_80`, `eps=0.3`, `t=7`)
- max `x`: `0.384615` (at `DisjCliq_26x3`, `eps=0.2`, `t=4`)
- violations:
  - `amp <= 1/(1-x)`: `0`
  - `amp <= 1 + x`: `0`
  - `amp <= 1 + x/(2(1-x))`: `0`

`rho1` statistics:
- min `0.065252`, median `0.25`, p99 `0.436071`, max `0.444444`
- no sample had `rho1 > 0.5`

## Interpretation
The crude amplification bound is very loose. Two tighter candidates hold on all sampled steps:

- `amp <= 1 + x`
- `amp <= 1 + x/(2(1-x))`

Candidate B is notable because it matches the complete-graph asymptotic correction scale (half-strength first-order term vs crude geometric amplification).

## Suggested proof direction
A plausible route is Neumann expansion with a structural bound on mixed moments:

`dbar = (1/eps) tr(A) + (1/eps^2) tr(MA) + (1/eps^3) tr(M^2 A) + ...`

If one can prove a uniform coefficient bound

`tr(M^k A) <= (1/2) ||M||^k tr(A)` for all `k>=1`,

then candidate B follows:

`amp <= 1 + x/(2(1-x))`.

Empirically this is consistent with observed `rho1 <= 0.444444` and no `rho1 > 0.5` in sampled runs.

## Status
This is still empirical evidence, not a theorem yet. It gives a concrete amplification target inequality to attempt next in the draft.
