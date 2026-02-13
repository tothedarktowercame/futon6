# Problem 6 Cycle 3 Codex Verification

Date: 2026-02-13
Agent: Codex
Status: Completed

## Scope

Verified the Cycle 3 Neumann-alignment closure candidate on the required test set:

- Graphs: `K_40`, `K_80`, `ER_60(p=0.5)`, `Barbell_40` (n=80), `Star_40`, `Grid_8x5`
- Epsilons: `{0.2, 0.3, 0.5}`
- Greedy trajectory on Turan `I_0`, horizon `T = floor(eps*m0/3)` (with `max(1, ...)` guard)

## Artifacts

- Script: `scripts/verify-p6-cycle3-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle3-results.json`

## Command Run

```bash
python3 scripts/verify-p6-cycle3-codex.py \
  --out-json data/first-proof/problem6-codex-cycle3-results.json
```

## Task 1: Verify `tr(F_t) = 2(t - tau)`

Definitions used:
- `tau = tr(M_t)`
- `tr(F_t)` from matrix trace of cross-edge matrix
- cross-edge leverage sum computed directly from edge leverages

Results:
- `max |tr(F_t) - 2(t - tau)| = 12.0` (fails badly)
- Worst witness: `Star_40, eps=0.5, t=6` with `tr(F_t)=0`, `2(t-tau)=12`
- `max |tr(F_t) - cross_tau_sum| = 1.83e-13` (trace computation is consistent)

Correct identity found (numerically exact):

- `tr(F_t) = sum_{u in S_t} ell_u - 2*tr(M_t)`
- `max |tr(F_t) - (sum_{u in S_t} ell_u - 2*tau)| = 1.78e-14`

So `2(t-tau)` is not the right general identity (it only aligns in highly uniform settings).

## Task 2: Verify `rho_k <= 1/2` for `k=1,2,3,4`

`rho_k = tr(M_t^k F_t) / (||M_t||^k * tr(F_t))`

Maxima across all tested steps/runs:

- `k=1`: max `rho_1 = 0.461538`
- `k=2`: max `rho_2 = 0.461538`
- `k=3`: max `rho_3 = 0.461538`
- `k=4`: max `rho_4 = 0.461538`

Violations of `rho_k > 1/2`: none for all `k=1..4`.

Extremal witness for all k: `K_80, eps=0.5, t=13`.

## Task 3: Verify full Neumann amplification bound

Checked at each step:

- `dbar <= dbar^{M=0} * (2-x)/(2(1-x))`
- where `x = ||M_t||/eps`, `dbar^{M=0} = tr(F_t)/(eps*r_t)`

Results:
- `max dbar / bound = 1.000000` (up to floating-point noise)
- Violations (`> 1`): none
- Tightest observed ratio: `1.0000000000000004` at `ER_60_p0.5_seed42_rep0, eps=0.3, t=1`

## Task 4: Track `x = ||M_t||/eps`

K_n formula check:
- Verified `x_t = t/(n*eps)` (with the `t=1` zero-rank edge case handled explicitly)
- `max |x - t/(n*eps)| = 3.05e-16`

Horizon behavior (`t=T` row per run):
- `x_t > 1/3` occurs in 2 runs:
  - `Barbell_40, eps=0.2, t=5, x=0.3750`
  - `Barbell_40, eps=0.5, t=13, x=0.3500`

So `x <= 1/3` at horizon is not universally true on this test set.

## Task 5: Verify `tr(M^k F) <= mu_max^k * (t - tau)` for `k=1,2,3`

Checked per-step with `mu_max = ||M_t||`.

Results:
- No violations for `k=1,2,3`
- Maximum positive gap `lhs-rhs` was `0` (floating-point level)

Also checked alternative RHS implied by the corrected trace relation:
- `mu_max^k * tr(F_t)/2`
- No violations for `k=1,2,3`

## Sanity check: `F_t <= Pi - M_t`

Verified Loewner-order sanity via `lambda_max(F_t + M_t - Pi)`:
- max observed value: `9.62e-14` (numerical zero)

## Key Question Answer

Does the algebraic chain

- `rho_k <= 1/2` -> `dbar <= dbar^{M=0} * (2-x)/(2(1-x))`

hold numerically for all tested instances?

**Yes.** On this entire required test suite, all `rho_k` checks (`k=1..4`) and all Neumann-bound checks passed.

Main caveat: the claimed auxiliary identity `tr(F_t)=2(t-tau)` is false in general; the numerically exact replacement is

- `tr(F_t)=sum_{u in S_t} ell_u - 2*tr(M_t)`.
