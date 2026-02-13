# Problem 6 Codex Cycle 1 Verification

Date: 2026-02-13
Agent: Codex
Status: Completed

## Scope

Cycle 1 tasks from pairing:

1. Formally check the correction decomposition algebra:

   d_bar = uniform_d_bar + correction

   correction <= (||M_t|| / (eps*(eps-||M_t||))) * tr(F_t)/r_t

   d_bar <= uniform_d_bar * eps/(eps-||M_t||)

2. Stress-test candidate constants c in:

   ||M_t|| <= c*eps

across a mixed graph suite and epsilon grid.

## Commands Run

```bash
python3 scripts/verify-p6-cycle1-c-gap.py \
  --out-json data/first-proof/problem6-codex-cycle1-results.json
```

## Test Grid

- Graphs: 29 total
- Families:
  - Complete: K_20, K_40, K_60, K_80
  - Cycle: C_40, C_80
  - Path: P_40, P_80
  - Barbell: Barbell_20, Barbell_30, Barbell_40
  - Erdos-Renyi: n in {40, 60, 80}, p in {0.2, 0.4, 0.6}, 2 reps each
- Epsilon grid: {0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
- Nontrivial steps analyzed: 725
- RNG seed: 12345

## Results

### 1) Algebra checks (formal identities/inequalities)

All passed:

- decomposition exact failures: 0
- correction-bound failures: 0
- d_bar-bound failures: 0

So on all tested steps, the correction formula and both upper bounds are
numerically consistent with machine precision.

### 2) Constant c stress test for ||M_t|| <= c*eps

Global:

- max ||M_t||/eps = 0.3125
- q99(||M_t||/eps) = 0.2917
- min barrier gap fraction (eps-||M_t||)/eps = 68.75%

c-pass table (all 725 steps):

- c = 0.20: FAIL
- c = 0.25: FAIL
- c = 0.30: FAIL
- c = 1/3: PASS
- c = 0.35: PASS

c-pass table (progress <= 80% of horizon):

- c = 0.30: PASS
- c = 1/3: PASS

Interpretation: late steps are where ratios cross 0.30; a full-horizon
uniform statement should target c = 1/3 (or slightly larger), while
two-phase statements can use c = 0.30 in the early/mid regime.

### 3) Worst cases

- Worst ratio case:
  - graph: Barbell_40
  - eps: 0.4
  - t: 9
  - ||M_t||/eps: 0.3125
  - d_bar: 0.6506

- Worst d_bar case:
  - graph: K_80
  - eps: 0.5
  - t: 12
  - d_bar: 0.7179
  - uniform_d_bar: 0.6000
  - ||M_t||/eps: 0.3000

## Implications for Claude (Cycle 2)

1. Safe target for a global norm lemma on tested families:
   ||M_t|| <= eps/3.

2. If proving eps/3 directly is hard, a two-phase theorem is supported:
   - Phase A (<= 0.8 horizon): ||M_t|| <= 0.30*eps
   - Phase B (late steps): allow up to eps/3 and close with existing
     d_bar margins.

3. The correction algebra is now cleanly validated; remaining work is purely
   proving the norm-growth inequality itself (not fixing formula errors).

## Artifacts

- Script: `scripts/verify-p6-cycle1-c-gap.py`
- Machine summary: `data/first-proof/problem6-codex-cycle1-results.json`
