# Problem 6 Cycle 6 Codex Verification

Date: 2026-02-13  
Agent: Codex  
Base handoff: `data/first-proof/problem6-codex-cycle6-handoff.md`

Artifacts:
- Script: `scripts/verify-p6-cycle6-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle6-results.json`

## Executive Summary

Cycle 6 gives a clear empirical split between bridge options:

1. **Task 1 / Task 2 (Order-comparison bridge A): not supported.**
   - Barrier greedy does **not** satisfy the partial-averages leverage-degree bound in general.
   - Many counterexamples were found.
2. **Task 3 (Modified construction bridge B): strongly supported.**
   - Modified leverage-order barrier greedy reaches horizon in all tested runs.
   - Barrier is preserved (`max ||M||/eps = 0.9866 < 1`).
   - `dbar0` stays below 1 with wide margin (`max = 0.7333`).
3. **Task 4 (Direct induction probe): strongly supported numerically.**
   - The tested local delta condition holds at every step in all runs.
4. **Task 5 (Strict-light threshold): fixed.**
   - With heavy defined as `tau_e >= eps` (with tolerance), knife-edge cases disappear in exhaustive small-n scan.

Suite coverage for Tasks 1-4: 29 graph instances x 4 eps values = **116 runs**.

## Task 1: Numerical Comparison (three trajectories)

Compared at every run:
- `barrier_greedy`: min `tr(Y_t(v))` among barrier-feasible candidates,
- `leverage_prefix`: first `t` vertices sorted by static leverage degree `ell_v`,
- `modified_greedy`: process leverage order, add only if barrier-feasible.

Tracked per step:
- `sum_{u in S_t} ell_u`,
- `dbar0_t = tr(F_t)/(r_t eps)`,
- `tau_internal_t`,
- partial-averages bound `2t(m-1)/m`.

### Key empirical answers

1. **Does barrier greedy stay under partial-averages bound?**
   - No.
   - Violations: **187** step instances.
   - Worst gap: `+0.7382` at `C5b_Bip_n120_p0.332_i2`, `eps=0.5`, `t=20`.

2. **Does modified greedy find T vertices?**
   - Yes, in all runs.
   - Failure count: **0/116**.

3. **How do trajectories compare on `dbar0`?**
   - `dbar0_modified <= dbar0_barrier` at all matched steps in this scan.
   - Worst observed trajectory maxima:
     - barrier: `0.8081`
     - modified: `0.7333`
     - leverage-prefix: `0.7333`

## Task 2: Order-Comparison Lemma (bridge A)

### Approach 1 (barrier score to leverage degree)

No formal inequality bridge was established from `tr(Y_t(v))` to full static `ell_v` that would force partial-averages compatibility.

### Approach 2 (selected vertex below average leverage degree)

Counterexamples found.

- `ell_{v_t} <= 2(m-1)/m` fails in many steps.
- Counterexample count: **205** step instances.

### Approach 3 (stochastic domination by sorted prefix)

Empirical proxy (sum-ell vs partial bound) fails (same 187 violations), so this route is not currently viable as-is.

## Task 3: Modified Construction (bridge B)

Implemented exactly the proposed leverage-order barrier greedy:
- sort `I0` by nondecreasing `ell_v`,
- iterate in that order,
- add `v` iff barrier-feasible (`||Y_t(v)|| < 1`).

### Results

- Reaches `T = floor(eps*m/3)` in **all 116 runs**.
- Barrier maintained:
  - worst `||M||/eps = 0.986597` (`Reg_100_d10`, `eps=0.5`, `t=16`).
- `dbar0` bound is clean:
  - worst `dbar0 = 0.733333` (`K_50_50`, `eps=0.3`, `t=10`).

This is strong evidence that the clean proof path is Bridge B, not Bridge A.

## Task 4: Direct Induction (bridge C probe)

For barrier trajectory, tracked:

\[
\Delta_t := \ell_{v_t}^{R_{t+1}} - \ell_{v_t}^{S_t},\qquad
\text{rhs}_t := (r_t-1)\varepsilon - \operatorname{tr}(F_t).
\]

Tested condition: `Delta_t < rhs_t` at each step.

### Results

- Condition holds for **all 726 checked steps**.
- Failure count: **0**.
- Recurrence identity sanity check:
  - `trF_{t+1} = trF_t + Delta_t`
  - max absolute error: `8.53e-14`.

So the local induction condition is numerically very robust in this suite.

## Task 5: Strict-light fix (`tau >= eps` heavy)

Implemented threshold with floating tolerance:
- heavy iff `tau_e >= eps - 1e-12`,
- light iff `tau_e < eps - 1e-12`.

Exhaustive small-n scan (24 graphs x 3 eps = 72 rows):
- `count(max_dbar0 >= 1) = 0`.
- K10 knife-edge fixed:
  - `K_10, eps=0.2` now gives `m0=1`, `horizon=0`, `max_dbar0=0`.

## Bottom Line

- **Bridge A (order-comparison lemma for standard barrier greedy)** is empirically contradicted.
- **Bridge B (modified leverage-order barrier greedy)** is empirically successful across the full tested suite and keeps both barrier feasibility and `dbar0 < 1` margins.
- **Bridge C (direct induction local condition)** is also strongly supported numerically.
- Strict threshold cleanup is resolved and removes the knife-edge case.

Most pragmatic next step: formalize Bridge B as the main construction and use Task 4-style local inequality as a backup/auxiliary lemma.
