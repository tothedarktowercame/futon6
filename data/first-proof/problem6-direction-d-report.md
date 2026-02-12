# Problem 6: Direction D Probe Report (Near-rank-1 Reformulation)

Date: 2026-02-12
Author: Codex

### Direction D: Near-rank-1 reformulation

**What was tried:**
Implemented a dedicated Direction D diagnostic to test the handoff hypothesis
that the rank gap remains near 1 up to `n<=48`.

Added script:
- `scripts/verify-p6-gpl-h-direction-d.py`

The script reuses `find_case2b_instance` from `scripts/verify-p6-gpl-h.py`,
runs the same barrier-greedy trajectory, and records for each step:
- `gap_t(v) = tr(Y_t(v)) / ||Y_t(v)||` on active vertices,
- step aggregates (`mean_gap`, `max_gap`),
- rank-1 energy ratio `||Y_t(v)||^2 / ||Y_t(v)||_F^2`.

Commands run:
1. `python3 scripts/verify-p6-gpl-h-direction-d.py --nmax 48 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --c-step 0.5 --json-out data/first-proof/problem6-direction-d-results.json`
2. `python3 scripts/verify-p6-gpl-h-direction-d.py --nmax 24 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --c-step 0.5 --json-out data/first-proof/problem6-direction-d-results-n24.json`

---

**What happened:**

### Main run (`n<=48`)

- Case-2b instances: `218`
- Step rows: `463`

Global step stats:
- `mean_gap`: mean `1.1409`, p90 `1.4640`, p95 `1.6907`, max `2.2186`
- `max_gap`: mean `1.2398`, p90 `1.7131`, p95 `1.8464`, max `2.2186`
- rank-1 ratio `||Y||^2 / ||Y||_F^2`: mean `0.9552`, p05 `0.7919`, min `0.7354`

Threshold checks for the handoff viability criterion:
- fraction with `max_gap <= 1.05`: `0.6004`
- fraction with `max_gap <= 1.20`: `0.6004`
- fraction with `max_gap <= 1.50`: `0.7754`

Hard failures concentrate in dense families:
- `K`: max gap `2.219`
- `ER`: max gap `2.029`

Top hard rows:
- `K_48, eps=0.30, t=6`: `mean_gap=max_gap=2.2186`
- `K_40, eps=0.30, t=5`: `mean_gap=max_gap=2.1361`
- `ER_48_p0.5, eps=0.30, t=6`: `mean_gap=1.5681`, `max_gap=2.0293`

### Baseline comparison (`n<=24`)

- Case-2b instances: `52`
- Step rows: `49`
- `mean_gap` mean: `1.0521`
- `max_gap` mean: `1.0744`
- `max_gap <= 1.05` fraction: `0.8367`
- worst `max_gap`: `1.4967`

So the rank gap clearly worsens with scale and density.

---

**Exact failure point (if applicable):**

Direction D required a practically-uniform near-rank-1 regime (handoff target:
rank gap staying close to 1, e.g. `<=1.05`). This fails on the expanded regime:

1. Only about `60%` of step rows satisfy `max_gap <= 1.05` at `n<=48`.
2. Dense late-step rows have `max_gap` in the `2.0-2.22` range.
3. Therefore the reduction
   `Y_t(v) ~ sigma_v q_v q_v^T` is not uniformly accurate enough for a
   theorem-level replacement of GPL-H in current form.

In short: near-rank-1 is a good heuristic, not a universal bridge.

---

**Partial results (if any):**

1. Quantified scale effect:
   - small regime (`n<=24`): close to rank-1,
   - larger regime (`n<=48`): substantial degradation in dense families.
2. Isolated where Direction D is strongest:
   - `RandReg`, `Dumbbell`, `DisjCliq` remain much closer to rank-1.
3. Produced reusable artifacts for follow-up theorem fitting:
   - `data/first-proof/problem6-direction-d-results.json`
   - `data/first-proof/problem6-direction-d-results-n24.json`

---

**Surprises:**

1. The degradation is strongly family-dependent: complete and ER graphs drive
   almost all severe violations, while sparse/structured families stay close.
2. Even where `max_gap` is large, GPL-H min-score behavior remains good in this
   run (`max min_score = 0.740741`), suggesting the obstacle is theorem shape,
   not existence of good vertices.

---

**New dead ends discovered:**

1. **Uniform rank-gap universality (`<=1.05`)** is false in the tested range.
2. **Pure shadow-family closure via rank-1 replacement** is not viable as a
   stand-alone unconditional argument for Case-2b.

---

**Verdict:** FAILS AT uniform near-rank-1 universality;
NARROWS GAP TO a density-aware/step-aware variant of Direction D.

A plausible next Direction D variant is conditional:
prove a transfer result under a bounded-density or bounded-gap hypothesis,
then combine with a separate argument handling dense late-step rows.

## Files

- `scripts/verify-p6-gpl-h-direction-d.py` — Direction D diagnostic script
- `data/first-proof/problem6-direction-d-results.json` — full run (`n<=48`)
- `data/first-proof/problem6-direction-d-results-n24.json` — baseline (`n<=24`)
- `data/first-proof/problem6-direction-d-report.md` — this report
