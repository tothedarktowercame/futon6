# Futon6 Hotspot Audit: TryHarder Loops and Missed Pivots (2026-03-05)

Goal: identify where effort increased without proportionate understanding, and define concrete stop/pivot rules.

## Evidence Snapshot

- Frontier review judgement coverage is zero:
  - `superpod-frontier-trial-review.json`: `0/67`
  - `superpod-frontier-trial-review-ungated.json`: `0/96`
  - `superpod-frontier-trial-mo-review.json`: `0/70`
  - `superpod-frontier-trial-mo-review-ungated.json`: `0/96`
- Frontier problem state files are still `spec_lock_status: fail` with pending formal statement/quantifiers/regime.
- P7 rerun matrix (3 arms x 2 reps) showed weak quality separation and strong latency separation.
- P7 unresolved concentration remains high at node level across runs:
  - `p7-problem`: unresolved `11/11`, verified `0`
  - `p7-s6`: unresolved `9/9`, verified `0`
  - `p7-synthesis`: unresolved `8/8`, verified `0`
- `run-frontier-superpod-trial.py` retrieval path is SE/MO superpod-only (no arXiv retrieval source).

## Key Moments (Hotspots)

### H1. Frontier runs before Spec-Lock

- Moment: Frontier FM retrieval trials were run while FM state files still had `spec_lock_status: fail`.
- Symptom: candidate quality was discussed despite unresolved formal statement and quantifier regime.
- Cost: retrieval quality judgments became ungrounded; difficult to know what "relevant" means.
- Missed trigger: `spec_lock_status != pass` should have blocked retrieval experiments.
- Better move: force SPEC completion first (formal statement, quantifiers, parameter regime, output type).

### H2. Retrieval tuning without labeling

- Moment: multiple frontier trial variants (gated/ungated, MO/non-MO) were produced, but no judgements were filled.
- Symptom: scoring output was all zeros (`pairs=0`) because no labels existed.
- Cost: repeated runs produced no epistemic update.
- Missed trigger: after first trial, required minimum label quota was not enforced.
- Better move: pause all reruns until at least 30 labelled pairs per problem (or equivalent budget).

### H3. Structural-gate oscillation without decisive signal

- Moment: concern that overlap gate kills structural novelty led to ungated reruns.
- Symptom: ungated structural pool had high zero-overlap noise (especially FM-002/003), but no human labels to confirm lift.
- Cost: debate moved to mechanism speculation instead of measured precision/lift.
- Missed trigger: source-sliced judgement metrics were available but not populated.
- Better move: label source buckets first, then decide gate policy by measured structural lift.

### H4. P7 rerun expansion with tie-heavy outcomes

- Moment: P7 moved from 2-arm to 3-arm x 2 reps (54 judgments).
- Symptom: pairwise comparisons were mostly ties, quality deltas small.
- Cost: large runtime consumed for limited learning signal.
- Missed trigger: tie-dominance threshold not used as stop condition.
- Better move: stop when pairwise ties exceed ~70% and switch to qualitative node-level diagnosis.

### H5. Latency problem became the work

- Moment: processed arm timeouts/retries and stalls repeatedly shaped run strategy.
- Symptom: significant effort moved into resiliency mechanics rather than math insight.
- Cost: infrastructure progress, but weak theorem-level progress.
- Missed trigger: no explicit cap on latency-debug budget per experimental question.
- Better move: one hardening cycle, then freeze infra and continue only with stable arm (MO-only baseline).

### H6. Node-level uncertainty persisted but broad reruns continued

- Moment: same P7 nodes remained unresolved across runs (`p7-problem`, `p7-s6`, `p7-synthesis`).
- Symptom: broad reruns diluted effort over already stable nodes.
- Cost: compute spent where status was already robust.
- Missed trigger: unresolved-node concentration should trigger targeted rerun mode.
- Better move: claim-only broad pass, then wired rerun only uncertain node IDs.

### H7. Citation-dependent gaps treated as retrieval gaps

- Moment: several unresolved proof failures are theorem/citation gaps (reviewer: P2/P7/P8 in particular).
- Symptom: repeated corpus retrieval was used where primary-source theorem resolution was required.
- Cost: low-yield search over secondary discussions.
- Missed trigger: "missing theorem identifier" not treated as escalation criterion.
- Better move: immediate primary-source track (arXiv/papers/books) when core claim needs a named theorem.

### H8. Pipeline quality risk accepted too long before strict gates

- Moment: superpod pipeline had very low Stage 6 parse success and low Stage 7 categorical consistency.
- Symptom: downstream artifacts were used despite known weak foundations.
- Cost: trust calibration overhead and potential false confidence.
- Missed trigger: preflight/readiness gates were not enforced early enough.
- Better move: readiness contract as hard prerequisite before downstream research claims.

## Proposed Stop/Pivot Rules (Operational)

1. Spec gate:
- Do not run retrieval experiments when problem state `spec_lock_status != pass`.

2. Label gate:
- After first trial, no rerun allowed until label coverage reaches minimum quota (`>=30/problem` or agreed equivalent).

3. Tie gate:
- If pairwise tie rate exceeds `70%` after 2 replicates, stop broad reruns and switch to targeted node diagnosis.

4. Unresolved-node gate:
- If any node is unresolved in all runs (`verified=0`), switch to `--node-id` targeted mode only.

5. Citation gate:
- If blocker text is "needs theorem/citation" or equivalent, pivot to primary-source acquisition (arXiv/books/papers), not more MO/SE retrieval.

6. Infra budget gate:
- Cap latency-hardening effort per question (single cycle), then freeze infra and continue with stable arm.

## Immediate Priority List (to improve understanding, not runtime)

1. Frontier: complete Spec-Lock fields for FM-001/002/003 before any further retrieval reruns.
2. Frontier: label existing four review files first; compute source-sliced lift from labels.
3. First-Proof hotspots: run targeted node deep dives on persistent unresolved nodes (`P7: p7-problem, p7-s6, p7-synthesis`; `P3 synthesis cluster`).
4. Citation-critical problems (P2/P7/P8): open a primary-source track and record exact theorem IDs needed for closure.

