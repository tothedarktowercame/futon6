# FutonZero Follow-On: Next 10 Days Hotspot Queue (Learning Yield Ordered)

Date: `2026-03-05`  
Mission anchor: `futon0/holes/missions/M-futonzero-mvp.md`  
Operational anchor: `holes/handoffs/hotspot-hardening-sprint-2026-03-05.md`

## Capability-Yield Frame (from FutonZero)

Queue order is based on expected gain across FutonZero capability dimensions:

- `Task capability delta`: expected reduction in unresolved hotspot claims.
- `Discipline adaptation delta`: expected improvement in PSR/PUR/PAR + gate behavior.
- `Pathway utility`: expected improvement in reusable policy for upcoming First-Proof problems.

Weighted learning-yield score (1-5):  
`0.5 * task_delta + 0.3 * discipline_delta + 0.2 * pathway_utility`

## Hard Invariants (must hold for all slots)

- No broad reruns while stubborn-node gate is failing.
- No frontier retrieval reruns while `spec_lock_status != pass`.
- No retrieval-policy claims without label coverage (`>=30/problem` per review file).
- Citation blockers must pivot to primary sources immediately (no MO/SE-only loop).

## Single Queue (10 Days, Highest Yield First)

| Rank | Day | Work Package | Hotspot/Gate Focus | Capability Target | Yield |
|---|---:|---|---|---|---:|
| 1 | 1 | FM Spec-Lock closure (`FM-001/002/003`) | Spec gate fail -> pass | Discipline + pathway grounding before execution | 4.8 |
| 2 | 2 | Frontier labeling bootstrap (first review file to quota) | Label gate fail -> partial | Evidence-over-assertion discipline | 4.6 |
| 3 | 3 | Complete label quota on all 4 frontier review files + source-sliced scoring | Label gate fail -> pass | Pathway policy signal (guided vs noise) | 4.5 |
| 4 | 4 | P7 targeted node run (`p7-problem`, `p7-s6`, `p7-synthesis`) | Stubborn nodes with highest persistence | Task delta on central blocker cluster | 4.4 |
| 5 | 5 | Citation dependency table for `P2/P7/P8` (theorem IDs + source status) | Citation-gated blockers | Discipline pivot quality + future reuse | 4.3 |
| 6 | 6 | P3 synthesis-cluster targeted run (`p3-problem`, `p3-s1..s5`, `p3-synthesis`) | 92.6% unresolved problem lane | Task delta + lemma decomposition skill | 4.1 |
| 7 | 7 | P9 hotspot package replay from stepper snapshot | High unresolved + many hotspots | Transfer to non-P3/P7 structure | 3.9 |
| 8 | 8 | P2 and P5 hotspot package passes (node-targeted only) | Medium-high unresolved lanes | Generalization of targeted method | 3.7 |
| 9 | 9 | P10 then P1 hotspot package passes + no-regression check on strong lanes (`P8`) | Stability + regression guard | Pathway robustness for release-readiness | 3.6 |
| 10 | 10 | Incoming-problem dry-run protocol (simulate next First-Proof release) | End-to-end gate discipline | Reusable capability loop for next batch | 3.8 |

## Slot Exit Evidence (for FutonZero accounting)

Each slot is complete only with durable artifacts:

- `task delta evidence`: node-level before/after counts (`verified/plausible/gap/error/parse`).
- `discipline evidence`: explicit PSR + PUR + gate decision note in handoff.
- `pathway evidence`: what policy changed (preserve/adapt/drop) and why.
- `durability`: report and machine-readable outputs committed (or persisted under storage roots with path references).

## Stop/Pivot Triggers (queue-level)

- Pairwise tie rate `>70%` after 2 reps on a package -> stop broad comparison; switch to node diagnosis.
- Any node unresolved across all runs (`verified=0`) -> node-only mode; no full-problem rerun.
- Latency hardening exceeds one cycle for same question -> freeze infra path, continue with stable arm.

## Expected Outcome by Day 10

- Frontier gates moved from `FAIL` to measurable `PASS` states where currently blocked.
- Persistent hotspots reduced or converted into explicit theorem/citation dependency records.
- A repeatable, capability-scored hotspot workflow ready for the next First-Proof problem release window.
