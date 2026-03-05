# Hotspot Hardening Sprint (5-10 Days, 2026-03-05)

Objective: increase proof-system reliability on known hotspots before investing in greenfield FM solving.

Principle: no more broad reruns unless a gate is satisfied.

## Daily Control Panel

Run at start/end of each day:

```bash
python3 scripts/hotspot-dashboard.py \
  --output-md holes/handoffs/hotspot-dashboard-$(date +%F).md
```

Current gate status (day 0): all `FAIL`.

## Gates (Hard Rules)

1. Spec gate:
- Frontier retrieval experiments are blocked until all FM state files have `spec_lock_status: pass`.

2. Label gate:
- No retrieval knob tuning unless each review file reaches `>=30` labels/problem (90 total labels/file).

3. Stubborn-node gate:
- Broad proof reruns are blocked when stubborn nodes exist; only targeted `--node-id` passes are allowed.

4. Citation gate:
- If a blocker is theorem/citation identity, pivot to primary-source track immediately (arXiv/books/papers), not MO/SE reruns.

## Workstreams

### WS1. Proof Hotspot Closure (existing 10 problems)

Target first:
- P7: `p7-problem`, `p7-s6`, `p7-synthesis`
- P3: synthesis cluster (`p3-problem`, `p3-s1..s5`, `p3-synthesis`)

Method:
- claim-only broad pass only once (if needed)
- wired targeted reruns on uncertain nodes only
- stop if tie/noise dominates; switch to citation/lemma decomposition

### WS2. Frontier Discipline (FM-001/002/003)

Goal is readiness, not solving.

Required outputs per FM file:
- formal statement
- quantifiers
- parameter regime
- output format
- forbidden substitutions
- `spec_lock_status: pass`

Then fill existing review JSON labels before any new runs.

### WS3. Primary-Source Dependency Closure

Build theorem-ID dependency tables for citation-critical blockers (start with P2/P7/P8):
- blocker claim
- required theorem/proposition
- exact source candidate
- status: confirmed / ambiguous / missing

## 10-Day Skeleton

Day 1:
- Freeze broad reruns.
- Run dashboard baseline.
- Convert FM state files from `SPEC fail` to draft `SPEC pass` candidates.

Day 2:
- Finalize FM Spec-Lock fields (all three files).
- Start labeling Frontier review files (first pass, top-ranked candidates).

Day 3:
- Complete minimum label quota for one review file.
- Run scorer and source-sliced metrics.

Day 4:
- Continue label completion for remaining review files.
- Decide structural gate policy from measured lift, not intuition.

Day 5:
- P7 targeted node pass only (`--node-id` unresolved set).
- Produce node-level blocker decomposition (logic gap vs citation gap vs model variance).

Day 6:
- P3 targeted synthesis cluster pass.
- Extract minimal missing-lemma list.

Day 7:
- Citation track deepening for P2/P7/P8 (primary-source table v1).
- Verify which blockers are genuinely open vs just uncited.

Day 8:
- Second targeted passes only on nodes with changed assumptions/evidence.
- No re-run of stable nodes.

Day 9:
- Consolidate: updated hotspot report + dashboard trend.
- Decide readiness for new First-Proof release response mode.

Day 10:
- Dry-run protocol for incoming problems:
  - Spec-Lock
  - citation-first triggers
  - node-targeted verification
  - explicit stop/pivot conditions

## Exit Criteria (before FM solving push)

1. Dashboard gates:
- Spec gate `PASS`
- Label gate `PASS`
- Stubborn-node gate improved (or explicitly localized with citation dependencies)

2. Proof-side:
- Persistent hotspot nodes reduced or fully mapped to named theorem dependencies.

3. Process-side:
- Every rerun justified by a changed hypothesis/evidence source, not “try harder”.

## Decision Rule for FM Solving Attempts

Only attempt substantive FM solving if all are true:
- FM Spec-Lock complete
- Label gate passed on existing retrieval artifacts
- hotspot trend is improving (not flat) for at least 2 consecutive daily dashboards

Otherwise: continue hardening and dependency closure.
