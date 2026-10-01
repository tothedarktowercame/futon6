# M-mark8 — same-budget adaptive mining and inspectable results

Status: open

## Purpose

Mark8 improves how a fixed model-call budget is allocated and makes the
allocation observable. Mark7 artifacts and accounting remain the comparison
baseline. A Mark8 result must state both its model-call budget and the Mark7
policy it is compared with.

## Invariants

1. WARP and TAPESTRY are independently configurable, enabled by default, and
   pinned in the immutable run manifest.
2. Immutable run artifacts and item accounting are authoritative. Runtime
   charts and a later Neo4j projection are derived views and can be rebuilt.
3. An allocator cannot create calls: every policy report proves that input and
   output slot totals are equal.
4. Pre-call features read only frozen corpus, citation, and encyclopedia data.
   Tests forbid model-produced artifacts from entering a pre-call feature.
5. A new allocation signal remains report-only until a frozen offline gate
   establishes predictive value. It is never promoted because it explains the
   same outputs from which it was computed.

## First bounded slice: report-only wooliness

For a passage `p`, report these components separately:

- `U`: share of used terms not defined locally, earlier in the paper, or in the
  frozen concept encyclopedia;
- `C`: share of cited references that do not resolve in the frozen citation
  index;
- `D`: mean citation-hop distance to the nearest resolvable definition,
  clipped at three hops and normalized to `[0,1]`.

The initial diagnostic is

`W(p) = clip(0.45 U + 0.25 C + 0.30 D, 0, 1)`.

The weights are frozen configuration, not learned from the evaluation labels.
This slice adds a deterministic CLI and tests, emits `wooliness.json`, and
renders wooliness-versus-outcome and operator-attention reports. It changes no
live allocation behavior.

Promotion to an allocator input requires all of:

- deterministic replay from the same inputs;
- monotonic component tests and fixtures where adding an in-paper definition
  or resolving a citation lowers `W`;
- a no-model-output import/read test;
- AUC at least 0.65 for frozen Mark7 weak-extraction labels and a comparison
  against the best existing pre-call proxy;
- a prospective holdout or exploration tranche, because retrospective labels
  exist only for candidates Mark7 selected.

## Allocation slices after validation

1. Offline replay of S4 selection at the same global acceptance total.
2. Deterministic prechecks for missing clause spans, cyclic step proposals, and
   prompt-size overflow, with saved slots returned to the deferred pool.
3. A refusal/error queue distinguishing deterministic skips, retryable parse
   failures, timeouts, and truncations.
4. Batchwise routing using post-call outcomes while retaining a declared
   exploration share.
5. A compact structural-neighbour browser over accepted graphs and embeddings.
6. A rebuildable Neo4j projection for multi-hop post-run queries; Neo4j is not
   required by the runner or scheduler.

## Runtime reports

The report surface includes region-pressure, budget flow, recovery reasons,
wooliness versus outcomes, marginal value by paper, an operator top-K queue,
and vocabulary-cluster health. Every chart names its source artifacts and the
decision it is intended to inform.

## Acceptance

- [x] Mark8 has an isolated branch based on the default-on WARP/TAPESTRY work.
- [x] The wooliness diagnostic, promotion gates, and Neo4j boundary are
      declared before implementation.
- [ ] The report-only wooliness slice passes unit, determinism, and forbidden-
      input tests.
- [ ] Offline S4 policies are compared at an identical global call budget.
- [ ] The refusal/error queue preserves every attempted or skipped item with a
      reason and retry disposition.
- [ ] A prospective run compares Mark8 with the frozen Mark7 baseline.

## Evidence and correction to the analytics handoff

The initial analytics design is
`holes/mark8-runtime-analytics-design.md` in the canonical checkout. Its A1
description treats Mark7 as a flat-cap policy. The current runner source already
declares `exposition-first-scaled-cap/v3`, implemented by
`run_manifest.scaled_cap` and consumed by
`mark3_extract_expository_candidates.py`. The 100-paper result may have been
created by an older flat-cap revision, so replay must read the run manifest and
accounting actually archived with that result instead of inferring its policy
from current source.
