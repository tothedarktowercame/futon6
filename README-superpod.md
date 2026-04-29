# README: Superpod Status

Last updated: 2026-04-29

This file is the operator-facing summary of where the superpod pipeline
stands now, across the live `mark2` lane, the in-code `mark3` upgrades,
and the new legacy-TeX normalization work.

It is not a replacement for the mission docs. It is the short state of
play.

## 1. What is live vs. what is new

There are now two distinct layers to keep straight:

- **mark2** is the live batch/coordinator workflow for broad arXiv runs.
- **mark3** is the next runner shape for arXiv batches: same general
  substrate, but with better paper-aware pattern tagging, explicit Stage 6
  coverage semantics, geometric artifacts, and eprint-first arXiv handling.

The important scoping point is:

- mark3 **extends** mark2; it does not replace the mark2 state machine.
- mark3 changes are **forward-applicable**. They do not require iterative
  reprocessing of already-completed batches.
- any retrospective rerun is a separate second-pass decision.

Primary mission docs:

- [M-superpod-mark2.md](/home/joe/code/futon6/holes/missions/M-superpod-mark2.md)
- [M-superpod-mark3.md](/home/joe/code/futon6/holes/missions/M-superpod-mark3.md)

## 2. Current operational lane

The live transfer lane is:

- server-side staging: `linode-chicago:~/mark2/`
- Joe-side archive: `~/code/storage/mark2/`

Semantics:

- `~/mark2/inbox/` holds batch tarballs for Rob to pull.
- `~/mark2/outbox/` holds returned result tarballs.
- `~/mark2/mark2 status` is the authoritative state view.
- the Chicago host is a **transfer surface**, not the long-term archive.

Current archival convention:

- numbered input/output history is kept under `~/code/storage/mark2/`
- preserved mfuton input bundles live under `~/code/storage/mark2/inbox/`
- preserved result bundles live under `~/code/storage/mark2/outbox/`

See also:

- [storage/mark2/README-OPERATOR.txt](/home/joe/code/storage/mark2/README-OPERATOR.txt)

## 3. What is already landed in code

The runner-side work for mark3 and `E-mfuton-silver` is already committed
in futon6.

### ArXiv runner upgrades

In [scripts/superpod-job.py](/home/joe/code/futon6/scripts/superpod-job.py):

- Stage 3 now has an arXiv-aware prompt path instead of relying only on the
  math.SE Q&A prompt premise.
- arXiv batches can auto-pick local `eprints/` when present, so the richer
  paper pipeline is reached without manual flag discipline.
- Stage 6 now emits explicit coverage/status records rather than ambiguous
  silent holes.
- slot-distinctness is enforced more explicitly for
  `situation_S` / `xiang_salience` / `arrow_constraint`.
- Stage 9a emits `geometry.json` and records geometry stats in the run
  manifest.

### Legacy TeX normalization

In [src/futon6/legacy_tex_normalize.py](/home/joe/code/futon6/src/futon6/legacy_tex_normalize.py)
and [src/futon6/paper_hypergraph.py](/home/joe/code/futon6/src/futon6/paper_hypergraph.py):

- declared `\newtheorem` alias expansion
- declared `\newenvironment` theorem/proof wrapper expansion
- undeclared env-alias recovery for common legacy names like `thm`, `lem`,
  `defi`
- wrapper-macro expansion such as `\be{...}` / `\ee{...}`
- command-style theorem/proof alias recovery such as `\let\lem\lemma`
- exact `\paragraph{Lemma.}`-style claim synthesis
- provenance carried through as `native`, `alias_expanded`, or
  `prose_synthesized`

This work is tracked in:

- [E-mfuton-silver.md](/home/joe/code/futon6/holes/excursions/E-mfuton-silver.md)

### Acceptance harness

There is now a checked-in replay harness:

- [eval-legacy-tex-normalization.py](/home/joe/code/futon6/scripts/eval-legacy-tex-normalization.py)

It compares baseline vs normalized Stage 5d behavior on either:

- an entity list plus local eprint directory, or
- an embedded batch tarball

## 4. What the evidence says so far

### Legacy-source lift

The current normalization wave is already materially useful on older arXiv
source.

Observed local replay results:

- 200-paper old-source sample: papers with at least one claim block
  `119 -> 187`
- `ct-validation/arxiv-superpod-mit-eprint-pilot-500`: among 495 locally
  replayable eprints, papers with claim blocks `230 -> 391`
- same pilot: papers with proof blocks `212 -> 236`
- same pilot: claim nodes `3675 -> 8078`

On that pilot replay, new claim provenance was:

- `3779` native
- `4299` alias-expanded
- `0` prose-synthesized

So the lift is coming mainly from source-structural recovery, not from a
free-form hallucination layer.

### Current floor

The remaining misses should currently be treated as an explicit floor, not
as license for unbounded parser widening.

Known floor classes:

- papers with no recoverable theorem/proof surface in the available source
- sources whose structure is custom expository formatting rather than
  theorem-block syntax
- custom command families not yet covered by the high-confidence rules
- source layouts whose useful body is missing or too degraded

New widening work should start from named residual buckets with examples and
acceptance targets, not from ad hoc heuristic drift.

## 5. What is still pending

### Operational rollout

The code is ahead of the operational rollout.

What still depends on execution-side follow-through:

- Rob pulling the updated futon6 runner and using it on future arXiv runs
- deciding whether mark3 runs under a separate manifest/command surface or
  as an explicit mark2-adjacent invocation convention
- any deliberate second-pass rerun of earlier numbered batches

### Evaluation still to finish

The early-batch acceptance replay is already strong, but the modern-source
positive control should still be completed as a longer offline job:

- replay mfuton bundles through the same baseline-vs-normalized harness
- confirm no material regression on already-good modern source

The harness is present; the remaining issue is runtime, not missing code.

## 6. Note on mfuton chronology

Rob has changed the mfuton harvesting policy going forward:

- old lane: `metadataPrefix=arXiv`, effectively latest-update chronology
- new lane: `metadataPrefix=arXivRaw`, sorted by the `v1` submission date

Implication:

- futon6 runner code is unchanged by this
- future mfuton batches should be more faithful to original-submission
  chronology
- pre-change and post-change mfuton batches are not cleanly comparable as
  identical time slices

That matters for interpretation of batch composition, not for the mark3
runner logic itself.

## 7. Older superpod docs

Some older repo-root superpod notes refer to the earlier StackExchange /
MathOverflow production run rather than the current arXiv batch lane.

In particular:

- [KNOWN_LIMITATIONS.md](/home/joe/code/futon6/KNOWN_LIMITATIONS.md)
- [SUPERPOD-RERUN-NOTES.md](/home/joe/code/futon6/SUPERPOD-RERUN-NOTES.md)

Those remain useful historical records, but they are not the authoritative
state summary for the current arXiv mark2/mark3 work.

## 8. Short version

If you need the one-paragraph summary:

The live superpod workflow is still `mark2` through the Chicago
transfer/archive lane, but the runner has now been materially upgraded in
code for the next arXiv phase. The important mark3-side changes are
arXiv-aware Stage 3 behavior, eprint-first paper handling, explicit Stage 6
coverage semantics, geometric artifacts, and a legacy-TeX normalization layer
that already shows large structural-recall gains on older arXiv source. The
main remaining work is operational rollout and disciplined acceptance replay,
not open-ended redesign.
