# README: Superpod Status

Last updated: 2026-05-20

This file is the operator-facing summary of where the superpod pipeline
stands now across the live `mark2` lane, the in-code `mark3` upgrades, and
the latest arXiv eprint-source QC fix.

It is not a replacement for the mission docs. It is the short state of play.

## 1. What is live vs. what is new

There are still two distinct layers to keep straight:

- **mark2** is the live batch/coordinator workflow for broad arXiv runs.
- **mark3** is the next runner shape for arXiv batches: same general
  substrate, but with better paper-aware pattern tagging, explicit Stage 6
  coverage semantics, geometric artifacts, and eprint-first arXiv handling.

The important scoping point is:

- mark3 **extends** mark2; it does not replace the mark2 state machine
- mark3 changes are **forward-applicable** and do not require forced
  retrospective reruns
- any retrospective rerun is a separate second-pass decision

Primary mission docs:

- [M-superpod-mark2.md](/home/joe/code/futon6/holes/missions/M-superpod-mark2.md)
- [M-superpod-mark3.md](/home/joe/code/futon6/holes/missions/M-superpod-mark3.md)

## 2. Current operational lane

The live transfer lane is:

- server-side staging: `linode-chicago:~/mark2/`
- Joe-side archive: `~/code/storage/mark2/`

Semantics:

- `~/mark2/inbox/` holds batch tarballs for Rob to pull
- `~/mark2/outbox/` holds returned result tarballs
- `~/mark2/mark2 status` is the authoritative state view
- the Chicago host is a transfer surface, not the long-term archive

Current archival convention:

- numbered input/output history is kept under `~/code/storage/mark2/`
- preserved mfuton input bundles live under `~/code/storage/mark2/inbox/`
- preserved result bundles live under `~/code/storage/mark2/outbox/`

See also:

- [storage/mark2/README-OPERATOR.txt](/home/joe/code/storage/mark2/README-OPERATOR.txt)
- [futon6-status.py](/home/joe/code/futon6/scripts/futon6-status.py)

For a concrete operator snapshot that joins the live Linode state to the
local archive mirrors, run:

```bash
~/code/futon6/scripts/futon6-status.py
```

## 3. What is already landed in code

The runner-side work for mark3 and `E-mfuton-silver` is already landed in
futon6.

### ArXiv runner upgrades

In [superpod-job.py](/home/joe/code/futon6/scripts/superpod-job.py):

- Stage 3 has an arXiv-aware prompt path instead of relying only on the
  math.SE Q&A premise
- arXiv batches can auto-pick local `eprints/` when present, so the richer
  paper path is reached without manual flag discipline
- Stage 5 NER/scope detection is now eprint-first for arXiv paper runs
  instead of mining duplicated abstract text
- Stage 5 now emits a combined `discourse-wiring.json` artifact
  alongside `scopes.json`, carrying scopes plus wires, ports, and labels
- Stage 5 learned-term output now feeds forward into Stage 5c and Stage 5d,
  so paper technique extraction and paper hypergraphs can use terms learned
  earlier in the same run
- Stage 6 emits explicit coverage/status records rather than ambiguous
  silent holes
- slot-distinctness is enforced more explicitly for
  `situation_S` / `xiang_salience` / `arrow_constraint`
- Stage 9a emits `geometry.json` and records geometry stats in the run
  manifest

### Legacy TeX normalization

In [legacy_tex_normalize.py](/home/joe/code/futon6/src/futon6/legacy_tex_normalize.py)
and [paper_hypergraph.py](/home/joe/code/futon6/src/futon6/paper_hypergraph.py):

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

- an entity list plus local eprint directory
- an embedded batch tarball

## 4. Critical operator rule: paper mining must be eprint-backed

For arXiv runs, the critical current rule is:

- if you mean to mine papers, supply or inherit local `eprints/`
- do not treat abstract-only output as a valid paper-mining run

Reason:

- a May 2026 QC pass found that Stage 5 scope detection had been running on
  `question-body + answer-body`, which for arXiv entities was effectively the
  abstract duplicated, not the paper body
- this produced misleadingly low scope coverage even on mathematically rich
  category-theory papers
- the runner now uses the same eprint-first text-source logic across Stage 5,
  Stage 5b, Stage 5c, and Stage 5d, and records `text_source_counts` plus
  eprint load status in the manifest/stage-status sidecars
- when `--paper-eprint-dir` is set for arXiv paper stages, the runner now
  refuses to silently write zero-eprint abstract-only Stage 5 output

The practical interpretation is:

- low scope coverage is not meaningful evidence unless the manifest confirms
  that the paper stages actually loaded eprints
- the first things to inspect on any arXiv paper run are:
  - `manifest.json -> stage5_stats.text_source_counts`
  - `manifest.json -> stage5_stats.eprint_status_counts`
  - `manifest.json -> stage5_stats.total_discourse_records`
  - `manifest.json -> stage5_stats.total_wires / total_ports / total_labels`
  - `manifest.json -> stage_status.technique_ner.text_source_counts`
  - `manifest.json -> stage_status.paper_hypergraph.text_source_counts`
  - `manifest.json -> stage9a_stats.eprint_text_used`

If those do not show eprint usage, treat the run as QC-failed for paper
mining, not as a weak-but-valid result.

## 5. Laptop-safe running instructions

This laptop has no GPU. That does **not** block the paper-mining lane. It
only blocks embeddings, LLM inference, graph embeddings, and FAISS.

For a CPU-safe arXiv replay that still exercises the corrected eprint-backed
paper stages, use:

```bash
python3 scripts/superpod-job.py \
  --input-dir /ABS/PATH/TO/batch-input \
  --arxiv-jsonl BATCH.jsonl \
  --site arxiv.math \
  --output-dir /ABS/PATH/TO/output \
  --paper-eprint-dir eprints \
  --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \
  --discover-terms \
  --discover-structures \
  --discover-terms-eprint-dir eprints \
  --discover-terms-pm-seed /home/joe/code/futon6/data/dictionary/entries-pm-seed.edn \
  --discover-terms-nlab-seed /home/joe/code/futon6/data/dictionary/entries-nlab-seed.edn \
  --discover-terms-nnexus-stopwords /home/joe/code/nnexus/lib/NNexus/StopWordList.pm \
  --discover-terms-nnexus-snapshot /home/joe/code/nnexus/lib/NNexus/resources/database/snapshot-6-2014.sqlite \
  --skip-embeddings \
  --skip-llm \
  --skip-clustering \
  --skip-graph-embed \
  --skip-faiss
```

Why these flags are the current default-safe lane:

- `--paper-eprint-dir eprints` forces the paper stages onto the real source
  tree rather than the abstract surrogate
- `--discover-terms-eprint-dir eprints` makes open-world term discovery read
  the same source family
- the seed flags make Stage 5 classify extracted terms against the current
  PM seed, nLab seed, and NNexus concept snapshot while still retaining
  provisional genuinely new terms
- `--discover-structures` makes Stage 5 mine term-dense uncovered residual
  sentences into learned structure signatures and write a simple
  structure/term loss summary
- the same learned-term stream is then reused in Stage 5c and Stage 5d, so
  the runner can accumulate terminology in-run instead of requiring a
  separate downstream novelty pass
- `--skip-embeddings --skip-llm --skip-clustering --skip-graph-embed --skip-faiss`
  keeps the run CPU-safe while still giving:
  - Stage 5 NER/scope output
  - `discourse-wiring.json` with scope + wire + port + label records
  - `candidate-new-terms.jsonl` with seed-aware novelty labels
  - `learned-term-dictionary.jsonl` with provisional OED-style entries
  - `learned-structure-candidates.json` with reusable residual signatures
  - `learned-structure-summary.json` with structure loss and seed-match stats
  - `qc-preregister.json` with historical baseline checks against archived
    `mark2` runs
  - Stage 5c classical technique extraction
  - Stage 5d classical paper hypergraphs
  - Stage 9a geometry

Expected post-run checks:

- Stage 5 should report `text source: eprint=N, abstract-fallback=0`
- Stage 5 should also report a learned-dictionary summary such as:
  - `new=...`
  - `seed-known-missing-from-kernel=...`
  - `rhs_supported=...`
- Stage 5 should now also report:
  - `Discourse coverage: ...`
  - `wires=...`
  - `ports=...`
  - `labels=...`
- Stage 5c and Stage 5d should also report eprint-only text use
- Stage 9a should report nonzero eprint text coverage

If any of those fall back to abstracts for a supposedly paper-backed batch,
stop and treat the run as a provenance failure.

## 5a. Distributed Proofreaders loop

For structure-learning work, the current recommended audit loop is:

```bash
python3 scripts/build-uncovered-sentence-audit.py
```

That script:

- chooses fresh papers not already used in the daisychain ledger
- measures current Stage 5 discourse coverage
- emits only the residual uncovered sentences for manual review
- advances a ledger so the next run shifts to new papers instead of
  repeatedly tuning on the same examples

The main outputs are:

- `data/showcases/distributed-proofreaders/latest-audit.json`
- `data/showcases/distributed-proofreaders/latest-audit.html`
- `data/showcases/distributed-proofreaders/daisychain-ledger.json`

The intent is to grow a reusable seed kit of structure patterns, not to
paper-special-case the detector around one or two hand-picked texts.

## 6. What the evidence says so far

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

### Latest two-paper CPU replay

A concrete post-fix replay on two real `math.CT` papers from `batch-008`
using the CPU-safe invocation above produced:

- Stage 5 text source: `eprint=2`, `abstract-fallback=0`
- Stage 5 scope coverage: `100%`
- Stage 5 NER coverage: `100%`
- Stage 5c text source: `eprint=2`, `abstract=0`
- Stage 5d text source: `eprint=2`, `abstract=0`
- Stage 9a eprint coverage: `2/2`

That is the current compact positive control that the paper-mining lane is
actually reading papers rather than abstract surrogates.

### Current floor

The remaining misses should still be treated as an explicit floor, not as
license for unbounded parser widening.

Known floor classes:

- papers with no recoverable theorem/proof surface in the available source
- sources whose structure is custom expository formatting rather than
  theorem-block syntax
- custom command families not yet covered by the high-confidence rules
- source layouts whose useful body is missing or too degraded

New widening work should start from named residual buckets with examples and
acceptance targets, not from ad hoc heuristic drift.

## 7. What is still pending

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

## 8. Note on mfuton chronology

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

## 9. Older superpod docs

Some older repo-root superpod notes refer to the earlier StackExchange /
MathOverflow production run rather than the current arXiv batch lane.

In particular:

- [KNOWN_LIMITATIONS.md](/home/joe/code/futon6/KNOWN_LIMITATIONS.md)
- [SUPERPOD-RERUN-NOTES.md](/home/joe/code/futon6/SUPERPOD-RERUN-NOTES.md)

Those remain useful historical records, but they are not the authoritative
state summary for the current arXiv mark2/mark3 work.

## 10. Short version

If you need the one-paragraph summary:

The live superpod workflow is still `mark2` through the Chicago
transfer/archive lane, but the runner has now been materially upgraded in
code for the next arXiv phase. The important current point is that paper
mining must be eprint-backed: Stage 5 NER/scope detection, Stage 5c
technique extraction, Stage 5d paper hypergraphs, and Stage 9a geometry are
only trustworthy when the manifest confirms real eprint usage. The laptop
safe invocation is now the CPU-only paper path with embeddings/LLM/FAISS
disabled, and the latest two-paper replay confirms that this path is reading
papers rather than abstract surrogates.
