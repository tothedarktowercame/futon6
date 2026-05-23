# README: Superpod Status

Last updated: 2026-05-21

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

## 5. Running instructions

Two variants share most of the command, but **Rob's superpod run is
NOT just "drop the `--skip-*` flags"**: removing `--skip-llm` enables
Stage 6 with backend `local-llm`, which the help text on
`scripts/superpod-job.py:6454` notes achieves only ~10% parse rate
without schema constraints. Schema-constrained backends (`codex`,
`gemini`) reach ~100% — passing the 80% gate at
`scripts/superpod-job.py:8047`. The canonical superpod invocation
therefore must add `--stage6-backend codex` (or `gemini`).

Paths in the commands below are **repo/scratch-relative**, matching
the discipline of `scripts/run-arxiv-handoff.sh:52`. Absolute paths
to Joe-local dirs were a portability footgun: missing `--ner-kernel`
*silently skips Stage 5 entirely* (`scripts/superpod-job.py:7148`),
while missing seed/snapshot files only warn and degrade discovery
quality (`scripts/superpod-job.py:6753`).

- **For Rob's superpod (full GPU)**: drop the five `--skip-*` flags
  AND add `--stage6-backend codex` (or `gemini`).
- **For laptop CPU replay**: keep the `--skip-*` flags as shown so
  the paper-mining lane still exercises Stage 5 / 5c / 5d / 9a + the
  new structure-learning loop without GPU. Stage 6 is skipped
  transitively by `--skip-llm`, so the backend choice doesn't matter
  here.

In both modes the QC outputs print in the same shape and the
structure-learning loop runs.

### Laptop CPU replay (default-safe)

Run from the futon6 repo root; eprint dirs and seed paths are
relative to that root.

```bash
python3 scripts/superpod-job.py \
  --input-dir /ABS/PATH/TO/batch-input \
  --arxiv-jsonl BATCH.jsonl \
  --site arxiv.math \
  --output-dir /ABS/PATH/TO/output \
  --paper-eprint-dir eprints \
  --ner-kernel data/ner-kernel-clean.tsv \
  --discover-terms \
  --discover-structures \
  --discover-terms-eprint-dir eprints \
  --discover-terms-pm-seed data/dictionary/entries-pm-seed.edn \
  --discover-terms-nlab-seed data/dictionary/entries-nlab-seed.edn \
  --discover-terms-nnexus-stopwords ../nnexus/lib/NNexus/StopWordList.pm \
  --discover-terms-nnexus-snapshot ../nnexus/lib/NNexus/resources/database/snapshot-6-2014.sqlite \
  --skip-embeddings \
  --skip-llm \
  --skip-clustering \
  --skip-graph-embed \
  --skip-faiss
```

### Rob's superpod (full GPU)

Same command as above, but: drop all five `--skip-*` flags AND add
`--stage6-backend codex` (or `gemini`) so Stage 6 actually clears
its 80% parse-rate gate. Add the topic priors so the symbol-grounding
arbitration consults MSC + SE-corpus priors (see §9).

```bash
python3 scripts/superpod-job.py \
  --input-dir /ABS/PATH/TO/batch-input \
  --arxiv-jsonl BATCH.jsonl \
  --site arxiv.math \
  --output-dir /ABS/PATH/TO/output \
  --paper-eprint-dir eprints \
  --ner-kernel data/ner-kernel-clean.tsv \
  --discover-terms \
  --discover-structures \
  --discover-terms-eprint-dir eprints \
  --discover-terms-pm-seed data/dictionary/entries-pm-seed.edn \
  --discover-terms-nlab-seed data/dictionary/entries-nlab-seed.edn \
  --discover-terms-nnexus-stopwords ../nnexus/lib/NNexus/StopWordList.pm \
  --discover-terms-nnexus-snapshot ../nnexus/lib/NNexus/resources/database/snapshot-6-2014.sqlite \
  --stage6-backend codex
```

If the superpod environment doesn't have `nnexus` checked out
alongside futon6, omit the two `--discover-terms-nnexus-*` flags
rather than passing absolute Joe-local paths — discovery degrades
gracefully rather than failing the run.

Why these flags are the current default-safe lane:

- `--paper-eprint-dir eprints` forces the paper stages onto the real source
  tree rather than the abstract surrogate
- `--discover-terms-eprint-dir eprints` makes open-world term discovery read
  the same source family
- the seed flags make Stage 5 classify extracted terms against the current
  PM seed, nLab seed, and NNexus concept snapshot while still retaining
  provisional genuinely new terms
- `--discover-structures` makes Stage 5 mine term-dense uncovered residual
  sentences into learned structure signatures, classify them by discourse
  verb (scope / label / wire), and emit gated candidates that can be
  replayed on a future run via `--discover-structures-seed-json`
- the same learned-term stream is then reused in Stage 5c and Stage 5d, so
  the runner can accumulate terminology in-run instead of requiring a
  separate downstream novelty pass
- `--skip-embeddings --skip-llm --skip-clustering --skip-graph-embed --skip-faiss`
  keeps the run CPU-safe while still giving:
  - Stage 5 NER/scope output
  - `discourse-wiring.json` with scope + wire + port + label + comment records
  - `candidate-new-terms.jsonl` with seed-aware novelty labels
  - `learned-term-dictionary.jsonl` with provisional OED-style entries
  - `learned-structure-candidates.json` with reusable residual signatures
    (coarse cluster signature + `full_signatures` per cluster + `predicted_kind`)
  - `learned-structure-summary.json` with structure loss and seed-match stats
  - `audit-summary.json` with per-paper inhabited/outer/straddled term
    counts and tree-aware depth distribution on a random sample of entities
  - `qc-preregister.json` with historical baseline checks against archived
    `mark2` runs plus the new structure-learning gates and headline
  - Stage 5c classical technique extraction
  - Stage 5d classical paper hypergraphs
  - Stage 9a geometry

Two new flags with safe defaults — only override if Rob wants:

- `--stage5-loss-log-interval N` (default 500): print a running loss
  snapshot every N entities during the Stage 5 loop. Set to 0 to silence.
- `--audit-sample-size K` (default 30): how many entities the inline
  end-of-job audit classifies. 30 is cheap; 0 disables the audit entirely.

Expected post-run checks (failure of any indicates the run isn't valid):

- **Provenance**: Stage 5 reports `text source: eprint=N, abstract-fallback=0`;
  Stage 5c and Stage 5d report eprint-only text use; Stage 9a reports
  nonzero eprint coverage. Fallback to abstracts on a paper-backed batch
  is a provenance failure — stop and investigate.
- **Stage 5 baseline output**: `Discourse coverage`, `wires=`, `ports=`,
  `labels=` numbers print.
- **Periodic loss snapshots**: every 500 entities, lines like
  `[N/T] loss snapshot: term entities_with_ner=…, structure uncovered=…
  (with_terms=…), interaction free_floating=NN.N%, seed_matches=…`.
- **Audit summary**: at end of Stage 5, lines like
  `Audit sample (30 entities): inhabited=…, outer=… (frontier NN.N%),
  straddled=… -> audit-summary.json` followed by a depth distribution
  line `Audit depth distribution: d1:N, d2:N, d3:N, … (max_depth=N)`.
- **Structure-learning headline** (when `--discover-structures` is on):
  `Structure-learning headline: discovered=N, classified=M (label=…,
  scope=…, wire=…), gated=K`. The `gated` count is what the next run
  could replay via `--discover-structures-seed-json` to test transfer.
- **QC**: `Preregistered QC: pass|warn|fail (N gates) -> qc-preregister.json`
  with new gates `structure_learning_capture`, `gated_pattern_yield`,
  `structure_seed_replay` (if seed JSON was loaded).
- **Comment scopes**: `Comment scopes: N comment/unreachable records
  across M entities` (24% of source by chars is typical; >50% may indicate
  a corpus with heavy meta-content).

## 5a. Distributed Proofreaders loop

For structure-learning work, the current recommended audit loop is:

```bash
python3 scripts/build-uncovered-sentence-audit.py
```

That script:

- chooses fresh papers not already used in the daisychain ledger
- runs Stage 5 discourse detection over each
- enriches each uncovered residual with known-term hits from the
  NER kernel and a normalized structure-seed signature
- buckets cross-paper signatures by their discourse-verb backbone
- gates the resulting candidates into `learned-discourse-patterns.json`
  (default: `paper_count >= 2` and a recognizable `predicted_kind` of
  `scope` / `label` / `wire`)
- advances a ledger so the next run shifts to new papers instead of
  repeatedly tuning on the same examples

The default outputs are:

- `data/showcases/distributed-proofreaders/latest-audit.json` — full
  audit, including per-paper coverage, residuals, term hits, and
  aggregated `structure_seed_candidates`
- `data/showcases/distributed-proofreaders/latest-audit.html` — same as
  HTML for browsing
- `data/showcases/distributed-proofreaders/learned-discourse-patterns.json` —
  gated patterns from this run, deployable into a follow-on audit
- `data/showcases/distributed-proofreaders/daisychain-ledger.json` —
  records advance across runs

### Replay matcher (cross-batch firing detection)

Pass a prior audit JSON to surface cross-batch signature firings:

```bash
python3 scripts/build-uncovered-sentence-audit.py \
  --seed-signatures-json data/showcases/distributed-proofreaders/latest-audit.json \
  --no-advance-ledger
```

The audit then runs an in-order subsequence matcher: every residual in
the current run is checked against the prior signatures. The matched
prior is recorded on each residual as `matched_prior_signature`, and
the run-level report carries `seed_signatures_loaded` and
`seed_matches_applied`. This is how you measure whether learning from
batch A transfers to batch B.

### Promotion loop (learned patterns feed back into detection)

To apply previously-learned patterns as live detectors:

```bash
python3 scripts/build-uncovered-sentence-audit.py \
  --learned-patterns-json data/showcases/distributed-proofreaders/learned-discourse-patterns.json \
  --no-advance-ledger
```

The audit then:

- loads each pattern's regex + `predicted_kind`
- calls `nlab_wiring.detect_learned` over every paper's text
- runs an **anti-clobber filter** (ON by default): only learned records
  whose match span lies outside the union of existing
  scope/wire/port/label spans count toward coverage. Records that pile
  onto already-covered prose are kept for diagnostics but excluded
  from the discourse list so the coverage delta is a clean signal.
- emits `learned_records_emitted_total`, `learned_records_total`
  (kept), `learned_records_clobbered_total` in the report

To inspect all firings including the clobbered ones, use
`--no-learned-anticlobber`.

### Recommended cross-batch iteration

```bash
# 1. Audit batch A → patterns from A
python3 scripts/build-uncovered-sentence-audit.py \
  --paper-id ID1 --paper-id ID2 ... \
  --out-json /tmp/A.json \
  --learned-patterns-out /tmp/A-patterns.json \
  --no-advance-ledger

# 2. Audit batch B with A's patterns → measure lift
python3 scripts/build-uncovered-sentence-audit.py \
  --paper-id ID_N+1 ... \
  --out-json /tmp/B-with.json \
  --learned-patterns-json /tmp/A-patterns.json \
  --no-advance-ledger

# 3. Compare B-with against a B-baseline run (no --learned-patterns-json)
#    on the same paper IDs to read the coverage delta.
```

The intent is to grow a reusable seed kit of structure patterns, not to
paper-special-case the detector around one or two hand-picked texts.
Promotion details and the open loss-of-loss work are tracked in
[`holes/missions/M-structure-seed-promotion.md`](holes/missions/M-structure-seed-promotion.md).

## 5b. Demos to inspect

The QC viewer renders per-paper pages with structural overlays. Both the
v1 baseline (scope-only) and the v2 (tree-aware + inhabited terms +
depth coloring + comment scopes) live under
`data/showcases/`. The files are gitignored regeneratable artifacts;
Rob can rebuild them locally after a run with:

```bash
python3 scripts/build-batch-008-qc-viewer.py \
  --paper-id 0710.3853v1 --paper-id 0802.0600v1 \
  --paper-id 0711.1739v1 --paper-id 0712.4211v1 \
  --out-html data/showcases/batch-008-math-ct-qc-v2.html \
  --out-json data/showcases/batch-008-math-ct-qc-v2.json \
  --out-page-dir data/showcases/batch-008-math-ct-qc-v2-pages
```

Then open:

- [`batch-008-math-ct-qc-v2.html`](data/showcases/batch-008-math-ct-qc-v2.html) —
  index with per-paper frontier counts and kernel-term breakdowns
- per-paper pages under `batch-008-math-ct-qc-v2-pages/` — full overlay
  with nested scope marks, inhabited (purple) vs outer (teal) kernel
  terms, comment/unreachable scopes (grey strikethrough), and a depth
  distribution line

For the structure-learning loop's residual proofreading view:

- [`distributed-proofreaders/latest-audit.html`](data/showcases/distributed-proofreaders/latest-audit.html) —
  uncovered sentences ranked by kernel-term density, grouped by paper
- [`distributed-proofreaders/learned-discourse-patterns.json`](data/showcases/distributed-proofreaders/learned-discourse-patterns.json) —
  gated patterns ready to feed into a next-iteration audit via
  `--learned-patterns-json`

What to look at in v2:

- The legend bar shows the **depth palette** (d1 amber, d2 rose, d3
  violet, d4 indigo, d5+ slate w/ dashed outline) plus the term swatches
  (teal = outer, purple = inhabited, grey strikethrough = comment).
- **Per-paper frontier counts**: `inhabited / outer (scope-development
  frontier) / total`. Outer = candidates for future scope work.
- **Depth distribution** under each paper: `d1:N d2:N d3:N …`. The
  Galois actions paper (`0711.1739v1`) currently shows terms nested
  six levels deep; the flat renderer would have hidden ~80% of those.
- The `0802.0600v1` (Balanced category theory) page is the clearest
  example of high frontier ratio — most of its kernel terms sit in
  unannotated prose, which is exactly the structure-learning target.

## 5c. Preregistration: what we expect to learn from the renewed mining

This section commits in writing — before Rob's first batch lands — to
what we predict and how we'll measure it. Anything not on this list
that surfaces will count as an unexpected finding rather than a
confirmation.

### Metrics we will track per batch

1. **Aggregate frontier ratio** (`audit_outer_terms / audit_total`)
   over the audit sample. Reported in the QC headline.
2. **Inhabitation rate** between consecutive batches: `(outer_before
   - outer_after) / outer_before` on the same paper set when
   `--discover-structures-seed-json` carries forward. A positive value
   means learned patterns are migrating residual terms into scope.
3. **Gated pattern yield** (`headline_summary.gated_for_promotion`):
   candidates that cleared `paper_count >= 2 AND predicted_kind set`.
4. **Depth distribution**: `audit_depth_distribution` and
   `audit_max_depth`. We expect mass at d1–d3 and a long thin tail
   into d4–d6.
5. **Comment-scope share**: total `comment/unreachable` chars / total
   source chars. A sanity check on corpus quality.
6. **Free-floating term ratio** during the run: prints every 500
   entities. We expect it to stabilize (not climb) as the batch progresses.

### Predictions for batch sized 1000+ papers

- **Gated pattern yield ≥ 5**. At 30-paper sample we got 1; scaling
  argument: more papers → more cross-paper recurrence clears the
  `paper_count >= 2` gate. If yield is < 2, the prefilter is too
  strict or the discourse-verb taxonomy is missing common cues.
- **Max depth ≥ 5** on at least 10% of sampled papers. Real arXiv
  math.* papers routinely nest `env/proof > bind/let > bind/typed >
  constrain/relation`. Lower max-depth suggests detector regressions.
- **Frontier ratio between 15% and 40%** on the audit sample
  aggregate. Below 15% means scopes are already covering everything
  (good but suspicious); above 40% means the detector is mostly
  watching from the sidelines (which is what we want to reduce).
- **Comment-scope share between 5% and 30%** of source chars.
  Outside that range suggests the comment detector is mis-firing or
  the corpus has unusual meta-content.
- **At least three distinct `predicted_kind` values** in the gated
  candidates (scope, label, wire all represented). If only one kind
  shows up, the discourse-verb taxonomy is funneling everything
  through one category and the classifier needs widening.

### Patterns we specifically expect to clear the gate

From the 30-paper batch-008 evidence and the audit's 9-paper run, we
got `be obtain` (label) and `be introduce` (label) at small N. At
1000+ papers we expect to additionally see:

- `we prove that be` (label) — recurrent theorem-statement frame
- `let be` (scope) — basic binding cue when the math content uses
  `\let` macros or short let-be constructions
- `we study and` (label) — paper-level framing seen in introductions
- `assume that` / `suppose that` (scope) — proof-internal binding
- `notice that` / `observe that` (wire) — discourse connectives

If none of those clear `paper_count >= 2` at 1000 papers, something
is wrong with the aggregation prefilter, not with the corpus.

### Closing the loop across batches

Rob's first batch produces `learned-structure-candidates.json`. The
second batch loads that as `--discover-structures-seed-json` and we
read two new numbers from the QC headline:

- `seed_matches_applied` — how many residuals in batch B were
  recognized via signatures learned from batch A. Non-zero confirms
  cross-batch transfer.
- `entities_with_seed_matches` — diversity of where matches fired.

The first batch with non-zero replay is the "the loop closed on real
data" milestone.

### Stopping rule

Iteration stops when **inhabitation rate per cycle drops below 1%**
across a representative random sample, OR when the gated-pattern
yield stops growing across two consecutive batches. At that point
the next bottleneck is the detector itself (more cue verbs, more
scope shapes), not the learning loop.

The full discipline is documented in
[`holes/missions/M-structure-seed-promotion.md`](holes/missions/M-structure-seed-promotion.md)
section 7.

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

## 9. Symbol-grounding readiness for 5K Arxiv slice (Gate P6)

Status: **all preparation gates passed; awaiting Joe's go/no-go.**

The symbol-grounding pipeline (the Stage-5 CPU portion that runs
end-to-end per paper, before the GPU components) has cleared its
five preparation gates from
`holes/missions/M-symbol-grounding-scaling-plan.md`.

### Gate-by-gate evidence

**Gate P1 — Wikipedia gold extractor.** Built
`scripts/build-grounding-gold-wikipedia.py`. 7027 entries, 13376 gold
pairs. Combined with PM (469 pairs) and ProofWiki (14939 pairs);
nLab held-out as test corpus (no train signal).

**Gate P2 — Combined gold ≥ 1500 pairs across ≥ 800 entries.**
17475 cross-source gold pairs after F4 (literature-lifted strategy
merge proposer returned "no merges" → strategies are independent,
not redundant).

**Gate P3 — Strategy gating + canon-ancestry comparison.** Three
gating rounds shipped via `--disable-strategy`; ancestry-match mode
via `--match-mode ancestry --ancestry-index data/canon-ancestry-pm.json`.

**Gate P4 — Precision ≥ 30% with recall stable.** PM gold, 200
held-out entries, ancestry-match mode, no topic priors:

| | precision | recall | F1 |
|---|---:|---:|---:|
| best single strategy (let-binding) | 29.0% | – | – |
| weighted-avg per strategy | 13.6% | – | – |
| **arbitrated (Bayesian)** | **32.4%** | **22.1%** | **26.3%** |

Above-target precision; arbitrated recall stable.

Topic-aware priors (MSC + SE corpus) are wired through
`bayesian_grounding.combine_strategy_votes` via `context_factors` but
are *configurable per use case*:

| eval surface | priors? | result |
|---|---|---|
| arxiv coherence vs nLab vocab | ON | +13.2pp (47.9% → 61.1%) |
| PM gold precision/recall | OFF | priors hurt recall ~5pp |

PM gold has cross-MSC references (a logic-tagged entry citing a
topology canon) that the MSC prior wrongly suppresses; arxiv papers
are genuinely topic-coherent, so the prior helps. Default for the
production run: priors ON.

**Gate P5 — 100-paper production shakedown.** Two contrasting pools
processed end-to-end via `scripts/p5-production-shakedown.py`:

| | broad-non-CT (mfuton-001) | CT-pure |
|---|---:|---:|
| papers processed | 99 | 88 |
| wall time | 23.6s | 29.3s |
| throughput | 4.19 papers/s | 3.0 papers/s |
| total emissions | 12530 | 12391 |
| mean canons / paper | 126.6 | 140.8 |
| unique canons | 1112 | 1122 |
| in-nLab (all emissions) | 49.1% | 66.4% |
| in-nLab (high-confidence, p≥0.5) | 42.6% | 55.5% |
| RSS end | 648 MB | 653 MB |
| malformed bindings | 0 | 0 |
| MSC prior canons added (online) | 351 | 493 |

All P5 checks pass: no OOM, no crashes, no malformed bindings,
progress logging fires, every strategy represented, online MSC-prior
updates land cleanly. The 17.3pp in-nLab gap is correct-direction
(nLab is CT-skewed; broad papers still hit 49% because general math
vocabulary is shared) — the pipeline isn't CT-overfit.

### Scale economics for the 5000-paper slice

At 3–4 papers/sec single-threaded CPU, the symbol-grounding portion
alone is ~25 min for 5000 papers wall-time-equivalent (independent
of GPU stages). The superpod parallelises across many workers, so
real wall-clock will be dominated by Stage 3 LLM / Stage 6 anyway.

What 5000 papers buys us (per scaling-plan §3):
- ~half a million per-binding fingerprints into the canon store
- ~5000 MSC-prior updates (high-confidence emissions) — meaningful
  shift in P(canon | MSC) distribution for canons that surface often
- enough strategy-emission volume for the per-strategy reliability
  posteriors to tighten well below 5% credible-interval width

### Recommended invocation

Use the **superpod (full GPU)** command in §5 as the base, then add
the topic-prior flags below. The cleaned NER kernel
(`data/ner-kernel-clean.tsv`, 18937 entries) filters out the
"stable → StableMarriageProblem" garbage shape that Joe spotted in
batch-008. `--update-msc-prior` writes the online-EM-updated MSC
prior at end of run for the next batch to pick up.

```
--ner-kernel data/ner-kernel-clean.tsv     # already in §5 command
--msc-prior data/topic-prior-msc.json
--se-corpus-prior data/topic-prior-se-corpus.json
--update-msc-prior data/topic-prior-msc-updated.json
--stage6-backend codex                      # already in §5 command
```

(`--stage6-backend codex` and the relative `--ner-kernel` path are
inherited from the §5 superpod-variant; listed here only as a
reminder of why those are non-default vs the laptop-CPU command.)

### Open questions for Gate P6

1. **Which 5000 papers?** The math.CT pool has 9742 eprints; mfuton-001
   has another 5000 broader math.*. Two natural slices:
   - First 5000 of math.CT by arxiv id (oldest first) — keeps
     experiment comparable to the existing batch-008 work.
   - 5000 mixed-domain from mfuton-001/002 — exercises priors on
     the broader vocabulary the pipeline will see in arxiv-at-large.
2. **Priors on or off?** Default recommendation: ON for arxiv. Joe's
   call. Trivial to flip via the CLI flags.
3. **Pre-flight canon-store seeding.** We have an aggregate from PM
   + ProofWiki + Wikipedia + nLab (`data/canon-store-pm-pw-wiki/`).
   Use it as starting prior for arbitration, or start cold?

### What we'll learn from the run

- whether the 61.1%-in-nLab coherence number (30-paper sample) holds
  on a 5000-paper run, or decays as topical diversity broadens
- whether the MSC online-EM updates converge or wander
- per-strategy reliability tightening — the credible-interval widths
  will tell us which strategies need more evidence vs which are
  already well-characterised

## 10. Older superpod docs

Some older repo-root superpod notes refer to the earlier StackExchange /
MathOverflow production run rather than the current arXiv batch lane.

In particular:

- [KNOWN_LIMITATIONS.md](/home/joe/code/futon6/KNOWN_LIMITATIONS.md)
- [SUPERPOD-RERUN-NOTES.md](/home/joe/code/futon6/SUPERPOD-RERUN-NOTES.md)

Those remain useful historical records, but they are not the authoritative
state summary for the current arXiv mark2/mark3 work.

## 11. Short version

If you need the one-paragraph summary:

The live superpod workflow is still `mark2` through the Chicago
transfer/archive lane, with the runner materially upgraded for arXiv work.
Paper mining must be eprint-backed: Stage 5 NER/scope detection, Stage 5c
technique extraction, Stage 5d paper hypergraphs, and Stage 9a geometry
are only trustworthy when the manifest confirms eprint usage. The single
canonical command lives in section 5 (drop the `--skip-*` flags on
superpod, keep them on the laptop). On top of that baseline, Stage 5 now
runs a structure-learning loop that mines term-dense uncovered residuals
into discourse-verb-classified candidate signatures (`learned-structure-
candidates.json`), can replay a prior batch's signatures via subsequence
match (`--discover-structures-seed-json`), reports periodic loss snapshots
during the run and an audit summary at the end (`audit-summary.json`),
and gives the QC report a headline block showing what was learned. The
preregistration in section 5c commits to which numbers we'll track and
what we expect them to look like before the first big batch lands; the
demo pointers in section 5b show what the per-paper visualization looks
like under the new tooling. Section 9 is the symbol-grounding readiness
report (Gate P6 hand-off): all five preparation gates (P1–P5) have
passed — arbitrated precision 32.4% with 22.1% recall on PM gold,
nLab coherence 61.1% on arxiv math.CT, production shakedown clean on
both broad-non-CT and CT-pure 100-paper pools — awaiting Joe's go/no-go
on a 5000-paper Arxiv slice and choice of slice/prior configuration.
