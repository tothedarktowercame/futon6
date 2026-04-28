# Excursion: mfuton silver — legacy TeX normalization from modern silver targets

**Date opened:** 2026-04-28
**Status:** open / planning
**Owner mission:** `futon6/holes/missions/M-superpod-mark3.md`
**Theory sister excursion:**
`futon3/holes/excursions/E-old-arxiv-block-detection.md`
**Metrics sister excursion:**
`futon3/holes/excursions/E-substrate-metrics.md`
**Audience:** Joe (pipeline), Rob (batch execution / QC)

## Purpose

Turn the mfuton lineage into a **silver-target corpus** for improving
legacy arXiv source handling without weakening any mark2/mark3
invariants.

The practical question is not "can we make old papers look modern by
wishful parsing?" It is narrower:

- given that mfuton papers already produce good claim/proof structure,
- and given that older arXiv papers often carry full source but in
  older macro idioms,
- can we add a normalization layer that maps legacy source into the
  canonical block vocabulary the existing parser already understands?

This excursion is the substrate-side implementation plan for that
question.

## Why "mfuton silver"

The mfuton batches are not gold annotation. They are **silver**:
high-quality pipeline outputs produced from full modern eprint sources
with the current extraction stack.

They are useful because they give us a concrete target for what "good
enough structural extraction" looks like in this system:

- non-trivial `with_claim_blocks`,
- good theorem/proof linkage,
- dense `paper-hypergraphs.json`,
- stable `T_total` / F2 geometry downstream.

The silver target is therefore not a theorem statement database. It is
the **output behavior** of the current stack on papers whose source it
already parses well.

## Hypothesis

Most of the vintage gap is front-end, not back-end.

More precisely:

1. `src/futon6/paper_hypergraph.py` and
   `src/futon6/theorem_extraction.py` already work when the source uses
   explicit theorem/proof environments.
2. Older arXiv papers fail disproportionately because they use legacy
   macro conventions, plain-TeX theorem heads, or multi-file source
   layouts that are not normalized into that environment vocabulary.
3. Therefore the right intervention is a **legacy-source
   normalization pass** before Stage 5d's classical parser, not a
   workaround in later stages.

## Non-goals / invariants

This excursion is invalid if it achieves recall by violating substrate
invariants. In particular:

- Do not weaken the eprint guard or silently fall back to abstract-only
  paper stages.
- Do not hallucinate theorem/proof blocks without provenance.
- Do not bypass the classical parser by stuffing LLM guesses directly
  into final hypergraphs as if they were native blocks.
- Do not special-case specific batches in a way that breaks global
  coherence.

If the legacy source cannot support a trustworthy block projection, the
result must be an explicit degraded/failure record, not a fake success.

## Implementation shape

The intended architecture is:

1. Load full source exactly as today from the eprint path.
2. Run a **legacy normalization pass** on a copy of the source.
3. Feed the normalized text into the existing environment-based parser.
4. Preserve provenance saying which blocks came from native environments
   vs synthesized legacy normalization.
5. Measure improvement with the same downstream metrics already used for
   mfuton / regular comparisons.

The normalization pass should aim to produce canonical markers such as:

- `\begin{theorem}...\end{theorem}`
- `\begin{lemma}...\end{lemma}`
- `\begin{proposition}...\end{proposition}`
- `\begin{corollary}...\end{corollary}`
- `\begin{proof}...\end{proof}`

while preserving original spans and the reason each synthesized block
was introduced.

## Work packages

### WP1. Failure taxonomy on staged early batches

Sample the early batches Joe is staging and classify the front-end
failures. Expected buckets:

- custom theorem aliases via `\newtheorem{thm}{Theorem}` and friends
- AMS/plain-TeX forms such as `\proclaim`, `\demo`, `\enddemo`
- prose theorem heads (`Theorem 2.1.` / `Proof.`) without env syntax
- sectioning and `\input` / `\include` stitching failures
- source files whose body is too degraded to normalize safely

Deliverable:
- a short bucketed report with counts and 3-5 representative examples
- explicit "do not attempt" classes if some source families are too noisy

### WP2. Canonicalization of declared theorem macros

Before doing any fuzzy prose detection, handle the cheap structural win:

- parse `\newtheorem` declarations
- map aliases like `thm`, `lem`, `prop`, `cor`, etc. to canonical claim
  environments
- rewrite those alias environments in a normalized copy of the source

This is the highest-confidence path because the source itself declares
the semantics.

Deliverable:
- deterministic rewrite layer with tests on synthetic fixtures and a few
  real legacy examples

### WP3. Plain-TeX / prose theorem-head detection

For sources that still have no blocks after WP2:

- detect theorem-like statement heads
- detect proof starts and proof terminators
- synthesize block markers only when local evidence is strong enough

Suggested initial gates:

- head appears near line start, not mid-sentence
- body spans at least one full sentence
- paired proof marker exists within a bounded window, or the statement
  is otherwise structurally isolated
- external-reference contexts (`By Theorem 3`, `[12, Theorem 4.2]`) are
  explicitly excluded

Deliverable:
- conservative v0 detector tuned for precision first

### WP4. Provenance and artifact shape

Extend the intermediate artifact shape so normalized legacy blocks are
traceable:

- native environment block
- alias-expanded environment block
- prose-synthesized block

The consuming hypergraph should be able to distinguish these cases in
metadata, even if they project to the same node type.

Deliverable:
- metadata schema change proposal plus runner integration points

### WP5. Acceptance harness against mfuton silver and old-batch reality

Evaluate on two fronts:

- **Positive control:** mfuton batches must not materially regress under
  the normalization pass.
- **Target set:** older staged batches should show improved structural
  recall.

Primary metrics:

- `stage_status.paper_hypergraph.with_claim_blocks`
- theorem/proof linkage rate
- `paper_hypergraphs.json` density
- F2 / `T_total` summary via the existing math-corpus metric scripts

Success is not "matches mfuton." Success is:

- clear lift over the old-batch baseline
- no hidden regression on mfuton
- explicit accounting of synthesized-vs-native structure

## Proposed acceptance targets

These are deliberately conservative and should be revised once WP1
lands.

1. On mfuton positive-control batches:
   - no more than 5% relative drop in `with_claim_blocks`
   - no more than 5% relative drop in theorem/proof linkage

2. On an early-batch evaluation slice:
   - substantial lift in `with_claim_blocks` over current baseline
   - F2 improves meaningfully from the older-arXiv eprint-on floor
   - synthesized legacy blocks remain a minority of all detected blocks
     on papers where native environments already exist

3. On manual audit:
   - false positives are legible and classifiable, not opaque
   - every synthetic block can be traced back to a source cue

## Current status (2026-04-28)

The first implementation wave is now landed in
`src/futon6/legacy_tex_normalize.py`:

- declared `\newtheorem` alias expansion
- declared `\newenvironment` theorem/proof wrapper expansion
- standard undeclared env-alias recovery for names like `thm`, `lem`,
  `defi`
- structured wrapper-macro expansion for `\be{...}` / `\ee{...}` style
  shorthands
- command-style theorem/proof alias expansion for `\let\lem\lemma`,
  `\let\eth\endtheorem`, `\let\prf\proof`, etc.
- exact `\paragraph{Lemma.}`-style claim synthesis with explicit
  `prose_synthesized` provenance

Empirical status from local replay:

- on a 200-paper old-source sample, papers with at least one claim block
  rose from `119` to `187`
- on `ct-validation/arxiv-superpod-mit-eprint-pilot-500`, among the 495
  locally replayable eprints, papers with claim blocks rose from `230`
  to `391`

## Current do-not-attempt floor

The residual after the first implementation wave should currently be
treated as explicit floor, not as license for unbounded parser creep.

Current floor classes are:

- papers with no visible theorem/proof surface in the available source
  at all
- sources whose remaining structure is custom expository formatting
  rather than theorem-block syntax
- custom command families whose heads and terminators are not yet
  recoverable by the existing high-confidence rules
- source layouts whose body is effectively missing or too degraded for a
  trustworthy projection

The correct next step for any of these classes is a new audited bucket
proposal with examples and acceptance criteria, not ad hoc widening of
the normalizer.

## Likely code touch points

- `src/futon6/paper_hypergraph.py`
- `src/futon6/theorem_extraction.py`
- `scripts/superpod-job.py`
- tests for paper-stage extraction and eprint-backed fixtures

Possible new files:

- `src/futon6/legacy_tex_normalize.py`
- fixtures covering alias environments and plain-TeX theorem/proof heads

## Sequencing relative to mark3

This is **not a prerequisite for mark3 v0**.

Mark3's current job is:

- arXiv-aware Stage 3
- eprint-defaulting
- Stage 6 coverage / slot distinctness
- geometry artifact

`E-mfuton-silver` is the follow-on if the older-arXiv vintage gap
still matters after mark3 is instantiated and regular batches are worth
lifting.

## Cross-references

- `futon3/holes/excursions/E-old-arxiv-block-detection.md` — theory-side
  statement of the same vintage-gap problem
- `futon3/holes/excursions/E-substrate-metrics.md` §1 — empirical F2
  gap that motivates this work
- `futon6/holes/missions/M-superpod-mark3.md` — substrate mission this
  excursion hangs from
