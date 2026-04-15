# Preregistration: Batch-002 (Superpod Mark 2)

**Date written:** 2026-04-15 (prior to Rob executing batch-002)
**Parent missions:** M-superpod-mark2 (execution), M-paper-reverse-morphogenesis (design)
**Batch size:** ~5,000 arXiv math papers (of 570,209 in manifest)
**Pipeline version:** futon6 @ commit 43e506d (main, 2026-04-15)
**Author:** Joe Corneli (+ Claude as drafting partner)
**Executor:** Rob, on the UT Austin superpod (8× A100 80GB, one node)

## What this document is, and what it is not

This is a **scientific preregistration** for the ML-bearing claims of
batch-002. It commits in advance to (a) what we predict, (b) what counts
as evidence for or against each prediction, (c) which arms of the
pipeline we are comparing, (d) what we will not do during analysis, and
(e) what follow-on actions are predetermined by which results.

The purpose is to prevent the three failure modes that plague informal
ML evaluation:

1. **Post-hoc storytelling** — rationalizing whatever we see as "what we
   expected all along."
2. **Garden of forking paths** — running many analyses, reporting the
   ones that look good, burying the rest.
3. **Moving the goalposts** — redefining "success" after seeing results.

This is *not* a design document (that is M-paper-reverse-morphogenesis)
or an execution runbook (M-superpod-mark2). It is the contract we make
with ourselves about how batch-002 will be interpreted.

Each subsequent batch gets its own preregistration, informed by the
prior batch's results. Batch-003's preregistration will reference this
document as its baseline.

## Background (one paragraph)

Mark 1 showed that BGE retrieval on arXiv math.CT papers accidentally
found a paper whose *technique* named the problem for one of our
learn-to-swim canaries (C7: Pettis integral as monad algebra map). Mark
2 is the bet that this accident can be made the default case, by
extracting technique-level terminology and the argumentative skeleton
of each paper, reconstructing what problem each paper actually solves
from those features, and scoring the reconstruction against the paper
itself. The pipeline improvements land incrementally across batches; the
reconstruction corpus eventually trains a forward problem-solving model
(downstream consumer: M-apm-solutions). Batch-002 tests the extraction
feasibility claim at scale and pilots the reconstruction-sufficiency
claim on a small manual sample.

## Claims under test

Batch-002 makes four claims. Only the first is directly testable at the
batch scale; the second is pilot-testable on a manual sample; the third
requires at least one more batch; the fourth is explicitly deferred.

### Claim 1 — Extraction is feasible and informative at arXiv scale

**IF** we run the new stages 5c (technique-level NER) and 5d (paper
hypergraph) over 5,000 arXiv math papers with full eprint LaTeX
sources,

**HOWEVER** the classical parser may be too strict (missing non-eponymous
technique phrases), the LLM arm may silently fail or hallucinate, and
the LaTeX environment parsing may miss papers that use non-standard
theorem macros,

**THEN** we should see: (a) a nontrivial technique vocabulary per paper;
(b) a substantive argumentative-skeleton hypergraph for the majority of
papers; (c) nonzero contribution from both extraction arms in both
stages; (d) high eprint-load coverage.

**BECAUSE** if any of (a)–(d) fail at batch scale, the reconstruction
claim (§ Claim 2) has no inputs to work from. Passing this claim is a
necessary condition for the whole program.

### Claim 2 — The extracted features are sufficient for reconstruction

**IF** a human analyst (Joe), blind to the paper's body prose, is shown
only the stage-5c technique list, the stage-5d hypergraph (nodes +
edges), and the stage-5 concept list for a paper,

**HOWEVER** the features may be too sparse to reconstruct the main
result, or too noisy to disentangle the central argument from tangential
material, or rich enough in aggregate but with the argumentative
"spine" missing,

**THEN** the analyst should be able to produce a recognizable sketch
(main result + 3–5 derivation steps with technique attributions) that
matches the actual paper when revealed.

**BECAUSE** if a human cannot reconstruct from these features, the LLM
stage 6 cannot either. A pilot failure here falsifies the stage-6
design *before* we spend compute on implementing it.

### Claim 3 — The learning loop produces measurable improvement over batches

**IF** we apply the four update channels (pattern library, technique
vocabulary, hypergraph features, prompts) between batches based on the
gaps observed in batch-002,

**HOWEVER** the updates may be cosmetic (not addressing the real gaps),
or over-fitted to batch-002's idiosyncrasies, or the improvements may
be within noise bands that masquerade as signal,

**THEN** batch-003, run on a different 5,000-paper sample with the
updated pipeline, should score higher on the preregistered metrics than
batch-002.

**BECAUSE** the "mining approach *is* ML" claim rests on this
batch-over-batch improvement curve. Without it, the learning loop is a
narrative, not a system.

**Status:** Not testable in batch-002 alone. This claim is preregistered
here so that batch-002's numbers serve as the committed baseline against
which batch-003 will be compared. Changing the metric definitions
between batches invalidates this claim's test — see § Anti-commitments.

### Claim 4 — The reconstruction corpus trains a useful forward model

**IF** the reconstruction corpus, once stage 6 and stage 11 are
implemented and running at scale, is used to train a model that takes
(problem, techniques, patterns) → sketched argumentative structure,

**THEN** that model should propose usable technique-and-structure
candidates for novel problems.

**Status:** Not tested in batch-002, and not testable against any
specific external problem set from this batch's output. Batch-002 is
a ~5,000-paper round-robin slice of the arXiv math manifest; its
content is uncorrelated with any particular downstream evaluation
target (e.g., APM prelim topics, Mathlib's LeanDojo coverage,
FrontierMath problem areas). The batch's role here is to produce
*input* for forward-model training — not to test whether the trained
model is useful against a specific target. Evaluation of Claim 4 is
the responsibility of the downstream consumer mission
(M-apm-solutions), and requires a corpus whose content has been
accumulated or filtered to cover the evaluation target's topics. See
M-superpod-mark2 § "Open items — future mark2" for the topic-targeted
batch question that Claim 4's eventual evaluation depends on.

Preregistered here so that the thread is documented end-to-end, not
so that batch-002 is taken to bear on it.

## Operational definitions

Each abstract claim resolves to specific numeric thresholds measured
from specific output artifacts.

### For Claim 1 — extraction

Metric E1.1 — **technique count per paper**. Computed from
`techniques.json` as `len(record["techniques"])` for each record.

Metric E1.2 — **technique arm intersection rate**. Across the whole
batch: `(#terms with extraction_source="both") / (#terms total)`.

Metric E1.3 — **per-paper hypergraph claim density**. For each paper
in `paper-hypergraphs.json`, the number of nodes of type `claim` (i.e.,
numbered theorems, lemmas, propositions, corollaries).

Metric E1.4 — **theorem–proof linkage rate**. Across papers that have
at least one claim node: `(#derivation edges) / (#claim nodes)`. A
claim without a linked proof has no derivation edge contributing to the
numerator.

Metric E1.5 — **hypergraph edge arm attribution**. Edge counts by
provenance: `classical`, `llm`, `both`.

Metric E1.6 — **eprint load rate**. From `stage_status.paper_hypergraph.
text_source_counts`: `eprint / (eprint + abstract)`.

### For Claim 2 — reconstruction-sufficiency pilot

Metric R2.1 — **analyst reconstruction match rate**. Protocol in
§ "Manual reconstruction pilot" below. One binary outcome per paper in
the 10-paper sample: "recognizable reconstruction" (yes/no), judged by
the preregistered matching criteria.

Metric R2.2 — **failure-mode distribution**. For each "no" in R2.1,
which of the four prespecified failure modes applies: (i) missing key
technique, (ii) hypergraph too sparse, (iii) hypergraph dense but
misleading, (iv) other (prose-based).

### For Claim 3 — baseline

Batch-002's values on E1.1, E1.2, E1.3, E1.4, E1.5, E1.6, R2.1 are
**frozen** as the batch-003 comparison baseline.

## Pre-committed thresholds

These thresholds are declared before any batch-002 output is inspected.

| Metric | Confirmatory threshold | Notes |
|---|---|---|
| E1.1 (techniques/paper) | median ≥ 5, mean ≤ 100 | math 10–30 pp. Outside this band signals over- or under-extraction. |
| E1.2 (arm intersection) | ≥ 0.20 | Both arms extract from same signal. If < 0.10, one arm is broken. |
| E1.3 (claims/paper, eprint-loaded) | median ≥ 3 for 70% of eprint-loaded papers | Papers without numbered theorems are fine; most math papers have ≥ 3. |
| E1.4 (theorem–proof linkage) | ≥ 0.60 | Most theorems have explicit proofs. Low rate signals LaTeX-parsing failure. |
| E1.5 (edge provenance) | `llm > 0` and `both > 0` | Both arms contribute; neither dominates 100/0. |
| E1.6 (eprint load rate) | ≥ 0.95 | Kept high because the mark2 coordinator bundles eprints. |
| R2.1 (pilot match rate) | ≥ 6 / 10 | Weak test — if humans can't, LLMs can't. |

A **confirmatory pass** on Claim 1 requires E1.1, E1.3, E1.4, E1.6 to
meet threshold. E1.2 and E1.5 are confirmatory-but-diagnostic: failure
doesn't block Claim 1, but points at which update channel needs
attention before batch-003. R2.1 is Claim 2's only binary test.

## Experimental arms — what batch-002 compares

Two extraction dimensions run in parallel with per-item arm provenance
recorded:

1. **Stage 5c arm** (`--technique-ner-arm both`): classical regex
   extraction + LLM few-shot extraction. Merge marks intersection as
   `both`, unique contributions as `classical` or `llm`.
2. **Stage 5d arm** (`--paper-hypergraph-arm both`): classical LaTeX
   block parsing + LLM implicit-edge pass. Merge marks duplicate
   edges as `both`, unique LLM edges as `llm`.

**Confirmatory comparisons** (preregistered):

- E1.2 tests whether the two stage-5c arms find overlapping terms
  (intersection rate).
- E1.5 tests whether the stage-5d LLM arm contributes uniquely
  (nonzero `llm` count) and confirms classical (nonzero `both` count).

**Exploratory comparisons** (recorded but not powering any claim):

- Per-sub-discipline variation (math.AP vs math.CT vs math.PR): which
  sub-disciplines do the arms perform differently on?
- Term-type stratification: do eponymous techniques ("Wantzel's
  theorem") skew classical? Do non-eponymous techniques ("spectral
  sequence computation") skew LLM? Answer is expected yes, but
  recording magnitudes informs future classical-pattern work.
- Eprint vs abstract fallback: for the small fraction that falls back
  to abstract (< 5% expected), how much worse is the extraction? This
  bounds how urgent it is to improve eprint loader robustness.

Exploratory analyses are allowed to generate *new* hypotheses, which are
then preregistered for batch-003. They cannot be used to defend or
attack Claim 1 retroactively.

Two dimensions that the spec names but batch-002 does not vary (both
collapse to a single arm for this batch):

- **Stage 6 reconstruction passes** — stage 6 is not yet rewritten for
  papers; batch-002 ships with the old SE-focused stage 6 (which will
  produce garbage on arXiv inputs but is harmless). Multi-pass
  comparison is deferred to the batch where the rewritten stage 6
  lands.
- **Stage 3 pattern library** — `flexiarg-only` (no mined patterns
  yet). The `flexiarg + mined` arm is meaningless for batch-002 because
  the mined library is empty on the first paper batch. This arm's
  comparison starts in batch-003.

## Manual reconstruction pilot (Claim 2 test)

**Sampling.** The pilot sample is 10 papers selected by seeded random
from batch-002's output. Seed is committed in advance:
`hash("batch-002 pilot", sha256)[:8]`. Papers selected *before* any
output is inspected.

**Pilot protocol** (per paper):

1. A helper script prepares a "pilot packet" containing only: paper's
   extracted techniques from `techniques.json`, paper's hypergraph
   nodes and edges from `paper-hypergraphs.json`, paper's concepts from
   `ner-terms.json`. No full prose. Paper title is shown (can't be
   hidden without defeating the test of "given this problem framing...").
2. Joe attempts, working alone and without re-reading the paper
   body, to produce a sketch: **main result statement** (1–2 sentences)
   plus **3–5 derivation steps**, each labelled with a technique from
   the extracted list.
3. Joe writes the sketch, timestamps it, commits it to
   `holes/missions/data/PREREG-batch-002-pilot/<arxiv-id>.md`.
4. *Only then* Joe reads the actual paper.
5. Joe judges whether the sketch is a "recognizable reconstruction"
   by the matching criteria below. Commits the judgment with
   rationale.

**Matching criteria** ("recognizable" = all three):

- The sketched main result mentions the paper's actual main result's
  mathematical object *and* its principal property. (Stricter than
  "mentions the object"; weaker than "states it precisely.")
- At least 3 of the sketched derivation steps correspond to distinct
  steps in the paper's actual argument, in any order.
- No derivation step in the sketch is factually contradicted by the
  paper (which would mean the hypergraph misled the reconstruction).

**Failure-mode taxonomy** (for R2.2):

- (i) **Missing key technique**: the paper's central technique wasn't
  in the stage-5c output. Signal → stage 5c needs pattern/prompt work.
- (ii) **Hypergraph too sparse**: insufficient derivation edges or
  claim nodes to infer argument structure. Signal → stage 5d needs
  edge-type expansion or classical-parser tuning.
- (iii) **Hypergraph dense but misleading**: enough edges, but they
  don't point at the argument's spine. Signal → stage 5d edge
  *quality* (not quantity) needs work.
- (iv) **Other** (prose-based, e.g., result is qualitative and doesn't
  reduce to a theorem statement). Signal → the extraction approach
  may have a scope limit.

**Pre-committed judgment blinding**. The sketch must be committed to
the repo *before* Joe reads the paper. Git timestamps enforce this.
Sketches that weren't committed before paper-read are excluded from
R2.1 count — no backfill permitted.

## Required artifacts — what the run must produce

For the preregistered analyses to be runnable, batch-002's output
directory must contain:

| File | Required | Purpose |
|---|---|---|
| `embeddings.npy` | yes | Stage 2 output; used for cross-batch BGE comparisons |
| `entities.json`, `relations.json`, `tags.json`, `stats.json` | yes | Stage 1 artifacts |
| `ner-terms.json` | yes | Stage 5 concept NER |
| `scopes.json` | yes | Stage 5 scope detection |
| `techniques.json` | **yes (new)** | Stage 5c output — Claim 1 depends on it |
| `paper-hypergraphs.json` | **yes (new)** | Stage 5d output — Claim 1 depends on it |
| `stage_status.*` in manifest | yes | `text_source_counts`, `arm_counts`, `edge_provenance` are read directly |
| Run stdout log | yes | Preserved by Rob as `batch-002-run.log` |

If any of the **yes (new)** files is missing or malformed, the batch is
invalid for Claim 1 testing — a re-run is required before analyses
proceed.

Note: `experiment_meta.json` (spec §"Experimental design") is not
produced by batch-002. The `stage_status` block records the same
per-stage arm counts; we accept the lesser instrument for this batch
and commit to shipping `experiment_meta.json` starting batch-003, when
stage 11 and the learning-loop script land.

## Anti-commitments (what we will NOT do)

1. **No prompt tuning during batch-002 analysis.** Stage 5c and 5d
   LLM prompts are frozen at commit 43e506d. Any change to a prompt
   post-run is a batch-003 commit, not a batch-002 adjustment.
2. **No regex tuning during batch-002 analysis.** Same applies to
   stage-5c classical patterns and stage-5d LaTeX parsers. Tuning
   observations go into a `batch-002-update-proposals.md` and are
   applied to code only after Claim 1's verdict is final.
3. **No threshold redefinition post-run.** The numeric thresholds in
   §"Pre-committed thresholds" are frozen. If a metric lands at 0.59
   against a 0.60 threshold, that is a *fail*, not a "close enough."
   Recording the near-miss as context for the next preregistration
   is fine; counting it as a pass is not.
4. **No cherry-picking spot-check papers.** The 10-paper pilot sample
   is seeded-random. Spot-checks outside the pilot are exploratory and
   cannot be used to defend Claim 2.
5. **No retroactive reinterpretation of exploratory analyses as
   confirmatory.** If an unplanned analysis suggests a new pattern,
   that pattern is preregistered for the next batch, not cited as
   evidence for the current one.
6. **No selective re-runs.** If the batch has issues on some
   sub-discipline, we don't re-run just that slice with tweaked
   settings and report the clean result. A re-run is either full or
   it is not reported as the batch-002 result.
7. **No ex-post-facto claim reweighting.** If Claim 1 passes on
   4 of 6 confirmatory thresholds, that is a partial pass, not a pass.
   Partial passes are recorded as such in § Results, with explicit
   listing of which thresholds missed.

## Decision tree — what each outcome commits us to

| Outcome on Claim 1 | Outcome on Claim 2 | Committed next action |
|---|---|---|
| Full pass (all 4 thresholds) | Pass (≥ 6/10) | Implement stages 6, 3, 11 as specified; run batch-003 with same arms + new stages. |
| Full pass | Fail (< 6/10) | Halt stage-6 implementation. Diagnose which failure mode dominates (R2.2); revisit the four-layer output schema or the hypergraph edge types accordingly. Batch-003 runs with 5c/5d only, revised. |
| Partial pass | Pass | Apply per-channel updates to the failed thresholds' corresponding channels; re-run batch-003 with no other changes to isolate the effect. |
| Partial pass | Fail | Treat as "not ready for stage 6." Repeat the diagnose-and-iterate cycle on 5c/5d. Pilot re-runs each batch until both claims pass. |
| Full fail (E1.6 < 0.95) | Any | **Infrastructure failure**, not a claim test. Fix eprint loading, re-run batch-002 from scratch with the same preregistration (this document is reused). |
| Full fail (other) | Any | Claim 1 falsified at scale. Rethink the extraction design. Consult M-paper-reverse-morphogenesis §Principles for which principle is suspect. |

## Results section (template — filled in after batch-002 returns)

> **Instructions to future-us:** when batch-002 returns, append results
> here. Do not edit above this line except to append clarifying notes
> in a `> ADDENDUM` block. The preregistration text is the contract;
> below is where we report what happened.

### Metrics

| Metric | Threshold | Observed | Pass/Fail |
|---|---|---|---|
| E1.1 | median ≥ 5, mean ≤ 100 | — | — |
| E1.2 | ≥ 0.20 | — | — |
| E1.3 | ≥ 3 for 70% | — | — |
| E1.4 | ≥ 0.60 | — | — |
| E1.5 | llm > 0 and both > 0 | — | — |
| E1.6 | ≥ 0.95 | — | — |
| R2.1 | ≥ 6/10 | — | — |

### Claim-level verdict

- Claim 1: [pass / partial pass / fail]. Notes:
- Claim 2: [pass / fail]. Notes:
- Claim 3: baseline recorded for batch-003 comparison.
- Claim 4: not tested (downstream).

### Exploratory findings (record, do not use as evidence for Claims 1–2)

- [per-sub-discipline variation observations]
- [term-type stratification observations]
- [eprint-vs-abstract delta observations]

### Surprises

Things that happened that we did not anticipate. Record these before
explaining them. (Explanations go in an addendum after recording.)

### Committed next actions

Per the decision tree, the next action is: [...]. Tagged in commit
messages as `batch-002 follow-up`.

## Cross-references

- Spec: `holes/missions/M-paper-reverse-morphogenesis.md`
- Execution: `holes/missions/M-superpod-mark2.md` (and its "Post-run
  health and quality checks" section, which covers operational
  diagnostics orthogonal to the claims tested here)
- Pilot packets (once produced): `holes/missions/data/PREREG-batch-002-pilot/`
- Batch-003 preregistration (forthcoming): `holes/missions/PREREG-batch-003.md`
