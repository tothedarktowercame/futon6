# Mission: Superpod Mark 3 — Pattern-Tagged + Geometrically Annotated arXiv Corpus

**Date:** 2026-04-27
**Status:** IDENTIFY
**Owner:** Rob (superpod runs), Joe (pipeline code + evaluation)
**Repos:** futon6 (pipeline), futon3c (downstream consumers),
futon3 (pattern application diagnostic — parent theory mission)
**Predecessor:** [M-superpod-mark2.md](M-superpod-mark2.md) — mark3 inherits
the mark2 substrate (10-stage pipeline, mark2 coordinator,
batch state machine, eprint plumbing as of `74cc161`) and
extends rather than replaces.
**Parent theory:**
[`futon3/holes/missions/M-pattern-application-diagnostic.md`](../../../futon3/holes/missions/M-pattern-application-diagnostic.md)
— mark3 is the substrate-side delivery for prototype 2 of that
mission (math-corpus-as-hypergraph; pattern application
diagnostic with witnesses).

## 1. IDENTIFY — naming the gap

### Motivation

Mark 2 produces 40,000+ processed papers with typed hypergraphs,
reverse-morphogenesis triples, embeddings, and per-stage chunk
files. Two structural defects make that output insufficient as a
substrate for **prototype 2 of M-pattern-application-diagnostic**:

1. **`pattern-tags.json` is empty for ≈99 % of papers.** Stage 3
   uses the math.SE-derived Q&A prompt with the futon6/P0
   25-pattern Q&A list. arxiv papers aren't Q&As; the LLM
   correctly returns `[]`. Confirmed by inspecting
   `output/stage3-pattern-tags-chunks/chunk-000.json` across
   `results-001..006` and `results-mfuton-001/002`. **The
   tagging stage runs cleanly but on the wrong premise.**
2. **eprint mode wasn't the default for the regular lineage.**
   The plumbing exists (`74cc161 "Harden mark2 eprint plumbing"`)
   and `mfuton-001/002` invocations exercise it, producing
   per-paper hypergraphs ≈ 20× denser than the regular lineage
   (n_nodes ≈ 222 vs. ≈ 9). Same parser; different invocation
   flag. **The richer substrate already exists in code; it just
   isn't reached by every batch.**

Plus three derived gaps named by the parent mission's IFR
(coverage, well-specified, repeatable, demonstrable value,
replicable under automation):

3. **Coverage gap on triples.** ≈4–12 % of papers in the sample
   have a silently-null `analysis` field in
   `reverse-morphogenesis.json`. The IFR's coverage property
   requires *every* turn to produce a record; a `null` reading
   mid-batch is indistinguishable from "this turn never
   happened."
4. **Slot-collapse gap.** ≈12 % of triples paraphrase `situation_S
   / xiang_salience / arrow_constraint` as one sentence, with
   `quality: {form: good, salience: good, arrow: good}` masking
   the collapse. The IFR's "well-specified" property means the
   slots must be *demonstrably distinct* before the quality
   check passes.
5. **No geometric-quantity artifacts.** The hypergraph is
   present, but T (tension scalar), ∇T, Δ are not computed at
   batch time. The discrete-DG framing committed by the parent
   mission requires these as first-class artifacts so downstream
   consumers don't recompute them per query.

### Theoretical anchoring

- **The parent mission's IFR**, with its five-property
  decomposition and the geometric commitment (tension as scalar
  field on hypergraph, gradient + Laplacian computable). See
  `M-pattern-application-diagnostic.md` §IFR, §"Geometric
  commitment", §"System invariant".
- **`library/futon-theory/task-as-arrow.flexiarg`** — patterns
  as BHK arrows, witnesses as proof objects. The math
  substrate's `derivation` edge `(claim → proof_of_target)` is
  literally a BHK arrow at edge level (validated in
  `futon3/holes/excursions/E-math-prototype-pilot.md`).
- **`library/futon-theory/structural-tension-as-observation.flexiarg`**
  — tension as the observable signal of mission progress.
- **`futon4/holes/mission-lifecycle.md`** — this mission follows
  the 7-phase derivation path; VERIFY explicitly checks for no
  regression on mark2's completion criteria.

### Scope in / out

**In scope:**

- Stage 3 prompt fork: arxiv-aware prompt builder taking
  `paper_id, title, abstract, theorem/proof excerpts`.
- Hierarchical arxiv pattern-set authoring: ≈20 entries
  (5 family parents + 15 leaves) seeded from
  `futon3/holes/excursions/E-math-prototype-pilot.md` Phase-2
  hand-tagging.
- Coverage discipline: every turn produces a record. Replace
  silent-null with explicit `{:status :failed, :reason ...}`
  shape.
- Slot-distinctness enforcement: the prompt requires the three
  slots to be substantively different; quality check rejects
  paraphrase-collapse with a flag rather than falsely passing.
- Geometric-quantity artifact: per-paper
  `output/geometry.json` carrying `{:T-total, :unpaired-claims,
  :hypergraph-laplacian-summary}` for batch-level queries.
- VERIFY checks ensuring no regression on mark2's completion
  criteria (see §VERIFY).
- INSTANTIATE: ship as `superpod-mark3` runner alongside mark2;
  do NOT cut over the existing mark2 outbox/manifest. mark3
  starts a fresh manifest cycle (or migrates with explicit
  metadata).

**Out of scope:**

- Replacing the mark2 coordinator state machine
  (`scripts/mark2`). It works.
- Persistent homology / Forman-Ricci curvature on the corpus
  (phase-2 geometric work, lives downstream).
- Cross-paper pattern-similarity propagation via
  `structural-similarity-index`.
- Reconciling Mark 1 / 2 / 3 outputs into a single canonical
  embedding. Successive runs on the same papers produce
  successive layers; canonicalisation is its own mission.
- Authoring patterns for non-math corpora (legal, scientific
  literature) — the schema must permit them, but this mission
  delivers only the arxiv-math instance.

### Completion criteria (how we'll know it worked)

1. On a fresh batch run with mark3:
   - `output/pattern-tags.json` has non-empty `:patterns` for
     ≥ 70 % of successful papers (vs. ≈ 1 % in mark2).
   - `output/reverse-morphogenesis.json` has zero null
     `analysis` fields; failures recorded explicitly with
     reason.
   - `output/geometry.json` exists per batch and per paper,
     with `T-total ∈ [0,1]` distribution non-degenerate
     (std > 0.1 across the batch).
2. Spot-check on 25 papers from mark3 output: hand-tagging
   agreement with pipeline-tagging ≥ 60 % (using the
   hierarchical pattern set; family-level agreement counted as
   half-credit).
3. Mark2 regression tests still pass (see §VERIFY).
4. `mark2 status` and `mark3 status` (separate manifests) show
   no interference; rolling back to mark2-only is a one-command
   operation if mark3 misbehaves.

### Relationship to other missions

| Mission | Relationship |
|---------|-------------|
| `M-superpod-mark2` | Predecessor; mark3 inherits substrate, mark3 doesn't break mark2's outputs |
| `M-pattern-application-diagnostic` (futon3) | Parent theory; mark3 is the prototype-2 substrate delivery |
| `M-paper-reverse-morphogenesis` | Stage 6 is here; mark3 fixes the slot-collapse + null-coverage issues in its output |
| `M-apm-solutions` (futon3c) | Downstream consumer of FAISS retrieval |
| `M-trip-journal` (futon5a) | Future consumer of typed pattern records |
| `E-math-prototype-pilot` (futon3) | The hand-tagging pilot that produced this mission's pattern-set seed |

### Source material

- `scripts/superpod-job.py` — the mark2 pipeline runner
  (10 stages, 6,273 lines as of 2026-04-27).
- `scripts/mark2` — coordinator state machine.
- `src/futon6/paper_hypergraph.py` — theorem-block parser
  (already supports the rich-hypergraph case mfuton uses).
- `src/futon6/theorem_extraction.py` — LaTeX theorem extraction
  utilities.
- `~/code/storage/mark2/outbox/results-mfuton-001..002/output/` —
  reference for the rich-hypergraph + reverse-morphogenesis
  output shape.
- `futon3/holes/excursions/E-math-prototype-pilot.md` — the
  25-paper hand-tagging that produced the candidate pattern
  set.
- GitHub issue [futon6 #45](https://github.com/tothedarktowercame/futon6/issues/45)
  — umbrella covering R-1..R-5 plus the rename to mark3 (this
  mission's INSTANTIATE deliverable).

### Owner / dependencies

- **Joe:** mission ownership, prompt + pattern-set authoring,
  geometric-quantity artifact spec, mark3 runner integration.
- **Rob:** superpod execution, stage-by-stage validation runs,
  manifest migration tooling, performance evaluation.
- **Dependencies:** mark2 substrate as-is; no upstream
  dependency on missions that haven't reached at least their
  IDENTIFY phase.

**IDENTIFY exit criterion (mission-lifecycle):** A human has
read the proposal and agrees the gap is real and the scope is
right. — *Pending review by Joe; sub-issues to be filed against
GitHub for Rob's review of the runner-side items.*

## 2. MAP — what's ready vs. what's missing

(See `mission-lifecycle.md` §MAP. Each Q below to be answered
with concrete findings, not speculation, before MAP exit.)

### Ready vs missing

| Item | Where | Status |
|------|-------|--------|
| 10-stage pipeline | `scripts/superpod-job.py` | **Ready** (mark2) |
| Mark2 coordinator | `scripts/mark2` | **Ready** |
| Theorem-block parser | `src/futon6/paper_hypergraph.py` | **Ready** (mfuton uses it) |
| eprint plumbing | `scripts/superpod-job.py` `_load_eprint_text_for_entity` (74cc161) | **Ready (in code, off by default)** |
| Reverse-morphogenesis Stage 6 | `scripts/superpod-job.py` Stage 6 | **Ready (with gaps: null-coverage, slot-collapse)** |
| Pattern-tag Stage 3 | `scripts/superpod-job.py` Stage 3 | **Ready (wrong prompt for arxiv)** |
| Hypergraph-tensor / GNN | `src/futon6/graph_embed.py` | **Ready (mark2 fixed training signal)** |
| Q&A pattern set (math.SE) | `PATTERNS` constant in `superpod-job.py` | **Ready (wrong-domain for arxiv)** |
| arxiv-proof-patterns set | — | **Missing** |
| Arxiv-aware Stage 3 prompt | — | **Missing** |
| Coverage discipline | — | **Missing** (silent-null is current behaviour) |
| Slot-distinctness enforcement | — | **Missing** |
| Geometry artifact (T, Δ) | — | **Missing** |
| eprint-mode-as-default for arxiv | — | **Missing (invocation default)** |
| mark3 manifest / coordinator fork | — | **Missing** |
| Pattern-tag round-trip test | — | **Missing** |

### MAP questions

- **Q1.** Where in `superpod-job.py` is `PATTERNS` declared, and
  is it loaded from a file or hard-coded? Can the source be
  swapped via a flag without code change to swap pattern set?
- **Q2.** What does the current `mark2.bak` history at
  `linode-chicago:~/mark2/` tell us about how often the runner
  has been hot-patched? Does mark3's "alongside mark2" framing
  match Rob's existing operational discipline?
- **Q3.** Is `eprint_mode` already advertised on a flag, and is
  the input-jsonl shape compatible with passing eprint-dir
  (i.e. does the regular batch jsonl carry the eprint path,
  or only the abstract)?
- **Q4.** What's the LLM prompt-budget envelope per paper for
  Stage 3? If the arxiv prompt with theorem/proof excerpts
  exceeds it, paragraph-level chunking is needed.
- **Q5.** Where does the `hypergraph-thread-ids.json` artifact
  come from, and does it already imply a per-paper
  geometric-quantity hook we can extend?
- **Q6.** What's the existing test harness for `superpod-job.py`?
  `tests/test_superpod_job_smoke.py` exists (74cc161); what
  does it cover, and where do mark3's new VERIFY tests slot in?

### Notes — what's already in mfuton-equivalent code

The following items appeared in mark1 → mark2 transitions and
are operational in mfuton's invocation:

- Eprint full-text loading (74cc161)
- Theorem-block parsing in `paper_hypergraph.py`
- Stage 6 reverse-morphogenesis chunked output
- `techniques.json` (mfuton has it; check if regular batches
  produce it under the right invocation)

Items posted as **separate GitHub issues for Rob** so they
can be handled discretely without bundling into this
mission's substantive work:

- (filed under #45 R-3) Default `--discover-terms-eprint-dir`
  to a sensible value when input is arxiv. Code exists; flip
  the default. Quick.

Items that **belong to this mission** (no equivalent in master
or mfuton):

- (R-1) arxiv-aware Stage 3 prompt fork
- (R-2) hierarchical `arxiv-proof-patterns.edn` curation
- Coverage discipline (no nulls)
- Slot-distinctness enforcement
- Geometric-quantity artifact
- (R-5) rename to `superpod-mark3`

## 3. DERIVE — TBD

To be expanded after MAP exit. Sketches of design directions
already locked-in by IDENTIFY scope:

- Pattern-set schema mirroring the M-pattern-application-
  diagnostic typed slots (`{:context, :tension, :move,
  :witness-shape, :domain}`) with `:domain :math-corpus` and
  exemplar paper-id references for each leaf pattern.
- Coverage record shape: for failed turns, emit
  `{:entry_id, :status :failed, :reason <enum>, :error
  <string>, :stage <kw>}` so downstream consumers get a
  uniform signal.
- Geometry artifact shape: `output/geometry.json` containing
  per-paper `{:paper-id, :T-total, :unpaired-claims,
  :Laplacian-summary {:max :argmax-vertex-id :min :argmin-
  vertex-id}, :computed-at}`. Cheap; ~1 ms per paper at
  hypergraph sizes we have.

## 4. ARGUE — TBD

To be filled at ARGUE phase. Pattern cross-reference must
search `futon3/library/` for relevant patterns; expect to find
matches in `futon-theory/` (BHK arrow framing),
`coordination/` (pipeline / batch discipline), and
`storage/` (durable-write / round-trip patterns).

## 5. VERIFY — anti-regression checks vs. Mark 2

This phase is held first-class because mission-lifecycle.md
calls it out as the structural-validation phase, and mark3's
"build alongside, don't break" framing makes anti-regression
checks the central VERIFY content.

### Mark 2 completion criteria that must continue to hold

(From M-superpod-mark2.md §"How we'll know it worked")

- [V-1] Pipeline runs on a corpus larger than math.CT
  (≥ 200 K papers feasible without code change).
- [V-2] Stage 9b embeddings don't collapse: pairwise cosine
  std > 0.10, validation accuracy < 90 %.
- [V-3] Hybrid retrieval (BGE + GNN) surfaces technique-relevant
  papers BGE-only misses, on the learn-to-swim canaries / pilot.
- [V-4] FAISS index serves `corpus_ws_bridge.py` consumers
  with measurably better precision than mark1.

These remain operationally critical. mark3 must not regress on
any of them.

### Mark 3 specific VERIFY items

- [V-5] mark2 batches in `~/mark2/outbox/` remain readable and
  queryable after mark3 lands. mark3 does NOT mutate mark2
  artifacts.
- [V-6] Round-trip on the new pattern-tag schema: write +
  read-back of a `pattern-tags.json` entry preserves all
  hierarchical-pattern fields.
- [V-7] Coverage smoke test: run mark3 on a 10-paper test
  jsonl that includes pre-known LLM-failure prompts; confirm
  every paper gets a record (success or `:failed` shape).
- [V-8] Geometry artifact correctness on hand-checked papers:
  T_total agrees with hand-counting unpaired claims to within
  ±1 paper-of-the-25-paper-pilot ground-truth.
- [V-9] No interference between mark2 and mark3 coordinator
  state: spawning a mark3 batch while mark2 batches are
  in flight produces correct routing (separate inbox/outbox,
  separate manifest).

### Tripwire fidelity contract (per mission-lifecycle §DERIVE)

mark3 is a **port/extend** mission relative to mark2, not pure
greenfield. A fidelity contract is required:

- **Preserve:** mark2's 10-stage pipeline semantics, the
  mark2 coordinator's batch state machine, the `output/`
  artifact filename conventions (`paper-hypergraphs.json`,
  `embeddings.npy`, etc.), the input-jsonl schema.
- **Adapt:** Stage 3 (prompt + pattern set forks per input
  domain), Stage 6 (slot-distinctness enforcement,
  null-coverage replacement). Compatibility assertion: a
  consumer reading `pattern-tags.json` from mark2 and from
  mark3 should be able to detect which schema it has via a
  `:schema-version` field; both schemas remain readable.
- **Drop:** None planned for mark3.

## 6. INSTANTIATE — TBD

To be expanded at INSTANTIATE phase. Provisional plan:

1. Branch `mark3-runner` off master (or mfuton-equivalent
   integration branch, depending on Q2).
2. Implement R-1, R-2, coverage, slot-distinctness,
   geometry-artifact in source.
3. Sub-issues filed for each so individual review remains
   tractable; PRs land discretely.
4. Smoke test `tests/test_superpod_job_smoke.py` extension
   per V-6/V-7/V-8.
5. Rename runner to `superpod-mark3.py` (or co-locate with a
   `--mode mark3` flag — decision at INSTANTIATE).
6. Coordinator fork: `mark3` script alongside `mark2`,
   independent inbox/outbox, separate manifest.
7. Operational rollout: Rob runs a 10-paper mark3 burnin
   batch; review; then a 1-batch (5 K paper) trial; review;
   then re-run regular results-001..006 with mark3 if all
   passes.

## 7. DOCUMENT — TBD

Held for INSTANTIATE-exit. Will produce:
- Docbook entries explaining mark3's prompt fork, the
  hierarchical pattern set, and the geometry artifact.
- Cross-references from M-pattern-application-diagnostic
  prototype-2 subsection to mark3's INSTANTIATE checkpoint.
- Updated `M-superpod-mark2.md` "Open items — future mark2"
  section noting mark3 supersedes those items.

## Checkpoints

### Checkpoint IDENTIFY-entry — 2026-04-27

**What was done:** Mission spawned, IDENTIFY phase populated
from M-pattern-application-diagnostic IFR + the 25-paper
hand-tagging excursion + the GH issue umbrella (#45). MAP
inventory drafted; questions Q1–Q6 listed for the MAP phase.
VERIFY phase pre-populated with mark2 anti-regression checks
(V-1..V-4) and mark3-specific items (V-5..V-9). Fidelity
contract (preserve / adapt / drop) named.

**Test state:** N/A (mission doc only).

**Next:** MAP phase. Answer Q1–Q6, then DERIVE.

### Checkpoint R-2 deliverable — 2026-04-27

**What was done:** R-2 (hierarchical arxiv-proof pattern set) authored
in `futon3/library/`, completing the Joe-side authoring track without
runner integration. Eleven files:

- 5 family parents in `futon3/library/math-strategy/`:
  `existence-result.flexiarg`,
  `characterization-result.flexiarg`,
  `structural-relation-result.flexiarg`,
  `property-of-object-result.flexiarg`,
  `clarification-meta.flexiarg` (meta-tag, not a pattern).
- 5 new leaves in `futon3/library/math-informal/`:
  `failure-mode-characterization.flexiarg`,
  `structural-characterization.flexiarg`,
  `structural-inclusion.flexiarg`,
  `complexity-classification.flexiarg`,
  `structural-equivalence.flexiarg`.
- 1 operator-readable index:
  `futon3/library/math-strategy/PAPER-SHAPES-INDEX.md` ties the
  hierarchy together with cross-references to existing
  math-informal leaves and exemplar arxiv paper-ids per family.

The taxonomy cross-references existing math-informal patterns
(e.g. `transport-across-isomorphism`, `find-the-right-abstraction`,
`split-into-cases`, `exhaustion-as-theorem`) rather than
duplicating them. The new leaves only fill genuine gaps the
existing library didn't cover.

**Per-flexiarg `@family` field**: each new leaf declares its
parent family (e.g. `@family math-strategy/characterization-result`)
so future tooling can derive the hierarchy by grep.

**Test state:** Schema only; no runner-side validation.

**Next:** R-1 (arxiv-aware Stage 3 prompt) consumes this
taxonomy as its choice space (see PAPER-SHAPES-INDEX.md
§"Use as a prompt choice space"). Authoring R-1 is the next
local-only deliverable; runner integration is downstream of
Rob's deploy.

### Checkpoint R-1 deliverable — 2026-04-27

**What was done:** R-1 (arxiv-aware Stage 3 prompt builder) authored
as a local module in `src/futon6/arxiv_pattern_prompt.py` with a
test suite in `tests/test_arxiv_pattern_prompt.py`. Three exports:

- `load_paper_shape_taxonomy(library_root)` — parses the 5 family
  parents + member leaves from `futon3/library/math-strategy/` and
  `math-informal/`. Reads `@flexiarg`, `@title`, `@family`,
  `member[…]`, and `! conclusion:` body for each pattern.
- `build_arxiv_pattern_prompt(paper_id, title, abstract,
  theorem_excerpts=None, proof_excerpts=None, ...)` — emits the
  prompt with the hierarchical choice space. Includes coverage and
  slot-distinctness rules in the prompt itself so the LLM knows
  when to emit `clarification-meta` rather than defaulting to
  `uncertain`.
- `parse_arxiv_pattern_response(raw)` — validates the LLM's JSON
  response against the loaded taxonomy. Rejects invalid families
  / leaves; enforces that `clarification-meta` carries a
  `:collapsed {:reason :explanation}` block. Returns
  `{:ok :family :leaf :family_confidence :leaf_confidence
  :rationale :collapsed :error}`.

**Test state:** 15 tests, 15 passed, 0 failures.
Covers: taxonomy load (5 families, leaves linked to families),
prompt builder (all families and excerpts present, abstract
clipping), response parser (well-formed / uncertain-leaf /
clarification-meta with-and-without-collapsed / invalid-family
/ invalid-leaf / no-json).

**Demo verified:** module renders a coherent prompt for
`arxiv-2604.20815v1` (Zarankiewicz dichotomy) with all 5 families
and 18 leaves in the expected hierarchical layout.

**Integration path:** Rob's mark3 deploy imports both functions.
The runner calls `build_arxiv_pattern_prompt(...)` at Stage 3
when input source is arxiv (see #46 for source-detection logic).
Output of `parse_arxiv_pattern_response(...)` lands in
`pattern-tags.json` with the new schema; coverage discipline
record fires when `parse_*` returns `:ok false`.

**Next:** Track B (geometry-on-existing-data demo for Rob).
Compute T_total + Laplacian summaries on the 40k papers already
in `~/code/storage/mark2/outbox/`. Materialises `E-Ttotal.md`
from stub to real findings; produces a compact reusable script
that mark3 can adopt verbatim for the geometry-artifact stage.
