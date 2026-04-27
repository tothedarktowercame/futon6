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

### Why this mission exists in this shape (pattern-work context)

mark3 is more than a runner cleanup. It is the math-corpus
prong of a broader pattern-application diagnostic: each agent
turn (or paper, in the arxiv prototype) should be coded as an
application of a **typed, well-specified, repeatable pattern of
demonstrable value**. Joe's parent theory mission frames
patterns as `context → tension → move` — a BHK-arrow shape
whose witness (proof, code change, theorem) makes the
application checkable. The math corpus is *prototype 2* of
that frame: the hand-tagged 25-paper pilot
([`futon3/holes/excursions/E-math-prototype-pilot.md`](../../../futon3/holes/excursions/E-math-prototype-pilot.md))
showed the substrate already produces the slot triple
(`situation_S`, `xiang_salience`, `arrow_constraint`) — what's
missing is the named-pattern overlay that R-1 + R-2 add.

The **R-1 / R-2 / R-3 / geometry-artifact** stack in this
mission isn't mark3-internal cleanup. Each piece corresponds
to one of the parent IFR's five properties:

- **R-1 (arxiv prompt) + R-2 (hierarchical pattern set):**
  *well-specified.*
- **Coverage discipline (no nulls):** *coverage.*
- **Slot-distinctness:** *repeatable.*
- **Geometry artifact (T_total, F₂):** *demonstrable value.*
- **Hierarchical taxonomy + family-level fallback:** *replicable
  under automation.*

Rob does not need to engage with the parent mission to do this
work — the deliverables here stand alone. But the framing
explains *why* the substrate edges in this particular
direction rather than alternative refactors. Cross-references:
[`E-math-prototype-pilot.md`](../../../futon3/holes/excursions/E-math-prototype-pilot.md)
(empirical seed), [`E-substrate-metrics.md`](../../../futon3/holes/excursions/E-substrate-metrics.md)
(F₂ score and successor metrics).

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

### MAP questions — answered (2026-04-27)

**Q1. PATTERNS hard-coded; no swap flag.**
`PATTERNS` is a literal list at `scripts/superpod-job.py:340`,
25 `(name, description)` tuples (≈ 2,300 chars). `PATTERN_NAMES`
derives from it at module load. **No flag swaps the source.**
Adding `--stage3-pattern-set <path>` to read an alternative list
is a small change (≈ 30 lines: argparse entry + load + branch in
`build_pattern_prompt`). The mark3 R-1 module
(`src/futon6/arxiv_pattern_prompt.py`) already loads from the
futon3 flexiarg directory via `load_paper_shape_taxonomy()`, so
the swap is implemented for the arxiv path; the only remaining
work is wiring `--source arxiv` to dispatch to it.

**Q2. mark2 has been hot-patched 6× over 7 days.**
`linode-chicago:~/mark2/` contains six `mark2.bak.YYYYMMDDTHHMMSSZ`
files: five on 2026-04-16 (rapid iteration day) and one on
2026-04-23. Each `mark2.bak` differs from the live `mark2`
binary, so they are real patches not no-ops. **Hot-patching
during burnin is Rob's operational discipline.** The
mark3-alongside-mark2 framing matches: a separate `mark3`
binary with its own `mark3.bak` cadence keeps the mark2 backup
chain undisturbed.

**Q3. eprint flag exists; jsonl shape carries metadata only.**
- Flag: `--discover-terms-eprint-dir <path>` (parser at
  `superpod-job.py:4405`), with sibling `--distinctor-eprint-dir`
  and a `--paper-eprint-dir` default-fallback chain at
  lines 4720-4729. The flag exists, is off by default, and has
  been used by mfuton invocations.
- Input-jsonl shape (per `src/futon6/stackexchange.py:688`,
  `load_arxiv_pairs()`): `{id, title, abstract, categories, date}`.
  **No `eprint_path` field.** Eprints come from a separate
  filesystem directory; `_load_eprint_text_for_entity()` at
  `superpod-job.py:1215` matches files by `glob(f"{paper_id_short}*")`.
  Therefore R-3 (eprint default for arxiv batches; #46) needs
  a filesystem convention for where eprints live (e.g.
  `~/mark2/eprints/` or wherever Rob stores them) plus a
  default-on-when-source-is-arxiv branch.

**Q4. Arxiv prompt is ≈ 7,600 chars; mark2 SE prompt is ≈ 4,500 chars.**
- mark3 arxiv prompt (built via R-1 module) with all 5 family
  parents, 18 leaves, abstract clipped at 1,200 chars, plus up
  to 3 theorem and 3 proof excerpts at 600 chars each: **7,571
  chars** measured directly. ≈ 1,900 tokens.
- mark2 SE prompt (`build_pattern_prompt`): 25-pattern list
  (2,310 chars) + Q[:700] + A[:900] + template ≈ **4,500 chars**.
- 67 % size increase. Comfortable within typical LLM windows
  (Mistral-7B-instruct: 8,192 tokens ≈ 32,000 chars; modern
  models: 128 K+ tokens). **No paragraph-level chunking needed
  for v0.** If a future extension wants to feed full proof
  bodies rather than excerpts, chunking will become necessary;
  parked.

**Q5. `hypergraph-thread-ids.json` is GNN-side; not the right hook.**
Emitted at `superpod-job.py:3522` after the GNN training pass,
as the index for `hypergraph-embeddings.npy`. It carries thread
ids, not per-paper geometric quantities. The right hook for
the mark3 geometry artifact is **a new per-paper sweep** over
`paper-hypergraphs.json` after Stage 5d completes — exactly
what `compute-paper-T.py` already does. Cost: ≈ 1 ms per paper;
runs in ≈ 4 s on a 5 K-paper batch. Output sibling artifact
`output/geometry.json` (or `paper-T.tsv`). See Track-B
checkpoint below.

**Q6. Test harness has 5 smoke tests; mark3 VERIFY tests slot
in cleanly.**
`tests/test_superpod_job_smoke.py` (221 lines, 5 tests):
- `test_superpod_job_ct_pipeline_smoke` — canonical end-to-end.
- `test_superpod_job_limit_defaults_thread_limit` — argument plumbing.
- `test_arxiv_paper_eprint_dir_feeds_all_paper_stages` — eprint flag flow.
- `test_arxiv_paper_hg_eprint_dir_is_legacy_alias` — flag aliasing.
- `test_arxiv_paper_eprint_dir_fails_when_no_sources_match` — failure path.

mark3 V-5..V-9 land as siblings:
- V-6 (round-trip on hierarchical pattern-tag schema) → new test
  using R-1's `parse_arxiv_pattern_response()` against a fixture.
- V-7 (coverage smoke) → new test that runs Stage 3 with
  pre-known LLM-failure prompts and asserts every paper gets a
  record (success or `:status :failed`).
- V-8 (geometry artifact correctness) → new test invoking
  `compute_paper_T.compute_one_paper()` on hand-checked
  fixtures.
- V-9 (mark2/mark3 coordinator non-interference) → integration
  test that spawns a mark3 batch alongside a mark2 batch on
  separate inbox/outbox dirs.

Existing harness pattern (smoke-test-with-tmp-path-fixtures)
fits all four naturally; **no test-infrastructure changes
needed**. R-1 already shipped its own test file
(`tests/test_arxiv_pattern_prompt.py`, 15 tests, all passing)
which V-6 builds on.

**MAP exit:** all six questions have concrete answers. Ready
for DERIVE.

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

### Checkpoint Track-B deliverable — 2026-04-27

**What was done:** Track B (geometry-on-existing-data demo) materialised
as `scripts/compute-paper-T.py` running on three batches already in
`~/code/storage/mark2/outbox/`. ~4 seconds per 5,000 papers; pure-Python.
Reads `output/paper-hypergraphs.json`, emits TSV with per-paper
`n_claims, n_unpaired, T_total, top_support_id, top_support_count` plus
a Δ-Laplacian proxy (`top_support_id` is the non-claim vertex co-
occurring most often with unpaired claims — load-bearing-concept
candidate).

**Findings landed in
`futon3/holes/excursions/E-Ttotal.md` v0:**

- F-1: T_total is a stable corpus property. mfuton-001 and mfuton-002
  (independent 5k-paper samples) produce near-identical distributions
  (mean 0.364 / 0.368, median 0.333 / 0.333). The geometry is real, not
  a sampling artifact.
- F-2: The eprint-off cost is now empirically measurable. Older
  `results-005` lineage has 83 % empty-claim papers (vs 28-32 % for
  mfuton); on the rest the T-mean shifts 0.36 → 0.53. **R-3 (eprint
  default) is backed by hard numbers, not architectural argument.**
- F-3: Distribution shape has meaningful resolution (p10=0.0, p90=0.78
  on mfuton). Plenty of dynamic range for downstream consumers.
- Cross-tab against the 25 hand-tagged papers shows directional
  family signal (characterization median T=0.19, structural-relation
  T=0.33), n=24 too small for stat-sig but rank ordering matches
  the qualitative reading.
- **Two silent-fail modes are now distinguishable:** triple-level
  (null analysis, 4 %) vs hypergraph-level (n_claims=0, 32 %).
  The mark3 coverage record needs both reasons named in DERIVE.
- Δ-Laplacian proxy spot-check on Zarankiewicz paper picks
  `technique:axis-parallel-box` as the load-bearing concept —
  matches the abstract's actual punchline. **Geometric framing
  operationally correct on real data with a one-line T definition.**

**Test state:** pure-Python compute step; 3 batches × 5k papers
processed without error. No automated tests for the analysis
script itself in v0.

**Integration path:** the script lifts verbatim into mark3's
pipeline as a per-batch geometry stage. The mission's §3 DERIVE
sketch for `output/geometry.json` should incorporate the
distinguishable silent-fail-reason discipline:

```clojure
{:status :failed
 :reason :triple-extraction      ; or :no-theorem-blocks, etc.
 :stage :stage6                  ; or :stage5d
 :substrate-mode :abstract-only} ; or :eprint, :degraded
```

**Demo for Rob:** the cross-batch comparison (mfuton vs
results-005) makes R-3 (eprint default) empirically motivated;
the cross-tab shows the substrate is doing useful work *now*
without further runner patches; the Zarankiewicz-load-bearing-
concept spot-check shows the geometric framing produces
mathematically sensible outputs without any tuning.

**Next:** mark3 MAP phase. Q1–Q6 in §2 still need answering
(code reads of `superpod-job.py`, `mark2`, the test harness).
Or, if Rob is ready: bundle R-1 + R-2 + the geometry script
into a draft mark3 branch for review.

### Checkpoint MAP-exit — 2026-04-27

**What was done:** answered all six MAP questions with concrete
code-read findings (line numbers, file paths, measured
prompt-budget numbers, test-harness inventory). Highlights:

- PATTERNS is a hard-coded literal at `superpod-job.py:340`; a
  `--stage3-pattern-set` flag is ≈ 30 lines to add. R-1's
  `load_paper_shape_taxonomy()` already implements the loader.
- mark2 has been hot-patched 6× in 7 days (`mark2.bak.*`
  files); mark3-alongside framing matches existing discipline.
- eprint flag exists (`--discover-terms-eprint-dir`); arxiv
  jsonl has no `eprint_path` field, so R-3 (#46) needs a
  filesystem convention plus a default-on branch.
- Arxiv prompt size measured: 7,571 chars (≈ 1,900 tokens) —
  comfortably within typical LLM windows; no chunking needed.
- `hypergraph-thread-ids.json` is GNN-side and not the right
  geometry hook; the right hook is a new sweep over
  `paper-hypergraphs.json` (already implemented in
  `compute-paper-T.py`).
- Test harness has 5 smoke tests at `test_superpod_job_smoke.py`;
  V-5..V-9 slot in as siblings without test-infrastructure
  changes.

Light cross-reference section ("Why this mission exists in this
shape") added at the top of the mission to point Rob at the
parent theory context (`M-pattern-application-diagnostic`,
`E-math-prototype-pilot`, `E-substrate-metrics`) without
requiring he engage with it for the substrate work. Mark3
deliverables stand alone; the framing just explains why the
shape is what it is.

**Test state:** N/A (mission doc only).

**Next:** DERIVE. The MAP findings constrain DERIVE concretely:
typed pattern-tag schema (R-1 done), coverage-record schema
with two distinguishable failure modes (per E-Ttotal §"Two
silent-fail modes"), geometry artifact spec (matches
`compute-paper-T.py` output shape), and the source-detection
branch (Q3 conclusion). DERIVE writeup is ≈ 1-2 hours of
focused work.
