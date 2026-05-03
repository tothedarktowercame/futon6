# Mission: Hyperreal Dictionary Planning

**Date:** 2026-04-29
**Status:** IDENTIFY
**Owner:** Joe
**Repo:** futon6
**Sequel architecture:** `futon3/holes/missions/M-live-geometric-stack.md`
**Immediate substrate dependencies:**
- `futon6/holes/missions/M-superpod-mark2.md`
- `futon6/holes/missions/M-superpod-mark3.md`
- `futon6/holes/excursions/E-mfuton-silver.md`
**Downstream consumer references:**
- `futon3/holes/missions/M-pattern-application-diagnostic.md`
- `futon3c/holes/missions/M-apm-solutions.md`
- `futon0/holes/missions/M-futonzero-mvp.md`
- `futon0/holes/missions/M-futonzero-prelim-practice.md`

## 1. Why this mission exists

The superpod pipeline is no longer just a speculative extractor. It is
becoming a real mathematical substrate:

- arXiv batches can be processed into per-paper theorem/proof hypergraphs
- mark3 adds arXiv-aware pattern tagging, explicit Stage 6 coverage
  semantics, and geometry artifacts
- the legacy-TeX normalization work substantially improves structural
  recall on older arXiv source

This changes the planning problem.

Earlier wishlist documents in `resources/` framed "extract structured
graphs from mathematical text" as the main unknown. That is no longer the
main unknown. The next question is:

> How do we turn batch-local superpod outputs into a live mathematical
> hypergraph substrate that can support navigation, explanation,
> formalisation packets, tutoring, and eventually self-play?

That question belongs in `futon6`, because the paper/mining pipeline and
the Hyperreal Dictionary code both live here.

## 2. Architectural inheritance

This mission is a sequel to `M-live-geometric-stack`, but not a duplicate.

`M-live-geometric-stack` closed the architecture question for the **code
stack as a living typed hypergraph**: one live store, explicit edge
taxonomy, computable geometric quantities, liveness as an invariant, and
multiple downstream consumers.

This mission asks whether the same architectural shape now makes sense for
the **mathematical corpus**:

- papers, claims, proofs, concepts, patterns, citations, authors, and
  versions as vertices
- theorem/proof, mention, citation, pattern-application, concept-alignment,
  formalization-target, and pedagogical edges as typed relations
- batch outputs as ingestion feeds into one living corpus store
- Hyperreal Dictionary as the concept/navigation layer over that store

The mission does **not** assume the full live geometric stack should be
ported mechanically. It is a planning-and-prototype mission whose job is
to separate:

- what is buildable now
- what is one layer away
- what still sounds plausible but remains science-fiction-adjacent

## 3. What the existing material suggests

The older resource documents (`AI4CI (Mathsistant).txt`, `ati.html`,
`sfi.html`) repeatedly ask for the same higher-level outcomes:

- a structured knowledge base from mathematical text
- modelling of process as well as content
- navigation / recommender / question-answering support
- pedagogical usefulness, not only retrieval
- a route toward formalisation
- eventually self-questioning / self-improving agent loops

The `futon0` missions sharpen the consumer side:

- `M-futonzero-mvp` wants visible tutor interventions, deterministic local
  artifacts, and policy comparison rather than black-box coaching
- `M-futonzero-prelim-practice` wants teaching tactics, capability
  trajectories, pre-computed navigation layers, and a feedback loop from
  learner behaviour back into the teaching substrate

Taken together, the picture is:

- `futon6` should own the mathematical corpus substrate and concept layer
- `futon0` / `futon3c` should consume that substrate for tutor-like and
  practice-like loops
- formalisation should begin as packetization / triage, not as a fantasy of
  uniform theorem closure

## 4. Mission objective

Produce a reality-based architectural plan for a Hyperreal-Dictionary-like
mathematical substrate built from superpod outputs, together with a small
set of prototypes that test the parts that appear buildable now.

This mission should answer:

1. What can be built immediately on top of the current superpod outputs?
2. What requires one additional architectural layer, but is now plausibly
   within reach?
3. What still belongs to longer-horizon research, and what would it cost to
   bring it into engineering range?

## 5. In scope

### In scope

- Define the architecture for a corpus-level live mathematical hypergraph
  store fed by superpod outputs
- Specify the relationship between superpod paper outputs and
  `scripts/hyperreal.py`
- Build small prototypes for the buildable-now layer
- Reality-check the tutoring / self-play / formalisation ambitions against
  the actual current substrate
- Produce rough effort estimates and sequencing for each lane

### Out of scope

- Full deployment of a global live arXiv hypergraph database
- Full tutoring system implementation
- Full self-play system implementation
- Full automatic formalisation at corpus scale
- Replacing mark2/mark3 mission ownership with this mission

## 6. Proposed outcomes

### Outcome A. Corpus architecture note

A concrete architecture note that answers:

- canonical vertex and edge classes for the corpus-level store
- how batch outputs map into that schema
- which artifacts remain per-batch and which are promoted to global state
- where geometric quantities live and how they are recomputed
- where provenance and versioning attach

### Outcome B. Buildable-now prototypes

At least two small prototypes from the following class:

- **paper → concept bridge**:
  map paper terms / claims / patterns into Hyperreal Dictionary objects
- **batch-level concept neighborhood report**:
  show which concepts, claims, and pattern families dominate a batch
- **formalization packet**:
  for a selected claim, emit local theorem/proof/dependency/pattern context
- **teaching-unit packet**:
  for a selected concept or theorem family, emit explanation / example /
  common pitfall scaffolding using existing extracted structure

These should be treated as probes, not product claims.

### Outcome C. Reality-check memo

A planning memo for the currently plausible-but-not-yet-built lanes:

- tutoring / learning-pathway systems
- self-play over mathematical content
- stronger formal/informal bridge work

For each lane, record:

- current substrate prerequisites already satisfied
- missing prerequisites
- likely first real deliverable
- main risk of self-deception
- rough effort band

### Outcome D. Sequenced roadmap

A staged roadmap that separates:

- **Stage 1:** immediate probes
- **Stage 2:** architecture-enabling work
- **Stage 3:** first true downstream consumers
- **Stage 4:** longer-horizon agentic loops

## 7. Rough feasibility bands

These bands are intentionally coarse and should be corrected by the
mission's own MAP/DERIVE work.

| Lane | Current status guess | Effort band |
|---|---|---|
| Paper → Hyperreal concept bridge | buildable now | days |
| Batch concept / pattern report | buildable now | days |
| Formalization packet exporter | one layer away | 1-2 weeks |
| Teaching-unit packet exporter | one layer away | 1-2 weeks |
| Global corpus hypergraph ingest v0 | architecturally clear, not built | 2-6 weeks |
| Tutor navigation / pathway policy consumer | plausible once packets exist | 2-6 weeks |
| Self-play over corpus content | still planning/research-heavy | 1-3 months+ |
| Broad automated formalisation | still selective / triage-first | months, not weeks |

## 8. Deliverable standard

This mission is invalid if it blurs planning and capability.

Specifically:

- A prototype must be explicitly labeled as a probe.
- A reality-check memo must say when something is still not engineered.
- Any tutoring/self-play claim must identify the exact substrate it uses.
- No mission text should imply that a global corpus hypergraph already
  exists if the deliverable is only a batch-local report.

## 9. Completion criteria

The mission reaches VERIFY when:

1. A written architecture note exists and names the corpus-level schema and
   ingestion boundary clearly.
2. At least two buildable-now probes exist and run on real superpod-derived
   data.
3. A reality-check memo exists for tutoring, self-play, and formalisation,
   with explicit missing prerequisites and effort bands.
4. The roadmap says which follow-on mission should own each next lane.

## 10. Likely first steps

1. Audit `scripts/hyperreal.py`, `scripts/validate-ct.py`, and current
   superpod output artifacts together, not separately.
2. Choose one real batch or pilot slice as the planning substrate.
3. Implement the smallest paper → concept bridge probe.
4. Implement one packetization probe:
   formalization packet or teaching-unit packet.
5. Only after that, write the longer tutoring/self-play strategy memo.

## 11. Naming note

The title says "planning" deliberately.

This mission is the place to turn an attractive cluster of ambitions into:

- prototype-backed near-term claims
- bounded architectural work
- honest sequencing for the still-plausible longer-range pieces

If the planning/probe mission succeeds, the actual build mission should
likely split by consumer:

- Hyperreal corpus substrate
- formalization packets
- tutoring / learning pathways
- self-play / agent loops
