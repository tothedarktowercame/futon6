# M-metric-harness — the next end-to-end run measures PROGRESS, not throughput

**Status:** HEAD complete; IDENTIFY authored; MAP complete (scratch-work survey);
DERIVE authored (3-axis taxonomy + per-phase & aggregate catalog); ARGUE authored
(paradox resolved); **VERIFY spiked** — concept-coverage accretion slope is real + steep
at small n (0.10→0.56 over k=1→10 on the 9,738-paper CT corpus); INSTANTIATE = build the
harness (2026-06-23).

*Follows `futon4/holes/mission-lifecycle.md`. Successor framing to the mark5 run
(`holes/mark5-ct100-results.md`); executes against the corrected pipeline in
`holes/linode-stepper-contract.md` and the cross-paper mining of
`holes/proofcheck-readiness.html`.*

---

## HEAD

- **Operator-voice anchor (Joe, 2026-06-22):** *"If we do one paper end-to-end we should
  already have metrics, and those metrics should improve if we do 10 papers end to end.
  If they are not improving then we should be able to pinpoint why."* The point is **not**
  "100 papers completed."
- **What's already felt to be true:** Phase 2 is cross-paper/holistic, so the substrate
  should *compound* — a held-out paper grounds better as the corpus grows. The
  comprehension floor is already corpus-relative; G-coverage was designed to rise with
  corpus-fraction. The progress signal is a **slope**, and it should already be latent at
  n=1.
- **Anti-glibness discipline:** throughput is not progress. A pile of graphs, or "N
  papers done," hides whether the holistic claim is real. Every headline number must be
  defined at n=1, expected to rise n=1→10, and **decomposed per stage** so a flat curve
  names the culprit. A flat/falling metric is a *finding to pinpoint*, never something to
  paper over with volume (the mark5 lesson, generalized).
- **Working-economy position:** this mission **underwrites the superpod scale decision** —
  we only pay for archive-scale once the small run shows the slope is real and diagnosable.
  It is underwritten by the corrected full-pipeline contract + the already-built stages.
- **Carried-forward tensions:** (i) the S7 macro-collapse cause is still open — paper-level
  lens vs fixed-vs-data-driven vocab (mark5 D1, Diagnostic 1); (ii) the **S4 expository
  GPU run has never been exercised**; (iii) the **S6 paper-graph(B) assembler is unbuilt**;
  (iv) the embedding/macro-vocab rework is **deferred** (separate work, not this mission).
- **Provenance:** this conversation (2026-06-22→23), as the post-mortem of mark5. Intake =
  operator principle stated after reviewing the readiness docs + the mark5 results.

**Exit criterion (HEAD):** Joe recognises this as the faithful shape of "what we need next
time"; the four tensions are named, not buried.

---

## 1. IDENTIFY

**Motivation.** mark5 ran end-to-end and produced 102 typed CLeans — but the only thing it
*demonstrated* was throughput. It ran one reasoning sibling (④, one proof/paper), skipped
② concepts and the entire ⑤ expository sibling and the comprehension layer, and took no
cross-paper measurement. So we cannot say whether the central Phase-2 claim — *the
representation improves as the corpus grows* — is true. The gap: **we have no instrument
that shows progress, and no run small enough to read the slope yet rich enough to exercise
the holistic features.**

**Theoretical anchoring.** Phase-1 (per-paper) vs Phase-2 (cross-paper holistic); the
corpus-relative comprehension floor (`E-comprehension-foundation.md`); "improves as we
run" (G-coverage rises with corpus-fraction, `coverage_inline.py`); the cross-paper
mining the readiness docs are built for. Adjacent to AIF+ (a metric trajectory is a
path-integral over the run, not a single scalar).

**Scope in:** (a) **the metrics harness** — compute the rising metrics at n=1 and n=10,
emit the slope + per-stage attribution; (b) a **small full-pipeline run** (10 then 20
whole papers, *everything on* — both siblings, concepts, comprehension, cross-paper
mining). **Scope out (deferred):** the embedding/macro-vocab rework (its own work, gated
on Diagnostic findings); archive scale; RAW-CTL (operator decides separately).

**Completion criteria (testable).**
1. Every headline metric is **computed at n=1** (defined, non-trivial value).
2. For each, the **n=1→10 slope** is emitted *with per-stage attribution*.
3. The metrics that *should* rise (G-coverage, comprehension, recurring-(type,concept)
   holes) **either demonstrably rise or yield a pinpointed reason for flatness**.
4. The run has **both reasoning siblings + concepts + comprehension + cross-paper mining
   ON** — verified, no thin slice (whole-paper unit, object B).
5. The **headline is the slope, not "N papers."**

**Relationship to other missions/docs.** *Depends on* the corrected contract's
needs-build items (S3 all-proofs extraction, S4 expository GPU run, S6 paper-graph(B)
assembler). *Enables* the superpod scale decision (`superpod-dag-contract.md`).
*Supersedes* mark5's throughput framing.

**Source material.** `linode-stepper-contract.md` (DAG + feature grid),
`proofcheck-readiness.html` (cross-paper mining), `pre-superpod-pipeline-readiness.html`
(phases), `mark5-ct100-results.md` (throughput-without-metrics), `E-comprehension-foundation.md`,
`data/mark5-ct100-run/` (102 artifacts in hand for harness prototyping).

**Owner & dependencies.** claude-1 + Joe; futon6 (CPU harness + the small run); a LLaMA/70B
box for the S3/S4 GPU stages.

### Candidate metrics (operator-seeded, 2026-06-23)

Joe's likely candidates — the intuition-level targets that motivate the harness; the
DERIVE table formalizes them. **The list is open** (more will surface in MAP).

- **Any-markup coverage % (per paper).** What fraction of a paper's content is covered by
  *any* mark at all. If a paper reads **50% covered, 50% is unmodelled** — earlier local
  experiments sat near this, and that gap is exactly what motivated attending to the
  **expository sections (⑤)**. Defined at n=1; should rise as the *full pipeline* (above
  all the expository sibling) is turned on. A flat-low value says the feature-set is
  missing whole regions, not that the paper is hard. **This is the direct test that we
  model the WHOLE paper, not just its proofs** — so its rise when ⑤ is on is the evidence
  the expository sibling earns its keep.
- **Symbol-grounding %, by symbol *kind*.** Not one number but a small, rigorous taxonomy,
  now that "symbol" has widened: (i) **variables inside expressions** (classic SFC2b),
  (ii) **named concepts** (the noun layer), (iii) **proof-moves as concepts** (the verb
  layer — a move is "grounded" iff it resolves to a known technique/pattern, else honestly
  flagged). The verb layer is the hard, interesting part; stay rigorous by requiring a
  cited definition / pattern-match or an explicit `:undefined`/`:thin` flag (the SFC2b +
  rung-3 discipline — never count an ungrounded symbol as grounded). Per-kind, defined at
  n=1; rises with substrate growth + as each kind is handled.
- **# concepts defined in the encyclopedia.** Even in CT (a good seed set), the count of
  concepts carrying a *real definition entry* should **rise as papers accrete** — each
  paper contributes definitions/usages. A pure cross-paper accretion metric: seed at n=1,
  climbing n=1→10. Flat ⇒ the encyclopedia isn't actually ingesting new definitions.
- **Weak-point identification + our confidence in it.** Flag the load-bearing-but-thin
  steps / undischarged gaps in a paper (rung-3 thin-move detector; the conjecture /
  weak-proof map) — but **paired with a confidence that distinguishes a weak *proof* from
  a weak *model*** (low comprehension ⇒ "study more / weak-extraction", *never*
  "weak-proof" — the comprehension verdict gate). Two-dimensional: (# flagged weak points,
  confidence/comprehension at each flag). Defined at n=1; the *confidence* should rise with
  the corpus (better grounding ⇒ more flags we can trust as real, fewer mistaken for model
  weakness).
- **(more to come)** — deliberately open; MAP-Q1 sorts which rise vs saturate at small n.

**Exit criterion (IDENTIFY):** Joe agrees the gap is real and the scope (harness + small
everything-on run, rework deferred) is right.

---

## 2. MAP — survey of prior runs / data / renders (2026-06-23)

*The "scratch work" is sprawling: ~15 run/experiment dirs, ~20 G of rendered HTML, ~35 G
of grabbed-back SE/MO data, plus per-feature slices. Surveyed via an Explore sweep +
targeted inventory. **Headline MAP finding at the bottom: no run ever varied n, so a
progress *slope* has literally never been measured — that absence is the mission's whole
point.***

### Runs / experiment slices

| run | type | date | location | produced | status |
|-----|------|------|----------|----------|--------|
| loop-run | laptop | 06-16 | `data/iatc-argument-graphs/loop-run` | 10 finals | early baseline |
| loop-run-dpdemo{,-fixed,-final} | laptop | 06-16 | same parent | render artifacts (0/2/1 finals) | superseded by dp-demo render |
| gh200 | laptop | 06-16 | `…/gh200` + showcases/gh200 (182 HTML) | 15 finals | baseline; `1308.1804` true-positive vacuous-edge catch |
| **loop-run-70b** (mark4 go-live) | laptop+vLLM 70B | 06-17 | `…/loop-run-70b` | 9 finals, checker 9/9 · substance 9/9 · grounding 21.4% | preregistered; the enriched baseline |
| **linode-stageA-20260618** | Linode | 06-18 | `…/linode-stageA-20260618` | 9 finals (top-level) | Stage-A staged run |
| **RAW-CTL** | Linode | 06-18 | `data/exp-20260618/loop-run-70b-raw` | 10 finals + eval; grounding **12.5%**, substance lower | **RAN** — raw degrades vs enriched ⇒ enrichment earns its keep |
| EXP-3b / BGE retrieval | Linode/laptop | 06-18 | `data/exp-20260618/{bge-cas-sel-3b,exp3b-context}` | 6+6 entries | the text-vs-structure retrieval arc |
| per-feature slices | laptop | 06-17/18 | `data/{rung3-technique,symbol-grounding,cas-select-steps,expository-scope-graphs}/loop-run-70b` | one slice each | each exercises ONE proofcheck stage in isolation |
| **mark5-ct100-run** | Linode | 06-22 | `data/mark5-ct100-run` | 106 graphs, 102 CLeans, embeddings, export | latest; macro-collapse (see results note) |

### Data stores

| store | size | count | note |
|-------|------|-------|------|
| eprints `storage/futon6/data/arxiv-math-ct-eprints` | 1.9 G | 9,798 `.tar.gz` (9,916 in metadata) | the math.CT corpus; active ingestion |
| warp substrate `data/warp` | 1.2 G | concept-index = **3,623 concepts** (Jun 18) | the (stale-for-mark5) concept model |
| candidates | — | iatc-candidates 10 · **ct200 199** · dpdemo 5 | ct200 = the mark5 pre-stage |
| grabbed-back corpora `storage/futon6` | **35 G se-data** · 613 M mo-processed · math/math-se | the P7 StackExchange/MO era |
| NER kernel `data/ner-kernel*` | 1.5 M | ~18.9 K terms | **SE-physics-derived (May 23) — stale for CT** |

### Rendered examples (the laptop "test-coverage" runs Joe recalls)

| showcase | size | when | shows |
|----------|------|------|-------|
| `ct-anatomy/golden` | **18 G / 9,774 HTML** | 06-15 | the big anatomy render (object-layer coverage) |
| `ct-anatomy/gh200` | 845 M / 182 HTML | 06-16/18 | gh200 anatomy + IATC goldens |
| `batch-008-math-ct-qc{,-v2,-v3}` | ~24 M | **05-20/23** | the math.CT QC / test-coverage pages |
| `distributed-proofreaders/*` | — | 05-20 | earlier DP render experiments |
| `ct-anatomy/dp-demo` (11) · `proofcheck-demo` (1) · `clean-demo` (9-proof) · `clean-ct200-demo` (mark5) | — | 06-17/22 | the proof-check + CLean demos |

### Ready vs missing (for the harness)

| ready — data exists to prototype a metric on | missing — the work |
|---|---|
| mark5 102 (structure/method) · loop-run-70b **enriched + raw** (substance/grounding contrast) · expository-scope-graphs slice (expository coverage) · symbol-grounding + rung3-technique slices · warp concept-index (encyclopedia count) · eprint corpus + S1 (any-markup coverage) | **the harness itself** (compute metrics + n=1→n=10 slope + per-stage attribution) |
| each candidate metric has a producing stage / a prior slice | **S3 all-proofs**, **S4 expository at scale**, **S6 paper-graph(B)** |
| | **a single consistent run that varies n** (none exists — see below) |

### MAP questions — partial answers

- **Q1 (which rise vs saturate):** *cannot yet be read from the archive* — every run is a
  single fixed-n snapshot; none sweeps n. Must be measured by the harness on one
  consistent run. The mark5 102 + the slices let us prototype the *metric definitions*, not
  the slope.
- **Q2 (n=1 baseline):** open (design in DERIVE) — leave-one-out vs self vs zero per metric.
- **Q3 (attribution):** *feasible* — the per-feature slices (rung3, symbol-grounding,
  expository-scope) prove each stage's output can be measured in isolation, so a flat
  top-line can be decomposed by re-running the same stage scripts per metric.

### Surprises (recorded before DERIVE locks in)

1. **RAW-CTL DID run** (06-18) — correcting the contract; enrichment earns its keep.
2. **No run ever varied n.** Every artifact is a fixed-size snapshot (10, 15, ~100). So
   the *slope* — the mission's core deliverable — has **never been measured**. That's
   exactly why there's no progress signal today: everything is a one-shot.
3. **Per-stage outputs already exist as isolated slices** (rung3, symbol-grounding,
   expository-scope) → per-stage attribution (Q3) is buildable by reuse.
4. The scratch is large + uneven (18 G golden render; 35 G stale SE data; NER kernel
   SE-derived) — the harness should ignore the stale corpora and measure on the live
   math.CT path.

---

## 3. DERIVE (2026-06-23)

### Metric taxonomy — three axes (they "rise" differently)

The core design move. Every metric is exactly one of:

- **ACCRETION** — rises with corpus size n (leave-one-out at n=1). The corpus *compounds*.
  **This axis IS the progress slope — the headline.** A held-out paper grounds / is
  comprehended better as the substrate grows.
- **COMPLETENESS** — rises when *features* are turned on; defined per-paper at n=1; roughly
  flat across n. Answers "how much of *each* paper do we model?" The signal is the
  **features-on jump** (proof-only → +expository), not n.
- **QUALITY / FLOOR** — should stay above a threshold; not n-dependent. Soundness of the
  extraction.

Conflating these is exactly how mark5 reported "100 papers done" over a stalled signal.
**The progress claim rests on the ACCRETION slopes**; COMPLETENESS shows feature value;
QUALITY are the gates. The harness reports all three, labelled by axis.

### Per-phase metric catalog (A=accretion · C=completeness · Q=quality)

| stage | metric | axis | n=1 baseline | moves with | flat ⇒ suspect |
|-------|--------|:----:|--------------|-----------|----------------|
| **S1 ① anatomy** | any-markup coverage % | C | paper-1 % covered by any mark | features-on (esp. ⑤) | a sibling/detector off |
| | mark-kind density (proof-move / def / expository / symbol) | C | per-paper breakdown | features-on | detector gaps |
| | wf=0, proof-region coverage | Q/C | per-paper | — | detector overfit |
| **S2 ② concepts** | # encyclopedia concepts defined | A | CT seed count | each paper accretes defs | encyclopedia not ingesting |
| | concept-coverage / G-coverage | A | paper vs substrate-of-1 | substrate grows | ② substrate / SFC |
| | prose-concept P/R, term-prior resolution | Q | per golden | — | detector quality |
| **S3 ④ IATC** | yield (pass/fail/retry) | Q | per run | — | generator / gates |
| | substance-%, grounding-% (warrant-resolution) | Q | per finals | — | shell-gaming / style |
| | structural diversity (distinct shapes) | Q | per run | — | template collapse |
| | anchor-faithfulness (R2a) | Q | per node | — | hallucinated source |
| **S4 ⑤ expository** | expository scope coverage % | C→sat | scopes on paper-1 | features + minted vocab (saturating ~35%) | vocab / hole-fill |
| | typed-hole fill rate (of 16 scopes) | C | per paper | hole-filling loop | the loop |
| | expository-argcheck pass | Q | per graph | — | malformed fills |
| **S5 comprehension** | comprehension floor (corpus-relative) | A | per-proof vs corpus-of-1 | nouns+strategy ground as n grows | R2d or STRAT-REC axis |
| | noun axis (R2d) / strategy axis (rung-3+STRAT-REC) | A | per-proof | substrate / co-learning | whichever axis is flat |
| | symbol-grounding %, by kind {var, named-concept, proof-move} | C+A | per-kind on paper-1 | each kind handled + substrate | the unhandled kind / grounding loop |
| | weak-points flagged + **confidence** | A (confidence) | per-proof | confidence rises with grounding | verdict gate / comprehension |
| **S6 paper-graph (B)** | B completeness (statement → proof-substructure-or-flag) | C | per paper | assembler + both siblings | a sibling missing |
| | proof↔statement attachment rate | C/Q | per paper | attachment heuristic | attachment logic |
| | expository connectivity (tissue linking proofs↔statements) | C | per paper | ⑤ on | expository sibling off |
| **S7 ⑧ embed** | G-entropy (macro-entropy + off-diag cosine) | Q | per run | vocab/weighting | macro collapse (mark5 D1) |
| | structure-vs-text retrieval gap (EXP-3 at scale) | A | n/a at n=1 | proof-space populates | embedding weighting |
| | method-spine diversity | Q | per run | — | typing collapse |
| **S8 ⑧ export** | row-counts match, ANN sanity | Q | per export | — | export bug |

### Aggregate metrics (across papers / the whole run)

- **The slope of every ACCRETION metric (n=1→n=10)** — *the* headline progress artifact.
- **Corpus completeness:** distribution of any-markup coverage across papers + the
  proof-only-vs-full-pipeline **features-on delta** (the evidence ⑤ earns its keep).
- **Cross-paper recurrence:** # recurring (type,concept) holes (df≥2) = the size of the
  conjecture / weak-proof map; grows with n.
- **Cross-paper retrieval:** structure-clustering-by-method-across-topics (the EXP-3
  "0.95 vs 0.24" claim, *reproduced at n* — does structure beat text at scale).
- **Comprehension distribution:** fraction comprehended vs "study-more", and how it shifts
  as n grows (with the weak-point confidence attached).

### n=1 baseline + slope mechanism (per axis)

- **ACCRETION → leave-one-out.** metric(held-out paper | substrate built from the other k),
  swept k=1..N. At n=1 the substrate is the single paper itself (self-grounding floor).
  **IF** we want "rises 1→10" **HOWEVER** raw metric-vs-n confounds "corpus helps" with
  "later papers happen to be easier" **THEN** measure against a *fixed held-out set* as the
  substrate grows **BECAUSE** that isolates the corpus contribution.
- **COMPLETENESS → per-paper distribution + features-on delta** (proof-only vs full pipeline);
  roughly flat in n by design.
- **QUALITY → per-paper vs floor + aggregate pass-rate.**

### Entity / relation types (harness data model)

- `MetricRecord {run-id, corpus-id, paper-id, stage, metric, axis, value, n, computable}`
  — `n` = substrate size for accretion metrics.
- `SlopeReport {metric, points:[(k,value)], slope, rising?, attribution-stage}`.
- Relation: each aggregate metric → its component per-paper `MetricRecord`s (for attribution).

### Data flow

Each stage emits `MetricRecord`s per paper → the harness collects → for ACCRETION metrics
it **replays at k=1..n** (leave-one-out / incremental) → `SlopeReport`s + per-stage
attribution → **the slope report** (the deliverable, and the superpod go/no-go input).

### Invariants

- Every required stage emits ≥1 `MetricRecord` per paper.
- Every ACCRETION metric is **leave-one-out computable at n=1** (nothing undefined-at-1).
- A flat/falling accretion slope **MUST resolve to a named stage** (`attribution-stage`
  non-null) — else the *harness* is incomplete, not the pipeline.
- COMPLETENESS metrics report the **features-on delta**, never a bare single number.

### IF/HOWEVER/THEN/BECAUSE (key decisions)

1. **IF** progress = "improves as we run" **HOWEVER** completeness metrics are ~flat in n
   **THEN** split the three axes and rest the progress claim on ACCRETION slopes **BECAUSE**
   conflating them is how mark5 mistook throughput for progress.
2. **IF** a slope is flat **HOWEVER** a single top-line hides the cause **THEN** require
   per-stage attribution **BECAUSE** "pinpoint why" is the operator's explicit requirement.
3. **IF** the verb layer (proof-move grounding) is fuzzy **HOWEVER** rigor matters **THEN**
   count a move grounded *only* on a cited definition / pattern-match, else an explicit
   `:thin`/`:undefined` flag **BECAUSE** an inflated grounding-% would be the worst lie.

### Anti-patterns (explicitly out)

"N papers completed" headline; one reasoning sibling; one proof per paper; raw counts
without the slope; an accretion slope reported without attribution.

---

## 4. ARGUE (2026-06-23)

### Pattern cross-reference (`futon3/library/`)

- **`futon-theory/progress-signal`** — the canonical pattern this mission instantiates:
  *measure progress via accumulating evidence + explicit health states so stall / dead
  work is caught early, not felt as vague stagnation.* The harness **extends** it from
  mission-progress to **corpus-progress** — the accretion slope IS the evidence-
  accumulation curve.
- **`collaboration-coherence/navel-gazing`** ("close every feedback loop with an action
  surface; reflection without one is noise / a Bateson double-bind") — the discipline
  against Joe's paradox. **Every metric must close to an action:** the `attribution-stage`
  is *where the fix goes*; the slope is *the scale / no-scale decision*. A metric admired
  for its own sake is exactly the failure mode this pattern names.
- **`f6/graph-enhanced-evaluation`** ("two agents, same questions, one has the graph; the
  difference measures the value of the structure") — the **teleological anchor**. This is
  the real question the harness's intermediate metrics only *approximate*: does the mined
  structure make downstream reasoning measurably better? The accretion slopes are cheap,
  early **proxies** for that bare-vs-graph-enhanced delta; the harness keeps a line of
  sight to it as ground truth.
- **`peeragogy/use-or-make`** — reuse before produce: the MAP found per-feature stage
  slices already on disk; the harness *composes existing stage outputs*, it doesn't mint
  new measurement machinery.

### Theoretical coherence — the paradox resolved

Joe's paradox (*metrics must not become the target; the real question is what the mined
data lets us **do***) is resolved by two patterns acting together: **navel-gazing** keeps
each metric tied to an action (a steering signal, not a score to game), and
**graph-enhanced-evaluation** keeps the *whole set* honest as a proxy for downstream
use-value (so we never optimize coverage-% at the expense of usefulness). The intermediate
per-stage metrics earn their place **only** as early, cheap, decomposable approximations
of the bare-vs-graph-enhanced delta — which is too expensive to run every iteration.

**Trade-off accepted:** the slopes are proxies (cheap, early, per-stage) for the true
use-value metric (graph-enhanced QA, expensive). The harness's job is to make the proxies
trustworthy enough to steer *between* rare ground-truth evaluations.

### Plain-language argument

We're building a gauge for the mining pipeline. Today we can run it on 100 papers and all
we learn is "it finished." The gauge instead asks: when we add papers, does the system
actually understand *more* — and if not, which part is stuck? It does this with a few
numbers that each (a) already mean something on a single paper, (b) should climb as papers
accumulate, and (c) point at the responsible stage when they don't. The numbers aren't the
goal — they're a cheap early stand-in for the real goal (does the mined structure make
downstream mathematical reasoning measurably better), which we keep honest by occasionally
checking them against an actual bare-vs-graph-enhanced test.

## 5. VERIFY (2026-06-23)

### Spike — concept-coverage accretion slope (CPU, free, data in hand)

Inverted `data/warp/concept-index.json` (3,623 concepts × **9,738 CT papers**) to
paper→concepts; for 20 held-out papers, measured coverage vs substrate size k using the
DERIVE leave-one-out mechanism:

```
 k (substrate)  :   1     2     5    10    20    50   100   200   400  9718
 distinct conc. : 102   274   551   844  1259  1798  2482  2959  3317  3623
 distinct DEFINED: 101   273   547   837  1243  1776  2442  2912  3260  3559
 held-out cov.  :0.098 0.226 0.422 0.564 0.683 0.822 0.918 0.967 0.994 1.000
```

**Result — the accretion slope is real and steep at small n.** Held-out concept-coverage
rises **0.10 → 0.56 over k=1→10 (~5.7×)** and saturates near 1.0 by k≈400; distinct
concepts (and distinct *defined* concepts) accrete monotonically.

**What this verifies (the riskiest DERIVE commitments):**
1. The leave-one-out slope mechanism produces a real, monotone rising curve on actual data.
2. The slope is steep precisely in the **1→20 range Joe targets** — "small first, read the
   slope" is sound; we don't need 100s of papers to see progress.
3. An ACCRETION metric behaves as the taxonomy predicts.

**Caveats (honest).** Uses the existing curated concept-index (HAPAX-filtered); coverage
here is "held-out concepts appearing in other papers' concept sets" — a proxy for
grounding-against-*definitions* (the fuller version rebuilds the def-substrate per k, an
INSTANTIATE concern). One metric spiked; the rest land in INSTANTIATE.

**Decision log.** No DERIVE revision needed — the spike confirms accretion slopes are real
and readable at small n. Risk that "there's no slope to find" is retired.

## 6. INSTANTIATE · 7. DOCUMENT — *forward path*

- **INSTANTIATE:** build the harness (`MetricRecord`/`SlopeReport` emit + leave-one-out
  replay + per-stage attribution); wire S3-all-proofs + S4 expository + S6 assembler; run
  10 then 20 whole papers everything-on; emit the slope + attribution. Keep a line of
  sight to `f6/graph-enhanced-evaluation` as the ground-truth proxy check.
- **DOCUMENT:** the slope report becomes the progress artifact (and the superpod
  go/no-go input).
