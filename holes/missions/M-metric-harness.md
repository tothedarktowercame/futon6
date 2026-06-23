# M-metric-harness — the next end-to-end run measures PROGRESS, not throughput

**Status:** HEAD complete; IDENTIFY authored (awaiting operator sign-off); MAP/DERIVE
seeded; INSTANTIATE deferred to next session (2026-06-23).

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
- **(more to come)** — deliberately open; MAP-Q1 sorts which rise vs saturate at small n.

**Exit criterion (IDENTIFY):** Joe agrees the gap is real and the scope (harness + small
everything-on run, rework deferred) is right.

---

## 2. MAP — *seeded; to complete next session*

**Ready vs missing.**

| ready (built) | missing (the work) |
|---|---|
| S1 anatomy, S2 substrate, S3 IATC spine, S5 comprehension scripts, S7/S8, all gates | **the metrics harness** (compute + slope + per-stage attribution) |
| each candidate metric has a producing stage (see DERIVE table) | **S3 all-proofs** extraction (mark3_extract_candidates → every proof) |
| 102 mark5 artifacts for harness prototyping | **S4 expository GPU run** (built cfec4f9, never exercised) |
| | **S6 paper-graph(B) assembler** (new component) |

**MAP questions (answer with findings, not speculation, next session):**
- **Q1** Which candidate metrics genuinely rise vs saturate at small n (10–20)? Prototype on the mark5 102 where possible.
- **Q2** What is the correct **n=1 baseline** for an inherently cross-paper metric (self-grounding? zero? leave-one-out)?
- **Q3** What is the per-stage **attribution mechanism** — how does a flat top-line decompose to "stage X isn't contributing"?

---

## 3. DERIVE — *seeded; the metric contract*

Every headline metric must be **(a) defined at n=1**, **(b) expected to rise n=1→10**,
**(c) decomposed per stage**.

| metric | stage | n=1 baseline | why it rises (with n, features, or accretion) | if flat, suspect |
|--------|-------|--------------|---------------------|------------------|
| **any-markup coverage %** (per paper) | ①+④+⑤ | paper-1 % covered by any mark | rises as the full pipeline (esp. ⑤ expository) is on — closes the unmodelled gap | a sibling/detector is off or missing whole regions |
| **symbol-grounding %**, by kind {var, named-concept, proof-move} | SFC2b/R2d/rung-3 | per-kind % grounded on paper-1 | substrate growth + each kind handled; verb-layer is the frontier | the grounding loop / the unhandled kind |
| **# encyclopedia concepts defined** | ② | seed-set count (CT seed) | each paper accretes new definitions | encyclopedia not ingesting new defs |
| concept-coverage / G-coverage | ②/R2d | paper-1 concepts vs substrate-of-1 | held-out terms more often grounded as substrate grows | ② substrate / SFC detector |
| comprehension floor (corpus-relative) | S5 | per-proof vs corpus-of-1 | more papers ground more nouns + strategies | R2d (nouns) or STRAT-REC (strategy) |
| recurring (type,concept) holes surfaced | WARRANT-NORM/PASS3 | 0 (no recurrence at n=1) | cross-paper gaps repeat (df≥2) | the (type,concept) keying |
| structure-retrieval discriminativeness | ⑧ | n/a (no neighbours) | proof-space populates → method clusters sharpen | embedding weighting (mark5 D1/D2) |
| expository scope coverage | ⑤ | scope-kinds on paper-1 | minted scopes cover more sentences (saturating) | expository vocab / hole-fill |
| strategy-recognizer recall | STRAT-REC | recall on paper-1 prose | co-learning grows the vocab | recognizer vocab growth |

**Anti-patterns (explicitly out):** "N papers completed" headline; one reasoning sibling;
one proof per paper; raw counts without the slope.

*IF/HOWEVER/THEN/BECAUSE and the harness data-flow design land here next session.*

---

## 4. ARGUE · 5. VERIFY · 6. INSTANTIATE · 7. DOCUMENT — *forward path*

- **ARGUE:** the harness is the AIF+ move (measure the trajectory, not a scalar); the
  plain-language pitch = "one paper already scores; ten should score better, and if not we
  know which stage to blame."
- **VERIFY:** prototype the rising metrics on the mark5 102 (CPU, free) before the GPU run.
- **INSTANTIATE:** build the harness; wire S3-all-proofs + S4 expository + S6 assembler;
  run 10 then 20 whole papers everything-on; emit the slope + attribution.
- **DOCUMENT:** the slope report becomes the progress artifact (and the superpod go/no-go input).
