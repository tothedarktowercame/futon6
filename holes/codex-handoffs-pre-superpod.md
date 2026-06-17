# Pre-Superpod — Codex handoffs (DRAFT for review · NOT dispatched)

Derived from claude-6's `holes/pre-superpod-pipeline-readiness.html`, one handoff
per non-`ready` card, in card order from 1.1. **These are written for review — none
have been dispatched.** Each follows the AGENTS.md handoff shape (goal · files ·
defect · fix · acceptance bar · gates) so a Codex agent could run it as-is once we
approve. Liabilities #1–#3 are already fixed; `ready`/`verified` cards (1.2, 1.3,
3.1, 3.2, 4.1–4.4, 7.2) get no handoff. Re-numbered H1… for dispatch tracking.

Gate legend (AGENTS.md): **PY** = `pytest tests/` + the card's own check; **BB** =
`clj-kondo` + `futon4/dev/check-parens.el` on any `.bb` touched. Every handoff ends
with "bell claude-1 back with a summary + commit shas."

---

## H1 — card 1.1 · Detector reasoning layer (OVERFIT)  · size L · PY
**Defect (verified):** the object layer is the validated floor (wf=0, tagged/math
100% cross-MSC), but the **reasoning layer is overfit** — claim/inference/illative
marks anchored on only **5/7 demo papers** after hand-iteration; **illative-coverage
is the named dominant gap**; text-style `Proof.` proofs (e.g. **1012.1220**) are
still missed entirely.
**Goal:** generalise the reasoning-layer detectors so illative/claim/inference marks
fire robustly beyond the tuned demo set, including text-style (non-environment)
proofs — without touching the object layer.
**Files:** `scripts/dp_paper_view.py` (`detect_inferences`, `detect_implications`,
`detect_proof_macros`, the reasoning-layer block ~L170–420); demo papers incl.
`1012.1220`; `scripts/dp_capabilities/`.
**Fix:** (a) build a held-out coverage harness over ≥15 papers beyond the 5/7;
(b) diagnose the 2/7 unanchored + why `1012.1220`'s text-style `Proof.` (vs a
`\begin{proof}` env) is missed; (c) widen heuristics and **gate on the held-out
set** to avoid re-overfitting. Discipline: **fix the detector, never the checker**.
**Acceptance:** illative/inference coverage reported on the held-out set, measurably
> the 5/7 baseline; `1012.1220` text-proof now anchored; `check_invariants.py`
wf-errors stay **0** on all probe classes (object layer unbroken).
**Gates:** PY + report the before/after coverage numbers.

## H2 — card 2.1 · Prose-concept layer (UNTESTED)  · size S · PY
**Defect (inferred):** `build_golden_paper.py` produces the named-term concept marks
that `dp_enrich` merges, but there is **no independent validation** — and "terms not
noticed" (DC-1) is the dominant visible demo defect.
**Goal:** a precision/coverage audit + a stated gate for the prose-concept layer.
**Files:** `scripts/build_golden_paper.py`, `scripts/dp_enrich.py`,
`holes/dp-defect-catalogue.md` (DC-1/DC-2).
**Fix:** a 100-row precision audit (sample of papers): of named math terms in prose,
what fraction get a concept mark (recall) and what fraction of marks are real
(precision); wire it as the **C-TERM-COVERAGE** measurement the catalogue names.
**Acceptance:** reported precision + recall on the sample; the residual DC-1 miss
rate quantified, not just asserted.
**Gates:** PY.

## H3 — card 2.2 · Term prior (UNTESTED)  · size S · PY
**Defect (documented):** `build_term_prior.py` is present with the OVERFED/HUNGRY/
HAPAX resolution design (E-prior-over-terms), but **no validation evidence seen**.
**Goal:** validate the df-prior's three behaviours on real terms.
**Files:** `scripts/build_term_prior.py`.
**Fix:** fixture set of known OVERFED (trim-to-core), HUNGRY (extend), HAPAX (df=1,
drop) phrases; assert the prior resolves each as designed; report counts on a corpus
slice.
**Acceptance:** each of the three cases demonstrated on real terms with numbers.
**Gates:** PY.

## H4 — card 2.3 · Concept encyclopedia (UNTESTED)  · size M · PY
**Defect (documented):** `build_concept_encyclopedia.py` assembles concept entries
(def passage + provenance + dep edges + centrality), but the deep semi-formalisation
is an explicit HOLE and the structure-first substrate is **unvalidated**.
**Goal:** validate the structure-first entries (the parts claimed real), and make the
semi-formal HOLE explicit/typed (not silently empty).
**Files:** `scripts/build_concept_encyclopedia.py`; NNexus/nLab index.
**Fix:** audit a sample of entries for (def-passage present, provenance link
resolves, ≥1 dep edge, centrality computed); emit a coverage table; ensure the
semi-formal slot is a declared `:hole`, not absent.
**Acceptance:** per-field completeness numbers on the sample; HOLE is typed.
**Gates:** PY.

## H5 — card 4.5 · Substance gate — the 1/15 failure (PARTIAL)  · size S · PY
**Defect (verified):** the substance gate caught 14/15 on the gh200 run; **the single
failure is unexamined** (also an open question in the doc footer).
**Goal:** identify the failing paper and classify: real shell/miss vs fixture issue;
fix or document.
**Files:** `scripts/substance_gate.py`; the gh200 run dir; the failing `.edn`.
**Fix:** locate the 1/15, run the gate with reasons, decide real-vs-fixture; if a
gate bug, fix + add a regression fixture; if a real catch, document it as expected.
**Acceptance:** the failure named, classified, and either fixed (with a fixture) or
documented as a true positive.
**Gates:** PY.

## H6 — card 4.6 · Render side-by-side (UNTESTED)  · size S–M · PY
**Defect (inferred):** `build_iatc_goldens.py` (CPU-marks vs GPU-marks demo) exists
but is untested, and the comparison is buried under the LaTeX preamble (the GPU/IATC
marks live only in the reconstructed passage).
**Goal:** finish + validate the side-by-side so the comparison is immediately legible.
**Files:** `scripts/build_iatc_goldens.py`; `data/showcases/ct-anatomy/dp-demo/`.
**Fix:** crop each paper to its passage window (slice text + rebase both columns'
mark offsets) so CPU-vs-GPU marks align on-screen; wire a link from `dp-demo/index.html`
(Mockups section); re-run after the enriched 70B re-run (H-headline) so it shows real
graphs.
**Acceptance:** demo renders all goldens window-focused, both columns marked-up text,
linked from the index; screenshot verified.
**Gates:** PY (+ visual screenshot check).

## H7 — card 5.1 · Expository region carve + scope vocabulary (UNTESTED)  · size M · PY
**Defect (documented):** `expository_region_extract.py` + `consolidate_scope_votes.py`
carve expository regions (leaf-section, `inflight` prose gaps) and mint scope vocab,
but the work **hadn't finalised** — and it had been pulled toward full IATC-grade
machinery (14 warrant/hole/inference refs) when the CPU carve should be the *lighter*
analogue of Weft (①). The *informal reasoning* over these regions is H8 (GPU), not here.
**Goal:** finalise + validate the CPU carve and the minted expository-scope
vocabulary; strip the IATC-grade reasoning machinery out of the carve (it belongs in
H8).
**Files:** `scripts/expository_region_extract.py` (940 L, uncommitted),
`scripts/consolidate_scope_votes.py`, `scripts/dp_enrich.py` (429 L, uncommitted).
**Fix:** confirm region carving on the goldens (e.g. 0905.0595 L202–208, the doc's
example); land the minted scope vocabulary; commit the two uncommitted scripts; gate
on region-coverage over a sample. Leave reasoning-reconstruction to H8.
**Acceptance:** carve validated on ≥5 goldens with a region-coverage number; scope
vocab minted + documented; scripts committed.
**Gates:** PY. **Decision for Joe:** confirm the carve/reasoning split (CPU carve here,
GPU reasoning in H8) before this lands.

## H8 — card 5.2 · Informal-reasoning reconstruction (VISION — no script)  · size XL / mission
**Defect (inferred):** the GPU sibling of IATC over expository regions — performatives
/ value / meta / analogy / motivation — **does not exist yet** (no script). It must
absorb the IATC categories dropped from illative-only IATC.
**Goal:** scaffold the candidate→GPU-loop→gate trio mirroring Phase ④, with an
**informal-reasoning checker** analogous to `iatc_argcheck`.
**Open design questions (Joe's call BEFORE any code):** (1) its own schema — what is
the informal-reasoning graph (perf[Assert/Suggest/Judge/Challenge/Query],
value[easy/plausible/beautiful/useful], meta[goal/strategy/analogy/generalise])?
(2) what does its `iatc_argcheck`-analogue check? (3) which categories land first?
**Recommendation:** this is **mission-sized and design-blocked**, not a one-shot
Codex handoff. Scope the schema + checker with Joe first (a design session), then it
splits into: H8a candidate extractor, H8b GPU loop, H8c informal checker/gate.
**Gates:** BB (the checker will be `.bb`) + PY, once scoped.

## H9 — card 6.1 · APM ↔ eprint scope coverage (UNTESTED)  · size M · PY
**Defect (documented):** `mark4_apm_structure_coverage.py` runs (3 match flavours,
coarsest→tightest) but is **not validated** here; design in
`holes/apm-structure-match-design.md`.
**Goal:** validate the coverage metric + establish its gate (what counts as signal).
**Files:** `scripts/mark4_apm_structure_coverage.py`;
`holes/apm-structure-match-design.md`; the frozen APM scopes + eprint scope pool.
**Fix:** confirm the type-only saturation vs type+multichar discrimination
(baseline last run: mean .26 / median .14, 13-proof tail ≥80%); decide whether the
tightest flavour or an embedding matcher is the real metric; state the gate.
**Acceptance:** the three flavours reported on the frozen set; a chosen metric + gate
documented; the disagreement-vs-keyword diagnostic noted as future arm.
**Gates:** PY.

---

## Not written as Codex handoffs (and why)
- **7.3 Eval harness re-run** — the *headline action* (enrichment-fed 70B re-run to
  replace the blind eval). Not a code fix; it's the GPU run + owner review, claude-6's
  current lane. Leave to that thread.
- **3.3 Landscape overlays · 7.1 Pipeline-runner→executor · 1.2 line-review** —
  optional/inspection-only or already-`ready`; defer unless prioritised.

## Suggested order (dependency-aware)
1. **H1** (detector reasoning) — upstream of every GPU reasoning stage; the dominant gap.
2. **H5, H6** — small, close out Phase ④ (the validated stage).
3. **H2/H3/H4** — concept substrate validation (parallelisable, small).
4. **H7** then **H8 design session** — the expository sibling (H7 unblocks H8).
5. **H9** — APM match validation.

## Remaining gaps — H2 (codex-2)

Implemented a reusable `C-TERM-COVERAGE` audit in
`scripts/build_golden_paper.py`: given independently sampled expected prose
terms, it reports sample rows, concept marks, true/false-positive marks,
precision, recall, and missed terms.

Gate evidence:

- focused pytest: `tests/test_pre_superpod_concept_substrate.py`;
- real 100-row sample from `data/showcases/ct-anatomy/golden/*.json`:
  precision `0.0724`, recall `0.14`, 221 concept marks, 16 true-positive marks,
  205 false-positive marks under the strict one-expected-term-per-row audit.

Remaining gap: the measurement is now wired and exposes the defect, but the
detector remains low-recall/noisy on this strict sample. DC-1 is quantified, not
solved.

## Remaining gaps — H3 (codex-2)

Implemented public term-prior helpers in `scripts/build_term_prior.py`:
`document_frequencies`, `build_index`, and `resolve_phrase`. The resolver now
returns explicit `OVERFED`, `HUNGRY`, `HAPAX`, or `KEPT` decisions with df
evidence.

Gate evidence on 200 golden rendered papers:

- `OVERFED`: `interesting abelian category` -> `category` (`resolved_df=194`);
- `HUNGRY`: `category of modules` -> `category of modules over` (`df=23`,
  `resolved_df=14`);
- `HUNGRY`: `model category` -> `cofibrantly generated model category`
  (`df=34`, `resolved_df=6`);
- `HAPAX`: `a-branes for the lg-model` -> dropped (`df=1`).

Remaining gap: the three behaviours are validated, but the OVERFED example shows
the current highest-df-core rule can trim too aggressively to a generic head
(`category`). A later precision pass should prefer semantically specific cores
when several eligible subphrases exist.

## Remaining gaps — H4 (codex-2)

Implemented `audit_entries` in `scripts/build_concept_encyclopedia.py` and made
the semi-formalisation gap a typed hole:
`{\"kind\":\"hole\",\"type\":\"formalise-structure\",...}` in JSON and
`{:kind :hole :type :formalise-structure ...}` in EDN.

Gate evidence from a temp `--n 50` build:

- def-passage completeness: `50/50` (`1.00`);
- provenance completeness: `42/50` (`0.84`);
- dependency-edge completeness: `28/50` (`0.56`);
- centrality completeness: `37/50` (`0.74`);
- typed-hole completeness: `50/50` (`1.00`).

Remaining gap: structure-first entries are now auditable and the formalisation
HOLE is typed, but provenance, dep edges, and centrality are incomplete for a
substantial minority of entries.

## Remaining gaps — H5 (codex-3)

Classified the GH200 substance-gate failure as a true positive, not a fixture
issue. `scripts/substance_gate.py --self-check` passes, and the GH200 batch has
exactly one failing graph:

- `data/iatc-argument-graphs/gh200/1308.1804.edn`
- reason: `:e-QE-supported` has `:premise :theorem-QE` and
  `:conclusion :theorem-QE`, so the graph contains vacuous X-implies-X
  reasoning even though the intended premises are present separately under
  `:given`.

Remaining gap: no code change is needed for H5; the failure should remain a
substance-gate catch until the graph is regenerated or manually repaired.

## Remaining gaps — H6 (codex-3)

Finished `scripts/build_iatc_goldens.py` so each golden is rendered in the
IATC passage window rather than from the LaTeX preamble. The builder now:

- derives the crop from graph source line ranges;
- slices the source text to that window;
- rebases both CPU anatomy marks and GPU/IATC marks into the cropped text;
- writes `data/showcases/ct-anatomy/dp-demo/mark4-iatc-goldens.html`.

Added `data/showcases/ct-anatomy/dp-demo/index.html` with a Mockups link to the
side-by-side page. Headless Chrome screenshot verification passed at:
`/tmp/futon6-screens/mark4-iatc-goldens.png`.

Remaining gap: the page still reflects the available
`loop-run-dpdemo-final` artifacts: only `0801.2567.edn` exists as a final graph,
with the other four papers falling back to `.attempts/*.attempt2.edn`.

## Remaining gaps — H7 (codex-3)

Finalised the CPU expository carve lane as a structural extractor, not an
IATC-grade reasoning reconstruction. Validation ran on five goldens:

- `0905.0595`: 1 region, 183/209 body lines, 87.56%;
- `0711.1761`: 84 regions, 845/2714 body lines, 31.13%;
- `0801.2567`: 117 regions, 645/1594 body lines, 40.46%;
- `0807.1872`: 1 region, 42/259 body lines, 16.22%;
- `0710.2254`: 6 regions, 19/194 body lines, 9.79%.

The documented 0905.0595 lines 202-208 are covered by
`0905.0595-leaf-0001` (`leaf-section`, lines 81-263). Scope-vote
consolidation is now repo-local under `holes/excursions/close-reading/` and
mints `connection/bridge-analogy` from 5 papers, 2 agents, and 5 votes. The
generated reports are:

- `holes/excursions/close-reading/consolidation-report.json`;
- `holes/excursions/close-reading/consolidation-report.md`.

Remaining gap: the carve/reasoning split should still be confirmed by Joe
before H8 work starts; H7 intentionally stops at structural region extraction
plus scope-vocabulary consolidation.
