# Mission: Structure-Seed Promotion — From Replay Labels to a Live Pattern Inducer

**Date:** 2026-05-20
**Status:** INSTANTIATE (sections 3.1–3.3 landed same session; 3.4 + measurement open)
**Owner:** Joe (POC complete on `nlab-wiring.py` / `build-uncovered-sentence-audit.py` / `superpod-job.py`); next-phase delegate TBD
**Predecessor work in same session:**
- structure-seed signature aggregation and Stage 5 hook (Codex,
  2026-05-20, uncommitted on `scripts/superpod-job.py`)
- POC shippability fixes (this turn):
  - tightened skeletonizer (dropped closed-class prepositions from
    `STRUCTURE_CUE_WORDS` in both copies; added be-verb lemmas)
  - subsequence matcher (`_match_structure_seed_signature` in
    `scripts/superpod-job.py`) replacing exact set-membership replay
  - audit-level `--seed-signatures-json` flag in
    `scripts/build-uncovered-sentence-audit.py`
  - 6 new unit tests + smoke test rewritten to exercise true
    cross-paper generalization (different paper, different terms,
    same backbone)
  - end-to-end evidence: 3 cross-batch firings on real batch-008
    papers (run-A: 0711.4904v1 / 0711.1887v1 / 0801.4067v1 →
    run-B: 0712.0418v1 / 0801.0350v1 / 0802.1450v1)

## 1. What the POC ships

The structure-seed replay loop now actually fires on fresh papers.
Concretely:

- the skeletonizer collapses analogous discourse constructions to
  the same signature across papers (was: paper-specific verbatim
  strings)
- a prior signature is treated as a *claim*: "the structural form
  has these N cue tokens in this order"; a new residual matches if
  the prior tokens appear, in order, as a subsequence of the new
  residual's signature
- the matcher enforces a minimum prior length (`min_tokens=3`) to
  prevent the degenerate `<term>` signature from labelling
  every term-dense sentence
- on real arXiv data, cross-batch matches are not zero. 3 firings
  across 3 fresh papers against 3 prior papers' worth of
  signatures.

## 2. What the POC does *not* ship

This is the part to read carefully.

### 2.1 The fired signatures are too generic to discriminate

The 3 cross-batch firings observed used these priors:
- `<math> <term> <math>` (2 firings)
- `<term> and <term>` (1 firing)

These are real recurring structural shapes — but they fire on
*almost any* term-dense math sentence. As a `learned/structure-seed`
label they don't add information beyond "this sentence has
mathematical content."

The lever isn't the matcher. It's the **signature** itself. Real
discourse structure ("we prove that X is Y", "we now consider X",
"Let X be a Y") lives in longer cue chains. After preposition
dropping those chains exist (the smoke test produces them), but on
real papers most uncovered residuals are bibliography fragments,
broken-encoding lines, or noun phrases without discourse verbs —
they don't generate long backbones.

### 2.2 No promotion to first-class detector

The current label `learned/structure-seed` lives only in
`discourse-wiring.json`. None of `detect_scopes` /
`detect_wires` / `detect_ports` / `detect_labels` consults the
learned signatures. So promoted signatures don't widen *future*
scope coverage — they only annotate residuals after the fact.

### 2.3 No classification

A fired signature isn't typed. Codex's enumerated next steps:
1. score signatures by recurrence and term density
2. classify them as likely `scope` / `wire` / `label`
3. emit candidate generalized patterns
4. apply those in a second-pass detector rather than only as labels

Steps 1 is partial (`structure_seed_candidates` carries `count` and
`paper_count`). Steps 2–4 are unbuilt.

## 3. What the next mission slice should do

In order of dependency:

### 3.1 Raise signature information content

Two complementary moves, neither requires the classifier:

- **Verb-keep, preposition-drop:** already done. Confirm by
  inspecting top-frequency 3+ token signatures across a 10-paper
  sample — they should look like real discourse constructions, not
  bibliography fragments. If they don't, the residual extractor is
  the wrong source: it surfaces low-coverage *prose*, which is
  exactly the prose that lacks discourse cues.
- **Discourse-cue prefilter:** for signatures to be useful as
  patterns, the residual sentence must contain at least one
  discourse verb (`prove`, `show`, `obtain`, `study`, `consider`,
  `define`, `denote`, `let`, `assume`, `recall`). Filter the
  candidate pool to such sentences before aggregation. Predicted
  effect: fewer signatures, all more informative.

### 3.2 Heuristic classification of signatures

Once 3.1 produces signatures with verbs, classify by cue verb:
- `let`, `define`, `denote`, `fix`, `write` → `scope` candidate
- `prove`, `show`, `obtain`, `derive` → `label` candidate
- `then`, `therefore`, `notice`, `observe` → `wire` candidate

Add a `predicted_kind` field on each candidate. Still annotation,
not promotion.

### 3.3 Regex emission with quarantine

For each candidate with `paper_count ≥ 2` and `predicted_kind`,
emit a candidate regex into a separate file
(`learned-discourse-patterns.json`), not into `nlab-wiring.py`
directly. The regex is a translation of the cue tokens into a
loose-match pattern (cue verbs as literal, placeholders as
`\b\w+\b` or `\$[^$]+\$`).

A second-pass detector in `nlab-wiring.py`
(`detect_learned(entity_id, text, learned_patterns)`) consumes this
file and emits records like the existing detectors. Patterns are
gated by `min_papers` / `min_count` thresholds so noisy candidates
don't poison the detector.

### 3.4 Loss-of-loss as stopping rule

`structure_loss` in the Stage 5 `learning_loss` block should
decrease when promoted patterns close residuals. Monitor it across
2–3 promotion cycles; stop when marginal benefit drops below some
threshold or when residuals are dominated by genuinely non-English
prose (German, encoding-damaged).

## 4. Out of scope for this mission

- The `learned-structure-summary.json` schema is currently flat; if
  the promoted-pattern surface needs evolution (e.g., per-pattern
  decay), bump schema and version it.
- Replacing the regex layer with embedding-based clustering is
  bigger work and shouldn't be done before 3.1–3.3 produce evidence
  that the regex layer hits a real ceiling.
- The two duplicate copies of `_STRUCTURE_CUE_WORDS` /
  `STRUCTURE_CUE_WORDS` (one each in `superpod-job.py` and
  `build-uncovered-sentence-audit.py`) should eventually live in
  one importable module. Tactically aligned for this POC; cleanup
  is a separate mechanical refactor.

## 5. Concrete unit of next work

A single PR-shaped slice:
1. add discourse-verb prefilter to
   `summarize_structure_seed_candidates` in both audit and
   superpod-job
2. add `predicted_kind` heuristic
3. emit `learned-discourse-patterns.json` with `paper_count ≥ 2`
   gate
4. add `detect_learned` in `nlab-wiring.py` that consumes it
5. wire `detect_learned` into the audit's per-paper coverage
   computation
6. measure: does sentence coverage on a fresh batch rise versus the
   POC baseline?

If 6 yields measurable lift, the loop has closed: learned
structure feeds back into detection. If not, the bottleneck is
upstream (residual extractor surfaces non-discourse prose) and the
next move is in that extractor, not in the learner.

## 6. Same-session implementation result (2026-05-20)

Steps 1–5 landed. Step 6 measurement run on real batch-008 papers
shows a partial result that calls for tuning, not a re-architecture.

### What shipped (same session)

- **Discourse-verb prefilter** in
  `summarize_structure_seed_candidates` (both
  `scripts/build-uncovered-sentence-audit.py` and
  `scripts/superpod-job.py`). Signatures without a discourse-verb
  cue are dropped before bucketing.
- **Coarse-signature clustering** (`coarse_discourse_signature`):
  signatures bucket on the discourse-verb + structural-connective
  backbone, so two papers' `<term> be introduce <cite>` and
  `<math> be introduce <term> <num>` aggregate to coarse signature
  `be introduce`. Each candidate carries `full_signatures` for the
  replay matcher.
- **`predicted_kind` heuristic**: scope > label > wire preference
  over discourse verbs in the signature.
- **Gate + regex emitter** (`build_learned_discourse_patterns`):
  `paper_count ≥ N` AND `predicted_kind ∈ {scope, label, wire}`;
  tokens translate to a loose-match regex with `.{1,120}?` gaps
  between cue anchors.
- **`detect_learned`** in `scripts/nlab-wiring.py`: consumes the
  gated pattern set; emits `learned/<predicted_kind>` records that
  count toward coverage just like any other detector hit.
- **Audit wiring**: `--learned-patterns-json` (input) and
  `--learned-patterns-out` (output) CLI flags. `learned-discourse-
  patterns.json` is emitted by every audit run.

### What the measurement showed

On a 9-paper batch-008 sample (audit-A):
- 8 structure-seed candidates after discourse-verb prefilter
- 1 candidate cleared the `paper_count ≥ 2` gate: coarse
  signature `be introduce`, predicted kind `label`, observed in
  `0801.4067v1` and `0711.1887v1`
- Gated patterns file: 1 entry

Applying that one pattern to a fresh 6-paper batch (audit-B):
- 1 learned record fired on `0711.0898v1`
- Sentence coverage on B-with-patterns vs B-baseline: **+0.0000
  across all 6 papers**

### Why coverage didn't lift on this sample

The `be introduce` regex matched a region of the test paper that
was already covered by existing scope/wire/port/label detection.
A learned-pattern record stacked on top of an existing scope
contributes no marginal coverage. The fire is real; the lift on
this small batch is zero because the existing detector is already
strong where the pattern fires.

### Open work (sections 3.4 + measurement)

- **Run on Rob-sized batches.** At 9 papers, only one signature
  clears `paper_count ≥ 2`. At 50–100 papers (the queue Rob is
  prepping), many more should survive and the marginal-coverage
  signal should be measurable.
- **Loss-of-loss stopping rule** (section 3.4) is unimplemented.
  Track `structure_loss` from `learning_loss` across promotion
  cycles; promote signatures whose addition reduces residuals.
- **Anti-clobber gating.** Optionally suppress `detect_learned`
  records that fall inside already-covered spans, to focus
  promoted patterns on widening coverage rather than annotating
  already-covered prose. This is a cheap filter at the audit
  layer.

### Tests added in this slice

- `test_signature_has_discourse_verb_filters_correctly`
- `test_predict_kind_scope_beats_label_when_both_present`
- `test_predict_kind_label_when_only_rhetorical_cue`
- `test_predict_kind_none_when_no_discourse_verb`
- `test_signature_to_regex_anchors_cue_backbone`
- `test_build_learned_discourse_patterns_gates_by_paper_count`
- `test_detect_learned_fires_on_text_via_loaded_pattern`
- `test_detect_learned_no_op_when_no_patterns`
- `test_detect_learned_skips_bad_regex_silently`

Test count after this slice: 122 passed.
