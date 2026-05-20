# Mission: Structure-Seed Promotion — From Replay Labels to a Live Pattern Inducer

**Date:** 2026-05-20
**Status:** IDENTIFY
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
