# CAS-SEL Tier-1, measured on arXiv math.CT

**Question it answers.** The deterministic selection path yields all-thin on
arXiv, so the card's own expectation was that real matches need the Tier-1 LLM
verify. The obvious-looking move was to retract CAS-SEL because it "does
nothing". The question put to it was the right one: *if we leave it out just
because it does nothing, what would we be missing?*

Now measured over **all 98 proof graphs** (16 papers), run to completion:

| | |
|---|---:|
| proofs with at least one verified match | **95 / 98** |
| proofs with none | 3 |
| total verified matches | **478** |
| distinct patterns matched | **26** |
| per-step failures (timeout etc.) | 12 |

So the answer is: we would be missing a working per-proof strategy recognizer.
It fires on 97% of proofs, and the deterministic path finds none of this.

## The sharper finding: the menu is for the wrong mathematics

The point of CAS-SEL is that each proof gets a check menu matched to *its*
sorry-topology instead of a uniform `{R2a,R2b,R2c}`. Selection works. What the
selected patterns then find is a `CHECK_MENU` built for a different field:

| what happens to a verified match | count | share |
|---|---:|---:|
| fires a real check | 157 | 32% |
| pattern is a menu key, but its list is empty | 89 | 18% |
| **pattern is absent from the menu entirely** | **232** | **48%** |

Those shares are stable: at 72 proofs they read 32% / 18% / 51%, at 98 they read
32% / 18% / 48%. The finding is a property of the corpus, not of how far the run
had got.

And from the other side — the menu keys that **never matched once** in 478
verified matches:

    estimate-by-bounding · induction-and-well-ordering
    separate-into-independent-pieces

Those are analysis patterns. The patterns the corpus actually produces, and that
the menu has never heard of, are categorical:

    transport-across-isomorphism (64) · find-the-right-abstraction (50)
    verify-universal-property (43) · structural-equivalence (20)
    dualise-the-problem (17) · encode-as-algebra · structural-inclusion

**The menu was specified from an analysis-shaped pattern population; math.CT
produces a categorical one.** That is not a bug in the selector — the selector
is what made the mismatch visible, and nothing else in the pipeline could have.

## Why this could not have been decided from the implementation diagram

Retracting CAS-SEL on the grounds that it produced nothing would have removed
the only instrument capable of reporting that the check menu does not fit the
corpus. The 232 absent-pattern matches are not a failure of the feature; they
are its most informative output, and they are legible only because it ran.

The same shape as LEAN-NL's 562 misses: the misses are the sample that says
where the vocabulary is thin.

## What follows

1. **Extend `CHECK_MENU` to the categorical patterns**, in frequency order —
   `transport-across-isomorphism`, `find-the-right-abstraction`,
   `verify-universal-property` alone account for 157 of 478 matches (33%).
2. **Fill the empty lists** (`unfold-the-definition` at 67 matches is the
   largest single empty entry — 18% of all matches select a key that fires
   nothing).
3. **Decide the three never-matched analysis keys**: keep as coverage for other
   fields, or mark them out-of-scope for math.CT. They cost nothing but they
   make the menu look better populated than it is for this corpus.

Until (1) and (2), CAS-SEL's readiness should read *selection witnessed, menu
under-specified for this corpus* — which is a different and much more useful
status than either "ready" or "does nothing".

## Provenance

- `scripts/cas_select.py --steps-dir data/cas-select-steps/run --backend openai
  --model glm-4.5-air --checkpoint /tmp/cassel-tier1.jsonl`, GLM-4.5-Air Q4 on
  Zone, `FUTON6_LLM_MAX_TOKENS=256`.
- Cost: ~7 min per proof serialized on a one-slot endpoint; 98 proofs in ~13h
  wall clock across restarts. Exit status `success`, 98 distinct proof ids. An uncapped
  `max_tokens` had made this ~15 min *per call* and put the pass on a ~9-day
  trajectory; see H24/H26 in `excursions/E-superpod-hardening.md`.

---

# Appendix: the deterministic recognizer, calibrated against Tier-1

W8's calibration half. Both instruments run over the **same 818 proof steps** of
the same 16 papers, so the comparison needs no adjustment:

| instrument | recognized | share | classes |
|---|---:|---:|---:|
| deterministic tactic-gesture recognizer (`strategy_recognizer.py`) | 76 | **9.3%** | 5 |
| LLM Tier-1 pattern verify (`cas_select.py --backend openai`) | 478 | **58.4%** | 26 |

A factor of six in coverage and of five in vocabulary. The deterministic side
recognizes only `obtain` (54), `suffices` (11), `intro` (5), `apply` (4),
`wlog` (1) — Lean-style tactic gestures, which is what its vocabulary contains.
742 of 818 moves are `ungrounded` to it.

Two things follow.

**This is the quantified case for the LLM residue pass.** rung-3 exists to handle
what the deterministic rungs cannot reach, and "cannot reach" now has a number
against a shared denominator rather than an impression. It also sets the bar any
cheaper method has to clear.

**The two instruments disagree about what a proof move even is.** The recognizer
looks for tactic gestures; Tier-1 looks for reasoning patterns. Neither
vocabulary is wrong, but 9.3% is not a *recall* figure for the recognizer — it
is the rate at which CT prose happens to contain Lean-shaped tactic words. Read
as recall it would badly understate a tool doing something else, and that
misreading is easy to make from the summary line alone, which prints
`recognized (grounded+thin) = 76/818 = 9.3%` with no mention of vocabulary.

The honest calibration statement is therefore: *on math.CT prose, tactic-gesture
recognition covers 9.3% of proof steps, and pattern-level LLM verification
covers 58.4%; the residue between them is what rung-3 is for.*

---

# Appendix B: the rung-3 residue pass (W8)

92 residue gaps over 16 papers, GLM-4.5-Air on Zone, ~28 min.

| | |
|---|---:|
| gaps asked | 92 / 92 |
| model-written questions | **92** |
| template fallback · endpoint error | **0 · 0** |
| classified `novel-technique` · `real-gap` | 5 · 87 |
| distinct question strings | **92 / 92** |
| median similarity to the deterministic template | **0.23** (none above 0.90) |

Every question is the model's own and distinct. The five `novel-technique`
classifications are legible as such — e.g. *"How does the pivotal structure on a
rigid monoidal category C induce a monoidal natural isomorphism τ between the
identity functor and the double dual?"* — and the `real-gap` questions name the
objects of the step they interrogate rather than restating its pattern.

**It took four passes to get here, and the first one reported clean.** The
progression is the point:

| pass | reported | actually |
|---|---|---|
| 1 | 88/92 model-written | every question was the deterministic template verbatim (median similarity **1.00**) |
| 2 | 78/92 model-written | template clause removed; but the model now ran past `max_tokens` and truncated mid-word |
| 3 | 78/92, 14 unparseable | `maxLength` fixed truncation; the 14 were mathematical questions containing LaTeX braces, which the JSON extractor could not parse |
| 4 | 92/92, 0 degraded | brace-balanced extraction |

Each pass fixed a real defect and exposed the next, which had been masked. Two
of the four were in instruments I had added earlier in the same session to catch
the previous fault. The provenance field (`source: model|template|error`) is
what made passes 1–3 legible at all; without it, pass 1 would have shipped as a
result.

**The lesson worth keeping is about what provenance measures.** `source: model`
records *authorship*, not *originality*: a pass can be 100% model-written and 0%
informative, and pass 1 was exactly that. The check that caught it was
similarity-to-stub — comparing the output against what the deterministic
fallback would have produced. Any LLM stage should carry both.
