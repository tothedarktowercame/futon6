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
