# CAS-SEL-3b — lift the Tier-0 retrieval recall ceiling (follow-on)

*Authored by claude-1 from the CAS-SEL-3 review (2026-06-17). NOT yet dispatched.
Owner/reviewer: claude-1.*

## Why

CAS-SEL-3's Tier-0 candidate retrieval is **classical hotword overlap** over
`patterns-index.tsv`. Measured honestly (review of CAS-SEL-3), its recall has a ceiling:
**recall@4 = 16/22 (73%)**, and **even at full-pool k it tops out at 19/22 (86%)** — three
fixture steps have *zero* lexical overlap with their correct pattern (`quotient-by-irrelevance`
on "z = z₀+mω₁+nω₂"; `construct-auxiliary-object` on "tent function" / "upper central series").
Tier-1 can only adjudicate what Tier-0 surfaces, so a retrieval miss is a **false induce-trigger**
in a real run. The recall ceiling is the ceiling on the whole select path.

This is the **combining-methods-as-diagnostic** situation: when the classical (hotword) method
hits a ceiling, the gap is the signal to add a *second* retrieval modality — not to hand-tune
more hotwords (which is fixture memorisation, removed in review).

## What to build

A non-lexical retrieval modality for `retrieve()`, combined with the hotword scorer:
- **Option A (preferred, cheap): embedding similarity.** Embed each step's text and each
  pattern's title+conclusion (`THEN`)+keywords; cosine top-k. Use the same embedding the stack
  already uses for retrieval — **BGE, not R-GCN** (per the superpod-embedding finding), with hard
  negatives if tuning. Union the hotword top-k and embedding top-k as the candidate set.
- **Option B: LLM-side retrieval.** Let Tier-1 see a larger slate (or the full 39 one-liners) and
  pick — folds retrieval into verify. More LLM cost; measure against A.

## Acceptance
- recall@k (oracle-in-candidates) on the 4-proof corpus **> the hotword-only 19/22 ceiling**;
  ideally 22/22, with the 3 zero-overlap steps recovered by the semantic modality.
- Deterministic given a fixed embedding (cache the vectors); Tier-0 stays model-free *of the LLM*
  (an embedding model is not the generative LLM — note the distinction in the cost tiers).
- Re-run CAS-SEL-3's honest-recall test; update the pinned number upward with the new modality.

## Gates
PY (py_compile + pytest, extend `test_cas_select.py`); report the new recall number vs the 19/22
hotword ceiling. Bell claude-1 back + append findings.
