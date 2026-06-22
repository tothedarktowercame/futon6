# E-comprehension-foundation — corpus-relative comprehension as the foundation

*claude-1 + Joe, 2026-06-22. The holistic architecture under the readiness
"cards": what the pipeline produces, why "improves as we run" is true (and where),
and the discipline that keeps the foundation solid so Phase 3 can assume it.*

## The principle (peng)

All of tai chi assumes peng and the other energies compose on it. Here the peng is
the **normalized vocabularies** — the noun (concept) vocabulary and the hole/
strategy vocabulary. Every Phase-3 technique (weak-proof detection, conjecture
finding, fill-by-retrieval) is a *composition* that assumes them. If the
foundation is fractured, every composition silently inherits the defect — and
still shows green on its own card. **Normalization is foundation, maintained
as-we-go — never a Phase-3 repair of a Phase-2 deficiency.**

## Two vocabularies, symmetric (the asymmetry to fix)

Today: noun concepts are discovered in Phase 1 and *used* in Phase 2 (grounding).
Hole/strategy concepts are discovered late (Phase 2, if at all) and *used for
nothing*. Fix: make the hole vocabulary **first-class**, mined and used exactly
like the noun vocabulary.

| layer | vocabulary | recurs? | role |
|---|---|---|---|
| noun | concepts (term-prior, encyclopedia, concept-index) | yes (saturating) | what the proof is *about* |
| hole/strategy | method tags + satiety + discharge-kind (+ conjecture/open-problem) | yes (controlled, 12 tags) | *how* the proof moves |
| detail | raw `:wanted` free-text | **no** (paper-local) | the specific gap; hygiene only |

Empirical (2026-06-22, `warrant_normalize.py` over 36 finals): raw `:wanted` slugs
are bespoke — 46 slugs, only within-paper paraphrase merges, **zero cross-paper
recurrence**. So the recurring/foundational key is **(hole-type, grounded-concept)**,
not the raw label. `segal-condition-implies-pullback-square-condition` →
`(reduce-to-known, weak-factorization-system)`, which *can* recur because the
concept is normalized. The 部首 reading: a hole is recognized by decomposing it
into known components (type + concept), the way 棚/稝/掤 are related through 朋.

## Comprehension as self-knowledge (the key move)

A proof's representation has a **comprehension score relative to the corpus**:

```
Comprehension(proof | corpus) =  R2d  (fraction of NOUNS grounded)
                              ⊕  rung-3 (fraction of STRATEGIES/moves grounded)
```

Both already exist and are already built as *"flagged gap → question, not a
verdict"*: R2d (noun-side, buckets defined/known/imported/undefined) and rung-3
(its verb-twin, buckets grounded-by-pattern/by-citation/thin/ungrounded). The
missing work is to **compose them into one per-proof number, make it
corpus-relative, and use it to gate the verdict.**

**The gate (the whole point):**
- **Low comprehension** (few concepts grounded, few strategies recognized) ⇒
  *weak EXTRACTION* — "I didn't understand this proof; I need a richer corpus / to
  study more / to ask for help." **Never** "the proof is flawed." (You don't call
  a proof wrong when you understood 1/10 of it.)
- **High comprehension + genuine unfilled holes** ⇒ *weak PROOF* — real
  mathematics left open. Only assertable *after* comprehension is established.

This dissolves the weak-proof / weak-extraction conflation with one composed
metric, and it is the disambiguation the bare hole-count could never make.

### Empirical: strategy is the binding axis (2026-06-22, `clean_comprehension.py`)

First run of the composed floor over `loop-run-70b` (9 proofs): the two axes are
**wildly asymmetric**. Noun comprehension is solid (mostly **1.00** — the concept
substrate works), but strategy comprehension is **near-zero** (rung-3 grounds
~0–0.30 of moves). Since `comp = min(N, S)`, the strategy axis gates everything →
8/9 `weak-extraction`, 1 `partial-comprehension`, **0 weak-proof** (the gate
correctly refuses to judge proofs we don't yet understand).

Conclusion: **the foundation is half-solid.** Nouns are grounded; strategies are
not — we know *what* these proofs are about but not *how* they argue. The lever
that raises comprehension is therefore the **strategy pattern pool** (CAS-SEL
retrieval / `.flexiarg` patterns / wiring rung-3's verifier for arXiv), **not**
more concept mining. Known contributor: rung-3's Tier-0 deterministic retrieval
"yields all-thin on arXiv" (the Tier-1 LLM verify is unwired), so some of the low
strategy score is detector gap, not true incomprehension — itself a foundation
item. The go/no-go test before Linode: does `comp` rise as the pattern pool grows?

**Wiring update (same day):** `clean_comprehension.py` now computes rung-3 **live**
from `cas_select` (so the floor reflects the current pattern pool — re-runs lift as
it grows) and gives **resolution-graded credit** (a verified pattern = full; a
`thin` recognized-but-unverified match = partial, the hologram's low-res layer;
conjectures excluded). Re-measure over `loop-run-70b`: verdicts shift from
`{weak-extraction:8, partial:1}` (binary) to `{weak-extraction:2,
partial-comprehension:7}` (graded) — **the floor responds to a strategy-layer
improvement, confirming corpus-relativity.** Still 0 `weak-proof` (nothing reaches
`comp>=0.8`) — correct: we recognize shapes, not verified detail. Corpus move
totals quantify the two CPU levers: **grounded=16, thin=50, ungrounded=9,
conjecture=3** → the dominant lever is **VERIFY** (convert the 50 thin moves to
grounded; only 10/39 patterns are currently "verifiable"), not retrieval (9
ungrounded). Verifying thin moves is where Tier-1 (LLM) or an expanded verifiable
registry pays off most.

## Why "improves as we run" is true — and where

- **Extraction** (raw IATC graph / CLean skeleton): one-shot per paper.
- **Comprehension** (grounding that structure against the corpus): **improves as
  the corpus grows** — more concepts and strategies become recognizable, so the
  *same* proof re-grounds to a better representation. This is re-grounding, not
  re-extraction; cheap CPU; foundation maintenance.
- **Phase-3 fill-by-retrieval**: also corpus-dependent — more papers ⇒ more
  candidate fills ⇒ higher fill-rate.

So the iterative gains are real but live in **re-grounding + fill**, not in
re-extraction. (Earlier framing said "the reasoning layer doesn't improve" —
corrected: its *comprehension* does.)

## The objective (statable, with a ceiling)

> "The best possible structural representation of this proof **given this corpus**"
> = max comprehension against the current corpus; the un-grounded remainder is the
> explicit **gap-to-ceiling**, shrinking monotonically as the corpus grows.

Not "we tried it once and it didn't work." Every proof carries its comprehension
score and its gap-to-ceiling, so the run *knows what it doesn't yet understand*.

## Build sequence (all CPU, no GPU / no usage limit)

1. **Ground holes to concepts** — extend R2d from "concepts a proof uses" to
   "the concept a hole wants"; key holes by (type, concept). *Foundation.*
2. **Compose the comprehension score** — R2d ⊕ rung-3 per proof, corpus-relative.
3. **Wire the verdict gate** — weakness assertions require high comprehension;
   else emit "study-more / ask-for-help" (the ArSE question path).
4. **Raw `:wanted` normalization** — light hygiene (merge within-paper paraphrases).
5. *Then* Phase 3 composes on this: weak-proof map, conjecture (open-problem)
   harvest, fill-by-retrieval — each assuming a solid (type, concept) foundation.

The **Linode stepper** runs Pass 1–2 and the re-grounding loop; Phase 3 mines the
result. The go/no-go before scaling is: does the comprehension score behave —
rise as the corpus grows, and separate weak-extraction from weak-proof on a pilot?

*Cross-refs:* `E-clean.md`, `proofcheck-readiness.html` (R2d, rung-3),
`clean-method-vocab.edn`, `scripts/{warrant_normalize,clean_hole_harvest}.py`.
