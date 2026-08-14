# README-etik — the mark

**Named by Joe, 2026-07-27.** `etik` = `kite` reversed. The opposite of a kitemark.

A **kitemark** is an authority certifying conformity: *this meets the standard*. Binary,
granted from above, a claim about the object, issued by a body with the standing to issue it.

An **etik** inverts every one of those axes:

| Kitemark | Etik |
|---|---|
| certifies conformity | **maps absence** — what is *not* there |
| binary pass | **a vector by grain** |
| a claim about the object | **a claim about the instrument** |
| issued by an authority | issued by an outside observer with no standing to certify |
| says "verified" | **refuses the word "verified"** |
| absence of a flag = pass | **N/A ≠ FAIL** |

> **A kitemark certifies what is there. An etik marks what isn't.**

That sentence is the whole design. The etik is the residual-sorry map, named.

## Why the name is righter than a reversal

**Etic** (Pike, 1954, from *phonemic* / *phonetic*). The **emic** account of a proof is the
author's own narrative in the author's own terms — including *"it is easy to see"*, *"left to
the reader"*, *"standard argument"*. The **etic** account is the outside observer's structural
description in the analyst's categories.

An etik mark is the etic reading of a proof. The whole position — outsider, observational,
corpus-wide rather than curated — is the etic stance toward mathematical practice.

**This is also what makes the mark writing rather than vandalism.** A mark saying *"this paper
is deficient"* is tagging someone's wall. A mark saying *"here is what I could not follow"* is
an etic observation, jointly owned, and what it certifies is a **gap in understanding**, not a
fault in the mathematics. The comprehension gate is not bolted on to make the corpus study
publishable — it is what the word already means.

Corollary (see `holes/../TN-deep-research-landscape-position-FINDINGS-2026-07-27.md` §D3a):
the etik is the *social licence* for a corpus-wide study of unjustified steps. A mark that
cannot say "this proof is weak" can be applied to 4,616 papers by someone with no chair to
defend. A kitemark cannot.

## What an etik contains

The mark **is** the conformance vector — not a summary of it. Four positions, the grain
containment lattice already implemented in `scripts/cas_cert.py`:

```
symbol  ⊂  concept  ⊂  technique  ⊂  proof
```

Per grain, one of: **filled** · **flagged** (a named gap → ArSE seed) · **N/A** (grain never
engaged on this paper).

Hard rules, inherited from CAS-CERT and non-negotiable:

1. **No overall verdict position.** No score, no grade, no aggregate glyph. The moment an etik
   renders a single summary judgement it has become a kitemark with the sign flipped, which is
   the one thing it must not be.
2. **N/A must be visually distinct from flagged.** If those ever collapse, the mark lies —
   "criterion never exercised" and "criterion violated" are different facts, and conflating them
   is exactly the failure the certificate exists to avoid.
3. **The gate FAILs only on mis-wire.** A low comprehension reading means *weak-EXTRACTION —
   study more, richer corpus*; **never** *weak-proof* (`scripts/clean_comprehension.py`).
4. **The word "verified" never appears.** The claim is *"well-formed wiring — every port filled
   or flagged."*

## Recognisability (the TAKI constraint)

The mark's value is cumulative, not per-instance: one etik seen anywhere should imply the whole
corpus. That imposes constraints the code does not currently enforce:

- **One stable glyph**, byte-identical rendering across every paper. No per-paper drift, no
  per-run restyling. Format changes are versioned events, not edits.
- Legible at a glance at small size — it will appear inline in listings, not only on a
  per-paper page.
- The four grain positions must be readable in fixed order, left to right, always present even
  when N/A.

**Trademark note:** the Kitemark is a live BSI registered trademark. The inversion here is
conceptual only. The etik glyph must not resemble BSI's mark — no visual quotation, no parody
form. Design it from the grain vector outward, not from the kitemark inward.

## Where it already exists in code

The object is built; the *name* and the *stable rendering* are what is new.

| Piece | File | State |
|---|---|---|
| Conformance vector by grain, N/A ≠ FAIL, gate FAILs only on mis-wire | `scripts/cas_cert.py` | built + reviewed |
| Rung ladder feeding it (0/1/R2a/R2b/R2c/R2d) | `scripts/iatc_semcheck.bb` | built + reviewed |
| Technique grain + phrased gap questions | `scripts/rung3_technique.py`, `scripts/rung3_residue_llm.py` | built; costed LLM pass pending |
| Comprehension gate | `scripts/clean_comprehension.py` | built |
| Corpus-scale hole harvest (the collection of etiks) | `scripts/clean_hole_harvest.py` | partial |
| Human rendering | `scripts/build_proofcheck_demo.py` → `proofcheck-demo/index.html` | built, 4 papers |

**Terminology to adopt:** one paper gets **an etik**. The corpus-wide collection is **the etik
map**. `holes/math-ct-full.ids.txt` (4,616 primary math.CT papers) is the first map's extent.

## Open

- **Glyph design.** Not started. Four positions × three states; must survive small-size inline
  rendering and must never suggest an aggregate.
- **Versioning.** An etik computed under extraction v1.3 is not comparable to one under a later
  ladder. The mark should carry its generator version, and the map should refuse to mix
  versions silently.
- **The hedging class is unmined.** *"it is easy to see" / "left to the reader" / "standard
  argument"* — confirmed by the 2026-07-27 probe to be unmined anywhere in the literature, and
  the most natural etik trigger we do not yet detect. Candidate next detector.
- **Does an etik ever get retracted?** If a later run understands a paper better, the previous
  etik was a true statement about the earlier instrument. Supersede, don't delete — the same
  discipline the learning loop applies to memories.
