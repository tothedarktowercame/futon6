# E-apm-cross-paper-dependency-graph — which APM papers can reuse each other's Lean

**Type:** excursion, linked to M-informal-lean-alignment. **Status:** PLANNED (not started).
**Raised:** 2026-10-25 by Joe, to come back to separately and with more focus.
**Feeds:** pudding-prover T3.3 (ArXiv math.CT mining). This excursion is one rung of an implicit curriculum toward T3.3, not a one-shot discharge of it.

## Question

The mark8 run over the APM informal solutions (`apm-501-xi-r1`) and the step–lemma alignment from M-informal-lean-alignment (`futon6/scripts/align_informal_lean.py`) together give, for most papers, the informal proof steps and the Lean lemmas that prove them. Lean-aligned proofs now exist for almost all of the corpus.

Can these be joined into a cross-paper dependency graph? The graph would answer two things:
- how connected the corpus is;
- for an open sorry (e.g. t98A01, Euler-characteristic multiplicativity for finite covers), which other papers' lemmas could be reused.

## Why it matters for T3.3

It is a concrete answer to "what is mark8 useful for?", at a scale already tested. It could be shown to Rob as the case for running mark8 over the math.CT section of the arXiv as a proof of concept.

## Not yet decided

- Edge rule: an aligned step proved by another paper's lemma, or a match above a threshold.
- What counts as a useful demo: a number, a rendered example, or both.
