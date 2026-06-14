# E-miner-v2-bv-combs — type the wiring diagrams with BV connectives (combs over :composes)

**Excursion (bounded, single-owner). Spun out 2026-06-14. Owner: a Claude
agent (conceptually rich — CT/BV typing). Bell claude-1 back with results + shas.**

## Goal
Type the mined wiring diagrams with **BV connectives**, building **combs over
the `:composes` skeleton**. This is an *application* of the CT mining that
deliberately puts **external exotype eustress** on the mining (Joe): a harder
downstream consumer (BV-typed combs) pressures the mining to emit well-typed,
genuinely-composable structure — and wherever the `:composes` skeleton is too
thin to type, that gap is the eustress signal that feeds back to improve the
mining.

## Scope / steps
1. Locate the mined `:composes` skeleton (grep `:composes` — `futon6/holes/golden-graphs/`,
   `futon5` CT DSL / wiring diagrams). Understand its current shape.
2. Define the BV connective typing (BV = the deep-inference connectives;
   combs = parametrised/comb-shaped morphisms over the composition skeleton).
3. Attempt to type the skeleton; **record every place the skeleton is
   insufficient to type** — that residue is the load-bearing output (it's the
   eustress: a typed-consumer's demands that the mining must rise to meet).
4. Feed the gap-list back as concrete mining-improvement items (cross-ref the
   CT-mining work: dp_paper_view scopes / the :composes detector).

## Acceptance
A BV-typing pass over the `:composes` skeleton; combs built where typeable; a
**gap-list** of where the mining's output is insufficient (the eustress
deliverable) with concrete feedback to the mining. Commit artifacts + this note.

## Constraints
Never restart the futon3c JVM. Co-Authored-By: Claude Fable 5
<noreply@anthropic.com>. Bell claude-1 back with {what typed, the gap-list/
eustress signal} + shas.
