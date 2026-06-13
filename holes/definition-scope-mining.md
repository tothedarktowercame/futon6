# Definition-scope mining — grounding the definiens (step-back, 2026-06-13)

Joe, looking at the rendered Let-binder "Let $A$ be a right $H$-comodule
algebra **in** $\C$": *"I am guessing about the semantics. Does 'in' mean
element-of? Or something else? We won't know until we have structure-mined
THAT. Each definition is effectively a scope. We could start by mining
Lean's mathlib, PlanetMath, and other easy-to-read sources."*

## The insight (the one that's been circling all conversation)

A **definition is a scope** — and a binder's **definiens is a forward
reference to a definition-scope** elsewhere. We have been marking
"$H$-comodule algebra in $\C$" as a *definiens* and resolving its head to a
concept *pointer* (nnexus/nlab). But a pointer is not a meaning:

- the concept-authority says "this term is known" (nnexus:comodule);
- it does NOT say what "in $\C$" *means* — and that is exactly the
  load-bearing semantics. ("in" here is almost certainly the
  **internalization** reading — an algebra/comodule *object internal to the
  monoidal category $\C$* (cf. mathlib `Mon_ C`, monoid objects), not
  element-of — but we MUST NOT assert it. We mine the definition.)

So the recognizer registry must grow a rung: from **notation + concept
pointers** to **definitions-as-scopes**. Each concept resolves to its
structured definition (its own definiendum/definiens), recursively — a
**definition dependency graph**: "$H$-comodule algebra in $\C$" is defined
via "$H$-comodule", "algebra in $\C$", "monoidal category $\C$", … each a
scope with its own definition. (This is the same dependency structure the
dark-tower excursion already measured on mathlib's import graph — formal
definitions ARE this graph.)

## Start with the easy, explicit sources (all on disk)

Arbitrary arXiv papers assume/omit/abbreviate their definitions — the worst
place to learn what a term means. Start where definitions are EXPLICIT:

1. **mathlib4** (`/home/joe/code/mathlib4/Mathlib`) — FORMAL, unambiguous.
   "algebra in a monoidal category" is literally a structure/typeclass;
   "in $\C$" is resolved by Lean's types, no prose ambiguity. The ground
   truth for semantics. (Already our 2nd substrate, dark-tower.)
2. **PlanetMath** (`/home/joe/code/planetmath`) — clean, self-contained,
   cross-linked prose definitions; NNexus's auto-linking was built FROM it,
   so it is already a definitional graph. The concept-authority's own
   source — close the loop by mining the definitions behind the pointers.
3. **nLab** (`nlab-content` + the 20,653 indexed names) — the CT-native
   definition source we already point at.

## How it extends the running capability

- The concept-authority (130,960 terms) currently returns a POINTER. The
  definition-scope miner makes it return a STRUCTURED DEFINITION — the
  definiens of a binder resolves to a scope with its own anatomy.
- The miner is the SAME structure-first machinery (detect_scopes finds
  `bind/define`, `is-called`, the C4 prose-definienda), now pointed at
  definition sources and recursively chained.
- It answers operator questions like Joe's "what does 'in $\C$' mean" by
  retrieval against mined definitions, not by the agent guessing — the
  honest, no-self-certification path.

## Worked test (the acceptance question)

Mine the definition of "algebra object / monoid object in a monoidal
category" from mathlib (`Mon_ C`) and PlanetMath, and answer: does "$A$ a
right $H$-comodule algebra **in** $\C$" mean element-of or internal-object?
Expected: internal object (the internalization reading) — but the artifact
is the MINED definition, with provenance, not this prose.

## Disposition

Next phase of M-distributed-proofreaders (or a sibling excursion
E-definition-scopes). The step from "recognize the terms" to "know what
the terms mean." Not started; this note is the charter. First concrete
step on Joe's word: mine mathlib + PlanetMath for the algebra-in-a-
monoidal-category definition and resolve the "in $\C$" question as the
worked example.
