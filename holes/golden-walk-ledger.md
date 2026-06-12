# Golden walk — operator rules ledger

Running ledger from the live proofreading walks (Joe at the cockpit,
paper-anatomy.el + *Paper Blocks* panel). Each rule is a *type-level*
finding: a correction that applies to a detector family, not just the
span it was found on. Site lines record where the rule was minted.

**Walk protocol (Joe, 2026-06-12, voice):** the walk is voice-driven —
Joe navigates and remarks aloud; Fable reads the cursor, interprets,
and records the tags/rules here. No operator key presses; the i/o/u/x/m
keys in paper-anatomy.el remain available but are not the primary
channel.

## R1 — $...$ is a math-expression scope with subterms

Anything between dollar signs is a `mexpr` envelope and should carry
typed subterms (membership, macro-call, group, ...). Display math
(`\[...\]`, eqnarray) likewise — known detector gap (R1b: MATH_BLOCK_RE
is inline-$ only).

- Minted: 2026-06-12, golden-paragraph walk (Fukaya paragraph).

## R2 — introduction vs use: "X is a Y" in an assumption block = bind, not constrain

A prose-announced assumption ("In this section we will assume that
{\it ...}") is an `assume` envelope, and "X is a Y" sentences inside it
are *binder sites* — they introduce X with its type — not relations
between already-known things. The detector currently reads the use
without the binder, so the introduced symbols stay free (the
Skolem-audit failure mode at paper scale: downstream scopes inherit a
dangling binding edge).

Honest anatomy of the specimen sentence: one `assume` envelope ⊃ two
`bind` scopes ($\C$ := closed braided monoidal category w/ (co)equalizers;
$H$ := flat Hopf algebra in $\C$) ⊃ one genuine `constrain` (braiding of
$\C$ is $H$-linear). The shipped markup collapsed all of it into a single
`constrain/relation` on the $H\in\C$ membership — not wrong, but the
fragment that survived.

- Site: 0809.2517 point ~122 ("In this section we will assume that
  {\it $\C$ is a closed braided monoidal category ...}").
- Tags: `m` (missed binds + assume envelope), `i` (incomplete constrain).
- Detector hook: emphasis blocks following assume-verbs ("assume",
  "suppose", "we will assume that") → assume envelope; "$X$ is a Y"
  inside → bind/typing.
- Minted: 2026-06-12, live walk, voice.

## R3 — undefined standard concepts = holes needing canon links (CONFIRMED)

Concept-shaped terms used with no in-paper definition (here: "Hopf
algebra", 23 occurrences, deduped to one hole) are correctly marked as
`hole` by the golden layer's concept-shaped filter. Joe confirmed live:
"It's a hole with no in-paper definition, needs a Canon link."

This generalizes: every such hole should eventually *resolve* to a
canon anchor (nLab entry / canon-fingerprint-store id) rather than stay
a bare flag. The hole mark is the sorry; the canon link is the term
that discharges it. Resolution pipeline = the grounding-layer port
(papers ↔ concepts bipartite graph, same shape as missions ↔ patterns).

**Refinement (Joe, voice, same walk):** "a hole with no in-paper
definition isn't a hole per se. It's a *pointer* — and it becomes a
problem if we can't find a reference on the outside of that pointer."
So the type splits, two-stage:

1. **pointer** — detection-time kind: an extern reference, presumed
   resolvable. Not a defect; linking is just deferred.
2. **hole-proper** — the *verdict after a failed canon lookup*. Only
   then is it a problem (a genuine sorry).

Hole status is the failure of resolution, not the absence of in-paper
definition. Same discriminator as the sorry typing discipline:
closes-by-construction (pointer → canon link found) vs needs
world-action (no canon entry exists → real gap, possibly a missing
canon page rather than a paper defect). Detector consequence: the
golden layer should emit `pointer`, and `hole` becomes a post-resolution
state, not a detection kind.

- Site: 0809.2517 point ~139 ("flat Hopf algebra" in the §2 assumption).
- Verdict: golden mark CORRECT (true positive for the concept-shaped
  filter — the ≥2-occurrence dedupe and kind-word head both behaved).
- Minted: 2026-06-12, live walk, voice.
