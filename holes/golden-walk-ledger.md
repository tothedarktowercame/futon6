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
