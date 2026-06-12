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

## R4 — bound-symbol occurrences: mexpr mark + use→binder edge (BUGBEAR)

Specimen: the `$\C$` in "the braiding of $\C$ is $H$-linear" carries
NO mark at all — overlays-at returns nil. Two stacked violations:

1. **R1 violation at detection level:** short single-symbol spans
   (`$\C$`, `$H$`) get no mexpr envelope. Every $-span is a
   math-expression scope — no length exemption.
2. **Missing use→binder edge:** `\C` is *defined in this very
   paragraph* ("$\C$ is a closed braided monoidal category ...", the
   R2 bind site ~30 chars earlier). A symbol occurrence whose binder
   exists in-paper should link to it. "According to the structure of
   the paper... it says right what it is in this paragraph" (Joe,
   voice). This is the golden-30 expectation stated at mint time —
   "I'd expect *all symbols* to have definitions" — now witnessed
   failing on the simplest possible case.

The symbol table is the missing artifact: per paper, every bound
symbol ($\C$, $H$, $\Upsilon$, ...) with its binder site, and every
occurrence carrying an edge back. Same shape as R3's pointer/canon
split, but *intra-paper*: occurrence = pointer, binder = the in-paper
canon. A symbol occurrence with NO in-paper binder and NO canon link
is then the real orphan.

- Site: 0809.2517 point ~173.
- Tags: `m` (missed mexpr on $\C$ and $H$), type-level.
- Detector hook: tokenize ALL $...$ spans regardless of length; build
  per-paper symbol table from R2 bind sites; emit use→binder edges.
- Minted: 2026-06-12, live walk, voice ("a serious bugbear").

## R5 — author-defined macros are a special definition type (macro-def)

`\C` is not primitive notation: the eprint defines it —
`\newcommand{\C}{{\mathcal C}}` (0809.2517, beseq2.tex:65). "This type
of author-defined macro should become a special type of definition
because we want to know what's going on with author-defined symbols"
(Joe, voice).

So a symbol like \C carries a TWO-TIER definition chain, and the
anatomy should show both tiers:

1. **macro-def** (lexical binder) — the preamble \newcommand /\def /
   \DeclareMathOperator: pure notation, maps surface form to TeX
   right-hand side. Lives outside the running prose, often in a .sty
   or a preamble-only .tex file.
2. **semantic bind** (R2) — the prose site that gives the symbol its
   mathematical content ("$\C$ is a closed braided monoidal
   category ...").

The R4 symbol table gains a notation column: symbol → macro-def site
+ RHS → semantic binder → uses. Author-defined symbols are exactly the
ones that can NEVER resolve via canon (no nLab page for \C) — their
entire meaning is intra-paper, which is why the chain must be complete:
an author-defined macro with a macro-def but no semantic bind is a
genuine orphan, the inverse failure of R3's pointer.

- Site: 0809.2517 (\C used at ~point 173; defined beseq2.tex:65).
- Detector hook: parse \newcommand/\renewcommand/\def/
  \DeclareMathOperator across ALL files in the eprint (including .sty)
  into macro-def scopes; join to the symbol table by surface form.
- Note: the fresh extractor flattens multi-file eprints — macro-defs
  in non-main files (here beseq2.tex feeding Galois2.tex) must still
  be swept.
- Minted: 2026-06-12, live walk, voice.

## R6 — large displays are scopes; `:=` in display = definition; diagram DSLs have subterms

Joe marked a whole display region live (0809.2517, chars 645–1239):
a `$$...$$` block of GrCalc graphical-calculus diagrams defining the
smash-product multiplication, `m_{A\# H} := <string diagram>`, plus
the unit. Marks at point: nil — nothing on the entire display.

Three findings stack:

1. **R1b witnessed live, $$-variant:** display math (`$$...$$` as well
   as `\[...\]`/eqnarray) gets no envelope. Every display block is a
   mexpr scope — the LARGE ones doubly so, since they carry the
   paper's actual constructions.
2. **`:=` inside display = definition site.** This display BINDS the
   multiplication and unit of $A\# H$. Definitions don't only live in
   prose ("X is a Y", R2) or \newcommand (R5) — they live in displayed
   equations, and `:=` (also `\equiv`, "defined by", \stackrel{def}{=})
   is the marker. The R4 symbol table must accept display-math binder
   sites.
3. **Diagram DSLs are structured subterm territory.** The GrCalc
   macros (\gbeg/\got/\gmu/\gcmu/\gbr/\glm/\gob...) are a typesetting
   DSL for string diagrams — cells, wires, multiplications, braidings.
   Subterm typing here = parsing the diagram grammar, not generic
   macro-call spans. (Affinity note: these ARE wiring diagrams — the
   paper's own idiom is the futon5 idiom; a GrCalc parser would land
   diagram structure we already know how to consume.)

- Site: 0809.2517 region 645–1239 ($$ display after "becomes an
  algebra in \C with multiplication and unit").
- Tags: `m` (missed display envelope, missed := bind), type-level.
- Detector hook: MATH_BLOCK_RE extended to $$...$$/\[...\]/eqnarray
  (R1b fix); := and friends inside display → bind scope on the LHS;
  GrCalc grammar as a subterm parser (stretch).
- Minted: 2026-06-12, live walk, voice ("now I'm going to mark a
  large display").

**R6 amendment (Joe, voice):** the display is "absolutely full of
user-defined symbols — \gbeg, \got, \gu, on and on. Those are not
standard LaTeX markup... all of those have some type of lexical
meaning, and by and large we could get at least some clues about that
by referring to the LaTeX preamble. But as it is, this whole thing is
just marked up as one big display scope with no internal structure at
all."

Verified: GrCalc3.sty defines 58 macros via \newcommand (\gbeg \gend
\gnl \gcl \got \gob \gmu \gcmu \glm \grm \gwmu \gbr ...). So R5's
macro-def sweep and R6's subterm parsing are ONE mechanism, not two:
the preamble/styles give every non-standard control sequence a
macro-def entry, and any occurrence of an author-defined macro inside
a display becomes a typed subterm *by lookup* — no diagram-grammar
intelligence needed for the first cut. "Not in the standard LaTeX/
AMS vocabulary" is the detector test; the macro-def table is the
decoder ring. A full GrCalc grammar parse stays the stretch goal;
macro-table-driven subterm marks are the immediately buildable rung.
