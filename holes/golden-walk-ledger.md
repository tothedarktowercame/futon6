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

## R7 — partial subterm markup is worse than none: parse the whole expression or flag it

Specimen: `$A\ot\Delta_H$` (0809.2517, ~point 1353 — "right
$H$-comodule given by $A\ot\Delta_H$"). The only subterm marked is the
`_H` subscript. Joe, voice: "I don't know what A is. I don't know if
\ot is an operator or some other type of symbol. And I don't know how
\Delta is defined in this paper."

A lone subscript mark inside an otherwise-unparsed expression ASSERTS
structure that isn't there — it reads as "this expression is
understood" when three of its four tokens are dark. Rule: subterm
markup within a mexpr is all-or-flagged — either every token resolves
through the chain, or the envelope carries an explicit
`parse-incomplete` mark so partial coverage can't masquerade as
coverage.

What the full chain yields here, each token answering one of Joe's
three questions (all verified in the eprint):
- `A` → semantic bind needed (the H-Azumaya algebra of the running
  construction) — R2/R4 territory.
- `\ot` → macro-def: `\newcommand{\ot}{\otimes}` (beseq2.tex:86) →
  canon pointer: tensor product. R5 lookup, trivially decodable.
- `\Delta_H` → comultiplication of H. In-paper evidence exists but is
  DIFFUSE: Sweedler-notation convention stated in applic3.tex for
  $\Delta_B$ generally; no single crisp binder for $\Delta_H$. This is
  the interesting case: a *convention-bound* symbol — bound by a
  notational convention paragraph, not a definition. The symbol table
  needs a third binder kind: macro-def / semantic-bind /
  **convention-bind** (Sweedler notation, "we write", "we denote by").
- `_H` → the one thing already marked, useless without its parent.

- Tags: `i` (incomplete — expression-level parse missing), type-level.
- Detector hook: mexpr subterm pass must tokenize the WHOLE $-span
  (R1) and join every control sequence through macro-table (R5) +
  symbol table (R4) + convention paragraphs (new); emit
  parse-incomplete on the envelope when any token fails to resolve.
- Minted: 2026-06-12, live walk, voice.

## R8 — symbols in section headings = forward declarations (promissory-bind)

Specimen: "\subsection{The map $\Upsilon$ assigning an $H$-Galois
object to an $H$-Azumaya algebra}" (0809.2517, ~point 309). Marks:
nil. Joe, voice: "a symbol that may not be defined yet... maybe
they're planning to define it."

The heading introduces $\Upsilon$ BEFORE any definition — but it is
not an orphan use: it is a *forward declaration*, and the heading
even carries the TYPE: "the map ... assigning an H-Galois object to
an H-Azumaya algebra" is literally a type signature,
Υ : H-Azumaya → H-Galois, stated in prose before the term is
constructed. That is exactly the sorry-arrow shape — type without
term, with the section body as the scope that promises to discharge
it.

Fourth binder kind for the symbol table:
- macro-def (R5) / semantic-bind (R2) / convention-bind (R7) /
  **promissory-bind** (R8): bound by sectioning structure; carries a
  prose type signature; DISCHARGED when a real definition site for
  the symbol appears within the section extent; an undischarged
  promissory-bind at section end is a genuine defect (promised, never
  delivered).

Bonus: this gives paper anatomy its proof-hole semantics for free —
a section = a sorry being discharged in real time; heading = the
:want, body = the construction. Papers and missions get the same
shape (HEAD as satisfaction conditions).

- Site: 0809.2517 ~point 309 (subsection heading).
- Tags: `m` (heading scope + promissory-bind unmarked), type-level.
- Detector hook: \section/\subsection titles parsed for $-spans →
  promissory-bind entries typed by the surrounding prose; check
  discharge within section extent.
- Minted: 2026-06-12, live walk, voice.

**R8 refinement (Joe, voice):** the heading introduction "is a scope
opening — a scope called: I'm introducing a new symbol which hasn't
been defined yet, but keep reading until you get the definition. A
symbol-binding scope that's going to be paid off later in the text."

So promissory-bind is not an annotation on the heading — it is a
SCOPE WITH EXTENT: it opens at the introduction site and closes at
the payoff (the discharging definition), not necessarily at the
section boundary. The interval between is the *debt region*: uses of
the symbol inside it are uses-under-promise — valid, but
forward-dependent, and the anatomy should render them as such (the
reader is carrying an obligation the author has taken out). Extent
semantics, in scope terms: binder at the open, body = debt region,
discharge = the close. An EOF before the close = the
promised-never-delivered defect, now detectable as an unclosed scope
rather than a missing row.

## R9 — `$X$-term` is a parametrized concept: one compound, two edges

Specimen (0809.2517, the Υ subsection heading and throughout):
"$H$-Galois object", "$H$-Azumaya algebra"; earlier "$H$-linear",
"$H$-comodule algebra", "$A\x A$-bimodule". Joe, voice: "this kind of
construction — dollar-sign symbol, dash, lexical term — shows up a
lot in mathematics... we're able to recognize the structural things
like Galois object or Azumaya algebra, but we're not connecting
them."

The pattern `$X$-<concept>` is a PARAMETRIZED CONCEPT, and the
current anatomy splits it down the middle: the concept detector sees
the lexical term, the math detector (when it fires at all, R4) sees
the $X$, and nothing records that they form one compound. The honest
parse is a single compound node with two outgoing edges:

1. **parameter edge** — $X$ → its binder via the R4 symbol table
   (here: H → the §2 semantic bind, "a flat Hopf algebra").
2. **concept edge** — the lexical head → canon pointer (R3): "Galois
   object", "Azumaya algebra" resolve to nLab; the H-parametrized
   FORM may itself have a canon page (nLab: Hopf–Galois object) —
   prefer the parametrized entry when it exists, fall back to the
   bare concept + parameter edge when it doesn't.

This is the bridge rule: R3 (concept pointers) and R4 (symbol table)
have so far been separate lanes; `$X$-term` compounds are where they
MUST join. It is also intensely productive: parametrized concepts are
how the paper builds its working vocabulary (every structure in this
paper is an H-something), so missing the connection means the
concept-lane reads the paper as generic category theory while the
symbol-lane reads it as untyped algebra.

- Tags: `i` (concept marks incomplete — parameter edge missing),
  type-level.
- Detector hook: regex family `\$([^$]+)\$-(\w[\w ]*)` joined at
  build time: parameter → symbol table, head → concept/canon lane;
  emit one compound scope spanning the whole construction.
- Minted: 2026-06-12, live walk, voice (utterance truncated
  mid-clause; the connection-failure point was complete).

**R9 naming + NER/concept split (Joe, voice):** the `$X$-` prefix is
a **mathematical decorator**. "A mathematical decorator is actually
an important part of the concept. It doesn't necessarily have to be
part of the named entity — but it needs to be part of the concept,
the mathematical concept."

So the two layers come apart cleanly:
- **named entity** (NER lane): the lexical head — "Galois object",
  "Azumaya algebra". The term-spotter / NER kernel keeps matching on
  these; decorators stay OUT of the entity dictionary (else the
  dictionary fragments into every parametrization an author invents).
- **concept** (graph lane): decorator INCLUDED — the node is
  H-Azumaya-algebra-with-H-bound-to-this-paper's-H, i.e. the compound
  of R9 with its parameter edge. Concepts are
  decorator-qualified entities, and the papers↔concepts bipartite
  graph must be built over THESE, not over bare entities — otherwise
  two papers "share a concept" when one decorates over a Hopf algebra
  and the other over a group, which is exactly the false-merge the
  embedding continuum already suffers.

Decorators also stack and iterate ($A\x A$-bimodule; right
$H$-comodule algebra — side-decorators like "right"/"left" are
decorators too, prose rather than $-form). Decorator grammar =
(side|$X$|$X,Y$)-entity, recursive.

## R10 — with-clauses are capability decorators; assumptions want a dependency map

Specimen: "$\C$ is a closed braided monoidal category with equalizers
and coequalizers" (0809.2517, the §2 assumption).

1. **Adjective stacks are decorator towers** — closed ∘ braided ∘
   monoidal (category): prefix prose decorators, each independently
   canon-resolvable, extending R9's grammar
   (adj* (side|$X$)-entity (with-clause)?).
2. **"with X and Y" is a CAPABILITY decorator** — not what the object
   is but structure it is required to HAVE, i.e. resources the
   section's proofs will consume (equalizers/coequalizers get used in
   specific constructions, not everywhere).
3. **The author announces a dependency map and leaves it implicit:**
   "Nevertheless, for some of the results not all of the assumptions
   will be necessary." Each assumption (closedness, braiding,
   (co)equalizers, flatness, H-linearity of braiding) is a separate
   obligation with its own set of consuming results. The anatomy
   should carry assumption → consuming-result edges; the author's own
   sentence is the license to build them. This is the with-clause's
   payoff: capability decorators are exactly the assumption units the
   dependency map is over.

- Tags: type-level (decorator grammar extension + new edge family).
- Detector hook: with-clause parser on bind sites; assumption units
  individuated per conjunct; (stretch) result-side consumption mining
  ("since \C has equalizers...", "by closedness...").
- Minted: 2026-06-12, live walk, voice.

**R10 sharpening (Joe, voice):** "Are we building up an object type
called closed-braided-monoidal-category-with-equalizers-and-
coequalizers? Or are the equalizers and coequalizers separately
supplied from that object? Wherever they are, these terms aren't
marked up — we have no concept of how they fit or relate structurally
to the closed braided monoidal category whatsoever."

Two layers:
1. **Baseline failure first:** "equalizers" and "coequalizers" are
   concept-shaped terms with perfectly good canon entries and they got
   ZERO marks — they didn't even reach R3 pointer status. Before any
   type-theory subtlety, the concept detector simply missed two
   textbook nouns (likely because they appear only inside the
   assumption sentence — singular/plural or context filtering).
2. **The bundling question is REAL and the anatomy shouldn't guess:**
   bundled type (category equipped with chosen (co)equalizers — a
   sigma/structure reading) vs property (such limits exist) vs
   separately-supplied structure. This distinction is itself
   canon-grade — nLab's "property, structure, and property-like
   structure" page IS the reference for it; having (co)equalizers is
   the textbook property-like-structure example (a choice is
   structure, unique up to unique iso, so existence behaves like a
   property). The honest move: mark the parts, emit an edge typed
   `equips?` (bundling-unresolved), and let the edge's resolution
   itself carry a canon link. The markup's job is to record the
   relation AND its unresolved bundling status — not to silently pick
   a reading the author didn't state.

- Detector consequence: with-clause conjuncts get concept marks
  ALWAYS (pointer at minimum); the with-edge carries a bundling slot
  {bundled | property | supplied | unresolved}.

## R11 — "this is why we really need hypergraphs": qualification re-enters built structure

Specimen: "the braiding of $\C$ is $H$-linear" (0809.2517, §2
assumption, final conjunct). Joe, voice: "We could have represented
this whole sentence with the S-expression parse tree, but this kind
of statement — the braiding of C is H-linear — is going BACK INTO
that structure and qualifying it. You could write that down as an
S-expression, but it's really a bit heavy."

Two findings, one representational and one binder-taxonomic:

1. **Fifth binder kind — decorator-bind:** "the braiding of $\C$" is
   a use whose binder is a DECORATOR. No symbol was ever assigned to
   the braiding; it exists in scope only because the "braided" in
   "closed braided monoidal category" supplies an implicit component
   (canonically c). Decorators don't just qualify concepts — they
   introduce nameable structure into scope. Binder taxonomy now:
   macro-def / semantic-bind / convention-bind / promissory-bind /
   **decorator-bind** (component supplied by a decorator, referenced
   by "the <component> of <object>").
2. **The anatomy's native representation is a hypergraph, not a parse
   tree.** The sentence parses as a tree, but this conjunct reaches
   back into structure built earlier in the same sentence (into the
   decorator tower's interior!) and qualifies it. As a 3-ary
   hyperedge: ends = {braiding-component (via decorator-bind into
   \C's tower), H (via semantic-bind), linearity (concept/canon)} —
   flat, local, cheap. As an S-expression: paths into a re-entrant
   tree — "a bit heavy" = the encoding cost of forcing cross-links
   through tree locality. This is the formal license for what the
   scopes.json ends/roles structure already gestures at, and it is
   exactly Arxana's native store (relations over anchored ends;
   substrate-2 = relations). Paper anatomy lands in the hypergraph,
   not in a treebank.

- Tags: type-level (representation + binder kind).
- Detector hook: "the <noun> of $X$" resolved against X's decorator
  tower (component lexicon per decorator: braided→braiding,
  monoidal→tensor/unit, closed→internal hom); constraint conjuncts
  emitted as n-ary hyperedges with role-typed ends.
- Minted: 2026-06-12, live walk, voice.

## R12 — \ref/\label/\cite: the author's OWN hypergraph is being dropped

Specimen: "according to Diagram \ref{A-B-bimod2}, the left and right
$A$-module structures of $A\# H$ are given..." (0809.2517, ~point
2637). Marks: nil.

This is the cheapest possible signal and we drop it: \label/\ref is
the author's own link layer — exact, machine-readable, zero
inference. The anatomy should harvest it as a typed edge family:

- **\label{X}** = anchor (and the \begin{equation}\label{...} pattern
  ties the anchor to an env-tex scope we already detect — the join is
  free).
- **\ref{X}** = intra-paper edge; TYPE comes from the prose word
  immediately before (Diagram/Lemma/Equation/Section/Proposition...) —
  the author tells us the referent's kind at every use site.
- **\cite[locus]{key}** = extern edge, the cross-PAPER analogue of
  R3's pointer: resolves through the bibliography to another paper,
  and the optional argument is a SUB-ANCHOR into the target ("recalling
  from \cite[Proposition 2.3]{Maj3}" — earlier in this same section —
  points at a specific proposition inside Majid). Citations with loci
  are exactly the paper↔paper edges the corpus graph wants, at
  proposition granularity rather than bibliographic granularity.

With R11 this completes the picture: the paper IS a hypergraph
already, partially authored by its writer (\ref/\cite edges, explicit
and exact) and partially implicit (binder chains, decorator
components, debt regions — R2–R11). The detector job is to harvest
the authored layer verbatim and infer the implicit layer; the shame
of the current pipeline is that it infers (badly) while discarding
what was authored.

- Tags: `m` (ref edge unmarked), type-level.
- Detector hook: \label/\ref/\eqref/\cite sweep (trivial regexes, all
  files); prose-word-before typing on refs; bibliography resolution +
  locus capture on cites; emit as hyperedges with anchored ends.
- Minted: 2026-06-12, live walk, voice.

## W1 — composition witness: "the left and right $A$-module structures of $A\# H$"

Not a new rule — a nine-word phrase that composes five existing ones,
making it the natural acceptance test for the next detector
generation:

- "left and right" = side-decorators (R10) DISTRIBUTING over a
  conjunction → two concepts from one phrase;
- "$A$-module structure" = parametrized concept (R9), parameter edge
  to A's binder;
- "of $A\# H$" = component-of construction (R11 family);
- $A\# H$ = author-constructed object whose binder is the :=
  string-diagram display (R6);
- the payoff equation \nu_{A\# H}^A under \label{left-right mod
  Asm.H} = the authored anchor (R12) for future refs.

Acceptance bar: a detector run that produces the full hyperedge set
for this ONE sentence — two decorator-distributed concept nodes, the
parameter edges, the component-of edge into the smash product's
display binder, and the label anchor — has implemented R6+R9+R10+R11+
R12 correctly. Currently: marks nil across the whole line.

- Site: 0809.2517 ~point 2640-2700.
- Minted: 2026-06-12, live walk, voice.

**W1 addendum — prose designation operators + the respectively-zip
(Joe, voice):** "are given respectively by <display>" is the
definition operator of this sentence and it is unmarked. Two pieces:

1. **"is given by" is a designation operator** — the prose sibling of
   R6's `:=`. The definition-verb family ("given by", "defined by",
   "we set", "denote(d) by", "determined by") binds its subject to the
   display/expression that follows, exactly as := does inside
   displays. R6's binder detection must run on prose verbs, not just
   display-internal symbols.
2. **"respectively" is a ZIP operator** — order-sensitive pairing
   between the distributed conjuncts ("left and right ... structures")
   and the sequence of payoff expressions (here ν^A_{A\# H} and its
   partner). The hyperedge emission: zip(conjuncts, displays) → one
   designation edge EACH, preserving order. Without the zip, the two
   structures share one blurry edge to one blob of math; with it, each
   gets its own binder. ("Respectively" is pure syntax with exact
   semantics — another authored signal in R12's class, free to
   harvest.)

## R13 — authored proof structure: goal, decomposition, discharge plan

Specimen (0809.2517, ~point 3724): "The first goal of this subsection
is to prove that if $A$ is an $H$-Azumaya algebra in $\C$, then
$(A\# H)^A$ is an $H$-Galois object ($H$ flat). The proof will be
completed once we prove: 1) $(A\# H)^A$ is an $H$-comodule algebra;
2) $(A\# H)^A$ is faithfully flat; 3) $can_{(A\# H)^A}$ is an
isomorphism." Then: "We proceed to prove 1)." Marks: nil — "it's just
hanging there... obviously trying to do something mathematical"
(Joe, voice).

What the author wrote, in proof-engineering terms, is a PROOF TREE
NODE in prose:

- "The first goal of this subsection is to prove that if..then.." =
  an explicit GOAL statement (and note: it is the discharge plan for
  R8's promissory $\Upsilon$ — the subsection heading promised the
  map, this sentence states the theorem that delivers it).
- "The proof will be completed once we prove: 1)..2)..3).." = a
  DECOMPOSITION step — the author runs a 'suffices' tactic by hand,
  splitting the goal into three obligations. Three typed sorries,
  opened in one sentence, each with its statement given.
- "We proceed to prove 1)" = ENTERING a subgoal — a scope open whose
  close is wherever 1) is discharged; same extent semantics as R8's
  debt regions, nested.

This is R12's lesson at the discourse level: the proof DAG is
AUTHORED ("goal", "it suffices", "once we prove", "we proceed to",
enumerate environments, "this completes the proof") — harvestable
with a small marker lexicon, no theorem-proving required. It is also
exactly the proof-anatomy showcase's vocabulary (goal/subgoal/
discharge) arriving in the wild, and the natural consumer of the
superpod discourse-wiring stage. Subsection anatomy becomes: heading
(promise) → goal (statement) → decomposition (3 obligations) →
discharge scopes 1,2,3 → promise closed.

- Tags: `m`, type-level; Joe notes fair-enough/new-instruction status:
  proof structure was not previously in scope — it is NOW.
- Detector hook: proof-discourse marker lexicon + enumerate items as
  obligation nodes + "prove i)" anaphora resolution to item anchors;
  emit goal/decomposes-into/discharges hyperedges.
- Minted: 2026-06-12, live walk, voice.
