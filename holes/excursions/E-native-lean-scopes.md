# E-native-lean-scopes — scopes generated from Lean, not mined from prose

**Type:** excursion (design, no code yet). **Raised:** 2026-11-05 by Joe, with Claude,
while outlining `holes/T-arxana-ts.md`. **Status:** open. **Waits on:** nothing for the
Lean-source half; the Lean-to-LaTeX renderer
([electric-samurai/mfuton-share#8](https://github.com/electric-samurai/mfuton-share/issues/8))
for the rendered-page half.

## 0. The question

Mark8 scopes are mined from prose by S1 regexes and S3/S4 model calls, then placed
by offset. A Lean file has the same structure as facts: binders, references,
dependencies, proof steps. EmitPython (`mfuton-share`) already reads those facts
into Python. So: what would a scope layer generated *natively* from Lean look like,
what does it give that LaTeXML-style markup of the same text does not, and what can
start before Rob's renderer lands?

The short answer Joe supposed is right, and can be made exact: LaTeXML and the
token-role contract (`\mBin`, `\mRel`, ...) label each token *locally*; native scopes
add the *edges* (occurrence to binder, reference to declaration, declaration to
declaration), and edges are what global inference runs over. Section 3 says which
S1/S3/S4 kinds become exact, which stay heuristic, and which do not exist on the
formal side at all.

## 1. What EmitPython exposes today (read from the share, 2026-11-05)

| Surface | Module | Addressing | Facts |
|---|---|---|---|
| Syntax tree | `syntax_types.SyntaxTypes.from_lean(positions=True)` | per-token **byte** span `(start, stop)`; a node's range is its leaf tokens' hull | kind per node (`Lean.Parser.Command.theorem`, `«term_+_»`, tactic kinds), idents, atoms, doc comments as real nodes, trivia comments synthesised with spans |
| Elaborated terms | `lean_expr_surface` (`ForallE`, `Lam`, `LetE`, `App`, `Const`, `BVar`, `FVar`, `MData`), loaded by `lean_expr_loader.load_const_surface` from the EmitPython emission | none (an `Expr` carries binder names, not positions) | statement (`type`) and proof (`value`) per constant; binder names, types, binder info; constant references; DAG-shared |
| Dependency closure | `blueprint_render/uses.collectUsed`, `get_used_constants` | declaration names | transitive `\uses`; `\leanok` = no `sorryAx` in the closure |
| Declaration surface | `entity_surface/nodes/leaves.py` (`LeanTheorem`, `LeanDefinition`, ...), `declaration_interface.py` | `start_line/end_line`, `start_col/end_col`, selection range | docstring, interface sites with role / binder kind / line / column / references (bound or not) |
| Source binding | `expression_binders_and_scope.LeanSourceBinding` | `statement_lines`, `proof_lines`, `source_ids`, `obligation_ids` | a Lean attribute that names the informal passage a declaration formalises |
| Proof graph artifact | `proof_graph/graph_core` | declaration node ids | typed graph (node-link JSON) over declarations and edges |
| Module surface | `native/module_surface` | names | imports, `constNames`, (next) env-extension entries |

Not present: Lean info trees or tactic goal states. EmitPython reads oleans and runs
the parser; it does not elaborate, and oleans do not store info trees. So per-tactic
"goal before / goal after" is not available from this share. Everything below is
written to that constraint; section 5 says what a small Lean-side export would add.

Two consequences for the Arxana core in `T-arxana-ts.md`:

- Lean spans are **bytes**; the core stores **codepoints**. Lean sources are dense
  in multi-byte characters (`∀`, `→`, `ℝ`, `≤`), so the conversion is real work and
  belongs in the importer, once, with a test on a file full of Unicode.
- Expr binders have no positions. The bridge from a `ForallE` binder to a source
  token is by binder name and order within the declaration's syntax node. Exact for
  statement binders (`(m n : Int)`), approximate inside tactic blocks (`intro x` may
  shadow). Report the bridge's confidence as `located`, not `exact`.

## 2. What LaTeXML (and the token-role contract) gives, and where it stops

A LaTeXML page of a paper gives block structure (`ltx_theorem`, `ltx_proof`,
sections), MathML with `alttext`, `data-sourcepos`, and operator/identifier tags.
The token-role contract Joe offered Rob in mfuton-share#8 (`\mNumber`, `\mBin`,
`\mRel`, `\mDelimiter`, roles taken from Lean Syntax/Expr, colours from a `.sty`,
HTML from latexml-oxide) makes the roles of a Lean-rendered page *exact* where a
paper's are guessed: the renderer knows a `+` is `HAdd.hAdd`.

Both stop at the token. Neither says:

- which binder this `x` belongs to, or whether two `x`s are the same variable;
- which declaration a name refers to, or what that declaration's statement is;
- what a proof rests on, transitively;
- whether a step is proved or `sorry`.

Those are relations between spans, and a rendering can only carry them by baking
ids into the output (an `id` per binder, a `data-ref` per use), which couples the
page to one scope vocabulary and violates the no-baking rule in
`futon4/README-essays.md`. The scope layer is where relations live; the page keeps
roles. This is the division of labour to propose on mfuton-share#8: **roles through
macros, scopes through the sidecar**, and one extra thing from the renderer, a
**source span per rendered token** (or at least per declaration), so the overlay's
resolver is `exact` on the blueprint page. Without that, the page falls to the
`align` resolver in `T-arxana-ts.md` §4, which works but is lossy on math.

## 3. The mapping: what each mark8 kind becomes on the Lean side

The S1 vocabulary (`render_scope_margin.MARK_KINDS`) is, read plainly, a list of
guesses at binding structure. In Lean every one is a fact, or does not exist.

### 3.1 S1 marks: binding and reference (exact)

| S1 kind (prose heuristic) | Lean fact | Source |
|---|---|---|
| `bind/let`, `let-binder`, `definiendum`/`definiens` | `ForallE`/`Lam` binder in the statement; `LetE`; a `def`'s name and value | Expr + syntax binder tokens |
| `bind/typed` (`f : A → B`) | binder with `binderType` | Expr |
| `assume/explicit` | a binder whose type is a Prop (`is_prop`, once emitted) | Expr |
| `quant/universal`, `quant/existential` | `ForallE` with Prop body; `App(Const Exists, Lam)` | Expr |
| `bind/summation`, `bind/product`, `bind/integral` | `App(Const Finset.sum / Finset.prod / MeasureTheory.integral, ..., Lam)` | Expr |
| `constrain/relation`, `constrain/such-that` | `App(Const Eq / LE.le / Membership.mem ...)` | Expr; roles already in the token contract |
| `symbol` / `symbol-grounded` and the binding **strategies** `in-environment`, `proved-statement`, `in-section`, `in-paper`, `unbound` | every `BVar` resolves by de Bruijn index to exactly one binder; every `FVar` to a section variable; every `Const` to a declaration. There is no `unbound` and no `assumed` rule | Expr |
| `concept` (lexicon match, grounded or not) | `Const` to a Mathlib or local declaration; grounded by construction, except `sorryAx` and axioms | Expr |
| `cite`, `ref` (dangling or not) | `Const` reference; never dangling | Expr, `collectUsed` |
| `env/theorem`, `env/proof`, `env/definition` | declaration kind and range (`LeanTheorem` ... with `start_line/end_line`) | declaration surface |
| `anaphor` | does not exist (no anaphora in terms) | |
| `role-gap`, `unknown`, `classified` (macro roles) | the token-role contract, exact | renderer |
| `text-mode`, `layout` | comments, docstrings, `moduleDoc` | syntax tree |

On the a00J02 page the `in-paper` strategy fires 58 times and is marked
"(assumed)" in the glossary. On the Lean side the same question has one answer per
occurrence and a span for the binder. That is the first concrete thing native scopes
offer: **every symbol use is an edge to its binder, with no assumed rule**, and the
edge can be drawn across the informal text once an informal symbol is aligned to a
Lean binder (`M-informal-lean-alignment`).

### 3.2 S3 proof graph: steps and warrants (exact at declaration grain, partial inside a proof)

| S3 notion | Lean fact | Grain |
|---|---|---|
| node `ref` (a result the proof points to) | `Const` in the proof value; `\uses` | exact |
| node `definition` | `Const` to a `def`/`structure` | exact |
| node `object` (an object the proof constructs) | `LetE`, `obtain`/`have` with a term, `Exists.intro` | syntax tree + Expr, approximate inside tactics |
| node `claim` | `have h : P := ...` (syntax node with span); the theorem's own statement | syntax tree |
| edge with warrant `stated` / `cited` / `missing-warrant` | every step has a term; `cited` = the step's head constant is a named lemma; `stated` = the term is built inline; **`missing-warrant` = `sorry`** (`\leanok` is its negation) | exact at declaration grain |
| `proof-move` (clearly, it is easy to see) | an automation tactic: `simp`, `decide`, `grind`, `omega`, `aesop`, `norm_num`. The hedge *with a certificate* | syntax tree |
| the quote-check (does a node's quote match its gloss) | not needed: a step's span is its syntax node | |

The `missing-warrant` ↔ `sorry` correspondence is the formal end of the test
`M-informal-lean-alignment` already states ("S3 holes coincide with Lean helper
lemmas"). Natively: a helper lemma is a declaration with in-degree 1 in the `\uses`
graph, computed, not judged. An informal hole aligned to such a lemma is a hole the
formalisation filled; an informal hole aligned to a `sorry` is a hole nobody has
filled, in either register. That pair of counts is a run-level number worth a chart.

Inside a tactic proof, step order and the local context after each tactic need info
trees (section 5). Without them the native S3 graph is: one node per `have`/`obtain`/
`show`/`calc` step with its span, edges from the constants each step's term mentions,
and warrants at the declaration level. That is already more than S3 gets right on
prose (87% of S3 nodes cite `s_i` as node `i` in mark7master-20260921).

### 3.3 S4 expository scopes: rhetorical, so mostly not native, with formal shadows

S4's vocabulary (`holes/excursions/expository-superpod-vocab.edn`) is rhetorical:
`rationale/telos`, `connection/*`, `open-problem/status`, `difficulty-assessment`,
`auxiliary-construction`, `generalisation`. Lean code states none of these; its
docstrings and the blueprint prose do. So on the Lean side S4 is not replaced, it is
**re-targeted**: the S4 stage runs over docstrings (`LeanDeclaration.docstring`),
`moduleDoc` blocks, and blueprint prose, with the region's formal neighbour known.

What changes is the **fill**. On prose, a fill is a phrase, checked only by
`echo` / `unanchored` / `bare-noun`; on APM, 17% of 8,677 fills were the slot
description copied back. On Lean, several holes can be filled with a **declaration
name**, which is anchorable (the declaration has a span) and checkable against the
graph:

| S4 kind | Formal shadow | Check |
|---|---|---|
| `auxiliary-construction` ("the key lemma") | a declaration in the theorem's `\uses` closure with in-degree 1 | fill ∈ closure |
| `connection/literature-gap` ("X proved ... but not ...") | a Mathlib constant the proof uses, or none | fill resolves |
| `connection/example-source` | an `instance` or `example` declaration | kind check |
| `generalisation` ("more generally") | a Mathlib lemma of which the local one is an instance | later: unification, `exact?` |
| `computes-invariant/calculation` | `decide` / `norm_num` / `rfl` proofs | tactic kinds |
| `difficulty-assessment` | automation succeeded (easy) or a long hand proof (hard) in the same file | proof length, tactic mix |
| `rationale/telos`, `open-problem/status`, `heuristic-plausibility` | none; prose only (`sorry` with a docstring is the nearest thing to an open problem) | flags as now |

So the second concrete offer: **a fill that names a declaration is a verifiable
fill.** The audit stops asking "is this phrase in the cited lines" and asks "is this
declaration in the closure", which is a graph query. That is the "global inference"
Joe pointed at, stated as a check the run can fail.

## 4. What to build, and in which order

The Lean-source half needs nothing from the renderer. It emits the same
`AnnotationSet` shape `T-arxana-ts.md` §3 defines, with one segment per `.lean` file,
so the TS core, the VS Code substrate and the web view consume it unchanged.

```
.lean files + oleans
  ├─ EmitPython parse (positions=True)      → tokens with byte spans, syntax kinds      [layer lean-syntax]
  ├─ const_surface (elaborated Expr)        → binders, bvar→binder edges, Const refs    [layer lean-binding]
  ├─ collectUsed / proof_graph              → declaration hyperedges, leanok, sorry     [layer lean-uses]
  ├─ docstrings + moduleDoc + blueprint     → S4 stage, fills may be declaration names [layer s4]
  └─ LeanSourceBinding / align_informal_lean → cross-segment edges to the .tex          [edges]
  → AnnotationSet JSON (codepoint offsets, one Segment per file, hash per file)
```

1. **`scripts/lean_scopes.py`** (futon6, Python, imports `mfuton`): for one Lean
   project, emit `lean-syntax` and `lean-binding` layers and the `lean-uses`
   hyperedges. Legend from a `LEAN_KINDS` table in the style of `MARK_KINDS`: an
   undefined kind stops the build. Check: `validate()` passes; every `BVar` has an
   edge; the byte-to-codepoint conversion round-trips on a Unicode-heavy file.
2. **Overlay on the `.lean` in VS Code** (`T-arxana-ts.md` phase 3) and on the
   source as plain text in the web view. No renderer needed. Check: hover a symbol,
   see its binder; click a declaration, see its `\uses` closure and whether it is
   `leanok`.
3. **Cross edges**: read `LeanSourceBinding.statement_lines/proof_lines` where
   present, and `align_informal_lean.py` pairs otherwise, as hyperedges between the
   `.tex` segment and the `.lean` segment. Now the a00J02 margin page and the Lean
   file are one set with two segments. Check: for the 7 complete-Lean problems in
   the Euler-characteristic evaluation set, every aligned pair resolves on both sides.
4. **S4 over docstrings** with declaration-name fills and the closure check.
   Measure the echo rate against the 17% prose baseline.
5. **Blueprint page** (after mfuton-share#8 lands): if the renderer emits a source
   span per token, `exact`; else `align`. Report the resolution histogram either way.

Items 1 to 4 can start now on the apm-lean corpus (`/home/joe/code/apm-lean/problems/<pid>/lean/Main.lean`, per
`README-apm-lean-ground-control.md`), with EmitPython at `ec03d27` on Linux, which
Joe reported working on the 4.29.0-rc8 oleans.

## 5. What a small Lean-side export would add

Info trees (`Lean.Elab.InfoTree`) carry, per syntax node, the elaborated term and,
per tactic, the goals before and after. With them:

- the Expr-to-token bridge becomes exact inside tactic proofs (`TermInfo` holds
  both the `Expr` and the `Syntax` with its position);
- S3-native steps get a goal state each, so a `have` step's "claim" is the goal it
  closes, not just its stated type;
- `proof-move` hedges (automation calls) get the goal they discharged, the thing a
  reader of "clearly" wants to see.

This is a few dozen lines of Lean run as a `lake env` script over the project, the
same posture as `lean-wiring` reading the compiler's pass list. It is not in the
share and is not EmitPython's job; note it as a later request, separate from #8.

## 6. The argument, for the record

```
IF:       scopes over Lean should support inference across a whole corpus
          (what rests on what, which holes are filled, which symbol is which).
HOWEVER:  rendering-level markup (LaTeXML, token-role macros) is local to a token,
          and baking relations into a page couples it to one vocabulary.
THEN:     generate scopes natively from EmitPython's surfaces as a sidecar
          AnnotationSet over the .lean files, keep roles in the renderer, and
          ask the renderer only for a source span per token.
BECAUSE:  in Lean the S1 binding guesses are facts (de Bruijn, Const), the S3
          warrant question has an exact answer at declaration grain (sorry), and
          S4 fills can be declaration names checked against the uses closure;
          none of that needs the renderer, all of it needs positions, which
          EmitPython already keeps.
```

Prediction error to watch for: the Expr-to-token bridge inside tactic blocks may be
worse than "approximate" on proofs that shadow names or use `<;>` combinators. If
the first run on the Euler-characteristic set shows more than a few percent of
`BVar` edges landing on the wrong token, bring section 5 forward.
