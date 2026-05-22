# Mission: Symbol Grounding — Structure Mining Inside `$...$` Math Content

**Date:** 2026-05-21
**Status:** IDENTIFY → MAP (sections 1–3); DERIVE for Layer 1 starting today
**Owner:** Joe (architecture + Batch One curator); claude-7 (implementation)
**Predecessor:** [M-structure-seed-promotion.md](M-structure-seed-promotion.md)
— the scope-learning loop this mission extends down into math content
**Supervised signal source:** `/home/joe/code/storage/futon6/data/first-proof/`
(First Proof Batch One: 373 files, 219 .md, agent-generated semantic markup
already using `nlab-wiring`-compatible discourse types — `bind/let`,
`constrain/such-that`, `wire/adversative`, typed `input_ports` /
`output_ports`, etc.)

## 1. Why this mission exists

The audit and the QC viewer currently treat `$...$` math blocks as opaque
LaTeX. A sentence like

> Let $\mathcal{C}$ be a monoidal category and $T : \mathcal{C} \to \mathcal{C}$
> be an opmonoidal monad.

renders with `Let` highlighted as a `bind/let` scope, `monoidal category`
and `opmonoidal monad` as kernel-term occurrences in the prose, **and the
entire `$T : \mathcal{C} \to \mathcal{C}$` block as raw LaTeX with no
internal markup at all**. The symbols `T`, `\mathcal{C}`, and `\to` carry
all the semantic load of the sentence — and we leave their meaning as
"an exercise to the reader."

Joe's framing (verbatim): *"Right now, they just aren't [grounded]. We
have mathematical terms but we leave the term-symbol correspondence as an
'exercise' to the reader — when in fact we could do it ourselves using a
refinement of existing approaches."*

The framework already supports nested scope markup, tree rendering, and
depth-aware classification. The new work is detector content + grounding
logic; nothing in the infrastructure changes.

## 2. The supervised signal

First Proof Batch One — kept under
`/home/joe/code/storage/futon6/data/first-proof/` — is 219 markdown files
with structured discourse markup. The vocabulary already matches
`nlab-wiring.py`: `bind/let`, `constrain/such-that`, `wire/adversative`,
`wire/consequential`, plus typed `input_ports` and `output_ports` carrying
labels like `concavity + f(0)=0 → subadditivity`. The math content lives
in backticked inline expressions (rather than `$...$`) but the conversion
to LaTeX is already in `/home/joe/code/futon6-upstream/data/first-proof/latex/`.

This gives us **paired data**:

- `(markdown source, semantic markup over inline math)` pairs from agent
  authoring
- `(rendered LaTeX, inferred markup)` — what we'd need to recover from a
  raw arXiv paper, the inverse problem

A symbol-grounding model can learn from the forward direction (which we
have) and validate against the inverse (which the structure-learning loop
will produce).

## 3. Three-layer arc

The arc is intentionally staged so each layer is independently shippable
and the visual demo improves at each step.

### Layer 1 — Regex-level math-scope detection (this commit / next 1–2)

Extend `scripts/nlab-wiring.py` with a new `detect_math_scopes()` family
that fires *inside* `$...$` blocks. Initial pattern targets:

- **Typed-arrow**: `f : X \to Y` → nested `math/typed-symbol > math/typed`
- **Named operators**: `\Hom(A, B)`, `\Spec(R)`, `\Pic(C)`, `\Tr(g)`,
  `\End(V)`, `\Aut(G)` → `math/named-functor` with argument scopes
- **Composition**: `g \circ f` → `math/composition`
- **Adjunction**: `T \dashv U` → `math/adjunction`
- **Set membership / inclusion**: `x \in X`, `A \subseteq B`,
  `A \subset B` → `math/membership`, `math/inclusion`
- **Binders**: `\sum_{i=1}^n`, `\prod_{...}`, `\bigcup_{...}`,
  `\int_0^1` — already partially handled in
  `_detect_symbolic_binders`; surface them as math scopes
- **Quantifiers**: `\forall x`, `\exists y` → `math/quantified`
- **Equations**: `X = Y`, `X \cong Y`, `X \simeq Y` → `math/relation`
- **Macros likely to denote category-theory objects**:
  `\mathcal{C}`, `\mathbf{C}`, `\cat{C}` → `math/category-symbol`
- **Identity/unit**: `\mathrm{id}_X`, `1_X` → `math/identity`
- **Variables**: single-letter symbols + subscripts as default
  `math/symbol` leaves so the renderer always emits *something*

The acceptance test for Layer 1 is the visible-density check on the v2
demo: the current screenshot shows `$...$` blocks as bare LaTeX; the
new demo should show inline scope marks inside math, layered at correct
depths (e.g., `math/typed-arrow > math/named-functor > math/symbol`).

Each new regex gets a focused test in `tests/test_nlab_wiring.py`.

### Layer 2 — Math AST sub-parser

Regex hits a ceiling around nested braces, TeX macro expansion, and
operator precedence. Layer 2 swaps the regex layer for a real parser:

- Try `pylatexenc.latex2text` and `pylatexenc.latexwalker` first — they
  produce a node tree.
- Walk the node tree; emit scope records per node with positions
  back-mapped to the original `$...$` content.
- Slot the resulting scope spans into `build_scope_tree` from
  `futon6.structure_seed`. The existing tree-aware renderer handles
  nesting depth automatically.
- Cross-check that the math-scope coverage matches what Layer 1 produced
  on a sample of papers; the AST should subsume the regex layer with
  minor improvements in nested cases.

If `pylatexenc` is too heavy or not available, fall back to a small
custom recursive-descent parser focused on the math constructs we care
about.

### Layer 3 — Symbol-term grounding via a defeasible strategy library

**Reframing (2026-05-21):** the right model is *not* a grounding
pipeline that builds a paper-level dictionary. It is a **library of
named, defeasible strategies** that each emit tentative `(symbol,
canon, scope-range)` bindings with provenance. Bindings can be
*defeated* by later evidence in the same paper (e.g., `Let X be a
finite abelian group` later in the paper narrows the scope of the
earlier `Let X be an abelian group`). Bindings are paper-local — no
symbol persists across papers. What *can* persist across papers is
**strategy effectiveness**: which strategies fire, how often, with
what corroboration.

Joe's verbatim guidance: *"Symbol grounding has to be per-paper, but
even so, maybe we can learn an approach that will improve symbol
grounding over all."* That "approach" is the strategy library, and
the cross-paper learning happens at the meta-level (hit rates,
corroboration rates, strategy composition).

#### Core data model

```python
@dataclass
class SymbolBinding:
    symbol: str                   # e.g., "X" or "\\mathcal{C}"
    canon: str | None             # e.g., "AbelianGroup"; None = unbound
    type_phrase: str              # the raw RHS, e.g., "abelian group"
    scope_start: int              # paper position where binding applies from
    scope_end: int                # exclusive end (later binding narrows this)
    confidence: str               # "high" / "medium" / "low"
    strategy: str                 # name of the strategy that produced this
    evidence_span: tuple[int, int]
    defeated_by: str | None       # binding id that narrowed/superseded this
```

`SymbolEnvironment` is a **piecewise** lookup: given a paper position
`p` and a symbol `X`, return the binding whose scope range contains
`p`. New bindings narrow the scope of earlier bindings rather than
deleting them — the original binding stays in the log with
`defeated_by` set so the meta-learning loop can see when and why each
strategy got overridden.

#### Strategy library (starter set)

| # | Strategy | Looks for | Confidence |
|---|---|---|---|
| 1 | `let-binding` | `Let $X$ be a Y` | high |
| 2 | `denotation` | `$X$ denotes Y`, `we denote by $X$ Y` | high |
| 3 | `fix-pattern` | `Fix $X$ as Y` | high |
| 4 | `the-Y-X` | `the Y $X$` (e.g. "the category $\\mathcal{C}$") | medium |
| 5 | `inline-is-a` | `$X$ is a Y` | medium |
| 6 | `color-channel` | preamble `\\newcommand{\\X}[1]{\\textcolor{...}{#1}}`. **This is the First Proof signal** — color-coded macros are a type channel (delimiters in pink, etc.). Color does not give us canonical names but it does cluster symbols into typed channels. | high (for the channel; type-name extraction still needed) |
| 7 | `notation-env` | `\\begin{notation}...\\end{notation}` blocks | high |
| 8 | `section-context` | section heading near first occurrence of symbol | low |
| 9 | `kernel-ambient` | NER kernel lookup with no in-paper declaration | low |

#### Defeasibility

Every strategy emits *defeasible* bindings:

- A `let-binding` at position 1024 binds `X → AbelianGroup` from 1024
  onward.
- A later `let-binding` at position 4500 binds `X → FiniteAbelianGroup`.
  The first binding's `scope_end` gets capped at 4500. The first
  binding is not deleted — it now applies only on `[1024, 4500)` and
  carries `defeated_by` pointing at the later one.
- Two strategies disagreeing at the same position is also a defeat:
  whichever has higher confidence wins; the loser remains in the log.
- Implicit defeat by *no later evidence* is fine — bindings naturally
  apply until end of paper if no narrowing occurs.

This makes the grounder honest about reading mathematics: locally
bound symbols stay locally bound, and the system records *where* and
*why* every claim was made, not just the conclusion.

#### Cross-paper meta-learning

For each strategy across a batch:
- `hit_rate` = total emitted bindings / total math-symbol occurrences
- `corroboration_rate` = bindings that agree (same canon) with another
  strategy / total bindings emitted
- `defeat_rate` = bindings that got narrowed by later evidence / total

Strategies with high `corroboration_rate` are trustworthy. Strategies
that fire alone and never get corroborated are either solo-correct on
their slice (good) or noise (bad). The QC headline reports these
per-strategy numbers each batch; over time we keep the high-trust
strategies and prune or constrain the low-trust ones.

#### Frontier extends to math

Every ungrounded symbol inside `$...$` becomes a math-frontier
candidate. The audit summary gains:
- `math_inhabited_terms` (symbols with at least one binding)
- `math_outer_terms` (symbols mentioned but not bound by any strategy)
- `math_inhabitation_rate`

These mirror the prose-level frontier counts but for symbol-grounding.

#### First Proof's actual role

First Proof Batch One uses **color-based markup** (delimiters in
pink, etc.) for visual symbol types in agent-authored proof content.
It is *not* a pre-labeled `(symbol, canon)` dataset. Its role here:

- **Calibration corpus** — known-rich math content where we can
  hand-judge whether the strategy library produces sensible bindings.
- **Color-channel reference** — the visual conventions inform
  strategy 6 above (preamble color macros as type channels).

The grounder needs no supervised labels to start: prose-pattern
strategies + kernel lookups are sufficient for a v1. First Proof
calibrates judgments, not weights.

## 4. Out of scope

- Generating new mathematics from grounded markup (the *forward* direction
  that Batch One implements). This mission is about *recovering* grounded
  markup from existing arXiv content.
- Cross-paper symbol resolution (when `\mathcal{C}` in paper A is the
  same category as `\mathcal{C}` in paper B). That's a separate
  citation-network problem.
- LLM-based grounding. Layer 3 starts with prose-context heuristics +
  Batch One supervised signal. An LLM-augmented layer could land later
  but isn't on the path here.

## 5. Success criteria

- **Layer 1**: v2 demo regenerates with visible scope marks inside
  `$...$` blocks on ≥80% of math-heavy papers in the demo set. New
  regex regression tests pass.
- **Layer 2**: math-scope coverage parity vs. Layer 1 (regex ceiling
  is reached on the same set), plus AST handles 5+ specific
  constructions the regex layer misses (named here as gold cases).
- **Layer 3**: on a 30-paper sample, symbol-grounding precision ≥ 50%
  measured against First Proof Batch One projection. Audit's
  `math_outer_count` per paper is non-trivial (the frontier metric
  extends into math, giving the loop a new target).

## 6. Dependencies

- `pylatexenc` (Layer 2). If unavailable on the superpod, custom parser
  is the fallback.
- First Proof Batch One files staying in place at
  `/home/joe/code/storage/futon6/data/first-proof/`. The data is the
  supervisory anchor; losing it would force us to invert from prose
  context alone.

## 6.5 Evidence (so far)

- **Layer 1** (regex math scopes): `nlab-wiring.detect_math_scopes` —
  shipped in the v2 viewer; commit `1b4f0a4` and follow-ons.
- **Layer 2** (math AST): `src/futon6/math_ast.py` — custom recursive-
  descent parser (pylatexenc not used); commit `c38e62b`. Tests in
  `tests/test_math_ast.py`.
- **Layer 3 core** (defeasible strategies): `src/futon6/symbol_grounding.py`
  with `SymbolBinding`, `SymbolEnvironment`, `merge_bindings`, three
  starter strategies (`let-binding`, `denotation`, `the-Y-X`); commit
  `dce28fa`. Tests in `tests/test_symbol_grounding.py`.
- **Layer 3 viewer wiring**: each grounded atom shows a purple badge
  with canon name + strategy-attribution tooltip; commit `432fa91`.
- **Strategy 6 (color-channel)** and **Strategy 7 (notation-env)** are
  open. Cross-paper meta-learning is now in place (see below).
- **Task 49 — NewcommandStrategy + role palette + cross-paper vocab**:
  commit `269017c`.
    - `\newcommand`/`\def`/`\DeclareMathOperator` harvested with
      balanced-brace body extraction.
    - First-Proof `math-proofread-style.sty` v0.9 palette ported as
      `math_ast.classify_atom_role()`. Each grounded mark's label chip
      gets a `role-X` CSS class so Greek letters render Mulberry,
      named operators BurntOrange, etc.
    - Hopf demo mark count: 1566 → 2293 (+\Cat, \M, \V, \K, \B, \C, \X).
    - Galois: 5691 → 6745; Martingale: 2504 → 2679.
    - `aggregate_newcommand_vocab(envs)` scaffolds cross-paper vocab
      learning (per-symbol body distribution + recurring `common` list).
- **Task 50 — strategy meta-learning**:
    - `compute_strategy_metrics(env)` produces per-paper emit / defeat
      / corroboration / solo counts.
    - `aggregate_strategy_metrics(metrics_by_paper)` sums across
      papers and reports defeat / corroboration rates.
    - Index page now has a "Strategy meta-learning (cross-paper)"
      section listing each strategy's trust signal.

## 7. Stopping rule

The structure-learning loop has a stopping rule (M-structure-seed-
promotion.md §7: inhabitation rate < 1% per cycle). The grounding loop
extends that to math: stop iterating when **symbol-grounding precision
plateaus on held-out Batch One data AND the math-frontier metric stops
falling across two cycles**.

At that point the next bottleneck is upstream: better symbol declaration
detection in prose, or richer cross-paper resolution. Those are followups,
not this mission.
