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

### Layer 3 — Symbol-term grounding (the actual point)

Once Layers 1–2 give us *what's there structurally* inside math, Layer 3
asks *what does it refer to*.

Mechanism:

1. **Prose-side symbol declarations**: scan prose around each math block
   for declaration patterns we already detect — `Let \$X\$ be a Y`,
   `\$X\$ denotes Y`, `Fix \$X\$ as Y`, `where \$X\$ is a Y`. The
   right-hand-side is typically a kernel-term hit.
2. **Per-paper symbol environment**: a dict
   `{symbol_text: {canon, declared_kind, declaration_position}}` built
   by scanning the whole paper. `X` declared as `category` near the
   top binds for subsequent math.
3. **Math-scope grounding**: when `detect_math_scopes` emits a
   `math/symbol` record for `X`, look up `X` in the per-paper env. If
   found, attach `canon=Category` to the scope content. The renderer
   then treats the symbol as an *inhabited term* inside math — just
   like the kernel-term overlay in prose.
4. **Frontier extends to math**: every ungrounded symbol becomes a
   `math/scope-development-frontier` candidate. The audit summary
   reports `math_inhabited` / `math_outer` / `math_total` alongside the
   prose-level frontier counts.

### Supervised validation via First Proof Batch One

The Batch One markup gives us truth labels for what symbol grounding
*should* produce on agent-authored content. Layer 3 includes:

- A converter from Batch One's port labels + discourse records into the
  same `(symbol, canon)` pairs the prose-side grounder produces.
- A precision/recall comparison on a held-out subset of Batch One.
- An honest report: if our prose-side grounder hits 60% precision on
  Batch One data, that's the upper-bound expectation on arXiv content
  where authors aren't writing with grounding in mind.

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

## 7. Stopping rule

The structure-learning loop has a stopping rule (M-structure-seed-
promotion.md §7: inhabitation rate < 1% per cycle). The grounding loop
extends that to math: stop iterating when **symbol-grounding precision
plateaus on held-out Batch One data AND the math-frontier metric stops
falling across two cycles**.

At that point the next bottleneck is upstream: better symbol declaration
detection in prose, or richer cross-paper resolution. Those are followups,
not this mission.
