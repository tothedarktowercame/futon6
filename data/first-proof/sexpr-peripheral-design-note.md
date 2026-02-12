# Design Note: S-Expressions, Peripherals, and Proof as Structure

*Emerged from standoff-annotation-layer session, 2026-02-12.
Participants: Joe, Claude (Opus 4.6).*

## Origin

While implementing a standoff annotation layer for the First Proof monograph
(to decouple tcolorbox box environments from clean .tex files that Codex
edits), we realized the standoff layer is a degenerate form of a much better
idea — and that the better idea connects the superpod extraction pipeline,
the futon3c peripheral model, and Arxana (futon5) into a single design.

## The Degeneracy Chain

The First Proof went through four representation stages, each fixing a
problem created by the previous one:

```
1. Inline LaTeX boxes     → destroyed by Codex rewrites (merge conflicts)
2. Standoff JSON + regex  → fragile anchoring into .tex files
3. Arxana graph           → proper typed edges, stable anchors
4. S-expr canonical form  → no anchoring needed; structure IS the content
```

Each step eliminates a class of degeneracy. Step 4 eliminates the
anchoring problem entirely because there is no "base document" to
anchor into — the semantic structure is primary, and LaTeX is a
rendering target.

## Core Principle: Idempotent Representation

IF: The superpod pipeline extracts structured representations from prose
HOWEVER: If that structure is then emitted as LaTeX and re-annotated,
  we've round-tripped through a lossy format for no reason
THEN: The extraction output format should be the authoring format
BECAUSE: This eliminates the entire class of degenerate annotation problems

The pipeline should be:

```
prose (raw) → superpod extract → s-expr (canonical) → compile to whatever
```

Not:

```
prose → superpod extract → s-expr → emit LaTeX → re-annotate LaTeX
```

The second path is what we just spent a session building infrastructure to
manage (standoff-boxes.json, apply-proof-boxes.py). That infrastructure
becomes unnecessary when the s-expr is the source of truth.

## S-Expressions as Mathematical Representation

### Why Lisp Syntax

Mathematical expressions are nested scopes. In Lisp, this is too obvious
to even state — the nesting IS the notation. The syntax cost is parens
and nothing else.

Compare the quadrifocal tensor from Problem 9:

```latex
Q^{(\alpha\beta\gamma\delta)}_{ijkl}
  = \det[A^{(\alpha)}(i,:); A^{(\beta)}(j,:); A^{(\gamma)}(k,:); A^{(\delta)}(l,:)]
```

```clojure
(quadrifocal α β γ δ
  (det (camera-row α i)
       (camera-row β j)
       (camera-row γ k)
       (camera-row δ l)))
```

The LaTeX version encodes structure (which slot, which camera, which row)
as visual position (superscript vs subscript vs argument position). The
reader must reconstruct the tree from the grid. The s-expr version makes
roles explicit — you can't accidentally confuse a camera index with a
row index because they occupy different positions in the tree.

This is also why Einstein invented summation convention, Penrose invented
graphical notation, and category theorists use string diagrams — all
attempts to recover structure flattened by index notation. Lisp just
doesn't flatten it in the first place.

### Failure Modes

LaTeX errors are typically structural — mismatched braces, lost scope,
environment nesting. These are hard to catch and produce garbage silently.

S-expression errors are typically dropped parens — mechanically catchable
by `read`. A one-line structural check replaces the regex nightmare of
heading-matching in apply-proof-boxes.py.

## Peripherals as Proof Scopes

### The futon3c Connection

The futon3c peripheral model defines scoped contexts: enter a peripheral,
certain actions become available, certain checks must be discharged,
exiting produces a certified result.

This maps directly onto proof construction:

```clojure
(peripheral :hadamard-product-interpretation
  :requires [(rank-check M) (rank-check Lambda)]
  :available [:decompose :bound :witness]
  :exit-when (verified? det-nonvanishing)

  ;; work happens here, inside the parens
  ;; the structure constrains what you can do
  ;; the paren doesn't close until exit-when is satisfied
  )
```

A complete proof is a composition of peripherals:

```clojure
(proof problem-9
  (peripheral :bilinear-form-reduction ...)
  (peripheral :rank-constraint
    (peripheral :minor-vanishing ...))
  (peripheral :algebraic-geometry-converse
    (peripheral :explicit-witness ...))
  (peripheral :tensor-factor-compatibility ...))
```

Each peripheral is a scope you enter, work inside, and can't exit without
the checks. The proof is done when all the parens close.

### Generation Cannot Outpace Checking

This is the key insight: within a peripheral, generation and checking are
the same act. You're not generating text and then checking it — you're
filling in a structure that won't close until it's valid. The paren IS
the gate.

The "generation outpacing checking" problem is literally: writing a
closing paren before filling in the body. Paredit won't let you do that.

### Why Peripherals, Not Agents

A peripheral is a mode change within the same reasoning process. Context
stays, capabilities change.

An agent hop serializes context across a boundary. When Problem 7 hit
the codimension gap obstruction, the creative insight (rotational
involutions resolve the parity tension) required simultaneously holding:
the lattice construction, the Fowler criterion, the dimension-parity
tension, and the surgery machinery. An agent handed "check equivariant
surgery" would have returned "no, gap hypothesis fails" — missing the
workaround entirely.

The peripheral model preserves this: you're still inside the proof,
holding full state, but you've entered a scope where surgery tools and
checks are foregrounded.

## Superpod Extraction: 10X and 100X

### Current State (1X)

The existing extraction pulls entities and terms. The First Proof's
"Key References from futon6 corpus" sections are almost entirely
PlanetMath definitional anchors ("PlanetMath: symplectic manifold").
None of the actual proof moves came from extracted material. The
extraction captures nouns when proofs need verbs.

### 10X: Extract Argument Patterns (no tensors)

Index by what a technique DOES, not what it IS:

```clojure
(technique
  :name "Titu's lemma for superadditivity"
  :pattern (show (>= (+ (/ a x) (/ b y))
                     (/ (square (+ a b)) (+ x y))))
  :precondition (and (> x 0) (> y 0))
  :source "math.SE #1234567")

(obstruction
  :context "equivariant surgery"
  :blocks "codimension-1 fixed sets"
  :reason "gap hypothesis requires codim >= 2"
  :workaround "rotational involution"
  :source "math.SE #2345678")
```

Categories to extract:
- **Techniques** — what transforms what, under what conditions
- **Obstructions** — what blocks what, known workarounds
- **Equivalences** — "X iff Y when Z"
- **Counterexamples** — "this fails because..." with specific failure

### 100X: Composable Wiring (with tensors)

Techniques become pluggable subgraphs:

```clojure
(wire
  (port :in  (sum-of-ratios [a x] [b y]))
  (port :out (ratio (square (sum a b)) (sum x y)))
  :transform titus-lemma
  :type-constraint (positive [x y]))
```

At 10X you search for relevant techniques and a human checks applicability.
At 100X the system checks applicability by attempting the wiring and
reporting type mismatches at the boundary.

## Arxana as Substrate

Arxana (futon5) provides the graph substrate. The standoff annotation
layer we built is a degenerate Arxana — a separate layer pointing into
base documents by anchors. Arxana provides:

- Typed edges (derived-from, blocks, enables, refines, contradicts)
- Stable anchors (not regex-matched substrings)
- Provenance tracking (which SE answer contributed to which proof step)

With s-expr canonical form, Arxana becomes the medium you work inside,
not a layer you apply on top. Adding a proof step = adding a node and
wiring it. Incorporating an SE result = splicing a subgraph. "Writing
up" = traversing the graph, emitting LaTeX or prose or brief.

The traceability gap in the First Proof (we can't tell whether SE
material helped because provenance was lost in conversation transcripts)
disappears when the proof graph IS the provenance record.

## Flexiformal Pathway Reframing

The CLAUDE.md pathway:

```
EXPLORE → ASSIMILATE → CANALIZE
```

In the s-expr model, the tree structure is already explicit at EXPLORE.
You're not adding structure as you mature — you're filling in structure
that's already there. A `(conjecture ...)` becomes `(theorem ... (proof ...))`
— same shape, more content. The representation doesn't change kind as
it matures.

The progression tree → wiring diagram → hypergraph is increasing the
dimensionality of the structural representation as needed, making more
relationships first-class. Each step is still s-expr compatible; each
step is still inside the same Arxana graph.

## Design Constraints for Superpod Job

1. Extraction output format = authoring format (idempotency)
2. Extract argument patterns, not just entities (10X)
3. Output as EDN/s-expr, not LaTeX (eliminate lossy round-trip)
4. Peripheral-compatible structure (scoped, with entry/exit conditions)
5. Arxana-ingestible (typed nodes and edges, not flat text)

## Evidence from This Session

- 25 annotations across 10 files required: 1 JSON manifest, 1 Python
  script (200 lines), regex-based heading matching, bottom-to-top line
  number tracking. All of this is accidental complexity from using LaTeX
  as the source of truth.
- The standoff layer works but is fragile: heading text must be matched
  as substrings, line positions must be resolved at apply-time, the
  output must be regenerated after any edit to the clean files.
- In the s-expr model: zero infrastructure needed. The box type is a
  node property. The content is the node's children. Editing the content
  edits the source of truth directly.
