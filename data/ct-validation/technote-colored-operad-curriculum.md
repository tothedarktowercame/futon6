# Technote: Colored Operad Curriculum

**Date:** 2026-02-10
**Context:** CT validation session — wiring diagram metatheory v3.0

## The Observation

The wiring diagram metatheory has three sorts of content that are currently
rendered flat (e.g. `bind/let · Let C,D be categories`):

1. **Meta-language** — the reasoning structure (Bind, Quantify, Constrain, Conclude...)
2. **Domain objects** — the mathematical content (Category, Functor, NatTrans...)
3. **Informal glue** — the English connectives and anaphora (because, but, the above...)

These three sorts interact in a typed way: a Bind block *takes* domain objects,
a wire/causal *connects* meta-blocks, a port *references* across the tree.
The type constraints are implicit in the current model. They should be explicit.

## The Structure: Colored Operad

In category theory, a **colored operad** (= multi-sorted operad, = symmetric
multicategory) is a wiring diagram where:

- Wires carry **colors** (= sorts, = types)
- Operations have **typed input/output ports** — only matching colors compose
- Composition respects the coloring — you can't plug a wire of color A into
  a port expecting color B

This is exactly what we need. The colors are:

| Color | Sort | Examples |
|-------|------|----------|
| `meta` | Metatheory component | Bind, Quantify, Constrain, Conclude |
| `domain` | Mathematical object | Category, Functor, NatTrans, Group |
| `informal` | Natural language glue | because, hence, but, the above |

A component like `Bind` has a typed interface:

```
Bind : domain* → meta
```

It takes zero or more domain-typed inputs (the things being bound) and produces
a meta-typed scope node. Currently we write:

```
c0["bind/let<br/>Let C,D be categories"]
```

In the colored operad, this becomes:

```
(meta/Bind (is-a C Category) (is-a D Category))
```

Or in a Scratch/Blockly visual:

```
┌─────────────────────────────────┐
│  Bind                           │  ← blue meta block
│  ┌──────────┐  ┌──────────┐    │
│  │ C : Cat  │  │ D : Cat  │    │  ← green domain ovals
│  └──────────┘  └──────────┘    │
└─────────────────────────────────┘
```

## The Pedagogical Insight: Quotient Functors as Curriculum

The metatheory has 30 component types. A kindergartener doesn't need 30 types.
But a kindergartener *does* reason — they bind variables ("Let's say this is
yours and this is mine"), quantify ("every single one"), constrain ("only the
red ones"), and conclude ("so we each get three").

The resolution: **quotient functors** over the colored operad.

```
Kindergarten:   Name It   Every    Only If    Because    So       But
                  │         │         │          │        │        │
Grade school:   Let       Every    Must be    Because  Therefore  But
                  │       /    \      │          │        │       / \
High school:    Let     ∀      ∃    Such that  Since    Hence   But  However
                  │     │      │      │          │        │       │     │
Undergrad:      bind   q/u   q/e   c/s-t      w/caus   w/cons  w/adv w/adv
                  │     │      │      │          │        │       │     │
Full metatheory: bind/  quant/ quant/ constrain/ wire/   wire/   wire/
                 let    univ   exist  such-that  causal  conseq  advers
```

Each level **un-quotients** one block into finer distinctions. The wiring
structure is preserved under quotient — the diagram still composes correctly.
Only the label resolution changes.

**A curriculum is a tower of quotient functors over a colored operad.**

The student learns:
1. Where each block goes (the wiring — invariant across levels)
2. That their familiar block was secretly two different things (the refinement)

This is exactly how Scratch works for programming: kids learn sequence, loop,
conditional as blocks. Later they learn that "loop" was secretly `for` vs
`while` vs `do-while`. The wiring doesn't change; the resolution does.

## The Other Axis: Concept Stratification by In-Links

The quotient tower gives the **vertical** axis (reasoning complexity). The
**horizontal** axis is concept difficulty, measurable by in-link count in a
knowledge graph.

### PlanetMath In-Link Stratification (9,451 entries, 18,326 links)

| Stratum | In-links | Count | Examples |
|---------|----------|-------|----------|
| Foundational | 30+ | 28 | Group (51), Equation (40), Triangle (40), Ring (31), Category (30) |
| Core | 10-29 | 190 | Axiom of Choice (28), Subgroup (35), MetricSpace, TopologicalSpace |
| Intermediate | 3-9 | 1,110 | FunctorCategory, NaturalTransformation, AbelianCategory |
| Specialized | 1-2 | 2,500 | Specific theorems, proofs, examples |
| Terminal | 0 | 5,337 | Advanced results, index pages, HoTT chapters |

The distribution follows a power law: 56% of concepts are never referenced
by other entries (terminal/leaf nodes), while the top 28 concepts account for
a disproportionate share of references.

**Most foundational** (most in-links):
```
 51  Group
 40  Equation
 40  Triangle
 36  SubstitutionNotation
 35  Subgroup
 33  Graph
 31  Ring
 30  Category
 28  AxiomOfChoice
```

**Most terminal** (zero in-links, most out-links — "survey" nodes):
```
 30 out  6.12 The flattening lemma (HoTT)
 30 out  7.2 Uniqueness of identity proofs and Hedberg's theorem
 29 out  Table of integrals
 28 out  High school mathematics (!)
 27 out  List of probability distributions
 21 out  Index of properties of topological spaces
```

Note "High school mathematics" has 28 out-links and 0 in-links — it's a
survey node that references many foundational concepts but is itself never
referenced. It's a *consumer* of the knowledge graph, not a *provider*.

### Popularity vs Prerequisite Depth

In-link count measures **popularity** (how often a concept is referenced), not
**prerequisite depth** (how much you need to know to understand it). These are
often inversely correlated:

- **Group** (51 in-links): axiomatically shallow — three axioms, stands alone.
  But universally referenced because everything *uses* groups.
- **Circle** (7 in-links): axiomatically deep — to *really* understand a circle,
  you need topology (S^1), group theory (SO(2) ≅ U(1) ≅ ℝ/ℤ), complex analysis
  (unit circle in ℂ). But few things explicitly cite "circle" as a prerequisite.

Circle *is-a* Group (it's SO(2)), but Group *is-not-a* Circle. The dependency
arrow goes one way. A proper curriculum ordering would use the **topological
sort of the dependency DAG** (prerequisite chain), not the popularity ranking.
The in-link count tells you what's *most connected*; the dependency depth tells
you what's *most composed*.

This means the horizontal axis should really distinguish:
- **Popularity** — what gets referenced most (good for: "what to teach first")
- **Depth** — what presupposes the most (good for: "what's hardest to reach")

A concept can be shallow-but-popular (Group) or deep-but-obscure (Circle-as-SO(2)).
The stratification is also corpus-relative: an elementary geometry textbook would
put Circle at the top of the popularity ranking; PlanetMath's algebra-heavy corpus
puts Group there instead.

### The 2D Curriculum (revised: actually 3D)

Combining the axes gives a 3D curriculum space, though in practice you project
to 2D by choosing which horizontal axis matters for your context:

```
                    Concept difficulty →
                    (in-link stratum: foundational → terminal)

Reasoning       ┌──────────────────────────────────────────┐
complexity      │                                          │
    │           │  "Name three        "Name all the        │
    │  Kinder   │   shapes"            quadrilaterals"     │
    │           │                                          │
    │           │  "Every shape        "Every prime has     │
    │  Grade    │   has sides"          a unique            │
    │           │                       factorization"     │
    │           │                                          │
    │  High     │  "∀ε>0 ∃δ>0         "∀ functors F,G:     │
    │  school   │   s.t. ..."          Nat(F,G) is..."    │
    ↓           │                                          │
                └──────────────────────────────────────────┘
```

The vertical axis (quotient tower) controls which reasoning blocks are available.
The horizontal axis (in-link stratum) controls which domain objects fill the slots.
Both are independently adjustable.

## Implementation: Blockly

Google's **Blockly** library (MIT license, JavaScript) is purpose-built for this.
It is the open-source engine behind MIT Scratch.

- Block shape/color = component sort (`meta` = blue, `domain` = green)
- Typed input slots = colored operad port constraints
- Stack connectors = wires (causal, consequential)
- Reporter blocks (oval) = domain objects that snap into slots

A quotient level is just a Blockly **toolbox configuration** — same engine,
fewer block types in the palette.

The connection constraints enforce the colored operad structure physically:
you *cannot* plug a wire where a domain object goes. That's the coloring
doing type-checking via the UI.

## Connection to Wiring Diagram Metatheory v3.0

The current metatheory (unified-metatheory.edn) already has the sort structure
implicitly:

- **Components** (30 types) = `meta` sort
- **Domain objects** = referenced by components but not typed in the metatheory
- **Wires** (5 types) = `informal` sort (connectives)
- **Ports** (11 types) = `informal` sort (anaphora)

To make it a colored operad, we would:

1. Add a `:sort` field to each type (`:meta`, `:domain`, `:informal`)
2. Add typed port signatures to components (e.g., `bind/let` takes `domain*`)
3. Define the quotient tower as a sequence of surjective maps on types
4. Build Blockly block definitions from the typed metatheory

The domain sort would draw from the NER kernel (19,236 terms) — these become
the green ovals that snap into meta block slots.

## The Punchline

The same formal structure — a colored operad with a tower of quotient functors —
lets a kindergartener explain why they have three apples and lets a grad student
analyze functor categories. The blocks are the same shape. The wiring is the same.
Only the resolution of the type labels and the complexity of the domain objects
change.

Defunding Sesame Street right when we can teach mathematical reasoning as
block play: wire/adversative.
