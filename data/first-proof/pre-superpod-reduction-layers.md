# Pre-Superpod Note: Reduction-Indexed Literature and Multi-Layer Problem Structure

*Emerged from First Proof sprint retrospective, 2026-02-12.
Participants: Joe, Claude (Opus 4.6).
Companion to: sexpr-peripheral-design-note.md*

## Origin

During the First Proof sprint, Problem 6 (epsilon-light subsets) generated
six handoff/dispatch documents and multiple closure attempts without honest
calibration on whether the gap was closeable. The failure mode: treating
"conditional result on an open problem" as a defect to repair, rather than
a legitimate outcome to characterize.

The better question was never "can you close the gap?" but "what does this
reduce to?" — and that question is layer-dependent.

## Observation: Reductions Are Layer-Relative

The same problem reduces to different things depending on which mathematical
layer you view it from. Problem 6 demonstrates this concretely:

| Layer | Reduction | Status |
|-------|-----------|--------|
| Spectral | Bound ‖L^{+/2} L_S L^{+/2}‖ ≤ ε | Set up correctly (Sec 1,4) |
| Concentration | Control martingale R* and ‖W_n‖ | Framework correct, bounds need leverage analysis |
| Combinatorial | Vertex selection with leverage score constraints | Star domination decomposition done (Sec 4a) |
| Sparsification | Adapt BSS (edge→vertex) without reweighting | **Structural gap**: BSS is fundamentally edge-based |

Each row is a valid reduction. Each lives in a different mathematical
world. The obstruction ("BSS doesn't adapt") is specific to the
sparsification layer. The concentration layer is fine — the setup is
correct. A closure attempt that ignores layers will thrash between them.

## The Futon3 Precedent: Group vs Circle

This is the same structure observed in the futon3 definitional layering:
- **Group** is more basic than **Circle** definitionally
- **Circle** is more basic than **Group** pedagogically

"More basic" is not absolute — it's relative to a layer (definitional,
pedagogical, computational, historical, ...). The ordering reverses
depending on which layer you query.

Similarly, the "most promising" reduction of a problem depends on the
layer:
- In the spectral layer, Problem 6 is almost done
- In the sparsification layer, Problem 6 has a structural gap
- These are not contradictory; they're different views

## Design Requirement: Multi-Layer Reduction Index

The superpod extraction should index literature not just by entities or
techniques, but by **reductions with layer tags**.

### Current (1X): Entity Index
```
Problem 6 → {Laplacian, spectral sparsification, leverage scores, BSS}
```
Flat. No structure. Tells you what concepts appear, not how they relate
or in which direction the implications flow.

### Target (10X): Reduction Index
```clojure
(reduction
  :from "vertex-induced ε-light selection"
  :to "martingale concentration on vertex-indexed PSD summands"
  :layer :concentration
  :status :setup-complete
  :source "Problem 6, Section 4")

(reduction
  :from "vertex-induced ε-light selection"
  :to "BSS edge sparsification"
  :layer :sparsification
  :status :structural-gap
  :obstruction "BSS selects edges with reweighting, not vertex subsets"
  :source "Problem 6, Section 5")
```

### Target (100X): Composable Multi-Layer Graph
```clojure
(problem "vertex-induced-epsilon-light"
  (layer :spectral
    (reduces-to "operator-norm-bound"
      :via "pseudoinverse whitening"
      :status :complete))
  (layer :concentration
    (reduces-to "matrix-freedman-bound"
      :via "star-domination + Doob martingale"
      :status :setup-complete
      :needs "leverage-score-analysis"))
  (layer :combinatorial
    (reduces-to "vertex-selection-with-leverage-constraints"
      :via "Z_v A_v decomposition"
      :status :partial))
  (layer :sparsification
    (reduces-to "BSS-adaptation"
      :via "edge-to-vertex analogy"
      :status :blocked
      :obstruction "fundamentally different objects")))
```

Now you can query: "show me all problems where the concentration layer
is complete but the sparsification layer is blocked" — and find
structural analogues to Problem 6.

## How This Changes Sprint Methodology

### Old Pattern (TryHarder)
```
identify gap → attempt closure → fail → identify gap → attempt closure → ...
```
Generates handoff documents. Does not accumulate knowledge.

### New Pattern (MapReductions)
```
identify problem → enumerate layers → find reduction in each layer →
characterize status per layer → query literature for analogues →
identify most promising layer for progress
```
Generates a multi-layer map. Each reduction is durable even if it doesn't
close the problem. The map accumulates across problems.

### Honest Calibration

For an open problem, the sprint outcomes are:
1. **Closed** — proof complete
2. **Reduced** — conditional result with clear assumptions (Problem 6)
3. **Mapped** — obstruction characterized, adjacent problems identified

All three are legitimate. Option 2 is not a defect. Option 3 is valuable
even without progress toward closure. The "confidence signal" that was
missing from Problem 6's sessions is: which layers are complete, which
are blocked, and is the blocked layer structurally necessary or
bypassable?

## Connection to Arxana / S-Expr Design

The multi-layer reduction index is naturally an Arxana graph:
- **Nodes**: problems, techniques, theorems, obstructions
- **Edges**: reduces-to, blocks, enables, analogous-to
- **Edge tags**: layer, status, source

The s-expr format (see companion note) gives both the authoring format
and the query format. Working inside the graph means reductions are
recorded as they're discovered, not reconstructed afterward.

The futon3 pattern library already indexes patterns with context-dependent
firing rules. Extending this to mathematical problems means: the "same"
pattern (e.g., "martingale concentration") fires differently in the
spectral layer vs the combinatorial layer, and the index reflects this.

## Concrete Next Steps for Superpod

1. **Extract reductions from SE/literature**, not just entities.
   "This answer shows that X reduces to Y via technique Z" is the
   target annotation, not "this answer mentions X."

2. **Tag reductions by layer.** The layer taxonomy for a given field
   (spectral, combinatorial, algebraic, geometric, ...) can be seeded
   from the First Proof problems and refined by extraction.

3. **Index obstructions as first-class objects.** "BSS doesn't adapt
   because it's an edge result" is as valuable as any technique.
   Obstructions are reductions that fail, with a typed reason.

4. **Build cross-problem queries.** "Find problems where concentration
   setup is complete but a structural gap blocks closure" — this is
   the query that would have saved six Problem 6 handoff documents.

## Evidence

- Problem 6 generated 6+ handoff/dispatch files across multiple sessions
  without honest layer-level status reporting
- The writeup itself (Sections 1-5) already implicitly contains the
  multi-layer structure — it just isn't indexed that way
- Problems 1-5, 7-10 also have implicit layer structure in their
  proof strategies; extracting it retroactively would test the framework
