# Mission: P8 Rational Reconstruction

**Date:** 2026-03-07
**Status:** VERIFY
**Owner:** futon6, with pattern discipline from futon3
**Depends on:** First-proof P8 material (complete), M-P7-rational-reconstruction
  (complete — method validated on branching proofs), math-strategy/ patterns
  (6 exist from P3+P7)
**Enables:** Validation of math-strategy patterns on symplectic geometry,
  hypothesis-category-check pattern (new), input to M-distributed-frontiermath

## 1. IDENTIFY

### Motivation

Problem 8 (does a polyhedral Lagrangian surface in R^4 with 4 faces per
vertex admit a Lagrangian smoothing?) has a complete solution
(`problem8-solution.md`), a reviewer assessment (REVIEWER.md: 2 critical,
1 major finding), a wiring diagram (10 nodes, 12 edges), stepper runs,
and Codex repair cycles. The current solution.md incorporates repairs
(vertex spanning lemma, edge crease smoothing, global Hamiltonian
argument) but the rational reconstruction works with the decision trace:
*why* were those moves made, and would pattern discipline have caught
the gaps before the reviewer did?

P8 tests the pattern library in a new domain (symplectic geometry) with
a different proof architecture from P3 (linear) and P7 (branching):

1. **Linear chain with a structural kernel.** The proof is mostly
   sequential (setup → local structure → key decomposition → consequences
   → smoothing → assembly) but step s3 (symplectic direct sum) is the
   structural heart — everything else follows from it. This is a
   "kernel proof" where one key insight (V1 + V2 decomposition) drives
   the rest.

2. **Category mismatch.** The most distinctive failure mode: invoking
   smooth symplectic results (Polterovich, Lalonde-Sikorav) on polyhedral
   objects. None of the existing 6 patterns specifically address "your
   objects don't live in the category the theorem requires." This needs
   a new pattern.

3. **Local-to-global assembly.** The proof does local smoothings at
   vertices and edges, then needs to compose them globally. This is a
   classic local-to-global move — but the original proof under-specified
   the compatibility and Hamiltonian isotopy arguments.

4. **Defensive branch.** Step s7 (3-face impossibility) is a challenge
   edge — explaining why 4 is the right valence — analogous to P7's
   Smith theory anti-obstruction.

### Theoretical anchoring

Same as P7 reconstruction:
- **Draft provenance standard** — PUR carries scope, graph is system of record
- **Rational reconstruction** — replay with discipline, record as evidence layers
- **Baldwin cycle** — P3/P7 patterns are Phase 1-2 (explored, assimilated);
  P8 tests Phase 3 (do they canalize to a third domain?)

Additional for P8:
- **Category discipline** — the finding that smooth results were applied
  to PL objects is a recurring AI proof failure. A category-check pattern
  would be reusable across many domains (algebraic geometry: scheme vs
  variety; topology: smooth vs topological manifold; analysis: continuous
  vs measurable).

### Scope

**In scope:**
- Replay P8 proof decomposition with virtual PSR/PUR
- Test whether existing 6 math-strategy/ patterns apply
- Write new pattern for hypothesis-category mismatch
- Record mathematical scope annotations on PURs
- Assess whether 3 reviewer findings would have been caught
- Assess P3/P7 pattern transferability to symplectic geometry

**Out of scope:**
- Replaying P9, P2, or P10 (follow-on missions)
- Building automated pattern suggestion tooling
- Actually rewriting the proof (repairs already done)

### Completion criteria

1. Every decision point in the P8 proof has a virtual PSR
2. Every proof step has a virtual PUR with mathematical scope
3. At least 1 new math-strategy/ pattern written (hypothesis-category-check)
4. Assessment of P3/P7 pattern transferability
5. Comparison table: original decomposition vs pattern-guided
6. Counterfactual assessment: would patterns have caught 3/3 findings?
7. Provenance JSON artifact (extends problem8-wiring.json)

### Source material

- `futon6/data/first-proof/problem8-solution.md` — the proof (9 sections)
- `futon6/data/first-proof/problem8-wiring.json` — wiring diagram (10 nodes, 12 edges)
- `futon6/data/first-proof/problem8-v1.mmd`, `problem8-v2.mmd`, `problem8-v3.mmd` — Mermaid diagrams
- `futon6/REVIEWER.md` §Problem 8 Findings — 3 findings (2 critical, 1 major)
- `futon3/library/math-strategy/` — 6 patterns from P3+P7
- `futon3/library/math-informal/` — 31 content patterns
- `futon6/holes/missions/M-P7-rational-reconstruction.md` — validated method

### Relationship to other missions

- **M-P7-rational-reconstruction** — direct predecessor. Tests whether
  the method and patterns transfer to symplectic geometry.
- **M-P3-rational-reconstruction** — grandfather mission. The original
  method validation.
- **M-distributed-frontiermath** — P8 provides another data point for
  which patterns govern AI proof construction.

## 2. MAP

### Q1: Decision points in the P8 proof

| Node | Section | Decision | What was chosen |
|------|---------|----------|-----------------|
| p8-problem | — | Problem framing | 4-face polyhedral Lagrangian → smoothing? |
| p8-s1 | §1 | Setup | Lagrangian Grassmannian Lambda(2), Maslov class, face/edge/vertex structure |
| p8-s2 | §2 | Local constraints | Edge-sharing kills 4/6 omega entries → only a=omega(e1,e3), b=omega(e2,e4) remain |
| p8-s3 | §3 | KEY: Symplectic direct sum | V1=span(e1,e3) + V2=span(e2,e4), block diagonal omega. Vertex spanning lemma. |
| p8-s4 | §4 | Maslov index | Decompose mu = mu1 + mu2, both 0 (back-and-forth, not winding) |
| p8-s4a | §5a | Vertex smoothing | Product structure K = C1 x C2 near v, smooth corners independently |
| p8-s5 | §6 | Edge crease smoothing | Generating function interpolation S_eps |
| p8-s6 | §7 | Global assembly | Disjoint supports → compose Hamiltonian isotopies (McDuff-Salamon 3.17) |
| p8-s7 | §5 | 3-face impossibility | Isotropic dimension bound: 3 isotropic vectors can't span 3D in R^4 |
| p8-c1 | — | Numerical validation | 998/998 valid configs → mu=0; without edge-sharing only 55% |

### Q2: Existing patterns that apply

**From math-informal/ (content patterns):**

| Decision point | Candidate pattern | Fit |
|----------------|-------------------|-----|
| p8-s1 (setup) | `unfold-the-definition` | Strong: define Lambda(2), Maslov class, polyhedral structure |
| p8-s2 (local constraints) | `exploit-symmetry` | Medium: cyclic adjacency kills omega entries |
| p8-s3 (direct sum) | `construct-auxiliary-object` | Strong: the V1+V2 decomposition IS the auxiliary object |
| p8-s4 (Maslov) | `reduce-to-known-result` | Medium: Maslov index decomposes via direct sum |
| p8-s4a (vertex smoothing) | `construct-an-explicit-witness` | Strong: product smoothing is the explicit construction |
| p8-s5 (edge smoothing) | `reduce-to-known-result` | Medium: generating function gives Lagrangian graph |
| p8-s6 (global) | `local-to-global` | Strong: assemble local smoothings globally |
| p8-s7 (3-face) | `argue-by-contradiction` | Strong: suppose 3D isotropic → contradiction |
| p8-c1 (numerical) | `numerical-scout` | Strong: sanity check, not theorem substitute |

**From math-strategy/ (P3+P7 patterns):**

| Decision point | Strategy pattern | Applies? |
|----------------|-----------------|----------|
| p8-s3 (vertex spanning claim) | `convention-bridge` | **Yes** — "guaranteed by topological submanifold condition" bridges from topology to linear algebra. The bridge needs proof. |
| p8-s5 (surgery invocation) | — | **Gap** — applying smooth surgery results to polyhedral objects is a category mismatch, not covered by existing patterns |
| p8-s6 (global composition) | `compose-independent-lemmas` | **Yes** — vertex smoothings + edge smoothings must compose. Independence = disjoint support. |
| p8-s7 (3-face impossibility) | `preemptive-objection-clearance` | **Partial** — this is more of a structural observation than an objection clearance. It explains *why* 4 is special, not "this doesn't obstruct." |

**Gaps (no existing pattern fits well):**

- **p8-s4a/s5 → s6 (category mismatch)**: The proof invokes
  Polterovich/Lalonde-Sikorav results (stated for smooth transverse
  Lagrangian intersections) on polyhedral objects with creases. No
  existing pattern governs "verify your objects live in the category
  the theorem requires." Candidate:
  `math-strategy/hypothesis-category-check`.

### Q3: Wiring diagram dependency structure

```
p8-problem ← p8-s1 (clarify: Lagrangian Grassmannian setup)
                ↓
              p8-s2 (reform: edge-sharing kills omega entries)
                ↓
              p8-s3 (assert: symplectic direct sum V1+V2)
              / | \
            /   |   \
  p8-s4   p8-s4a  p8-s7
  (Maslov=0)  (vertex   (3-face
              smoothing)  impossible)
            \   |
              \ |
              p8-s5 (edge crease smoothing; refs s4a, s4)
                ↓
              p8-s6 (global smoothing; refs s5, asserts to problem)
```

The proof is a **kernel proof**: s3 (symplectic direct sum) is the
structural heart. Everything downstream — Maslov vanishing (s4),
vertex smoothing (s4a), 3-face impossibility (s7) — flows from the
V1+V2 decomposition. The assembly (s5→s6) is the local-to-global move.

### Q4: Reviewer findings → decision points

| Finding | Severity | Decision point | Pattern gap |
|---------|----------|----------------|-------------|
| 1. Basis/nondegeneracy unjustified | Critical | p8-s3 | `convention-bridge` applies: "topological submanifold" ≠ "spanning in R^4" is exactly a convention bridge gap |
| 2. Surgery outside smooth hypotheses | Critical | p8-s5 (edge smoothing) | **New pattern needed**: `hypothesis-category-check` — smooth results applied to PL objects |
| 3. Global patching not established | Major | p8-s6 | `local-to-global` + `compose-independent-lemmas`: overlap compatibility and Hamiltonian composition need proof |

### Ready vs missing

**Ready:**
- P8 proof content: 9 sections, 10 nodes, 12 edges
- Existing patterns: 31 math-informal/ + 6 math-strategy/
- Reviewer findings: 3 findings mapped to decision points
- Wiring diagram: full dependency graph
- Validated method from M-P3 and M-P7

**Missing (the actual work):**
- 1 new pattern: `hypothesis-category-check`
- Virtual PSR for each of 9 decision points (excluding input + comment)
- Virtual PUR with mathematical scope for each step
- Assessment of P3/P7 transferability
- Comparison table
- Counterfactual assessment: would patterns have caught 3/3 findings?

## 3. DERIVE

### 3.1 New pattern written

One new pattern for `futon3/library/math-strategy/`:

| Pattern | Addresses |
|---------|-----------|
| `math-strategy/hypothesis-category-check` | p8-s5: smooth surgery invoked on polyhedral objects |

Combined with the 6 existing math-strategy/ patterns, the namespace
now has 7 patterns.

**Pattern definition:**

```
@flexiarg math-strategy/hypothesis-category-check
@title Hypothesis Category Check
@keywords category, smooth, PL, topological, algebraic, scheme, variety, hypothesis, mismatch, regularity
@audience mathematicians, proof-writers
@tone heuristic
@factor Discernment

! conclusion:
  When invoking a theorem, verify that your objects live in the category
  (smooth, PL, topological, algebraic, measurable, ...) that the theorem
  is stated for.

  + context: You want to apply a known result to objects in your proof,
    and the result's statement specifies regularity or categorical
    hypotheses on its inputs.

  + IF:
      The theorem you want to invoke is stated for objects in category C
      (e.g., smooth manifolds, algebraic varieties, measurable functions),
      and your objects live in a different category C' (e.g., PL complexes,
      schemes, continuous functions).

  + HOWEVER:
      Category mismatches are a common source of silent errors in proofs
      that cross domain boundaries. A theorem about smooth Lagrangian
      intersections does not apply to polyhedral creases. A result about
      algebraic varieties may fail for general schemes. The mismatch may
      be bridgeable (via a smoothing lemma, GAGA, regularity bootstrap)
      but the bridge itself requires proof.

  + THEN:
      (a) State the category of the theorem's hypotheses explicitly.
      (b) State the category of your objects explicitly.
      (c) If they differ, provide a bridge: a smoothing lemma, a
          regularity result, a comparison theorem, or a specialization
          argument that places your objects in the required category.
      (d) If no bridge exists, the theorem does not apply — find an
          alternative or prove a new version in your category.

  + BECAUSE:
      Cross-category theorem application is the second most common gap
      in AI-generated proofs (after convention bridges). The fix is
      mechanical: check the fine print. The cost of checking is minutes;
      the cost of not checking is a critical reviewer finding.
```

### 3.2 Pattern assignment per node

| Step | Node | Pattern | Rationale |
|------|------|---------|-----------|
| 1 | p8-problem | (input) | Problem framing, no PSR |
| 2 | p8-s1 | `math-informal/unfold-the-definition` | Define Lambda(2), Maslov class, face/edge/vertex |
| 3 | p8-s2 | `math-informal/exploit-symmetry` | Cyclic adjacency + Lagrangian condition kills omega entries |
| 4 | p8-s3 | `math-informal/construct-auxiliary-object` + `math-strategy/convention-bridge` | Build V1+V2 decomposition; bridge "manifold condition" → "spanning" |
| 5 | p8-s4 | `math-informal/reduce-to-known-result` | Maslov decomposes via direct sum; each factor has winding 0 |
| 6 | p8-s4a | `math-informal/construct-an-explicit-witness` | Product smoothing: K = C1 x C2, smooth each corner |
| 7 | p8-s5 | `math-informal/reduce-to-known-result` + `math-strategy/hypothesis-category-check` | Generating function gives Lagrangian graph; CHECK: smooth results applied to PL objects |
| 8 | p8-s6 | `math-informal/local-to-global` + `math-strategy/compose-independent-lemmas` | Assemble local smoothings; disjoint supports → Hamiltonian composition |
| 9 | p8-s7 | `math-informal/argue-by-contradiction` | Suppose 3-face vertex in R^4 → isotropic dim bound → contradiction |
| — | p8-c1 | `math-informal/numerical-scout` | Sanity check: 998/998 → mu=0 |

Note: p8-s3, p8-s5, and p8-s6 each require two patterns (one content,
one strategy). The strategy patterns address exactly the three reviewer
findings.

### 3.3 Virtual PSR/PUR Replay

#### vPSR-1: p8-s1 — Setup

- **Step**: p8-s1 (§1 — Lagrangian planes in R^4)
- **Pattern chosen**: `math-informal/unfold-the-definition`
- **Candidates considered**: `unfold-the-definition` (primary),
  `find-the-right-abstraction` (rejected: we're not abstracting,
  we're laying down concrete definitions)
- **Rationale**: The first move is to establish the stage: symplectic
  R^4, Lagrangian Grassmannian Lambda(2) = U(2)/O(2), pi_1 = Z
  (Maslov class), face/edge/vertex structure of a polyhedral Lagrangian.
- **Scope**: p8-s1
- **Confidence**: high

#### vPUR-1: p8-s1

- **Step**: p8-s1
- **Pattern**: `math-informal/unfold-the-definition`
- **Scope**: p8-s1
- **Actions taken**: Defined symplectic form omega = dx1 ^ dy1 + dx2 ^ dy2.
  Lagrangian plane: 2-dim subspace L with omega|_L = 0.
  Lagrangian Grassmannian Lambda(2) = U(2)/O(2), dim 3, pi_1 = Z.
  Polyhedral Lagrangian: faces are Lagrangian planes, edges are creases,
  vertices are multi-plane singularities.
- **Outcome**: success — all definitions in place
- **Prediction error**: none
- **Gap detected**: none (this step is clean)

#### vPSR-2: p8-s2 — Local constraints

- **Step**: p8-s2 (§2 — edge-sharing structure at 4-valent vertex)
- **Pattern chosen**: `math-informal/exploit-symmetry`
- **Candidates considered**: `exploit-symmetry` (the cyclic adjacency
  IS a symmetry that constrains omega), `unfold-the-definition`
  (rejected: we're not just unfolding, we're deriving consequences)
- **Rationale**: Four faces L_1,...,L_4 in cyclic order. Each face
  L_i = span(e_{i-1,i}, e_{i,i+1}). Lagrangian condition
  omega(e_{i-1,i}, e_{i,i+1}) = 0 for each face. This cyclic
  constraint kills 4 of 6 independent omega entries, leaving only
  a = omega(e1,e3) and b = omega(e2,e4).
- **Scope**: p8-s2
- **Confidence**: high

#### vPUR-2: p8-s2

- **Step**: p8-s2
- **Pattern**: `math-informal/exploit-symmetry`
- **Scope**: p8-s2
- **Actions taken**: Wrote omega matrix in basis (e1,e2,e3,e4).
  Cyclic adjacency + Lagrangian → omega(e_i, e_{i+1}) = 0 for all
  i (mod 4). Kills 4 entries, leaves a and b.
- **Outcome**: success — constraint structure clear
- **Prediction error**: none
- **Gap detected**: none

#### vPSR-3: p8-s3 — Symplectic direct sum (KEY STEP)

- **Step**: p8-s3 (§3 — R^4 = V1 + V2, vertex spanning lemma)
- **Pattern chosen**: `math-informal/construct-auxiliary-object` +
  `math-strategy/convention-bridge`
- **Candidates considered**: `construct-auxiliary-object` (primary:
  V1+V2 is the auxiliary structure that makes everything else work),
  `convention-bridge` (secondary: the claim that {e1,...,e4} spans R^4
  bridges from "topological submanifold condition" to "linear algebra
  spanning" — this bridge needs proof),
  `find-the-right-abstraction` (rejected: the decomposition is concrete,
  not abstract)
- **Rationale**: The omega matrix in basis (e1,e3,e2,e4) is block
  diagonal. This gives V1 = span(e1,e3), V2 = span(e2,e4), a
  symplectic direct sum. BUT this requires {e1,...,e4} to be a basis
  of R^4. The original v1 claimed this was "guaranteed by topological
  submanifold condition" — a convention bridge assertion without proof.
- **Scope**: p8-s3
- **Confidence**: medium (the decomposition is clear IF spanning holds;
  spanning itself needs careful argument)

#### vPUR-3: p8-s3

- **Step**: p8-s3
- **Pattern**: `construct-auxiliary-object` + `convention-bridge`
- **Scope**: p8-s3
- **Actions taken**: Constructed V1 = span(e1,e3), V2 = span(e2,e4).
  Showed omega is block diagonal in (e1,e3,e2,e4) basis. Asserted
  {e1,...,e4} spans R^4.
- **Outcome**: **partial** — decomposition is correct IF spanning holds
- **Prediction error**: **significant.** The `convention-bridge`
  pattern's HOWEVER clause warns: "what looks like 'the same thing'
  may hide a genuine mathematical difference." Here, "topological
  submanifold" and "edge vectors span R^4" are genuinely different
  properties. A 2-manifold embedded in R^4 need not have edge
  directions spanning all 4 dimensions.
- **Gap detected**: **Reviewer Finding 1 (critical).** The convention
  bridge from "topological submanifold condition" to "spanning" is
  exactly the gap. The `convention-bridge` pattern demands: "prove
  the relationship holds in your parameter regime." The v2 repair
  (vertex spanning lemma, proved algebraically using isotropic
  dimension bounds) is exactly what the pattern would have required
  from the start.
- **Reviewer alignment**: STRONG — `convention-bridge` catches Finding 1.

#### vPSR-4: p8-s4 — Maslov index vanishes

- **Step**: p8-s4 (§4 — Maslov = 0)
- **Pattern chosen**: `math-informal/reduce-to-known-result`
- **Candidates considered**: `reduce-to-known-result` (Maslov
  additivity under symplectic direct sum is a known property),
  `split-into-cases` (rejected: it's a decomposition, not a case split)
- **Rationale**: The Maslov index decomposes: mu = mu1 + mu2 via the
  V1+V2 direct sum. Each mu_j traces a back-and-forth path in RP^1,
  giving winding number 0. Total: 0+0 = 0.
- **Scope**: p8-s4
- **Confidence**: high (clean algebraic argument)

#### vPUR-4: p8-s4

- **Step**: p8-s4
- **Pattern**: `math-informal/reduce-to-known-result`
- **Scope**: p8-s4
- **Actions taken**: Applied Maslov index additivity under symplectic
  direct sum. Traced each factor's loop: back-and-forth, not winding.
  Both mu_j = 0.
- **Outcome**: success — Maslov vanishing proved algebraically
- **Prediction error**: none
- **Gap detected**: none (this step is clean; depends on s3 being
  correct, which it is once spanning is proved)

#### vPSR-5: p8-s4a — Vertex smoothing

- **Step**: p8-s4a (§5a — product smoothing at vertices)
- **Pattern chosen**: `math-informal/construct-an-explicit-witness`
- **Candidates considered**: `construct-an-explicit-witness` (primary:
  the product smoothing C1^sm x C2^sm is the explicit construction),
  `exploit-symmetry` (partial: the product structure IS a symmetry,
  but the pattern is about using symmetry to simplify, not to construct)
- **Rationale**: The V1+V2 decomposition gives K near v = C1 x C2
  (product of corners). Replace each corner with a smooth curve.
  Product of curves in symplectic factors is automatically Lagrangian
  (omega restricted to 1-dim = 0). Smooth immersion because tangent
  vectors in complementary V_j.
- **Scope**: p8-s4a
- **Confidence**: high (explicit, checkable construction)

#### vPUR-5: p8-s4a

- **Step**: p8-s4a
- **Pattern**: `math-informal/construct-an-explicit-witness`
- **Scope**: p8-s4a
- **Actions taken**: Constructed C1^sm, C2^sm as smooth roundings of
  the corner curves. Verified: product is Lagrangian (dim argument),
  smooth (complementary tangent vectors), agrees with K outside
  delta-neighborhood.
- **Outcome**: success — vertex smoothing constructed and verified
- **Prediction error**: none
- **Gap detected**: none (this step is clean; the product structure
  makes the Lagrangian property automatic)

#### vPSR-6: p8-s5 — Edge crease smoothing

- **Step**: p8-s5 (§6 — generating function interpolation)
- **Pattern chosen**: `math-informal/reduce-to-known-result` +
  `math-strategy/hypothesis-category-check`
- **Candidates considered**: `reduce-to-known-result` (primary:
  generating function interpolation is a standard technique),
  `hypothesis-category-check` (secondary: the Polterovich/Lalonde-Sikorav
  results cited for surgery/smoothing are stated for smooth transverse
  Lagrangian intersections),
  `construct-an-explicit-witness` (rejected: it's a reduction to a
  technique, not a witness construction)
- **Rationale**: The edge crease smoothing uses the generating function
  S_eps = chi(x1/eps) S1 + (1-chi) S2, whose graph y = grad S_eps is
  automatically Lagrangian (graph of exact 1-form). This part is clean.
  However, the v1 proof cited Polterovich/Lalonde-Sikorav for the
  smoothing lemma, and those results require smooth transverse
  intersections — not polyhedral creases.
- **Scope**: p8-s5
- **Confidence**: medium (the generating function argument is clean
  but the original theorem invocation was problematic)

#### vPUR-6: p8-s5

- **Step**: p8-s5
- **Pattern**: `reduce-to-known-result` + `hypothesis-category-check`
- **Scope**: p8-s5
- **Actions taken**: Applied generating function interpolation for
  edge crease smoothing. Also invoked Polterovich/Lalonde-Sikorav for
  the surgery/smoothing context.
- **Outcome**: **partial** — the generating function argument is
  self-contained and correct, but the citation of smooth surgery
  results is problematic
- **Prediction error**: **significant.** The `hypothesis-category-check`
  pattern asks: "state the category of the theorem's hypotheses (smooth
  transverse Lagrangian intersections) and the category of your objects
  (polyhedral creases). They differ. Where is the bridge?"
  The v1 proof asserted the bridge ("the polyhedral model produces a
  smooth transverse setup") without proving it. The v2 repair
  restructured the argument to use the generating function directly,
  bypassing the need for smooth surgery results — but the original gap
  was real.
- **Gap detected**: **Reviewer Finding 2 (critical).** The
  `hypothesis-category-check` pattern catches this directly: smooth
  symplectic results do not apply to polyhedral objects without a
  smoothing-to-immersion lemma. The pattern demands either (a) proving
  the bridge, or (b) finding an alternative argument in the correct
  category. The v2 chose (b).
- **Reviewer alignment**: STRONG — pattern catches Finding 2.

#### vPSR-7: p8-s6 — Global assembly

- **Step**: p8-s6 (§7 — compose vertex + edge smoothings globally)
- **Pattern chosen**: `math-informal/local-to-global` +
  `math-strategy/compose-independent-lemmas`
- **Candidates considered**: `local-to-global` (primary: assemble
  local smoothings into a global one), `compose-independent-lemmas`
  (secondary: vertex and edge smoothings are independent pieces
  that must compose), `reduce-to-known-result` (rejected: the global
  composition is not a simple theorem invocation — it requires a
  support-control argument)
- **Rationale**: Vertex smoothings (in disjoint balls B_i) and edge
  smoothings (in disjoint tubular neighborhoods) must compose to a
  global Hamiltonian isotopy. The `local-to-global` pattern demands:
  "check compatibility on overlaps." The `compose-independent-lemmas`
  pattern demands: "verify independence — is each piece self-contained?"
- **Scope**: p8-s6
- **Confidence**: medium (the local pieces are clean; the assembly
  needs explicit support-control)

#### vPUR-7: p8-s6

- **Step**: p8-s6
- **Pattern**: `local-to-global` + `compose-independent-lemmas`
- **Scope**: p8-s6
- **Actions taken**: Stated: vertex smoothings in disjoint balls
  (commute). Edge smoothings in disjoint tubular neighborhoods
  (commute with each other and with vertex balls). Composition is
  Hamiltonian (McDuff-Salamon Prop 3.17).
- **Outcome**: **partial** — the disjoint-support argument is
  sketched but not fully verified in v1
- **Prediction error**: **moderate.** The `local-to-global` pattern's
  HOWEVER clause warns: "not every property is local. You must verify
  that local data can be patched: the pieces must be compatible on
  overlaps." The v1 proof summarized the compatibility but did not
  demonstrate it. The `compose-independent-lemmas` pattern asks: "is
  each piece self-contained?" — which requires checking that edge
  smoothings don't interfere with vertex smoothings at the boundaries
  of vertex balls.
- **Gap detected**: **Reviewer Finding 3 (major).** The `local-to-global`
  pattern catches this: the compatibility at ∂B_i (where vertex
  smoothing meets edge smoothing) needs an explicit argument. The
  `compose-independent-lemmas` pattern reinforces: "independence must
  be genuine, not assumed." The v2 repair added Lemma 7.1 (disjoint
  Hamiltonian supports → commutative composition) and explicit
  compatibility at ∂B_i.
- **Reviewer alignment**: STRONG — both patterns catch Finding 3.

#### vPSR-8: p8-s7 — 3-face impossibility

- **Step**: p8-s7 (§5 — why 3-face doesn't work)
- **Pattern chosen**: `math-informal/argue-by-contradiction`
- **Candidates considered**: `argue-by-contradiction` (primary:
  suppose 3-face vertex exists → isotropic dim bound → contradiction),
  `preemptive-objection-clearance` (partial: this explains why 4 is
  special, which is somewhat defensive, but the argument is a genuine
  impossibility proof, not just "this doesn't obstruct")
- **Rationale**: 3 edges with omega(e_i, e_j) = 0 for all pairs →
  span is isotropic. But max isotropic dim in R^4 is 2. Three
  independent isotropic vectors can't exist. Contradiction.
- **Scope**: p8-s7
- **Confidence**: high (clean linear algebra)

#### vPUR-8: p8-s7

- **Step**: p8-s7
- **Pattern**: `math-informal/argue-by-contradiction`
- **Scope**: p8-s7
- **Actions taken**: Stated: 3-face vertex → all omega(e_i,e_j) = 0
  → span is isotropic → dim ≤ 2 in R^4 → can't have 3 independent
  edges. Contradiction.
- **Outcome**: success — impossibility clean
- **Prediction error**: none
- **Gap detected**: none (this step is clean; it's a structural
  observation, not where the reviewer found problems)

#### vPSR-9: p8-c1 — Numerical validation

- **Step**: p8-c1 (numerical verification)
- **Pattern chosen**: `math-informal/numerical-scout`
- **Candidates considered**: `numerical-scout` (only candidate)
- **Rationale**: 998/998 random valid 4-valent configurations give
  mu = 0. Without edge-sharing: only 55%. This is a sanity check,
  not a theorem substitute.
- **Scope**: p8-c1
- **Confidence**: high (supplementary evidence)

#### vPUR-9: p8-c1

- **Step**: p8-c1
- **Pattern**: `math-informal/numerical-scout`
- **Scope**: p8-c1
- **Actions taken**: Ran verify-p8-maslov-v2.py on 998 random configs.
  All give mu = 0. Comparison: without edge-sharing, only 55% give
  mu = 0.
- **Outcome**: success — numerical evidence supports algebraic proof
- **Prediction error**: none
- **Gap detected**: none (the `numerical-scout` pattern correctly
  flags this as supplementary, not primary evidence)

### 3.4 Counterfactual assessment

| Finding | Severity | Node | Pattern that catches it | How |
|---------|----------|------|------------------------|-----|
| 1. Basis/nondegeneracy unjustified | Critical | p8-s3 | `convention-bridge` | HOWEVER warns: "same thing with different notation may hide genuine difference." "Topological submanifold" ≠ "spanning." THEN demands: prove the bridge in your setting. |
| 2. Surgery outside smooth hypotheses | Critical | p8-s5 | `hypothesis-category-check` | Pattern demands: state theorem's category (smooth), state your objects' category (PL), provide bridge or find alternative. |
| 3. Global patching not established | Major | p8-s6 | `local-to-global` + `compose-independent-lemmas` | `local-to-global` HOWEVER: "verify pieces compatible on overlaps." `compose-independent-lemmas` HOWEVER: "independence must be genuine." Both demand explicit support-control argument. |

**Result: 3/3 findings caught** (all strong alignment).

This is the cleanest result across the three reconstructions:
- P3: 3/3 caught (validated method)
- P7: 4/4 caught (3 strong, 1 moderate)
- P8: 3/3 caught (all strong)

The new `hypothesis-category-check` pattern is the key addition for P8.
Without it, Finding 2 would have been caught only weakly by
`reduce-to-known-result` ("verify all hypotheses") — the category
mismatch is a specific, named failure mode that deserves its own pattern.

### 3.5 Pattern transferability assessment

**P3 patterns → P8:**

| P3 pattern | Used in P8? | Transfer quality |
|------------|-------------|-----------------|
| `convention-bridge` | Yes (p8-s3) | **Strong** — "topological submanifold → spanning" is exactly a convention bridge. Catches Finding 1. Third successful deployment (P3, P7, P8). |
| `non-circularity-check` | No | Not applicable — P8 has no construction-independence concern. |
| `compose-independent-lemmas` | Yes (p8-s6) | **Strong** — disjoint-support composition maps perfectly. Third successful deployment. |

**P7 patterns → P8:**

| P7 pattern | Used in P8? | Transfer quality |
|------------|-------------|-----------------|
| `route-exploration-and-pivot` | No | Not applicable — P8 is a linear proof, no multi-approach exploration. |
| `constraint-tension-resolution` | No | Not applicable — no parameter tension in P8. |
| `preemptive-objection-clearance` | Partial (p8-s7) | Weak — the 3-face impossibility is a structural observation, not an objection clearance. Better matched by `argue-by-contradiction`. |

**Transfer summary:** P3 patterns are highly reusable (2/3 transfer to
both P7 and P8). P7 patterns are specific to multi-approach/branching
proofs (0/3 transfer to P8's linear proof). This confirms that the
pattern library has two layers:

1. **Universal math-strategy patterns** (convention-bridge,
   compose-independent-lemmas): apply across proof architectures
2. **Architecture-specific patterns** (route-exploration-and-pivot,
   constraint-tension-resolution, preemptive-objection-clearance):
   apply to specific proof shapes

The new P8 pattern (hypothesis-category-check) is likely universal —
category mismatches can occur in any proof that invokes external results.

### 3.6 Comparison: original vs pattern-guided

| Aspect | Original (ad hoc) | Pattern-guided (replay) |
|--------|-------------------|------------------------|
| Decomposition | 10 nodes, kernel proof (s3 central) | Same structure — patterns don't change the decomposition |
| Vertex spanning (s3) | v1: "guaranteed by submanifold condition" | `convention-bridge` demands: prove bridge. **Gap caught → vertex spanning lemma** |
| Direct sum (s3) | Clean once spanning holds | `construct-auxiliary-object` confirms clean |
| Maslov (s4) | Clean | `reduce-to-known-result` confirms clean |
| Vertex smoothing (s4a) | Clean | `construct-an-explicit-witness` confirms clean |
| Edge smoothing (s5) | v1: cited smooth surgery on PL objects | `hypothesis-category-check` demands: state categories, provide bridge. **Gap caught → generating function restructure** |
| Global assembly (s6) | v1: "compose... compatible at boundaries" | `local-to-global` + `compose-independent-lemmas` demand: explicit support-control, overlap compatibility. **Gap caught → Lemma 7.1** |
| 3-face (s7) | Clean | `argue-by-contradiction` confirms clean |
| Numerical (c1) | Clean | `numerical-scout` confirms: supplementary only |
| Reviewer findings caught | 0/3 (found by external reviewer) | **3/3 (found by pattern discipline)** |

### 3.7 Cross-problem pattern usage matrix

| Pattern | P3 | P7 | P8 | Universal? |
|---------|----|----|----|----|
| `convention-bridge` | s4 (star/non-star) | s2 (ordinary/Bredon PD) | s3 (submanifold/spanning) | **Yes** |
| `non-circularity-check` | s1 (CTMC construction) | — | — | P3-specific |
| `compose-independent-lemmas` | s6 (combine pieces) | s6 (conditional conclusion) | s6 (global assembly) | **Yes** |
| `route-exploration-and-pivot` | — | s4 (4 approaches) | — | Multi-approach only |
| `constraint-tension-resolution` | — | s4 (parity tension) | — | Multi-approach only |
| `preemptive-objection-clearance` | — | s5 (Smith theory) | — | Defensive proofs only |
| `hypothesis-category-check` | — | — | s5 (smooth on PL) | **Yes** (predicted) |

The three "universal" patterns (`convention-bridge`,
`compose-independent-lemmas`, `hypothesis-category-check`) form a
**core triad** for AI proof validation:
1. Are conventions properly bridged?
2. Are pieces properly composed?
3. Are theorems applied in the right category?

These three patterns alone would catch 8/10 reviewer findings across
P3+P7+P8.

## 4. ARGUE

The central finding is that the pattern library now divides into
**universal** and **architecture-specific** patterns:

- **Universal** (convention-bridge, compose-independent-lemmas,
  hypothesis-category-check): caught findings across all three problems,
  across three different mathematical domains (combinatorics/probability,
  algebraic topology, symplectic geometry), across three different proof
  architectures (linear, branching, kernel).

- **Architecture-specific** (route-exploration-and-pivot,
  constraint-tension-resolution, preemptive-objection-clearance,
  non-circularity-check): each pattern catches specific failure modes
  tied to specific proof shapes.

IF: we want a minimal pattern discipline for AI proof generation
HOWEVER: 7 patterns is already a lot to check at every proof step
THEN: prioritize the universal triad as mandatory checks; apply
  architecture-specific patterns only when the proof shape triggers them
BECAUSE: the universal triad catches 80% of findings with 3/7 of the
  patterns, and the triggering conditions for architecture-specific
  patterns are identifiable from proof structure (branching → route
  patterns; conditional → composition patterns; defensive → clearance)

## 5. VERIFY

Full replay completed: 9 vPSRs + 9 vPURs covering all non-trivial nodes.
Counterfactual: 3/3 findings caught, all with strong alignment.

The pattern discipline is validated across three problems and three
mathematical domains. The universal triad emerges as the key finding.

## 6. INSTANTIATE

Artifacts produced:
- `futon3/library/math-strategy/hypothesis-category-check.flexiarg` (to commit)
- `futon6/holes/missions/M-P8-rational-reconstruction.md` (this file)
- `futon6/data/first-proof/problem8-provenance.json` (TODO: extend wiring with vPSR/vPUR layers)

### Completion criteria check

1. ✓ Every decision point has a virtual PSR (9 vPSRs for non-input nodes)
2. ✓ Every proof step has a virtual PUR with mathematical scope
3. ✓ 1 new math-strategy/ pattern written (hypothesis-category-check)
4. ✓ P3/P7 pattern transferability assessed
5. ✓ Comparison table produced
6. ✓ 3/3 reviewer findings caught by pattern discipline
7. ○ Provenance JSON artifact (remaining TODO)

**Status: VERIFY complete, INSTANTIATE partial (pattern commit + provenance JSON pending).**
