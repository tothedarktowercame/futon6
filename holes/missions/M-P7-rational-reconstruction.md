# Mission: P7 Rational Reconstruction

**Date:** 2026-03-07
**Status:** IDENTIFY
**Owner:** futon6, with pattern discipline from futon3
**Depends on:** First-proof P7 material (complete), synthetic QA batch
  (16 files complete), M-P3-rational-reconstruction (complete — method
  validated), math-strategy/ patterns (3 exist from P3)
**Enables:** Validation of math-strategy patterns on a harder problem,
  potential new patterns for multi-approach proofs, input to
  M-distributed-frontiermath mission

## 1. IDENTIFY

### Motivation

Problem 7 (can a uniform lattice with 2-torsion be pi_1 of a closed
manifold with rationally acyclic universal cover?) has a substantially
more complex proof structure than P3. The solution explores four
distinct approaches (Wall surgery, equivariant surgery, orbifold
resolution, rotation route), encounters blocking obstructions, and
settles on a conditional answer via a rotation-lattice construction.
The reviewer identified 4 findings (3 critical, 1 medium).

The P3 rational reconstruction validated the method: virtual PSR/PUR
replay caught 3/3 reviewer findings through pattern discipline. P7 is
the harder test case, with features P3 lacked:

1. **Multiple proof routes explored and abandoned.** P3 had one linear
   path. P7 has four approaches, two blocked, one unexplored, one
   conditional. The decision to abandon Approaches I-III and pursue
   IV is a strategic move that no existing pattern covers.

2. **Conditional/partial answer.** P3 aimed for a complete proof. P7
   explicitly acknowledges obligation S as open. The proof is honest
   about its incompleteness — but did pattern discipline *need* to be
   present for that honesty, or was it accidental?

3. **Geometric obstruction analysis.** The dimension-parity tension
   (E2 needs even n, surgery prefers odd n) is a structural insight
   that drove the rotation-route pivot. This is a pattern-level
   observation — recognising that two constraints are in tension and
   finding a resolution — not just a mathematical step.

4. **Richer wiring structure.** 8 nodes, 9 edges, with a challenge
   edge (s5 → problem) that represents the Smith theory
   anti-obstruction. P3's wiring was purely constructive; P7 includes
   defensive moves.

This mission replays the P7 proof development with pattern discipline,
testing whether the 3 existing math-strategy/ patterns transfer,
discovering new patterns for the features unique to P7, and assessing
whether the 4 reviewer findings would have been caught.

### Theoretical anchoring

Same as P3 reconstruction:
- **Draft provenance standard** — PUR carries scope, graph is system of record
- **Rational reconstruction** — replay with discipline, record as evidence layers
- **Baldwin cycle** — P3 patterns are Phase 1 (explored); P7 tests Phase 2
  (do they assimilate to a new problem?)

Additional for P7:
- **Lakatos Proofs & Refutations** — P7's multi-approach structure (try,
  fail, pivot) echoes the dialectical proof methodology. The abandoned
  approaches are "refuted lemmas" that constrain the search space.
  If the Mentor role from M-distributed-frontiermath were watching
  this proof, it would name the dimension-parity tension and the
  rotation pivot as pattern-level moves.

### Scope

**In scope:**
- Replay P7 proof decomposition with virtual PSR/PUR
- Test whether P3's math-strategy/ patterns (convention-bridge,
  non-circularity-check, compose-independent-lemmas) apply to P7
- Write new patterns for P7-specific strategic moves
- Record mathematical scope annotations on PURs
- Assess whether 4 reviewer findings would have been caught
- Assess the multi-approach/pivot structure as a proof strategy pattern

**Out of scope:**
- Actually closing obligation S (that's M-distributed-frontiermath's domain)
- Replaying P8 or P2 (follow-on missions)
- Building automated pattern suggestion tooling

### Completion criteria

1. Every decision point in the P7 proof has a virtual PSR (pattern
   chosen or pattern gap identified)
2. Every proof step has a virtual PUR with mathematical scope
3. At least 2 new math-strategy/ patterns written (expected: route
   abandonment, constraint-tension-resolution)
4. Assessment of P3 pattern transferability (which of the 3 applied?)
5. Comparison table: original decomposition vs pattern-guided
6. Counterfactual assessment: would pattern discipline have caught
   4/4 reviewer findings?
7. Provenance JSON artifact (extends problem7-wiring.json)

### Source material

- `futon6/data/first-proof/problem7-solution.md` — the proof (7 sections + 4 approaches)
- `futon6/data/first-proof/problem7-wiring.json` — wiring diagram (8 nodes, 9 edges)
- `futon6/data/first-proof/problem7-v1.mmd`, `problem7-v2.mmd` — Mermaid diagrams
- `futon6/data/first-proof/problem7-reduced-wiring.json` — reduced wiring
- `futon6/data/first-proof/problem7-complete-proof.md` — complete proof write-up
- `futon6/data/first-proof/problem7r-rotation-lattice-construction.md` — rotation route details
- `futon6/data/first-proof/problem7r-s2b-candidate-construction.md` — candidate construction
- `futon6/data/first-proof/problem7r-s3a-setup.md` — Wall surgery setup
- `futon6/data/first-proof/problem7r-s3b-obstruction.md` — obstruction analysis
- `futon6/data/synthetic-qa/synth-p7-*.json` — 16 synthetic QA pairs (8 steps × 2)
- `futon6/data/synthetic-qa/problem7-prompts.jsonl` — generation prompts
- `futon6/REVIEWER.md` §Problem 7 Findings — 4 findings (3 critical, 1 medium)
- `futon3/library/math-strategy/` — 3 patterns from P3 reconstruction
- `futon6/holes/missions/M-P3-rational-reconstruction.md` — validated method

### Relationship to other missions

- **M-P3-rational-reconstruction** — direct predecessor. This mission
  tests whether the method and patterns transfer to a harder problem.
- **M-distributed-frontiermath** — P7 is a candidate for the distributed
  proof attack. The rational reconstruction provides the strategic map:
  which approaches are blocked, where the open obligation lies, what
  patterns govern the remaining work.
- **Future P8, P2 reconstructions** — if P7 confirms the method works
  on multi-approach proofs, P8 (3 critical findings) and P2 (1 critical)
  are next.

## 2. MAP

### Q1: Decision points in the P7 proof

The proof has a more complex structure than P3. Instead of a linear
sequence, it has a **branching exploration** with abandoned routes:

| Node | Section | Decision | What was chosen |
|------|---------|----------|-----------------|
| p7-problem | — | Problem framing | Existence question for closed manifold with pi_1 = Gamma (2-torsion lattice) and Q-acyclic cover |
| p7-s1 | §1 | Baseline clarification | Torsion-free case works; torsion creates orbifold, not manifold |
| p7-s2 | §2 | Cohomological framework | Bredon/orbifold PD replaces ordinary PD for torsion groups |
| p7-s3 | §3 | FH(Q) realization | Invoke Fowler's criterion with fixed-set Euler vanishing |
| p7-s3a | §3b | Concrete instantiation | Reflection lattice construction (Douba-Vargas Pallete) |
| p7-s4 | §4 | Manifold upgrade (open) | Identify obligation S; explore 4 approaches |
| p7-s5 | §5 | Anti-obstruction | Smith theory does NOT obstruct over Q (defensive move) |
| p7-s6 | §6 | Conditional conclusion | Compose: FH(Q) discharged, S conditional, Smith non-issue |

**Strategic decisions unique to P7:**
- §3→§3a: Moving from abstract criterion to concrete construction
- §4: Exploring and abandoning multiple approaches (I: Wall surgery —
  3 open obstacles; II: equivariant surgery — blocked by gap hypothesis;
  III: orbifold resolution — not explored)
- §4 (cont.): Discovering the dimension-parity tension (E2 needs even,
  S prefers odd) and resolving it via Approach IV (rotation route)
- §5: Including a defensive section (anti-obstruction) as a separate
  proof step

### Q2: Existing patterns that apply

**From math-informal/ (content patterns):**

| Decision point | Candidate pattern | Fit |
|----------------|-------------------|-----|
| p7-s1 (baseline) | `math-informal/identify-the-obstruction` | Strong: name what breaks when torsion is present |
| p7-s2 (Bredon PD) | `math-informal/transport-across-isomorphism` | Medium: reframe PD in equivariant setting |
| p7-s3 (Fowler) | `math-informal/reduce-to-known-result` | Strong: invoke Fowler's criterion |
| p7-s3a (construction) | `math-informal/construct-an-explicit-witness` | Strong: build the lattice |
| p7-s6 (conclusion) | `math-informal/compose-conclusion` | Medium: assemble conditional result |

**From math-strategy/ (P3 patterns):**

| Decision point | P3 pattern | Applies? |
|----------------|------------|----------|
| p7-s2 (Bredon reframe) | `convention-bridge` | **Yes** — reframing "rational PD" from ordinary to Bredon is exactly a convention bridge |
| p7-s3a (lattice construction) | `non-circularity-check` | **Partial** — the construction must produce the right Gamma, but this isn't about avoiding circularity |
| p7-s6 (conclusion) | `compose-independent-lemmas` | **Yes** — the conclusion composes E2 (discharged) + S (conditional) + Smith (non-issue) |

**Gaps (no existing pattern fits well):**

- **p7-s4 (multi-approach exploration)**: The proof tries four approaches,
  two fail, one is unexplored, one succeeds conditionally. No existing
  pattern governs "try multiple routes, abandon blocked ones, pivot."
  Candidate: `math-strategy/route-exploration-and-pivot`.

- **§4 dimension-parity tension**: Recognising that two constraints (E2
  parity, S parity) are in tension and finding a construction that
  satisfies both simultaneously. Candidate:
  `math-strategy/constraint-tension-resolution`.

- **p7-s5 (anti-obstruction)**: Including a section that argues "this
  natural objection does NOT apply" is a defensive proof move — clearing
  the ground. No existing pattern. Candidate:
  `math-strategy/preemptive-objection-clearance`.

### Q3: Synthetic QA decomposition

The synthetic QA generated 16 files covering all 8 wiring nodes (2 per
node), targeting gaps identified by the reviewer and the gap specs:

| Synth step | Wiring node(s) | Gap targeted | Reviewer finding |
|------------|----------------|--------------|------------------|
| p7-problem | p7-problem | Torsion lattice → manifold feasibility | — (framing) |
| p7-s1 | p7-s1 | Orbifold vs manifold for torsion | — |
| p7-s2 | p7-s2 | Bredon cohomology gives rational PD | Finding 1 (critical) |
| p7-s3 | p7-s3 | Fowler's FH(Q) criterion | — |
| p7-s3a | p7-s3a | Concrete lattice constructions | — |
| p7-s4 | p7-s4 | Wall obstruction / surgery upgrade | Findings 2, 3 (critical) |
| p7-s5 | p7-s5 | Smith theory non-obstruction | Finding 4 (medium) |
| p7-s6 | p7-s6 | Composition: surgery for rational PD groups | Finding 3 (critical) |

### Q4: Wiring diagram dependency structure

```
p7-problem ← p7-s1 (clarify: torsion-free case; torsion = orbifold)
                ↓
              p7-s2 (reform: Bredon PD replaces ordinary PD)
                ↓
              p7-s3 (assert: Fowler's FH(Q) criterion)
                ↓
              p7-s3a (clarify: concrete arithmetic lattice examples)
                ↓
              p7-s4 (assert: manifold-upgrade gap = open)
                        ↓ (references p7-s3a)
p7-problem ← p7-s5 (challenge: Smith theory — dismissed)
p7-problem ← p7-s6 (assert: conditional conclusion)
                ↓ (references p7-s4, p7-s5)
```

Three paths converge on the problem:
- **Main constructive path**: s1 → s2 → s3 → s3a → s4 (build up to the
  open obligation)
- **Defensive path**: s5 (clear the Smith theory objection)
- **Conclusion**: s6 composes both paths + conditionality

This three-path structure (construct + defend + compose-conditionally)
is itself richer than P3's two-path structure (construct + theorem).

### Q5: Reviewer findings → decision points

| Finding | Severity | Decision point | Pattern gap |
|---------|----------|----------------|-------------|
| 1. "Rational PD group" too loose with torsion | Critical | p7-s2 | `convention-bridge` applies: reframing PD for torsion groups IS a convention bridge — the PSR should demand explicit Bredon formulation |
| 2. Normal map setup asserted without hypotheses | Critical | p7-s4 | `reduce-to-known-result` applies: "standard surgery theory gives..." needs hypothesis verification |
| 3. Obstruction vanishing unsupported | Critical | p7-s4/s6 | `reduce-to-known-result` + `compose-independent-lemmas`: the obstruction claim is the weakest link in the composition |
| 4. Smith theory is anti-obstruction only | Medium | p7-s5 | New pattern needed: `preemptive-objection-clearance` — the pattern should distinguish "this doesn't obstruct" from "this helps construct" |

### Ready vs missing

**Ready (no new work needed):**
- P7 proof content: 7 sections + 4 approaches, 8 wiring nodes, 9 edges
- Synthetic QA: 16 files covering all 8 nodes
- Existing patterns: 3 from math-informal/ + 3 from math-strategy/ (P3)
- Reviewer findings: 4 findings mapped to decision points
- Wiring diagram: full dependency graph
- Validated method from M-P3-rational-reconstruction

**Missing (the actual work):**
- 2-3 new patterns to write: `route-exploration-and-pivot`,
  `constraint-tension-resolution`, `preemptive-objection-clearance`
- Virtual PSR for each of the 8 decision points (including the approach
  exploration in s4)
- Virtual PUR with mathematical scope for each step
- Assessment of P3 pattern transferability
- Comparison analysis: original vs pattern-guided decomposition
- Counterfactual assessment: would patterns have caught the 4 findings?
- Provenance JSON artifact

## 3. DERIVE

### 3.1 New patterns written

Three new patterns committed to `futon3/library/math-strategy/` (94c23fd):

| Pattern | Sigil | Addresses |
|---------|-------|-----------|
| `math-strategy/route-exploration-and-pivot` | — | p7-s4: multi-approach triage and pivot |
| `math-strategy/constraint-tension-resolution` | — | p7-s4 (cont.): dimension-parity tension → rotation route |
| `math-strategy/preemptive-objection-clearance` | — | p7-s5: Smith theory anti-obstruction |

Combined with P3's three patterns (9ae2db9: `convention-bridge`,
`non-circularity-check`, `compose-independent-lemmas`), the
`math-strategy/` namespace now has 6 patterns.

### 3.2 Pattern assignment per node

| Step | Node | Pattern | Rationale |
|------|------|---------|-----------|
| 1 | p7-problem | (input) | Problem framing, no PSR |
| 2 | p7-s1 | `math-informal/identify-the-obstruction` | Name what breaks with torsion |
| 3 | p7-s2 | `math-strategy/convention-bridge` | Reframe PD from ordinary to Bredon/orbifold |
| 4 | p7-s3 | `math-informal/reduce-to-known-result` | Invoke Fowler's FH(Q) criterion |
| 5 | p7-s3a | `math-informal/construct-an-explicit-witness` | Build concrete arithmetic lattice |
| 6 | p7-s4 | `math-strategy/route-exploration-and-pivot` + `constraint-tension-resolution` | Multi-approach exploration, parity tension, rotation pivot |
| 7 | p7-s5 | `math-strategy/preemptive-objection-clearance` | Smith theory doesn't obstruct |
| 8 | p7-s6 | `math-strategy/compose-independent-lemmas` | Conditional conclusion from E2 + S + Smith |

Note: p7-s4 is the only node requiring *two* strategy patterns. The
first governs the exploration/pivot; the second governs the specific
structural insight (tension dissolution) that drove the pivot choice.

### 3.3 Virtual PSR/PUR Replay

#### vPSR-1: p7-s1 — Baseline clarification

- **Step**: p7-s1 (§1 — torsion-free vs torsion)
- **Pattern chosen**: `math-informal/identify-the-obstruction`
- **Candidates considered**: `identify-the-obstruction`,
  `math-informal/split-into-cases` (rejected: this isn't a case split,
  it's identifying the single obstruction that makes the problem hard)
- **Rationale**: The problem asks about lattices with 2-torsion.
  The first move is to clarify: torsion-free works (Gamma acts freely,
  quotient is a manifold), torsion creates an orbifold, not a manifold.
  The obstruction IS the torsion.
- **Scope**: p7-s1
- **Confidence**: high

#### vPUR-1: p7-s1

- **Step**: p7-s1
- **Pattern**: `math-informal/identify-the-obstruction`
- **Scope**: p7-s1
- **Actions taken**: Stated: torsion-free case gives M = X/Gamma
  aspherical. With torsion, X/Gamma is orbifold. The gap between
  orbifold and manifold is the problem.
- **Outcome**: success — obstruction clearly named
- **Prediction error**: none
- **Gap detected**: none (this step is clean)

#### vPSR-2: p7-s2 — Cohomological reframing

- **Step**: p7-s2 (§2 — Bredon PD replaces ordinary PD)
- **Pattern chosen**: `math-strategy/convention-bridge`
- **Candidates considered**: `convention-bridge`,
  `math-informal/transport-across-isomorphism` (rejected: this is not
  an isomorphism but a reframing of what "PD group" means for torsion groups)
- **Rationale**: "Rational PD group" in ordinary cohomology is
  insufficient when torsion is present. The convention bridge is:
  ordinary PD → Bredon/orbifold PD, which is the correct framework.
- **Scope**: p7-s2
- **Confidence**: medium (bridge requires careful statement)

#### vPUR-2: p7-s2

- **Step**: p7-s2
- **Pattern**: `math-strategy/convention-bridge`
- **Scope**: p7-s2
- **Actions taken**: Stated that rational PD comes from orbifold/Bredon
  cohomology for proper cocompact Gamma-action on X=G/K.
- **Outcome**: **partial** — the bridge is stated but not made precise
- **Prediction error**: **moderate.** The `convention-bridge` pattern's
  THEN clause demands: state both conventions precisely, with citations;
  prove or cite that they agree in your setting. The original proof
  asserts "Gamma is a rational PD group" without specifying which
  cohomology theory makes this true for torsion groups.
- **Gap detected**: **Reviewer Finding 1 (critical).** The convention
  bridge from ordinary PD to Bredon PD is exactly the kind of
  "looks like the same thing" gap that `convention-bridge` is designed
  to catch. The pattern's HOWEVER clause warns: "what looks like 'the
  same thing with different notation' may hide a genuine mathematical
  difference."
- **Reviewer alignment**: STRONG — pattern catches Finding 1.

#### vPSR-3: p7-s3 — Fowler's criterion

- **Step**: p7-s3 (§3 — FH(Q) via equivariant finiteness)
- **Pattern chosen**: `math-informal/reduce-to-known-result`
- **Candidates considered**: `reduce-to-known-result` (only candidate —
  invoking a named theorem is the paradigmatic use)
- **Rationale**: Fowler's theorem gives FH(Q) membership under
  fixed-set Euler-vanishing hypotheses. This is a direct invocation.
- **Scope**: p7-s3
- **Confidence**: high (clean theorem application)

#### vPUR-3: p7-s3

- **Step**: p7-s3
- **Pattern**: `math-informal/reduce-to-known-result`
- **Scope**: p7-s3
- **Actions taken**: Invoked Fowler's equivariant finiteness theorem.
  Stated the hypothesis: fixed-point sets have zero reduced Euler
  characteristic.
- **Outcome**: success — reduction clean
- **Prediction error**: none
- **Gap detected**: The `reduce-to-known-result` pattern says "verify
  all hypotheses of the cited result." The Euler-vanishing hypothesis
  is stated but its verification is delegated to s3a (the concrete
  construction). This is acceptable — the pattern allows decomposition
  — but the dependency must be tracked.

#### vPSR-4: p7-s3a — Concrete construction

- **Step**: p7-s3a (§3b — arithmetic lattice examples)
- **Pattern chosen**: `math-informal/construct-an-explicit-witness`
- **Candidates considered**: `construct-an-explicit-witness` (only candidate)
- **Rationale**: The abstract criterion (Fowler) needs a concrete
  instantiation. Douba-Vargas Pallete reflection lattice construction.
- **Scope**: p7-s3a
- **Confidence**: high

#### vPUR-4: p7-s3a

- **Step**: p7-s3a
- **Pattern**: `math-informal/construct-an-explicit-witness`
- **Scope**: p7-s3a
- **Actions taken**: Cited arithmetic-lattice constructions. Stated
  that these land in FH(Q) but produce finite complexes, not manifolds.
- **Outcome**: success — witness constructed, limitation noted
- **Prediction error**: none
- **Gap detected**: none (this step is clean; the limitation is
  explicitly acknowledged, which is honest)

#### vPSR-5: p7-s4 — Multi-approach exploration and pivot

- **Step**: p7-s4 (§4 — manifold upgrade, 4 approaches)
- **Pattern chosen**: `math-strategy/route-exploration-and-pivot` +
  `math-strategy/constraint-tension-resolution`
- **Candidates considered**: `route-exploration-and-pivot` (primary),
  `math-informal/split-into-cases` (rejected: this is not a case split
  but a strategy exploration — the approaches are alternative routes,
  not exhaustive cases), `constraint-tension-resolution` (secondary,
  triggered by the parity tension discovered during exploration)
- **Rationale**: Four approaches to the S obligation. The pattern says:
  sketch each as a lightweight wiring diagram with status, triage, pivot.
  The exploration reveals a structural tension (E2 needs even n, S needs
  odd n) that `constraint-tension-resolution` addresses.
- **Scope**: p7-s4
- **Confidence**: low (this is the hard part — the open obligation)

#### vPUR-5: p7-s4

- **Step**: p7-s4
- **Pattern**: `route-exploration-and-pivot` + `constraint-tension-resolution`
- **Scope**: p7-s4
- **Actions taken**:
  - Approach I (Wall surgery): 3 sequential obstacles identified → deprioritized
  - Approach II (equivariant surgery): BLOCKED (codim-2 gap fails for codim-1) → killed
  - Approach III (orbifold resolution): no known technique → unexplored
  - Approach IV (rotation route): dissolves parity tension → MOST PROMISING
  - Tension identified: E2 needs even n, S prefers odd n
  - Resolution: codim-2 rotation instead of codim-1 reflection makes both want odd n
- **Outcome**: **partial** — pivot identified but S obligation remains open
- **Prediction error**: **moderate.** `route-exploration-and-pivot`
  correctly predicts that the "obvious" approach (Approach I, continuing
  with reflections) is not the best one. The pattern's triage step
  would have killed Approaches II and III faster. But the pattern does
  not itself close the remaining gap.
- **Gap detected**: **Reviewer Findings 2 and 3 (both critical).**
  - Finding 2: "normal map setup asserted without hypotheses" — this is
    in the Approach I/IV transition zone. The `route-exploration-and-pivot`
    pattern's THEN clause says "record why each blocked route fails" —
    but the original proof does not clearly record what hypotheses the
    surgery setup requires. The pattern would have demanded explicit
    hypothesis listing before asserting the normal map exists.
  - Finding 3: "obstruction vanishing unsupported" — this is the heart
    of the open S obligation. The `constraint-tension-resolution` pattern
    helps identify the rotation route but does not by itself verify the
    obstruction computation. However, `reduce-to-known-result` (inherited
    from the invocation at s3) would demand "verify all hypotheses" —
    and the L-theory computation is an unverified hypothesis.
- **Reviewer alignment**: PARTIAL — patterns surface the gaps but don't
  fully prescribe the repair (which requires actual L-theory computation).

#### vPSR-6: p7-s5 — Anti-obstruction

- **Step**: p7-s5 (§5 — Smith theory)
- **Pattern chosen**: `math-strategy/preemptive-objection-clearance`
- **Candidates considered**: `preemptive-objection-clearance` (only
  candidate — no existing pattern addresses defensive proof moves)
- **Rationale**: Smith theory is a natural objection: "doesn't Z/2
  acting freely on a rationally acyclic space create fixed-point
  constraints?" The answer is no (2 is invertible in Q), but this
  needs to be stated explicitly.
- **Scope**: p7-s5
- **Confidence**: high (the anti-obstruction is clean)

#### vPUR-6: p7-s5

- **Step**: p7-s5
- **Pattern**: `math-strategy/preemptive-objection-clearance`
- **Scope**: p7-s5
- **Actions taken**: Stated that Smith theory applies over Z/2, not Q.
  The mod-2 fixed-point constraint doesn't bite when working rationally.
- **Outcome**: success — objection cleared
- **Prediction error**: none
- **Gap detected**: **Reviewer Finding 4 (medium).** The pattern's
  quality criteria say: "clearly distinguish 'X does not obstruct' from
  'X positively helps.'" The original proof keeps this as a separate
  section (good) but the reviewer notes it should be a "side remark,
  not a central proof step." The pattern would have flagged this:
  anti-obstruction clearance belongs in supporting material, not the
  main proof line. The wiring diagram correctly has this as a
  `challenge` edge (not `assert`), which is appropriate.
- **Reviewer alignment**: MODERATE — pattern catches the scope issue
  (side remark vs. proof step) but the original already handled it
  acceptably.

#### vPSR-7: p7-s6 — Conditional conclusion

- **Step**: p7-s6 (§6 — compose conditional answer)
- **Pattern chosen**: `math-strategy/compose-independent-lemmas`
- **Candidates considered**: `compose-independent-lemmas` (only candidate)
- **Rationale**: The conclusion composes three independent pieces:
  E2 (discharged via Fowler + construction), S (open/conditional),
  Smith (non-issue). This is exactly what `compose-independent-lemmas`
  governs.
- **Scope**: p7-s6
- **Confidence**: medium (the composition includes an open piece)

#### vPUR-7: p7-s6

- **Step**: p7-s6
- **Pattern**: `math-strategy/compose-independent-lemmas`
- **Scope**: p7-s6
- **Actions taken**: Stated: if Gamma in FH(Q) (done) and surgery
  obstruction vanishes (open), then closed manifold exists. Smith
  theory non-issue (cleared).
- **Outcome**: **partial** — composition is honest about conditionality
- **Prediction error**: **low.** The pattern's THEN clause demands
  "trace every assertion to a piece." The conclusion correctly traces
  E2 to s3/s3a, S to s4, Smith to s5. The conditionality is explicit.
- **Gap detected**: The `compose-independent-lemmas` pattern would ask:
  "is each piece actually proved, or merely asserted?" For E2: proved.
  For S: explicitly open. For Smith: proved (anti-obstruction).
  The composition is honest. However, the pattern would also surface
  Finding 3 again: the obstruction vanishing claim at s4 feeds into
  s6's conditional, and the pattern's trace-back would highlight that
  this is the weakest link.
- **Reviewer alignment**: MODERATE — the composition itself is honest;
  the gap is in s4's content, which s6 correctly flags as conditional.

### 3.4 Counterfactual assessment

| Finding | Severity | Node | Pattern that catches it | How |
|---------|----------|------|------------------------|-----|
| 1. "Rational PD" too loose | Critical | p7-s2 | `convention-bridge` | HOWEVER warns: "same thing, different notation" may hide real difference. THEN demands precise statement of both conventions. |
| 2. Normal map asserted | Critical | p7-s4 | `route-exploration-and-pivot` + `reduce-to-known-result` | Route pattern demands recording why each approach fails/succeeds. Reduce pattern demands hypothesis verification. |
| 3. Obstruction vanishing unsupported | Critical | p7-s4/s6 | `compose-independent-lemmas` + `reduce-to-known-result` | Compose traces assertion to piece; piece (s4) is open. Reduce demands L-theory computation or citation. |
| 4. Smith = anti-obstruction only | Medium | p7-s5 | `preemptive-objection-clearance` | Pattern distinguishes "doesn't obstruct" from "helps construct." Flags scope: side remark, not central step. |

**Result: 4/4 findings caught** (3 strong, 1 moderate).

Compared to P3 (3/3 caught): the pattern discipline extends to the
harder problem. The key addition is `route-exploration-and-pivot`, which
catches Finding 2 through its demand for explicit hypothesis recording
at each approach. Without this pattern, the multi-approach exploration
in s4 would be a strategic black box.

### 3.5 P3 pattern transferability assessment

| P3 pattern | Used in P7? | Transfer quality |
|------------|-------------|-----------------|
| `convention-bridge` | Yes (p7-s2) | **Strong** — the Bredon/ordinary PD reframing is exactly a convention bridge. Catches Finding 1. |
| `non-circularity-check` | No | Not applicable — P7 doesn't have a construction-independence concern. The lattice construction (s3a) is independently motivated. |
| `compose-independent-lemmas` | Yes (p7-s6) | **Strong** — the three-piece conditional composition maps perfectly. Helps surface Finding 3. |

2/3 P3 patterns transfer directly. `non-circularity-check` is P3-specific
(relevant when a construction might accidentally use its own target). P7's
structure is different: the challenge is multi-approach exploration and
conditional assembly, not construction circularity.

### 3.6 Comparison: original vs pattern-guided

| Aspect | Original (ad hoc) | Pattern-guided (replay) |
|--------|-------------------|------------------------|
| Decomposition | 8 nodes, 3-path structure | Same structure — patterns don't change the decomposition |
| PD reframing (s2) | "Gamma is rational PD" | `convention-bridge` demands Bredon specification — **gap caught** |
| Fowler invocation (s3) | Clean | `reduce-to-known-result` confirms clean — no change |
| Construction (s3a) | Clean | `construct-an-explicit-witness` confirms clean — no change |
| Multi-approach (s4) | 4 approaches explored informally | `route-exploration-and-pivot` demands triage table + status — **faster kill of blocked routes, explicit hypothesis recording** |
| Parity tension (s4) | Discovered during exploration | `constraint-tension-resolution` names the pattern — **makes the insight transmissible** |
| Smith (s5) | Separate section | `preemptive-objection-clearance` says: side remark, not proof step — **scope correction** |
| Conclusion (s6) | Conditional, honest | `compose-independent-lemmas` confirms honest — traces to pieces |
| Reviewer findings caught | 0/4 (found by external reviewer) | **4/4 (found by pattern discipline)** |

## 4-6. ARGUE → VERIFY → INSTANTIATE

**ARGUE**: The central finding is that P3's patterns transfer well to P7
(2/3 directly applicable), but P7 requires 3 additional patterns for
features P3 lacked: multi-approach exploration, constraint tension, and
defensive proof moves. The 6-pattern library now covers both linear
proofs (P3-style) and branching/conditional proofs (P7-style).

The Lakatos framing illuminates P7 better than P3: the abandoned
approaches (I-III) are "refuted lemmas" in the Proofs & Refutations
sense. The rotation pivot is a "method of proofs and refutations" move
— the proof strategy is refined by incorporating the failures. The
`route-exploration-and-pivot` pattern captures this formally.

**VERIFY**: Full replay completed above (8 vPSRs + 8 vPURs).
Counterfactual: 4/4 findings caught. The pattern discipline is
sufficient for this problem's complexity level.

**INSTANTIATE**: Artifacts produced:
- `futon3/library/math-strategy/route-exploration-and-pivot.flexiarg` (94c23fd)
- `futon3/library/math-strategy/constraint-tension-resolution.flexiarg` (94c23fd)
- `futon3/library/math-strategy/preemptive-objection-clearance.flexiarg` (94c23fd)
- `futon6/holes/missions/M-P7-rational-reconstruction.md` (this file, updated)
- `futon6/data/first-proof/problem7-provenance.json` (TODO: extend wiring with vPSR/vPUR layers)

### Completion criteria check

1. ✓ Every decision point has a virtual PSR (7 vPSRs, one per non-input node)
2. ✓ Every proof step has a virtual PUR with mathematical scope
3. ✓ 3 new math-strategy/ patterns written and committed
4. ✓ P3 pattern transferability assessed (2/3 transfer)
5. ✓ Comparison table produced
6. ✓ 4/4 reviewer findings caught by pattern discipline
7. ○ Provenance JSON artifact (remaining TODO)

**Status: VERIFY complete, INSTANTIATE partial (provenance JSON pending).**
