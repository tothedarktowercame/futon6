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

## Phases 3-6: To be executed

The DERIVE → ARGUE → VERIFY → INSTANTIATE phases follow the same
structure as M-P3-rational-reconstruction:

**DERIVE**: Write new patterns, define vPSR/vPUR for each node, design
the comparison method. Key new element: the multi-approach exploration
in §4 requires a vPSR that covers the *sequence* of approach choices,
not just a single pattern selection.

**ARGUE**: Test P3 pattern transferability. The central question: do
`convention-bridge` and `compose-independent-lemmas` work the same way
on P7, or does the conditional/multi-approach structure require
modifications? Also: does the Lakatos Proofs & Refutations framing
(from M-distributed-frontiermath) illuminate the P7 proof structure
better than the pattern-replay method alone?

**VERIFY**: Full virtual replay of 8 nodes + approach exploration.
Counterfactual assessment against 4 reviewer findings.

**INSTANTIATE**: `problem7-provenance.json` extending `problem7-wiring.json`.
New math-strategy/ patterns committed to futon3/library/.
