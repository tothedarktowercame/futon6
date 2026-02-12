# Process Patterns from the P6 and P7 Proof Journeys

**Date:** 2026-02-12
**Context:** P6 — 17 commits, 1 method wiring library, 1 verification script, 6 exhausted technique families, 1 remaining lemma. P7 — 18 commits, 22+ arXiv papers mined, 5 hypothetical proof architectures, 1 unconditional result, 1 open obligation.

---

## The Arcs

### Problem 6 (epsilon-light subsets)

```
LIBRARY (method wiring)  →  REDUCE (leverage threshold)  →  EXHAUST (6 techniques)  →  FORMALIZE (GPL-H)
     1 commit                   2 commits                      8 commits                   6 commits
```

**Timeline:**
- Feb 11: Initial solution (star domination + matrix Bernstein, conditional)
- Feb 11: Codex review catches 5 issues (Jensen direction, Schur order, etc.)
- Feb 12: Method wiring library built (D1-D10, 10 technique diagrams)
- Feb 12: Leverage Threshold Lemma + Turán bound → Cases 1, 2a proved (c₀ = 1/3)
- Feb 12: 6 subsample-and-concentrate techniques exhaust → quadratic-vs-linear wall proved
- Feb 12: Case 2b formalized as GPL-H with H1-H4 hypotheses
- Feb 12: Barrier greedy trajectory verifier (313 + 2040 trajectories, θ ≤ 0.667)
- Feb 12: MSS/KS mapping shows no existing theorem directly yields GPL-H
- Feb 12: Claude → Codex handoff with prioritized directions

**Status:** One lemma away. Cases 1 and 2a closed. Case 2b requires GPL-H (grouped paving lemma).

### Problem 7 (lattices with 2-torsion)

```
SPLIT (obligations)  →  DISCHARGE (E2)  →  MINE (literature)  →  TRIAGE (proof paths)  →  IDENTIFY (rotation route)
     2 commits             3 commits           5 commits              4 commits                 4 commits
```

**Timeline:**
- Feb 11: Initial solution (too strong claims, conflated E2 and S)
- Feb 11: Two Codex reviews catch structural errors (unproven lemmas, overclaimed steps)
- Feb 12: Clean split into E2 (finite CW) and S (manifold upgrade)
- Feb 12: E2 discharged via Fowler criterion + reflection lattices (even dim)
- Feb 12: S-branch: 3 approaches analyzed, all with unresolved obstacles
- Feb 12: Gap-focused arXiv mining (22 papers, curated KB)
- Feb 12: 5 hypothetical proof architectures mapped (H1-H5)
- Feb 12: Rotation route (H1) identified as most promising — dissolves parity tension
- Feb 12: Problem reduced to single lattice-existence question (number theory)

**Status:** E2 unconditional. S obligation open. Rotation route is the recommended path.

---

## What Worked

### 1. Method wiring library as pre-proof infrastructure (P6)

Before attempting the epsilon-light proof, a systematic library of 10 related methods was
constructed, each with a typed wiring diagram showing the shape signature:

```
Q (objective) → D (decomposition) → M (mechanism) → C (certificate) → O (output) → B (bridge to target)
```

Each diagram D1-D10 was assessed for "bridge status" — whether its output matches the target
theorem. Every diagram was `partial` or `none`, pinpointing exactly where each method fails
to apply.

This is different from a literature survey. The wiring diagrams are typed: they don't just
say "this paper is related" but specify the exact structural mismatch between the method's
output (typically edge-weighted sparsifiers) and the target (vertex-induced subsets).

**Evidence:** D2 (BSS barrier method) has bridge status `partial` — the core technique
(spectral barrier potential) transfers, but the atom structure differs. This led directly to
the Case 2b barrier greedy strategy, which adapted D2's potential function to vertex-level
multi-rank updates.

### 2. Exhaustion as theorem (P6)

Six subsample-and-concentrate techniques were systematically tried, each with quantitative
failure bounds:

| Technique                    | c₀ bound   | Bottleneck                           |
|------------------------------|-----------|--------------------------------------|
| Trace / Markov               | sublinear | ‖M‖ ≤ tr(M), loses dim factor       |
| Star domination + Bernstein  | O(ε/log n)| Converts p² → p, destroys headroom  |
| Decoupling + MSS (1-shot)    | O(ε²)    | p_A = ε²/12 to shrink atoms         |
| Decoupling + MSS (recursive) | O(√ε)    | 4^k spectral vs 2^k vertex scaling  |
| Greedy / chromatic number    | O(ε)     | IS in d≈1/ε graph has size ε·|I|    |
| Rank spreading heuristic     | O(ε²)    | tr/rank is lower bound, not upper    |

All six hit the **same wall**: spectral contribution scales as q² (quadratic in sampling rate)
while set size scales as q (linear). This convergence proves the wall is fundamental to the
technique family, not an artifact of any one approach. The wall itself became a lemma:
"trace-only certification has a sublinear ceiling" (Step 1 in the proof attempt).

This extends the P4 pattern `structural-obstruction-as-theorem` to a whole technique class
rather than a single method.

### 3. Obligation decomposition (P7)

The initial P7 solution conflated two logically independent obligations:
- **E2**: Place Γ in FH(Q) (finite CW complex with rationally acyclic universal cover)
- **S**: Upgrade the finite CW complex to a closed manifold

Splitting these was the key structural insight. E2 was discharged unconditionally (Fowler
criterion + reflection lattices). S remains open — but the problem is now well-localized.

The split also revealed that E2 and S have **conflicting requirements**: E2 (for reflections)
needs even ambient dimension n (so the fixed set has odd dimension and χ = 0), while the
surgery obstruction computation prefers odd n. This tension is invisible when the obligations
are conflated.

### 4. Hypothetical proof wiring diagrams (P7)

Before investing in any particular proof path for the S obligation, five complete proof
architectures were sketched as wiring diagrams with explicit node status:

| Diagram | Key idea | Assessment |
|---------|----------|------------|
| H1 | Rotation route (codim-2, odd n) | MOST PROMISING — dissolves parity tension |
| H2 | Reflection + Wall surgery | 3 sequential obstacles, unfavorable parity |
| H3 | Reflection + equivariant surgery | BLOCKED (codim-2 gap fails for reflections) |
| H4 | Orbifold resolution | UNEXPLORED, no known technique |
| H5 | Odd-dim reflection | BLOCKED (Gauss-Bonnet kills Fowler) |

This "plan before act" approach immediately killed two paths (H3, H5) and deprioritized
two more (H2, H4), concentrating effort on H1. Without this triage, the default would have
been to pursue H2 (the "obvious" continuation of the reflection construction), which has
three sequential obstacles and structural headwinds.

### 5. Dimensional tension as first-class structural constraint (P7)

The dimension-parity tension between E2 and S was recognized not as a nuisance but as a
theorem about the problem:

> For reflections: E2 needs n even (fixed set has odd dim, χ = 0). S prefers n odd
> (favorable L-theory parity). These cannot be simultaneously satisfied.

This led directly to the rotation route (H1): replace the codimension-1 reflection with a
codimension-2 rotation, where both E2 and S want odd n. The tension dissolves, equivariant
surgery becomes available, and the problem reduces to a single lattice-existence question.

The pattern: **when two parts of a proof have conflicting requirements on a parameter,
the parameter choice itself is a structural constraint that may point to a different
construction.**

### 6. Systematic literature mining with wiring diagrams (P7)

22+ arXiv papers were mined and organized into curated knowledge bases with wiring diagrams.
This is qualitatively different from a reference list: each paper was assessed for its
exact structural contribution to the proof gap, producing a typed record:

```json
{"paper": "arXiv:1204.4667", "role": "Fowler criterion", "applies_to": "E2",
 "status": "directly used", "gap_in_paper": null}
```

vs.

```json
{"paper": "arXiv:1705.10909", "role": "Costenoble-Waner equivariant surgery",
 "applies_to": "S-branch", "status": "blocked for codim-1",
 "gap_in_paper": "codim-2 gap hypothesis"}
```

This made the literature search productive: instead of reading papers linearly, each paper
was immediately classified by its structural fit to the proof architecture.

### 7. Cross-agent review as structural pressure (P6 + P7)

Both problems went through Codex review after the initial solution. The reviews caught:
- P6: Jensen inequality direction, Schur complement order, heuristic/theorem conflation
- P7: Unproven lemmas claimed as proved, incorrect fixed-set calculations, overclaimed PD

These weren't minor edits — they corrected the logical structure of the proofs. The review
cycle forced the arguments through a filter that checked not just correctness but the
distinction between what's proved and what's claimed. This is the multi-agent equivalent
of peer review.

---

## What Didn't Work

### 1. Star domination (P6)

The original P6 solution's key technique: replace edge indicators Z_u Z_v with
(Z_u + Z_v)/2 (star domination), then apply matrix Bernstein. This converts the quadratic
dependence on sampling probability (E[p²]) to linear (E[p]), destroying the concentration
headroom.

Concrete demonstration: for the star graph, star domination gives ‖A_v‖ = 1/2 for ALL
vertices, making matrix concentration impossible — yet the problem is trivially solvable
(take all leaves, L_{G[S]} = 0).

**Pattern**: Linearization techniques that work for edge sparsification fail for vertex
selection because they discard the quadratic structure that makes the problem solvable.

### 2. Reflection construction for S obligation (P7)

The reflection construction (even n) perfectly discharges E2 but creates structural
headwinds for every S-branch approach:
- Equivariant surgery: BLOCKED (codim-2 gap fails for codim-1 fixed sets)
- Wall surgery: unfavorable even-dimensional L-theory parity
- Orbifold resolution: no known technique

The reflection was "too good" for E2 — it discharged the obligation so cleanly that the
natural impulse was to build on it for S. But the codimension-1 geometry that makes E2
easy (odd-dimensional fixed set → χ = 0) simultaneously blocks equivariant surgery.

**Pattern**: When an initial construction discharges one obligation perfectly but
structurally blocks the next, consider whether a different construction could discharge
both — even if it makes the first obligation slightly harder.

### 3. Initial overclaiming (P6 + P7)

Both problems' initial solutions overclaimed:
- P6: Jensen inequality applied in wrong direction; MSS extrapolated from linear to
  quadratic without flagging the gap
- P7: Multiple lemmas stated as proved without complete arguments; FH(Q) membership
  presented as sufficient when the manifold upgrade was the actual question

The overclaiming was caught by cross-agent review, but the pattern is worth noting:
**first drafts under time pressure systematically overclaim the strength of arguments.**
The review step is not optional.

### 4. Trace-only certification (P6)

Four of the six subsample-and-concentrate techniques rely on controlling trace(M) and
using ‖M‖ ≤ tr(M). This bound loses a factor of rank(M), which can be as large as n.
The "greedy with trace averaging" bound was proved to have a sublinear ceiling:
tr(M_S) ≤ D·t²/(m₀−t), which gives sublinear set size.

**Pattern**: When multiple approaches fail at the same quantitative step (trace → operator
norm conversion), the step itself is the bottleneck. Progress requires a technique that
controls the operator norm directly, not through trace.

---

## Cross-Problem Comparison

### P4 vs. P6: Decomposition Strategy

| Aspect | P4 | P6 |
|--------|----|----|
| Decomposition target | Symmetry subspaces | Leverage threshold |
| Cases | 5 (by symmetry reduction) | 3 (by spectral mass) |
| Proved cases | 4 of 5 | 2 of 3 |
| Remaining gap | 4D generic (Case 3c) | Spectrally dense interior (Case 2b) |
| Closure method | PHCpack homotopy | Unknown (GPL-H needed) |
| Gap type | Computational | Theorem-level |

P4's gap was computational (the algebraic elimination was infeasible for 4 variables, but
homotopy continuation handles it routinely). P6's gap is theorem-level (no known
mathematical result implies the needed bound). This is a qualitatively harder type of gap.

### P4 vs. P7: Proof Architecture

| Aspect | P4 | P7 |
|--------|----|----|
| Structure | Single inequality, case decomposition | Two obligations, multiple approaches |
| Verification | Computational (scripts) | Structural (reference chasing) |
| Agent role | Compute and certify | Read and synthesize |
| Failure mode | OOM/timeout | Wrong construction |
| Scripts created | 24 | 1 (verify-p6-gpl-h.py borrowed format) |

P4 is a "computational mathematics" problem — the truth is accessible through sufficiently
powerful computation. P7 is a "structural mathematics" problem — the truth depends on
finding the right construction, and computation helps only indirectly.

### P6 vs. P7: Remaining Gap Character

P6's gap (GPL-H) is a **single operator-valued averaging lemma** with concrete hypotheses
and strong numerical evidence. The gap is narrow but theorem-level.

P7's gap (S obligation) is a **construction problem** with multiple possible approaches and
no clear computational test. The gap is wide but might collapse if the right lattice exists.

---

## Proposed Patterns

### New patterns

#### 1. math-informal/technique-landscape-map

**When you have a problem that multiple standard techniques partially address,
build a typed library of those techniques before proving anything.** Each
technique gets a wiring diagram showing where it matches and where it fails.
The library reveals the exact structural gap between existing methods and the
target theorem.

Evidence: P6 method wiring library (D1-D10). The bridge-status classification
immediately showed that all 10 methods produce edge-weighted outputs, not
vertex-induced subsets — pinpointing the exact structural mismatch.

Existing: None (numerical-scout is about function landscapes; this is about
technique landscapes).

#### 2. math-informal/exhaustion-as-theorem

**When multiple approaches to the same step all fail at the same quantitative
barrier, prove that the barrier is fundamental to the technique class.** This
converts scattered failures into a structural result that eliminates the entire
class and redirects effort.

Evidence: P6's 6 subsample-and-concentrate techniques all hit the quadratic-vs-linear
wall. The trace-only ceiling theorem proves any trace-based certification is sublinear.

Existing: Extends structural-obstruction-as-theorem from single-method to class-level.

#### 3. agent/hypothetical-proof-architecture

**Before investing in a proof path, sketch multiple complete proof architectures
as wiring diagrams with node status (solid/open/blocked).** Triage immediately:
kill blocked paths, deprioritize paths with sequential obstacles, focus on paths
with independent sub-goals.

Evidence: P7's 5 hypothetical diagrams (H1-H5). H3 and H5 immediately killed. H1
identified as most promising despite not being the "obvious" continuation.

Existing: None (split-into-cases is about decomposing a single proof, not comparing
multiple proof architectures).

#### 4. math-informal/parametric-tension-dissolution

**When two parts of a proof have conflicting requirements on a parameter
(dimension, degree, codimension), recognize the tension as structural and
seek a construction that dissolves it.** The tension is a theorem about the
approach, not the problem.

Evidence: P7 dimension-parity tension. E2 needs even n, S prefers odd n. The rotation
route (codim-2 instead of codim-1) makes both obligations prefer odd n.

Existing: None.

#### 5. agent/reduction-to-kernel

**Reduce a complex open problem to the smallest possible missing lemma with
explicit hypotheses, then record it as a named conjecture.** The reduction
itself is proved work, and the named conjecture becomes a handoff target.

Evidence: P6's reduction from "universal c₀ for epsilon-light subsets" to GPL-H
(4 explicit hypotheses H1-H4, one conclusion). The reduction proposition
(L2* implies linear-size closure) is a proved theorem. P7's reduction from
"closed manifold with given π₁" to "arithmetic lattice with order-2 rotation
in Isom(H^{2k+1})".

Existing: Relates to scope-bounded-handoff but is about mathematical content,
not agent coordination.

### Specializations of existing patterns

#### 6. math-informal/numerical-trajectory-evidence (specializes numerical-scout)

**For discrete/combinatorial conjectures, run trajectory-level numerical
experiments on parameterized graph families.** Track worst-case quantities along
greedy or randomized paths. Report distributional statistics (median, 90th pct,
max) across families and parameter values.

Evidence: verify-p6-gpl-h.py — 313 baseline + 2040 randomized trajectories, plus
exhaustive small-state check (n ≤ 14). Worst observed score ≤ 0.667.
Parent: math-informal/numerical-scout.

#### 7. agent/typed-literature-mining (specializes scope-bounded-handoff)

**When mining literature for proof ingredients, classify each paper by its
structural fit to the proof architecture, not just topical relevance.** Record:
which proof node it addresses, whether it directly applies or has a gap, and
what the gap is.

Evidence: P7's 22+ paper curated KB with typed wiring diagrams per paper.
Parent: social/scope-bounded-handoff.

---

## Meta-Observation: The One-Lemma Horizon

Both P6 and P7 (and P4) converged to a state where most of the proof is done and
a single remaining piece is the bottleneck. But the character of the bottleneck differs:

| Problem | Remaining piece | Type | Accessibility |
|---------|----------------|------|---------------|
| P4 | Case 3c root enumeration | Computational | High (PHCpack) |
| P6 | GPL-H (grouped paving lemma) | Theorem-level | Unknown |
| P7 | Lattice with order-2 rotation | Existence (number theory) | Unknown |

P4's bottleneck was accessible to existing tools (homotopy continuation). P6 and P7's
bottlenecks require new mathematical content. This suggests a two-phase proof engineering
process:

1. **Reduce phase**: decompose, exhaust techniques, formalize the gap
2. **Kernel phase**: either close with existing tools (P4) or record as a named
   conjecture for future work (P6, P7)

The reduce phase is always tractable with sufficient effort. The kernel phase may not be —
but the reduce phase is still valuable, because it converts "hard problem" into "hard
problem minus everything that was tractable." The difference between "I can't prove this
theorem" and "the proof reduces to this one operator-averaging lemma with these 4 explicit
hypotheses" is enormous.

---

## Files Referenced

| File | Role in pattern discovery |
|------|--------------------------|
| `data/first-proof/problem6-proof-attempt.md` | Evidence for exhaustion-as-theorem, reduction-to-kernel |
| `data/first-proof/problem6-method-wiring-library.md` | Evidence for technique-landscape-map |
| `data/first-proof/problem6-claude-handoff.md` | Evidence for cross-agent handoff structure |
| `scripts/verify-p6-gpl-h.py` | Evidence for numerical-trajectory-evidence |
| `data/first-proof/problem7-solution.md` | Evidence for obligation-decomposition, parametric-tension |
| `data/first-proof/problem7-hypothetical-wirings.md` | Evidence for hypothetical-proof-architecture |
| `data/first-proof/problem7-gap-paper-kb.md` | Evidence for typed-literature-mining |
| `data/first-proof/problem7r-s2b-candidate-construction.md` | Evidence for obligation-decomposition |
| `data/first-proof/p4-process-patterns.md` | Prior art for this document's format |
