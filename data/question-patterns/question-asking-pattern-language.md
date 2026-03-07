# Question-Asking Pattern Language for Mathematical Research

**Date:** 2026-03-07
**Status:** Phase 3 — synthesized from MO corpus analysis + First Proof evidence
**Sources:**
- 95,321 MO reverse-morphogenesis situations (Phase 1 clusters)
- 395 process-facing MO situations (Phase 2 extraction)
- 15 First Proof process patterns from P4, P6, P7
- Phase 2 Pass 1 alignment analysis

---

## Why a Pattern Language for Questions

Mathematical research has a rich public record of *answers* — proofs, theorems,
techniques. But the *questions that drive research forward* are mostly invisible.
MathOverflow captures content-facing questions ("does X exist?", "what's the
connection between A and B?"). The process-facing questions that determine
research productivity — "am I investing in the right approach?", "what's the one
missing piece?" — are asked privately, if at all.

This pattern language names 8 question-asking moves, ordered from most to least
visible in public mathematical discourse. The first four have partial MO analogs;
the last four have essentially zero public presence despite being the highest-
productivity patterns in the First Proof data.

---

## Layer 1: Publicly Visible (MO analogs exist, weakened form)

### QP-1: LANDSCAPE SCOUT
**"What does the territory look like before I commit to a route?"**

Map the problem space numerically or structurally before choosing a proof
strategy. The scout identifies: critical points, boundary behavior, case
structure, minimum values, symmetries.

| | |
|---|---|
| **MO analog** | COMPUTABILITY_QUESTION (~3,669). MO asks "can X be computed?" |
| **Process upgrade** | "What does the computation *reveal* about proof strategy?" |
| **First Proof** | `numerical-scout` (P4: 11K starts, 3 searches mapped the landscape before algebraic proof), `numerical-trajectory-evidence` (P6: 2,353 trajectories) |
| **MO examples** | Cluster-distributed; no dedicated cluster |
| **Question template** | "Before proving f > 0 on D, where are the critical points? What's the case structure? How close to zero does f get?" |
| **When to deploy** | Start of any proof attempt on a continuous/algebraic problem |

### QP-2: TECHNIQUE LANDSCAPE
**"What methods get close, and where does each one break?"**

Before proving, build a typed library of applicable techniques. Each gets a
wiring diagram showing where it matches the target and where it fails. The
library reveals the exact structural gap.

| | |
|---|---|
| **MO analog** | CONNECTION_SEEKING (~31K), ANALOGY_TRANSFER (~1,586) |
| **Process upgrade** | MO asks about relationships between objects. This asks about *structural fit* of methods to a specific target, with typed mismatch analysis. |
| **First Proof** | `technique-landscape-map` (P6: D1-D10 with bridge status), `hypothetical-proof-architecture` (P7: H1-H5 with node status), `progressive-method-escalation` (P4: 6 methods in order) |
| **Question template** | "For target T, methods M1...Mk each partially apply. For each Mi, what exactly is the structural mismatch between Mi's output and T's requirement?" |
| **When to deploy** | When multiple standard techniques partially address the problem |

### QP-3: STRUCTURAL PROBE
**"Does this mechanism actually work, or is there a structural obstruction?"**

Before investing in an approach, check whether the problem has properties that
structurally block it. This is pre-commitment due diligence.

| | |
|---|---|
| **MO analog** | FAILURE_UNDERSTANDING (~3,859), COUNTEREXAMPLE_HUNT (~5,040) |
| **Process upgrade** | MO asks "why did X fail?" (post-hoc). This asks "will X work?" (pre-commitment). Temporal reversal. |
| **First Proof** | `structural-obstruction-as-theorem` (P4: interior zeros block Putinar certificates — proved before wasting more SOS effort), `exhaustion-as-theorem` (P6: 6 techniques, same wall) |
| **MO process-facing** | 38 FAILURE_CHARACTERIZATION situations. Best examples: obstruction to extending vector bundles (se-mathoverflow-51441), obstruction to spin-c structure (se-mathoverflow-306603) |
| **Question template** | "Before I invest in approach A, is there a structural reason it can't work for problems with property P?" |
| **When to deploy** | Before committing >1 hour to any single proof strategy |

### QP-4: FAILURE CHARACTERIZATION
**"This failed — is the failure a theorem about the problem?"**

When a method fails, convert the failure into positive structural knowledge.
The failure characterization is often as valuable as a successful proof step.

| | |
|---|---|
| **MO analog** | Partial overlap with FAILURE_UNDERSTANDING |
| **Process upgrade** | MO seeks *explanation*. This seeks a *theorem about the failure*. |
| **First Proof** | `structural-obstruction-as-theorem` (P4: "Putinar certificates are structurally impossible" — 7 scripts → 1 theorem), `exhaustion-as-theorem` (P6: "trace-only certification has a sublinear ceiling") |
| **MO process-facing** | 38 situations. Genuine obstruction-to-extension questions. |
| **Question template** | "Method M fails on problem P. Is the failure an artifact of implementation, or does P have a structural property that blocks all methods of type M?" |
| **When to deploy** | After any method failure, before moving on |

---

## Layer 2: Publicly Invisible (zero/near-zero MO presence, highest productivity)

### QP-5: THEOREM APPLICABILITY
**"Do the hypotheses of theorem T actually match my situation?"**

Check whether a known theorem's hypotheses structurally fit the problem at
hand. Not "does T apply?" (which is often obvious) but "where exactly does
the structural fit break down, and is the broken hypothesis essential?"

| | |
|---|---|
| **MO presence** | ~258 situations (0.3%). Almost never asked publicly. |
| **Why invisible** | Researchers check hypotheses privately. The question "does Theorem 3.2 apply when the space isn't compact?" feels answerable by reading the paper — but understanding structural fit is different from reading. |
| **First Proof** | `technique-landscape-map` (P6: 10 methods with typed bridge-status assessment showing exact mismatch) |
| **MO process-facing** | 258 THEOREM_APPLICABILITY situations. Best: Torelli theorem scope (se-mathoverflow-23848), Lefschetz to singular varieties (se-mathoverflow-57744) |
| **Question template** | "T requires H1...Hn. My situation satisfies H1...H(n-1) but not Hn. Is Hn essential, or is there a variant that relaxes it?" |
| **When to deploy** | When importing a result from the literature into a proof |

### QP-6: TENSION DISSOLUTION
**"Two parts of my proof have conflicting requirements — can I resolve them?"**

Recognize when proof components impose conflicting constraints on a shared
parameter (dimension, degree, codimension). The tension itself is a structural
fact about the approach, and dissolving it may require a fundamentally
different construction.

| | |
|---|---|
| **MO presence** | ~295 situations (0.3%). Invisible in public Q&A. |
| **Why invisible** | Tension between proof components is a *construction-level* concern. MO questions are about single objects or statements, not about engineering a proof from parts. You'd never post "my E2 obligation needs even n but my S obligation needs odd n." |
| **First Proof** | `parametric-tension-dissolution` (P7: E2 needs even n, S prefers odd n → rotation route makes both want odd n). Also P4: "elegant approach (SOS) is structurally blocked, ugly approach (case-by-case) works" — a different kind of tension between aesthetics and feasibility. |
| **MO process-facing** | 295 situations, ~100 genuine after filtering. Best: BGG category O not closed under extension — which axiom? (se-mathoverflow-361482), removing condition from a conjecture (se-mathoverflow-118465) |
| **Sub-patterns** | |
| | *Parametric tension*: two obligations conflict on a parameter value |
| | *Aesthetic tension*: the "right" approach is blocked, the "ugly" one works |
| | *Scope tension*: a lemma that handles one case blocks generalization |
| **Question template** | "Parts A and B need conflicting values of X. Is there a construction where both are compatible?" |
| **When to deploy** | When a proof has 2+ obligations or sub-goals with shared parameters |

### QP-7: KERNEL IDENTIFICATION
**"What's the one remaining lemma — stated with exact hypotheses?"**

Reduce a complex open problem to the smallest possible missing piece. The
reduction itself is proved work. The named kernel becomes a handoff target
(to literature search, collaborator, or future self).

| | |
|---|---|
| **MO presence** | ~4,457 by loose keyword count, but mostly false positives from math vocabulary ("reducible", "sufficient condition"). True process-facing KERNEL_IDENTIFICATION is rare. |
| **Why invisible** | The reduction work that produces the clean statement is invisible. The kernel, once cleanly stated, sometimes appears as a "reference request" — but without the reduction context. |
| **First Proof** | `reduction-to-kernel` (P6: "epsilon-light" → GPL-H with 4 hypotheses; P7: "closed manifold" → "arithmetic lattice with order-2 rotation") |
| **Question template** | "I've proved everything except lemma L. Here's L with exact hypotheses. Is L known?" |
| **When to deploy** | When a proof attempt has succeeded everywhere except one step |

### QP-8: CONFIDENCE INVERSION
**"Am I confident about the wrong things?"**

Meta-cognitive audit of your own certainty. The pattern: when an approach
*feels* elegant and general, check whether it's structurally blocked. When
an approach feels tedious and case-specific, it may work *because* it engages
with the problem's actual structure.

| | |
|---|---|
| **MO presence** | 62 situations (0.06%). The rarest process-facing pattern. |
| **Why invisible** | Completely meta-cognitive. Never a mathematical question. |
| **First Proof** | The "confidence anticorrelation" (P4 making-of: "High confidence self-assessments were the worst failures"), P4 SOS elegance trap (7 scripts invested in the "right" approach that was structurally blocked) |
| **MO process-facing** | 62 situations. Sub-patterns identified from close reading: |
| **Sub-patterns** | |
| | *Proof error discovery*: finding a flaw in a published or in-progress argument (21 of 62). E.g., Hensel's flawed proof of e transcendental (se-mathoverflow-416293), lemma in SGA3 seems incorrect (se-mathoverflow-395865), Lomonaco paper relies on open conjecture (se-mathoverflow-361408) |
| | *Intuition failure*: expecting X, computing Y (18 of 62). E.g., expected longest stick → 0 but it doesn't (se-mathoverflow-430355), sheaf functor should preserve products but doesn't (se-mathoverflow-79712), eigenvalue should be simple but two nodal domains appear (se-mathoverflow-339090) |
| | *Historical counterexample*: learning that a plausible claim was disproved (12 of 62). E.g., Dulac's proof was incomplete for 60 years (se-mathoverflow-96510), Hilbert's conjecture about eversion was wrong (se-mathoverflow-312023) |
| | *Definitional surprise*: a definition doesn't behave as expected (11 of 62). E.g., Big O notation applied to constants is confusing (se-mathoverflow-76327), rank definitions give inconsistent results (se-mathoverflow-158121) |
| **Question template** | "I feel confident about approach A. Is there a structural reason to distrust that confidence?" |
| **When to deploy** | Whenever you notice yourself feeling certain. The more certain, the more important to check. |

---

## Pattern Interaction Graph

The patterns aren't independent — they form a natural workflow:

```
QP-1 LANDSCAPE SCOUT
  │
  ▼
QP-2 TECHNIQUE LANDSCAPE ──► QP-5 THEOREM APPLICABILITY
  │                                    │
  ▼                                    ▼
QP-3 STRUCTURAL PROBE ◄──── QP-8 CONFIDENCE INVERSION
  │
  ▼ (if probe fails)
QP-4 FAILURE CHARACTERIZATION
  │
  ▼ (if multiple failures converge)
QP-6 TENSION DISSOLUTION
  │
  ▼ (when most of proof is done)
QP-7 KERNEL IDENTIFICATION
```

**Typical sequence in a proof attempt:**
1. Scout the landscape (QP-1)
2. Map which techniques get close (QP-2)
3. For each technique, check structural fit (QP-5) and probe for obstructions (QP-3)
4. When approaches fail, characterize the failures (QP-4)
5. If failures reveal conflicting requirements, dissolve the tension (QP-6)
6. When most of the proof is done, identify the kernel (QP-7)
7. Throughout: audit your confidence (QP-8)

---

## Evidence Summary

| Pattern | First Proof instances | MO corpus (95K) | Productivity |
|---------|---------------------|-----------------|--------------|
| QP-1 LANDSCAPE SCOUT | 2 (P4, P6) | ~1,825 (keyword) | High |
| QP-2 TECHNIQUE LANDSCAPE | 3 (P4, P6, P7) | ~5,272 (keyword) | High |
| QP-3 STRUCTURAL PROBE | 2 (P4, P6) | 38 (tight regex) | Very high |
| QP-4 FAILURE CHARACTERIZATION | 2 (P4, P6) | 38 (tight regex) | Very high |
| QP-5 THEOREM APPLICABILITY | 1 (P6) | 258 (tight regex) | Very high |
| QP-6 TENSION DISSOLUTION | 2 (P4, P7) | 295 (tight regex) | Critical |
| QP-7 KERNEL IDENTIFICATION | 2 (P6, P7) | ~4,457 (inflated) | Critical |
| QP-8 CONFIDENCE INVERSION | 2 (P4 meta-obs) | 62 (tight regex) | Very high |

**The inverse relationship**: the patterns with the highest research productivity
have the lowest public visibility. This is the core finding.

---

## Application to M-distributed-frontiermath

The pattern language maps directly onto the FM-001 mission roles:

| Role | Primary patterns |
|------|-----------------|
| **Prover** (codex-1, zcodex) | QP-1 (scout), QP-2 (technique landscape), QP-7 (kernel ID) |
| **Critic** (claude-1) | QP-3 (structural probe), QP-4 (failure characterization), QP-8 (confidence inversion) |
| **Mentor** (claude-2) | QP-6 (tension dissolution), QP-8 (confidence inversion), pattern naming |

The Critic's job is exactly QP-3 + QP-4 + QP-8: probe for structural
obstructions, characterize failures as theorems, and audit the team's
confidence. The Mentor's job is QP-6 + QP-8: detect tensions between
proof components and break persistence loops driven by false confidence.

### Concrete protocol additions for FM-001

1. **Phase 1 (SPEC)**: Deploy QP-1 before any proof attempt. Numerical scouting
   on small Ramsey instances to map the landscape.
2. **Phase 2 (FALSIFY)**: Deploy QP-3. For each falsification attempt, ask:
   "is there a structural reason the hypothesis must fail?"
3. **Phase 3 (CONSTRUCT)**: Deploy QP-2 at the start — which known techniques
   get close to Ramsey-for-book-graphs? For each technique, QP-5 — do the
   hypotheses match? When approaches fail, QP-4 — characterize the failure.
   Mentor watches for QP-6 (tensions) and QP-8 (false confidence).
4. **Phase 4 (VERIFY)**: QP-7 — state the remaining kernel cleanly.
5. **Phase 5 (MAP)**: Record which patterns were deployed and what they yielded.

---

## Files

| File | Role |
|------|------|
| `process-content-alignment.md` | Phase 2 Pass 1 (pre-cluster alignment) |
| `mo-situation-clusters.json` | Phase 1 clusters (30 domain clusters, 95K entries) |
| `process-facing-mo-situations.json` | Phase 2 extraction (395 process-facing MO situations) |
| `p4-process-patterns.md` | 8 First Proof patterns from P4 |
| `p6-p7-process-patterns.md` | 7 First Proof patterns from P6/P7 |
| This file | Phase 3 pattern language synthesis |
