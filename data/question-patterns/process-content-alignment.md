# Phase 2 Pass 1: Process–Content Pattern Alignment

**Date:** 2026-03-07
**Source:** First Proof process patterns (P4, P6, P7) × MO keyword taxonomy
**Status:** Pre-cluster pass — structural alignment only, pending Phase 1 cluster data

---

## Method

Cross-reference the 8 process-facing question patterns identified in the
handoff doc against:

1. The 12 content-facing MO keyword patterns (with frequency counts)
2. The 15 concrete First Proof patterns from `p4-process-patterns.md` (8)
   and `p6-p7-process-patterns.md` (7)

For each process pattern, assess:
- Does it have a content-facing analog on MO?
- If partial, what does the process version add?
- Which First Proof patterns instantiate it?
- What would a "question template" look like?

---

## Alignment Table

### 1. STRUCTURAL PROBE
**"Does this mechanism actually work?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | Partial — maps to FAILURE_UNDERSTANDING (3,859) and COUNTEREXAMPLE_HUNT (5,040) |
| Gap | MO version asks "why does X fail?" post-hoc. Process version asks "will X work?" pre-commitment. The temporal orientation is reversed. |
| First Proof instances | `structural-obstruction-as-theorem` (P4: 7 SOS scripts proved Putinar certificates structurally impossible), `exhaustion-as-theorem` (P6: 6 techniques all hit quadratic-vs-linear wall) |
| Question template | "Before I invest in approach A, is there a structural reason it can't work for problems with property P?" |
| Productivity | Very high — each instance eliminated an entire proof strategy |

### 2. THEOREM APPLICABILITY
**"Do the hypotheses of theorem T actually match my situation?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | ~0. This is almost never asked publicly. |
| Gap | Researchers check theorem hypotheses privately but don't post about it. The question "does Theorem 3.2 of [X] apply when the space isn't compact?" is answerable by reading the paper — but reading isn't the same as understanding the structural fit. |
| First Proof instances | `technique-landscape-map` (P6: 10 methods assessed for "bridge status" — exact structural mismatch between each method's output and the target theorem) |
| Question template | "Theorem T requires hypotheses H1...Hn. My situation satisfies H1...H(n-1) but not Hn. Is Hn essential, or is there a variant that relaxes it?" |
| Productivity | Very high — the P6 method wiring library immediately showed all 10 techniques produce edge-weighted outputs rather than vertex-induced subsets |

### 3. FAILURE CHARACTERIZATION
**"Why does this fail structurally, not just computationally?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | Partial — FAILURE_UNDERSTANDING (3,859) overlaps, but MO version is usually "why does method M give wrong answer?" not "what structural property of the problem blocks method M?" |
| Gap | Process version seeks a *theorem about the failure* — converting failure into positive knowledge. MO version seeks an *explanation of the failure* — understanding what went wrong. |
| First Proof instances | `structural-obstruction-as-theorem` (P4: interior zeros where all constraints are strict kill Putinar certificates — this is a theorem, not a bug report), `exhaustion-as-theorem` (P6: trace-only certification has a sublinear ceiling — proved as a lemma) |
| Question template | "Method M fails on problem P. Is the failure an artifact of my implementation, or does P have a structural property that blocks all methods of type M?" |
| Productivity | Very high — each characterization was as valuable as a successful proof step |

### 4. TENSION DISSOLUTION
**"Can I resolve conflicting requirements without compromise?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | ~0. This is invisible in public Q&A. |
| Gap | Tension between proof components is a *construction-level* concern. MO questions are about single mathematical objects or statements, not about engineering a proof from parts. You'd never post "my E2 obligation needs even n but my S obligation needs odd n" on MO. |
| First Proof instances | `parametric-tension-dissolution` (P7: E2 needs even n, S prefers odd n → rotation route makes both want odd n), also implicit in P4's "the elegant approach (SOS) is structurally blocked, the ugly approach (case-by-case resultant) works" |
| Question template | "Parts A and B of my proof have conflicting requirements on parameter X. Is there a construction where both parts are compatible?" |
| Productivity | Critical — the P7 rotation route was the key insight, and it came from recognizing the tension as a structural constraint pointing to a different construction |

### 5. KERNEL IDENTIFICATION
**"What's the one remaining lemma?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | ~0. Researchers don't usually post "here's my proof, what's the missing piece?" |
| Gap | This is a *reduction* operation — converting a complex open problem into a single named conjecture with explicit hypotheses. MO questions are usually about single mathematical statements, not about proof architecture. |
| First Proof instances | `reduction-to-kernel` (P6: "universal c₀ for epsilon-light subsets" reduced to GPL-H with 4 explicit hypotheses; P7: "closed manifold with given π₁" reduced to "arithmetic lattice with order-2 rotation") |
| Question template | "I've proved everything except one lemma L. Here's L with its exact hypotheses. Is L known, or does it follow from known results?" |
| Productivity | Critical — the reductions themselves are proved theorems, and the named conjectures become handoff targets |
| Note | This is the one process pattern that *could* appear on MO — and occasionally does (as "reference request" questions). But the reduction work that produces the clean statement is invisible. |

### 6. LANDSCAPE SCOUT
**"What does the territory look like before I commit to a route?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | Partial — COMPUTABILITY_QUESTION (3,669) overlaps for computational aspects. But the process version is about *mapping* before *proving*, not about computability per se. |
| Gap | MO asks "can X be computed?" Process version asks "what does the landscape of X look like, so I can choose a proof strategy?" The scout is pre-strategic; the MO question is post-strategic. |
| First Proof instances | `numerical-scout` (P4: 11,000+ starts, 3 independent critical point searches mapped the full landscape before any algebraic proof), `numerical-trajectory-evidence` (P6: 2,353 trajectories on parameterized graph families) |
| Question template | "Before proving that f > 0 on domain D, I want to know: where are the critical points? What's the case structure? How close to zero does f get?" |
| Productivity | High — the P4 numerical scout identified the case structure that guided the entire algebraic proof |

### 7. TECHNIQUE LANDSCAPE
**"What methods get close, and where do they break?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | Partial — ANALOGY_TRANSFER (1,586) and CONNECTION_SEEKING (31,179) overlap. "What's the relationship between method A and method B?" is a common MO question. |
| Gap | MO version asks about relationships between mathematical objects. Process version asks about the *structural fit* of methods to a specific target. The P6 wiring library is typed — each method gets a "bridge status" assessment showing exactly where it fails to connect to the target. |
| First Proof instances | `technique-landscape-map` (P6: D1-D10 wiring diagrams with bridge status), `hypothetical-proof-architecture` (P7: H1-H5 diagrams with node status solid/open/blocked), `progressive-method-escalation` (P4: resultant → Sturm → sign-counting → IA → SOS → homotopy) |
| Question template | "For target theorem T, methods M1...Mk each partially apply. For each Mi, what exactly is the structural mismatch between Mi's output and T's requirement?" |
| Productivity | High — killed 2 of 5 proof paths immediately (P7), identified the exact structural gap (P6) |

### 8. CONFIDENCE INVERSION
**"Am I confident about the wrong things?"**

| Dimension | Assessment |
|-----------|------------|
| MO analog | ~0. This is a meta-cognitive pattern, not a mathematical question. |
| Gap | Completely absent from content-level Q&A. This is about the *reliability of your own assessment*, which is never a mathematical question but is a critical research process question. |
| First Proof instances | The "confidence anticorrelation" meta-observation (P4 making-of: "High confidence self-assessments (P4, P9) were the worst failures. The easiest-feeling problems had the deepest gaps."), also P4: "the elegant approach (SOS) feels right but is structurally blocked; the ugly approach (case-by-case) works" |
| Question template | "I feel most confident about approach A. Is there a structural reason to distrust that confidence? What am I not seeing?" |
| Productivity | Very high — the single most expensive error in the First Proof was confidence in approaches that felt elegant but were structurally blocked |

---

## Summary: The Process–Content Gap

| Pattern | MO frequency | Process productivity | Gap type |
|---------|-------------|---------------------|----------|
| STRUCTURAL PROBE | ~9K (partial) | Very high | Temporal reversal (pre vs post) |
| THEOREM APPLICABILITY | ~0 | Very high | Invisible private work |
| FAILURE CHARACTERIZATION | ~4K (partial) | Very high | Explanation vs theorem |
| TENSION DISSOLUTION | ~0 | Critical | Construction-level, not object-level |
| KERNEL IDENTIFICATION | ~0 | Critical | Reduction work invisible |
| LANDSCAPE SCOUT | ~4K (partial) | High | Pre-strategic vs post-strategic |
| TECHNIQUE LANDSCAPE | ~33K (partial) | High | Typed fit vs general relationship |
| CONFIDENCE INVERSION | ~0 | Very high | Meta-cognitive, not mathematical |

**Key finding**: The four highest-productivity patterns (THEOREM APPLICABILITY,
TENSION DISSOLUTION, KERNEL IDENTIFICATION, CONFIDENCE INVERSION) have **zero
MO presence**. They are invisible in public mathematical discourse.

The four patterns with partial MO analogs (STRUCTURAL PROBE, FAILURE
CHARACTERIZATION, LANDSCAPE SCOUT, TECHNIQUE LANDSCAPE) exist on MO but in
*weakened form* — the MO version is reactive where the process version is
proactive, and explanatory where the process version is structural.

---

## Predictions for Phase 1 Clusters

When the MO situation clusters arrive from Codex, I predict:

1. **Large clusters** will correspond to EXISTENCE_WONDER and CONNECTION_SEEKING
   (the two most frequent keyword patterns). These are the bread-and-butter of
   public mathematical Q&A.

2. **Medium clusters** will split CURIOSITY_FROM_SURPRISE into at least two
   sub-patterns: "genuine surprise" (structural) and "pedagogical surprise"
   (student encountering a known result for the first time).

3. **Small/absent clusters** for anything resembling the four zero-MO-presence
   patterns. If any cluster maps to TENSION DISSOLUTION or KERNEL IDENTIFICATION,
   that's a strong signal that researchers occasionally *do* surface these
   questions publicly — and those examples would be the most valuable in the
   corpus.

4. **The 9.5% parse-rate bias** (Llama-3-8B succeeded on cleaner Q&A) means
   the clusters will over-represent well-formed content questions and
   under-represent the messy process-adjacent questions. The most interesting
   finds will be in the noise at the cluster periphery.

---

## Phase 2 Pass 2 (pending Phase 1 clusters)

When `mo-situation-clusters.json` arrives:

1. For each cluster, compute embedding distance to the 8 process-pattern
   descriptions above.
2. Flag clusters that are "near" process patterns (cosine similarity > 0.6 to
   the question template text).
3. For near-matches, pull representative examples and assess whether they're
   genuinely process-facing or just content-facing questions that happen to use
   similar vocabulary.
4. For the four zero-MO patterns, look for *any* cluster members within
   distance 0.7 — these would be rare public instances of normally-private
   research process questions.
