# Technote: Learn to Swim

*Validating the question-asking pattern language on arxiv proofs
before entering the Olympics.*

**Date:** 2026-03-15
**Status:** ROUTE (evidence landscape surveyed, execution path identified)
**Depends on:** Question-asking pattern language (Phase 3 complete),
proof peripheral (operational), M-P3/P7/P8 rational reconstructions
(complete)

---

## The metaphor

Validating a mathematics methodology on FrontierMath problems is like
validating your ability to swim by entering the Olympics. You need to
get in the pool first.

The pool: arxiv papers with known proofs. We elide the proof. We
reconstruct it using the question-asking pattern language (QP-1
through QP-8) as structural gates in the proof peripheral. We can
always peek at the answer. The goal is not "did we get the right
answer" but "did we produce a better explanation than the original."

## What already exists

### The question-asking pattern language (Phase 3, complete)

8 named patterns, ordered by public visibility vs. research
productivity:

| Pattern | Public visibility (MO) | Research productivity |
|---------|----------------------|---------------------|
| QP-1 LANDSCAPE SCOUT | ~1,825 | High |
| QP-2 TECHNIQUE LANDSCAPE | ~5,272 | High |
| QP-3 STRUCTURAL PROBE | 38 | Very high |
| QP-4 FAILURE CHARACTERIZATION | 38 | Very high |
| QP-5 THEOREM APPLICABILITY | 258 | Very high |
| QP-6 TENSION DISSOLUTION | 295 | Critical |
| QP-7 KERNEL IDENTIFICATION | ~4,457 (inflated) | Critical |
| QP-8 CONFIDENCE INVERSION | 62 | Very high |

**Core finding**: the highest-productivity patterns have the lowest
public visibility. THEOREM APPLICABILITY, TENSION DISSOLUTION, KERNEL
IDENTIFICATION, and CONFIDENCE INVERSION are essentially invisible on
MathOverflow (0.06%–0.3%) despite being the moves that actually change
research trajectories.

Source: `data/question-patterns/question-asking-pattern-language.md`

### The proof peripheral (operational)

- 9-phase cycle machine with structural enforcement
- 5-mode protocol: SPEC → FALSIFY → CONSTRUCT → VERIFY → MAP
- FALSIFY gate: must attempt to disprove before constructing
- Obligation ledger with DAG dependencies
- Dead-end honesty policy (failed routes are append-only)
- TryHarder licensing with kill conditions

Source: `futon3c/src/futon3c/peripheral/proof_shapes.clj`

### Post-hoc validation (complete)

The flywheel has already turned once:

1. **First Proof** (10 problems, ad hoc) → raw process traces
2. **P4/P6/P7 process patterns** → 15 named patterns extracted
   post-hoc from the actual proof campaigns
3. **M-P3/P7/P8 rational reconstruction** → replay with pattern
   discipline, virtual PSR/PURs, new `math-strategy/` patterns
4. **Reverse Morphogenesis** → 95K MO situations mined → 12
   content-facing patterns identified
5. **Question-asking pattern language** → 8 patterns synthesized
   from MO corpus + First Proof, with role mapping and interaction
   graph

The rational reconstructions validated the approach post-hoc: the
reviewer-identified gaps in P3 (star/non-star bridge asserted not
proved, irreducibility too compressed) map to specific pattern
failures that QP-5 (theorem applicability) and QP-3 (structural
probe) would have surfaced.

### The role mapping (specified)

| Role | Primary patterns |
|------|-----------------|
| **Prover** (codex) | QP-1 (scout), QP-2 (technique landscape), QP-7 (kernel ID) |
| **Critic** (claude-1) | QP-3 (structural probe), QP-4 (failure characterization), QP-8 (confidence inversion) |
| **Mentor** (claude-2) | QP-6 (tension dissolution), QP-8 (confidence inversion) |

### What's missing

The patterns aren't installed as structural gates in the proof
peripheral. The question-asking pattern language exists as a document;
it hasn't been wired into the SPEC→MAP protocol as checklist prompts
or phase-gated requirements. The flywheel has turned backward
(retro); it hasn't turned forward (prospective).

---

## The proposal: arxiv proofs with proofs elided

### Why arxiv, not FrontierMath

- **Known answers**: we can peek. The goal is explanation quality, not
  discovery.
- **Training data concern is irrelevant**: LLMs have seen these proofs,
  but the goal isn't reproduction — it's producing proofs that are
  *better as explanations* than the originals. Decision traces, dead
  ends characterized as theorems, tensions named and dissolved,
  confidence inversions caught. None of this is in the training data
  because nobody publishes it.
- **Gentler failure mode**: a failed reconstruction of a known proof
  teaches you about the methodology. A failed attempt at an open
  problem teaches you nothing except "that was hard."
- **Scale**: arxiv has millions of papers. FrontierMath has hundreds.

### What "better" means

A futonic proof is better than a standard proof if it includes:

1. **Decision trace**: at each choice point, what alternatives were
   considered and why this route was taken (virtual PSRs)
2. **Dead ends as theorems**: failed approaches characterized
   structurally, not just abandoned (QP-4)
3. **Tension dissolution narrative**: where conflicting requirements
   appeared and how they were resolved (QP-6)
4. **Confidence audit**: where the proof team's confidence was
   miscalibrated and what corrected it (QP-8)
5. **Kernel identification**: the key lemma stated cleanly with
   exact hypotheses, separable from the surrounding machinery (QP-7)
6. **Technique landscape**: which methods were tried, where each
   broke, what the structural mismatch was (QP-2)

A reader of the futonic proof should understand not just *that* the
theorem is true, but *why this proof strategy works and others don't*.

### Concrete protocol

1. **Select a paper.** Criteria: proof is non-trivial (>1 page), the
   result is interesting, the proof involves at least one non-obvious
   choice point. Start with areas where we have patterns: combinatorics,
   graph theory, algebra (from First Proof experience).

2. **Elide the proof.** Keep: theorem statement, definitions, notation.
   Remove: proof text. Preserve: any lemmas stated in the paper (their
   statements, not proofs).

3. **Run the proof peripheral.** SPEC → FALSIFY → CONSTRUCT → VERIFY
   → MAP, with QP patterns as structural gates:
   - SPEC: QP-1 (landscape scout) mandatory before any proof attempt
   - FALSIFY: QP-3 (structural probe) — attempt to disprove or find
     a counterexample
   - CONSTRUCT: QP-2 (technique landscape) at entry, QP-5 (theorem
     applicability) for each imported result, QP-4 (failure
     characterization) after each failed approach
   - VERIFY: QP-7 (kernel identification) — state the key lemma cleanly
   - Throughout: QP-8 (confidence inversion) at every commitment point
   - Mentor watches for QP-6 (tension dissolution)

4. **Compare.** Place the futonic proof alongside the original.
   Score on the six "better" criteria above. Where does the futonic
   proof add explanatory value? Where does it add noise?

5. **Extract patterns.** Any new proof strategy pattern discovered
   during reconstruction goes into `math-strategy/`. Any question-
   asking pattern refinement feeds back into the QP language.

### Scale

Start with 3 papers (same scale as M-P3/P7/P8 rational
reconstruction). If the methodology adds explanatory value on 3,
scale to 10. If it adds value on 10, the flywheel is validated.

FrontierMath is what you do *after* the flywheel is spinning.

---

## Connection to the three pillars

This validates the thesis at its most demanding:

- **The Argument**: if the pattern language produces better
  mathematical explanations than standard proofs, that's evidence for
  S3 (pattern transfer is real) and S6 (structural constraints
  crystallize into enforceable law) at a level that software
  engineering alone can't provide. Mathematics is unforgiving.

- **The Invariants**: the 5-mode protocol (SPEC→MAP) with FALSIFY
  gating IS a structural invariant. If it catches errors that ad-hoc
  approaches miss, the precision metaphor is validated in the hardest
  domain.

- **The Missions**: if the seven-phase methodology works for
  mathematics — where rigour is non-negotiable and dead ends are
  expensive — it works everywhere. The futonic approach isn't a
  software methodology applied to math; it's a *knowledge methodology*
  that happens to have been developed in software first.

---

## The "why" (author's note)

The broader motivation: things happen for a reason, but people don't
typically understand how things work very well. They don't understand
themselves very well either. As a result, opportunities to learn from
practice — from one's own experience and from others' — are not used
well, and not much is learned.

The question-asking pattern language is a concrete instance of this
diagnosis. The highest-productivity research moves (theorem
applicability checking, tension dissolution, confidence inversion) are
invisible in public mathematical discourse. Mathematicians use them
privately but don't share them. A system that installs these patterns
as structural gates — so that every proof attempt is forced through
the same inquiry discipline that productive researchers use
intuitively — would be making the invisible visible.

This isn't about competing with AI mathematicians on benchmark scores.
It's about producing mathematics that teaches — proofs that explain
not just *that* something is true, but *why this route and not that
one*, *where the confidence was misplaced*, and *what the dead ends
revealed*. The Lakatosian approach: specifications (conjectures,
structured inquiry) are the fulcrum; LLMs are the lever.

---

## Horizon: provably minimal understanding paths

The deepest version of this work isn't "produce a proof" or even
"produce a better proof." It's: **given a reader's prior knowledge,
produce the provably shortest path to understanding this proof.**

A proof is a logical structure. Understanding a proof is a *path*
through that structure. Different people need different paths depending
on what they already know. The minimal path for someone who knows
homological algebra but not field theory is different from the minimal
path for someone who knows both but not the specific construction
technique.

### Why this is tractable

Three things in the stack compose into the machinery for this:

**1. The obligation DAG is the proof's dependency graph.**
The proof peripheral already represents proofs as DAGs — nodes are
concepts/lemmas, edges are dependencies. Minimality on a DAG is
well-defined: shortest path from what you know to the theorem,
weighted by learning cost per edge.

**2. The question-asking patterns are the edges.**
QP-5 (theorem applicability) is the edge "do I already know enough to
apply this theorem?" QP-7 (kernel identification) is "what's the
single node I'm missing?" Personalisation comes from which edges have
zero cost for *this* reader — because they already know that step.

**3. The AIF observation model gives the cost function.**
Each reader's prior knowledge is an observation vector. Edges with
high precision (reader already knows this) have low cost. Edges with
low precision (new concept) have high cost. The minimal understanding
path is the policy that minimises total free energy from the reader's
current beliefs to the state where the proof is understood. This is
literally what AIF computes — it's not metaphor.

### The architecture

```
Reader's prior knowledge (observation vector)
    │
    ▼
Proof obligation DAG (nodes = concepts/lemmas, edges = dependencies)
    │
    ▼
Personalised shortest path (weighted by reader's knowledge gaps)
    │
    ▼
Ordered sequence of explanations (each one is a QP-guided
reconstruction of one edge in the minimal path)
```

### Why this is novel

Nobody is working on provably minimal understanding paths. Proof
assistants (Lean, Coq) verify correctness. Textbooks give one
fixed path. Tutoring systems adapt pacing but not structure. The
conjunction of proof DAGs, question-asking patterns, personalised
observation vectors, and AIF-based path minimisation doesn't exist
anywhere else.

### Why this is a product

"Give me the shortest path to understanding this paper, given what I
already know" is something every graduate student and every researcher
would use. The value proposition isn't "solve math problems" — it's
"make mathematical knowledge accessible on a personalised basis, with
provable minimality guarantees."

This connects to the broader "why": opportunities to learn from
practice are wasted because the paths to understanding are not made
explicit. The minimal understanding path is the antidote — it tells
you exactly what you need to learn, in what order, skipping everything
you already know, with each step explained by the question-asking
pattern that makes it intelligible.

---

## Files referenced

| File | What it provides |
|------|-----------------|
| `data/question-patterns/question-asking-pattern-language.md` | QP-1 through QP-8 |
| `data/question-patterns/mo-situation-clusters.json` | 95K MO situations clustered |
| `data/question-patterns/process-facing-mo-situations.json` | 395 process-facing situations |
| `data/first-proof/p4-process-patterns.md` | 8 process patterns from P4 |
| `data/first-proof/p6-p7-process-patterns.md` | 7 process patterns from P6/P7 |
| `holes/missions/M-P3-rational-reconstruction.md` | Post-hoc validation (COMPLETE) |
| `holes/missions/M-P7-rational-reconstruction.md` | Post-hoc validation (COMPLETE) |
| `holes/missions/M-P8-rational-reconstruction.md` | Post-hoc validation (COMPLETE) |
| `futon3c/src/futon3c/peripheral/proof_shapes.clj` | Proof peripheral shapes |
| `futon3c/src/futon3c/peripheral/proof_backend.clj` | Proof peripheral tools |
| `futon3/library/f6/*.flexiarg` | 10 math-specific patterns |
| `holes/handoffs/question-asking-as-reverse-morphogenesis.md` | ← theory |
| `holes/handoffs/question-asking-pattern-mining-from-mo-rm-2026-03-06.md` | Mining handoff |
