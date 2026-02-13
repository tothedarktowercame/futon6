# Design Pattern: Layer-Switching Under Coaching

*Emerged from Problem 6 breakthrough, 2026-02-13.
Participants: Joe (coach), Codex (prover), Claude (analyst/retrospective).*

## The Pattern

When an agent is stuck in a TryHarder loop on a single mathematical layer,
a coaching intervention that forces layer enumeration can unlock an elementary
solution in a different layer.

## Problem 6: Before and After

### Before (TryHarder loop, ~8 hours)

The agent attempted to close Problem 6 (epsilon-light vertex subsets) by
adapting Batson-Spielman-Srivastava edge sparsification to vertex selection.
This generated:

- `problem6-proof-attempt.md` (35KB)
- 6+ handoff/dispatch documents (Directions A-D, GPL-H attack, closure attempts)
- `problem6-gpl-h-counterexample.md` (K_{t,r} broke the formulation)
- `problem6-gpl-h-attack-paths.md` (three formal approaches to close a gap
  that might be structural)
- 20 verification scripts

The pattern: identify gap -> attempt closure -> fail -> generate new dispatch
-> repeat. Each cycle was technically competent but confined to the
sparsification layer, where the obstruction (BSS is fundamentally edge-based)
may be structural.

### The coaching intervention

```
"I think you should think out of the box a bit. What kind of problem is
this? What kind of proof applies to this kind of problem? How would you
teach it to an undergraduate? How would you teach it to a grad student?
Where, in reality, do people learn about this kind of problem? What kind
of person finds this kind of problem easy? Are there 'tricks' (symmetries)
that would make some of your Zeno's Paradoxes go away?"
```

Key questions and what they force:

| Question | What it forces |
|----------|---------------|
| "What kind of problem is this?" | Layer enumeration — name the mathematical frameworks |
| "What kind of proof applies?" | Reduction identification per layer |
| "How would you teach it to an undergrad?" | Identify what's *elementary* about the problem |
| "What kind of person finds this easy?" | Identify the right mathematical community |
| "Are there symmetries/tricks?" | Look for structural cancellations |

### After (layer switch, ~15 minutes of thinking + execution)

The agent found an elementary proof chain:

1. **Turán's theorem** -> independent set I_0 with |I_0| >= epsilon*n/3
   (all internal edges light)
2. **Leverage filter** -> remove high-leverage vertices, keeping I_0' >= epsilon*n/12
3. **Barrier greedy** -> at each step, pick vertex with minimum spectral norm
4. **PSD trace bound** -> ||Y|| <= tr(Y) for PSD matrices
5. **Pigeonhole** -> if average trace < 1, some vertex has trace < 1,
   therefore spectral norm < 1, therefore barrier maintained

**For K_n**: dbar = 2t/(n*epsilon) = 2/3 at T = epsilon*n/3 steps. **Exact. Proved.**

**For general graphs**: dbar < 1 proved at M_t = 0; verified numerically
(440 steps, max dbar = 0.641) with 36% margin. Single remaining gap:
formal dbar < 1 bound at M_t != 0.

### What changed

| Aspect | Before | After |
|--------|--------|-------|
| Layer | Sparsification (BSS adaptation) | Combinatorial (Turán + greedy) |
| Machinery | Interlacing families, Borcea-Brändén, MSS | PSD trace bound + pigeonhole |
| Complexity | Heavy | Elementary |
| K_n status | Not proved | Proved exactly, c = 1/3 |
| General status | Structural gap (BSS edge-vertex mismatch) | Narrow gap (M_t != 0 amplification) |
| Gap character | Possibly structural | Probably technical |

## The Layer Analysis (retrospective)

Problem 6 has four mathematical layers. The coaching forced their enumeration:

| Layer | Reduction | Status (before) | Status (after) |
|-------|-----------|-----------------|----------------|
| Spectral | Bound ||L^{+/2} L_S L^{+/2}|| <= epsilon | Complete | Complete (unchanged) |
| Concentration | Matrix Freedman on vertex PSD summands | Framework correct | Framework correct (unchanged) |
| Combinatorial | Turán + leverage filter + barrier greedy | Not attempted | **Complete** (K_n proved) |
| Sparsification | Adapt BSS edge->vertex | **Structural gap** | Bypassed entirely |

The breakthrough was not "trying harder" in the sparsification layer. It was
recognizing that the combinatorial layer — which was never attempted — offers
an elementary path that bypasses the sparsification layer entirely.

## Why the Coaching Worked

### 1. It forced layer enumeration

"What kind of problem is this?" is not answerable within a single layer.
It requires stepping back to see the problem from multiple mathematical
perspectives. This is the first step of the strategy checklist from Part IV.

### 2. It forced pedagogical reduction

"How would you teach it to an undergraduate?" implicitly asks: "What's the
simplest version of this problem?" The simplest version is K_n, which has
uniform leverage scores (tau_e = 2/n) and admits an exact computation.
The agent found dbar = 2/3 for K_n — the proof that unlocked everything.

### 3. It broke the expertise trap

"What kind of person finds this easy?" reframes the problem from "what
sophisticated machinery do I need?" to "who would find this obvious?"
The answer: someone who thinks combinatorially about independent sets and
uses basic matrix inequalities. Not someone steeped in BSS machinery.

### 4. It suggested structural cancellation

"Are there tricks (symmetries) that would make Zeno's Paradoxes go away?"
The PSD trace bound + pigeonhole is exactly such a trick: instead of
controlling each vertex's spectral norm individually (Zeno's Paradox of
infinite case analysis), bound the average trace and invoke pigeonhole.
One inequality replaces infinitely many.

## Transferable Principles

### For coaching agents

1. **Don't dispatch; reframe.** "Close the gap in Section 5" generates
   another TryHarder cycle. "What kind of problem is this?" generates
   a layer switch.

2. **Ask pedagogical questions.** "How would you teach this?" forces
   identification of the elementary core. The elementary core is often
   the proof.

3. **Ask sociological questions.** "Who finds this easy?" identifies the
   right mathematical community and therefore the right techniques.

4. **Suggest structural cancellation.** Many gaps arise from treating
   each case separately when a global bound (averaging, symmetry,
   trace inequality) handles all cases at once.

### For agents working on problems

5. **Enumerate layers before attacking.** The strategy checklist from
   Part IV: enumerate layers, find reductions, assess status per layer,
   characterize obstructions, look for analogues.

6. **Try the elementary layer first.** The combinatorial layer (Turán +
   pigeonhole) was never attempted before the coaching. It was the
   simplest and it worked.

7. **Distinguish structural from technical gaps.** The BSS adaptation gap
   may be structural (BSS is fundamentally about edges). The M_t != 0
   amplification gap is probably technical (36% empirical margin, clear
   bootstrap structure). Work on technical gaps, not structural ones.

8. **The undergraduate test.** If you can't explain the proof strategy to
   an undergraduate, you may be in the wrong layer. The final proof
   chain (Turán + trace bound + pigeonhole) is undergraduate-accessible.

## Connection to Other Patterns

- **Problem 7 (layer-switching)**: Blocked in codimension-1 surgery,
  switched to codimension-2 rotations. Same pattern, different domain.

- **Problem 4 (creative reduction)**: Found the Phi_3 * disc identity by
  asking "what algebraic relationship does the reduction demand?" — the
  demand shaped the search. Here, asking "what's elementary?" shaped the
  search.

- **Problem 8 (structural decomposition)**: Found the symplectic direct sum
  that made everything fall out. Here, the PSD trace bound + pigeonhole is
  the structural observation that makes everything fall out.

## Evidence

- 6+ handoff documents before coaching (TryHarder loop)
- 15-minute thinking pause after coaching
- Complete K_n proof with c = 1/3 (exact)
- 440/440 numerical verifications with 36% margin
- Single narrow remaining gap (M_t != 0 amplification)
- 20 verification scripts with zero violations

## The Sentence

When stuck, don't ask "how do I close this gap?" Ask "what kind of problem
is this, and who would find it easy?"
