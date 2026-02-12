# Checkpoint Note: Verification Dynamics in the First Proof System

**Date:** 2026-02-11
**Context:** After completing P4 n=3 proof + stress testing, mid-cycle on
Codex reviewer + Claude responder for P2, P7, P8 gaps.

---

## The generation/verification asymmetry

All 10 problem solutions were generated in a ~60-minute burst (7 min/problem).
The reviewer critique landed ~4.5 hours later. Fixes were applied in ~30 minutes
(~10 min/problem). Each stage ran at LLM generation speed, not at
mathematical verification speed.

This is the same structure as MMCA rule exploration in futon5: generation
(bit flipping / proof sketching) is cheap; evaluation (detecting edge-of-chaos /
checking whether cited theorems deliver claimed results) is expensive.

## What the Proofs-and-Refutations cycle actually looks like

The system running right now:

1. **Prover** generates candidate proof (fast, cheap, over-confident)
2. **Critic** (Codex reviewer) identifies structural gaps
3. **Responder** (another Claude) attempts repairs at generation speed
4. **Second critic** catches "confidence laundering" — claims that got
   stronger through rewrites without the math catching up

This is a genuine Lakatos dialectic. The key Lakatos insight: **refutations
are more valuable than proofs**. P4's honest "this is open" is a better
outcome than P7's promoted-to-universal-theorem.

## Three failure modes observed

### Confidence laundering (P7 — worst case)

Each rewrite made the claim stronger rather than more honest:
- v1: "vanishes by parity" (bare assertion, Medium-low confidence)
- Fix 1: Added contradictory AHSS argument (bumped to Medium)
- Fix 2: Promoted to "automatically zero for all Gamma" (universal claim)

The confidence rating tracked the *ambition* of the claim, not the *solidity*
of the argument.

### Fix-by-citation vs. fix-by-proof (P2 vs P8)

| Problem | Fix strategy | Outcome |
|---------|-------------|---------|
| P8 | **Proved the claim** (vertex spanning lemma from geometry) | Recoverable |
| P2 | **Cited theorems** (BZ 1976) that plausibly imply the claim | Needs verification |
| P7 | **Asserted universal theorem** citing close-but-not-matching refs | Least recoverable |

Fixes that derive the needed fact from the problem's own structure (P8)
succeed. Fixes that close gaps by citing theorems that are close-but-not-quite
(P7) don't.

### The 28-minute fix cycle

Reviewer critique at 22:48, fix at 23:16. That's generation speed, not
verification speed. The result: fixes that look like proofs (lemma statements,
proof sketches, citations) but sometimes aren't.

## What worked: P4 as the success case

P4 had a **checkable claim** (a numerical inequality). This forced honesty:
- n=2: proved (equality)
- n=3: proved (Cauchy-Schwarz, symbolically verified in SymPy)
- n>=4: honestly flagged as open, with 35K+ numerical tests confirming the
  conjecture and a research brief identifying proof strategies

The difference: P4's claim could be stress-tested computationally. P2/P7/P8
claims (existence of objects, vanishing of obstructions, representation-theoretic
spanning) cannot be numerically spot-checked. So "looks like a proof" failure
modes persisted through the fix cycle.

## Codex's current assessment

- **P8: most recoverable.** Conditional on precise nondegeneracy hypothesis,
  or prove that hypothesis from geometric setup.
- **P2: recoverable but harder.** Needs verification that BZ 1976 actually
  delivers the fixed-W spanning claim.
- **P7: least recoverable.** Obstruction-vanishing step needs a precise
  theorem tailored to this lattice family, or retreat to conditional status.

## What this means for the superpod pipeline

The bottleneck is **number of critic cycles before convergence**, not compute
or token cost:
- P8 converged in one fix round (real proof added)
- P2 might converge in two (if someone checks the BZ citation)
- P7 may need the critic to force a retreat to conditional status —
  itself a valid proof-theoretic outcome

### Connection to futon3c agency model

The futon3c peripheral model could compress the cycle because peripherals
can run verification sub-tasks **in parallel**. The critic could spawn
focused verification peripherals ("does Ranicki Prop 22.34 apply to lattices
with torsion?") while the main agent continues with other problems.

The speedup lives not in faster generation but in **parallelized verification
with memory transfer** back to the main context.

### Connection to MMCA (futon5)

Same structure: the interesting dynamics live at the boundary between
"trivially generatable" and "meaningfully evaluable." Building the detector
that distinguishes edge-of-chaos from frozen/chaotic is the hard problem
in both settings.

## Status summary

| Problem | Status | Convergence |
|---------|--------|-------------|
| P4 n=2 | PROVED | Done |
| P4 n=3 | PROVED (symbolic) | Done |
| P4 n>=4 | Open (research brief ready) | Needs new math |
| P8 | In critic cycle | ~1 more round |
| P2 | In critic cycle | ~2 more rounds |
| P7 | In critic cycle | May need retreat to conditional |

We are not shipping failed proofs. The system catches problems. This is
verification at glacial scale — but at least it is happening.
