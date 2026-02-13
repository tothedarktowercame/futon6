# Claude ↔ Codex Pairing Context: Problem 6

## Current State (2026-02-13)

**One formal gap remains** in the P6 epsilon-light subset proof:
bounding ‖M_t‖ during the barrier greedy to control amplification.

Everything else is proved or empirically closed with large margin.

## What's Proved

1. **Turán step**: I₀ ≥ εn/3 with all internal edges light (τ_e ≤ ε)
2. **Foster bound**: avg leverage degree < 2 over I₀
3. **d̄ < 1 when M_t = 0**: d̄ ≤ 2t/(ε·r_t) ≤ (2/3)/(1-ε/3) < 1
4. **Pigeonhole + PSD trace**: d̄ < 1 ⟹ ∃v with ‖Y_t(v)‖ < 1
5. **K_n exact**: d̄ = 2t/(nε), c = 1/3

## The Last Gap

**Prove**: ‖M_t‖ ≤ cε for some c < 1 throughout the barrier greedy.

**Why it matters**: When M_t ≠ 0, d̄ is amplified by factor ε/(ε-‖M_t‖).
If ‖M_t‖ ≤ ε/4, then d̄ ≤ (4/3)·(2/3)/(1-ε/3) = (8/9)/(1-ε/3) < 1.

**Empirical status**: ‖M_t‖/ε ≤ 0.30 across all tested cases (367+ steps).
The barrier gap is ≥ 70% of ε at every step.

## Attack Vectors for the Gap

### A. Log-determinant potential
Φ(t) = log det(εI - M_t) decreases at each step. If the decrease per step
is bounded by the leverage contribution, a capacity argument may bound ‖M_t‖.
See `verify-p6-mt-growth.py` for trajectory data.

### B. BSS-style barrier function
The original Batson-Spielman-Srivastava method uses upper AND lower barrier
potentials. Our greedy only uses the upper barrier. Adding a lower barrier
(or showing the greedy implicitly maintains one) could directly bound ‖M_t‖.

### C. Leverage degree argument
Each added vertex contributes ≤ ℓ_v/ε to ‖M_t‖ (roughly). With the leverage
filter (ℓ_v ≤ C_lev/ε), each step adds ≤ C_lev/ε² to the spectral norm.
Over T = εm₀/3 steps: ‖M_T‖ ≤ T·C_lev/ε² = C_lev·m₀/(3ε). This is too
large. Need to exploit cancellation/subadditivity of spectral norm.

### D. Direct d̄ bound without M_t control
Instead of bounding ‖M_t‖ separately, bound tr(B_t F_t) directly using
the structure of how F_t and M_t co-evolve. The diagnostic shows they are
spectrally misaligned (F_t has only 44% weight in M_t's top eigenspace).

### E. Tighten the horizon
Instead of T = εm₀/3, use T = εm₀/4 or εm₀/5. This reduces d̄_uniform
and creates more room for amplification. Trade-off: smaller |S| (worse
constant c).

## Key Files

| File | What it contains |
|------|-----------------|
| `scripts/verify-p6-dbar-mechanism.py` | **Mechanism diagnostic** — decomposes d̄ into 4 mechanisms |
| `scripts/verify-p6-dbar-bound.py` | d̄ < 1 verification (440 steps) |
| `scripts/verify-p6-mt-growth.py` | M_t growth + log-det potential tracking |
| `scripts/verify-p6-gpl-h.py` | Base barrier greedy implementation |
| `scripts/verify-p6-leverage-aware-greedy.py` | Leverage-filtered variant |
| `data/first-proof/problem6-solution.md` | Full proof writeup |
| `data/first-proof/problem6-dbar-mechanism.md` | Mechanism discovery writeup |
| `data/first-proof/problem6-gpl-h-attack-paths.md` | Earlier attack path analysis |
| `data/first-proof/problem6-kk-extremal-analysis.md` | K_n exact analysis |

## Pairing Protocol

### For Claude
- Run diagnostic scripts, analyze output, formulate proof strategies
- Write proof arguments and update `problem6-solution.md`
- Generate verification tasks for Codex

### For Codex
- Verify algebraic identities and formal bounds
- Search literature for relevant theorems (BSS barriers, spectral sparsification)
- Run numerical experiments on specific graph families
- Check edge cases that might break proposed bounds

### Cycle Structure
1. **Claude proposes** a formal bound for ‖M_t‖ or direct d̄ control
2. **Codex verifies** the bound on test cases + checks algebra
3. **If it holds**: Claude writes it up, Codex reviews
4. **If it fails**: Codex reports the counterexample, Claude adjusts

### Key Invariants
- d̄ < 1 at every step (verified 440+ steps, 0 violations)
- Max d̄ = 0.718 (28% margin)
- Uniform d̄ (M_t=0 case) ≤ 0.666 (33% margin)
- Foster bound: tr(F_t) ≤ 2t always
- The K_n formula d̄ = 2t/(nε) is exact and proves c = 1/3

## Quick Start for New Agent

```bash
cd /home/joe/code/futon6

# Understand the mathematical framework
cat data/first-proof/problem6-solution.md

# See the mechanism discovery
cat data/first-proof/problem6-dbar-mechanism.md

# Run the diagnostic
python3 scripts/verify-p6-dbar-mechanism.py

# Run the base verification
python3 scripts/verify-p6-dbar-bound.py

# Track M_t growth
python3 scripts/verify-p6-mt-growth.py
```
