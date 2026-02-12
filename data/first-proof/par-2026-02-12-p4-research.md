# Post-Action Review: P4 n>=4 Proof Research + Verification Dynamics

**Session**: P4 n>=4 proof research + verification dynamics analysis
**Date**: 2026-02-12
**Duration**: ~4.5 hours
**Status**: Research cycle complete, open blockers identified

---

## Patterns Used

| Pattern | Type | Notes |
|---------|------|-------|
| confidence-laundering-detection | inferred | Root cause analysis of P2/P7/P8: fixes promoted claims without upgrading proofs |
| research/strategy-portfolio | inferred | Three strategies with explicit probabilities (35%/25%/15%) |
| numerical-validation-as-gate | inferred | 35K+ tests forcing honest characterization of open vs. proved |
| wiring-diagram-infrastructure | inferred | Typed edges enable targeted critique vs. whole-document rewrites |
| generation/verification asymmetry | inferred | MMCA/futon5 parallel: generation cheap, evaluation expensive |

## What Went Well

- **Voiculescu discovery**: Free Stam inequality IS our P4 inequality in the
  limit — validates conjecture theoretically and provides proof template
- **Strategy B algebraic proof**: Differentiation commutes exactly with ⊞_n
  (clean, exact identity for all n)
- **Strategy A diagnostic**: Identified that 1/Phi_4 is indefinite yet
  superadditive — rules out naive convexity route, focuses future work
- **Honest characterization**: P4 n>=4 openly flagged as open with strategies
  + blockers, not overclaimed
- **Parallel coordination**: Three agents (two Claudes + Codex) working on
  different aspects without interference
- **Checkpoint notes**: Captured system-level insights (verification dynamics,
  wiring diagram value) that feed back into process improvement

## What Could Improve

- Strategy A Codex research pipeline created but not yet executed
- Jensen gap direction (convex vs concave in cumulants) identified as critical
  but not fully resolved — low valid trial count (17-75) due to real-rootedness
  constraint limits confidence
- Strategy B failure mode not fully explored — didn't investigate whether a
  different functional (not 1/Phi_n) could induct properly
- No computer algebra (SOS relaxation) attempted for n=4 specifically
- Codex/Claude coordination is asynchronous; a discovery-gate model would
  reduce wasted computation on strategies proven unviable mid-way

## Prediction Errors

| Expected | Actual | Magnitude |
|----------|--------|-----------|
| Finitizing Voiculescu would be direct | Jensen gap gates entire approach, direction unknown | 0.7 |
| Differentiation commutativity → induction | Commutes exactly, but inequality direction flips at Step A | 0.6 |
| n=4 follows from n=3 scaling | Jump n=3→4 loses disc identity, gains cross-terms, breaks algebra | 0.5 |
| Verification cycle = failure mode | Verification cycle = working glacial system | 0.4 |

## Key Insight

1/Phi_n is superadditive in cumulant space WITHOUT being convex. The Hessian
is indefinite (both positive and negative eigenvalues) yet
f(κ_p + κ_q) >= f(κ_p) + f(κ_q) holds. The proof must exploit structure
beyond definiteness — possibly the real-rootedness constraint on the domain.

## Open Blockers

1. **Jensen gap direction** — gates Strategy A viability
2. **Phi_n/Phi_{n-1} relationship** — currently breaks Strategy B induction
3. **n=4 SOS decomposition** — computer algebra not yet attempted
4. **Codex Strategy A pipeline** — staged but not yet executed

## Suggestions

1. Run Codex Strategy A pipeline (`run-research-codex-p4-stam.py --limit 6`)
2. Investigate whether real-rootedness cone constraint enables superadditivity
   despite indefinite Hessian
3. Try SOS relaxation for n=4 via computer algebra (SymPy or DSOS)
4. Create proof attempt failure log as reusable learning artifact
5. Archive Voiculescu connection prominently — potential MO post
6. Revisit Strategy B with different functional (not 1/Phi_n)
7. Strengthen Codex/Claude coordination with discovery-gate model

## Lineage

### Commits (this session)
- `439aabe` — checkpoint-verification-dynamics.md
- `bf62e14` — wiring diagram infrastructure analysis
- `9efda5b` — Voiculescu 1998 free Stam discovery
- `ed99dca` — three proof strategies + Codex research script
- `827f4c6` — Strategy B: differentiation commutes with ⊞_n
- `9e3dbb6` — Strategy A: indefinite Hessian, superadditive anyway

### Key Files
- `data/first-proof/checkpoint-verification-dynamics.md`
- `data/first-proof/problem4-ngeq4-proof-strategies.md`
- `data/first-proof/problem4-ngeq4-research-brief.md`
- `scripts/verify-p4-strategy-a.py`
- `scripts/verify-p4-strategy-b.py`
- `scripts/run-research-codex-p4-stam.py`

### Evidence Added
- 35K+ numerical tests (n=4..8), 0 violations
- Voiculescu (1998) free Stam = asymptotic analog of P4 inequality
- Exact algebraic proof: (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)
- 1/Phi_4 Hessian indefinite in cumulant space (75/75 trials)
- Superadditivity holds despite non-convexity (0 violations)
- Cumulant additivity exact: κ_4 = a_4 - (1/12)a_2² under ⊞_4
