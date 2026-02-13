# Unified Codex Repair Handoff: P1, P2, P8, P9

Date: 2026-02-13
Agent: Claude
Script: `scripts/codex-unified-repair.py`

## Context

The closure-validation-audit and per-problem codex-results.jsonl files
identified specific gaps in Problems 1, 2, 8, and 9. This script
targets those gaps with numerical verification and algebraic probes.

Problem 7 is excluded: its gaps (surgery obstruction computation for
the S obligation) require topological arguments not amenable to
numerical verification.

## Gap inventory (what the script targets)

### Problem 1: Phi^4_3 measure equivalence under smooth shifts

| Gap | Location | Script Task | Priority |
|-----|----------|-------------|----------|
| Young's inequality chain (cubic controlled by quartic) | Section 5, Lemma 5.1 | P1-T1 | HIGH |
| Wick expansion of V(phi-psi) - V(phi) | Section 4, lines 99-112 | P1-T2 | MEDIUM |
| Equivalence chain logic (pushforward preserves ~) | Section 6, Lemma 6.1 | P1-T3 | HIGH |

**Nature of gaps**: Presentational. The proof is mathematically correct
but leaves key steps implicit (pushforward preserves equivalence,
distributional lift of pointwise Young's inequality). The Codex audit
(p1-s6) correctly flagged the missing step.

### Problem 2: Universal test vector for Rankin-Selberg integrals

| Gap | Location | Script Task | Priority |
|-----|----------|-------------|----------|
| Laurent ring units (monomial characterization) | Section 2, Lemma 2.1 | P2-T1 | MEDIUM |
| PID submodule structure | Section 3a, Lemma 3a.1 | P2-T2 | HIGH |
| GL_n-equivariance computation | Section 3a, Key Step | P2-T3 | HIGH |
| Conditional on H_FW (newvector theorem) | Section 3a, lines 151-158 | N/A | NOTE |

**Nature of gaps**: The algebraic infrastructure is sound. The main
dependency is H_FW (the newvector test-vector theorem from the local
newform literature), which is a cited theorem input, not a gap in
the local reasoning. The proof explicitly notes this conditionality.

### Problem 8: Lagrangian smoothing of polyhedral surfaces

| Gap | Location | Script Task | Priority |
|-----|----------|-------------|----------|
| Maslov index = 0 (extend 998/998) | Section 4, lines 157-176 | P8-T1 | HIGH |
| Vertex spanning lemma | Section 3, line 89 | P8-T2 | HIGH |
| Symplectic direct sum decomposition | Section 3, Theorem v2 | P8-T3 | HIGHEST |
| Winding number explicit computation | Section 4, "back-and-forth" | P8-T4 | HIGH |

**Nature of gaps**: The proof is algebraic and correct, but the Maslov
winding argument was informal ("back-and-forth path, winding 0").
P8-T4 fills this with an explicit angle computation. The edge
nondegeneracy (forward reference at line 89) is resolved by the
vertex spanning lemma (proved below it).

### Problem 9: Polynomial detection of rank-1 scaling

| Gap | Location | Script Task | Priority |
|-----|----------|-------------|----------|
| Explicit witness det(M) = -24 | Section 4, lines 134-156 | P9-T1 | HIGHEST |
| Universal converse (all non-rank-1 lambda) | Section 4 | P9-T2 | HIGHEST |
| Forward direction (rank-1 => minors vanish) | Section 3 | P9-T3 | HIGH |
| n=5 to n>=5 extension (Lemma 4.1) | Section 4, lines 165-169 | P9-T4 | HIGH |
| Tensor factor compatibility lemma | Section 5, lines 189-197 | P9-T5 | HIGH |

**Nature of gaps**: The forward direction is proved. The critical gap
is the converse universality: the proof only exhibits one specific
non-rank-1 lambda witness, but needs to hold for ALL non-rank-1
lambda. P9-T2 probes this numerically. If 0 violations across
thousands of random lambda, the converse is strongly supported.
Formal closure requires an algebraic argument (e.g., specialization
of the polynomial nonvanishing statement).

## Running

```bash
cd /home/joe/code/futon6
python scripts/codex-unified-repair.py --seed 42

# Run specific problems:
python scripts/codex-unified-repair.py --problems 8 9

# Different seed:
python scripts/codex-unified-repair.py --seed 123
```

## Expected outputs

- `data/first-proof/codex-unified-repair-results.json` (detailed per-task results)
- `data/first-proof/codex-unified-repair-verification.md` (summary report)

## What to look for in results

### P1
1. **P1-T1**: Young's inequality C_eps = 27/(256*eps^3) verified? All samples
   satisfy |x|^3 <= eps*x^4 + C_eps?
2. **P1-T2**: Wick expansion matches direct computation? Discrepancy < 1e-8?
   Cubic term dominates?
3. **P1-T3**: Equivalence chain logic closes? The gap (pushforward preserves ~)
   is a single missing lemma, not a mathematical error.

### P2
1. **P2-T1**: Laurent ring units are monomials? No false units found?
2. **P2-T2**: PID argument confirmed? (Standard algebra, not really in question.)
3. **P2-T3**: Equivariance ratio near 1.0? (Discrete approximation, exact match
   not expected, but ratio should be within 0.3 of 1.0.)

### P8
1. **P8-T1**: Maslov index = 0 for ALL 10000 valid configurations?
   (0 violations expected; any violation would falsify the theorem.)
2. **P8-T2**: All configs spanning R^4? (0 non-spanning expected.)
3. **P8-T3**: Symplectic decomposition holds for all configs? Off-block entries
   vanish, a and b nonzero?
4. **P8-T4**: Explicit winding computation confirms total = 0 via angle cancellation?

### P9
1. **P9-T1**: det(M) = -24 exactly? det(Omega) = 0? Lambda not rank-1 (ratio
   contradiction)?
2. **P9-T2**: **CRITICAL** — 0 violations across 2000 random non-rank-1 lambda?
   If any lambda has ALL tested minors vanishing, that's a potential counterexample
   to the converse. Report the lambda and cameras for investigation.
3. **P9-T3**: All rank-1 lambda give vanishing minors? (0 violations expected.)
4. **P9-T4**: Extension works for n=6,7,8? (All non-rank-1 lambda detected via
   5-camera subsets?)
5. **P9-T5**: Tensor factorization: forward (rank-1 => all matricizations rank 1)
   and converse (all matricizations rank 1 => rank 1) both hold?

## Repair verdicts (expected after Codex run)

| Problem | Expected Verdict | Remaining Work |
|---------|-----------------|----------------|
| P1 | All tasks verified | Add explicit pushforward-preserves-~ lemma to proof text |
| P2 | All tasks verified | H_FW conditionality is documented, not a gap |
| P8 | All tasks verified | Winding computation fills the informal gap |
| P9 | T1-T5 verified, but converse needs formal proof | Write algebraic specialization argument for universal converse |

## Interaction with other verification

- P6 has its own verifier chain (C7, C8, C9) — not included here.
- P7 is excluded (topological gaps, not numerically testable).
- P3, P4, P5, P10 are not covered (different gap structures).
