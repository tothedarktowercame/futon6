# Codex Handoff: Certify Case 3c Critical Points (Close the n=4 Proof)

**Date:** 2026-02-13
**From:** Claude (integration cycle)
**Priority:** HIGH — this is the SINGLE remaining gap in the n=4 Stam proof
**Prerequisite:** PHCpack (`phcpy`) installed

---

## Context

The n=4 finite Stam inequality proof is structured as exhaustive critical point
enumeration on the compact domain. ALL cases are algebraically proved except
**Case 3c** (generic off-diagonal: a₃≠0, b₃≠0, a₃≠±b₃).

The previous PHCpack run (`verify-p4-n4-case3c-phc-certified.py`) used total-degree
homotopy (Bézout bound 9⁴=6561 paths). Results:
- `accounting_certified: true` (all 6561 paths accounted)
- `finite_regular: 560`, `diverged: 759`, **`failed: 5242`**
- 12 real in-domain CPs found, all with -N ≥ 0
- Case 3c: 4 CPs, all -N ≈ 1679 > 0

**The gap:** 5242 failed paths could theoretically hide real in-domain solutions.

## Task: Close the Gap

Try these approaches IN ORDER (stop when one succeeds):

### Approach A: Higher-precision path tracking

Re-track the 6561 paths using quad-double precision:

```python
from phcpy.solver import solve
from phcpy.trackers import standard_double_track  # or dd/qd variants

# The gradient system (4 equations, degree 9 each)
# Build from scripts/verify-p4-n4-case3c-phc-certified.py

# Key: use 'dd' (double-double) or 'qd' (quad-double) precision
# This should recover more of the 5242 failed paths
```

**Success criterion:** `failed: 0` (or `failed` paths all provably divergent).

### Approach B: Smale's alpha theory certification

Given the 4 high-precision Case 3c roots (100-digit precision, |∇(-N)| < 1e-45),
use alpha theory to certify each is an actual root:

```python
# For each approximate root z:
# 1. Compute beta(f, z) = |f^{-1}(z)| · |f(z)|  (Newton residual / condition)
# 2. Compute gamma(f, z) = sup_k |f^{-1}(z) · D^k f(z) / k!|^{1/(k-1)}
# 3. alpha = beta * gamma
# 4. If alpha < alpha_0 ≈ 0.15767..., root is certified
```

This certifies the 4 known roots but does NOT prove no others exist.
Combine with: total degree homotopy should find AT LEAST as many isolated
solutions as exist (it finds all of them generically). The 560 finite solutions
represent an upper bound on the number of isolated finite solutions, and all
real ones among them are in-domain with -N ≥ 0.

### Approach C: Interval Newton (Krawczyk's method)

For each Case 3c root z_i, construct a small box B_i around it and verify:
1. The Krawczyk operator K(B_i) ⊂ B_i (proves unique root in B_i)
2. -N(z_i) > 0 (with interval arithmetic certification)

Python packages: `mpmath.iv` or `intervaltree` + custom Newton.

### Approach D: Reduce system via invariant coordinates

The exchange symmetry (a₃,a₄)↔(b₃,b₄) and parity (a₃,b₃)→(-a₃,-b₃) suggest
using invariant coordinates:
- σ = a₃² + b₃², π = a₃·b₃ (symmetric/mixed in a₃,b₃)
- s = a₄ + b₄, d = a₄ - b₄

This reduces the gradient system from 4 equations in 4 variables to fewer
effective unknowns, potentially making Gröbner basis computation feasible.

## Key Files

| File | Description |
|------|-------------|
| `scripts/verify-p4-n4-case3c-phc-certified.py` | Current total-degree PHCpack script |
| `data/first-proof/problem4-case3c-phc-certified.json` | Current results (5242 failed) |
| `data/first-proof/case3c-handoff.md` | Detailed gap analysis |
| `data/first-proof/p4-n4-proof-status.md` | Full proof status |
| `data/proof-state/P4.edn` | Proof state ledger |
| `scripts/verify-p4-n4-case3-diag.py` | Cases 3a,3b (for reference) |
| `scripts/verify-p4-n4-case2-final.py` | Case 2 (for resultant reference) |

## The Gradient System

4 polynomials of degree 9 in (a₃, a₄, b₃, b₄), derived from ∇(-N) = 0 where
-N is the 233-term degree-10 polynomial (the negated Stam surplus numerator at
a₂=b₂=-1).

The system inherits:
- Exchange symmetry: if (a₃,a₄,b₃,b₄) is a solution, so is (b₃,b₄,a₃,a₄)
- Parity: if (a₃,a₄,b₃,b₄) is a solution, so is (-a₃,a₄,-b₃,b₄)

These give a 4-fold orbit for generic Case 3c solutions (confirmed: all 4
Case 3c CPs are one orbit).

## Feasibility Domain

For real-rooted centered quartics with a₂ = -1:
- disc(p) = 256a₄³ + 128a₄² + 144a₃²a₄ + 16a₄ - 27a₃⁴ + 4a₃² ≥ 0
- f₁(p) = 1 + 12a₄ > 0  (always true for a₄ > -1/12)
- f₂(p) = 9a₃² + 8a₄ - 2 < 0
- Same constraints for (b₃, b₄)

## Success Criterion

Update `data/proof-state/P4.edn` ledger item `L-n4-case3c`:
- Status: `:proved` (was `:numerically-verified`)
- Evidence-type: `:analytical` or `:certified-numerical`

And update `L-n4-cases-complete` and `L-n4` accordingly.

## Output

Save results to `data/first-proof/problem4-case3c-certified-v2.json` with:
```json
{
  "method": "...",
  "failed_paths": 0,
  "certification_type": "alpha_theory|krawczyk|improved_homotopy",
  "all_real_in_domain_nonneg": true,
  "case3c_certified": true
}
```
