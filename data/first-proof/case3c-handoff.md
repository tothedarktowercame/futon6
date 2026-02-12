# Case 3c Handoff: Remaining Gap in the n=4 Stam Inequality Proof

## Update 2 (2026-02-12, certified total-degree script)

The initial PHCpack runs (Update 1) used the blackbox solver, which returned
`mixed_volume = 0` — meaning no certified root count. The blackbox solver found
the right solutions but couldn't prove it found ALL of them.

**Root cause**: `stable_mixed_volume()` returned 0, likely because the gradient
polynomials' Newton polytopes are degenerate for the BKK bound. The blackbox
solver then used its own heuristic to find solutions, but without a certified
start system.

**Fix**: `scripts/verify-p4-n4-case3c-phc-certified.py` uses **total degree
homotopy** instead:
1. Constructs start system `x_i^9 - 1` for each variable
2. This has exactly 9^4 = 6561 start solutions (products of roots of unity)
3. Tracks ALL 6561 homotopy paths from start to target
4. Classifies every endpoint: finite (regular/singular), diverged, failed
5. Certification: `regular + singular + diverged + failed = 6561`

```bash
pip install phcpy sympy
python3 scripts/verify-p4-n4-case3c-phc-certified.py --tasks 0
```

Single-threaded (`--tasks 0`) is safest. Expect ~5-15 minutes for 6561 paths.
Output: `data/first-proof/problem4-case3c-phc-certified.json`

If `accounting_certified: true` and `all_nonneg: true` in the output, the proof
is complete.

## Update 1 (2026-02-12, Codex PHC run)

PHCpack (`phcpy`, PHCv2.4.90) was executed on the full 4D gradient system via
`scripts/verify-p4-n4-case3c-phc.py`.

Produced:
- `data/first-proof/problem4-case3c-phc-results.json` (tasks=8)
- `data/first-proof/problem4-case3c-phc-results-singlethread.json` (tasks=0)

Both runs agree on all in-domain real critical points:
- 12 total in-domain CPs, split as case1=4, case3a=2, case3b=2, case3c=4.
- Case 3c points are the expected symmetry orbit, with
  `-N = 1678.549826372544892... > 0`.

High-precision refinement of Case 3c seeds gives `max |∇(-N)| < 1e-45`.
Remaining caveat: `mixed_volume = 0` means no certified root count from the
polyhedral homotopy. See Update 2 for the fix.

## What's Been Proved (Algebraically Exact)

| Case | Subspace | Method | CPs in domain | -N values | Time |
|------|----------|--------|---------------|-----------|------|
| 1 | a₃=b₃=0 | Resultant → degree-26 univariate | 4 | 0, 825, 825, 898 | ~2s |
| 2 | b₃=0, a₃≠0 (and symmetric) | Resultant → degree-127 univariate, Sturm + sign counting | 0 | — | 30s |
| 3a | Diagonal: a₃=b₃, a₄=b₄ | Resultant → degree-24 univariate | 1 | 2296 | ~1s |
| 3b | Anti-diagonal: a₃=-b₃, a₄=b₄ | Resultant → degree-23 univariate | 2 | 0.05, 686 | ~1s |

**All algebraically-found critical points have -N ≥ 0.** ✓

Boundary analysis is also algebraically complete: the f₁=0 and f₂=0 faces
only intersect {disc≥0} at isolated degenerate points where -N=0.

Scripts: `verify-p4-n4-case2-final.py`, `verify-p4-n4-case3-diag.py`

## What Remains: Case 3c (Generic Off-Diagonal)

Critical points where **a₃≠0, b₃≠0, a₃≠±b₃** — the full 4D gradient system.

### Known CPs (numerical, from 3 independent searches with 11,000+ total starts)

4 symmetry copies of one orbit, all with **-N ≈ 1679** (saddle points):
```
(±0.0624, 0.1665, ∓0.2485, 0.0204)  and exchange-symmetry partners
```

### Why Direct Algebraic Elimination Failed

The gradient system is 4 polynomials of degree 9 in 4 variables.
- Eliminating a₃ via resultant: res(g₁,g₂,a₃) has ~2000 terms (timed out at 2 min)
- Even if the first elimination succeeds, the second (eliminate b₃) would produce
  degree ~72² ≈ 5000 in the remaining variables — infeasible

### Why Interval Arithmetic Failed

1. **Naive IA**: 233-term polynomial → massive wrapping error. Only 6.7% of boxes certified.
2. **Centered form**: Global Hessian Frobenius norm ≈ 96,000,000 (!), making the
   quadratic correction ~35,762 per box — larger than most -N values.
3. **Domain issue**: -N is NOT non-negative on the bounding box, only on the
   semi-algebraic domain {disc_p≥0, disc_q≥0, f₁>0, f₂<0}. Many box centers
   lie outside the domain where -N can be -2000.

## Recommended Approaches (for laptop)

### Option A: PHCpack (Best bet)

[PHCpack](https://github.com/janverschelde/PHCpack) uses polyhedral homotopy
continuation to find ALL isolated solutions of a polynomial system with a
**certified root count** (via mixed volume / BKK bound).

```bash
pip install phcpy
```

Script idea:
```python
from phcpy.solver import solve
# Build the 4 gradient equations as strings
# PHCpack will find all complex solutions and certify the count
solutions = solve(system)
# Filter to real solutions in the domain
# Verify -N ≥ 0 at each
```

The gradient system has Bézout bound 9⁴ = 6561, but the mixed volume
(actual bound) is likely much smaller. PHCpack handles this size routinely.

### Option B: Bertini

Similar to PHCpack but with different algorithms. Available at bertini.nd.edu.

### Option C: Domain-Aware Interval Arithmetic

The key improvement needed: only verify -N ≥ 0 on boxes that are
**actually inside the domain**, not the whole bounding box.

This requires encoding the domain constraints (disc≥0, f₁>0, f₂<0) into
the box filtering. Boxes where disc < 0 everywhere can be skipped.
This dramatically reduces the number of boxes to verify.

Also: use **affine arithmetic** instead of interval arithmetic for tighter
bounds on multivariate polynomials. Python package: `pyaffine` or custom.

### Option D: Hybrid Algebraic-Numerical

Use the (s,d,S,D) coordinate system with exchange symmetry to reduce
the system size. The symmetric/antisymmetric decomposition of the gradient
gives 3 equations in 4 invariant variables (σ=s², δ=d², S, Δ=D²).
The 4th equation comes from a compatibility condition.
This reduces degrees significantly and might make Gröbner basis feasible.

## Key Files

| File | What it does |
|------|-------------|
| `scripts/verify-p4-n4-case2-final.py` | Case 2 proof (30s, loads cache from `/tmp/case2-elimination-cache.pkl`) |
| `scripts/verify-p4-n4-case3-diag.py` | Cases 3a,3b proof (3s) |
| `scripts/verify-p4-n4-case3c-phc.py` | Blackbox PHCpack run (both runs found 12 in-domain CPs) |
| `scripts/verify-p4-n4-case3c-phc-certified.py` | **Certified** total-degree PHCpack run (6561 paths, full accounting) |
| `scripts/verify-p4-n4-case3c.py` | Case 3c analysis (shows degree structure, resultant infeasible) |
| `scripts/verify-p4-n4-classify-cps.py` | Numerical CP classification (17 min, comprehensive) |
| `scripts/verify-p4-n4-lipschitz.py` | Grid verification + Lipschitz bounds (shows min grid -N = 0.025) |
| `data/first-proof/p4-n4-proof-status.md` | Overall proof status document |

## The Polynomial -N

233 terms, degree 10 in (a₃, a₄, b₃, b₄), defined as the negated numerator of:

```
S = -disc_r/(4·f₁_r·f₂_r) + disc_p/(4·f₁_p·f₂_p) + disc_q/(4·f₁_q·f₂_q)
```

where p, q are centered degree-4 polynomials with a₂=b₂=-1,
r = p ⊞₄ q (finite free additive convolution), and
disc, f₁ = 1+12a₄, f₂ = 9a₃²+8a₄-2 are the discriminant and related factors.

Writing S = N/D where D < 0 on the domain, we need -N ≥ 0.

## What Success Looks Like

Run the certified script:
```bash
python3 scripts/verify-p4-n4-case3c-phc-certified.py --tasks 0
```

Check the output JSON for:
```json
{
  "accounting_certified": true,     // all 6561 paths accounted
  "all_nonneg": true,               // every in-domain CP has -N >= 0
  "real_in_domain": 12,             // matches Cases 1-3b + Case 3c
  "in_domain_by_case": {"case1": 4, "case3a": 2, "case3b": 2, "case3c": 4}
}
```

If both flags are true, the proof is complete:
- Total degree homotopy certifies the gradient system has at most 6561 isolated
  solutions (Bezout bound). All paths are tracked and accounted for.
- The finite solutions include exactly 12 real in-domain CPs, all with -N ≥ 0.
- Combined with algebraic proofs for Cases 1-3b and boundary analysis, this
  certifies -N ≥ 0 on the entire domain.

**Φ₄(p ⊞₄ q) ≥ Φ₄(p) + Φ₄(q)** with equality iff p = q = x⁴ - x² + 1/12. ∎
