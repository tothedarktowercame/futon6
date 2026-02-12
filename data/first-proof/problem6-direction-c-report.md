# Problem 6: Direction C Probe Report (Fixed-Block Interlacing)

Date: 2026-02-12
Author: Codex

## Direction C: Fixed-block interlacing / average characteristic polynomial

**What was tried:**
Implemented a trajectory-level diagnostic for the dispatch Direction C:
for each Case-2b step, define
`p_v(x)=det(xI-Y_t(v))`, compute
`pbar_t(x)=(1/r_t) sum_v p_v(x)`, then compare
`largest_real_root(pbar_t)` against `min_v ||Y_t(v)||` and test pairwise
interlacing rates among the `p_v` root sets.

Script added:
- `scripts/verify-p6-gpl-h-direction-c.py`

Commands run:
1. `python3 scripts/verify-p6-gpl-h-direction-c.py --nmax 24 --eps 0.12 0.15 0.2 0.25 0.3 --c-step 0.5`
2. `python3 scripts/verify-p6-gpl-h-direction-c.py --nmax 32 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --c-step 0.5`

Additional active-only check (inline diagnostic):
- Recomputed Direction C metrics restricting to vertices with nonzero load
  (`||Y_t(v)|| > 1e-10`) to avoid trivial early-step zero-load effects.

---

**What happened:**
The averaged characteristic polynomial route gives useful quantitative control
but does not yet provide a theorem-level bridge.

Main outcomes from script pass (`n<=32`):
- Case-2b instances analyzed: 93
- Steps analyzed: 227
- `max_t min_v ||Y_t(v)|| = 0.740741` (still < 1 in these tests)
- `max largest_real_root(pbar_t) = 0.821995` (< 1 in this raw pass)
- Full sampled pairwise interlacing held in 206/227 steps; failed in 21/227.

Active-only check (nonzero-load vertices only):
- Active-step rows: 125
- `max min_active_score = 1.0000000000000018`
- `max largest_real_root(pbar_active) = 1.0060437842918053`
- Steps with `largest_real_root(pbar_active) < 1`: 119/125
- Full sampled pairwise interlacing: 110/125
- Worst sampled pairwise interlacing rate: ~0.758

So once we remove trivial zero-load vertices, both phenomena appear:
1. nontrivial interlacing failures,
2. occasional averaged-polynomial root crossing 1.

---

**Exact failure point (current):**
The MSS step

`largest_root(pbar_t) < 1  =>  exists v with ||Y_t(v)|| < 1`

requires a common-interlacing hypothesis (or equivalent real-stability
closure) for the family `{p_v}` at each step. We do not have this.

Empirical evidence against taking common interlacing as automatic:
- sampled pairwise interlacing is not universal (rates down to ~0.758).

Even if one used `pbar` as a heuristic bound, active-only diagnostics show
`largest_real_root(pbar)` can reach/exceed 1 in hard steps.

Therefore, the current Direction C route does not close GPL-H directly.

---

**Partial results:**
1. New computable witness family for Direction C:
   `p_v`, `pbar`, root-gap `largest_root(pbar)-min_v score(v)`, and
   per-step sampled interlacing rate.

2. Quantitative narrowing:
   - In many tested steps, `largest_root(pbar)` is below 1 and not far above
     the best score; this suggests the polynomial-average perspective carries
     genuine spectral-spreading information not captured by matrix averaging.

3. Structural split identified:
   - The obstacle is localized to active nonzero-load subsets and to missing
     common-interlacing structure; early-step zero-load vertices can mask this.

---

**Surprises:**
1. On the raw (all-vertices) family, `largest_root(pbar)` stayed below 1 in all
   tested steps up to `n=32`, even when common interlacing was not universal.
2. On active-only families, occasional `largest_root(pbar) >= 1` appears,
   indicating the raw all-vertex averaging can hide the hard regime.

---

**New dead ends discovered:**
1. "Assume common interlacing by analogy" appears unjustified for the fixed
   grouped family `{Y_t(v)}`. Sampling shows genuine interlacing failures.
2. "Use `largest_root(pbar)<1` as an empirical universal surrogate" is not
   robust in active-only hard steps.

---

**Verdict:** FAILS AT interlacing-hypothesis step (for full closure).

Direction C currently **narrows the gap quantitatively** and provides a useful
spectral diagnostic, but does not prove GPL-H. To close via Direction C, one
would need an additional theorem establishing an interlacing/real-stability
property for the fixed grouped family `{p_v}` (or a replacement mechanism that
bounds `min_v ||Y_t(v)||` from polynomial data without common interlacing).
