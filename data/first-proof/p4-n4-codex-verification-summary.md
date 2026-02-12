# P4 n=4 Codex Verification Summary (Non-PHC)

## Run metadata
- Date: 2026-02-12
- Target commit: `3c69d90` (`origin/master`)
- Verification worktree: `/tmp/futon6-3c69d90`
- Method: rerun cited `verify-p4-n4-*` scripts and compare outputs against `data/first-proof/p4-n4-proof-status.md`

## Commands run
- `python3 scripts/verify-p4-n4-algebraic.py`
- `python3 scripts/verify-p4-n4-case2-final.py`
- `python3 scripts/verify-p4-n4-case3-diag.py`
- `python3 scripts/verify-p4-n4-global-min.py`
- `python3 scripts/verify-p4-n4-global-min2.py`
- `python3 scripts/verify-p4-n4-taylor-bound.py`
- `python3 scripts/verify-p4-n4-lipschitz.py` (stopped after grid/Lipschitz stages completed)
- `python3 scripts/verify-p4-n4-critical-points.py` (stopped after Case 1 completed)
- `python3 scripts/verify-p4-n4-case3c.py` (stopped after elimination bottleneck reproduced)
- `python3 scripts/verify-p4-n4-sos-d12-scs.py` (dependency missing)
- `python3 scripts/verify-p4-n4-sos-reduced.py` (dependency missing)

## Artifacts
- Raw logs were captured under `/tmp` during the run (for example: `/tmp/p4-n4-algebraic.log`, `/tmp/p4-n4-case2-final.log`, `/tmp/p4-n4-case3-diag.log`, `/tmp/p4-n4-taylor-bound.log`, `/tmp/p4-n4-global-min.log`, `/tmp/p4-n4-global-min2.log`, `/tmp/p4-n4-lipschitz.log`, `/tmp/p4-n4-critical-points.log`).

## Headline result
- Non-PHC algebraic core is reproduced.
- Case 1, Case 2, Case 3a, Case 3b all show in-domain critical points with `-N >= 0`.
- Equality-point Hessian is positive definite with eigenvalues `{3/4, 21/8, 6, 8}`.
- Large grid verification reproduces `529,984` in-domain points and `min -N = 0.025017`.
- Remaining rigorous gap is still Case 3c exhaustive certification (PHC/Gröbner path).

## Per-claim verification outcomes
- `Phi_4 * disc` identity: verified symbolically.
- Surplus numerator construction: reproduced (233 terms, degree 10).
- Case 1 (`a3=b3=0`) critical points: reproduced.
- In-domain CPs reproduced: `(1/12,1/12)` with `-N ~ 0`, plus asymmetric pair `-N ~ 824.57`, plus diagonal `-N ~ 898.16`.
- Case 2 (`b3=0, a3!=0`) elimination/Sturm proof: reproduced with `Interior critical points found: 0`.
- Case 3a/3b exact subcases: reproduced with all reported interior CPs having `-N >= 0`.
- Equality-point local structure: reproduced (`grad=0`, Hessian PD).
- Grid evidence: reproduced (`50^4` grid, `529,984` domain points, `min -N = 0.025017`).
- Case 3c algebraic bottleneck: reproduced; direct elimination stalls at `res(g1,g2,a3)`.

## Reproducibility notes
- `verify-p4-n4-case3c.py` reached `Step 2: Compute res(g1, g2, a3)...` and remained there without completion in bounded runtime; this matches the handoff narrative that direct elimination is infeasible.
- `verify-p4-n4-taylor-bound.py` completed and reported a tiny negative optimum (`-3.81987775e-11`) in outer-region local optimization, consistent with numerical tolerance effects near boundaries rather than a structural counterexample.
- `verify-p4-n4-global-min2.py` prints a "potential counterexample", but the point lies effectively on a singular boundary (`b4 ~= -1/12`, where denominators degenerate), so it is not evidence against the interior-domain claim structure.

## Blockers in this environment
- `verify-p4-n4-sos-d12-scs.py` failed: `ModuleNotFoundError: No module named 'scs'`.
- `verify-p4-n4-sos-reduced.py` failed: `ModuleNotFoundError: No module named 'cvxpy'`.
- These block re-running the SOS infeasibility claims in this specific runtime.

## Conclusion
- The commit `3c69d90` restructuring is consistent with the claim that the remaining rigorous dependency is Case 3c certification.
- All other major non-PHC components were reproduced or partially reproduced with consistent outputs.
- Final closure still requires certified exhaustive Case 3c root accounting (PHCpack/Bertini or equivalent).
