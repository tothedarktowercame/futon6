# Problem 4 Stam-for-r: Codex Task Report

Date: 2026-02-13
Base commit: `e4882e0`

## Completed Artifacts

- Task 1 script: `scripts/prove-p4-stam-for-r-n3-symbolic.py`
- Task 1 data: `data/first-proof/problem4-stam-for-r-n3-results.json`
- Task 2 script: `scripts/verify-p4-score-projection.py`
- Task 2 data: `data/first-proof/problem4-score-projection-results.json`
- Task 4 script: `scripts/analyze-p4-n4-sos-bruteforce.py`
- Task 4 data: `data/first-proof/problem4-n4-sos-bruteforce-results.json`
- Task 3 analysis: `data/first-proof/problem4-ct-analysis.md`

## Task 1: n=3 Symbolic Stam-for-r

Status: symbolic reduction completed; no counterexample found in global searches.

Key formula (centered cubic `x^3 + a2 x + a3`):

`r(a2,a3) = 3(-2 a2^3 + 27 a3^2) / (a2 (4 a2^3 + 27 a3^2))`

In positive-scale coordinates (`a2=-s<0`, `a3=u`):

`1/r = s(4s^3 - 27u^2) / (3(2s^3 + 27u^2))`

Reduction:

- Write `u = c s^(3/2) x`, `v = c t^(3/2) y` with `c=2/sqrt(27)`, `x,y in [-1,1]`.
- Let `lam = s/(s+t)`.
- Stam-for-r becomes `F(lam,x,y) >= 0` on `[0,1] x [-1,1]^2`.

Search results (`problem4-stam-for-r-n3-results.json`):

- differential evolution minimum: `-1.11e-16` (numerical zero)
- random search (3M): minimum `3.29e-08`
- dense slices: minima at numerical zero

Interpretation: reduced inequality is numerically consistent with exact nonnegativity; symbolic closed form obtained.

## Task 2: Score Decomposition / Cauchy-Schwarz Route

Status: simple linear/affine projection models are strongly falsified.

Across `n=3..6` with 40k samples each:

- global affine fit test `R^2` ~ `0.01-0.02`
- global affine relative error mean ~ `0.71-0.87`
- diagonal projection models have extremely high residuals
- per-sample scalar blend also poor (error grows with `n`)

Interpretation: if a finite Blachman/score-projection proof exists, it is not captured by linear or diagonal affine decompositions in sorted-score coordinates.

## Task 4: n=4 SOS / Brute Force

Status: symbolic reduction + brute-force/global search completed; full SOS solver blocked by environment.

Symbolic Stam numerator (`a2=b2=-1`) structure:

- degree: `10`
- monomials: `233`
- symmetries: swap (`p<->q`) and reflection (`a3,b3 -> -a3,-b3`)

Environment blockers:

- `cvxpy` unavailable
- `scs` unavailable

Brute-force/global results (30k normalized samples + DE search):

- Stam surplus minimum found: `7.20e-04` (random), `2.38e-11` (global optimization)
- Stam-for-r gap minimum found: `3.45e-02`
- no violations observed in this run

Interpretation: n=4 remains consistent with positivity; solver infrastructure is the current blocker for a full SOS certificate in this workspace.

## Task 3: Category-Theory Organization

See `problem4-ct-analysis.md`.

Main output:

- formalizes `(RR_n, ⊞_n)` and functionals `1/Φ`, `1/r` as lax monoidal candidates
- frames the open step as finding a natural transformation from `1/r` to `1/Φ`
- ties Task 2 failure to the need for nonlinear, configuration-dependent projection morphisms
- gives concrete follow-up algebraic tests for correction factors `C_n`

## Overall Conclusion

1. n=3 Stam-for-r has a clean symbolic reduction and strong global numerical confirmation.
2. The straightforward score-projection linear route is not viable as-is.
3. n=4 evidence is strong but a certified SOS proof is blocked by missing SDP tooling.
4. CT framing is useful for narrowing proof obligations, especially through transformation laws between `1/r` and `1/Φ`.
