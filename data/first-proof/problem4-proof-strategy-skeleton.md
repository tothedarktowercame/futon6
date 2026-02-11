# Problem 4 Proof Strategy Skeleton

Date: 2026-02-11  
Status: Research synthesis only (no complete proof yet)

## Goal

For monic real-rooted degree-\(n\) polynomials \(p,q\), prove
\[
\frac{1}{\Phi_n(p\boxplus_n q)} \ge \frac{1}{\Phi_n(p)} + \frac{1}{\Phi_n(q)},
\]
where
\[
\Phi_n(p)=\sum_i\left(\sum_{j\ne i}\frac{1}{\lambda_i-\lambda_j}\right)^2.
\]

## Verified Inputs We Can Use

1. MSS finite additive convolution formula and Haar model (arXiv:1504.00350):
- coefficient weights \(\frac{(n-i)!(n-j)!}{n!(n-k)!}\),
- representation \(p\boxplus_n q = \mathbb{E}_Q[\chi_x(A+QBQ^\*)]\).

2. Finite cumulants (Arizmendi-Perales, arXiv:1611.06598):
- explicit coefficient-cumulant conversion,
- additivity of finite cumulants under \(\boxplus_n\),
- finite-to-free convergence.

3. Root majorization structure (arXiv:1811.06382, arXiv:2108.07054):
- \(\lambda(p\boxplus_n q)\prec \lambda(p)+\lambda(q)\),
- submodularity/diminishing-returns patterns for root transforms,
- finite-free-position matrix realization and further majorization consequences.

## Recommended Route: B + C Hybrid

1. **Re-express target in symmetric-root coordinates**  
Define \(F(\lambda_1,\dots,\lambda_n):=1/\Phi_n\).  
Use translation and scaling reductions to normalize a root configuration.

2. **Bridge \(\boxplus_n\) to a root-order inequality class**  
Use majorization/submodularity results for \(\boxplus_n\) to control how root vectors move.

3. **Find sufficient property of \(F\)**  
Establish a condition (Schur-convexity/concavity on a constrained region, or a submodular inequality along a valid interpolation path) that implies
\[
F(\lambda(p\boxplus_n q))\ge F(\lambda(p))+F(\lambda(q)).
\]

4. **Close with MSS/AP structure**  
Use bilinearity/cumulant additivity to justify algebraic steps that require decomposing \(\boxplus_n\) operations.

## Secondary Route: Cumulant-Coordinate Route A (Guarded)

1. Express \(F\) in finite cumulants \(\kappa^{(n)}\) using AP formulas.
2. Check whether a restricted convexity/superadditivity statement holds on the real-rooted image of coefficient space.
3. If true only on that image (not globally), state and prove domain-restricted convexity.

This route is plausible but currently unsupported by theorem-level evidence.

## Open Lemmas (Current Gaps)

1. A usable monotonicity principle linking \(F=1/\Phi_n\) to known majorization relations.
2. A submodularity-style inequality directly for \(F\), analogous to largest-root transforms.
3. A precise matrix/Haar identity turning \(\Phi_n\) into an expectation-controlled quantity.

## Immediate Next Checks

1. Symbolic \(n=3,4\): compute \(F\) on majorization-comparable root tuples to test Schur behavior.
2. Numerical path tests: interpolate between \(\lambda(p)+\lambda(q)\) and \(\lambda(p\boxplus_n q)\) and test discrete second differences of \(F\).
3. Attempt a derived inequality from known submodularity transforms to bound \(F\) from below.
