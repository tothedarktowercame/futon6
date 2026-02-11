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

1. ~~A usable monotonicity principle linking \(F=1/\Phi_n\) to known majorization relations.~~
   **ELIMINATED**: F is not Schur-convex or Schur-concave. Majorization alone
   does not control F.
2. ~~A submodularity-style inequality directly for \(F\), analogous to largest-root transforms.~~
   **ELIMINATED**: F is not submodular in root coordinates (~50% violation rate).
3. A precise matrix/Haar identity turning \(\Phi_n\) into an expectation-controlled quantity.
   **STILL OPEN** — this is now the critical path.
4. (NEW) A **root-regularity** argument: show that ⊞_n increases the uniformity
   of root spacing in a way that controls 1/Phi_n. This is the qualitative
   mechanism (unevenly spaced roots have high Coulomb energy), but needs a
   quantitative formulation.

## Immediate Next Checks — COMPLETED

Verification script: `scripts/verify-p4-schur-majorization.py`

### Check 1: Schur-convexity/concavity (NEGATIVE)

F = 1/Phi_n is **NEITHER Schur-convex NOR Schur-concave** in root coordinates.
Tested 3000 pairs per n with doubly-stochastic majorization:

| n | Schur-convex violations | Schur-concave violations |
|---|------------------------|-------------------------|
| 3 | 640 (21.3%)           | 2355 (78.5%)            |
| 4 | 904 (30.1%)           | 2094 (69.8%)            |
| 5 | 954 (31.8%)           | 2043 (68.1%)            |

F **tends** toward Schur-convex behavior (more spread => larger F) but with
significant exceptions. On the symmetric subfamily (-s, 0, s), F = 2s²/9 IS
monotone increasing, but asymmetric root configurations break this — e.g.,
(-2, 0.5, 1.5) has spread 6.5 but F=0.40, while (-1.5, 0, 1.5) has spread
4.5 but F=0.50.

**Key insight:** F depends on root REGULARITY (uniformity of spacing), not
just total spread. Unevenly spaced roots have high Coulomb energy (close
pairs dominate), giving small 1/Phi_n. This is exactly the pattern where
⊞_n helps: free convolution regularizes root spacing.

### Check 2: Interpolation paths (MIXED)

Along the majorization path from λ(p⊞q) to λ(p)+λ(q):

| n | monotone increasing | monotone decreasing | neither |
|---|--------------------|--------------------|---------|
| 3 | 833 (83.3%)        | 128 (12.8%)        | 39 (3.9%) |
| 4 | 660 (66.0%)        | 196 (19.6%)        | 144 (14.4%) |
| 5 | 532 (53.2%)        | 181 (18.1%)        | 287 (28.7%) |

F mostly increases from convolution roots to componentwise sum (consistent
with rough Schur-convexity along these paths), but this weakens at larger n.
Path second differences are mixed — no clean convexity/concavity.

### Check 3: Submodularity (NEGATIVE)

~49-50% violations for all n — effectively random. F is NOT submodular in
root coordinates.

### Revised Assessment

**Routes A and B+C both face obstacles from these results:**

- **Route A (cumulant convexity):** Global Schur-convexity fails, so any
  convexity argument must be domain-restricted (to the image of real-rooted
  polynomials under the cumulant map).
- **Route B+C (majorization):** The majorization λ(p⊞q) ≺ λ(p)+λ(q) alone
  is insufficient because F is not Schur-monotone. The proof MUST use the
  specific structure of how ⊞_n moves roots (regularization of spacing),
  not just the majorization inequality.

**Most promising direction remains:** a Haar/random-matrix argument that
directly controls Phi_n of the expected characteristic polynomial via
properties of the unitary orbit A + QBQ*, exploiting that averaging over
Q regularizes root spacing.
