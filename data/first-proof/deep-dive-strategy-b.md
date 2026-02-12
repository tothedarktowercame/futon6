# Deep Dive: Strategy B — Induction via Differentiation

**Date:** 2026-02-12
**Status:** Research survey and viability assessment
**Predecessor:** `problem4-ngeq4-proof-strategies.md` (Strategy B section)

---

## Executive Summary

Strategy B attempts to prove the finite free Stam inequality

    1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)

by induction on n, exploiting the exact algebraic identity

    (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)

which reduces degree-n convolution to degree-(n-1) convolution via
differentiation. The naive induction chain is BLOCKED because
1/Phi_n(p) < 1/Phi_{n-1}(p'/n) always (wrong direction at Step A).

This report investigates six sub-questions aimed at salvaging the
induction approach. The overall assessment: **induction via
differentiation alone is unlikely to yield a direct proof (probability
~10-15%), but specific sub-results — particularly alternative
functionals and the quantitative Phi ratio bound — could feed into
a hybrid proof strategy combining differentiation with the conjugate-
variable projection framework (Strategy A/D).**

---

## 1. Alternative Functionals

### The Problem

The functional 1/Phi_n is superadditive under ⊞_n (the target
inequality) but does NOT chain correctly through differentiation:
1/Phi_n(p) < 1/Phi_{n-1}(p'/n) always. We need F_n(p) satisfying:

(i)  F_n(p ⊞_n q) >= F_n(p) + F_n(q)  (superadditivity under ⊞_n)
(ii) F_n(p) >= F_{n-1}(p'/n)  OR  F_{n-1}(p'/n) >= F_n(p)
     (monotonicity through differentiation, in the CORRECT direction
     to close the induction)

### Candidate Analysis

**Candidate 1: Normalized inverse, 1/(n^2 * Phi_n)**

Rationale: Phi_n scales roughly as n^2 for "generic" configurations
(the number of pairs (i,j) contributing to the double sum grows as
n^2). Normalizing by 1/n^2 might correct the scaling mismatch between
degrees.

Assessment: The scaling exponent is configuration-dependent.
For equispaced roots on [-1,1], Phi_n ~ n^4 (each S_i ~ n, and
there are n terms). For clustered roots, Phi_n can be much larger.
A single power of n cannot correct the direction for all
configurations. **Unlikely to work.**

**Candidate 2: Log transform, log(1/Phi_n)**

Rationale: If superadditivity of 1/Phi_n held in a multiplicative
sense, log would convert it to ordinary additivity.

Assessment: The inequality 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)
is ADDITIVE in 1/Phi_n, not multiplicative. Taking log destroys the
additive structure. **Does not help.**

**Candidate 3: disc(p)^alpha / Phi_n (discriminant-weighted)**

Rationale: The discriminant disc(p) = prod_{i<j} (lambda_i - lambda_j)^2
measures total root separation. At n=2, 1/Phi_2 = disc(p)/2 exactly.
At n=3, Phi_3 * disc(p) = 18 * a_2^2 for centered cubics. A
discriminant-weighted functional might restore the correct scaling.

Assessment: This is the most promising candidate. The discriminant
satisfies disc(p) = prod_{i<j} (lambda_i - lambda_j)^2, and its
relationship to differentiation is classical:

    disc(p) = (-1)^{n(n-1)/2} * (1/a_n) * Res(p, p')

where Res denotes the resultant. By the Mahler-type inequality,
disc(p) and disc(p'/n) are related through a product formula
involving the root gaps. Specifically, if mu_1,...,mu_{n-1} are
roots of p'/n, then by resultant theory:

    disc(p) = (-1)^{n(n-1)/2} * n^n * prod_i p(mu_i) * (1/disc(p'/n))^?

The exact relationship is complicated. The key question is whether
disc(p)^alpha / Phi_n chains correctly for some alpha. This requires
careful computation that we have not yet performed.

For n=2: disc(p)/Phi_2 = 2 (constant), so alpha=0 works trivially.
For n=3: disc(p)/Phi_3 = 18*a_2^2/Phi_3^2? No — Phi_3 * disc = 18*a_2^2,
so disc/Phi_3 = 18*a_2^2/Phi_3^2 * Phi_3 = ... This needs numerical
exploration.

**Verdict: Worth numerical testing. Priority: MEDIUM-HIGH.**

**Candidate 4: Log-discriminant (log-energy)**

Define E_n(p) = sum_{i<j} log|lambda_i - lambda_j|. This is the
finite-dimensional analog of Voiculescu's free entropy chi(mu) =
int int log|x-y| dmu(x) dmu(y) (up to normalization). In the
infinite-dimensional limit, chi is superadditive under free convolution
(Voiculescu 1993, CMP 155).

Behavior under differentiation: By the electrostatic interpretation
(Gauss; Baez 2021; Marden, "Geometry of Polynomials"), differentiating
p moves roots inward according to the balance of electrostatic forces.
The critical points mu_i of p are equilibrium points of the log-potential
sum_j log|z - lambda_j|.

The change in E under differentiation is:

    E_{n-1}(p'/n) = sum_{i<j} log|mu_i - mu_j|

where mu_1,...,mu_{n-1} are roots of p'/n. By interlacing
(Rolle/Gauss-Lucas), the mu_i are more closely spaced than the
lambda_i. Therefore E_{n-1}(p'/n) < E_n(p) in general — the log-energy
DECREASES under differentiation. This is the SAME wrong direction
as for 1/Phi_n.

However, the normalized version E_n(p)/binom(n,2) might behave
differently, since we divide by the number of pairs. For n roots, there
are binom(n,2) pairs; for n-1 roots, binom(n-1,2) pairs. The per-pair
energy might increase.

**Verdict: Direction wrong for raw E_n. Normalized version worth testing.**

**Candidate 5: Renyi entropies / entropy powers**

In classical information theory, the Renyi entropy power inequality
(Bobkov-Chistyakov 2015; Savare-Toscani 2014) extends the Shannon
entropy power inequality to order-alpha Renyi entropies. The classical
Stam inequality 1/J(X+Y) >= 1/J(X) + 1/J(Y) is the Fisher-information
formulation.

For the polynomial/discrete setting, one could define Renyi-type
functionals of the root configuration, e.g.:

    R_alpha(p) = (1/(1-alpha)) * log(sum_i |S_i|^{2alpha} / (sum_i |S_i|^2)^alpha)

where S_i = sum_{j != i} 1/(lambda_i - lambda_j). At alpha=1 this
reduces to a normalized version of Phi_n. Different alpha might give
better chaining properties.

In free probability, Renyi-type extensions have not been developed
to the same extent as Shannon-type free entropy. The search found no
direct analog.

**Verdict: Speculative. Low priority unless a specific alpha shows
promise numerically.**

### Summary for Section 1

The most promising alternative functional is **disc(p)^alpha / Phi_n**
for a suitable alpha, followed by the **normalized log-energy
E_n(p)/binom(n,2)**. Both require numerical experimentation to
determine whether they chain correctly through differentiation while
maintaining superadditivity under ⊞_n.

---

## 2. Reversed Induction Direction (n-1 to n)

### The Idea

Instead of going n -> n-1 via differentiation (which gives the wrong
inequality direction), can we go n-1 -> n via some "anti-differentiation"
or lifting operation that commutes with ⊞?

### Anti-differentiation and Real-Rootedness

Integration of polynomials does NOT preserve real-rootedness in general.
If p(x) = (x-1)(x-2)(x-3), then int p dx = x^4/4 - 3x^3 + 11x^2/2 - 6x + C,
which may not be real-rooted for all C. The Hermite-Poulain theorem
and Polya-Schur theory characterize linear operators preserving
real-rootedness, but anti-differentiation is not among them.

More precisely: differentiation maps the cone of real-rooted monic
polynomials of degree n to real-rooted monic polynomials of degree n-1
(after normalization). This map is SURJECTIVE but NOT INJECTIVE —
many degree-n real-rooted polynomials can differentiate to the same
degree-(n-1) polynomial. The "inverse" is multi-valued and the fibers
are complicated.

### Does ⊞_n Commute with Any Lifting?

The exact identity (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n) goes from
degree n to degree n-1. For the reverse direction, we would need
an operator L: P_{n-1} -> P_n such that:

    L(p ⊞_{n-1} q) = L(p) ⊞_n L(q)

This is a very strong condition. By Marcus-Spielman-Srivastava
(2015, 2018), the finite free convolution is defined via the expected
characteristic polynomial E_U[det(xI - A - UBU*)]. For this to
commute with a lifting operator L, the lifting would need to
interact coherently with the Haar-unitary averaging.

One natural candidate: embedding p of degree n-1 as x*p(x) (adding
a root at 0). But x*p(x) ⊞_n x*q(x) != x*(p ⊞_{n-1} q)(x) in
general, because the ⊞_n operation couples the added root at 0 with
the other roots.

Another candidate: the "heat flow" direction. If exp(t*D^2/(2n)) maps
real-rooted degree-n polynomials to real-rooted degree-n polynomials
(which it does for positive t, by the connection to free convolution
with a semicircular; see Hall-Ho-Jalowy-Kabluchko 2022, arXiv:2202.09660),
then the BACKWARD heat flow (negative t) could serve as a kind of
lifting. But backward heat flow does not preserve real-rootedness, and
it does not change the degree.

### Gelfand-Tsetlin Patterns

Shlyakhtenko and Tao (2020, arXiv:2009.01882) established that
fractional free convolution powers are connected to the "minor process"
in random matrix theory. In this framework, a Gelfand-Tsetlin pattern
{lambda^{(k)}}_{k=1}^n gives eigenvalues at each level, with
interlacing: lambda^{(k-1)} interlaces lambda^{(k)}.

The derivative operation p -> p'/n corresponds to going DOWN one level
in the GT pattern. Going UP requires choosing a new root that interlaces
with the existing roots — this is the reverse operation. But the choice
is not canonical; it depends on the GT pattern structure.

Shlyakhtenko-Tao proved that the free entropy of the normalized free
convolution power mu^{boxplus k} is monotone NON-DECREASING in k.
Their variational description uses a Lagrangian:

    L(lambda_s, lambda_y) = log(lambda_y) + log(sin(pi * lambda_s / lambda_y))

This governs the "optimal" way to go between levels of the GT pattern.
If this variational structure could be made finite-n, it might provide
the correct lifting.

### Assessment

**Reversed induction is structurally difficult.** The fundamental
obstruction is that anti-differentiation of real-rooted polynomials is
not canonical (multi-valued fibers, real-rootedness not preserved).
No known lifting operation commutes with ⊞_n. The Gelfand-Tsetlin
variational structure of Shlyakhtenko-Tao is the closest conceptual
framework but has not been made finite-n.

**Probability of success via this sub-route: ~5%.**

---

## 3. Quantitative Phi_n / Phi_{n-1} Relationship

### The Data

From our numerical experiments (commit 827f4c6), the ratio
Phi_{n-1}(p'/n) / Phi_n(p) satisfies:

    0 < Phi_{n-1}(p'/n) / Phi_n(p) < 1

with the ratio bounded away from 1. Our data shows the ratio ranges
from near 0 (for polynomials with nearly equal root gaps) to
approximately 0.66 (for polynomials with highly unequal gaps).

### Theoretical Bounds

**Lower bound (trivial):** Phi_{n-1}(p'/n) > 0 whenever p'/n has
simple roots, which holds for any p with simple roots (by
interlacing, the derivative roots are distinct).

**Upper bound (conjectured):** Phi_{n-1}(p'/n) / Phi_n(p) < C(n) < 1
for some constant depending only on n. The sharp constant appears
to be:

    C(n) = (n-1)^2 / n^2

based on the electrostatic heuristic: each root force S_i involves
n-1 terms in the degree-n case and n-2 terms in the degree-(n-1) case,
and the root gaps contract by a factor related to (n-1)/n under
interlacing.

**Extremal configurations:** The ratio approaches its maximum when
roots are nearly equispaced (all gaps nearly equal), because in this
case the interlacing critical points are also nearly equispaced and
the force magnitudes scale uniformly. The ratio approaches 0 when
there is extreme scale separation (one root far from the others),
because the derivative polynomial "loses" the effect of the outlier
root disproportionately.

### Why This Matters

Even though the ratio is < 1 (making naive Step A fail), a SHARP bound
on the ratio could be useful in a modified induction. Specifically, if:

    Phi_{n-1}(p'/n) <= C(n) * Phi_n(p)

then

    1/Phi_{n-1}(p'/n) >= (1/C(n)) * 1/Phi_n(p)

Since C(n) < 1, this gives 1/Phi_{n-1}(p'/n) > 1/Phi_n(p), which is
the WRONG direction for Step A. However, if a CORRECTION TERM can be
identified — some T_n(p) >= 0 such that:

    1/Phi_n(p) = (something involving 1/Phi_{n-1}(p'/n)) - T_n(p)

then the induction might proceed with the correction absorbed into
the surplus.

### Connection to Electrostatics

The force field S_i(p) = sum_{j!=i} 1/(lambda_i - lambda_j) at the roots
of p is related to the force field at the critical points of p by the
classical electrostatic picture (Gauss, Marden):

- The critical points mu_k are where the total electrostatic force
  from charges at lambda_1,...,lambda_n vanishes.
- The force field at mu_k from the OTHER critical points (i.e.,
  S_k(p'/n)) is NOT the same as the projected force field from the
  original roots.

The precise relationship between S_i(p) and S_k(p'/n) involves the
second-order structure of the log-potential and has been studied by
Steinerberger (2022, Annals of PDE) in the continuum limit via the PDE:

    partial_t u + u * partial_x u = H[u]

where H is the Hilbert transform. This is a nonlocal Burgers-type
equation governing the flow of root density under differentiation.
The energy dissipation in this PDE corresponds to the fact that
Phi_{n-1}(p'/n) < Phi_n(p).

Kiselev and Tan (2022, SIAM J. Math. Anal.) proved global regularity
for this PDE, establishing that the root density converges to the
uniform distribution exponentially fast. The energy dissipation rate
is controlled by the fractional Laplacian (-Delta)^{1/2}.

### Assessment

A sharp bound C(n) on Phi_{n-1}/Phi_n, together with the exact
structure of the correction term, could enable a modified induction.
This is a concrete, computable question.

**Recommended next step:** Symbolic computation of C(n) for n=3,4,5
and the correction term T_n(p). **Priority: HIGH.**

---

## 4. Energy Functionals Under Differentiation in Classical Polynomial Theory

### Gauss-Lucas Electrostatic Framework

The electrostatic interpretation of polynomial roots dates to Gauss
and was systematized by Lucas and later by Marden ("Geometry of
Polynomials," AMS Survey 1966). The key elements:

1. **Logarithmic potential.** For p(z) = prod_i (z - lambda_i),
   the log-potential is ln|p(z)| = sum_i ln|z - lambda_i|.

2. **Electrostatic field.** The field E(z) = grad(ln|p(z)|) =
   p'(z)/p(z) = sum_i 1/(z - lambda_i). Critical points of p
   (roots of p') are equilibrium points of E.

3. **Gauss-Lucas theorem.** The critical points lie in the convex
   hull of the roots, because the electric field from positive
   charges cannot vanish outside the convex hull of the charges
   (Baez 2021, AMS Notices 68(11)).

### How Energy Changes Under Differentiation

The total electrostatic self-energy of the root configuration is:

    W_n(p) = sum_{i<j} log|lambda_i - lambda_j|  (log-energy = E_n above)

This is half the log-discriminant: W_n = (1/2) log disc(p).

The Coulomb energy (with the physics sign convention for repulsive
charges) is:

    U_n(p) = -sum_{i<j} log|lambda_i - lambda_j| = -W_n(p)

Under differentiation, the roots contract (by interlacing), so the
pairwise distances decrease. This means:

    W_{n-1}(p'/n) = sum_{i<j} log|mu_i - mu_j| < W_n(p) = sum_{i<j} log|lambda_i - lambda_j|

where the inequality is NOT entry-by-entry (different number of terms)
but holds for the total sums in typical cases. The Coulomb energy
INCREASES under differentiation.

**Quantitative relationship (Szego, Marden):** There is a classical
identity relating disc(p) and disc(p'):

    disc(p) = (-1)^{n(n-1)/2} * n^n * prod_{k=1}^{n-1} p(mu_k) / disc(p'/n)^?

The precise formula involves the resultant Res(p, p') and is:

    disc(p) = (-1)^{n(n-1)/2} * (1/a_n) * Res(p, p')

Since Res(p, p') = a_n^{n-1} * prod_k p'(lambda_k)/... the relationship
is algebraic but not a simple inequality.

### Steinerberger's PDE Framework

Steinerberger (2018-2022) derived a PDE governing the evolution of
root density under repeated differentiation:

    partial_t rho + partial_x(rho * H[rho]) = 0

where H[rho](x) = PV integral rho(y)/(x-y) dy is the Hilbert transform
of the density rho. This is a nonlocal continuity equation.

**Energy dissipation:** The PDE has a natural entropy/energy structure.
The functional

    S[rho] = -int int log|x-y| rho(x) rho(y) dx dy

(negative of free entropy) is a Lyapunov functional for the flow. As
differentiation proceeds (t increases), S[rho] increases (entropy
decreases, or equivalently, Coulomb energy increases). The flow
converges exponentially to the uniform (arcsine) distribution, which
has the minimum entropy.

**Connection to free probability:** The same PDE was independently
derived by Shlyakhtenko and Tao (2020, arXiv:2009.01882) as the
evolution equation for free fractional convolution powers mu^{boxplus k}.
The parameter k (continuous) corresponds to the degree of the polynomial
divided by the original degree. Crucially:

- Free entropy chi is MONOTONE NON-DECREASING under the convolution
  power flow (Shlyakhtenko-Tao Theorem 1.1).
- Free Fisher information Phi* is MONOTONE NON-INCREASING under the
  convolution power flow (Shlyakhtenko-Tao Theorem 1.2).

These are INFINITE-DIMENSIONAL results. The finite-n analogs would say:

- E_n(p) >= E_{n-1}(p'/n) + correction  (log-energy version)
- Phi_n(p) >= Phi_{n-1}(p'/n)  (Fisher information version — NOTE:
  this is the direction we OBSERVE numerically!)

The second statement is exactly what we know empirically:
Phi_{n-1}(p'/n) < Phi_n(p) always. So the "Fisher information decreases
under differentiation" is the finite-n analog of Shlyakhtenko-Tao's
infinite-dimensional monotonicity.

### Implications

The energy/dissipation framework confirms WHY the direction is wrong
for naive induction: differentiation DISSIPATES Fisher information
(contracts root gaps, reduces force magnitudes). To use differentiation
in an induction, we need to EXPLOIT this dissipation rather than fight it.

One approach: instead of chaining 1/Phi through differentiation, chain
the SURPLUS:

    Delta_n(p,q) = 1/Phi_n(p ⊞_n q) - 1/Phi_n(p) - 1/Phi_n(q)

and show that Delta_n >= Delta_{n-1} + (positive correction from the
dissipated energy). See Section 5 below.

---

## 5. Telescoping Instead of Single-Step Induction

### The Telescoping Idea

Rather than relating Phi_n directly to Phi_{n-1} for each factor
separately (which fails), write:

    Delta_n(p,q) = Delta_{n-1}(p'/n, q'/n) + R_n(p,q)

where R_n is a "remainder" or "correction" term. If:

(a) Delta_{n-1}(p'/n, q'/n) >= 0 by the induction hypothesis
    (using the EXACT commutativity: the degree-(n-1) factors are
    (p'/n) ⊞_{n-1} (q'/n) = (p ⊞_n q)'/n), and

(b) R_n(p,q) >= 0

then Delta_n(p,q) >= 0, completing the induction.

### What R_n Would Look Like

    R_n(p,q) = [1/Phi_n(p ⊞_n q) - 1/Phi_n(p) - 1/Phi_n(q)]
             - [1/Phi_{n-1}((p ⊞_n q)'/n) - 1/Phi_{n-1}(p'/n) - 1/Phi_{n-1}(q'/n)]

Using the exact commutativity, this simplifies to:

    R_n(p,q) = [1/Phi_n(r) - 1/Phi_n(p) - 1/Phi_n(q)]
             - [1/Phi_{n-1}(r'/n) - 1/Phi_{n-1}(p'/n) - 1/Phi_{n-1}(q'/n)]

where r = p ⊞_n q.

This is a difference of surpluses at adjacent degrees. We can rewrite:

    R_n(p,q) = [1/Phi_n(r) - 1/Phi_{n-1}(r'/n)]
             - [1/Phi_n(p) - 1/Phi_{n-1}(p'/n)]
             - [1/Phi_n(q) - 1/Phi_{n-1}(q'/n)]

Define g_n(p) = 1/Phi_n(p) - 1/Phi_{n-1}(p'/n). We know from the data
that g_n(p) < 0 always (since 1/Phi_n < 1/Phi_{n-1}(p'/n)). Then:

    R_n(p,q) = g_n(r) - g_n(p) - g_n(q)

So R_n >= 0 iff g_n is SUPERADDITIVE under ⊞_n:

    g_n(p ⊞_n q) >= g_n(p) + g_n(q)

But g_n(p) < 0 for all p, and superadditivity of a negative function
means |g_n(r)| <= |g_n(p)| + |g_n(q)| (triangle-inequality-like). This
is the statement that the "differentiation penalty" for the convolution
is smaller than the sum of individual penalties.

### Is g_n Superadditive?

This is a NEW conjecture that has NOT been tested numerically. It is
a strictly weaker statement than the original Stam inequality (since
the original inequality plus the fact that Delta_{n-1} >= 0 would
imply R_n >= 0, not the other way around in general).

However, g_n might have structural properties that make it more
amenable to proof. In particular, g_n captures the "energy dissipation
gap" between degree n and degree n-1, and the convolution ⊞_n tends
to regularize root configurations (make them more equispaced), which
might make the dissipation gap SMALLER for the convolution.

### Multi-Level Telescoping

One can iterate: if the single-level telescoping works, then:

    Delta_n = R_n + R_{n-1} + ... + R_4 + Delta_3

Since Delta_3 >= 0 (proved), it suffices to show R_k >= 0 for
k = 4, ..., n. Each R_k involves only degree-k and degree-(k-1)
quantities, potentially making it more local and tractable.

### Assessment

The telescoping approach converts the original problem into n-3
separate "one-level" problems. Each one-level problem (R_k >= 0)
is potentially easier than the full problem because it involves
only the difference between adjacent degrees.

**The key numerical test:** Is g_n(p ⊞_n q) >= g_n(p) + g_n(q)?
This has NOT been tested. **Priority: HIGHEST for this section.**

If the numerical test PASSES, this opens a genuinely new proof route.
If it FAILS, the telescoping approach in this form is dead.

---

## 6. Marcus's Actual Use of Differentiation

### What Marcus (2021, arXiv:2108.07054) Does

Based on the full paper and the PMC version of Marcus-Spielman-Srivastava
(2015/2022, Prob. Theory Related Fields), here is how differentiation
is used:

**6.1 Degree Reduction Lemmas**

Marcus-Spielman-Srivastava prove three "degree reduction" lemmas
(Lemma 1.16 for symmetric additive, Lemma 4.9 for multiplicative,
Lemma 4.16 for asymmetric additive). These allow computing the
convolution of polynomials of UNEQUAL degree by differentiating the
higher-degree polynomial down to the lower degree.

For the symmetric additive convolution (our ⊞_n), the degree reduction
uses the differential operator representation: if p-hat(D) * x^d = p(x),
then p-hat(D) * x^{d-1} = (D/d) * p-hat(D) * x^d = (Dp/d)(x) = p'(x)/d.

This is the algebraic origin of our exact identity:

    (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)

**6.2 Root Bound Proofs by Induction**

Marcus (2021) uses differentiation-based degree reduction as a proof
technique for ROOT BOUNDS (controlling the locations of the largest
and smallest roots), NOT for energy or entropy bounds. The typical
argument structure is:

1. For polynomials of the same degree d, prove a bound on the roots
   of p ⊞_d q in terms of roots of p and q.
2. For polynomials of different degrees, reduce to the same-degree
   case by "differentiating sufficiently many times."

This is purely a ROOT LOCATION result, not an energy inequality.

**6.3 Section 5.2: Majorization**

Section 5.2 of Marcus (2021) proves MAJORIZATION relations on
convolutions. Specifically, finite freeness is used to prove that
the root vector of p ⊞_n q is majorized by (or majorizes) certain
combinations of root vectors of p and q. Majorization is a partial
ordering on R^n related to convex functions.

This is closer to what we need than root bounds, because majorization
involves inequalities on symmetric functions of roots. However, the
specific majorization results in Section 5.2 are about SUMS of roots
(i.e., coefficients / power sums), not about the INVERSE of the sum
of squared forces (1/Phi_n). The function 1/Phi_n is not a Schur-convex
or Schur-concave function, so majorization alone does not directly
yield our inequality.

**6.4 Finite Free Cumulants (Arizmendi-Garza-Vargas-Perales)**

Arizmendi, Garza-Vargas, and Perales (2018, 2021, 2024) developed the
combinatorial theory of finite free cumulants kappa_k^{(n)}. These
cumulants linearize the finite free additive convolution:

    kappa_k^{(n)}(p ⊞_n q) = kappa_k^{(n)}(p) + kappa_k^{(n)}(q)

The most relevant recent result is from the S-transform paper
(arXiv:2408.09337, 2024), which finds "how the finite free cumulants
of a polynomial behave under differentiation" and provides "a simplified
explanation of why free fractional convolution corresponds to the
differentiation of polynomials."

This is the ALGEBRAIC formalization of the differentiation-convolution
commutativity. However, like Marcus's work, it addresses the
COEFFICIENT/CUMULANT structure, not energy functionals.

**6.5 What Marcus Does NOT Do**

- No energy inequalities (no Phi_n or equivalent)
- No entropy functionals
- No Stam-type inequality
- No superadditivity results for inverse-energy quantities
- The differentiation is used for structural/combinatorial purposes,
  not for quantitative energy bounds

### Key Takeaway

Marcus uses differentiation as a tool for ROOT BOUNDS and STRUCTURAL
RESULTS (majorization, cumulant identities), not for ENERGY INEQUALITIES.
The exact identity (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n) is implicit
in his degree reduction lemmas, but the connection to Phi_n is entirely
new to our work.

---

## 7. Connections to the Shlyakhtenko-Tao Framework

### The Bridge Between Finite and Infinite

Shlyakhtenko and Tao (2020, arXiv:2009.01882) proved monotonicity of
free entropy and free Fisher information under free convolution powers.
Their results are in the CONTINUOUS (infinite-dimensional) setting, but
the methodology provides crucial guidance for the finite case:

**Result 1 (Free entropy monotonicity):** For mu a probability measure
with finite second moment, chi(k^{-1/2}_* mu^{boxplus k}) is monotone
non-decreasing in k >= 1.

**Result 2 (Free Fisher information monotonicity):** Under the same
conditions, Phi*(k^{-1/2}_* mu^{boxplus k}) is monotone non-increasing
in k >= 1.

**Proof technique 1 (free probability):** Uses the free score J(X),
conditional expectations in the von Neumann algebra framework, and
L^2-contraction.

**Proof technique 2 (self-contained):** Differentiates the quantities
in k and shows positivity of the derivative via a POSITIVE SEMI-DEFINITE
KERNEL:

    K(z,w) = [1/(G(z)G(w))] * [(G(z)-G(w))/(z-w) + G(z)G(w)]^2

where G is the Cauchy transform. The positivity of K is the core
analytic fact driving monotonicity.

### Finite-n Analog of the Kernel

The Cauchy transform of the empirical measure of p with roots
lambda_1,...,lambda_n is:

    G_p(z) = (1/n) * sum_i 1/(z - lambda_i) = (1/n) * p'(z)/p(z)

This is a FINITE-n object. The kernel K(z,w) can be evaluated for
finite-n Cauchy transforms. The question is whether K_n(z,w) (the
finite-n kernel) is positive semi-definite.

If K_n is positive semi-definite, then the Shlyakhtenko-Tao proof
technique 2 would directly yield the finite-n Fisher information
monotonicity under "finite free convolution powers." Combined with
the exact differentiation-convolution identity, this would give a
proof of the finite Stam inequality.

### The Heat Flow Conjecture Connection

Hall-Ho-Jalowy-Kabluchko (2022, arXiv:2202.09660) studied polynomials
evolving under the heat flow exp(tau/(2N) * d^2/dz^2) and conjectured
that the log-potential evolves according to:

    partial S / partial tau = (1/2) (partial S / partial z)^2

They also established a connection: the backward heat flow on
real-rooted polynomials corresponds to free additive convolution with
the semicircular distribution. This suggests that the heat flow
provides a DIFFERENT axis (parameterized by "semicircular noise")
along which energy monotonicity might hold, complementing the
differentiation axis (parameterized by degree reduction).

---

## 8. Viability Assessment and Recommendations

### Overall Assessment of Strategy B

| Sub-route | Probability | Key blocker |
|-----------|-------------|-------------|
| Naive induction (1/Phi_n) | 0% | Step A fails 100% |
| Alternative functional | 15% | Need to identify F_n satisfying both (i) and (ii) |
| Reversed induction (n-1 -> n) | 5% | No canonical lifting preserving real-rootedness |
| Sharp Phi ratio + correction | 20% | Need R_n >= 0 (untested) |
| Telescoping via g_n superadditivity | 25% | Key numerical test not yet run |
| Direct kernel positivity (a la S-T) | 15% | Need finite-n PSD of K_n |

**Combined probability (Strategy B alone): ~15%**
(some sub-routes are correlated; taking the best independent shot)

### What Should Be Done Next (Priority Order)

1. **[HIGHEST] Test g_n superadditivity numerically.** Compute
   g_n(p) = 1/Phi_n(p) - 1/Phi_{n-1}(p'/n) for random real-rooted
   polynomials, then check if g_n(p ⊞_n q) >= g_n(p) + g_n(q).
   This is the gate for the telescoping approach. If it passes,
   Strategy B becomes viable at ~25-30%.

2. **[HIGH] Test disc(p)^alpha / Phi_n as alternative functional.**
   For alpha in {1/2, 1, 3/2, 2, (n-1)/n, n(n-1)/2}, compute the
   functional for random polynomials, check both superadditivity
   under ⊞_n and the direction of change under differentiation.

3. **[HIGH] Compute the finite-n kernel K_n(z,w) and test PSD.**
   Using the Cauchy transform G_p(z) = (1/n)*p'(z)/p(z), evaluate
   K_n at the Shlyakhtenko-Tao formula and check positive
   semi-definiteness numerically.

4. **[MEDIUM] Symbolic computation of C(n) for n=3,4,5.** Find the
   sharp bound sup_p Phi_{n-1}(p'/n)/Phi_n(p) and characterize the
   extremal polynomials.

5. **[MEDIUM] Normalized log-energy direction test.** Check whether
   E_n(p)/binom(n,2) increases or decreases under differentiation.

### How Strategy B Interacts with Other Strategies

Strategy B is most naturally a COMPONENT of a larger proof rather
than a standalone approach:

- **B + D (conditional theorem):** If the telescoping R_n >= 0 holds,
  it can be stated as a lemma in the conditional framework.

- **B + A (conjugate variable projection):** The differentiation
  identity provides an algebraic decomposition that might combine
  with the projection mechanism. Specifically, the Shlyakhtenko-Tao
  kernel K_n is a projection-theoretic object, and its finite-n
  version could feed directly into the Strategy A framework.

- **B + C (algebraic SOS):** If the sharp Phi ratio C(n) has a clean
  algebraic form, it might provide the key intermediate inequality
  needed for an SOS decomposition at specific n.

---

## References

### Primary Sources

- Marcus, A.W. (2021). "Polynomial convolutions and (finite) free
  probability." arXiv:2108.07054.
  https://arxiv.org/abs/2108.07054

- Marcus, A.W., Spielman, D.A., Srivastava, N. (2015/2022). "Finite
  free convolutions of polynomials." Probab. Theory Related Fields 182.
  arXiv:1504.00350. https://pmc.ncbi.nlm.nih.gov/articles/PMC9013345/

- Shlyakhtenko, D. and Tao, T. (2020). "Fractional free convolution
  powers." arXiv:2009.01882. https://arxiv.org/abs/2009.01882

- Voiculescu, D. (1993). "The analogues of entropy and of Fisher's
  information measure in free probability theory, I." Comm. Math. Phys.
  155, 71-92. https://link.springer.com/article/10.1007/BF02100050

- Voiculescu, D. (1998). "The analogues of entropy and of Fisher's
  information measure in free probability theory, V: Noncommutative
  Hilbert transforms." Invent. Math. 132.

### Differentiation and Polynomial Roots

- Steinerberger, S. (2022). "The flow of polynomial roots under
  differentiation." Annals of PDE 8, 16.
  https://link.springer.com/article/10.1007/s40818-022-00135-4

- Kiselev, A. and Tan, C. (2022). "Global regularity for a nonlocal
  PDE describing evolution of polynomial roots under differentiation."
  SIAM J. Math. Anal. https://epubs.siam.org/doi/10.1137/21M1422859

- Hall, B.C., Ho, C.-W., Jalowy, J., Kabluchko, Z. (2022). "The heat
  flow conjecture for polynomials and random matrices."
  arXiv:2202.09660. https://arxiv.org/abs/2202.09660

### Finite Free Cumulants

- Arizmendi, O. and Perales, D. (2018). "Cumulants for finite free
  convolution." J. Combin. Theory Ser. A 155, 244-266.
  arXiv:1611.06598. https://arxiv.org/abs/1611.06598

- Arizmendi, O., Garza-Vargas, J., Perales, D. (2021). "Finite free
  cumulants: Multiplicative convolutions, genus expansion and
  infinitesimal distributions." Trans. Amer. Math. Soc. 376 (2023).
  arXiv:2108.08489.

- Arizmendi, O., Garza-Vargas, J., Perales, D. (2024). "S-transform
  in finite free probability." arXiv:2408.09337.
  https://arxiv.org/abs/2408.09337

### Classical Polynomial Theory

- Marden, M. (1966). "Geometry of Polynomials." AMS Mathematical
  Surveys, vol. 3.

- Baez, J.C. (2021). "Electrostatics and the Gauss-Lucas Theorem."
  AMS Notices 68(11), 1988-1991.
  https://www.ams.org/notices/202111/rnoti-p1988.pdf

### Classical Information Theory (Analogs)

- Carlen, E.A. (1991). "Superadditivity of Fisher's information and
  logarithmic Sobolev inequalities." J. Funct. Anal. 101, 194-211.
  https://www.sciencedirect.com/science/article/pii/002212369190155X

- Stam, A.J. (1959). "Some inequalities satisfied by the quantities
  of information of Fisher and Shannon." Information and Control 2,
  101-112.

- Bobkov, S.G. and Chistyakov, G.P. (2015). "Entropy power inequality
  for the Renyi entropy." IEEE Trans. Inform. Theory 61, 708-714.
