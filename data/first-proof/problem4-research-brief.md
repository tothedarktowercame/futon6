# Library Research Brief: Problem 4 — Root Separation Under Finite Free Convolution

**Purpose:** Targeted search over mathoverflow and math.stackexchange corpus
to find a correct proof strategy for the superadditivity inequality. We know
the result is TRUE (numerically verified) but the original proof is BROKEN.

**Priority:** HIGH — this is the only problem where we have a confirmed
mathematical error in the proof, not just a confidence gap.

---

## What we claimed

The inequality

    1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)

holds for monic real-rooted polynomials p, q of degree n, where
Phi_n(p) = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2 is the Coulomb
self-energy of the roots, and ⊞_n is the Marcus-Spielman-Srivastava (2015)
finite free additive convolution.

## What we know for certain

1. **The inequality IS true** — 0 violations in 8000 random trials (n=2,3,4,5)
2. **Equality at n=2** — 1/Phi_2 = (a-b)^2/2, linear in the sole cumulant
3. **Strict inequality for n >= 3** — min ratio 1.0001 at n=3, growing with n
4. **Specific to ⊞_n** — fails ~40% under plain coefficient addition

## What is WRONG in our proof

**Error 1: Direction.** We claimed "concavity + f(0)=0 implies
superadditivity." This is mathematically backwards. Concavity + f(0)=0
gives SUBadditivity: f(a) + f(b) >= f(a+b). For superadditivity one
needs CONVEXITY + f(0)=0.

**Error 2: False concavity claim.** We claimed "1/Phi_n is concave in the
cumulants." Numerical midpoint tests show 1/Phi_n is NEITHER globally
convex NOR concave in coefficient space (50/50 violations at n=3, trending
toward more convexity violations at higher n).

**Error 3: Coefficient-cumulant conflation.** The polynomial coefficients
a_k are NOT finite free cumulants. The ⊞_n formula has cross-terms
(e.g., c_2 = a_2 + (2/3)a_1*b_1 + b_2 at n=3). The actual finite free
cumulants require Möbius inversion on non-crossing partitions
(Arizmendi-Perales 2018).

## What we need from the corpus

A correct proof strategy. The three most promising directions:

### Direction A: Convexity in finite free cumulant space

If 1/Phi_n happens to be convex (not concave!) as a function of the
Arizmendi-Perales finite free cumulants kappa_k^(n), then superadditivity
follows from f convex + f(0)=0. The coefficient-space test doesn't rule
this out because the coefficient→cumulant map is nonlinear.

### Direction B: Random matrix / Haar integration argument

Since p ⊞_n q = E_U[char(A + UBU*)], maybe a direct argument through
properties of Haar-random unitary conjugation works. Jensen's inequality
applied to some functional of the eigenvalues?

### Direction C: Algebraic argument via the MSS bilinear formula

The coefficient formula c_k = sum_{i+j=k} w(n,i,j) a_i b_j has specific
falling-factorial weights. Maybe the superadditivity follows from an
algebraic identity involving these weights.

---

## Research Questions (search these)

**Q4.1: Has anyone studied Phi_n or 1/Phi_n in the context of free convolution?**

Search terms:
- "root separation" "free convolution"
- "Coulomb energy" "free convolution" OR "free probability"
- Phi_n "characteristic polynomial" roots separation
- "electrostatic" "free convolution"
- "log-gas" "finite free"

What we're looking for: Any functional of roots that has been shown to be
superadditive, subadditive, convex, or concave under ⊞_n. Even if it's not
exactly 1/Phi_n, a related result would indicate the proof technique.

**Q4.2: Properties of finite free cumulants (Arizmendi-Perales)**

Search terms:
- "finite free cumulants" Arizmendi OR Perales
- "finite R-transform"
- "finite free" cumulant moment
- "finite free convolution" cumulant linearization
- arXiv:1611.06598

What we're looking for: Explicit formulas for finite free cumulants in terms
of coefficients (moment-cumulant relation). Also: any convexity/monotonicity
results for functions expressed in finite free cumulant coordinates.

**Q4.3: Superadditivity results under free convolution**

Search terms:
- "superadditive" "free convolution"
- "free entropy" superadditivity
- "free Fisher information" convexity
- Voiculescu "free entropy" "convex"
- "reciprocal" energy "free convolution"

In classical free probability, the free entropy chi and free Fisher
information Phi* have known convexity properties. If 1/Phi_n is analogous
to free entropy (which IS known to be superadditive under free convolution),
this would give both the proof and the conceptual explanation.

**Q4.4: Root gap / minimum spacing under ⊞_n**

Search terms:
- "root gap" "free convolution" OR "finite free"
- "minimum spacing" eigenvalues "free convolution"
- "interlacing" "root separation"
- Marcus Spielman Srivastava "root" gap OR spread OR spacing
- "barrier method" roots "free convolution"

MSS's original work used a barrier method for interlacing. Root gap
behavior under ⊞_n is closely related to 1/Phi_n (by Cauchy-Schwarz,
1/Phi_n <= sum_{i<j}(lambda_i - lambda_j)^2 / n(n-1)^2).

**Q4.5: Is this a known conjecture?**

Search terms:
- Spielman "root separation" conjecture
- "First Proof" problem 4 OR Spielman
- site:arxiv.org 2602.05192
- "Spielman" "finite free convolution" open problem

The problem is attributed to Spielman, who is one of the creators of
finite free convolution. He may have posed this on MO or discussed it
in a talk/paper. Finding the original context would be extremely valuable.

**Q4.6: Haar integration and eigenvalue functionals**

Search terms:
- "Haar measure" "expected characteristic polynomial" eigenvalue
- "random unitary" conjugation "eigenvalue" functional
- E_U "char" OR "characteristic polynomial" A + UBU*
- "Weingarten" characteristic polynomial

The random matrix model p ⊞_n q = E_U[char(A + UBU*)] might allow
direct computation via Weingarten calculus or HCIZ-type integrals.

---

## Key References

- Marcus, Spielman, Srivastava (2015), "Interlacing Families II," Annals of
  Math 182(1), 327-350. arXiv:1504.00350.
  [Defines ⊞_n, proves real-rootedness preservation, random matrix model]

- Arizmendi, Perales (2018), "Cumulants for finite free convolution,"
  JCTA 155, 244-266. arXiv:1611.06598.
  [Finite free cumulants that linearize under ⊞_n — the right coordinates]

- Marcus (2021), "Polynomial convolutions and (finite) free probability,"
  arXiv:2108.07054.
  [Survey including Section 5.2 on majorization. May contain energy-type
  results or open problems. HIGH PRIORITY to read in detail.]

- Marcus, Spielman, Srivastava (2018), "On the Further Structure of the
  Finite Free Convolutions," arXiv:1811.06382.
  [Generalizes barrier method, conjectures new root bounds. Counterexample
  to multivariate extension. Relevant for proof technique.]

- Voiculescu (1993), "The analogues of entropy and of Fisher's information
  in free probability theory, I," Comm. Math. Phys. 155(1), 71-92.
  [Free entropy chi is superadditive under free convolution. If 1/Phi_n
  is a finite analogue of free entropy, this gives the conceptual frame.]

## Conceptual Lead: Free Entropy Analogy

Voiculescu's free entropy chi(mu) satisfies:

    chi(mu ⊞ nu) >= chi(mu) + chi(nu)

This is EXACTLY the form of our inequality. If 1/Phi_n is a finite
analogue of free entropy (both measure "how spread out" a distribution
is), then the proof strategy may be:

1. Identify 1/Phi_n as a discretization of chi
2. Show the superadditivity proof for chi descends to the finite case
3. The classical proof of chi superadditivity uses free Fisher
   information Phi* and the relation d/dt chi(mu_t) = -1/(2*Phi*(mu_t))
   along the free Brownian motion semigroup

This is speculative but could be the "right" proof. Search for MO posts
connecting finite free convolution to free entropy.

## Search Strategy Notes

1. **MathOverflow strongly preferred**: This is research-level free
   probability / random matrix theory. MO >> math.SE for relevance.

2. **Author search priority**: Posts by or mentioning Spielman, Marcus,
   Srivastava, Arizmendi, Perales, or Voiculescu are highest value.

3. **Date range**: Finite free convolution was introduced in 2015. Most
   relevant posts are 2015-2026, with the Arizmendi-Perales cumulant
   paper from 2017/2018 being a key inflection point.

4. **What counts as a hit**: We need a PROOF TECHNIQUE, not just the
   result. Specifically valuable:
   - Any convexity result for a root functional under ⊞_n
   - Explicit finite free cumulant formulas (the moment-cumulant map)
   - Connections between 1/Phi_n and free entropy/Fisher information
   - Any post by Spielman about this problem or related conjectures
   - The barrier/interlacing method applied to energy-type functionals

5. **Return format**: For each hit, return:
   - Post title, URL/ID, author, date
   - Key claim or technique (1-2 sentences)
   - How it relates to our gap (proof technique / conceptual / citation only)
   - Confidence that it helps close the gap (high / medium / low)
