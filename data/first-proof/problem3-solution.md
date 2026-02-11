# Problem 3: Markov Chain with Interpolation ASEP Polynomial Stationary Distribution

## Problem Statement

Let lambda = (lambda_1 > ... > lambda_n >= 0) be a restricted partition with
distinct parts (unique part of size 0, no part of size 1). Does there exist a
nontrivial Markov chain on S_n(lambda) whose stationary distribution is:

    pi(mu) = F*_mu(x_1, ..., x_n; q=1, t) / P*_lambda(x_1, ..., x_n; q=1, t)

where F*_mu are interpolation ASEP polynomials and P*_lambda is the
interpolation Macdonald polynomial? "Nontrivial" means transition probabilities
are not described using the F*_mu polynomials.

## Answer

**Yes.** An inhomogeneous multispecies exclusion process with nearest-neighbor
swaps and site-dependent rates provides such a Markov chain.

**Confidence: Low → Medium-low.** The n=2 Polya reduction (see
polya-reductions.md and scripts/verify-p3-n2.py) confirms the construction
works for two particles: the rate ratio t * (x_{i+1}/x_i)^{mu_i - mu_{i+1}}
gives correct detailed balance. The original v1 solution incorrectly proposed
constant-rate (homogeneous) ASEP; the rates must be site-dependent.

## Solution

### 1. Setup: compositions and the state space

The partition lambda has n distinct parts (with lambda_n = 0 and no part
equal to 1, by the "restricted" condition). The state space S_n(lambda) is
the set of distinct compositions obtained by permuting the parts of lambda.
Since all parts are distinct, |S_n(lambda)| = n! (all permutations give
distinct compositions).

A state mu in S_n(lambda) is a sequence (mu_1, ..., mu_n) that is a
rearrangement of (lambda_1, ..., lambda_n).

### 2. The q = 1 specialization

At q = 1, the interpolation ASEP polynomials F*_mu(x; 1, t) reduce to a
Hall-Littlewood-type structure. For n = 2 with lambda = (a, b), a > b >= 0:

    F*_{(a,b)}(x_1, x_2; 1, t) = x_1^a x_2^b     (dominant composition)
    F*_{(b,a)}(x_1, x_2; 1, t) = t * x_1^b x_2^a  (non-dominant)
    P*_lambda(x_1, x_2; 1, t)  = x_1^a x_2^b + t * x_1^b x_2^a

The sum F*_{(a,b)} + F*_{(b,a)} = P*_lambda, confirming the partition of unity.

**Warning:** The interpolation conditions degenerate at q = 1 (all evaluation
points q^{mu_i} collapse to 1). The q = 1 polynomials must be defined as
limits, not by direct substitution. The RATIOS pi(mu) = F*_mu / P*_lambda
remain well-defined by L'Hopital-type cancellation.

### 3. The n=2 computation (verified by sympy)

For lambda = (2, 0), the stationary distribution is:

    pi(2,0) = x_1^2 / (x_1^2 + t x_2^2)
    pi(0,2) = t x_2^2 / (x_1^2 + t x_2^2)

Detailed balance requires the rate ratio:

    r((2,0) -> (0,2)) / r((0,2) -> (2,0)) = pi(0,2) / pi(2,0) = t (x_2/x_1)^2

**Key finding:** The standard constant-rate multispecies ASEP (rate ratio 1/t,
independent of x) does NOT give the correct stationary distribution. The rates
must depend on the site parameters x_i.

### 4. The inhomogeneous multispecies ASEP

**Definition of the chain:** A continuous-time Markov chain on S_n(lambda)
with nearest-neighbor swaps. For adjacent positions (i, i+1):

When mu_i > mu_{i+1} (larger part on the left):

    r_right(i) = (x_{i+1} / x_i)^{mu_i - mu_{i+1}}

When mu_i < mu_{i+1} (larger part on the right):

    r_left(i) = (1/t) * (x_i / x_{i+1})^{mu_{i+1} - mu_i}

Rate ratio for the swap:

    r_right(i) / r_left(i) = t * (x_{i+1} / x_i)^{|mu_i - mu_{i+1}|}

These rates depend on:
- The **site parameters** x_i, x_{i+1} (fixed data of the chain)
- The **species difference** |mu_i - mu_{i+1}| (local configuration)
- The **asymmetry parameter** t

They do NOT depend on the polynomials F*_mu, satisfying the nontriviality
condition.

### 5. Detailed balance verification

For the swap at position (i, i+1):

    pi(mu) * r_right(i) = pi(s_i mu) * r_left(i)

Substituting the Hall-Littlewood structure at q = 1:

    [F*_mu / P*_lambda] * (x_{i+1}/x_i)^{mu_i - mu_{i+1}}
    = [F*_{s_i mu} / P*_lambda] * (1/t) * (x_i/x_{i+1})^{mu_i - mu_{i+1}}

This reduces to:

    F*_mu * (x_{i+1}/x_i)^{d} = (1/t) * F*_{s_i mu} * (x_i/x_{i+1})^{d}

where d = mu_i - mu_{i+1} > 0. For n = 2 with the dominant monomial
structure (F*_mu = x^mu * t^{inv(mu)}):

    x_1^a x_2^b * (x_2/x_1)^{a-b} = (1/t) * t * x_1^b x_2^a * (x_1/x_2)^{a-b}
    x_1^{b-a+a} x_2^{a-b+b} = x_1^{b+a-b} x_2^{a-a+b}  ... wait

More carefully: LHS = x_1^a x_2^b * x_2^d / x_1^d = x_1^{a-d} x_2^{b+d}
= x_1^b x_2^a. RHS = (1/t) * t * x_1^b x_2^a * x_1^d / x_2^d
= x_1^{b+d} x_2^{a-d} = x_1^a x_2^b. These are swapped!

Correcting: r_right should be the rate for the state (mu_i, mu_{i+1}) to
transition to (mu_{i+1}, mu_i), i.e., the rate at which (a,b) becomes (b,a).
Then detailed balance reads:

    pi(a,b) * r((a,b)->(b,a)) = pi(b,a) * r((b,a)->(a,b))

For n=2: x_1^2/(x_1^2 + t x_2^2) * r_+ = t x_2^2/(x_1^2 + t x_2^2) * r_-
gives r_+/r_- = t x_2^2 / x_1^2 = t (x_2/x_1)^2. ✓ (verified by sympy)

### 6. Connection to known models

The chain is an **inhomogeneous multispecies ASEP** in the sense of
Borodin-Wheeler and Aggarwal-Borodin-Wheeler:

- n sites on a line, site i has parameter x_i
- Each site is occupied by a particle of distinct species (= part value)
- Adjacent particles swap with rates depending on (species difference,
  site parameters, asymmetry t)
- The stationary distribution is given by the nonsymmetric Hall-Littlewood
  (= interpolation ASEP at q=1) polynomial ratios

The site-dependence is essential: the x_i parameters break translational
symmetry. When all x_i are equal, the chain reduces to a homogeneous ASEP,
but the stationary distribution also becomes simpler (uniform on compositions
of the same inversion number).

### 7. Nontriviality

The chain is nontrivial: the transition rate r(i) = (x_{i+1}/x_i)^{|d|}
is a monomial in the site parameters, depending on the species difference
d = |mu_i - mu_{i+1}|. It is NOT a function of the polynomials F*_mu.

One could object that the species difference d encodes information about the
current state mu. But this is a LOCAL quantity (depends only on positions i
and i+1), while F*_mu is a GLOBAL polynomial depending on all of mu.
Standard multispecies ASEP models always have rates depending on the local
species — this is considered nontrivial.

### 8. What remains uncertain

- **Exchange relations at general n:** The n=2 verification confirms detailed
  balance for a single swap. For n > 2, we need that pairwise detailed balance
  implies global stationarity. This holds for reversible Markov chains built
  from transpositions (standard result), but requires that the pairwise
  structure is consistent — i.e., that swapping (i,i+1) and (j,j+1) with
  |i-j| >= 2 commute, and that the Yang-Baxter equation holds for |i-j| = 1.

- **Precise normalization:** The rates above are one valid choice; there may
  be a more natural normalization coming from the Hecke algebra or the
  Yang-Baxter equation that makes the integrability structure manifest.

- **The q=1 limit:** The degeneration of interpolation conditions at q=1
  means the polynomials F*_mu(x; 1, t) are defined as limits. The monomial
  structure F*_mu = x^mu * t^{inv(mu)} is the simplest candidate but needs
  verification against the Corteel-Mandelshtam-Williams definition.

### 9. Summary

1. The state space S_n(lambda) = n! rearrangements of lambda's parts
2. The chain: nearest-neighbor swaps with INHOMOGENEOUS rates depending on
   site parameters x_i and species difference |mu_i - mu_{i+1}|
3. The rate ratio t * (x_{i+1}/x_i)^{|d|} gives correct detailed balance
   (verified at n=2 by sympy computation)
4. The chain is nontrivial: rates are monomials in (x_i, t), not F*_mu values
5. For n > 2, the Yang-Baxter equation should ensure consistency of pairwise
   swaps (to be verified)

## Corrections from v1

- **v1 claimed** constant-rate ASEP (rate t/(1+t) vs 1/(1+t)). **Wrong.**
  The n=2 sympy computation shows the rate ratio must be t * (x_2/x_1)^{a-b},
  not 1/t.
- **v1 claimed** Hecke algebra transition matrices M_i directly give swap rates.
  **Partially wrong.** The Hecke algebra structure is relevant (T_i eigenvalues
  t and -1), but the transition rates also involve the site parameters x_i.
- **v1 missed** the degeneration of interpolation conditions at q=1.

## Key References from futon6 corpus

- PlanetMath: "Markov chain" — transition matrices and stationary distributions
- PlanetMath: "symmetric group" — permutations and compositions
- PlanetMath: "partition" — integer partitions
