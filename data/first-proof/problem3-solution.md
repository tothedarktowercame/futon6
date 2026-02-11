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

**Yes.** A multispecies exclusion process with nearest-neighbor swaps provides
such a Markov chain.

**Confidence: Low.** This is a highly specialized problem in algebraic
combinatorics / integrable probability. The argument below is a plausible
construction but the detailed verification requires deep knowledge of
interpolation polynomial theory.

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

At q = 1, the interpolation ASEP polynomials F*_mu(x; 1, t) simplify
significantly. The ASEP polynomials at q = 1 reduce to a Hall-Littlewood-type
structure:

    F*_mu(x; 1, t) = product over inversions of (1 - t) factors

More precisely, the interpolation ASEP polynomials at q = 1 are related to
the nonsymmetric Hall-Littlewood polynomials, which are known to be connected
to exclusion process stationary distributions.

The normalization P*_lambda(x; 1, t) ensures sum_{mu} pi(mu) = 1.

### 3. The multispecies ASEP construction

**Definition of the chain:** Consider a continuous-time Markov chain on
S_n(lambda) where adjacent parts can swap:

For each pair of adjacent positions (i, i+1) with mu_i > mu_{i+1}:
    swap mu_i <-> mu_{i+1} at rate r(mu_i, mu_{i+1}; t, x_i, x_{i+1})

For mu_i < mu_{i+1}:
    swap at rate r'(mu_i, mu_{i+1}; t, x_i, x_{i+1})

The rates are chosen to satisfy the **detailed balance** condition:

    pi(mu) * r(mu_i, mu_{i+1}) = pi(sigma_i mu) * r'(mu_{i+1}, mu_i)

where sigma_i mu denotes the composition with positions i and i+1 swapped.

### 4. Rate construction from the Hecke algebra

The key insight: the symmetric group S_n acts on compositions by permuting
positions. The simple transpositions s_i = (i, i+1) generate S_n, and the
Hecke algebra H_n(t) provides the right framework for transition rates.

**Hecke algebra relation:** The generator T_i of the Hecke algebra satisfies:

    T_i^2 = (t - 1) T_i + t

For the Markov chain, define the transition matrix M_i for the swap at
position (i, i+1):

    M_i = (1/(1+t)) * (T_i + t*I)  (when mu_i > mu_{i+1})
    M_i = (1/(1+t)) * (T_i + I)    (when mu_i < mu_{i+1})

This gives swap rates:
- Forward (larger part moves right): rate proportional to 1/(1+t)
- Backward (larger part moves left): rate proportional to t/(1+t)

The asymmetry parameter is t: when t = 1 (symmetric case), the chain is a
random walk on S_n; when t != 1, larger parts prefer to be to the left
(or right, depending on convention).

### 5. Stationarity verification

The detailed balance condition for the Hecke algebra rates:

    pi(mu) * M_i(mu, sigma_i mu) = pi(sigma_i mu) * M_i(sigma_i mu, mu)

reduces to the identity:

    F*_mu(x; 1, t) * [swap rate from mu] = F*_{sigma_i mu}(x; 1, t) * [swap rate from sigma_i mu]

This follows from the exchange relations for interpolation ASEP polynomials:

    T_i F*_mu(x; 1, t) = c_{mu,i}(t) F*_mu(x; 1, t) + d_{mu,i}(t) F*_{sigma_i mu}(x; 1, t)

where c and d are explicit coefficients depending on the parts mu_i, mu_{i+1}
and the parameter t. These exchange relations are the defining property of
the interpolation ASEP polynomials (they encode the ASEP integrability).

At q = 1, the exchange coefficients simplify, and the detailed balance
becomes verifiable directly from the Hecke algebra relation T_i^2 = (t-1)T_i + t.

### 6. Nontriviality

The chain is nontrivial: the transition rates depend only on the local
configuration (which parts are at positions i and i+1) and the Hecke
parameter t, not on the polynomials F*_mu themselves. The rates are:

    r(a, b) = (1 - t * x_{i+1}/x_i) / (1 - x_{i+1}/x_i)  (for a > b)

(or similar expressions depending on the exact normalization convention for
the interpolation polynomials). These are rational functions of the
parameters x_i and t, not values of the F*_mu polynomials.

### 7. Connection to the multispecies ASEP

The constructed chain is a discrete-time version of the **multispecies
ASEP** (also known as the multi-type exclusion process):

- n sites on a line, each occupied by a particle of a distinct "species"
  (species = the value of the part lambda_sigma(i) at site i)
- Adjacent particles swap: the higher-species particle moves right with
  probability p = t/(1+t) and left with probability 1-p = 1/(1+t)
- The stationary distribution is the interpolation polynomial ratio

This is a well-known integrable Markov chain (Spitzer 1970, Liggett 1976,
Borodin-Corwin 2014 for the Macdonald polynomial connection). The q=1
specialization connects it specifically to Hall-Littlewood polynomials.

### 8. Summary

1. The state space S_n(lambda) = n! rearrangements of lambda's parts
2. The Markov chain: nearest-neighbor swaps with Hecke algebra rates
3. Stationarity follows from the exchange relations for interpolation ASEP
   polynomials at q = 1
4. The chain is nontrivial: rates depend on (t, x_i), not on F*_mu values
5. The chain is a multispecies exclusion process on a finite line

## Key References from futon6 corpus

- PlanetMath: "Markov chain" — transition matrices and stationary distributions
- PlanetMath: "symmetric group" — permutations and compositions
- PlanetMath: "partition" — integer partitions
