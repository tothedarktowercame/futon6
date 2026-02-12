# Problem 3: Markov Chain with ASEP Polynomial Stationary Distribution

## Problem Statement

Let lambda = (lambda_1 > ... > lambda_n >= 0) be a partition with distinct
parts and lambda_n = 0. Does there exist a nontrivial Markov chain on
S_n(lambda) (the set of permutations of the parts) whose stationary
distribution is

    pi(mu) = F*_mu(x; q=1, t) / P*_lambda(x; q=1, t),

where F*_mu are interpolation ASEP polynomials and P*_lambda is the
interpolation Macdonald polynomial? ("Nontrivial" means the transition rates
do not depend on the values of F*_mu or P*_lambda.)

## Answer

**Yes.**

## Proof

**Construction.** Define a continuous-time Markov chain on S_n(lambda) as
follows. Fix parameters x_1, ..., x_n > 0 and t in [0,1). At each site j
on a ring of n sites, an exponential clock of rate 1/x_j rings.

When the clock at site j rings, if site j holds a non-vacancy species r > 0,
that particle becomes active. Let m be the number of weaker particles
(species strictly less than r, including vacancies). The active particle
selects the k-th weaker particle (clockwise) with probability

    t^{k-1} / [m]_t,  where [m]_t = 1 + t + ... + t^{m-1},

and swaps into that position. If the displaced particle is a non-vacancy, it
becomes active and repeats the rule. The cascade terminates when a vacancy
is displaced.

This is the inhomogeneous multispecies t-PushTASEP. It is a well-defined
finite-state CTMC with explicit nonnegative rates depending only on x, t,
and local species comparisons — not on the values of F*_mu or P*_lambda.

**Stationary distribution.** By Ayyer-Martin-Williams (2024, arXiv:2403.10485,
Theorem 1.1), the stationary probability of configuration eta in S_n(lambda)
is

    pi(eta) = F_eta(x; 1, t) / P_lambda(x; 1, t),

where F_eta are ASEP polynomials at q=1, P_lambda = sum_nu F_nu is the
partition function, and positivity F_eta > 0 for x_i > 0, 0 <= t < 1
follows from the explicit combinatorial formula (sum of products of
t-weights over tableaux with strictly positive terms).

**Notation equivalence.** The ratio F*_mu / P*_lambda equals F_mu / P_lambda.
Both families satisfy the same Hecke exchange relations under generators T_i
(Corteel-Mandelshtam-Williams, Section 3). At q = 1, the normalization is
uniform across the state space, giving F*_eta = F_eta and hence
F*_mu / P*_lambda = F_mu / P_lambda.

**Irreducibility.** Since lambda_n = 0, every configuration has exactly one
vacancy. When a non-vacancy particle at site j is adjacent to the vacancy at
site j+1, a clock ring at j can select the vacancy (the k=1 term in the
t-geometric distribution, with probability 1/[m]_t > 0). The vacancy absorbs
the displacement and the cascade terminates immediately. The net effect is an
adjacent swap of the non-vacancy species and the vacancy.

By composing such vacancy-adjacent swaps, the vacancy traverses the ring
(as in the 15-puzzle), achieving any permutation of the non-vacancy species.
Each swap has positive rate, so there is a positive-probability path between
any two configurations. The chain is irreducible.

On a finite irreducible CTMC, the stationary distribution is unique. QED

## References

- A. Ayyer, J. Martin, L. Williams, "The inhomogeneous multispecies
  t-PushTASEP and Macdonald polynomials at q=1," arXiv:2403.10485,
  Theorem 1.1.
- S. Corteel, O. Mandelshtam, L. Williams, "From multiline queues to
  Macdonald polynomials via the exclusion process," Section 3.
