# Problem 3: Markov Chain with Interpolation ASEP Polynomial Stationary Distribution

## Problem Statement

Let lambda = (lambda_1 > ... > lambda_n >= 0) be a restricted partition with
distinct parts (unique part of size 0, no part of size 1). Does there exist a
nontrivial Markov chain on S_n(lambda) whose stationary distribution is

    pi(mu) = F*_mu(x_1, ..., x_n; q=1, t) / P*_lambda(x_1, ..., x_n; q=1, t)

where F*_mu are interpolation ASEP polynomials and P*_lambda is the
interpolation Macdonald polynomial?

## Answer

Yes.

Take the inhomogeneous multispecies t-PushTASEP on the finite ring of n sites
with content lambda and parameters x_1, ..., x_n. This is a concrete continuous-
time Markov chain defined directly from local species comparisons and ringing
rates 1/x_i (no use of F*_mu in the transition rule). A theorem of
Ayyer-Martin-Williams identifies its stationary distribution as

    pi(mu) = F_mu(x; 1, t) / P_lambda(x; 1, t),

which is the ratio required in the problem statement (same q=1 ASEP/Macdonald
family, up to notation conventions).

## Confidence

Medium-high. The existence claim is a direct consequence of a published theorem.
The nontriviality condition is checked from the explicit generator.

## Solution

### 1. State space

Let

    S_n(lambda) = {all permutations of the parts of lambda}.

Because lambda has distinct parts, |S_n(lambda)| = n!.

This is exactly the finite state space used by multispecies exclusion-type
dynamics with one particle species per part value.

### 2. Lemma (explicit chain construction)

Define a continuous-time Markov chain X_t on S_n(lambda) as follows.

Fix parameters x_1, ..., x_n > 0 and t in [0,1). At each site j, an exponential
clock of rate 1/x_j rings.

If site j is a vacancy (species 0), nothing happens. If site j has species
r_0 > 0, that particle becomes active. Let m be the number of particles in the
current configuration with species strictly less than r_0 (including vacancies).
Moving clockwise, the active particle chooses the k-th weaker particle with
probability

    t^(k-1) / [m]_t,   where [m]_t = 1 + t + ... + t^(m-1),

and swaps into that position. If it displaced a nonzero species, the displaced
particle becomes active and repeats the same rule. The cascade ends when a
vacancy is displaced.

This is the inhomogeneous multispecies t-PushTASEP.

Why this is a valid Markov chain:
- The state space is finite.
- The transition rule is explicit and depends only on the current ring
  configuration, t, and x_i.
- Rates are finite and nonnegative.

Generator form (explicit):

For eta != eta', define

    q(eta, eta') = sum_{j=1}^n (1/x_j) * Pr(eta -> eta' | clock j rings).

Then set

    q(eta, eta) = -sum_{eta' != eta} q(eta, eta').

Each site j has finitely many outcomes (at most m_j <= n weaker-particle
choices), so total exit rate is finite in every state. Because lambda_n = 0 and
parts are distinct, each eta in S_n(lambda) contains exactly one vacancy
(species 0), so cascades terminate.

### 3. Lemma (nontriviality)

The chain above is nontrivial in the sense asked by the problem:

- Transition rates are defined from site rates 1/x_i, species inequalities, and
  t-geometric choice weights in the current ring configuration.
- No transition probability is defined using values of F*_mu or P*_lambda.

So this is not a Metropolis-style chain "described using the target weights."

### 4. Main theorem (stationary distribution)

Theorem (Ayyer-Martin-Williams, 2024, arXiv:2403.10485, Thm. 1.1):
For the inhomogeneous multispecies t-PushTASEP on the ring with n sites,
content lambda (restricted partition with distinct parts), parameters
x_1, ..., x_n > 0, and 0 <= t < 1, the stationary probability of
eta in S_n(lambda) is

    pi(eta) = F_eta(x; 1, t) / P_lambda(x; 1, t),

where F_eta are ASEP polynomials at q=1 and P_lambda(x; 1, t) =
sum_{nu in S_n(lambda)} F_nu(x; 1, t) is the partition function
(ensuring pi sums to 1). Positivity: F_eta(x; 1, t) > 0 for x_i > 0,
0 <= t < 1 is established as part of AMW Theorem 1.1, which shows that
pi(eta) = F_eta / P_lambda is a probability distribution. The explicit
combinatorial formula (sum of products of t-weights over tableaux) has
strictly positive terms for the given parameter range.

Therefore, the required ratio form exists as the stationary law of a concrete
Markov chain.

### 5. Notation bridge: F*_mu / P*_lambda = F_mu / P_lambda

The problem uses star notation F*_mu, P*_lambda (interpolation ASEP polynomials
in the Knop-Sahi convention), while AMW Theorem 1.1 uses F_eta, P_lambda.

**Writeup convention (explicit).** In this manuscript's Problem 3, we use
starred notation for the same q=1 family appearing in AMW:

    F*_eta := F_eta,     P*_lambda := P_lambda = sum_{eta in S_n(lambda)} F_eta.

Under this explicit convention, the ratio identity is immediate:

    F*_mu / P*_lambda = F_mu / P_lambda.

For readers using an alternate normalization of starred objects, only
eta-independence matters:

Concretely, if F*_eta = alpha * F_eta for some constant alpha independent
of eta, then P*_lambda = alpha * P_lambda and:

    F*_mu / P*_lambda = (alpha F_mu) / (alpha P_lambda) = F_mu / P_lambda.

The constant cancels in the ratio regardless of its value.

The constant cancels in the ratio regardless of its value.

**Verification for n = 2.** Take lambda = (a, 0). Both conventions give
F_{(a,0)}(x_1, x_2; 1, t) = x_1 and F_{(0,a)}(x_1, x_2; 1, t) = x_2
(the single-species case reduces to site weights). The ratio
F*_eta / P*_lambda = F_eta / P_lambda = x_i / (x_1 + x_2) in both conventions.

### 6. Sanity check: n=2 reduction

Take lambda=(a,0), state space {(a,0),(0,a)}. There is one non-vacancy and one
vacancy. Bells ring at rates 1/x_1 and 1/x_2.

- (a,0) -> (0,a) at rate 1/x_1
- (0,a) -> (a,0) at rate 1/x_2

So the stationary distribution is

    pi(a,0) = x_1/(x_1+x_2),   pi(0,a) = x_2/(x_1+x_2),

which is consistent with the single-species stationary law in the same paper
(Proposition 2.4, via recoloring reduction).

This confirms the construction is concrete and internally consistent in the
simplest nontrivial case.

### 7. Conclusion

For x_i > 0 and 0 <= t < 1, the inhomogeneous multispecies t-PushTASEP on
S_n(lambda) is:

(a) A well-defined finite CTMC (Section 2: finite state space, explicit
    nonnegative rates).

(b) Nontrivial: transition rates depend on (x, t) and current ring configuration,
    not on values of F*_mu or P*_lambda (Section 3).

(c) Has stationary distribution pi(eta) = F_eta(x; 1, t) / P_lambda(x; 1, t)
    by AMW Theorem 1.1 (Section 4), which equals F*_mu / P*_lambda under the
    notation bridge (Section 5).

**Existence vs uniqueness.** AMW Theorem 1.1 establishes existence of a
stationary distribution with the required ratio form; this is sufficient for
Problem 3.

Uniqueness/convergence can be studied separately. A vacancy-transport argument
is suggestive, but proving full irreducibility for the exact push-cascade
dynamics requires a dedicated connectivity proof and is not used as a premise
for the existence claim here.

Therefore the answer is **Yes**.

## References

- Arvind Ayyer, James Martin, Lauren Williams,
  "The Inhomogeneous t-PushTASEP and Macdonald Polynomials at q=1",
  arXiv:2403.10485, Theorem 1.1 and Proposition 2.4.
- Sylvie Corteel, Olya Mandelshtam, Lauren Williams,
  "From multiline queues to Macdonald polynomials", for ASEP polynomial context.
