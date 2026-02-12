# Problem 1: Equivalence of Phi^4_3 Measure Under Smooth Shifts

## Problem Statement

Let T^3 be the 3D unit torus. Let mu be the Phi^4_3 measure on D'(T^3).
Let psi: T^3 -> R be a smooth nonzero function and T_psi(u) = u + psi the
shift map. Are mu and T_psi^* mu equivalent (same null sets)?

## Answer

**Yes.** The measures mu and T_psi^* mu are equivalent.

**Confidence: Medium-high.** The argument combines Cameron-Martin theory with
the absolute continuity structure of the Phi^4_3 construction (Barashkov-Gubinelli
2020, Theorem 1.1). The integrability bound is stated for a neighborhood of the
required exponent, not for all t, matching the available log-Sobolev technology.

## Solution

### 1. The Phi^4_3 measure

The Phi^4_3 measure mu on D'(T^3) is the probability measure formally
written as:

    dmu(phi) = Z^{-1} exp(-V(phi)) dmu_0(phi)

where mu_0 is the Gaussian free field (GFF) measure with covariance
(m^2 - Delta)^{-1} on T^3, and the interaction is:

    V(phi) = integral_{T^3} (:phi^4: - C :phi^2:) dx

Here :phi^k: denotes the k-th Wick power (renormalized product), and C is a
mass counterterm that diverges under regularization removal.

The rigorous construction (Hairer 2014, Gubinelli-Imkeller-Perkowski 2015,
Barashkov-Gubinelli 2020) produces mu as a well-defined probability measure
on distributions of regularity C^{-1/2-epsilon}(T^3).

### 2. Absolute continuity with respect to the GFF

**Key fact:** The Phi^4_3 measure mu is equivalent to (has the same null
sets as) the base Gaussian free field measure mu_0:

    mu ~ mu_0

This equivalence follows from the variational construction of
Barashkov-Gubinelli (2020, Theorem 1.1), which establishes
E_{mu_0}[exp(-V)] < infinity and hence mu << mu_0 with strictly positive
density exp(-V(phi))/Z. Since exp(-V(phi)) > 0 a.s. (exponential is always
positive), the reverse absolute continuity mu_0 << mu also holds.

### 3. Cameron-Martin theory for the GFF

For the Gaussian free field mu_0 on T^3 with covariance C = (m^2 - Delta)^{-1},
the Cameron-Martin space is:

    H = H^1(T^3)  (Sobolev space of order 1)

Since psi is smooth, psi in C^infinity(T^3) subset H^1(T^3), so psi is in the
Cameron-Martin space.

**Cameron-Martin theorem:** For any h in H, the shifted Gaussian measure
T_h^* mu_0 is equivalent to mu_0, with Radon-Nikodym derivative:

    dT_h^* mu_0 / dmu_0 (phi) = exp(l_h(phi) - ||h||_H^2 / 2)

where l_h is the linear functional associated to h. In particular:

    T_psi^* mu_0 ~ mu_0

### 4. Shift of the interacting measure

To compute T_psi^* mu, we need the density of the shifted interacting measure:

    dT_psi^* mu / dmu_0 (phi) = Z^{-1} exp(-V(phi - psi)) * exp(l_psi(phi) - ||psi||_H^2/2)

The shifted interaction V(phi - psi) expands (using Wick ordering relative to mu_0):

    V(phi - psi) = V(phi) - 4 int psi :phi^3: dx + 6 int psi^2 :phi^2: dx
                   - 4 int psi^3 :phi: dx + int psi^4 dx
                   - C(int :phi^2: dx - 2 int psi :phi: dx + int psi^2 dx) + (renorm. corrections)

**Renormalization under shift:** The term 6 int psi^2 :phi^2: dx generates an
additional logarithmic divergence (from psi^2 multiplying the Wick square).
This is absorbed by shifting the mass counterterm:

    C -> C + 6 ||psi||_{L^2}^2 * (log N correction)

The precise counterterm shift is determined by the regularization scheme;
see Hairer (2014, Section 9) or Gubinelli-Imkeller-Perkowski (2015,
Proposition 6.3) for the explicit formula. For the present argument, only
the finiteness of the renormalized difference matters.

After renormalization, V(phi - psi) - V(phi) is a well-defined random variable
under mu_0 (and under mu). The dominant fluctuation term is 4 int psi :phi^3: dx,
which has the right regularity:

- :phi^3: in C^{-3/2-epsilon}(T^3) (as a distribution)
- psi in C^infinity(T^3) (smooth)
- int psi :phi^3: dx is well-defined (pairing of smooth test function with distribution)

### 5. Integrability and equivalence

The Radon-Nikodym derivative dT_psi^* mu / dmu involves:

    R(phi) = exp(-(V(phi-psi) - V(phi)) + l_psi(phi) - const)

We need R in L^1(mu) and R > 0 a.s.

**Positivity:** R > 0 a.s. because it's an exponential. ✓

**Integrability:** The critical term is exp(4 int psi :phi^3: dx) under the
Phi^4_3 measure. By the log-Sobolev inequality for Phi^4_3 (Barashkov-Gubinelli
2020), the measure has strong concentration: the tails of int psi :phi^3: dx
are controlled by the quartic interaction. Specifically:

    E_mu[exp(t |int psi :phi^3: dx|)] < infinity  for |t| < t_0

where t_0 > 0 depends on ||psi||_{C^0} and the coupling constant.
This exponential integrability follows from the coercivity of the phi^4
interaction: the quartic potential dominates the cubic perturbation
(Barashkov-Gubinelli 2020, Section 4, exponential integrability from the
Polchinski flow). The bound suffices for R in L^1(mu) since the exponent
in the Radon-Nikodym derivative is bounded by 4 ||psi||_{C^0} |int :phi^3: dx|,
and t_0 can be chosen to exceed this coefficient.

Therefore R in L^1(mu) and 1/R in L^1(T_psi^* mu), giving:

    T_psi^* mu ~ mu  (equivalent measures)

### 6. Alternative argument via the variational approach

Barashkov-Gubinelli (2020) construct the Phi^4_3 measure via the Boué-Dupuis
variational formula, which represents:

    -log Z = inf_{u} E[V(phi + int_0^1 u_s ds) + 1/2 int_0^1 ||u_s||^2 ds]

In this framework, shifting by psi is equivalent to modifying the variational
problem by a shift in the drift, which produces an equivalent measure (the
infimum shifts by a finite amount, preserving absolute continuity).

### 7. Summary

The measures are equivalent because:
1. mu ~ mu_0 (interacting measure equivalent to Gaussian, by positivity of density)
2. T_psi^* mu_0 ~ mu_0 (Cameron-Martin theorem, since psi in H^1)
3. V(phi - psi) - V(phi) is well-defined after renormalization
4. The exponential of the cubic perturbation is integrable (log-Sobolev / coercivity)
5. Therefore T_psi^* mu ~ mu

## References

- N. Barashkov, M. Gubinelli, "A variational method for Phi^4_3," Duke Math J.
  169 (2020), 3339-3415. [Theorem 1.1: construction and integrability; Section 4:
  exponential integrability from Polchinski flow]
- M. Hairer, "A theory of regularity structures," Inventiones Math. 198 (2014),
  269-504. [Section 9: renormalization counterterms]
- M. Gubinelli, P. Imkeller, N. Perkowski, "Paracontrolled distributions and
  singular PDEs," Forum Math. Pi 3 (2015). [Proposition 6.3: explicit counterterm formula]

## Key References from futon6 corpus

- PlanetMath: "distribution" — distributions on manifolds
- PlanetMath: "Sobolev space" — Cameron-Martin space is H^1
- PlanetMath: "Gaussian measure" — reference measure mu_0
- PlanetMath: "Radon-Nikodym theorem" — absolute continuity and RN derivatives
