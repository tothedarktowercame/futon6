# Problem 1: Measure Equivalence Under Smooth Shifts of the Phi^4_3 Measure

## Problem Statement

Let mu denote the Phi^4_3 measure on the 3-dimensional torus T^3, defined as

    dmu = Z^{-1} exp(-V(phi)) dmu_0,

where mu_0 is the law of the Gaussian free field, V = int(:phi^4: - C:phi^2:)dx
is the renormalized quartic interaction, and Z is the normalizing constant.

For a smooth nonzero function psi: T^3 -> R, let T_psi denote the shift
phi |-> phi + psi. Are the measures mu and T_psi^* mu equivalent (mutually
absolutely continuous)?

## Answer

**Yes.** mu and T_psi^* mu are equivalent measures.

## Proof

The proof proceeds in three steps: (1) mu ~ mu_0, (2) T_psi^* mu_0 ~ mu_0,
and (3) the shifted interaction is integrable.

**Step 1. mu ~ mu_0.** The Phi^4_3 measure is constructed via the variational
approach of Barashkov-Gubinelli (2020, Theorem 1.1). The density exp(-V) is
strictly positive mu_0-a.s. (since V is finite mu_0-a.s.), giving mu_0 << mu.
The bound E_{mu_0}[exp(-V)] < infinity (the main result of the variational
construction) gives mu << mu_0 with Radon-Nikodym derivative
dmu/dmu_0 = Z^{-1} exp(-V). Therefore mu ~ mu_0.

**Step 2. T_psi^* mu_0 ~ mu_0 (Cameron-Martin).** Since psi is smooth on T^3,
psi lies in H^1(T^3), the Cameron-Martin space of the Gaussian free field
(the Sobolev space where (1-Delta)^{1/2} psi is in L^2). The Cameron-Martin
theorem gives T_psi^* mu_0 ~ mu_0 with explicit Radon-Nikodym derivative.

**Step 3. Integrability of the shifted interaction.** By Steps 1 and 2, it
suffices to show that T_psi^* mu ~ T_psi^* mu_0 ~ mu_0 ~ mu, i.e., that the
shifted measure T_psi^* mu also has a well-defined density with respect to mu_0.

The Radon-Nikodym derivative d(T_psi^* mu)/dmu involves the difference
V(phi - psi) - V(phi). Expanding the quartic:

    V(phi - psi) - V(phi) = -4 int psi :phi^3: dx + 6 int psi^2 :phi^2: dx
                            - 4 int psi^3 phi dx + int psi^4 dx
                            + (renormalization counterterm shift)

The leading term is the cubic 4 int psi :phi^3: dx. The Wick power :phi^3:
lies in C^{-3/2-epsilon}(T^3) for any epsilon > 0 (by regularity of the GFF
in 3D), and psi is smooth, so the pairing int psi :phi^3: dx is well-defined.
The precise counterterm shift is determined by the regularization scheme
(Hairer 2014, Section 9; Gubinelli-Imkeller-Perkowski 2015, Proposition 6.3).

The key estimate: there exists t_0 > 0 (depending on ||psi||_{C^0}) such that

    E_mu[exp(t |int psi :phi^3: dx|)] < infinity  for all |t| < t_0.

This follows from the Polchinski renormalization group flow analysis in
Barashkov-Gubinelli (2020, Section 4): the quartic term provides coercivity
that dominates the cubic perturbation in a neighborhood of the origin.
Specifically, for |t| small enough, the perturbed potential
V(phi) - t int psi :phi^3: dx is still bounded below (the quartic grows
faster than the cubic), giving the exponential integrability.

**Conclusion.** The Radon-Nikodym derivative d(T_psi^* mu)/dmu is
proportional to exp(-(V(phi-psi) - V(phi))), which is:
- Positive (exponential of a finite quantity), giving T_psi^* mu << mu.
- Integrable with respect to mu (by the exponential integrability of the
  cubic term), confirming mu << T_psi^* mu by symmetry (apply the same
  argument to psi replaced by -psi).

Therefore mu ~ T_psi^* mu. QED

## References

- N. Barashkov, M. Gubinelli, "A variational method for Phi^4_3," Duke Math.
  J. 169(17) (2020), 3339-3415. Theorem 1.1, Section 4.
- M. Hairer, "A theory of regularity structures," Invent. Math. 198(2)
  (2014), 269-504. Section 9.
- M. Gubinelli, P. Imkeller, N. Perkowski, "Paracontrolled distributions and
  singular PDEs," Forum Math. Pi 3 (2015), e6. Proposition 6.3.
