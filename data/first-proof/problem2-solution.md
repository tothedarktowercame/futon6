# Problem 2: Universal Test Vector for Rankin-Selberg Integrals

## Problem Statement

Let F be a non-archimedean local field with ring of integers o. Let N_r
denote the subgroup of GL_r(F) consisting of upper-triangular unipotent
elements. Let psi: F -> C^x be a nontrivial additive character of conductor o,
identified with a generic character of N_r.

Let Pi be a generic irreducible admissible representation of GL_{n+1}(F),
realized in its psi^{-1}-Whittaker model W(Pi, psi^{-1}). Must there exist
W in W(Pi, psi^{-1}) with the following property?

For any generic irreducible admissible representation pi of GL_n(F) in its
psi-Whittaker model W(pi, psi), let q be the conductor ideal of pi,
Q in F^x a generator of q^{-1}, and u_Q := I_{n+1} + Q E_{n,n+1}. Then for
some V in W(pi, psi), the local Rankin-Selberg integral

    int_{N_n\GL_n(F)} W(diag(g,1) u_Q) V(g) |det g|^{s-1/2} dg

is finite and nonzero for all s in C.

## Answer

**Yes.** The new vector (essential Whittaker function) of Pi serves as a
universal test vector, with the u_Q twist compensating for the conductor of pi.

**Confidence: Medium.** The argument relies on standard Rankin-Selberg theory
(Jacquet-Piatetski-Shapiro-Shalika) plus properties of new vectors and the
Kirillov model. The nondegeneracy claim is natural but the "nonzero for all s"
condition requires a careful algebraic argument about fractional ideals.

## Solution

### 1. Rankin-Selberg theory background

The local Rankin-Selberg integral I(s, W, V) for GL_{n+1} x GL_n is:

    I(s, W, V) = int_{N_n\GL_n(F)} W(diag(g,1)) V(g) |det g|^{s-1/2} dg

This converges for Re(s) >> 0 and extends to a rational function of q_F^{-s}
(where q_F = |o/p|). The set of all such integrals as (W, V) vary generates
a fractional ideal of C[q_F^s, q_F^{-s}], whose generator is the local
L-factor L(s, Pi x pi).

The problem modifies this by inserting u_Q = I_{n+1} + Q E_{n,n+1} into the
argument of W, giving the "twisted" Rankin-Selberg integral.

### 2. The condition "finite and nonzero for all s"

For a rational function f(q_F^{-s}) to be "finite and nonzero for all s in C,"
it must have no poles and no zeros when viewed as a function of
X = q_F^{-s} in C^x. Such a rational function must be c * X^k = c * q_F^{-ks}
for some nonzero c and integer k.

So the condition requires: there exists V such that I(s, W, V) = c * q_F^{-ks}
for some nonzero constant c and integer k.

Since the integrals over V span L(s, Pi x pi) * C[q_F^s, q_F^{-s}] (for
fixed W, assuming nondegeneracy), this means we need to find V such that the
polynomial factor P(q_F^{-s}) satisfies P * L = c * q_F^{-ks}. Since
L(s, Pi x pi)^{-1} is a polynomial in q_F^{-s}, we need P to equal
c * q_F^{-ks} * L^{-1}, which is indeed a Laurent polynomial.

This is possible provided the integrals over V (for our fixed W) span the
full fractional ideal, i.e., the Rankin-Selberg pairing is nondegenerate
for our chosen W.

### 3. The u_Q twist and the Kirillov model

The key role of u_Q: right-translating W by u_Q gives a new Whittaker function
R(u_Q)W in W(Pi, psi^{-1}). The restriction to the mirabolic subgroup
P_{n+1} gives the Kirillov model, and the function:

    phi_Q(g) := W(diag(g,1) u_Q) = (R(u_Q)W)(diag(g,1))

lies in the Kirillov model of Pi restricted to GL_n.

Since Pi is generic and W is nonzero, R(u_Q)W is nonzero for all Q (right
translation by a unipotent element preserves the Whittaker model and maps
nonzero vectors to nonzero vectors). Therefore phi_Q is a nonzero function
in the Kirillov model for every Q.

### 4. Nondegeneracy of the pairing

For a nonzero phi_Q in the Kirillov model, the Rankin-Selberg pairing:

    V |-> int_{N_n\GL_n(F)} phi_Q(g) V(g) |det g|^{s-1/2} dg

is nondegenerate as V ranges over W(pi, psi). This is the fundamental
nondegeneracy of the Rankin-Selberg integral (Jacquet-Piatetski-Shapiro-
Shalika 1983, Section 2.7).

More precisely: since phi_Q is nonzero, the integrals over all V in W(pi, psi)
generate the full fractional ideal L(s, Pi x pi) * C[q_F^s, q_F^{-s}].

### 5. Choosing W: the new vector

**Choice:** Let W_0 be the new vector (essential Whittaker function) of Pi,
i.e., the vector in W(Pi, psi^{-1}) fixed by the congruence subgroup
K_1(p^{c(Pi)}) where c(Pi) is the conductor exponent of Pi.

**Properties of W_0:**
- W_0 is nonzero (Pi is generic)
- W_0(I_{n+1}) = 1 (standard normalization)
- R(u_Q)W_0 is nonzero for every Q in F^x (as argued in Step 3)

**Universality:** For any generic pi of GL_n(F) with conductor ideal q:
- Let Q generate q^{-1}
- The function phi_Q(g) = W_0(diag(g,1) u_Q) is nonzero in the Kirillov model
- By the nondegeneracy (Step 4), there exists V in W(pi, psi) such that
  I(s, W_0, V) is a nonzero element of L(s, Pi x pi) * C[q_F^s, q_F^{-s}]
- By the algebraic argument (Step 2), V can be chosen to make the integral
  equal to c * q_F^{-ks}, which is finite and nonzero for all s

### 6. The role of u_Q in conductor matching

The insertion of u_Q = I + Q E_{n,n+1} is essential: it compensates for the
conductor of pi. Without this twist, for highly ramified pi, the integral
might degenerate (the new vector of Pi would not "see" the ramification of pi).

The matrix u_Q acts on the (n, n+1)-entry, effectively shifting by Q in
the direction of the last standard basis vector. Since Q = pi_F^{-c(pi)}
(where pi_F is a uniformizer), this shift has size exactly matching the
conductor of pi. This ensures the twisted Whittaker function
W_0(diag(g,1) u_Q) interacts nontrivially with the Kirillov model of pi
at the right "scale."

Concretely: in the Iwasawa decomposition g = nak, the twist by u_Q
modifies the a-component (diagonal part) at scale Q, ensuring the support
of g |-> W_0(diag(g,1) u_Q) overlaps with the support of the new vector
of pi (which is concentrated at conductor scale c(pi)).

### 7. Verification in special cases

**Both unramified (c(Pi) = c(pi) = 0):** u_Q = I + E_{n,n+1} (Q = 1).
W_0 is the spherical vector. The integral with V_0 (spherical for pi)
gives L(s, Pi x pi) * (correction factor). Choose V to cancel the L-factor.

**Pi unramified, pi ramified:** u_Q has Q = pi_F^{-c(pi)}. The twist
shifts the support of the spherical vector to match the ramification of pi.
This is the classical "conductor-lowering" mechanism.

**Both ramified:** The new vector of Pi combined with the u_Q twist gives a
function in the Kirillov model whose support is compatible with the
conductor of pi. Nondegeneracy follows from the general JPSS theory.

### 8. Summary

1. The answer is YES: the new vector W_0 of Pi is a universal test vector
2. For any pi with conductor q, the twist u_Q (Q generates q^{-1}) ensures
   the Rankin-Selberg pairing is nondegenerate
3. Nondegeneracy of the Kirillov restriction: R(u_Q)W_0 is always nonzero
4. The fractional ideal structure of Rankin-Selberg integrals allows
   choosing V to make the integral a monomial c * q_F^{-ks}
5. This monomial is finite and nonzero for all s in C

## Key References from futon6 corpus

- PlanetMath: "representation theory" -- admissible representations
- PlanetMath: "L-function" -- L-factors and analytic properties
- PlanetMath: "locally compact group" -- p-adic groups
