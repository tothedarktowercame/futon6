# Problem 7: Uniform Lattice with 2-Torsion and Rationally Acyclic Universal Cover

## Problem Statement

Suppose Gamma is a uniform lattice in a real semisimple Lie group, and Gamma
contains an element of order 2. Is it possible that Gamma is the fundamental
group of a closed manifold whose universal cover is acyclic over Q?

## Status

**Conditional yes.** The CW-complex obligation (E2) is discharged
unconditionally for two lattice families — reflections in even dimension and
rotations in odd dimension. The manifold-upgrade obligation (S) is open,
reduced to a computable surgery obstruction in the structurally favorable
rotation setting.

## Obligation E2: Gamma in FH(Q) — DISCHARGED

### Fowler's Criterion

Fowler (arXiv:1204.4667): if a finite group G acts on a finite CW complex Y
such that for every nontrivial subgroup H < G and every component C of the
fixed set Y^H, chi(C) = 0, then the orbifold extension lies in FH(Q) — there
exists a finite CW complex with the given fundamental group and rationally
acyclic universal cover.

### Reflection Construction (even dimension)

Take an arithmetic uniform lattice Gamma_0 < Isom(H^n) containing a
reflection tau, via Douba-Vargas Pallete (arXiv:2506.23994). Choose n even.
Let pi = Gamma_0(I) be a deep congruence subgroup (torsion-free by Minkowski).
Set Gamma = <pi, tau>. The fixed set of tau on M = H^n/pi has dimension n-1
(odd), so chi = 0. Fowler gives Gamma in FH(Q).

### Rotation Construction (odd dimension)

Let k = Q(sqrt(2)), f = (1-sqrt(2))x_0^2 + x_1^2 + ... + x_n^2 (n+1
variables). Under the two real embeddings, f has signatures (n,1) and
(n+1,0), so SO(f) gives SO(n,1) at one place and a compact group at the
other. Gamma_0 = SO(f, Z[sqrt(2)]) is a cocompact arithmetic lattice in
SO(n,1) (Borel-Harish-Chandra + Godement criterion).

The element sigma = diag(1, -1, -1, 1, ..., 1) is in SO(f, Z[sqrt(2)]):
it preserves f (negates x_1, x_2 in the x_1^2 + x_2^2 summand), has
determinant +1, and has integer entries. Its fixed set on H^n is H^{n-2}
(codimension 2).

For n = 2k+1 odd: fixed-set dimension n-2 = 2k-1 is odd, so chi = 0.
Fowler gives Gamma in FH(Q).

**Both constructions discharge E2.**

## Obligation S: Manifold Upgrade — OPEN

Problem 7 asks for a **closed manifold**, not just a finite CW complex.
Upgrading requires surgery theory. Five candidate approaches were analyzed:

### Approach IV (Rotation Route — Recommended)

The rotation construction in odd dimension n = 2k+1 has three structural
advantages:

1. **Dimension-parity tension dissolved.** E2 and S both favor odd n (unlike
   the reflection case, where E2 needs even n but surgery prefers odd n).
2. **Equivariant surgery available.** The codimension-2 gap hypothesis
   (Costenoble-Waner, arXiv:1705.10909) is satisfied.
3. **Favorable L-theory parity.** The surgery obstruction lives in
   L_{2k+1}(Z[Gamma]) tensor Q with odd total degree.

The S obligation splits into two sub-options:

**S-rot-II (equivariant surgery, recommended).** Cut out a tubular
neighborhood of F, cap off the boundary. The Browder-Lopez de Medrano
framework applies. Key findings:

- 2-primary obstructions (Browder-Livesay, UNil, Arf) vanish rationally.
- The Davis-Luck Z/2 exclusion (arXiv:2303.15765), which blocks the
  *integral* version, does NOT apply to the rational version.
- AHSS computation for n = 7: ker(res) = H_2(F; Q), where F = H^5/C is
  the fixed-point manifold.
- If b_2(F) = 0, the rational obstruction vanishes unconditionally and
  the integral obstruction is torsion (killable by finite-cover trick).

**Immediate question:** Does b_2(F) = 0 for the arithmetic 5-manifold
F = H^5/C arising from the Q(sqrt(2)) construction? This is a question
about the cohomology of congruence subgroups of SO(4,1).

**S-rot-I (Wall surgery, fallback).** Same three-obstacle structure
(Poincare complex, normal map, obstruction) but with favorable odd parity.
ker(res) = Q + H_1(F; Q) — always at least 1-dimensional, so strictly
harder than S-rot-II.

### Deprioritized Approaches

- **Approach I (Wall surgery, reflection):** Three sequential obstacles with
  unfavorable even L-theory parity.
- **Approach II (equivariant surgery, reflection):** BLOCKED — codim-2 gap
  fails for codim-1 fixed sets.
- **Approach III (orbifold resolution):** No known technique.

## Smith Theory Non-Obstruction

Smith theory over Z/p constrains mod-p homology of fixed sets but does not
obstruct Q-acyclicity. The transfer homomorphism shows fixed sets can be
rationally trivial without contradiction.

## Theorem

Let Gamma be a cocompact lattice extension 1 -> pi -> Gamma -> Z/2 -> 1
constructed from an arithmetic lattice containing an order-2 isometry, via
either the reflection route (n even, n >= 6) or the rotation route (n odd,
n >= 7). Then:

(a) **(Unconditional)** Gamma in FH(Q).

(b) **(Open)** Whether there exists a closed manifold M with pi_1(M) = Gamma
and rationally acyclic universal cover remains unresolved. The rotation route
reduces the problem to computing a single surgery obstruction in the
structurally favorable odd-dimensional setting, with the equivariant
obstruction potentially vanishing if b_2 of the fixed-point manifold is zero.

## References

- J. Fowler, arXiv:1204.4667. (Fowler's criterion)
- A. Douba, F. Vargas Pallete, arXiv:2506.23994. (Reflection lattices)
- A. Borel, Harish-Chandra, Ann. Math. 75 (1962). (Arithmetic lattices)
- A. Bartels, F. Farrell, W. Luck, arXiv:1101.0469. (Farrell-Jones)
- S. Costenoble, S. Waner, arXiv:1705.10909. (Equivariant surgery)
- J. F. Davis, W. Luck, arXiv:2303.15765. (Z/2 exclusion — integral only)
