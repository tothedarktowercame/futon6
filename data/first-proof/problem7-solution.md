# Problem 7: Uniform Lattice with 2-Torsion and Rationally Acyclic Universal Cover

## Problem Statement

Suppose `Gamma` is a uniform lattice in a real semisimple Lie group, and
`Gamma` contains an element of order `2`. Is it possible that `Gamma` is the
fundamental group of a closed manifold whose universal cover is acyclic over
`Q`?

## Status in This Writeup

**Not fully proved here.**

This revision isolates what is proved unconditionally, what is proved for a
nearby finite-CW version, and what remains open in the manifold realization
step for the torsion case.

## 1. Baseline Geometry

Let `G` be connected real semisimple, `K < G` maximal compact, and
`X = G/K` contractible. For a uniform lattice `Gamma < G` with torsion,
`X/Gamma` is a compact orbifold (not a manifold).

So the torsion-free argument `M = X/Gamma` does not apply directly.

## 2. Cohomological Structure (Unconditional)

For proper cocompact actions on contractible `X`, the orbifold/Bredon
framework gives rational Poincare-duality-type structure in dimension
`d = dim(X)`.

This is standard in the Brown-Luck orbifold/equivariant cohomology setup.

## 3. Strong Nearby Result: Finite-CW Realization (`FH(Q)`)

A theorem of Fowler gives a concrete criterion for getting a finite CW complex
with prescribed orbifold fundamental group and rationally acyclic universal
cover:

- If a finite group action on a finite complex has all nontrivial fixed-point
  components of Euler characteristic zero, then the orbifold group lies in
  `FH(Q)` (i.e., admits a finite CW model with rationally acyclic universal
  cover).

Fowler also gives arithmetic-lattice examples (in odd-dimensional hyperbolic
settings) where this criterion applies, yielding lattice extensions in `FH(Q)`.

This is substantial evidence for the torsion-orbifold direction, but it is a
**finite-complex** statement, not yet a closed-manifold statement.

## 4. Why This Does Not Yet Solve Problem 7

Problem 7 asks for a **closed manifold** `M` with `pi1(M)=Gamma` and
`H_*(M_tilde;Q)=0` for `*>0`.

Two nontrivial upgrades are still needed:

1. `FH(Q)` -> closed manifold with the **same** fundamental group.
2. Verification of the relevant rational surgery obstruction for the chosen
   torsion lattice.

## 5. What Existing Manifold Theorems Cover

Avramidi proves rational manifold models for **duality groups with finite
classifying space** (`BΓ` finite), producing manifolds-with-boundary (and then
closed models via reflection-group constructions) with rationally acyclic
universal covers.

This theorem package is powerful but is tailored to torsion-free duality-group
input. It does not by itself close the torsion-lattice case asked in Problem 7.

## 6. Conditional Theorem (Current Proof-Level Output)

Assume:

- **(E2)** A concrete uniform lattice `Gamma` with an order-2 element is shown
  to satisfy the finite-complex criterion needed for `Gamma in FH(Q)`.
- **(S)** The manifold-upgrade surgery obstruction for this `Gamma` vanishes in
  the required dimension/range.

Then there exists a closed manifold `M` with `pi1(M)=Gamma` and
`H_*(M_tilde;Q)=0` for `*>0`.

So the current output is: **conditional yes** under `(E2)+(S)`.

## 7. Practical Next Step to Close the Gap

For a specific cocompact lattice family with 2-torsion:

1. Prove the fixed-set Euler-vanishing hypothesis needed to place `Gamma` in
   `FH(Q)`.
2. Compute (or cite) the exact rational surgery obstruction class for that same
   `Gamma`.
3. Execute the manifold-upgrade step with explicit references to the precise
   surgery theorem used.

## References

- J. Fowler, *Finiteness Properties of Rational Poincare Duality Groups*,
  arXiv:1204.4667.
- G. Avramidi, *Rational Manifold Models for Duality Groups*,
  arXiv:1506.06293.
- K. S. Brown, *Cohomology of Groups*.
- W. Luck, *Transformation Groups and Algebraic K-Theory*.
- C. T. C. Wall, *Surgery on Compact Manifolds*.
- A. Bartels, F. T. Farrell, W. Luck, Farrell-Jones results for cocompact
  lattices in virtually connected Lie groups.
