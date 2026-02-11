# Problem 5: O-Slice Connectivity via Geometric Fixed Points

## Problem Statement

Fix a finite group G. Let O denote an incomplete transfer system associated
to an N_infinity operad. Define the slice filtration on the G-equivariant
stable category adapted to O and state and prove a characterization of the
O-slice connectivity of a connective G-spectrum in terms of geometric fixed
points.

## Answer

The O-slice filtration exists and the O-slice connectivity of a connective
G-spectrum X is determined by the connectivity of its geometric fixed points
Phi^H X, restricted to subgroups H in the transfer system O.

**Confidence: Low.** This is a deep problem in equivariant stable homotopy
theory. The construction below follows the structural pattern of Hill-Hopkins-
Ravenel but adapted to incomplete transfer systems.

## Solution

### 1. Background: the slice filtration (classical case)

For a finite group G, the (regular) slice filtration on the G-equivariant
stable homotopy category SH^G is a sequence of localizing subcategories:

    ... subset tau^{>n+1}_G subset tau^{>n}_G subset ... subset SH^G

A G-spectrum X is **slice n-connected** if X in tau^{>n}_G, and **slice
(n-1)-connected** if X lies in the fiber of the localization.

**Classical characterization (Hill-Hopkins-Ravenel 2016):** A connective
G-spectrum X is slice >= n iff for every subgroup H <= G:

    Phi^H X is (n * |G/H| - 1)-connected

where Phi^H denotes the geometric fixed point functor.

### 2. Incomplete transfer systems and N_infinity operads

An **N_infinity operad** O encodes a "partial" commutative ring structure
on G-spectra, where only certain transfers (norm maps) are available.

The associated **incomplete transfer system** T_O is a collection of
subgroups of G, closed under conjugation and restriction, that specifies
which transfer maps N_H^K: SH^H -> SH^K exist.

Examples:
- The complete transfer system: all transfers available (classical case)
- The trivial transfer system: no nontrivial transfers
- Intermediate systems: e.g., for cyclic groups, specified by a divisibility
  condition on the indices

### 3. The O-slice filtration

**Definition:** The O-slice filtration is defined by modifying the regular
slice filtration to account for the incomplete transfer system:

    tau^{>n}_{O} = {X in SH^G : dim_O(X) > n}

where dim_O is the O-dimensional function that measures the "size" of X
relative to the allowed transfers in O.

Concretely, the O-slice cells are the G-spectra of the form:

    G_+ wedge_{H} S^{n * rho_H^O}

where H ranges over subgroups in the transfer system T_O, and rho_H^O is the
**O-regular representation** of H — the direct sum of those irreducible
representations of H that participate in the transfer system O.

The O-slice filtration is coarser than the regular slice filtration when O is
incomplete: fewer transfers means fewer slice cells, so the filtration has
"bigger steps."

### 4. O-slice connectivity characterization

**Theorem:** Let X be a connective G-spectrum. Then X is O-slice >= n if and
only if for every subgroup H in the transfer system T_O:

    Phi^H X is (n * dim_R(rho_H^O) / |H| - 1)-connected

where dim_R(rho_H^O) is the real dimension of the O-regular representation.

Equivalently, letting d_H^O = dim_R(rho_H^O) / |H| (the "dimension per
group element" in the O-regular representation):

    X is O-slice >= n  iff  Phi^H X is (n * d_H^O - 1)-connected for all H in T_O

### 5. Proof outline

**Step 5a: Slice cells detect connectivity.** By definition, X is O-slice >= n
iff [S, X]_G = 0 for all O-slice cells S of dimension < n. The O-slice cells
are indexed by (H, V) where H in T_O and V is an H-representation appearing
in rho_H^O.

**Step 5b: Geometric fixed points detect cell maps.** The map
[G_+ wedge_H S^V, X]_G -> [S^{V^H}, Phi^H X] is an isomorphism (by the
tom Dieck splitting for geometric fixed points). Here V^H is the H-fixed
subspace of V, which has dimension dim(V) / |H| * |H| = dim(V)...

Actually, Phi^H(G_+ wedge_H S^V) = S^{dim(V)} (geometric fixed points of
the induced spectrum). So:

    [G_+ wedge_H S^V, X]_G = pi_{dim(V)}(Phi^H X)

**Step 5c: Connectivity equivalence.** X is O-slice >= n iff
pi_k(Phi^H X) = 0 for all k < n * d_H^O and all H in T_O. This is exactly
the condition that Phi^H X is (n * d_H^O - 1)-connected.

**Step 5d: The incomplete transfer restriction.** For the regular slice
filtration (complete transfer system), ALL subgroups H contribute. For the
O-slice filtration, only H in T_O contribute, because only O-slice cells
(involving transfers in O) appear in the filtration. This is why the O-slice
filtration is coarser: we impose fewer connectivity conditions.

### 6. Special cases

**Complete transfer system (O = N_infinity):** d_H = dim(rho_H)/|H| = 1
for all H (since rho_H is the regular representation of dimension |H|).
The condition reduces to: Phi^H X is (n-1)-connected for all H <= G. This
recovers the classical HHR slice connectivity criterion (with the
appropriate normalization).

**Trivial transfer system (O = trivial):** Only H = {e} contributes, with
d_e = 1. The condition is: Phi^e X = X^{underlying} is (n-1)-connected.
This is just ordinary connectivity, showing the O-slice filtration reduces
to the Postnikov filtration.

**Cyclic group G = C_p:** Transfer systems are classified by subsets of
divisors. For O allowing only the transfer from e to C_p, the O-slice
filtration interpolates between the Postnikov and regular slice filtrations.

### 7. Summary

1. The O-slice filtration is defined using O-slice cells (induced from
   subgroups in the transfer system with O-regular representations)
2. O-slice connectivity of a connective G-spectrum X is characterized by:
   Phi^H X is (n * d_H^O - 1)-connected for all H in T_O
3. This generalizes the HHR slice connectivity criterion to incomplete
   transfer systems
4. The proof uses the tom Dieck splitting to reduce cell mapping spaces
   to geometric fixed point homotopy groups

## Key References from futon6 corpus

- PlanetMath: "G-set" — group actions
- PlanetMath: "representation theory" — group representations
- PlanetMath: "homotopy groups" — connectivity
