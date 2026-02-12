# Problem 7 Reduced: Strategy Mining from arXiv

## Scope

This note mines proof **strategy modules** from the curated 22-paper set and
maps them to the reduced P7 wiring diagram:

- Wiring JSON: `data/first-proof/problem7-reduced-wiring.json`
- Key target nodes:
  - `p7r-s2a`, `p7r-s2b` (E2: place `Gamma` in `FH(Q)`)
  - `p7r-s3a`, `p7r-s3b` (S: manifold-upgrade + obstruction closure)

## Reduced theorem (for reference)

If for a concrete uniform lattice `Gamma` with order-2 torsion:

1. `E2`: one proves `Gamma in FH(Q)` (finite CW model with Q-acyclic universal cover), and
2. `S`: one proves a manifold-upgrade with `pi_1` preserved and surgery obstruction closed,

then there exists a closed manifold `M` with `pi_1(M)=Gamma` and
`H_*(M_tilde;Q)=0` for `*>0`.

## Strategy module library (wiring motifs)

### M1: Equivariant finiteness via fixed-point Euler data

- Core sources:
  - `1204.4667` (Fowler), Main Theorem + Section 5 examples
  - `math/0008070`, `1707.07960` (Wall finiteness obstruction background)
- Motif wiring:
  - finite-group action on finite complex
  - -> fixed-set Euler characteristics vanish on nontrivial subgroups/components
  - -> equivariant finiteness obstruction vanishes
  - -> orbifold extension group in `FH(Q)`
- Maps to:
  - primary: `p7r-s2a`
  - secondary: `p7r-s2b` (instantiation checks)
- Transfer value:
  - direct and high for the E2 branch.
- Known caveat:
  - Fowler's arithmetic lattice example is strongest for odd-order actions; explicit
    order-2 lattice instantiation still needs separate verification.

### M2: Rational duality-group to manifold-with-boundary, then closed model

- Core sources:
  - `1506.06293` (Avramidi), Theorems 3/17 + reflection-group closure
  - `1204.4667` (reflection-group method context)
- Motif wiring:
  - duality group + finite classifying space
  - -> rational equivariant Moore-space input
  - -> rational surgery (`pi-pi` style interface) to manifold-with-boundary
  - -> reflection-group method to closed manifold with Q-acyclic universal cover
- Maps to:
  - primary: `p7r-s3a`
  - partial: `p7r-s3b`
- Transfer value:
  - high for surgery setup architecture.
- Known caveat:
  - this module is naturally torsion-free/duality-group framed; direct transfer to
    torsion lattice `Gamma` requires additional extension-equivariant bridge steps.

### M3: Lattice Farrell-Jones and assembly reduction

- Core sources:
  - `1101.0469`, `1401.0876` (FJ for cocompact/arbitrary lattices in virtually connected Lie groups)
  - `math/0510602`, `1003.5002`, `1007.0845`, `1805.00226`, `1204.2418`, `2507.11337`
- Motif wiring:
  - lattice class identified
  - -> FJ (with coefficients) available
  - -> assembly identifies relevant `K/L` groups with equivariant homology terms
  - -> obstruction computation is reduced to explicit family-level algebra/topology
- Maps to:
  - primary: `p7r-s3b`
  - support: `p7r-s3a`
- Transfer value:
  - high for computational reduction, not by itself a vanishing theorem.

### M4: Bredon/orbifold cohomology scaffold for torsion actions

- Core sources:
  - `1311.7629` (Bredon-Poincare duality groups)
  - `0705.3249` (orbifold translation groupoids + Bredon invariants)
  - `math/0312378` (classifying spaces for families)
- Motif wiring:
  - proper action with finite stabilizers
  - -> Bredon/orbifold cohomology formulation
  - -> duality and family-classifying-space language fixed
  - -> precise input domain for M1 and M3
- Maps to:
  - support: `p7r-s2a`, `p7r-s3a`
- Transfer value:
  - medium/high as formal infrastructure; usually not the final existence step.

### M5: Surgery subgroup / closed-manifold obstruction control

- Core sources:
  - `0905.0104` (closed-manifold subgroup vs inertia subgroup in surgery obstruction groups)
  - `math/0306054` (explicit L-group computations in torsion-containing examples)
- Motif wiring:
  - identify target `L`-group and geometric subgroup corresponding to closed-manifold realizations
  - -> compute obstruction image and detect vanish/non-vanish
  - -> infer realizability in closed-manifold category
- Maps to:
  - primary: `p7r-s3b`
- Transfer value:
  - medium/high for making S precise once reduction from M3 is done.

## Composition templates relevant to reduced P7

### Template A (E2-first, Fowler-forward)

- Shape:
  - M4 -> M1 -> (E2 complete)
  - then M3 + M5 -> (S complete)
  - then `p7r-s5`
- Strength:
  - closest to current reduced theorem structure.
- Main risk:
  - order-2 concrete lattice instantiation in `p7r-s2b`.

### Template B (S-first, Avramidi-forward)

- Shape:
  - M4 -> M2 gives manifold model for a related duality-group input
  - then extension/quotient transfer argument to torsion lattice
  - then M3/M5 for obstruction compatibility
- Strength:
  - strong constructive geometry for manifold-upgrade.
- Main risk:
  - nontrivial extension from torsion-free subgroup model to full `Gamma`.

### Template C (assembly-first obstruction workflow)

- Shape:
  - M3 computes/reduces obstruction terms
  - M5 handles geometric subgroup + closure criteria
  - M1 optionally supplies E2 model as fallback/parallel branch
- Strength:
  - good if vanishing can be shown directly for selected lattice family.
- Main risk:
  - reduction may still stop at "computable but not computed."

## 22-paper mapping by primary utility

- `p7r-s2a` / E2 criterion:
  - `1204.4667`, `math/0008070`, `1707.07960`
- `p7r-s2b` / concrete instantiation:
  - `1204.4667`, `math/0302218`, `0902.2480`
- `p7r-s3a` / manifold-upgrade setup:
  - `1506.06293`, `1311.7629`, `math/0312378`
- `p7r-s3b` / obstruction computation/vanishing:
  - `1101.0469`, `1401.0876`, `math/0510602`, `1003.5002`, `1007.0845`,
    `1805.00226`, `1204.2418`, `0905.0104`, `math/0306054`, `0901.0442`
- context and auxiliary splitting/structure:
  - `1110.2041`, `2107.00614`, `2507.11337`, `0705.3249`

## Immediate next extraction pass (highest ROI)

1. From `1204.4667`: isolate a checklist usable as a node-level verifier for
   `p7r-s2b` (exact fixed-set hypotheses and outputs).
2. From `1506.06293`: isolate the minimal surgery input contract for
   `p7r-s3a` (what must be true before the manifold-upgrade step).
3. From `1101.0469` + `1401.0876`: pin down the exact lattice classes and
   coefficient settings needed so `p7r-s3b` has a valid assembly interface.
4. From `0905.0104`: determine when "closed-manifold subgroup" identification
   can be used as a final closure criterion in the chosen dimension range.
