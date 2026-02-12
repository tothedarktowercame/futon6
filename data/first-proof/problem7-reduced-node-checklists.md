# Problem 7 Reduced: Node-Level Done Criteria

## Purpose

Concrete completion criteria for reduced-P7 target nodes:

- `p7r-s2a`, `p7r-s2b` (E2 branch: `Gamma in FH(Q)`)
- `p7r-s3a`, `p7r-s3b` (S branch: manifold-upgrade + obstruction closure)

Derived from theorem-level extraction in:
- `1204.4667` (Fowler)
- `1506.06293` (Avramidi)
- `1101.0469`, `1401.0876` (Farrell-Jones for lattices)
- `0905.0104` (closed-manifold subgroup / assembly interface)

## `p7r-s2a` checklist (Fowler criterion module)

### Target statement

Show an orbifold extension group `Gamma` lies in `FH(Q)` using fixed-set Euler
data.

### Required inputs

1. A finite group action `G ↷ Bπ` on a finite complex.
2. Group identification:
   - `Gamma = pi_1((EG × Bπ)/G)` (orbifold fundamental group).
3. Fixed-set hypothesis:
   - for all nontrivial `H < G` and each connected component `C` of `(Bπ)^H`,
     `χ(C) = 0`.

### Completion criterion

Cite Fowler Main Theorem (`1204.4667`) to conclude:
- `Gamma ∈ FH(Q)`; equivalently
- there exists finite CW `X` with `pi_1(X)=Gamma` and `X~` rationally acyclic.

### Validation notes

- Fowler also notes a necessary-condition direction involving cyclic-subgroup
  fixed-set Euler characteristics (Section 4.1 framing). Use this as a sanity
  check against candidate actions.

## `p7r-s2b` checklist (concrete lattice-family instantiation)

### Target statement

Produce an explicit uniform lattice family with order-2 torsion and verify the
fixed-set Euler hypotheses needed for `p7r-s2a`.

### Status: DISCHARGED

Completed via the even-dimensional reflection route
(`problem7r-s2b-candidate-construction.md`):

- Arithmetic reflection lattice `Gamma_0 < Isom(H^n)` with `n` even >= 6
  (Douba-Vargas Pallete, arXiv:2506.23994).
- Congruence subgroup `pi = Gamma_0(I)` gives closed manifold `M = H^n/pi`.
- Extension `1 -> pi -> Gamma -> Z/2 -> 1` with order-2 torsion.
- Fixed set: totally geodesic hypersurface, dimension `n-1` (odd), so `chi = 0`.
- Fowler criterion applies: `Gamma in FH(Q)`.

See `problem7r-s2b-candidate-construction.md` for full details.

## `p7r-s3a` checklist (manifold-upgrade setup interface)

### Target statement

Set up a manifold realization step with explicit `pi_1` control for the chosen
group family.

### Available theorem assets

1. Avramidi Theorem 17 (`1506.06293`):
   - if `Gamma` is a `d`-dimensional duality group with finite `BΓ`, then for
     `r≥2`, `d+r≥5`, there is a compact `(d+r+1)`-manifold-with-boundary
     `(M,∂)` with `pi_1(M)=pi_1(∂)=Gamma` and rationally acyclic universal
     cover.
2. Avramidi rational surgery pipeline:
   - Theorems 14/16 give rational `pi-pi` / rational surgery interfaces used
     to produce these manifold models.

### Completion criterion

Show all required hypotheses are met for the selected `Gamma` and dimension
range, and record:
1. exact `d`, `r`, and dimension constraints;
2. finite-classifying-space and duality assumptions;
3. explicit point where `pi_1` is fixed as `Gamma`.

### Critical caveat

- Avramidi's closure to a closed manifold via reflection methods may alter the
  fundamental group unless extra control is shown. For reduced P7, any closure
  step must preserve final `pi_1 = Gamma`.

### Status: PROVISIONALLY CLOSED

All surgery prerequisites verified (`problem7r-s3a-setup.md`):

- P1: finitely presented (lattice) — verified
- P2: finite rational Poincare complex from E2 — verified
- P3: dimension >= 5 (n = 6) — verified
- P4: degree-1 normal map existence (via torsion-free cover) — verified
- P5: pi_1 preservation (surgery below middle dimension) — verified
- P6: rational acyclicity preserved — verified

Single remaining obligation passes to `p7r-s3b`.

## `p7r-s3b` checklist (obstruction computation + closure)

### Target statement

Compute/cite the relevant surgery obstruction group/class and prove the needed
vanishing in the selected lattice case.

### Available theorem assets

1. Farrell-Jones lattice coverage:
   - `1101.0469`: cocompact lattices in virtually connected Lie groups satisfy
     K/L-theoretic FJ (with additive-category coefficients, `VCyc` family).
   - `1401.0876`: extends to arbitrary lattices in virtually connected Lie
     groups.
2. Assembly/L-theory reduction framework:
   - surveys and computation templates (`1003.5002`, `1007.0845`, `1805.00226`,
     `math/0510602`, `1204.2418`).
3. Closed-manifold subgroup interface:
   - `0905.0104` Theorem A/B relate inertia/closed-manifold subgroups and
     assembly images under stated dimension/decoration conditions.

### Completion criterion

1. Identify the exact obstruction group (decoration and orientation character
   included) for the chosen dimension.
2. Use FJ to reduce to an assembly/equivariant-homology calculation in the
   allowed family.
3. Prove vanishing (or required membership) of the specific obstruction class.
4. Verify that this implies realizability in the closed-manifold category for
   the same `Gamma`.

### Critical caveats

- FJ gives computability/reduction, not automatic vanishing.
- Assembly image vs closed-manifold subgroup equality depends on hypotheses
  (dimension, decorations, orientation character, stabilization regime). Do not
  silently assume equality in unsupported settings.

### Status: OPEN (advanced)

The FJ reduction is complete (`problem7r-s3b-obstruction.md`):

- UNil terms vanish rationally (Connolly-Davis), so `E_{Fin}` suffices.
- Computation reduces to `H_6^{Z/2}(M; L tensor Q)`.
- Transfer argument: `res(sigma(f)) = 0` (pi-level problem already solved).
- Obstruction localized to fixed-set contribution:
  `sigma(f) in Q + H_3(F; Q) + H_1(F; Q)`.
- Dimension-parity tension identified: E2 needs even n, S prefers odd n.
- Five resolution strategies proposed (A through E).

Proof remains **conditional on `sigma(f) = 0`**, but the obstruction is now
localized to a specific subspace determined by fixed-set topology.

## Fast execution order (recommended)

1. `p7r-s2a`: lock theorem interface from Fowler Main Theorem. **Done.**
2. `p7r-s2b`: produce true order-2 instantiation. **Done** (reflection route).
3. `p7r-s3a`: fix surgery setup with `pi_1` control. **Done** (Wall framework).
4. `p7r-s3b`: run FJ reduction plus obstruction computation. **Advanced**
   (obstruction localized to fixed-set piece; not yet closed).
5. Only then assert `p7r-s5`. **Blocked on s3b.**

## Current bottleneck

The obstruction `sigma(f) in Q + H_3(F; Q) + H_1(F; Q)` is the single
remaining mathematical obligation. The most promising resolution paths are:

- **Strategy B** (lattice selection): find congruence subgroup with small
  `b_1(F)`, `b_3(F)`.
- **Strategy C** (equivariant signature): compute the equivariant signature
  defect for the specific normal map.
- **Strategy E** (conditional): accept the conditional statement with the
  tightened obstruction characterization.
