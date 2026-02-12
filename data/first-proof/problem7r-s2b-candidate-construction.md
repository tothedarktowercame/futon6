# Problem 7 Reduced: `p7r-s2b` Candidate Construction (Query-Set Pass)

Date: 2026-02-12

## Target node

`p7r-s2b`: instantiate the Fowler fixed-set criterion with a **uniform lattice family with 2-torsion** and verify the fixed-set Euler conditions needed for `Gamma in FH(Q)`.

## Wiring shape for this node

- `Q`: produce explicit `1 -> pi -> Gamma -> Z/2 -> 1` with `Gamma` a cocompact lattice containing order-2 torsion.
- `A`: realize a `Z/2` action on finite `Bpi` (preferably a closed hyperbolic manifold).
- `F`: identify all connected components of `(Bpi)^{Z/2}`.
- `E`: prove `chi(C)=0` for every fixed component `C`.
- `O`: apply Fowler Main Theorem to conclude `Gamma in FH(Q)`.
- `B`: verify that this `Gamma` is exactly the lattice family targeted in Problem 7.

## Strategy modules mined from the query set

### S2B-M1 (direct criterion)

- Source: Fowler `arXiv:1204.4667`.
- Payload: if all fixed components for nontrivial subgroups have vanishing Euler characteristic, the orbifold extension group is in `FH(Q)`.
- Role: `E -> O`.

### S2B-M2 (order-2 reflective arithmetic lattice family)

- Source: Douba-Vargas Pallete `arXiv:2506.23994`, Remark 5.
- Payload: arithmetic uniform lattice `Gamma_0 < Isom(H^n)` containing a reflection; principal congruence covers yield closed manifolds with induced reflection and fixed totally geodesic hypersurface.
- Role: `Q -> A -> F`.

### S2B-M3 (finite-group realization fallback)

- Source: Belolipetsky-Lubotzky `arXiv:math/0406607`.
- Payload: any finite group can occur as full isometry group of a compact hyperbolic manifold.
- Role: alternative `Q/A` constructor when reflection-specific structure is unavailable.
- Limitation: does not by itself give fixed-set Euler control.

### S2B-M4 (fixed-set geometry filter)

- Source: Chen `arXiv:2501.11610`.
- Payload: involution-fixed-set formulas can extract global cobordism/parity information from fixed sets.
- Role: support for `F/E` sanity checks in involution cases.
- Limitation: not needed if Euler vanishing is automatic from odd dimension.

### S2B-M5 (periodic-action constraint filter)

- Source: Avramidi `arXiv:1106.1704`.
- Payload: strong constraints on homotopically trivial periodic diffeomorphisms in locally symmetric/aspherical settings.
- Role: reject impossible action models; not a constructor.

### S2B-M6 (finite-subgroup complexity control)

- Source: Samet `arXiv:1209.2484` (abstract-level pass from query results).
- Payload: quantitative control of finite subgroup/isotropy complexity in lattices.
- Role: bookkeeping support for large-family instantiation.

## Candidate completion route R1 (reflection-even-dimension route)

This is the strongest route from the current pass.

1. Pick **even** `n` and use the arithmetic setup in `2506.23994` (Remark 5):
   - `Gamma_0 = O'_f(O_k) < Isom(H^n)` is a uniform lattice,
   - `Gamma_0` contains a reflection `tau`.
2. Let `pi = Gamma_0(I)` be a sufficiently deep principal congruence subgroup:
   - `M = pi \ H^n` is a closed hyperbolic manifold,
   - `tau` induces an involution `tau_bar` on `M`.
3. Set `G = <tau_bar> ≅ Z/2` acting on `Bpi := M`.
4. Fixed set check:
   - `Fix(tau_bar)` is a (possibly disconnected) closed, embedded, totally geodesic hypersurface (`2506.23994`), so each component has dimension `n-1`.
   - Since `n` is even, `n-1` is odd, so every closed component `C` has `chi(C)=0`.
5. Fowler criterion (`1204.4667`) now applies for `G=Z/2`:
   - only nontrivial subgroup is `H=Z/2`,
   - every component of `(Bpi)^H` has zero Euler characteristic,
   - hence the orbifold extension group `Gamma = pi_1((EG x Bpi)/G)` lies in `FH(Q)`.
6. Structural output:
   - `1 -> pi -> Gamma -> Z/2 -> 1`,
   - `Gamma` has order-2 torsion,
   - `Gamma` is a cocompact lattice as a finite extension of cocompact `pi` inside `Isom(H^n)`.

## Node-level status after this pass

- `Q` (explicit order-2 lattice family): **satisfied** via `2506.23994` route.
- `A` (finite action model on finite `Bpi`): **satisfied** (`M` compact manifold).
- `F` (fixed set identification): **satisfied** (totally geodesic hypersurface components).
- `E` (Euler-vanishing): **satisfied** for even `n` by odd-dimensional fixed components.
- `O` (Fowler implication): **satisfied**.
- `B` (match to P7 reduced target): **satisfied at `p7r-s2b` level**.

## Remaining caution points (for proof-writeup integration)

1. Keep groups distinguished in notation (`Gamma_0`, `pi`, `Gamma`) to avoid extension/lattice ambiguity.
2. Cite precisely the statement in `2506.23994` that gives nonempty fixed hypersurface in the congruence cover setting.
3. State explicitly that the finite-group action used in Fowler is `Z/2` and that no other nontrivial subgroup checks are needed.
4. Keep this closure scoped to `p7r-s2b` (it does not resolve the manifold-upgrade branch `S`).

## Practical integration target

The reduced P7 pipeline can now treat `p7r-s2b` as provisionally closed and shift effort to:
- `p7r-s3a` (manifold-upgrade with `pi_1` control),
- `p7r-s3b` (explicit obstruction computation/vanishing).
