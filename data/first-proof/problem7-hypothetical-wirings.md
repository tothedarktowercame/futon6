# Problem 7: Hypothetical Proof Wiring Diagrams

Date: 2026-02-12

These are candidate proof architectures for obligation S (manifold upgrade).
Each diagram represents a possible complete proof path, with nodes marked as
either **solid** (established), **open** (needs work), or **blocked** (known
obstruction). The goal is to identify which paths are worth pursuing rather
than holding all possibilities in memory.

## Diagram H1: Rotation Route (MOST PROMISING — new)

**Key idea:** Use a rotational involution (codimension-2 fixed set) in odd
ambient dimension. This dissolves the dimension-parity tension: E2 works
(odd-dim fixed set has chi = 0), surgery benefits from odd L-theory parity,
and the codimension-2 gap hypothesis is satisfied for equivariant surgery.

```
[E2-rot] Lattice with order-2 rotation in Isom(H^{2k+1})
    |
    | Fixed set: H^{2k-1} (codim 2, odd-dim, chi = 0)
    v
[Fowler] chi(C) = 0 for all fixed components --> Gamma in FH(Q)
    |
    v
[S-branch: two sub-options]
    |
    +--[S-rot-I] Wall surgery in dim 2k+1 (odd)
    |     |
    |     | L_{2k+1}(Z[Gamma]) tensor Q: favorable parity
    |     | AHSS: q = 0 mod 4 forces p odd, better vanishing
    |     v
    |   [Obstruction] sigma(f) = 0?  (OPEN — but structurally favorable)
    |
    +--[S-rot-II] Equivariant surgery on (M, tau)
          |
          | Codim-2 gap hypothesis: SATISFIED
          | Costenoble-Waner framework applies
          v
        [Equivariant surgery obstruction] (OPEN — computable)
```

### Node status

| Node | Status | Detail |
|---|---|---|
| E2-rot: lattice existence | **OPEN (key question)** | Need arithmetic lattice in Isom(H^{2k+1}) with order-2 rotation. Belolipetsky-Lubotzky (math/0406607) guarantees Z/2 as isometry group, but does not control codimension of fixed set. Douba-VP constructs reflections specifically. Need a "rotational" analogue. |
| Fowler application | **Solid (if E2-rot)** | Same Fowler criterion; chi = 0 for odd-dim fixed set is automatic. |
| S-rot-I: Wall surgery | **Open** | Same three obstacles as Approach I (Poincare complex, normal map, obstruction), but the obstruction computation benefits from odd parity. Whether the first two obstacles are easier in odd dim is unclear. |
| S-rot-II: Equivariant surgery | **Open (newly viable)** | Codim-2 gap satisfied → Costenoble-Waner applies. Need to compute equivariant surgery obstruction for this specific action. |

### Why this is the most promising path

1. **Dimension-parity tension dissolved.** E2 and S both want odd n.
2. **Equivariant surgery becomes available.** The codim-2 gap (which blocks
   Approach II for reflections) is satisfied for rotational involutions.
3. **Two parallel S-branch options.** If equivariant surgery is computable,
   it bypasses the FH(Q) complex entirely.
4. **Single remaining bottleneck: lattice existence.** Everything reduces to
   finding the right arithmetic lattice.

### Key question for lattice existence

An order-2 rotation in SO(2k+1, 1) acts on H^{2k+1} with fixed set H^{2k-1}
(codimension 2). Algebraically, this corresponds to an element conjugate to
diag(-1, -1, 1, ..., 1) in the defining representation. The question: do
arithmetic uniform lattices in SO(2k+1, 1) contain such elements?

Candidate sources:
- Belolipetsky-Lubotzky (math/0406607): any finite group as isometry group,
  but fixed-set codimension not controlled.
- Arithmetic lattice constructions via quadratic forms over number fields:
  an element of order 2 in the lattice corresponds to an involution of the
  quadratic form. Reflections negate one coordinate; rotations negate two.
  The arithmetic existence question is whether the lattice contains such
  a "2-rotation."
- Possibly simpler: products. If Gamma_0 < Isom(H^{2k-1}) is a cocompact
  lattice and we embed H^{2k-1} -> H^{2k+1} totally geodesically, can we
  find a lattice in Isom(H^{2k+1}) that contains both Gamma_0 (stabilizing
  H^{2k-1}) and a rotation exchanging the two normal directions?


## Diagram H2: Reflection Route, Approach I (CURRENT — in trouble)

**Key idea:** Even n, reflection, FH(Q) complex, Wall surgery.

```
[E2-refl] Reflection lattice in Isom(H^{2k}), k >= 3  -- DISCHARGED
    |
    v
[Fowler] Gamma in FH(Q)  -- DISCHARGED
    |
    v
[P2] Y is rational Poincare complex?  -- OPEN (obstacle 1)
    |
    v
[P4] Degree-1 normal map f: M_0 -> Y?  -- OPEN (obstacle 2)
    |
    v
[Obstruction] sigma(f) in L_{2k}(Z[Gamma]) tensor Q = 0?  -- OPEN (obstacle 3)
    |
    | Even L-theory: unfavorable parity
    v
[Closed manifold M with pi_1 = Gamma]
```

### Assessment: THREE successive open obstacles, with structural headwinds
on the third. The dimension-parity tension is inescapable for reflections.
Not recommended for further investment unless obstacles 1-2 are resolved
by a reference.


## Diagram H3: Reflection Route, Approach II (BLOCKED)

**Key idea:** Even n, reflection, equivariant surgery to eliminate fixed set.

```
[E2-refl] -- DISCHARGED
    |
    v
[(M, tau, F)] Closed manifold M^{2k} with involution tau, fixed set F^{2k-1}
    |
    v
[Gap hypothesis check] Codim-1 fixed set: FAILS codim-2 gap  -- BLOCKED
    |
    X (Costenoble-Waner framework does not apply)
```

### Assessment: DEAD for codimension-1 fixed sets. The standard equivariant
surgery theories (Costenoble-Waner, and likely Dovermann-Schultz) require
codimension >= 2 gaps. This approach cannot proceed without developing
entirely new codimension-1 equivariant surgery theory, which is out of scope.


## Diagram H4: Reflection Route, Approach III — Orbifold Resolution (UNEXPLORED)

**Key idea:** Even n, reflection, resolve the mirror singularity in H^n/Gamma.

```
[E2-refl] -- DISCHARGED
    |
    v
[H^n/Gamma] Compact orbifold with mirror singularity along image of F
    |
    v
[Resolution] Resolve mirror singularity to closed manifold?  -- OPEN
    |
    | Must preserve: pi_1 = Gamma AND rational acyclicity of univ. cover
    | Standard approaches (cut-and-double) change pi_1
    v
[Closed manifold M with pi_1 = Gamma]
```

### Assessment: No known construction technique. Standard orbifold resolution
methods (cutting, doubling, blowing up) typically alter pi_1. Would need a
fundamentally new geometric construction. Not recommended unless a specific
resolution method is identified.


## Diagram H5: Odd-dim Reflection Route (BLOCKED by Gauss-Bonnet)

**Key idea:** Odd n, reflection. Would give favorable L-theory parity.

```
[E2-refl-odd] Reflection lattice in Isom(H^{2k+1})?
    |
    | Fixed set: H^{2k} (even-dimensional closed hyperbolic manifold)
    | Gauss-Bonnet: chi(H^{2k}/pi_F) = (-1)^k * c_k * vol != 0
    v
[Fowler check] chi(C) != 0  -- FAILS
    |
    X (Fowler criterion not satisfied)
```

### Assessment: DEAD. Even-dimensional closed hyperbolic manifolds always
have nonzero Euler characteristic (Gauss-Bonnet). Reflections in odd
dimensions produce even-dimensional hyperbolic fixed sets, so chi != 0
and Fowler fails. This is WHY the original construction needed even n.


## Summary: Recommended Priority Order

1. **H1 (Rotation route):** Most promising. Dissolves all known tensions.
   Single bottleneck is lattice existence (arithmetic question). Investigate
   whether order-2 rotations exist in arithmetic lattices of Isom(H^{2k+1}).

2. **H2 (Reflection + Wall surgery):** Current approach, documented in
   detail. Three open obstacles with structural headwinds. Only pursue if
   a reference resolves obstacles 1-2 simultaneously.

3. **H4 (Orbifold resolution):** Unexplored, needs a specific technique.
   Low priority unless a resolution-with-pi_1-control method is found.

4. **H3 (Reflection + equivariant surgery):** Blocked by gap hypothesis.

5. **H5 (Odd-dim reflection):** Blocked by Gauss-Bonnet.


## Key References for H1 (Rotation Route)

- Costenoble-Waner, arXiv:1705.10909 (equivariant surgery with gap hyp)
- Belolipetsky-Lubotzky, arXiv:math/0406607 (finite groups as isometry
  groups of hyperbolic manifolds)
- Douba-Vargas Pallete, arXiv:2506.23994 (reflections in congruence
  manifolds — need rotational analogue)
- Fowler, arXiv:1204.4667 (FH(Q) criterion)
- Bartels-Farrell-Luck, arXiv:1101.0469 (FJ for cocompact lattices)
