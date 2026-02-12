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
[E2-rot] Lattice with order-2 rotation in Isom(H^{2k+1})  -- DISCHARGED
    |
    | Fixed set: H^{2k-1} (codim 2, odd-dim, chi = 0)
    v
[Fowler] chi(C) = 0 for all fixed components --> Gamma in FH(Q)  -- DISCHARGED
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
    +--[S-rot-II] Equivariant surgery on (M, sigma)  -- RECOMMENDED
          |
          | Costenoble-Waner gap hypothesis: SATISFIED (codim 2 ≥ 2)
          | Dovermann-Schultz gap: NOT satisfied (dim F > dim M / 2)
          |
          | Key finding: 2-primary obstructions vanish rationally
          | Rational obstruction: ker(res) in L_{2k+2} ⊗ Q = H_2(F;Q)
          |
          | FLAT NORMAL BUNDLE ARGUMENT (Step A):
          | ν flat → e(ν)⊗Q = 0 → S(ν) ≃_Q F×S¹
          | → intersection form on H_3(S(ν);Q) is HYPERBOLIC
          | → Witt class = 0 → θ = 0 unconditionally
          v
        [Rational obstruction: VANISHES (flat ν, Chern-Weil)]
          |
          | Integral obstruction: also vanishes
          | For Norm(I) > 2: holonomy ρ trivial → ν trivial
          | → S(ν) = F × S¹ → integrally hyperbolic → θ = 0
          v
        [Surgery succeeds] → M' with free Z/2 action
          |
          v
        [N = M'/(Z/2)] closed manifold, π₁ = Γ, univ. cover Q-acyclic
```

### Node status

| Node | Status | Detail |
|---|---|---|
| E2-rot: lattice existence | **DISCHARGED** | Arithmetic lattice `SO(f, Z[sqrt(2)])` with `f = (1-sqrt(2))x_0^2 + x_1^2 + ... + x_n^2`, `n` odd. The involution `sigma = diag(1,-1,-1,1,...,1)` is an order-2 rotation with codim-2 fixed set. See `problem7r-rotation-lattice-construction.md`. |
| Fowler application | **DISCHARGED** | Fixed set dim `n-2` is odd → `chi = 0`. Fowler Main Theorem gives `Gamma in FH(Q)`. |
| S-rot-I: Wall surgery | **Open** | Same three obstacles as Approach I (Poincare complex, normal map, obstruction), but the obstruction computation benefits from odd parity: ker(res) = Q ⊕ H_1(F;Q) vs Q ⊕ H_3(F;Q) ⊕ H_1(F;Q) for reflections. Fallback option. |
| S-rot-II: Equivariant surgery | **DISCHARGED (θ = 0)** | "Cut and cap" method. **Rational:** flat ν → e(ν)⊗Q = 0 → intersection form on S(ν) hyperbolic → θ⊗Q = 0. **Integral:** congruence condition (Norm(I)>2) forces trivial holonomy → ν trivial → S(ν) = F×S¹ → θ = 0 integrally. Surgery succeeds → M' with free Z/2 action → N = M'/(Z/2) with π₁=Γ. See `problem7r-s-rot-obstruction-analysis.md`. |

### Why this path succeeds (rationally)

1. **Dimension-parity tension dissolved.** E2 and S both want odd n.
2. **Lattice existence resolved.** The arithmetic construction over `Q(sqrt(2))`
   provides the needed lattice. See `problem7r-rotation-lattice-construction.md`.
3. **Equivariant surgery becomes available.** The Costenoble-Waner codim-2 gap
   (which blocks Approach II for reflections) is satisfied for rotations.
4. **S-rot-II bypasses the FH(Q) complex entirely.** No Poincaré complex or
   normal map needed — works directly with (M, σ).
5. **2-primary obstructions vanish rationally.** The Davis-Lück Z/2 exclusion
   (which is about integral/2-primary phenomena) does not apply since P7 only
   needs rational acyclicity. Browder-Livesay, UNil, Arf all vanish over Q.
6. **Flat-normal-bundle argument kills the rational obstruction.** The normal
   bundle ν is flat (totally geodesic embedding), so e(ν) ⊗ Q = 0
   (Chern-Weil). This forces the intersection form on S(ν) to be rationally
   hyperbolic, giving zero Witt class and hence θ = 0 unconditionally.
   **The b₂(F) question is mooted.**
7. **Integral obstruction also vanishes.** For congruence ideal I with
   Norm(I) > 2: the integrality of rotation matrices over Z[√2] + the
   congruence condition force trivial holonomy → ν trivial → e(ν) = 0 →
   S(ν) = F × S¹ → intersection form integrally hyperbolic → θ = 0.

### Lattice existence: RESOLVED

The construction uses a quadratic form `f = (1-sqrt(2))x_0^2 + x_1^2 + ... + x_n^2`
over `Q(sqrt(2))`. The two real embeddings give signatures `(n,1)` and `(n+1,0)`,
so `SO(f, Z[sqrt(2)])` is a cocompact arithmetic lattice in `SO(n,1)` (Borel-
Harish-Chandra; cocompactness by Godement criterion via anisotropy). The element
`sigma = diag(1, -1, -1, 1, ..., 1)` is in `SO(f, Z[sqrt(2)])`, has order 2,
and fixes `H^{n-2}` (codimension 2). A congruence subgroup `pi = Gamma_0(I)`
with `I` coprime to 2 gives a torsion-free normal subgroup with `sigma notin pi`,
yielding the extension `1 -> pi -> Gamma -> Z/2 -> 1`.

Full details: `problem7r-rotation-lattice-construction.md`.


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


## Summary: Status

1. **H1 (Rotation route) — S-rot-II: DISCHARGED.** E2 discharged, S discharged.
   Equivariant surgery obstruction θ = 0 (rational: flat ν + Chern-Weil;
   integral: trivial holonomy from congruence condition). Closed manifold N
   with π₁ = Γ and rationally acyclic universal cover exists.

2. **H1 (Rotation route) — S-rot-I:** Fallback (no longer needed). Same
   three-obstacle structure as H2 with favorable odd L-theory parity.

3. **H2 (Reflection + Wall surgery):** Three open obstacles with structural
   headwinds. Deprioritized.

4. **H4 (Orbifold resolution):** Unexplored, needs a specific technique.

5. **H3 (Reflection + equivariant surgery):** Blocked by gap hypothesis.

6. **H5 (Odd-dim reflection):** Blocked by Gauss-Bonnet.


## Key References for H1 (Rotation Route)

- Costenoble-Waner, arXiv:1705.10909 (equivariant surgery with gap hyp)
- Belolipetsky-Lubotzky, arXiv:math/0406607 (finite groups as isometry
  groups of hyperbolic manifolds)
- Douba-Vargas Pallete, arXiv:2506.23994 (reflections in congruence
  manifolds — need rotational analogue)
- Fowler, arXiv:1204.4667 (FH(Q) criterion)
- Bartels-Farrell-Luck, arXiv:1101.0469 (FJ for cocompact lattices)
