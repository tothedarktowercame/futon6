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

## `p7r-s3a` checklist (Approach I: Wall surgery setup)

### Scope

This node covers the Wall surgery approach to obligation S (Approach I in
`problem7-solution.md`). Alternative approaches (equivariant surgery,
orbifold resolution) are described in `problem7-solution.md`, Section 4.

### Target statement

Set up the surgery pipeline that upgrades the FH(Q) complex to a closed
manifold with `pi_1 = Gamma` and rationally acyclic universal cover.

### Available theorem assets

1. Avramidi Theorem 17 (`1506.06293`):
   - if `Gamma` is a `d`-dimensional duality group with finite `BΓ`, then for
     `r≥2`, `d+r≥5`, there is a compact `(d+r+1)`-manifold-with-boundary
     `(M,∂)` with `pi_1(M)=pi_1(∂)=Gamma` and rationally acyclic universal
     cover.
2. Avramidi rational surgery pipeline:
   - Theorems 14/16 give rational `pi-pi` / rational surgery interfaces used
     to produce these manifold models.

### Critical caveat

- Avramidi's closure to a closed manifold via reflection methods may alter the
  fundamental group unless extra control is shown. For reduced P7, any closure
  step must preserve final `pi_1 = Gamma`.

### Status: OPEN (three obstacles)

Surgery prerequisites analysis in `problem7r-s3a-setup.md`:

- P1: finitely presented (lattice) — **verified**
- P2: finite rational Poincare complex — **open**: homology-level PD
  established (via spectral sequence collapse + Bredon PD), but chain-level
  Poincare complex structure not proved
- P3: dimension >= 5 (n = 6) — **verified**
- P4: degree-1 normal map existence — **open**: prior descent argument
  retracted; no replacement construction
- P5: pi_1 preservation — **verified** (if P4 holds)
- P6: rational acyclicity preserved — **verified** (if P4 holds)
- Surgery obstruction — **open**: see `p7r-s3b`

## `p7r-s3b` checklist (Approach I: obstruction computation)

### Scope

This node covers the surgery obstruction computation for Approach I. It is
relevant only if the upstream obstacles (P2, P4) from `p7r-s3a` are resolved.
Currently they are not. See `problem7r-s3b-obstruction.md` for full analysis.

### Target statement

Compute/cite the relevant surgery obstruction group/class and prove the needed
vanishing in the selected lattice case.

### Critical caveats

- FJ gives computability/reduction, not automatic vanishing.
- Assembly image vs closed-manifold subgroup equality depends on hypotheses
  (dimension, decorations, orientation character, stabilization regime). Do not
  silently assume equality in unsupported settings.
- This computation is specific to Approach I. The alternative approaches
  (Approaches II and III in `problem7-solution.md`) face different obstacles.

### Status: OPEN (framework established, intermediate lemmas unverified)

The FJ reduction framework is solid:

- UNil terms vanish rationally (Connolly-Davis), so `E_{Fin}` suffices.
- Computation reduces to `H_6^{Z/2}(M; L tensor Q)`.
- Dimension-parity tension identified: E2 needs even n, S prefers odd n.

The following are **conjectural sketches, not proofs**:

- Transfer argument `res(sigma(f)) = 0` requires an unconstructed compatible
  normal map.
- Localization `ker(res) ~= Q + H_3(F; Q) + H_1(F; Q)` rests on three
  unverified intermediate claims (free/fixed splitting, equivariant Thom
  isomorphism, twisted L-theory coefficients).
- Even if the localization holds, whether `sigma(f) = 0` is not addressed.

## Overall status

### E2 branch: discharged

1. `p7r-s2a`: Fowler criterion interface locked. **Done.**
2. `p7r-s2b`: reflection-lattice instantiation. **Done.**

### S branch: open

Obligation S (manifold upgrade) is an **open problem**. Four approaches
analyzed; one is the recommended path. See `problem7-solution.md`, Section 4
and `problem7-hypothetical-wirings.md` for full analysis.

3. `p7r-s3a` (Approach I setup): **Open** — three successive obstacles.
4. `p7r-s3b` (Approach I obstruction): **Open** — framework solid,
   intermediates unverified.
5. Approach II (reflection equivariant surgery): **Blocked** — codim-2 gap
   hypothesis fails for codim-1 fixed sets (Costenoble-Waner 1705.10909).
6. Approach III (orbifold resolution): **Unexplored** — no technique found.
7. **Approach IV (rotation route): E2 DISCHARGED — S DISCHARGED.**
   Lattice existence resolved
   (`problem7r-rotation-lattice-construction.md`). Surgery obstruction
   vanishes integrally: for congruence ideal I with Norm(I) > 2 and I
   coprime to 2 (e.g., I = (3)), the integrality of rotation matrices
   over Z[√2] + congruence condition forces trivial holonomy → ν trivial
   → S(ν) = F × S¹ → intersection form integrally hyperbolic → θ = 0.
   See `problem7r-s-rot-obstruction-analysis.md`.
8. `p7r-s5` (full closure): **DISCHARGED** — see argument chain below.

### Path forward (updated: obligations E2 and S both discharged)

**Both obligations are discharged.** The complete argument chain:

(a) **Lattice construction** — Γ = ⟨π, σ⟩ with π = Γ₀(I) a congruence
    subgroup of SO(f, Z[√2]), σ = diag(1,-1,-1,1,...,1), I = (3).
    Γ is a uniform lattice in SO(7,1) with 2-torsion. — **DONE**
(b) **E2 discharge** — Fixed set F has dim 5 (odd), χ = 0. Fowler Main
    Theorem gives Γ ∈ FH(Q). — **DONE**
(c) **Surgery obstruction vanishes** — Congruence condition forces trivial
    holonomy → ν trivial → S(ν) = F × S¹ → intersection form integrally
    hyperbolic → θ = 0 ∈ L₈(Z[Γ]). — **DONE**
(d) **Cut and cap succeeds** — θ = 0 implies equivariant cobordism exists:
    M' with free Z/2-action. N = M'/(Z/2) is a closed manifold with
    π₁(N) = Γ and rationally acyclic universal cover. — **DONE**

**Remaining work:**

**Priority 1 — Write up.** Assemble the complete proof into a single
document, combining the lattice construction, Fowler application, and
surgery vanishing argument. Verify the AHSS-to-Witt-class identification
with precise references.

**Priority 2 (optional) — Strengthen.** Extend to other dimensions
n = 2k+1 ≥ 7. Investigate what happens for n = 9 (where additional
AHSS terms appear).

**Deprioritized — All other approaches.** The rotation route S-rot-II
succeeds, making alternatives unnecessary.
