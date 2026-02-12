# Problem 7 Student Dispatch: Verification + Step B Exploration

**Date:** 2026-02-12
**From:** Claude (advisor)
**To:** Codex (student-explorer)
**Pattern:** agent/student-dispatch

---

## 1. Context

Problem 7 asks: if Gamma is a uniform lattice in a real semisimple Lie group
containing an element of order 2, can Gamma be the fundamental group of a
closed manifold whose universal cover is acyclic over Q?

**Answer: conditional yes.** The argument is 90% complete. Here is the chain:

### What's proved

1. **Lattice construction (DONE).** Over k = Q(sqrt(2)), the quadratic form
   f = (1 - sqrt(2))x_0^2 + x_1^2 + ... + x_n^2 gives Gamma_0 = SO(f, Z[sqrt(2)])
   as a cocompact arithmetic lattice in SO(n,1). The element sigma = diag(1,-1,-1,1,...,1)
   is an order-2 rotation in SO(f, Z[sqrt(2)]) with codimension-2 fixed set. For
   n = 2k+1 odd (concretely n = 7), the extension Gamma = <pi, sigma> (pi a
   congruence subgroup) is a uniform lattice with 2-torsion.

2. **E2 obligation (DONE).** The fixed set F has dimension n-2 = 2k-1 (odd),
   so chi(F) = 0. By Fowler's criterion (arXiv:1204.4667), Gamma in FH(Q):
   there exists a finite CW complex with pi_1 = Gamma and rationally acyclic
   universal cover.

3. **S obligation — equivariant surgery setup (DONE).** The codimension-2
   gap hypothesis (Costenoble-Waner, arXiv:1705.10909) is satisfied. The
   Browder-Lopez de Medrano "cut and cap" framework applies: remove a tubular
   neighborhood of F, cap off the boundary S(nu) with a manifold carrying a
   free Z/2 action.

4. **AHSS computation (DONE).** Via Farrell-Jones + AHSS for n = 7:
   ```
   ker(res) tensor Q = H_2(F; Q)   [at AHSS position (4,4)]
   ```
   where F = H^5/C is the fixed-point manifold (arithmetic hyperbolic 5-manifold).

5. **Rational obstruction vanishes (DONE — Step A).** The flat-normal-bundle
   argument establishes theta tensor Q = 0 unconditionally:
   ```
   nu flat (totally geodesic)
     => e(nu) tensor Q = 0                    [Chern-Weil]
     => H*(S(nu); Q) = H*(F; Q) tensor Q[u]/(u^2)    [Gysin splitting]
     => intersection form on H_3(S(nu); Q) is hyperbolic    [cup product]
     => Witt class = 0                        [hyperbolic => Witt-trivial]
     => theta = 0 in H_2(F; Q)               [AHSS localization]
     => rational surgery obstruction vanishes
   ```
   This is **unconditional in b_2(F)** — the earlier question about whether
   b_2(F) = 0 for arithmetic hyperbolic 5-manifolds is mooted.

### What's NOT proved (the remaining gap)

6. **Step B: integral (torsion) obstruction.** Since theta tensor Q = 0, the
   integral obstruction theta in L_8(Z[Gamma]) is torsion. The plan is a
   **finite-cover trick**: pass to a congruence subgroup Gamma' subset Gamma
   where the torsion is killed. Gamma' is still a uniform lattice with 2-torsion
   (it still contains sigma). Problem 7 asks about the existence of such a
   lattice, not about a specific one, so replacing Gamma by Gamma' is allowed.

   **This step has not been formalized.** Another Claude instance is currently
   working on it after commit 8ce9771.

---

## 2. Dead ends with reasons

### DE1: Reflection route for S — BLOCKED
The reflection construction (codim-1 fixed set) has a dimension-parity tension:
E2 requires n even (so chi(F^{n-1}) = 0), but surgery prefers n odd. Moreover,
the codim-2 gap hypothesis (Costenoble-Waner) fails for codim-1 fixed sets.
**The entire equivariant surgery approach is unavailable for reflections.**

### DE2: Wall surgery (S-rot-I) — STRUCTURALLY HARDER
Wall surgery on the Fowler complex has three sequential obstacles: Poincare
complex structure (chain-level, not just homological), degree-1 normal map
(Spivak fibration topological reduction), and the surgery obstruction. For
S-rot-I, ker(res) = Q + H_1(F; Q), which is always at least 1-dimensional
(the Q from the fundamental class of F is inescapable). **S-rot-II is strictly
easier.**

### DE3: Branched double cover quotient
Taking Q = M/(Z/2) as a topological manifold gives pi_1(Q) = pi/(sigma-coinvariants),
which is strictly smaller than Gamma. The torsion element sigma becomes a
meridional loop around the branch locus, contractible in Q. **Wrong pi_1.**

### DE4: The b_2(F) = 0 approach (MOOTED)
The original plan was: if b_2(F) = 0 then ker(res) = 0 and we're done.
Research showed b_2(F) = 0 is not guaranteed by vanishing theorems, and for
deep congruence levels b_2 > 0 is expected (Millson-Raghunathan).
**Mooted by Step A** — the flat-normal-bundle argument shows theta = 0
regardless of b_2(F).

### DE5: Davis-Luck manifold model (Z/2 excluded)
The Davis-Luck theorem (arXiv:2303.15765) proves manifold models exist for
odd-order G = Gamma/pi but explicitly excludes Z/2 due to 2-primary
obstructions (UNil, Browder-Livesay, Arf). These obstructions all vanish
rationally, so the Z/2 exclusion **does not apply** to the rational version
we need for Problem 7. But the Davis-Luck theorem cannot be cited directly.

---

## 3. Tasks

This dispatch has two parts: **verification** (check established claims) and
**exploration** (advance Step B).

### Part I: Verification of the argument chain

Verify each node of the proof independently. For each, you should:
- Re-derive the claim from first principles (not just check notation)
- Identify any hidden assumptions
- Flag any step that requires more justification

**V1: Lattice construction.**
Verify that f = (1-sqrt(2))x_0^2 + x_1^2 + ... + x_n^2 gives a form with
signatures (n,1) and (n+1,0) under the two real embeddings of Q(sqrt(2)).
Verify that sigma = diag(1,-1,-1,1,...,1) preserves f, has det(sigma) = +1,
and has entries in Z[sqrt(2)]. Verify cocompactness via the Godement criterion
(f is anisotropic over k).

**V2: Fowler application.**
Verify that the fixed set of sigma on H^n has dimension n-2 (codimension 2),
and that for n = 2k+1 odd, the fixed-set components are closed odd-dimensional
manifolds with chi = 0. Check that Fowler's Main Theorem (arXiv:1204.4667)
applies: the only nontrivial subgroup of Z/2 is itself, all fixed components
have chi = 0, therefore Gamma in FH(Q).

**V3: AHSS computation.**
Verify the computation of L_8(Z[Gamma]) tensor Q via the AHSS:
- Farrell-Jones gives L_*(Z[Gamma]) tensor Q = H_*^{Or(Gamma)}(E_{Fin}Gamma; L tensor Q)
- UNil vanishes rationally (Connolly-Davis)
- The coefficient system M_q decomposes into augmentation + sign factors
- Both factors have 4-periodicity (trivial orientation twist from codim-2)
- For n = 7 at total degree 8: sign factor at (4,4) gives H_2(F; Q)
- Therefore ker(res) tensor Q = H_2(F; Q)

Check the critical claim: **no w-twisted terms appear** because the normal
representation has det(-Id on R^2) = +1, unlike the reflection case where
det = -1 introduces twisted L-theory.

**V4: Flat-normal-bundle argument (Step A).**
This is the key new argument (commit 8ce9771). Verify each step:
- F is totally geodesic in M (from the arithmetic construction) => nu is flat
- Chern-Weil: flat connection => all real characteristic classes vanish =>
  e(nu) tensor Q = 0
- Gysin sequence: e = 0 => H*(S(nu); Q) = H*(F; Q) tensor Q[u]/(u^2) with
  u^2 = 0
- Cup product computation: base x base = 0 (dim F = 5 < 6), fiber x fiber = 0
  (u^2 = 0), base x fiber = Poincare duality pairing on F
- Therefore intersection form on H_3(S(nu); Q) is block off-diagonal
  (hyperbolic)
- Hyperbolic form has zero Witt class
- Surgery obstruction theta at AHSS position (4,4) = 0

**V5: Davis-Luck exclusion non-applicability.**
Verify that the three 2-primary obstructions (UNil, Browder-Livesay, Arf) all
vanish after tensoring with Q. Specifically:
- UNil_*(Z; Z, Z) tensor Q = 0 (Connolly-Davis)
- For n = 7: Browder-Livesay invariant in Z/2, and Z/2 tensor Q = 0
- Arf invariant is 2-torsion, hence vanishes rationally

### Part II: Exploration of Step B (integral obstruction / finite-cover trick)

The rational obstruction vanishes (Step A). The integral obstruction theta is
torsion. Step B must show that a finite cover kills this torsion.

**E1: Formalize the finite-cover trick.**
The argument sketch is:
- theta in L_8(Z[Gamma]) is torsion; let ord(theta) = d
- By the congruence subgroup property for arithmetic groups, Gamma has many
  normal finite-index subgroups
- Choose Gamma' normal in Gamma with [Gamma : Gamma'] divisible by d
- Gamma' still contains sigma (automatic if [Gamma : Gamma'] is odd,
  since sigma has order 2)
- The restriction of theta to Gamma' is killed by the transfer-restriction
  relation: res circ tr = [Gamma : Gamma'] * id, so d | [Gamma : Gamma']
  implies the restriction has zero obstruction

Investigate: is this argument complete? What exactly is "restriction of the
surgery problem to Gamma'"? The replacement is: (M, sigma) -> (M', sigma')
where M' -> M is the covering corresponding to pi' = Gamma' cap pi, and sigma'
is the lifted involution. Does the equivariant surgery obstruction transform
correctly under this covering?

**E2: The transfer-restriction issue.**
The transfer map tr: L_*(Z[pi']) -> L_*(Z[pi]) and restriction res satisfy
res circ tr = sum_{g in pi/pi'} g_*. But the surgery obstruction lives in
L_*(Z[Gamma]), not L_*(Z[pi]). How does the transfer work for the group
extension 1 -> pi -> Gamma -> Z/2 -> 1?

Specifically: if theta in L_8(Z[Gamma]) has order d, and Gamma' < Gamma has
index m with d | m, does the transfer-restriction relation
res_{Gamma'}^{Gamma} circ tr_{Gamma'}^{Gamma} = m * id hold in this L-group?
The answer should be yes (this is standard for decorated L-groups), but verify.

**E3: Congruence subgroup property (CSP).**
The CSP for SO(f) over Z[sqrt(2)] — verify that SO(f, O_k) has the CSP, so
every finite-index normal subgroup contains a principal congruence subgroup.
This is needed to guarantee existence of Gamma' with the right index. The
CSP for arithmetic lattices in SO(n,1) with n >= 2 follows from
Raghunathan-Venkataramana (1989) and Serre's conjecture.

**E4: sigma in Gamma' — the parity subtlety.**
If [Gamma : Gamma'] = m is odd, then sigma (order 2) cannot be in the kernel
of Gamma -> Gamma/Gamma' (which has odd order). So sigma in Gamma'. But if
m is even, sigma might map nontrivially. Investigate: can we always choose
Gamma' with [Gamma : Gamma'] odd and divisible by d? This requires d to be
odd. If d is a power of 2, we need m even and sigma in Gamma', which requires
more care. What are the possible orders of torsion in L_8(Z[Gamma])?

**E5: Alternative — Avramidi's rational surgery.**
Avramidi (arXiv:1506.06293) develops a rational surgery theory that works
directly with rational Poincare duality complexes, bypassing integral
obstructions entirely. Does this framework apply to the equivariant setting?
If so, Step B is unnecessary — the rational vanishing from Step A suffices.
This would be the strongest conclusion but requires verifying Avramidi's
framework handles equivariant actions.

---

## 4. Report format

```markdown
## Part I: Verification

### V[1-5]: [claim name]

**Verdict:** CONFIRMED / CONFIRMED WITH CAVEAT / GAP FOUND

**Re-derivation:**
[Your independent computation/argument, not just restating the claim]

**Hidden assumptions found (if any):**
[Anything the original argument takes for granted]

**Confidence:** high / medium / low

## Part II: Exploration

### E[1-5]: [direction name]

**What was tried:**
[1-3 sentences]

**What happened:**
[Where the argument stands]

**Exact gap (if any):**
[The specific step that needs more work]

**Partial results:**
[Any progress, even if incomplete]

**Surprises:**
[Unexpected findings]

**Verdict:** CLOSES STEP B / NARROWS TO [specific claim] / NEEDS [what]
```

---

## 5. Success criteria

**Full closure of Step B** (any of these suffices):
- Rigorous proof of the finite-cover trick: explicit construction of Gamma'
  with torsion obstruction killed.
- Proof that the integral obstruction theta is itself zero (without needing
  a finite cover).
- Application of Avramidi's rational surgery to the equivariant setting,
  making Step B unnecessary.

**Partial progress** (still valuable):
- Identification of the exact group-theoretic condition on Gamma' needed.
- Bound on the order d of the torsion obstruction (e.g., d divides some
  explicit number depending on n and the lattice).
- Verification that transfer-restriction works correctly in this L-group context.

**Verification value:**
- Confirmation of all 5 claim-nodes strengthens the paper.
- Finding a gap in any V-node is extremely valuable — better to find it now.

---

## 6. Another Claude is working on Step B

Commit 8ce9771 added Step A (flat-normal-bundle argument). Another Claude
instance is currently working on the remaining hole after that commit. Your
verification tasks (Part I) are independent of that work and can proceed in
parallel. For the exploration tasks (Part II), check the latest state of
`problem7-solution.md` and `problem7r-s-rot-obstruction-analysis.md` before
starting — there may be new commits.

---

## Files to read

| File | What it contains |
|------|-----------------|
| `data/first-proof/problem7-solution.md` | Full solution doc — lattice construction, E2, S approaches, theorem |
| `data/first-proof/problem7r-s-rot-obstruction-analysis.md` | AHSS computation, flat-normal-bundle argument (Step A), all references |
| `data/first-proof/problem7r-rotation-lattice-construction.md` | Detailed lattice construction (f, sigma, congruence subgroup) |
| `data/first-proof/problem7-hypothetical-wirings.md` | Wiring diagrams for all 5 approaches |
| `data/first-proof/problem7-writeup.md` | Compact (< 5 page) writeup |
| `data/first-proof/p6-p7-process-patterns.md` | Process patterns from the proof journey |
