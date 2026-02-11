# Response to Reviewer

Scope: point-by-point response to `REVIEWER.md` (critic pass, 2026-02-11)
Date: 2026-02-11

Convention: **[A]** = accept, **[PA]** = partially accept, **[R]** = rebut/clarify.

---

## Problem 1: Phi^4_3 Measure Equivalence

### R1.1 (Critical): stated equivalence `mu ~ mu_0'` not justified at this generality

**[PA]** The reviewer is right that line 40-43 asserts `mu ~ mu_0'` without
a precise theorem citation. However, the supporting argument is already
present in the text: Section 2 (lines 47-50) derives this from positivity of
`exp(-V)` and the integrability `E_{mu_0}[exp(-V)] < infinity`, which is
the main analytical result of the Barashkov-Gubinelli (2020) construction
(their Theorem 1.1). The equivalence is not a separate imported theorem --
it *is* the content of the construction.

**Proposed fix:** Add explicit citation: "This equivalence follows from the
variational construction of Barashkov-Gubinelli (2020, Theorem 1.1), which
establishes `E_{mu_0}[exp(-V)] < infinity` and hence `mu << mu_0` with
strictly positive density. Since `exp(-V) > 0` a.s., the reverse absolute
continuity `mu_0 << mu` also holds."

### R1.2 (Major): exponential moments "for all `t < infinity`" too strong

**[A]** Agreed. The claim at line 112 that `E_mu[exp(t |int psi :phi^3: dx|)] < infinity`
for all real `t` is stronger than what the log-Sobolev + coercivity argument
directly gives. The proof only needs a *neighborhood* of the required
exponent.

**Proposed fix:** Replace with: "There exists `t_0 > 0` (depending on
`||psi||` and the coupling constant) such that
`E_mu[exp(t |int psi :phi^3: dx|)] < infinity` for `|t| < t_0`.
This suffices for `R in L^1(mu)` since the exponent in the Radon-Nikodym
derivative is bounded by `4 ||psi||_{C^0} |int :phi^3: dx|`, and `t_0`
can be chosen to exceed this coefficient." Cite Barashkov-Gubinelli (2020),
Section 4 (exponential integrability from the Polchinski flow).

### R1.3 (Medium): renormalization-shift formula is schematic

**[PA]** The `log N correction` at line 87 is intentionally schematic -- the
proof's logic does not depend on the exact form of the counterterm shift,
only on the fact that `V(phi - psi) - V(phi)` is well-defined after
renormalization (line 89). The dominant term `4 int psi :phi^3: dx` is
already identified with its regularity checked (lines 90-95).

**Proposed fix:** Add a sentence: "The precise counterterm shift is
determined by the regularization scheme; see Hairer (2014, Section 9) or
Gubinelli-Imkeller-Perkowski (2015, Proposition 6.3) for the explicit
formula. For the present argument, only the finiteness of the renormalized
difference matters." This keeps the section conceptual while anchoring
the claim.

---

## Problem 2: Universal Test Vector for Rankin-Selberg Integrals

### R2.1 (Critical): universality for fixed `W` is unproved

**[A]** This is the central gap. The proof assumes (line 59-66) that for
fixed `W`, the integrals over `V` span the full fractional ideal. The
standard JPSS result (Section 2.7) establishes this when *both* `W` and `V`
vary. For fixed `W`, one needs that the Kirillov restriction `phi_Q` is
"sufficiently generic" -- a claim that is natural but not proved in the text.

**Proposed fix:** This requires a new lemma: "For the new vector `W_0` of
a generic `Pi`, and any `Q in F^x`, the restricted function
`g -> W_0(diag(g,1) u_Q)` generates the full Kirillov model of `Pi|_{GL_n}`
as a `GL_n`-module." This should follow from the Bernstein-Zelevinsky theory
of the Kirillov model (the restriction of a generic representation to the
mirabolic generates a representation containing all Kirillov functions), but
the argument needs to be written. Alternatively, cite Jacquet (2009,
Proposition 2.3) if it covers this case.

### R2.2 (Critical): nonzero `R(u_Q)W` does not automatically imply nonzero restriction `phi_Q`

**[PA]** The reviewer's concern is valid in principle: restriction from the
ambient Whittaker space to `diag(GL_n, 1)` can kill vectors. However,
lines 79-81 argue this specifically: `R(u_Q)` is right translation by a
unipotent element, which preserves the Whittaker model. The Kirillov model
of `Pi` (restriction to the mirabolic `P_{n+1}`) is *faithful* for generic
representations -- a nonzero Whittaker function restricts to a nonzero
Kirillov function. This is standard (Bernstein-Zelevinsky 1976, Section 5)
but should be cited explicitly.

**Proposed fix:** Add citation to BZ 1976 or Cogdell's survey (2004) for
faithfulness of the Kirillov restriction for generic representations.
Explicitly state: "For generic `Pi`, the map `W -> W|_{P_{n+1}}` from the
Whittaker model to the Kirillov model is injective (BZ, Theorem 5.21),
so `R(u_Q)W_0 != 0` implies `phi_Q != 0`."

### R2.3 (Major): jump from ideal language to single-`V` monomial output

**[A]** The argument at line 113-114 leaps from "the integrals span the
fractional ideal" to "there exists `V` giving a monomial `c q_F^{-ks}`."

**Proposed fix:** Make the linear algebra explicit: "The fractional ideal
`I = L(s, Pi x pi) * C[q_F^s, q_F^{-s}]` is a free rank-1 module over
`C[q_F^s, q_F^{-s}]` generated by `L(s, Pi x pi)`. Since
`L(s, Pi x pi)^{-1}` is a polynomial in `q_F^{-s}`, the monomial
`c q_F^{-ks}` lies in `I` for appropriate `k`. By the spanning property, some
`V` realizes this element."

### R2.4 (Medium): conductor matching section is heuristic

**[A]** Section 6 (lines 118-132) uses narrative language ("scale overlap").
This should be replaced with precise support statements from newvector theory.

**Proposed fix:** Replace with: "By the Casselman-Shalika formula, the
support of the new vector `W_0` of `Pi` is contained in
`N_{n+1} diag(o^x, ..., o^x, 1) K_1(p^{c(Pi)})`. Right translation by
`u_Q` shifts this support by `Q` in the `(n, n+1)` coordinate, ensuring
overlap with the support of `K_1(p^{c(pi)})`-fixed vectors in `W(pi, psi)`.
See Jacquet-Piatetski-Shapiro-Shalika (1981), Section 5."

---

## Problem 3: Markov Chain with Interpolation ASEP Stationary Distribution

### R3.1 (Major): star/non-star normalization bridge asserted, not proved

**[PA]** The bridge argument (Section 5, lines 107-131) is more than pure
assertion -- it gives a conditional proof: *if* the starred and unstarred
families differ by a global constant `alpha` independent of `eta`, the ratio
cancels. The remaining claim (that `alpha` is `eta`-independent) appeals to
the uniform leading-term convention under Hecke exchange relations
(lines 126-128). We acknowledge this step is compressed.

**Proposed fix:** Two options:
(a) Add a direct computation for small `n` (e.g., `n = 2, 3`) showing
    `F*_eta / F_eta` is constant across `eta in S_n(lambda)`, then state the
    general result follows from the same exchange relation argument.
(b) Cite the explicit comparison in Corteel-Mandelshtam-Williams (Section 3),
    who define both normalizations and note their equivalence at `q = 1`.
Option (b) is preferred if the citation supports it cleanly.

### R3.2 (Major): irreducibility claim too compressed

**[A]** The claim at lines 169-172 that "any configuration can reach any
other through push cascades" is correct but needs a constructive argument.

**Proposed fix:** Add: "Irreducibility follows from a sorting argument.
Given any two configurations `eta, eta'` in `S_n(lambda)`, `eta` can reach
`eta'` through adjacent transpositions. Each adjacent transposition `(j, j+1)`
occurs when the clock at site `j` rings and the `t`-geometric choice selects
the particle at site `j+1`. Since `t in [0,1)` and all parts are distinct,
each selection has probability `t^k / [m]_t > 0` for some `k < m`. Hence
every adjacent transposition has positive rate, giving a positive-probability
path between any two states." This is a standard argument for multispecies
exclusion processes (see Ayyer-Martin-Williams Section 3.2 for the analogous
claim in their setting).

### R3.3 (Medium): positivity needs direct citation

**[A]** Line 96-97 asserts positivity of `F_eta(x; 1, t)` narratively.

**Proposed fix:** Cite AMW Theorem 1.1 directly: "Positivity
`F_eta(x; 1, t) > 0` for `x_i > 0`, `0 <= t < 1` is stated as part of
Theorem 1.1 (Ayyer-Martin-Williams, arXiv:2403.10485), which establishes that
`pi(eta) = F_eta / P_lambda` is a probability distribution (hence each
summand is nonneg) with explicit combinatorial formula showing strict
positivity."

---

## Problem 4: Root Separation Under Finite Free Convolution

### R4.1 (Critical): headline answer overstates what is proved

**[PA]** The reviewer asks us to change the verdict. We note that the
writeup *already* qualifies the answer significantly:
- Line 28-29: "numerically verified, proof incomplete"
- Line 33-34: "The original proof via 'concavity...' contains errors"
- Section 5a header: "What the proof requires (GAP)"
- Lines 216-222: "What remains open: the analytic proof"

However, the one-word answer "Yes" on line 28 is admittedly misleading
when read in isolation.

**Proposed fix:** Change "**Yes** (numerically verified, proof incomplete)"
to "**Conjecturally yes, with strong numerical evidence.** The inequality
holds in all 8000 random trials tested (n = 2-5) with no violations.
An analytic proof remains open." This better reflects the status at a glance.

### R4.2 (Major): derivative-at-root formulas not proof-rigorous

**[A]** Lines 47-69 mix regularized and singular notation around root
evaluations.

**Proposed fix:** Rewrite Section 1 using only the algebraic identity
`p'(lambda_i) = prod_{j != i} (lambda_i - lambda_j)` (which is exact, no
limiting procedure needed), and define `Phi_n` directly as
`sum_i [sum_{j != i} 1/(lambda_i - lambda_j)]^2` without going through
log-derivative notation. The connection to `F''_A` can be kept as a remark,
not a derivation step.

### R4.3 (Medium): degree-2 convolution display incomplete

**[A]** The ellipsis at line 186 is not acceptable.

**Proposed fix:** Either provide the complete `n = 2` formula (which is
simple: `(p ⊞_2 q)(x) = x^2 - (a_1 + b_1)x + (a_2 + b_2 + a_1 b_1 / 2)`)
or remove the incomplete expansion and keep only the worked symmetric
specialization at lines 188-199, which is already complete and correct.

---

## Problem 5: O-Slice Connectivity via Geometric Fixed Points

### R5.1 (Major): reverse implication relies on unproved "verbatim localization"

**[A]** Lines 113-115 claim the Hill-Yarnall reverse direction "applies
verbatim" after restriction to `F_O`-local spectra. This needs justification.

**Proposed fix:** Add a reduction lemma: "The Hill-Yarnall reverse argument
(Section 2 of arXiv:1703.10526) constructs Postnikov sections using the
slice cells as building blocks and the geometric fixed-point detection
criterion to control connectivity at each stage. In the `F_O`-local setting:
(1) the building blocks are the `F_O`-restricted slice cells (same
construction, restricted indexing), (2) the detection criterion is the same
`Phi^H` test for `H in F_O` (with `Phi^K = 0` for `K notin F_O` by the
locality assumption), and (3) the Postnikov tower converges because the
localizing subcategory is generated by the same cells. The only input from
the ambient theory is the isotropy separation sequence (HHR Section 4),
which holds in any localizing subcategory of `SH^G`."

### R5.2 (Major): subgroup-family reduction may lose indexing-system data

**[PA]** The proof *already* acknowledges this limitation explicitly. The
"Scope / Caveat" section (lines 134-147) states: "Our `F_O` extracts the
subgroup-level data; it does not use the finer `K -> H` admissibility
structure for `K != e`. For problems where that finer data matters, a
fuller treatment would be needed."

We agree this caveat should be more prominent.

**Proposed fix:** Move the caveat into the theorem statement itself: restate
as "Characterization Theorem (subgroup-family level)" and add a remark that
the full indexing-system formulation is an open extension. This makes the
scope limitation visible without misrepresenting the result.

### R5.3 (Medium): filtration properties not checked

**[A]** Line 43 introduces `tau_{>= n}^O` without verifying monotonicity
and exhaustiveness.

**Proposed fix:** Add: "Monotonicity (`tau_{>= n+1}^O subseteq tau_{>= n}^O`)
is immediate: the generating set for `n+1` is a subset of that for `n`
(if `k|H| >= n+1` then `k|H| >= n`). Exhaustiveness
(`bigcup_n tau_{>= n}^O = SH^G_{F_O-local}`) holds because every object is
built from cells with finite `k|H|`. Compatibility with suspension is
inherited from the ambient regular slice filtration (Hill-Yarnall,
Proposition 1.3)."

---

## Problem 6: Epsilon-Light Subsets of Graphs

### R6.1 (Major): final existential claim depends on unnamed imported theorem

**[R]** The proof already handles this transparently. Section 5 (lines 137-146)
explicitly states: "This writeup does not reprove that theorem; it uses it as
an explicit external dependency." The conclusion in Section 6 (lines 148-161)
is explicitly labeled "conditional." The phrase "explicitly conditional" appears
in the section header.

We agree a precise theorem reference would strengthen the writeup, but the
reviewer's characterization of an "unnamed imported theorem" understates the
degree to which the proof signals this dependency.

**Proposed fix:** Add a specific reference: "The universal existence result
is due to [Spielman-Srivastava (2011), Theorem 1.1] / [Batson-Spielman-Srivastava
(2012)] or the appropriate source. Insert theorem statement and hypotheses
here." (Exact citation to be confirmed -- the result may be the BSS
twice-Ramanujan sparsification theorem or a related graph pruning result.)

### R6.2 (Major): concentration section is setup-only

**[PA]** Correct that Section 4 sets up the martingale framework without
computing explicit bounds for `R_*` and `||W_n||`. This is intentional:
the proof separates what it can establish internally (the setup) from
what requires the external theorem (the universal constant). Computing
`R_*` and `||W_n||` in terms of graph parameters would complete the
concentration argument but is the hard part of the external theorem.

**Proposed fix:** Add a remark: "To obtain a self-contained concentration
bound, one would need `R_* <= C_1 * epsilon` and `||W_n|| <= C_2 * epsilon^2`
for graph-dependent constants `C_1, C_2`. Bounding these requires leverage
score analysis (tau_e bounds) that is the core content of the external theorem
referenced in Section 5."

---

## Problem 7: Uniform Lattice with 2-Torsion

### R7.1 (Critical): "rational PD group" loose with torsion

**[A]** Line 52-59 states rational Poincare duality via orbifold cohomology
but uses the phrase "rational PD group" without the careful orbifold/Bredon
framing needed when `Gamma` has torsion.

**Proposed fix:** Replace with: "Since `Gamma` acts properly and cocompactly
on the contractible space `X`, the Bredon cohomology
`H^*_{Gamma}(X; R_Q)` (with the rational constant coefficient system)
satisfies Poincare duality by the orbifold PD theorem (see Brown, *Cohomology
of Groups*, Chapter VIII, or Luck, *Transformation Groups and Algebraic
K-Theory*, Section 6.6). Concretely, `H^*(Gamma; Q) = H^*(X/Gamma; Q)`
satisfies `H^k = H^{d-k}` where `d = dim(X)`."

### R7.2 (Critical): normal map existence asserted without hypotheses

**[A]** Line 70-74 says "standard surgery theory" gives the degree-1 normal
map, but the standard machinery (Wall, Chapter 9) requires finitely presented
`Gamma`, finite `d`-dimensional Poincare complex, and `d >= 5`.

**Proposed fix:** State explicitly: "Gamma is finitely presented (lattice in
a Lie group), and `B Gamma` has the rational homotopy type of the finite
complex `X / Gamma` (orbifold with finitely many cells). For `d = dim(X) >= 5`
(guaranteed when `G` has real rank >= 3, or using `SO(2k+1,1)` with `k >= 3`),
the surgery exact sequence applies. The degree-1 normal map `f: M_0 -> B Gamma`
is constructed from a Thom transversality argument on a finite Poincare
complex representing `B Gamma` rationally (see Wall, *Surgery on Compact
Manifolds*, Section 9.4, or Luck-Reich, *The Baum-Connes and Farrell-Jones
Conjectures*, Section 2)."

### R7.3 (Critical): obstruction vanishing "by parity" unsupported

**[A]** Lines 99-101 and 116 assert vanishing "by parity" without
computation. The reviewer is right that this is too coarse.

**Proposed fix:** Expand: "For `d` odd (specifically `d = 2k+1` with the
`SO(2k+1,1)` family), the rational surgery obstruction lies in
`L_{2k+1}(Z[Gamma]) tensor Q`. By the Farrell-Jones isomorphism and
Ranicki's algebraic surgery sequence, this reduces to
`H_{2k+1}^{Gamma}(X; L tensor Q)`. The Atiyah-Hirzebruch spectral sequence
collapses rationally because `L_*(Z) tensor Q = Q` in degrees `0 mod 4`
and `0` otherwise. Since `2k+1` is odd, the relevant `L`-groups at each
cell level are in odd degrees, hence zero. Therefore the rational obstruction
vanishes." Add explicit reference to Ranicki (1992), Proposition 15.11.

If this argument has a gap (e.g., if the equivariant spectral sequence is
more complex), we should weaken to: "conditional on vanishing of the
odd-dimensional rational surgery obstruction, which is expected for
dimensional parity reasons but requires verification for the specific
lattice family."

### R7.4 (Medium): Smith-theory discussion is anti-obstruction only

**[A]** Agreed. Section 7 (lines 120-136) shows that Smith theory does NOT
obstruct the construction, which is necessary but does not contribute
positively to the construction.

**Proposed fix:** Relabel Section 7 as "Remark: absence of Smith-theory
obstruction" and add: "This section addresses a natural objection -- that
torsion elements would force fixed points on a rationally acyclic covering
space -- and explains why it does not apply. It does not contribute to the
constructive argument, which is entirely in Section 5."

---

## Problem 8: Lagrangian Smoothing of Polyhedral Surfaces

### R8.1 (Critical): basis/nondegeneracy claim not justified by submanifold condition alone

**[PA]** Lines 72-73 claim `{e_1, e_2, e_3, e_4}` is a basis, justified by
"the topological submanifold condition." The reviewer correctly notes that
a topological 2-manifold condition alone does not force 4 edge directions to
span `R^4`.

However, the argument is stronger than the reviewer suggests: the 4 edges
are edge vectors of a polyhedral surface in `R^4` with 4 faces meeting at a
vertex. If these 4 edges lay in a 3-dimensional subspace, the 4 faces (each
spanned by two consecutive edges) would all lie in that 3-space, making the
polyhedral complex a surface in `R^3`, not a genuine surface in `R^4`. For a
Lagrangian surface, this is impossible: a Lagrangian plane in `(R^4, omega)`
has dimension 2, and two distinct Lagrangian planes generically span `R^4`.

**Proposed fix:** Make this explicit: "Add a generic position hypothesis:
the 4 edge vectors span `R^4`. This holds whenever the polyhedral complex
is not contained in a hyperplane, which is the generic case for Lagrangian
surfaces in `R^4` (since a single Lagrangian 2-plane already spans a
2-dimensional subspace, and distinct Lagrangian planes generically span
`R^4`). We state this as an explicit nondegeneracy hypothesis rather than
deriving it from the submanifold condition."

### R8.2 (Critical): surgery invoked outside smooth hypotheses

**[A]** Lines 149-151 invoke Polterovich/Lalonde-Sikorav for smooth
transverse Lagrangian intersections, but the input is polyhedral (creased).

**Proposed fix:** Insert a smoothing-to-immersion lemma before the surgery
step: "Before applying Lagrangian surgery, we first smooth each crease.
At a 4-valent vertex with the `V_1 oplus V_2` decomposition, the two
sheets (Section 6) are piecewise-linear Lagrangian surfaces with a single
crease each. Each crease is smoothed by a generating-function interpolation
(as described in Section 7b), producing smooth Lagrangian immersions in a
neighborhood of `v`. The two smoothed sheets then cross transversally at
`v` (the `V_1 oplus V_2` decomposition persists under `C^1`-small
perturbation), placing us in the setting of the Polterovich surgery theorem."

### R8.3 (Major): global patching not established

**[A]** Lines 161-176 summarize global smoothing but don't address
compatibility between local surgeries.

**Proposed fix:** Add a support-control argument: "Each vertex surgery is
performed in a ball `B_v` of radius `r_v` around `v`, chosen small enough
that `B_v` contains no other vertex. Since the vertices are isolated points
of a polyhedral complex, such radii exist. The surgeries at distinct vertices
have disjoint support, so they commute. Edge smoothings (Section 7b) are
performed after all vertex surgeries, on the remaining crease curves. These
creases are compact 1-manifolds (arcs between resolved vertex neighborhoods),
and the generating-function interpolation is local along each arc. A
partition-of-unity argument gives global smoothness. The composed isotopy
`K_t` is Hamiltonian because it is a finite composition of compactly
supported Hamiltonian isotopies (a standard fact in symplectic topology,
see McDuff-Salamon, *Introduction to Symplectic Topology*, Proposition 3.17)."

---

## Problem 9: Polynomial Detection of Rank-1 Scaling

### R9.1 (Major): converse relies on genericity heuristic

**[PA]** The reviewer says the genericity argument is "asserted but not shown
in-file." The proof does give the argument (lines 119-125): the condition
`det(Lambda circ Omega) = 0` for all triples is a polynomial condition on
camera parameters; for specific cameras it fails (explicit construction);
therefore it fails Zariski-generically.

However, the "explicit construction" is not actually exhibited in the text.

**Proposed fix:** Add an explicit witness: "For `n = 5`, choose 5 cameras
`A^(i)` as rows of generic `3 x 4` matrices (e.g., random rational entries).
Compute `Q^(1,2,3,4)` and a non-rank-1 scaling `lambda` (e.g.,
`lambda_{abgd} = 1` for all non-identical tuples). Evaluate the 3x3 minor
for a specific choice of row/column triples and verify it is nonzero. This
exhibits the required polynomial as not identically zero, establishing the
genericity claim." Include the numerical verification or reference a script.

### R9.2 (Major): Hadamard-product rank gives only upper bound

**[PA]** The upper bound `rank(Lambda circ Omega) <= rank(Lambda) * rank(Omega)`
is indeed only one direction. But lines 119-125 address the converse via the
genericity/polynomial-nonvanishing argument, not via Hadamard rank bounds.
The upper bound at line 116-117 is stated as context, not as the proof of
the converse.

**Proposed fix:** Restructure Section 4 to make the logical flow clearer.
Move the Hadamard bound to a remark, and foreground the polynomial
nonvanishing argument: "The converse follows from an algebraic-geometry
argument, not from Hadamard rank bounds. Define `P(A^{(1)}, ..., A^{(n)}) =
det(Lambda circ Omega)` for a specific minor choice. This is a polynomial in
camera entries. We exhibit a point where `P != 0` (the explicit witness above),
so `P` is not the zero polynomial, hence it is nonzero on a Zariski-dense
open set."

### R9.3 (Medium): iff from three matricizations should be spelled out

**[A]** Line 139 states the rank-1 iff without proof.

**Proposed fix:** Add: "If all three matricizations have rank 1, then:
`lambda_{abgd} = f_{ab} g_{gd}` (from (1,2) vs (3,4)) and
`lambda_{abgd} = h_{ag} k_{bd}` (from (1,3) vs (2,4)). From the first,
`lambda_{a_1 b_1 g d} / lambda_{a_2 b_2 g d} = f_{a_1 b_1} / f_{a_2 b_2}`,
independent of `g, d`. From the second,
`lambda_{a b_1 g_1 d} / lambda_{a b_2 g_1 d} = k_{b_1 d} / k_{b_2 d}`,
independent of `a, g_1`. Cross-referencing these separations forces
`f_{ab} = u_a v_b` and `g_{gd} = w_g x_d`, giving rank-1."

---

## Problem 10: RKHS-Constrained Tensor CP via PCG

### R10.1 (Major): preconditioner quality is hypothesis-level

**[PA]** The proof already frames spectral equivalence as a hypothesis, not a
derived result. Lines 131-136 introduce `(1-delta)P <= A_tau <= (1+delta)P`
as an explicit assumption and derive the convergence bound from it. This is
standard practice in PCG literature (state the spectral equivalence condition,
derive iteration count, then analyze the condition separately for specific
applications).

However, the reviewer is right that no sufficient conditions are given for
when `delta` is bounded.

**Proposed fix:** Add a remark: "The spectral equivalence `(1-delta)P <= A_tau
<= (1+delta)P` holds with `delta` bounded away from 1 when the sampling
pattern satisfies a *restricted isometry*-type condition: `||D - (q/N)I||`
is small relative to `lambda`. For uniform random sampling with `q >= C n log n`
(for a universal constant `C`), matrix concentration results (Tropp 2011,
Theorem 1.6) give `delta = O(sqrt(n log n / q))` with high probability.
Under this regime, `kappa = O(1)` and PCG converges in `O(log(1/eps))`
iterations." This connects the hypothesis to concrete sufficient conditions
without over-claiming.

### R10.2 (Medium): asymptotic claim understates setup cost

**[A]** Lines 162-165 simplify to `O(n^3 + t(n^2 r + q r))`, but the `n^3`
Cholesky dominates for large `n`.

**Proposed fix:** Add: "When `n` is large enough that the `O(n^3)` setup
dominates (i.e., `n^3 > t(n^2 r + q r)`, equivalently `n > t r` and
`n^3 > t q r`), the per-ALS-step cost is effectively `O(n^3)`. In this
regime, low-rank kernel approximations (e.g., Nystrom approximation with
rank `p << n`, reducing the kernel factorization to `O(n p^2)`) or iterative
inner solves (conjugate gradient on `K_tau y = z`, cost `O(n^2)` per inner
iteration) can replace the exact Cholesky, reducing the setup to
`O(n p^2 + t(n p r + q r))`. This is a well-known practical optimization
(see Rudi-Calandriello-Rosasco, 2017) and is compatible with the PCG
framework as presented."

---

## Summary of Actions

### Classification of reviewer findings

| Problem | Finding | Severity | Response | Action Required |
|---------|---------|----------|----------|-----------------|
| P1 | 1.1 | Critical | PA | Add BG2020 Thm 1.1 citation |
| P1 | 1.2 | Major | A | Weaken integrability claim |
| P1 | 1.3 | Medium | PA | Add formula reference, keep conceptual |
| P2 | 2.1 | Critical | A | New lemma on Kirillov generation |
| P2 | 2.2 | Critical | PA | Add BZ1976 citation for faithfulness |
| P2 | 2.3 | Major | A | Explicit linear algebra in fractional ideal |
| P2 | 2.4 | Medium | A | Replace with Casselman-Shalika support |
| P3 | 3.1 | Major | PA | Add small-n computation or CMW citation |
| P3 | 3.2 | Major | A | Add sorting/transposition argument |
| P3 | 3.3 | Medium | A | Cite AMW Thm 1.1 for positivity |
| P4 | 4.1 | Critical | PA | Soften headline to "conjecturally yes" |
| P4 | 4.2 | Major | A | Rewrite via algebraic identity only |
| P4 | 4.3 | Medium | A | Complete formula or remove |
| P5 | 5.1 | Major | A | Add reduction lemma |
| P5 | 5.2 | Major | PA | Move caveat into theorem statement |
| P5 | 5.3 | Medium | A | Add monotonicity/exhaustiveness check |
| P6 | 6.1 | Major | R | Already marked conditional; add reference |
| P6 | 6.2 | Major | PA | Add remark on what bounds are needed |
| P7 | 7.1 | Critical | A | Reframe in Bredon/orbifold terms |
| P7 | 7.2 | Critical | A | State explicit hypotheses and citation |
| P7 | 7.3 | Critical | A | Expand parity argument or weaken |
| P7 | 7.4 | Medium | A | Relabel as remark |
| P8 | 8.1 | Critical | PA | Add nondegeneracy hypothesis explicitly |
| P8 | 8.2 | Critical | A | Insert smoothing lemma before surgery |
| P8 | 8.3 | Major | A | Add support-control/partition-of-unity |
| P9 | 9.1 | Major | PA | Add explicit witness computation |
| P9 | 9.2 | Major | PA | Restructure to foreground polynomial argument |
| P9 | 9.3 | Medium | A | Add tensor factor compatibility lemma |
| P10 | 10.1 | Major | PA | Add sufficient conditions (RIP-type) |
| P10 | 10.2 | Medium | A | Add regime caveat and alternatives |

### Priority order for revisions

1. **P2** (2 critical gaps at core of the argument)
2. **P7** (3 critical gaps, weakest proof overall)
3. **P8** (2 critical gaps, but framework is stronger)
4. **P4** (1 critical, but only requires headline softening)
5. **P1** (1 critical, fix is a citation addition)
6. **P5, P3, P9** (major gaps, all fixable with stated additions)
7. **P6, P10** (mostly citation/caveat additions)

This ordering aligns with the reviewer's recommendation to treat P2, P7,
and P8 as highest-priority rewrites.
