# Reviewer Notes (Critic Pass)

Scope: proofs in `data/first-proof/problem{1,2,3,4,5,6,7,8,9,10}-solution.md`  
Mode: Proofs-and-refutations (adversarial validity check)  
Date: 2026-02-11

## Summary Verdict

- `P1`: plausible, but one theorem-level citation gap and one overstrong integrability claim.
- `P2`: not proof-grade yet; universal-test-vector step is still unproved.
- `P3`: existence claim is plausible, but notation-bridge and irreducibility arguments are under-justified.
- `P4`: currently evidence-backed conjecture, not a proof.
- `P5`: improved framing, but reverse implication is still largely asserted.
- `P6`: algebra cleanup is good, but existential result remains explicitly conditional.
- `P7`: weakest; deep surgery/FJ steps are asserted, not established.
- `P8`: central geometric-to-surgery bridge is not yet rigorous.
- `P9`: strong structure, but converse genericity step is still heuristic.
- `P10`: strong algorithmic draft, but convergence speed is conditional on unproved spectral assumptions.

## Problem 1 Findings

### 1. Critical: stated equivalence `mu ~ mu0'` is not justified at this generality
- Ref: `data/first-proof/problem1-solution.md:40`
- Issue: the text asserts full equivalence with a Gaussian reference measure. In 3D `Phi^4` constructions, what is immediate is often absolute continuity wrt a finite-dimensional regularization, not a direct global equivalence statement on the final singular measure without precise theorem citation/hypotheses.
- Repair:
  - Cite exact theorem and hypotheses for quasi-invariance under Cameron-Martin shifts in the final renormalized measure.
  - If unavailable, weaken to “expected/known in the constructed regime” and mark as conditional.

### 2. Major: exponential moments “for all `t < infinity`” is likely too strong
- Ref: `data/first-proof/problem1-solution.md:112`
- Issue: global mgf finiteness for all real `t` for cubic observables is stronger than what many concentration/log-Sobolev arguments directly provide.
- Repair:
  - Replace by the exact integrability bound needed for RN derivative (a neighborhood around the required coefficient), with theorem citation.
  - Avoid all-`t` unless explicitly proven in cited source.

### 3. Medium: renormalization-shift formula is schematic
- Ref: `data/first-proof/problem1-solution.md:83`
- Issue: “log N correction” and shifted counterterm are stated informally.
- Repair:
  - Either give explicit regularization-level formula with limit statement, or move to a theorem citation and keep this section conceptual.

## Problem 2 Findings

### 1. Critical: universality for fixed `W` is unproved
- Ref: `data/first-proof/problem2-solution.md:59`, `data/first-proof/problem2-solution.md:65`, `data/first-proof/problem2-solution.md:94`
- Issue: argument assumes that for fixed `W`, varying `V` spans full JPSS zeta ideal. This is the core claim and not established in text.
- Repair:
  - Add a precise theorem for “essential/new vector gives full zeta ideal after `u_Q` twist” (or reformulate as conjectural if no citation).

### 2. Critical: nonzero `R(u_Q)W` does not automatically imply nonzero restriction `phi_Q`
- Ref: `data/first-proof/problem2-solution.md:79`, `data/first-proof/problem2-solution.md:81`
- Issue: restriction to embedded `diag(g,1)` can vanish even when ambient Whittaker vector is nonzero.
- Repair:
  - Prove restriction nonvanishing with Kirillov/newform support argument, or cite exact proposition.

### 3. Major: jump from ideal language to single-`V` monomial output is not justified
- Ref: `data/first-proof/problem2-solution.md:113`
- Issue: existence of one `V` realizing `c q_F^{-ks}` needs concrete linear-algebra argument in zeta integral space for fixed `W`.
- Repair:
  - Add explicit basis/decomposition argument or theorem.

### 4. Medium: conductor matching section is heuristic
- Ref: `data/first-proof/problem2-solution.md:118`
- Issue: narrative (“scale overlap”) is plausible but not proof.
- Repair:
  - Replace with precise support/newvector invariance statements.

## Problem 3 Findings

### 1. Major: star/non-star normalization bridge is asserted, not proved
- Ref: `data/first-proof/problem3-solution.md:107`, `data/first-proof/problem3-solution.md:123`
- Issue: the argument needs a precise citation that the starred interpolation normalization differs by a global factor independent of state `eta` in this `q=1` specialization. As written, this is plausible but unsupported.
- Repair:
  - Cite an explicit theorem/proposition identifying the two conventions in this regime.
  - If only partial equivalence is known, state it conditionally and limit the final claim.

### 2. Major: irreducibility claim is too compressed
- Ref: `data/first-proof/problem3-solution.md:166`, `data/first-proof/problem3-solution.md:170`
- Issue: “any configuration can reach any other through push cascades” is central for uniqueness, but no constructive reachability proof is provided.
- Repair:
  - Add a short constructive argument (adjacent transpositions or canonical sorting path with positive probability at each step), or cite a known irreducibility proposition for this exact inhomogeneous chain.

### 3. Medium: positivity of stationary weights needs direct citation in stated parameter range
- Ref: `data/first-proof/problem3-solution.md:96`
- Issue: positivity is likely true here, but this sentence should point to a precise result in AMW rather than rely on narrative.
- Repair:
  - Add direct citation for positivity/nonnegativity of `F_eta(x;1,t)` when `x_i>0`, `0<=t<1`.

## Problem 4 Findings

### 1. Critical: headline answer overstates what is proved
- Ref: `data/first-proof/problem4-solution.md:28`, `data/first-proof/problem4-solution.md:34`, `data/first-proof/problem4-solution.md:203`
- Issue: the file says “Yes” while also stating the analytic proof is incomplete. Numerics are valuable evidence, but they do not settle a universal statement.
- Repair:
  - Change verdict to “open / conjecturally true with strong numerical evidence.”
  - Keep numerical section as support, not proof.

### 2. Major: derivative-at-root formulas are not proof-rigorous as written
- Ref: `data/first-proof/problem4-solution.md:47`, `data/first-proof/problem4-solution.md:121`
- Issue: identities involving regularized `F_A'` / `F_A''` at roots are presented heuristically around singular points; this needs precise limiting identities to be valid.
- Repair:
  - Replace with explicit algebraic identities in terms of root differences only, avoiding singular log-derivative notation at roots.

### 3. Medium: degree-2 convolution display is incomplete
- Ref: `data/first-proof/problem4-solution.md:186`
- Issue: the formula includes ellipsis (`...`) in a key derivation step, which is not acceptable in a proof document.
- Repair:
  - Provide full coefficient expression or remove the incomplete line and keep only the fully worked symmetric specialization.

## Problem 5 Findings

### 1. Major: reverse implication relies on an unproved “verbatim localization” step
- Ref: `data/first-proof/problem5-solution.md:109`, `data/first-proof/problem5-solution.md:114`
- Issue: the proof says Hill-Yarnall reverse direction applies verbatim in the `F_O`-local subcategory; that is plausible but nontrivial and currently asserted, not demonstrated.
- Repair:
  - Add a formal reduction lemma showing the localized category and restricted generators satisfy the same detection hypotheses.

### 2. Major: subgroup-family reduction may lose essential indexing-system data
- Ref: `data/first-proof/problem5-solution.md:136`, `data/first-proof/problem5-solution.md:139`
- Issue: replacing full admissible `H`-set data by `F_O={H : e->H admissible}` is explicitly coarser. The theorem proven is therefore for a reduced model, not necessarily the full transfer-system notion the problem may intend.
- Repair:
  - Either restate the problem as subgroup-family-level only, or add treatment of general admissible `K->H` data.

### 3. Medium: filtration properties are not checked explicitly
- Ref: `data/first-proof/problem5-solution.md:43`
- Issue: calling this a “slice filtration” normally requires explicit monotonicity/exhaustiveness checks (`tau_{>=n+1}^O subseteq tau_{>=n}^O`, compatibility under suspension conventions).
- Repair:
  - Add short verification or cite a framework result that the restricted generator scheme defines a genuine filtration.

## Problem 6 Findings

### 1. Major: final existential claim depends on an unnamed imported theorem
- Ref: `data/first-proof/problem6-solution.md:139`, `data/first-proof/problem6-solution.md:145`, `data/first-proof/problem6-solution.md:159`
- Issue: the key universal lower bound is external, but no precise theorem identifier is provided.
- Repair:
  - Name and cite the exact theorem (authors, statement, hypotheses) or rephrase final answer as strictly conditional conjectural transfer.

### 2. Major: concentration section is setup-only, not a complete probabilistic bound
- Ref: `data/first-proof/problem6-solution.md:132`, `data/first-proof/problem6-solution.md:133`
- Issue: the text correctly sets up matrix Freedman/Bernstein, but does not provide concrete bounds for `R_*` and `||W_n||`; without those, no quantitative existence follows from this section alone.
- Repair:
  - Supply explicit graph-dependent bounds and optimize sampling parameter to obtain a concrete high-probability statement.

## Problem 7 Findings

### 1. Critical: “Gamma is rational PD group” is too loose with torsion
- Ref: `data/first-proof/problem7-solution.md:52`
- Issue: with torsion, ordinary PD-group language is delicate; Bredon/orbifold formulations are needed.
- Repair:
  - Reframe in proper equivariant/orbifold cohomology terms and cite exact theorem.

### 2. Critical: existence of normal map setup is asserted without enough hypotheses
- Ref: `data/first-proof/problem7-solution.md:70`
- Issue: “standard surgery theory gives degree-1 normal map to `BΓ`” is nontrivial for torsion groups.
- Repair:
  - State required finiteness and Poincare pair conditions explicitly, with citation.

### 3. Critical: obstruction vanishing claim is unsupported
- Ref: `data/first-proof/problem7-solution.md:99`, `data/first-proof/problem7-solution.md:116`
- Issue: “vanishes by parity” is too coarse for this setting.
- Repair:
  - Provide explicit L-theory computation/citation for the chosen lattice family.
  - Or weaken conclusion to “conditional on vanishing of specified obstruction class.”

### 4. Medium: Smith-theory discussion is anti-obstruction only
- Ref: `data/first-proof/problem7-solution.md:122`
- Issue: does not help construct the manifold.
- Repair:
  - Keep as side remark, not a central proof step.

## Problem 8 Findings

### 1. Critical: basis/nondegeneracy claim at vertex is not justified by manifold condition alone
- Ref: `data/first-proof/problem8-solution.md:72`, `data/first-proof/problem8-solution.md:73`
- Issue: “{e1,e2,e3,e4} is a basis, guaranteed by topological submanifold condition” is too strong. A topological 2-manifold condition does not by itself force these four edge directions to span all of `R^4`.
- Repair:
  - Add a separate generic-position/nondegeneracy hypothesis and prove where it is used, or rework the argument to avoid requiring full spanning.

### 2. Critical: local surgery theorem is invoked outside its stated smooth hypotheses
- Ref: `data/first-proof/problem8-solution.md:149`, `data/first-proof/problem8-solution.md:150`, `data/first-proof/problem8-solution.md:151`
- Issue: Polterovich/Lalonde-Sikorav results are for smooth transverse Lagrangian intersections; here the objects are polyhedral with creases. The bridge from polyhedral model to smooth transverse setup is asserted, not proved.
- Repair:
  - Insert a rigorous local smoothing-to-immersion lemma that produces the required smooth transverse model before applying surgery.

### 3. Major: global patching of local Hamiltonian isotopies is not established
- Ref: `data/first-proof/problem8-solution.md:161`, `data/first-proof/problem8-solution.md:175`, `data/first-proof/problem8-solution.md:176`
- Issue: composability of local surgeries/edge smoothings needs support-control and compatibility checks; this is currently summarized but not demonstrated.
- Repair:
  - Add disjoint-support scheduling/partition-of-unity argument, with an explicit isotopy gluing lemma.

## Problem 9 Findings

### 1. Major: converse step relies on genericity heuristic, not completed proof
- Ref: `data/first-proof/problem9-solution.md:104`, `data/first-proof/problem9-solution.md:124`
- Issue: “not identically zero by explicit construction” is asserted but not shown in-file.
- Repair:
  - Add explicit witness camera configuration and computed nonzero minor, or cite established determinantal result covering this setting.

### 2. Major: Hadamard-product rank argument gives only upper bound, converse needs lower-bound mechanism
- Ref: `data/first-proof/problem9-solution.md:116`
- Issue: upper bound `<=4` does not imply rank 3 exists generically.
- Repair:
  - Replace with explicit transversality argument or direct polynomial nonvanishing certificate.

### 3. Medium: final iff from “all three matricizations rank 1” should be spelled out algebraically
- Ref: `data/first-proof/problem9-solution.md:139`
- Issue: statement is true, but current text compresses compatibility argument.
- Repair:
  - Add short tensor-factor compatibility lemma.

## Problem 10 Findings

### 1. Major: preconditioner quality is hypothesis-level, not derived
- Ref: `data/first-proof/problem10-solution.md:90`, `data/first-proof/problem10-solution.md:130`, `data/first-proof/problem10-solution.md:132`
- Issue: efficient convergence depends on spectral equivalence assumptions (`D ~ cI`, bounded `delta`), but no conditions are given that guarantee these for the sampled tensor design.
- Repair:
  - Add coherence/leverage assumptions (or RIP-style condition) and derive a concrete bound for `kappa(P^{-1}A_tau)`.

### 2. Medium: asymptotic claim can understate dominant setup cost
- Ref: `data/first-proof/problem10-solution.md:146`, `data/first-proof/problem10-solution.md:163`
- Issue: simplification to `O(n^3 + t(n^2 r + qr))` is correct algebraically, but in many RKHS settings `n^3` Cholesky is prohibitive and can dominate practical runtime.
- Repair:
  - Add regime caveat and mention alternatives (low-rank kernel approximations / iterative inner solves) when `n` is large.

## Recommended Next Actions

1. Treat `P2`, `P7`, and `P8` as highest-priority rewrites (current critical gaps affect correctness, not just exposition).
2. Recast `P4` as open/conjectural unless an analytic proof is added.
3. For `P3`, `P5`, `P6`, and `P10`, keep the frameworks but explicitly label conditional steps and add missing bridge lemmas/citations.
4. Keep `P1` and `P9` conclusions with moderated confidence until theorem-level citations close the remaining major gaps.
