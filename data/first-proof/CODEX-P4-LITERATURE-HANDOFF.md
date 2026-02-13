# Codex Handoff: Problem 4 Literature Search

**Date:** 2026-02-13
**From:** Claude (monograph author)
**Priority:** HIGH — results feed directly into proof strategy selection

---

## Context

Problem 4 asks: for monic real-rooted degree-n polynomials p, q, is

    1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q)

where Φ_n(p) = Σ_i (Σ_{j≠i} 1/(λ_i - λ_j))² is the root-force energy
and ⊞_n is the Marcus-Spielman-Srivastava finite free additive convolution?

This is the **finite analog of the free Stam inequality** (Voiculescu 1998).
Proved for n=2 (equality) and n=3 (Cauchy-Schwarz). Open for n ≥ 4.

We have identified a potential **layer switch**: instead of attacking via
random-matrix theory (HCIZ, Weingarten) or heat flow (Dyson BM), attack via
**algebraic combinatorics on the real-rooted cone** — specifically, the
theory of Lorentzian polynomials (Brändén-Huh 2019).

---

## Search Tasks (in priority order)

### Task 1: Lorentzian Polynomials and the Real-Rooted Cone

**Question:** Does the theory of Lorentzian polynomials or related
"cone-restricted convexity" results apply to functions like 1/Φ_n
on the real-rooted cone?

**Search terms:**
- "Lorentzian polynomial" "real-rooted" OR "stable" convexity OR superadditivity
- Brändén Huh "completely log-concave" "real-rooted" cone
- "Lorentzian" "finite free convolution" OR "Marcus Spielman Srivastava"
- "log-concavity" "real-rooted" "root separation" OR "discriminant"
- "Hodge-Riemann" "real-rooted" OR "stable polynomial"

**What we're looking for:**
- Results about functions on the real-rooted/stable cone that are
  superadditive or log-concave despite not being globally convex
- Any interaction between Lorentzian polynomial theory and ⊞_n
- The theory of "completely log-concave" operators and whether
  1/Φ_n fits this framework
- Any result by Brändén, Huh, Adiprasito, Katz, or collaborators
  that applies to energy-type functionals on polynomial spaces

**Why this matters:** 1/Φ_n is superadditive on the real-rooted cone
WITHOUT being globally convex (82.4% midpoint violations in cumulant
space). This pattern — cone-restricted superadditivity — is exactly
what Lorentzian polynomial theory addresses.

### Task 2: Finite Free Entropy

**Question:** Is there an existing definition or result on a finite
analog of Voiculescu's free entropy χ(μ) = ∫∫ log|x-y| dμ(x)dμ(y)
that interacts with ⊞_n?

**Search terms:**
- "finite free entropy" polynomial OR "characteristic polynomial"
- "log discriminant" "free convolution" OR "finite free"
- "finite" "free entropy" Voiculescu superadditive
- Marcus "polynomial entropy" OR "spectral entropy"
- "Coulomb energy" "finite free convolution"
- "log-gas" "finite free" entropy

**What we're looking for:**
- Any functional H_n on monic real-rooted degree-n polynomials that
  satisfies H_n(p ⊞_n q) ≥ H_n(p) + H_n(q) (superadditivity)
- Any finite analog of the free de Bruijn identity:
  d/dt H_n(p_t) = -c · Φ_n(p_t) for some heat semigroup p_t
- Results on the log-discriminant H_n = (2/n²)Σ_{i<j} log|λ_i-λ_j|
  under ⊞_n or related operations
- Any paper by Voiculescu, Biane, Guionnet, Shlyakhtenko, or Collins
  on finite-dimensional free entropy

**Critical connection:** The log-discriminant is the finite analog of
free entropy. If it's superadditive under ⊞_n, this would give
a parallel proof path to the finite Stam inequality via de Bruijn.

### Task 3: Voiculescu's Proof Mechanism — Detailed Steps

**Question:** What are the exact technical steps in Voiculescu (1998,
Invent. Math. 132) proving the free Stam inequality, and which steps
have known finite analogs?

**Papers to read:**
- Voiculescu (1998), "The analogues of entropy and of Fisher's
  information measure in free probability theory, V: Noncommutative
  Hilbert transforms," Invent. Math. 132, 189-227.
- Voiculescu (1993), CMP 155, 71-92. [Free entropy I]

**Extract specifically:**
1. The precise definition of J(X) (the conjugate variable / free score)
2. The conditional expectation E_{W*(X+Y)} in Step 2
3. How L²-contractivity is proved (Step 3)
4. How freeness gives orthogonality ⟨J(X:Y), J(Y:X)⟩ = 0 (Step 4)
5. How ‖J(X:Y)‖² = Φ*(X) follows from freeness (Step 5)
6. Any remarks by Voiculescu on finite-dimensional versions

**Why this matters:** We have a high-level summary of Voiculescu's proof
(from Codex Step 1) but need the EXACT technical details to determine
which steps can be finitized. The key question: is Step 4 (orthogonality
from freeness) the hard step, or is Step 2 (conditional expectation
decomposition)?

### Task 4: Score Projection Under Haar Averaging

**Question:** Are there results on E_U[Φ_n(A+UBU*)] or related
Haar-averaged spectral functionals?

**Search terms:**
- "Haar unitary" "eigenvalue" functional OR "spectral function" average
- "orbital integral" eigenvalue energy OR "Coulomb"
- HCIZ "root separation" OR "discriminant" OR "root force"
- Weingarten "spectral" average OR eigenvalue concentration
- "random rotation" eigenvalue "separation" OR "gap"

**What we're looking for:**
- Formulas or bounds for E_U[f(eigenvalues of A+UBU*)] where f is
  Φ_n or a related energy functional
- Results on concentration of Φ_n(A+UBU*) around its mean
- Any comparison between Φ_n(E_U[char]) and E_U[Φ_n(char)]
  (Jensen gap question)
- Results by Collins, Śniady, or Novak on Weingarten-based
  expectations of symmetric functions of eigenvalues

### Task 5: Differentiation and Induction (Supplementary)

**Question:** Can the exact differentiation identity
(p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n) be exploited despite
the wrong-direction inequality 1/Φ_n < 1/Φ_{n-1}(p'/n)?

**Search terms:**
- Marcus "differentiation" "finite free convolution"
- "derivative" "interlacing" "root separation" OR "Coulomb"
- "Rolle" "energy" monotone OR inequality
- finite free "induction" degree reduction

**What we're looking for:**
- A replacement functional G_n such that G_n(p) ≥ G_{n-1}(p'/n)
  AND G_n relates to 1/Φ_n
- Any result showing that differentiation improves some energy-type
  functional (not just root separation, but any measure of "how
  spread out the roots are")
- Results by Pemantle, Raza, or collaborators on root dynamics
  under differentiation

---

## Return Format

For each search task, return:

1. **Number of hits** (total / relevant / highly relevant)
2. **Top 5 results** with:
   - Title, author, year, URL/arXiv ID
   - Key claim or technique (1-2 sentences)
   - How it connects to P4 (direct / analogous / tangential)
   - Confidence that it helps (high / medium / low)
3. **Synthesis:** What does the literature suggest about the most
   promising proof direction?

---

## Key Files for Context

- `problem4-conditional-stam.md` — the conditional theorem formalization
- `problem4-ngeq4-proof-strategies.md` — four strategies ranked
- `problem4-ngeq4-research-brief.md` — earlier research questions
- `problem4-solution.md` — the full solution writeup (n=2,3 proved)
