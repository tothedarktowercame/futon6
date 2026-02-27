# First Proof: Comparison of Our Solutions with Official Solutions

**Date:** 2026-02-14
**Source:** Official solutions released at `codeberg.org/tgkolda/1stproof`, authored by
Abouzaid, Blumberg, Hairer, Kileel, Kolda, Nelson, Spielman, Srivastava, Ward,
Weinberger, and Williams.

---

## Scorecard

| # | Topic | Our answer | Official answer | Match? |
|---|-------|-----------|----------------|--------|
| 1 | Phi^4_3 measure shift equivalence | YES | **NO** | WRONG |
| 2 | Rankin-Selberg universal test vector | YES | YES | Correct |
| 3 | Interpolation ASEP Markov chain | YES | YES | Correct |
| 4 | Finite free Stam inequality | Conjectural YES | YES (proved) | Correct direction, incomplete proof |
| 5 | O-slice connectivity | Partial YES | YES | Correct direction, incomplete |
| 6 | Epsilon-light subsets | Partial YES | YES (c = 1/42) | Correct direction, incomplete proof |
| 7 | Lattice with 2-torsion, rationally acyclic cover | Conditional YES | **NO** | WRONG |
| 8 | Lagrangian smoothing, 4-valent vertices | YES | YES | Correct |
| 9 | Rank-1 scaling detection for quadrifocal tensors | YES | YES | Correct |
| 10 | RKHS-constrained CP via PCG | YES | YES | Correct |

**Bottom line:** 6 correct answers, 2 correct-direction but incomplete proofs, 2 wrong answers.

---

## Detailed Comparison

### Problem 1 — Phi^4_3 Measure Quasi-Shift Invariance

**We said:** YES (equivalent), via Cameron-Martin + log-Sobolev.

**Official answer:** NO. The measures mu and T*_psi mu are **mutually singular** for
any smooth psi not identically zero.

**What went wrong:** We committed the exact error the official commentary flags as the
primary AI failure mode: we assumed the Phi^4_3 measure is equivalent to the Gaussian
free field measure, then correctly deduced quasi-invariance under smooth shifts from that
false premise. The Cameron-Martin space argument is valid for the GFF but does not
survive the Phi^4 interaction.

**Official proof strategy:** Hairer constructs a distinguishing event B_gamma using Wick
powers and logarithmically divergent renormalization constants. The event has full measure
under mu but null measure under the shifted measure. The proof requires regularity
structures machinery to handle the divergent constants — the kind of delicate cancellation
analysis that cannot be bypassed by formal density arguments.

**Lesson:** The Phi^4_3 measure is *not* equivalent to the free field. This is a deep fact
about the nonperturbative nature of the interaction in 3D. Our heuristic that "the density
exp(-V) is strictly positive and finite mu_0-a.s." glossed over the critical issue that
renormalization constants diverge under the shift, creating genuinely new singular behavior.

---

### Problem 2 — Rankin-Selberg Universal Test Vector

**We said:** YES, via the new vector of Pi in the Kirillov model framework.

**Official answer:** YES.

**Comparison:** Both arrive at the correct answer. Our approach centered on the
new vector / essential Whittaker function serving as universal test vector, using
Bernstein-Zelevinsky theory and the Kirillov model for nondegeneracy. The official
solution takes a somewhat different route through the Godement-Jacquet functional
equation and Mellin inversion with carefully constructed Schwartz functions, but
the essential ingredients (newvector theory, conductor matching via u_Q) overlap
substantially.

The official commentary notes that AI systems typically either (a) constructed W
depending on pi (missing universality), or (b) tried to make the integrand constant on
its support (impossible due to central characters). We avoided both pitfalls by anchoring
on the new vector from the start.

**Status:** Solid match. Difference is primarily in proof architecture, not substance.

---

### Problem 3 — Interpolation ASEP Polynomial Markov Chain

**We said:** YES, citing the multispecies t-PushTASEP from Ayyer-Martin-Williams 2024.

**Official answer:** YES, via the *interpolation* t-Push TASEP with signed two-line queues.

**Comparison:** We identified the right theorem (AMW) and the right Markov chain
family. However, the official commentary raises a subtle but important point: the problem
asks specifically about *interpolation* ASEP polynomials (F*_mu with the starred/interpolation
notation), not the standard ASEP polynomials. The official solution requires a novel
construction using signed two-line queues with intricate weight systems — going beyond
what's in the published AMW paper, which handles the unstarred case.

The official commentary flags "problem substitution" — solving the known related problem
(standard ASEP polynomials) rather than the interpolation variant — as a common AI
error. Our solution may be partially susceptible to this critique, depending on exactly how
the notation bridge between starred and unstarred polynomials is interpreted. Our writeup
did explicitly address the notation correspondence (F*_mu = F_mu in AMW), but the official
solution suggests deeper combinatorial work is needed.

**Status:** Correct answer, but the proof may be solving a slightly easier problem.

---

### Problem 4 — Finite Free Stam Inequality

**We said:** Conjectural YES. Proved for n=2 (equality) and n=3 (via Titu's lemma).
Numerical verification for n >= 4 (8000+ trials, 0 violations).

**Official answer:** YES, with a complete proof for all n.

**Comparison:** We got the right answer and proved the low-dimensional cases. The
official proof uses three steps: (1) relate score vectors of the convolution to those of
the factors via a Jacobian, (2) bound the Jacobian norm using hyperbolic polynomial
theory (Bauschke et al.), (3) apply a Blachman-type argument with Cauchy-Schwarz.

Our n=3 proof (Phi_3 * disc = 18 a_2^2 + Titu's lemma) is a valid special case but
doesn't generalize. The official commentary notes that AI attempts failed to discover
the Hessian/Jacobian connection to hyperbolic polynomial theory — we similarly didn't
find this bridge.

The cross-term observation (plain coefficient addition fails ~40% of the time, confirming
the bilinear structure of finite free convolution is essential) was a good diagnostic, and
the numerical evidence correctly predicted the answer.

**Status:** Right answer, right low-dimensional proofs, but missing the key technique
(hyperbolic polynomial Hessians) for the general case.

---

### Problem 5 — O-Slice Connectivity via Geometric Fixed Points

**We said:** Partial YES. Subgroup-family-indexed characterization using F_O-local
regular slice cells.

**Official answer:** YES, with full characterization.

**Comparison:** Our formulation (restrict slice cells to subgroups in F_O, characterize
connectivity via Phi^H for H in F_O) aligns with the official approach, which also uses
nullification, geometric fixed points, and downward induction on the subgroup lattice.

The official commentary actually notes that AI solutions for this problem were
"essentially correct" in outline, with the main deficiency being sketched rather than
rigorous arguments and missing hypotheses. Our solution fits this profile: structurally
correct but at the subgroup-family level rather than the full indexing-system level.

**Status:** Correct framework, needs tightening to full rigor.

---

### Problem 6 — Epsilon-Light Subsets of Graphs

**We said:** Partial YES. K_n proved with c = 1/3. General graph: vertex-level
feasibility (GPL-V) verified but not proved.

**Official answer:** YES with c = 1/42 for all graphs.

**Comparison:** We proved the K_n case with a better constant (1/3 vs 1/42) for that
specific graph family. The official proof uses a greedy algorithm with a modified BSS
barrier function that sums only the largest sigma eigenvalues, combined with leverage
score analysis. Our approach used similar ingredients (barrier greedy, leverage scores,
Foster's theorem) but couldn't close the gap for general graphs where the barrier
degree d-bar exceeds 1.

The official solution's key insight is the modified barrier function Phi^u_sigma
(restricting to top-sigma eigenvalues), which we didn't employ. This is what enables
the general result with the universal constant.

Interestingly, the official commentary notes that no AI system came close to the full
proof — ChatGPT frankly stated inability, and Gemini gave only vague handwaving.
Our K_n proof and the extensive numerical work (731+ base runs, GPL-V analysis)
represent substantially more progress than the AI baselines they tested against.

**Status:** Significant partial result. Missing the modified barrier function that
unlocks the general case.

---

### Problem 7 — Uniform Lattice with 2-Torsion, Rationally Acyclic Cover

**We said:** Conditional YES (E2 discharged via Fowler; S conditional on surgery
obstruction vanishing).

**Official answer:** NO. Such a compact manifold cannot exist.

**What went wrong:** We correctly identified Fowler's criterion and constructed
lattices satisfying the finite-complex obligation (E2). But the problem asks about a
*closed manifold*, not a finite CW complex. The official solution shows that the upgrade
from finite complex to closed manifold is not merely a technical step that might go
through — it is fundamentally obstructed.

The official proof uses the Novikov conjecture (assembly map injectivity for lattices in
semisimple Lie groups) combined with a cobordism argument: any closed manifold with
the required properties would force a contradiction via the relationship between the
L-class of the manifold and its involution fixed-point set. The surgery obstruction
doesn't just "possibly vanish" — it provably *doesn't* vanish.

The official commentary flags exactly our type of error: using Fowler's paper to show
lattices *can* have finite-complex rational type, then incorrectly expecting the manifold
upgrade to work. The key false step in typical AI proofs is invoking "multiplicativity
of Euler characteristic in finite covers," which fails for infinite complexes.

**Lesson:** Fowler's theorem shows something *can* be done at the CW complex level,
but the Novikov conjecture / L-theory machinery shows it *cannot* be done at the
manifold level. We correctly identified the tension but resolved it in the wrong
direction.

---

### Problem 8 — Lagrangian Smoothing of 4-Valent Polyhedral Surfaces

**We said:** YES, via symplectic direct sum decomposition + product smoothing +
generating function edge smoothing.

**Official answer:** YES.

**Comparison:** Both arrive at the correct answer with overlapping ideas. Our proof
and the official proof both exploit the key structural fact: at a 4-valent Lagrangian
vertex, the symplectic orthogonality constraints force a symplectic direct sum
decomposition R^4 = V_1 + V_2.

However, the official solution takes a more elegant route via *conormal fibrations* and
*smoothing function spaces*, which enables a clean global construction without the
local-to-global compatibility issues that the official commentary identifies as the main
AI failure mode. Our proof handled the global assembly via disjoint supports and
composition of Hamiltonian isotopies (McDuff-Salamon), which works but is more
coordinate-heavy.

The official commentary notes that the best AI solutions got the local analysis right
but broke at global extension — specifically, either asserting disjoint neighborhoods
exist when they don't, or performing vertex moves that invalidate edge moves.
Our solution addressed this with explicit disjoint-support arguments, which appears
to handle the compatibility issue correctly, though the conormal fibration approach
is cleaner.

**Status:** Correct answer and sound proof, with a less elegant but functional
approach to the global extension.

---

### Problem 9 — Polynomial Detection of Rank-1 Scaling

**We said:** YES, via 3x3 minors of the rank-2 bilinear form structure.

**Official answer:** YES, via 5x5 minors of block tensor flattenings.

**Comparison:** Both reach YES through the determinantal structure of the Q-tensor,
but by different routes. Our approach exploits the rank-2 bilinear form that arises
from fixing two camera-row pairs, giving degree-3 coordinate functions (3x3 minors).
The official approach works with Tucker decomposition showing multilinear rank
<= (4,4,4,4), giving degree-5 coordinate functions (5x5 minors of flattenings).

The official commentary notes that one AI solution (NoInternet-040226) was
"essentially correct" using the same 5x5 minor approach. Our 3x3 minor route is
actually a different valid construction — potentially more efficient (lower degree)
if it covers all the necessary matricizations.

The official proof's "only if" direction (non-rank-1 implies some minor nonzero)
requires meticulous case analysis. Our converse used a polynomial nonvanishing argument
(explicit n=5 witness with det = -24), which is valid but less systematic.

**Status:** Correct answer via a genuinely different (and arguably more refined)
construction. The degree-3 vs degree-5 difference is notable.

---

### Problem 10 — RKHS-Constrained Tensor CP via PCG

**We said:** YES, with matrix-free PCG in O(n^2 r + qr) per iteration, and an improved
preconditioner P_new = c(G tensor K_tau^2) + lambda(I_r tensor K_tau) achieving
delta < 1 under uniform sampling.

**Official answer:** YES.

**Comparison:** This is the one problem where the official commentary is *positive*
about AI performance, noting that the best AI solution "was correct and actually better
than human solution in lowering computational complexity." The subsampled Kronecker
product matrix-vector product idea was called "obvious in hindsight but not previously
seen."

Our solution includes the key ingredients: matrix-free matvec avoiding the N-dependence,
eigendecomposition-based preconditioner application, and the K_tau^2 correction to
the preconditioner (our P10-C001 -> P10-C002 improvement cycle). The spectral
equivalence analysis (delta_old in [5.2, 22.7] vs delta_new < 1) directly parallels
the official discussion.

**Status:** Strong match. This is where our work most closely aligns with (and
potentially matches or exceeds) the official solution.

---

## Themes and Takeaways

### Where we did well

1. **Correct answer on 8/10 problems** (counting direction, not proof completeness).
   Better than the AI baselines tested by the problem authors (GPT-5.2 Pro and
   Gemini 3.0 Deep Think).

2. **Problem 10** is a clear win — our improved preconditioner and complexity analysis
   match the official solution's substance.

3. **Problem 8** produced a correct proof with all the key structural insights.

4. **Problem 9** found a different and possibly more efficient construction (degree-3
   vs degree-5 polynomials).

5. **Problems 4 and 6** demonstrated substantial partial results (proved special cases,
   extensive numerical verification) that go well beyond what the tested AI baselines
   produced.

### Where we went wrong

1. **Problem 1:** The Cameron-Martin / free-field-equivalence assumption is the canonical
   AI error on this problem. The Phi^4_3 measure's nonperturbative character in 3D makes
   it fundamentally different from the free field — a fact that requires regularity
   structures to properly understand.

2. **Problem 7:** We correctly identified all the relevant machinery (Fowler's criterion,
   surgery theory, assembly maps) but resolved the finite-complex vs. manifold tension
   in the wrong direction. The Novikov conjecture provides the definitive obstruction
   that we treated as merely "conditional."

### Structural observations

- The two wrong answers (Problems 1 and 7) share a pattern: we built a plausible
  argument from correct components but missed a deep obstruction. In Problem 1, the
  obstruction is the divergence of renormalization constants under shift. In Problem 7,
  the obstruction is the Novikov conjecture / L-theory constraint.

- Both wrong problems are also the ones where the official answer is NO/FALSE, while
  our default tendency was toward YES. This "YES bias" — treating plausible positive
  constructions as sufficient without rigorously excluding obstructions — is worth
  being aware of in future work.

- The problems where we excelled (8, 9, 10) tend to have more explicit algebraic or
  computational structure, where the key insight can be verified concretely. The problems
  where we struggled (1, 7) require deep structural theorems from specialized subfields
  (regularity structures, surgery theory / Novikov conjecture) where formal reasoning
  can't easily substitute for domain mastery.

---

## What the Official Document Says About AI Performance Generally

The authors tested GPT-5.2 Pro and Gemini 3.0 Deep Think. Common failure modes:

1. **Foundational errors:** Assuming false premises, then reasoning correctly from them (us on P1)
2. **Problem weakening:** Solving easier variants (flagged for P2, P3)
3. **Citation without substance:** Quoting papers without understanding content (P1)
4. **Local vs. global gaps:** Correct local analysis, broken global assembly (P8)
5. **False theorems:** Inventing plausible results that are false (P7)
6. **Problem substitution:** Solving the known related problem instead of the actual one (P3)

The authors also announce a **Batch 2** with a formal benchmarking phase, formal grading,
and verification of autonomous solution production.
