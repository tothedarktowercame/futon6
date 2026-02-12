# Process Patterns from the P4 n=4 Proof Journey

**Date:** 2026-02-12
**Context:** 24 scripts, 3 diagram versions, 5 failed approaches, 4 algebraic breakthroughs, 1 remaining gap.

---

## The Arc

The P4 n=4 Stam inequality proof attempt followed this trajectory:

```
EXPLORE (numerical)  →  CERTIFY (algebraic)  →  FAIL (SOS/IA)  →  DECOMPOSE (symmetry)  →  HANDOFF (PHCpack)
     7 scripts              7 scripts              7 scripts           6 scripts              1 note
```

**Timeline:**
- Feb 11 18:28 — First P4 draft (concavity argument, fundamentally wrong)
- Feb 11 22:16 — Concavity error found, numerical verification confirms conjecture
- Feb 11 23:19 — 35K stress tests, zero violations
- Feb 12 00:42 — Three proof strategies proposed
- Feb 12 (session 2) — Key identity proved, surplus numerator computed (233 terms)
- Feb 12 (session 3) — Case 1 exact (resultant, degree 26)
- Feb 12 (session 3) — SOS infeasibility proved structural (7 scripts, degrees 10-14)
- Feb 12 (session 4) — Case 2 exact (30s, degree 127 — after 4 failed attempts)
- Feb 12 (session 4) — Cases 3a, 3b exact (3s each)
- Feb 12 (session 4) — Case 3c: interval arithmetic OOMs, handoff created
- Feb 12 (session 5) — Codex closes Case 3c via PHCpack

**Scripts by approach:**
| Approach | Scripts | Outcome |
|----------|---------|---------|
| Foundational (identity, surplus) | 1 | Key identity proved |
| Critical point search (numerical) | 4 | Landscape mapped |
| SOS / Positivstellensatz | 7 | Structural infeasibility proved |
| Perturbation / global min | 3 | Hessian PD at x₀ |
| Resultant elimination (algebraic) | 6 | Cases 1, 2, 3a, 3b proved |
| Interval arithmetic | 2 | Failed (wrapping, OOM) |
| Grid / Lipschitz / Taylor | 3 | Numerical but not rigorous |

The ratio: **7 failed-approach scripts** (SOS) to **6 successful ones** (resultant). The failures weren't wasted — they proved structural results and narrowed the search space.

---

## What Worked

### 1. Numerical scouting before algebraic proof
The critical point enumeration scripts (11,000+ starts, 3 independent searches) mapped the full landscape before any algebraic proof was attempted. This identified:
- The case structure (which symmetry subspaces have CPs)
- The number and location of critical points per subspace
- The minimum -N values (0 at x₀, 0.05 at boundary-adjacent, 685+ elsewhere)

Without this map, the algebraic case decomposition would have been guesswork.

### 2. Symmetry-stratified elimination
The breakthrough was decomposing by symmetry subspaces in order of increasing difficulty:
1. a₃=b₃=0 (exchange symmetry reduces 4D→2D) — trivial resultant
2. b₃=0, a₃≠0 (parity reduces system size) — hard but doable resultant chain
3. Diagonal a₃=b₃ (exchange symmetry reduces 4D→2D) — easy resultant
4. Anti-diagonal a₃=-b₃ (exchange+parity reduces 4D→2D) — easy resultant
5. Generic (no symmetry reduction) — infeasible without homotopy continuation

Each step used the previous step's techniques and boundary conditions. The ordering was critical: solving easier cases first provided confidence and identified patterns (parity, domain constraints) that fed into harder cases.

### 3. Domain constraint sharpening
The single most impactful insight for Case 2: disc_q = 16·b₄·(4b₄-1)² ≥ 0 forces b₄ ≥ 0. This narrowed the root search from [-1/12, 1/4] to [0, 1/4], eliminating most candidate roots. Without this, the back-substitution phase would have had too many candidates.

More generally: before computing, sharpen the domain constraints. Semi-algebraic constraints often imply simpler constraints on individual variables that dramatically reduce computation.

### 4. Hybrid numerical-exact certification
When Sturm's theorem ran for 4 hours on a degree-70 polynomial (huge integer coefficients in the GCD chain), the solution was:
1. Compute approximate roots via numpy (fast, O(n³) in degree)
2. Place exact rational test points between consecutive roots
3. Evaluate the polynomial exactly at each test point
4. Count sign changes (= number of real roots in interval, by IVT)

This "sign-counting" method ran in 0.2 seconds. It's as rigorous as Sturm's theorem — it uses exact arithmetic for the certification step — but avoids the expensive polynomial GCD chain that makes Sturm impractical for high degrees with large coefficients.

### 5. Restart-safe computation
After a 4-hour Sturm computation was killed with no results, all subsequent scripts used pickle caching:
```python
CACHE_FILE = "/tmp/case2-elimination-cache.pkl"
# Save after each expensive step; load on restart
```
This is a simple discipline, but it saved hours of re-computation.

### 6. Failure as structural insight
The 7 SOS scripts weren't wasted effort. They proved:
- At x₀, all SOS multipliers must vanish (constraint polys are strictly positive)
- This forces σ₀(x₀) = -N(x₀) = 0, but σ₀ is SOS and vanishing creates contradictions
- Therefore: Putinar certificates are **structurally impossible** for this problem

This is a theorem about the problem, not just a failure of computation. It immediately eliminated an entire proof strategy and redirected effort toward algebraic elimination.

### 7. Multi-agent handoff with structured notes
When interval arithmetic OOMed the Linode, the response wasn't "give up" but:
1. Write what's proved and what remains
2. Describe why current approaches failed (with specific numbers)
3. Recommend the next approach with installation instructions
4. Define what success looks like
5. List all relevant files

This handoff note (`case3c-handoff.md`) enabled Codex to pick up the work on a different machine hours later.

---

## What Didn't Work

### 1. Concavity / convexity arguments
The initial proof draft claimed 1/Φ_n is concave in cumulants. This was:
- Wrong direction (concavity gives SUBadditivity)
- Wrong claim (1/Φ_n is neither convex nor concave)
- Masked by a correct conjecture (the inequality IS true numerically)

**Pattern**: When a proof attempt confirms a true conjecture, the proof errors are invisible to numerical testing. Only structural critique (edge-type checking in the wiring diagram) catches them.

### 2. SOS certificates (7 scripts)
Putinar's Positivstellensatz is the standard tool for certifying polynomial positivity on semi-algebraic sets. It failed here because -N has an interior zero — the unique equality point where all constraint polynomials are strictly positive. This is a structural obstruction, not a degree limitation.

**Pattern**: Interior zeros where all constraints are strict kill Putinar certificates. Check for this before investing in SOS computation.

### 3. Interval arithmetic (2 scripts)
- Naive IA: 233-term polynomial → massive wrapping error (only 6.7% certified)
- Centered form: Hessian Frobenius norm ≈ 96,000,000 → quadratic correction larger than function values
- Both: domain-unaware (verified boxes where -N can be -2000, outside the domain)

**Pattern**: Interval arithmetic on high-term-count polynomials over semi-algebraic domains requires domain-aware filtering as a prerequisite, not an afterthought.

### 4. Direct 4D resultant elimination
res(g₁, g₂, a₃) with ~2000 terms timed out at 2 minutes. Even if it succeeded, the second elimination would produce degree ~5000. The 4D system is simply too large for direct elimination.

**Pattern**: Resultant elimination scales as O(d₁·d₂) in degree. When total degree exceeds ~50 and term counts exceed ~500, resultants become infeasible. Decompose first.

### 5. Sturm's theorem at high degree
Sturm sequences require polynomial GCDs, which for degree-70 polynomials with 100+ digit integer coefficients become hour-long computations. The theoretical runtime is polynomial but the constant factors are enormous for exact integer arithmetic.

**Pattern**: Sturm's theorem is practical for degree ≤ 40 with moderate coefficients. Above that, hybrid methods (sign-counting) are faster while retaining rigor.

---

## Diagram Evolution

The three Mermaid diagrams track the evolution of understanding:

**v1** (Feb 11 18:28): Initial Claude draft. Shows concavity argument. 10 nodes. The "obvious" proof.

**v2** (Feb 11 23:00): Post-reviewer. Softer conclusion, retracted claims. Still the conceptual frame of exploration. 10 nodes.

**v3** (Feb 12, session 4): Proof architecture. Case decomposition, color-coded status, technique nodes, failure nodes. 17 nodes. The actual proof structure.

The jump from v2 to v3 represents a phase transition: from "exploring a conjecture" to "engineering a proof." The diagram vocabulary changed from IATC performatives (assert, challenge, clarify) to proof engineering (proved, pending, failed, technique).

---

## Proposed Patterns

### New patterns (not in futon3 library)

#### 1. math-informal/numerical-scout
**When you want to prove a statement about a continuous function on a domain,
first map the landscape numerically before attempting algebraic proof.**

The scout identifies: critical points, boundary behavior, potential case structure,
minimum values, symmetries. This map guides the algebraic strategy. Without it,
you're guessing which approach to try.

Evidence: P4 numerical scripts preceded algebraic proof by 2 sessions.
Existing pattern: None (estimate-by-bounding is about bounding, not scouting).

#### 2. math-informal/structural-obstruction-as-theorem
**When a proof method fails, characterize the failure as a theorem about the
problem's structure rather than abandoning the approach silently.**

The SOS infeasibility wasn't just "it didn't work" — it was "interior zeros
structurally block Putinar certificates." This insight is as valuable as a
successful certificate: it eliminates an entire proof strategy and redirects
effort.

Evidence: 7 SOS scripts → structural infeasibility theorem → informed case decomposition.
Existing pattern: None (split-into-cases is about succeeding, not learning from failure).

#### 3. math-informal/hybrid-certification
**When a pure algebraic method is too slow and a pure numerical method isn't
rigorous, combine them: use numerical computation to locate, exact arithmetic
to certify.**

The sign-counting method: numpy for approximate root locations, then exact
rational evaluation at test points. 0.2s vs 4 hours. Same rigor as Sturm.

Evidence: Case 2 degree-70 factor.
Existing pattern: None.

#### 4. agent/computation-before-sharpening
**Before expensive computation, exploit problem constraints to shrink the
search space. Semi-algebraic constraints often imply simpler variable bounds.**

disc_q ≥ 0 ⟹ b₄ ≥ 0 halved the search interval. Domain-specific constraints
should be extracted and simplified before being fed to general-purpose solvers.

Evidence: Case 2 domain narrowing.
Related: math-informal/estimate-by-bounding (but this is pre-computation, not bounding).

#### 5. agent/restart-safe-checkpoint
**Long computations must save intermediate results so they survive interruption.
Use pickle/serialization after each expensive step.**

Learned from: 4-hour Sturm computation killed with no results. All subsequent
scripts used cache files. Simple discipline, large payoff.

Related: coordination/bounded-execution (broader; this is a specific instantiation).

#### 6. agent/progressive-method-escalation
**Try the simplest applicable method first. Escalate only when simpler methods
fail, and record why they failed.**

Sequence in P4: resultant (simple) → Sturm (classical) → sign-counting (hybrid) →
interval arithmetic (numerical) → SOS (optimization) → homotopy continuation (heavy).
Each failure narrowed the space of viable approaches.

Related: agent/budget-bounds-exploration (about budgets; this is about method ordering).

### Specializations of existing patterns

#### 7. math-informal/symmetry-stratified-elimination (specializes split-into-cases)
**When a polynomial system has symmetry, decompose into invariant subspaces
ordered by increasing dimension.** Solve the most symmetric subspace first;
use its techniques and boundary conditions to inform the next.

Evidence: Cases 1→2→3a→3b→3c ordered by symmetry reduction effectiveness.
Parent: math-informal/split-into-cases.

#### 8. agent/resource-exhaustion-handoff (specializes social/scope-bounded-handoff)
**When computation exceeds available resources, create a structured handoff
with: what's proved, what remains, why it failed, recommended tools, success
criteria, relevant files.**

Evidence: case3c-handoff.md after interval arithmetic OOMed.
Parent: social/scope-bounded-handoff.

---

## Meta-Observation: The Confidence Anticorrelation

From the making-of narrative: "High confidence" self-assessments (P4, P9) were
the worst failures. The easiest-feeling problems had the deepest gaps.

This echoes in the P4 journey: the "obvious" approach (SOS certificates) led to
7 scripts and structural infeasibility. The "ugly" approach (case-by-case
resultant elimination) led to the actual proof.

**The pattern**: When a proof method feels elegant and general, check whether
it's blocked by structural obstructions before investing. When a method feels
tedious and case-specific, it may be the one that works, because it's engaging
with the problem's actual structure rather than trying to abstract it away.

---

## Files Referenced

| File | Role in pattern discovery |
|------|--------------------------|
| `scripts/verify-p4-n4-sos*.py` (7 files) | Evidence for structural-obstruction-as-theorem |
| `scripts/verify-p4-n4-case2-*.py` (5 files) | Evidence for progressive-method-escalation |
| `scripts/verify-p4-n4-case2-final.py` | Evidence for hybrid-certification, restart-safe-checkpoint |
| `scripts/verify-p4-n4-case3-diag.py` | Evidence for symmetry-stratified-elimination |
| `scripts/verify-p4-n4-classify-cps.py` | Evidence for numerical-scout |
| `data/first-proof/case3c-handoff.md` | Evidence for resource-exhaustion-handoff |
| `data/first-proof/problem4-v1/v2/v3.mmd` | Evidence for diagram evolution tracking understanding |
| `data/first-proof/making-of.md` | Evidence for confidence anticorrelation |
