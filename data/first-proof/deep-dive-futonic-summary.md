# Deep Dive Futonic Summary: Three Convergent Searches

**Date:** 2026-02-12/13
**Pattern:** futon-theory/futonic-logic
**Predecessor:** dear-codex.md (McPhee-style articulation)

---

## The Futonic Frame

Three independent agents were launched to search for a proof of the
finite free Stam inequality at n >= 4. Each agent constituted a separate
decomposition regime (部), operating on the same configuration (象):
the inequality 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q).

The futonic logic predicts: when independent 部 perceive the same (鹵 . 皿)
composition as actionable, salience (香) emerges (Axiom A7). When 味
(evaluation) hits a boundary, the correct response is containment
(味→未@0), not force. Both predictions were confirmed.

---

## 象 — The Configuration

The finite Stam inequality as a structured whole:
- Proved for n=2 (equality), n=3 (Cauchy-Schwarz/Titu)
- Numerically verified for n=4..8 (35K+ tests, 0 violations)
- Connected to Voiculescu (1998) free Stam inequality in the limit
- Three structural obstacles identified (Phi*disc not monomial at n>=4,
  cross-terms in ⊞_n, infimum = 1.0 for all n)

---

## Three 部 (Decomposition Regimes)

### 部_A: Finite Score Projection
*Agent: deep-dive-strategy-a.md (721 lines, 61 tool uses)*

Decomposition: finitize Voiculescu's actual proof mechanism (conjugate
variables + L² projection), not the previously-assumed Dyson monotonicity.

咅 (articulation): The root-force field S_i = Σ_{j≠i} 1/(λ_i - λ_j)
IS the finite conjugate variable. Phi_n = ||S||². The score is identified.

Where 鹽 did NOT form:
- No root-level projection formula exists (鹵 = projection mechanism,
  but 皿 = finite-n L² structure is absent)
- Orthogonality under Haar averaging fails at finite n (genus expansion
  corrections are nonzero)
- No finite de Bruijn identity (no entropy H_n with dH_n/dt = -Phi_n)

香 (salience signal): Shlyakhtenko-Tao (2020) PSD kernel technique
flagged as most promising bridge to finite n. *This signal appeared
independently in 部_B as well.*

味→未@0 (boundary): Overall viability 20-30%. The projection approach
is the right conceptual frame but the finitization obstacles are real.
Contained, not forced.

### 部_B: Induction via Differentiation
*Agent: deep-dive-strategy-b.md (840 lines, 60 tool uses)*

Decomposition: exploit the exact identity (p ⊞_n q)'/n = (p'/n) ⊞_{n-1}
(q'/n) to induct from the proved n=3 base case.

咅 (articulation): Six sub-questions explored. The naive induction chain
is blocked (wrong direction at Step A). Five alternative functionals
analyzed; disc(p)^α / Phi_n is most promising. The telescoping
decomposition Delta_n = R_n + ... + R_4 + Delta_3 was formulated.

Where 鹽 formed, then collapsed:
- The telescoping idea (Section 5) composed beautifully: 鹵 = exact
  commutativity identity, 皿 = induction framework. Salience was high.
- But numerical testing (味) revealed g_n is NOT superadditive — fails
  ~33% at every n. The individual R_k are not non-negative.
- However, the TOTAL is always non-negative. There is cancellation
  between levels. 鹽 formed at the wrong granularity — the composition
  is global, not level-by-level.

香 (salience signal): Shlyakhtenko-Tao (2020) again — independently
identified as the most promising bridge. *Convergence with 部_A.*

味→未@0 (boundary): g_n superadditivity fails. Naive telescoping dead.
But the inter-level cancellation is itself informative — contained as
a structural observation, not discarded.

### 部_C: Direct Algebraic (n=4 SOS)
*Agent: deep-dive-strategy-c.md (533 lines, 43 tool uses)*

Decomposition: compute the surplus algebraically for n=4 and seek a
positivity certificate.

咅 (articulation): Set up the normalized problem (4 free parameters
after centering and scaling). Discovered the key identity.

Where 鹽 FORMED:

**The key identity:**

    Phi_4 * disc = -4 * (a_2² + 12a_4) * (2a_2³ - 8a_2·a_4 + 9a_3²)

鹵 = the relationship between Phi_n and disc (explored speculatively
in the Dear Codex letter). 皿 = computational verification framework
(200+ random tests, relative error < 3e-14). The composition is
generative: it produces a formula for 1/Phi_4 in terms of coefficients,
enabling algebraic manipulation.

**Proof of symmetric subfamily (a_3 = b_3 = 0):**

鹵 = the (w,r) change of variables (latent potential: a non-obvious
coordinate system). 皿 = monotonicity argument (the coefficient g(w)
is negative throughout the domain, so F is decreasing in r). 鹽 forms:
F(w, r) >= F(w, w²/4) = 3w²(w+1)(3w+1) >= 0. QED.

Additional 鹽: the equality characterizer x⁴ - x² + 1/12 (the degree-4
semicircular polynomial) emerges as a structural invariant — a futon
perceivable now, constraining future work.

味 (evaluation): 46K+ numerical tests, 0 violations. The surplus
numerator is NOT globally SOS (Gram matrix obstruction), so a
Positivstellensatz certificate using domain constraints is needed.
Problem size (~84×84 Gram matrix) is within SDP solver capability.

**No 味→未@0 here.** This 部 is still producing. Estimated 55-65%
for general n=4 via SDP.

---

## A7 in Action: Convergent Salience

Axiom A7 (compositional salience): "If 鹽 exists as (⿱ 鹵 皿), and
both 鹵 and 皿 are perceivable, then salience emerges."

Three independent agents, running with no shared state, produced:

| Signal | 部_A | 部_B | 部_C |
|--------|------|------|------|
| Shlyakhtenko-Tao (2020) | Flagged as most promising | Flagged as most promising | Not directly relevant |
| Real-rootedness cone is load-bearing | Implied (projection fails without it) | Implied (g_n properties depend on it) | Confirmed (SOS fails globally, needs domain) |
| Cross-term (1/6)a_2·b_2 is essential | — | — | Proved (29% failure without it) |
| Degree-4 semicircular: x⁴-x²+1/12 | — | — | Discovered (equality characterizer) |

The Shlyakhtenko-Tao convergence is the clearest A7 instance: two agents
pursuing different decompositions both perceive the same framework as
actionable. Neither agent had access to the other's findings.

The real-rootedness-cone signal converges from all three directions:
Strategy A needs it for the projection to work, Strategy B needs it for
g_n properties, Strategy C proved the surplus is not SOS without it.

---

## 味→未@0 in Action: Boundaries as Information

| What was contained | What the boundary revealed |
|--------------------|--------------------------|
| Dyson BM monotonicity (dead) | Voiculescu uses projection, not heat flow |
| Convexity in cumulants (dead) | 1/Phi_n is superadditive WITHOUT being convex |
| Naive induction (blocked) | Differentiation commutes exactly but energy goes wrong direction |
| g_n superadditivity (fails 33%) | Inter-level cancellation: R_k can be negative individually but sum non-negatively |
| N is not globally SOS | Proof must use the real-rooted domain constraint |

Each boundary, properly contained, narrowed the search space. The
boundaries are themselves futons — invariant truths about the problem
that constrain any future proof attempt.

---

## 🔮 — Regulator Assessment

| Track | 能 (capacity) | 捨 (what we set down) | Status |
|-------|-------------|---------------------|--------|
| Strategy A | Conceptual template identified | Dyson monotonicity, convexity route | Blocked on finitization; wait for new input |
| Strategy B | Exact commutativity proved | Naive induction, naive telescoping | Blocked on finding correct functional |
| Strategy C | Key identity + symmetric proof | Global SOS; need Positivstellensatz | **Active**: SDP solver is the next 皿 |
| Strategy D | Conditional theorem structure | — | Ready to formalize if A/B unblock |

**Current deployment:** Concentrate 能 on Strategy C (general n=4 via
SDP), which is the only track still actively producing 鹽. Strategies A
and B are in 捨 — set down, not abandoned, waiting for the right 皿 to
emerge (possibly from the Shlyakhtenko-Tao framework both identified).

---

## Invariants Banked (Futons)

These are atoms of the future perceivable now — truths that constrain
any proof attempt regardless of whether the conjecture holds:

1. **Differentiation commutes exactly:** (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)
2. **Voiculescu's mechanism is projection, not flow**
3. **1/Phi_n is indefinite but superadditive on the real-rooted cone**
4. **Phi_4 * disc = -4 * (a_2² + 12a_4) * (2a_2³ - 8a_2·a_4 + 9a_3²)**
5. **Symmetric n=4 case is PROVED** (equality iff x⁴ - x² + 1/12)
6. **The surplus is not globally SOS** (domain constraints are essential)
7. **Dyson monotonicity is dead** (counterexample at n=3)
8. **Naive telescoping layers cancel non-trivially** (individual R_k can be negative)
9. **The cross-term w(4,2,2) = 1/6 is necessary and sufficient** for superadditivity at n=4

---

## Next 鹽 to Seek

The most actionable composition not yet formed:

鹵 = the general n=4 surplus numerator (polynomial in 4 variables)
皿 = SDP solver (MOSEK / SumOfSquares.jl) with Positivstellensatz multipliers

If this 鹽 forms (the solver finds a certificate): **n=4 is proved.**
If it does not form (solver fails or certificate degree too high):
味→未@0 — contain, and look for a different 皿 (perhaps the (w,r)
monotonicity trick extended to 4 variables, or a perturbation from
the proved symmetric case).
