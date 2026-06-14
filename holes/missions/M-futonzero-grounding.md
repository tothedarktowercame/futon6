# Mission: M-futonzero-grounding

*Ground the closed loop — make FutonZero learn from reality, not self-confirm.*

**Date:** 2026-06-09
**Status:** FOLDED into Campaign `C-falsifiable-missions` (2026-06-10, futon3c/holes/campaigns/) — this umbrella's diagnosis (closed loop) + grounding edges E1–E5 are the campaign's charter material; the three chartered constituents (M-peradam-grounding / M-pattern-posteriors / M-arguing-worlds) operationalize them. **Retired as a standalone mission** per `:O-capstone-form` → campaign. Kept as the diagnosis record.
**Owner:** claude-3 (authored as the sequel to `M-differentiable-substrate`; this mission is
cross-cutting and coordinates with `M-wm-policies` (claude-1/claude-4), `M-pudding-peradams`
(Joe/futon7), `E-possible-world-regulator`. Working title — see §0; Joe may rename.)
**Sources:** Fable (Opus2x) review of `M-differentiable-substrate` (2026-06-09, incorporated in
that doc's "Review" section); `futon2/docs/futonzero-alphazero.md` §1/§5 (claude-1 + Fable);
`M-differentiable-substrate` §8/§8.7 + the closed-loop synthesis;
`feedback_closed_learning_loop_grounding`.

## 0. Why a mission, not a memo

The Fable reviews didn't find bugs — they found that the impressive thing we built **is not the
thing it resembles**. FutonZero v1 (the gradient prior + the rollout + the cascade) is a real,
verified apparatus, but it is a **closed, self-referential loop**. Making it touch reality is an
open research problem with falsifiable tests and a kill criterion — exactly what a mission is for.
(Working name `M-futonzero-grounding`; candidates if the conceptual core (§3 E3b) leads: `M-out-bootstrap`,
`M-possible-world-argument`, `M-incorruptible-reward`.)

## 1. IDENTIFY — the diagnosis (the firm part)

Two independent Fable reviews converge:

- **Both ends of the loop are the author.** The SPEC the gradient descends toward is *authored*
  (the capability anchors — G3). The VALUE ranking outcomes is *self-graded* (Salingaros `C` /
  the search's `G(π)`, built from the producer's own `:delta-g`). Reward isn't external; the prior
  isn't trained from outcomes. So: my spec → my gradient → my prior → a search I shaped → ranked
  by my value → distilled back into my prior. **Nothing external touches it.**
- **It looks like AlphaZero but is structurally a single-agent MDP** (AlphaTensor-shaped). AlphaZero's
  *defining* feature is the closed **self-play** loop bottoming out in an **incorruptible terminal
  reward** — an adversary that is itself, generating an automatic curriculum. v1 has none of these.
- **Self-graded reward is Goodhart's door.** `C` scores a cascade coherent *without doing the
  work* (assemble six historically-co-applied patterns → a respectable `C` whether or not the
  ARGUE bears on the circumstance). Laundering `C` into reward is exactly the move the
  anti-laundering invariant forbids.
- **Such loops don't merely fail to improve — they can confidently converge to artifacts of the
  metric** (the 5 features' biases), *looking* confident (a peaked prior) while learning nothing real.

**The gap is the mission:** install the external grounding that makes the loop learn from reality.

## 2. What is NOT in question (don't re-litigate)

The v1/v2 **infrastructure is real and verified** and is not what this mission reopens: the
producer (`diffsub_emit.py`, scope-grain, 5565 nodes), the rollout (`futon2.aif.rollout`), the
locked drift-proof interface, the depth-chain reachability, the end-to-end green seam. This mission
is about the **learning claim**, which the infrastructure does not yet earn.

## 3. The grounding edges (named, from both reviews)

- **E1 — the value end: an incorruptible EXTERNAL reward.** The **peradam** (Pudding Prover's
  3-witness certificate: labor + arrow + fruit), NOT the self-graded `C` / self-estimated `G(π)`.
  `C` is the value *heuristic* that ranks candidates *for* the peradam; only a certified discharge
  is the reward. (= `M-differentiable-substrate` CH2 ↔ `M-pudding-peradams`, the discharge→peradam
  per-`:move/id` wiring; the anti-laundering invariant.) **Load-bearing.**
- **E2 — the spec end: a LEARNED, not authored, target.** G3 — the gradient currently descends
  toward an authored capability-anchor spec. E1 (value) and E2 (spec) are *the same hole from two
  sides*; grounding only one leaves the loop half-closed. Both ends must be externalised.
- **E3 — the endogenous engine (the OPEN conceptual core, held open per trust-the-method).** What
  generates the curriculum?
  - **(a) Self-play + adversary** (the AlphaZero path): R2 (search-result trains the prior) + the
    **Pudding Prover as REFUTER** (tries to refute a peradam claim) → genuinely two-player, importing
    the curriculum dynamics.
  - **(b) Argument-across-possible-worlds** (Joe's out-bootstrap — leading hypothesis): competing
    pattern-theoretic *buildouts* of the same circumstance, the more-**whole** one winning; the
    peradam is **EXOGENOUS** (the proof/anchor, like Lee Sedol — not the trainer). This drops the
    game *board* (no simulator): the cascade's `C` evaluates a possible world *directly*, so patterns
    (good-enough invariants) replace the simulated playout. Machinery already exists:
    `E-possible-world-regulator` (referee), the gradient-vs-rollout dialectic (disagreement-as-signal)
    (the argument), the judge-panel patterns (the tournament). Generative dialectic (curriculum) +
    adversarial refutation (the Pudding-Prover peradam) + the exogenous peradam-anchor (against
    Goodhart) may be the real shape — *"FutonZero rather than a port."*
- **E4 — the object mismatch: semilattice vs linear path.** A cascade is a **semilattice** (patterns
  overlap — "A City is Not a Tree"); `G(π) = Σ γ^t g(s_t)` is a path-integral over a **sequence**.
  Linearising betrays the partial-order where it matters most; **GFlowNets** (sample compositional
  objects ∝ reward) are the better-fitting machinery — nearly a definition of "assemble a
  pattern-language scored by wholeness." Reconciling the cascade lane (semilattice) with the rollout
  (linear) is open.
- **E5 — the parsimony budget is a hyperparameter (budget-6), not a principle.** Since `C` is monotone
  and bounded only externally, the budget *is* the regulariser. Why 6? An empirical coverage-knee, a
  tunable constant — and per the scale-invariant-degeneracy lesson, exactly where the next degeneracy
  could hide.

## 4. The falsifiable tests (kill criteria — the discipline)

A typed sorry beats a smooth claim. This mission must be falsifiable, not ceremony:

- **T1 — does multi-step search pay rent?** After scope-grain v2, **≥ ~15%** of top-ranked policies
  must be non-greedy (a multi-step `G(π)` strictly beating its greedy prefix). Else the rollout is
  over-engineering — **kill it**, let the cascade lane carry the value alone. (futonzero §5.)
- **T2 — is the unlocking emergent, not encoded?** The depth-5 chains are partly *by construction*
  (the eightfold-phase ordering I wrote). The real test: a capability flip on one mission opening
  reachable `:have`s on **another** mission (cross-mission, emergent). If unlocking is only the
  encoded workflow, multi-step search is rediscovering what we told it.
- **T3 — does arguing across worlds beat the best single buildout?** (Joe's test for E3b.) If
  arguing across competing buildouts doesn't beat the single best one, the dialectic is **ceremony**.
- **T4 — does the prior carry real information?** The peaked prior is *presentational* (manufactured
  by temperature from a 0.0127 score-spread). It carries information only once E1/R2 trains it from
  real outcomes. Until then it must carry **zero external weight**.

## 5. Methodology carried forward (the assurance edge)

**Structural witnesses don't cover semantic contracts** — the R1 re-flatten passed a green structural
handshake (21/3/7/0, exact) while violating the central semantic contract (is `:prior` consumed?).
Every grounding edge needs a **semantic** witness, not just a structural one. This *is* the
assurance-machinery narrative — and the **pitchable artifact** (Fable §6): the
**cascade-as-scored-ARGUE** — *"the system can tell you when its own case is weak"* (cursor `C`=1.145
vs on-ascent `C`≈9.9), legible to the UKRN/consulting audience, the bridge back to
`M-futon-forward-model`.

## 6. Success criteria / definition of done

The loop is **grounded** when: **(E1)** a real peradam — not `C` — trains the value via CH2; **(T1/T2)**
the kill-criteria are *run on v2 data and reported* (kill or keep, honestly — not assumed); **(T3)**
the possible-world-argument frame is tested against the single-best-buildout baseline; **(T4)** the
prior's information content is no longer a tuned hyperparameter. Until ≥1 of E1/E2 is live + witnessed,
the apparatus is described externally as **"a verified closed loop," NOT "a learning system."**

## 7. Log

- **2026-06-09 — IDENTIFY opened (claude-3).** Spawned from the two Fable (Opus2x) reviews — of
  `M-differentiable-substrate` and `futon2/docs/futonzero-alphazero.md` — which converge on the
  closed-loop diagnosis. The honest sequel to v1: stop building the loop, start grounding it.
  Conceptual core (E3) held open; leading hypothesis = Joe's argument-across-possible-worlds out-bootstrap.
