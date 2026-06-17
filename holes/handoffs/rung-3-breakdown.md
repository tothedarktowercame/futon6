# rung-3 breakdown — technique-grounding detector (the verb-twin of R2d)

*Breakdown of the `rung-3` card — **moved Phase C → Phase D** (the cascade) on 2026-06-17, because it
depends on the pattern menu. Owner excursion: **E-informal-proof-checking** (claude-1). Drafted by
claude-loop; rewritten 2026-06-17 to the technique-grounding framing (see the excursion's "rung-3 =
technique-grounding" note + CAS-0). DRAFT for review, not dispatched. Leads with an **empirical** spike.*

## IDENTIFY — what rung-3 actually is

Not a correctness judge. rung-3 is the **verb-twin of R2d**: R2d grounds the *nouns* (is each term defined /
known?); rung-3 grounds the *verbs* (is each reasoning move a recognized technique, or a **gap**?). It is a
**detector of thin / ungrounded moves, not an arbiter of truth** — it never asks "is this step true" (that is
the mathematics); it asks "is this step grounded in a known technique, or is there more work to do to even
explain it?". Finding *those* gaps is the use, exactly as finding undefined terms is.

CAS-0 already found the mechanism from the proof side: **a residual sorry = the matched pattern's undischarged
`HOWEVER` clause.** rung-3 is that, per-edge, with one sharpening below.

## MAP — what exists

```
WHAT to ground     IATC warranted edges {premise, warrant, conclusion, :source} — the proof MOVES.
THE TECHNIQUE      the cascade pattern menu (CAS-SEL): 36 math-informal + Pólya + the RM question/IATC
  SUBSTRATE        moves. rung-3 = technique-pattern-COVERAGE of moves, as R2d = concept-coverage of terms.
THE GAP→QUESTION   the RM question-pattern menu (EXISTENCE_WONDER, STRUCTURAL PROBE, …) phrases each gap;
                   ArSE threads carry the questions (typed-bell :ref).
CONJECTURES        author-declared gaps already in the prose (the expository open-problem/status move).
INFRA              vLLM + the OpenAI client (mark3_iatc_loop) for the residue only; CAS-0's worked proofs
                   (cas0-worked-*.md) + loop-run-70b moves as the example set.
```

## DERIVE — "thin" made sharp, and why the LLM share is small

**Technique patterns have a type:** **heuristic** (justifies a *strategy* — "reduce to an easier case",
"consider the generic point"; Pólya/RM-grade; does NOT justify a step) vs **verifiable** (justifies an
*inference* — "by [theorem] whose hypotheses hold", a computation, a def). The same pattern can be either
(`reduce-to-known-result` is a heuristic as a strategy, verifiable once the reduction map is exhibited).
**A cascade chains heuristics but must bottom out in verifiable leaves** — this is what makes it a
*mathematical* cascade, not a generic one.

So **"thin"** is sharp: a load-bearing step whose pattern bottoms out at the *heuristic* level where a
*verifiable* step is required, the verifiable step not exhibited. The **sorry typology**:
- **verifiable step** = a *filled* sorry — GROUNDED;
- **conjecture** = an *author-declared, acknowledged-unfilled* sorry — **credit it** (a corpus open-problem
  map, a first-class output, not a failure);
- **thin step** = an *undeclared* unfilled sorry presented as filled — **the detection target**;
- **ungrounded** = no matching technique at all — the worst gap.

This is the **per-edge instance of CAS-SEL**: fit the best-matching pattern to the move; the **residual** is
the gap, and its *type* (heuristic-leaf? no match?) is the gap's severity. Most moves match a pattern cheaply
(CAS-0 #1: 5/5 deterministically); **the LLM is needed only on the residue** — "is this thin step a valid
novel technique or a real gap?" — and even there the output is a **question, not a verdict**.

**The detector asks; it does not answer.** Each gap → an **ArSE question** ("how does the general case follow
from the example here?"), phrased by gap-type via the RM question-pattern menu. A recurring unanswered
question = a research frontier; an answered one **mints a new verifiable pattern** (the pattern-seeding loop,
now typed heuristic-vs-verifiable).

## ARGUE

> **IF** we want to find the thin/ungrounded steps over all of arXiv (where the real gaps are),
> **HOWEVER** judging step *correctness* is the math itself and has no gold,
> **THEN** make rung-3 **technique-pattern-coverage of moves** (the verb-twin of R2d) — match each move to the
> menu, flag the residual as a likely gap, *credit* author-declared gaps (conjectures), and emit each gap as
> an ArSE *question*,
> **BECAUSE** matching is mostly deterministic (CAS-0), the LLM only adjudicates the residue, a question is
> auditable where a verdict isn't, and conjectures give partial ground truth — and the residual-after-fit is
> a sharp, computable definition of "thin" that bottoms out in the verifiable/heuristic split.

## VERIFY — acceptance for the whole breakdown

1. The **deterministic residue is measured** (not assumed) on real moves — the LLM share is an empirical
   number, with the heuristic/verifiable typing applied.
2. On `loop-run-70b` + the CAS-0 worked proofs: per-move grounding bucket (grounded/thin/ungrounded),
   conjectures credited (not flagged), and each gap emitted as a phrased ArSE question.
3. The LLM is used **only on the residue**, output is a flagged question (`:likely-gap`), never a truth
   verdict; deterministic where deterministic.
4. Wired so the residue feeds the cascade's residual-sorry map (CAS-CERT) and the pattern-seeding loop.

## INSTANTIATE — sub-handoffs (rung-3-1 is an EMPIRICAL spike; depends on the menu being seeded)

> **Dependencies:** rung-3 is in Phase D — it needs the menu (CAS-0 seeding / CAS-SEL), shares the gap-map
> with R2d, and is prioritized by CAS-CERT. Sequence it **after** the cascade is seeded.

### rung-3-1 · Empirical spike — measure the residue + type the patterns · CPU (+ a little LLM)
**Goal (extends CAS-0 Q5):** on the moves of the CAS-0 worked proofs + a sample of `loop-run-70b` edges,
(a) **try to pattern-match each move deterministically** against the menu and **measure the residue** — that
number *is* the LLM share, don't guess it; (b) **type each matched pattern** heuristic vs verifiable, and
mark which residuals are heuristic-leaves (thin) vs no-match (ungrounded); (c) confirm **conjecture
recognition** distinguishes author-declared gaps; (d) draft the **gap→ArSE-question** mapping by gap-type.
**Deliverable:** `holes/excursions/rung-3-spec.md` (the residue number, the heuristic/verifiable typing, the
buckets, the question mapping), reviewed before rung-3-2. **Gate:** PY + the residue measurement.

### rung-3-2 · The technique-coverage detector (deterministic core) · CPU · BB/PY
**Depends on** rung-3-1 + CAS-SEL. **Goal:** per move, fit the best-matching menu pattern (reuse CAS-SEL's
select), classify grounded-by-pattern / grounded-by-citation / thin / ungrounded, credit conjectures. Emit a
per-paper **gap map** (the thin/ungrounded moves) + the buckets. **Acceptance:** reproduces rung-3-1's
hand-classification on the worked proofs; deterministic; no LLM in this stage. **Gate:** PY/BB + numbers.

### rung-3-3 · The LLM-on-residue + ArSE questions · LLM + CPU
**Depends on** rung-3-2 + CAS-CERT. **Goal:** for the thin/ungrounded residue only, an LLM pass decides
*novel-technique vs real-gap* and emits a **likely-gap → ArSE question** (phrased via the RM question-pattern
menu, opened as a typed-bell `:query` with an ArSE `:ref`). Prioritized by the residual-sorry map; bounded
budget. A recurring question that gets answered → mint a verifiable pattern. **Acceptance:** residue-only LLM
use; each gap is an auditable ArSE question, not a verdict; the residual-sorry map shrinks by the
resolved/grounded moves. **Gate:** PY + a human spot-check of the questions + the budget.

## Note
rung-3 is the **terminal fill** of `cascade → sorry → wiring` at edge grain: it grounds what it can in known
technique, *asks* about the rest, and what stays unanswered is the honest "where we're least sure" — the same
output as CAS-CERT's residual-sorry map, one level down. Its value is real only if rung-3-1 shows the
deterministic residue is small; that measurement is the gate on the rest.

## Findings — rung-3-1 empirical residue spike (codex-4, 2026-06-17)

Delivered `holes/excursions/rung-3-spec.md` plus `scripts/rung3_residue_spike.py`.

Measurement method: direct `cas_select.retrieve(..., k=4)` + `cas_select.verify(..., backend="stub",
oracle=...)` on the CAS-0 fixtures, deliberately avoiding `select_proof` because its fixture-only stub path
injects missed oracle patterns. The CAS menu loaded 39 math-informal patterns from the current committed
library/index.

Results:
- CAS-0 strict verified residue: **6/22 = 27.3%**. This is the empirical LLM/verifier share for the current
  oracle-backed worked-proof setting.
- CAS-0 buckets: **grounded 14**, **thin 2**, **ungrounded 6**. Pattern typing: **verifiable 14**,
  **heuristic 2**, **none 6**.
- `loop-run-70b` final graphs: 28 warranted edges. Because no oracle-backed verifier exists for those edges,
  strict verified residue is **28/28 = 100%**. Deterministic retrieval nevertheless proposes a top candidate
  for **28/28** edges; this is candidate reach, not correctness.
- `loop-run-70b` retrieval buckets: **thin 25**, **grounded-provisional 2**, **conjecture 1**. The conjecture
  recognizer credits author-declared/open-status phrasing such as `ought-to-*` rather than treating it as a
  hidden failure.

Gap-to-question mapping is drafted in the spec: heuristic leaves map to `STRUCTURAL PROBE`, no-match gaps to
`THEOREM APPLICABILITY` / `TECHNIQUE LANDSCAPE`, missing warrants to `KERNEL IDENTIFICATION`,
author-declared gaps to `EXISTENCE_WONDER` / `CONJECTURE_TESTING`, and obstruction residuals to
`OBSTRUCTION_IDENTIFICATION`.

Implication: the deterministic selector is useful and honestly bounded. CAS-0 still needs a semantic
retriever/verifier for the 27.3% residue; loop-run needs rung-3-3-style model judgement or new gold before
retrieval candidates can be counted as real matches.
