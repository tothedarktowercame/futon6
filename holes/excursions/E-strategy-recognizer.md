# E-strategy-recognizer — finding strategies in NL proofs (the means/ends discipline)

*claude-1 + Joe, 2026-06-22. The plan for the informal-proof strategy/tactic
recognizer that raises the strategy axis of the comprehension floor
([[E-comprehension-foundation]]). Written because the means/ends line drifts if
unstated.*

## The end (do not lose this)

**Find strategies in natural-language CT proofs.** The recognizer's output feeds
the comprehension floor's strategy axis (today ~0.25 on loop-run-70b — the binding
constraint). Lean / mathlib / Herald / ProofBridge are **means**, not the end:
we are not aligning to mathlib; mathlib is a *source of verified tactics* and a
*replication target*, nothing more.

| Asset | Role (means) | Not the end |
|---|---|---|
| Lean tactic set | verified strategy **vocabulary** (kernel furnishes the path-integral verification — see [[E-clean]] / M-wm-policies checkpoint) | — |
| Herald tactic-explanations + per-tactic decomposition | the **method** to seed + extract (tactic↔gesture↔proof-state triples) | not a dataset to swallow whole |
| ProofBridge joint NL/FL embedding + tactic-DAG-via-Lean-REPL | the **comparability bridge** + formal-side tooling, to adapt | not the alignment target |
| Herald round-trip + NLI check | the **validation** method for our recognizer | — |
| mathlib-CT | (a) CT-relevant tactic **source**; (b) inline **replication study** | **not** "align to mathlib" |
| **NL CT proofs** | **the goal — strategies live here** | — |

## Learn the methods, don't just grab the contents

What to ADAPT (not copy):
- **Herald** (arXiv 2410.10878, ICLR'25; Mathlib4): per-tactic decomposition — each
  tactic → its intermediate proof state (goal + var changes) → a localized
  statement → informalize, giving fine-grained **(tactic, proof-state, NL-gesture)**
  triples. Ships **tactic-explanations** (logical role per tactic type) — the
  semantic core of our tactic→gesture vocabulary, built NL→Lean; we **invert** it
  for recognition. Validation = round-trip (Lean compiler + back-translate + NLI).
  Tooling: Lean-Jixia static extraction. *No joint embedding (uses RAG retrieval).*
- **ProofBridge / NuminaMath-Lean-PF** (arXiv 2510.15681, ICLR'26): the **joint
  NL/FL embedding** (the comparability bridge we want), tactic-DAG via Lean REPL,
  retrieval-augmented fine-tuning, verifier-guided repair. Olympiad-domain data.

## Corpus shortlist (verified, 2026-06-22 deep-research run)

Gold proof/tactic-level + informal-proof aligned:
1. **Herald** — Mathlib4, ~45k NL↔Lean4 **proof** pairs, tactic-based + per-line
   proof states. *Closest to general/CT math; primary.*
2. **ProofBridge / NuminaMath-Lean-PF** — 38,951 NL↔Lean4 theorem+proof pairs,
   Lean 4.15, Apache/MIT, GitHub PrithwishJana/ProofBridge. *Seed; olympiad domain.*
3. **leanblueprint** projects (PFR teorth/pfr, sphere-eversion, FLT) — true
   LaTeX-proof ↔ Lean4-proof, research math. *Gold validation slice; assemble
   per-project, small-N, not a download.*

Formal tactic-trace backbone (no informal side — vocabulary/verification only):
LeanDojo Benchmark 4 (mathlib4, 122k proofs/259k tactics), Lean-Workbook (tactic
traces but NL **statement** only).

Gap: none is CT-*specific*; Herald (Mathlib4) + blueprints are closest. CT-specific
tactic distribution needs mathlib-CT extraction (the replication study).

## The co-learning architecture (world-class CT recognizer by end of Phase 2)

This is the tail-eating loop ([[E-comprehension-foundation]]) applied to strategy
recognition — the recognizer is **grown on CT, not imported**:

1. **Seed (starter kit, now):** Lean tactic vocabulary + Herald tactic-explanations
   (inverted to tactic→gesture) + a small adapted starter from Herald/ProofBridge.
2. **Grow on the CT NL-proof corpus (Phase 2):** each mined proof's *unrecognized*
   moves become candidate CT-specific strategy patterns (the high-level combinators
   above the atomic tactic level), gated + minted (df≥2, substance gate,
   author≠reviewer), fed back. Residue shrinks; recognizer becomes CT-specific.
3. **Inline mathlib-CT replication study:** run Herald's per-tactic decomposition on
   mathlib-CT to get CT-domain (tactic, proof-state, NL) triples — a co-learning +
   validation signal *for this run*, against which the recognizer's accuracy is
   measured. The goal stays NL recognition; mathlib-CT is the checkable ground truth.
4. **End state:** the comprehension floor's strategy axis rises as the recognizer
   improves — "best possible given this corpus," now on the strategy layer.

## Build sequence (CPU-first)

1. **Seed tactic→gesture vocabulary** — invert Herald's tactic-explanations + the
   Lean tactic set into a recognition table (tactic-class ← informal gestures).
   Corpus-independent; the first artifact. *(start here)*
2. **Recognizer v0** — deterministic/classical detector (the stack's hotword/pattern
   approach) mapping NL proof steps → tactic-class, expanding the comprehension
   floor's strategy vocabulary beyond the 39 flexiarg.
3. **Self-application validation** — Herald-style round-trip on aligned pairs
   (predict tactic-class from the NL step; check vs the real Lean tactic), starting
   on Herald, then the mathlib-CT replication slice.
4. **Re-measure the floor** — strategy axis should rise; that lift, repeated as the
   pool grows, is the corpus-relativity proof before any Linode spend.

## Results so far (2026-06-22)

- **Step 1 DONE** — `holes/clean/tactic-gesture-vocab.edn` (23 tactics / 135 gestures /
  17 residue + 8 conjecture markers; 9 method-tag cross-refs all resolve).
- **Step 2 DONE** — `scripts/strategy_recognizer.py`. Key finding: gestures live in the
  NL **prose**, not the IATC **claims**. On APM prose 12 tactic-classes fire
  (intro/apply/use/cases/ext/calc/obtain/gcongr/exact/rw/simp/contrapose); on the
  gesture-poor cas-select claim text only 4 generic ones (15.4% recognized).
- **Step 3 DONE** — recognition now runs on the candidate **`source-window` prose**
  and is wired into the floor as a complementary strategy signal
  (`strategy = max(rung-3, prose)`). Prose recovers strategy the claim-path lost:
  vs claim-level, 0709.0248/0712.0724/0801.3843/0708.1921 went `0.00 → 0.29–0.50`;
  vs rung-3, prose wins on 0708.2067 (`0.50→0.60`) and 0711.0473 (`0.57→0.71`).
  Complementary (0705.0452: rung-3 0.65, prose 0.00 — short window), so `max` is right.
- **Herald reconciliation DONE (2026-06-22)** — `scripts/herald_validate.py` over
  `FrenzyMath/Herald_proofs` (44,553 NL↔Lean4 proof pairs, pulled to
  `data/lean-nl/herald_proofs.parquet`). Self-application study (predict
  tactic-class from `informal_proof`, check vs `formal_proof`), 3000 proofs:
  set-level P=0.17 R=0.22 — but the breakdown is the point. **Discursive
  strategies recover well** (intro 0.95, apply 0.93, induction 0.65, contrapose
  0.61, constructor 0.58, suffices 0.56, cases 0.51); **bookkeeping tactics are
  silent in prose** (rw 0.09 [1470×], simp 0.16 [1385×], exact 0.03, refine/ext
  0.02). **Finding: gesture recognition has a principled ceiling at the discursive
  layer — and that IS the strategy layer.** The frequent mechanical tactics
  (rw/simp/exact) are elided in informal prose by design; their absence is correct,
  not a recogniser miss. Reconciliation actions: (1) tier the vocab —
  rw/simp/exact/refine/ext = *bookkeeping, silent, not a recognition target*;
  (2) tighten over-greedy gestures (intro/apply/use FP-rate 0.85–0.98).
- **Reconciliation APPLIED (2026-06-22)** — vocab re-tiered to **12 discursive
  recognition targets + 11 hidden-layer (bookkeeping) tactics** (Joe: "rw is a
  hidden layer; we can't recognize what isn't there"); over-greedy apply/use/intro
  gestures tightened. Re-validated on the discursive layer: **recall 0.22 → 0.66**
  (intro .87, contrapose .83, induction .71, suffices .71, obtain .66,
  apply/constructor/calc .62, cases .53); **78% of Lean tactic occurrences confirmed
  silent bookkeeping**. Residual: apply/suffices precision (greedy + whole-proof
  set-matching artifact), use over-tightened, by_contra/wlog sample-sparse.
- **mathlib-CT replication DONE (2026-06-22)** — `scripts/herald_ct_endtoend.py`.
  Herald is generated from Mathlib4, so its **CategoryTheory subset (3952 proofs)
  IS mathlib-CT with the NL side attached** — the replication slice, no LLM needed.
  End-to-end two-layer profile per proof: discursive (recognizer on prose) atop
  hidden (bookkeeping from the Lean trace). **Recognizer recall on CT discursive
  strategies 0.71** (intro .94, contrapose 1.0, suffices .86, constructor .75,
  induction .69, obtain .63, apply .61). **CT tactic signature** — discursive:
  apply/intro/constructor/obtain/cases (little induction); hidden: simp 2079, rw
  1541, exact 869, **ext 552, aesop_cat 116** (equational + naturality/diagram-chase,
  unlike olympiad's linarith/ring). The two-layer architecture works end-to-end on
  real CT proofs.
- **Next:** grow the discursive vocab on CT from the 562 recall-misses (co-learning);
  per-step attribution for precision; feed the two-layer profile into CLean/the floor.

*Cross-refs:* [[E-comprehension-foundation]], [[E-clean]], `CLEAN-LEAN-RELATION.md`,
`futon2/holes/M-wm-policies.md` (the G(π) checkpoint), Herald (arXiv 2410.10878),
ProofBridge (arXiv 2510.15681).
