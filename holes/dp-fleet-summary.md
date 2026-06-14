# DP fleet — run summary (for Joe's return, 2026-06-13)

The loom (Claude weft + Codex warp, orchestrated by claude-1 under /loop)
reached its goal: **good coverage**. Active grinding wound down at the
structural ceiling; the remaining gap needs a decision from you, not another
cheap lever.

## Final state (261-paper math.CT corpus)

| axis | result |
|---|---|
| **grounding** | **75%** (structural ceiling per claude-2's residue analysis) |
| **well-formedness** | **0 errors** (atomicity / nesting / sentence all clean) |
| math-span coverage | 100% (R1) · symbol tagging 100% |
| second layer (warp) | bibliography (9742 papers), concordance (78k terms / 23.5M obs), citation graph (30k edges, 12% linkage), concept-DEBT frontier |

Trajectory (full detail in `loss-ledger.md`): 52 → 53 → 57 → 70 → **75%**,
well-formedness 1 → (859 regression exposed by scaling) → **0**.

## Capabilities brought online this run

Weft: appositive typing (the big lever, +13pp), quantifier/where, defined-in-
paper, sub/superscript, display-`:=` (R6), def-equation/name-verb, Galois
alias-layer, noise context-classification, **$-parity root fix** (recovered
swallowed-prose denominators, +5pp, one root for two symptoms). Plus the
per-capability refactor (`dp_capabilities/`) that ended the merge-conflict class.

Warp: W1 bibliography, W3 concordance, W2 citation graph, corpus-DEBT report
(top holes: homotopy colimit, 2-category, dg category, …), citation↔DEBT bridge.

## What remains (your call — not loop-cheap)

1. **Irreducible debt (~25%)** — bound indices, dummy/generic variables used
   with no local introduction. Honest `sorry`-class debt; recorded, not chased.
2. **C-DEFINIENS-DEBT frontier (11k)** — concepts the corpus uses but never
   defines. This is the *second-layer payoff*, a feature: `corpus-debt-summary.md`.
3. **To push past 75%**: a deep-parse mission (display/diagram semantics — the
   GrCalc/codiagonal class). That's a new charter, not a cheap detector lever —
   "no third big lever; the curve flattened by design" (claude-2).
4. **Breadth**: scale beyond 261/9742 papers (doesn't raise %, makes the
   corpus + DEBT report more representative; throttle to protect the JVM).

## Process notes (honest)

- The adversarial checker earned its keep: it caught well-formedness **overfit**
  the moment scaling exposed it (859 errors), and a checker self-bug (the naive
  `$` regex) was found and fixed before it could dispatch garbage.
- **Full regeneration** is the propagation step — detector improvements don't
  reach the corpus number until golden is regenerated; the 859 wf errors were
  entirely stale-golden, not real bugs.
- Agent dispatch was **intermittent** (codex-3 / claude-4 stalled on the regen
  and wf tasks, ≥3 times — the roster/bell-routing issue). claude-1 ran the
  critical-path regens directly rather than re-dispatching into the gap.
- A load-induced JVM crash early on (too many concurrent agents + an unthrottled
  batch + API polling) forced the move to a moderated 2+2 cap with subprocess-
  isolated batches and no polling — which held for the rest of the run.

## Loop status

Wound down at the ceiling (goal met). Re-invoke `/loop` to resume if you want
breadth-scaling; charter a deep-parse mission to push past 75%.

## Archive-scale validation (2026-06-14, 948-paper clean corpus)

Correction to an earlier mid-run claim ("coverage invariants overfit at scale"):
that reading was off a STALE dashboard still containing the outlier 1001.4071,
whose malformed spans alone contributed ~76k untagged runs. With it quarantined,
the honest 948-paper numbers are:
- **grounded 76%, wf 0, C-SYM-TAGGED 67, C-MATH-NONNULL 0.**
The detector GENERALIZES: across 948 diverse papers, grounding holds at the
ceiling, well-formedness is clean, tagging/math coverage essentially complete.
The 4215 wf + ~76k untagged "regression" was ONE pathological outlier distorting
three metrics — not a generalization failure. Net: the structure-mining result
is validated at archive scale, not just the tuned 261-set. Remaining debt is the
honest ~24% ungrounded (bound indices / dummy vars) + the C-DEFINIENS-DEBT
frontier (the second-layer feature, scales with corpus). One deferred real item:
detector-hardening for the outlier-span class (so 1001.4071-type inputs can be
un-quarantined).
