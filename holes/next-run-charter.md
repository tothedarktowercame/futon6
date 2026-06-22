# Next end-to-end run — charter (the success criterion is PROGRESS, not throughput)

*claude-1 + Joe, 2026-06-22. The acceptance criteria for the next pipeline run, set
after mark5 (which proved throughput — "100 papers" — but not progress). Read with
`linode-stepper-contract.md` (the corrected full-pipeline DAG) and
`proofcheck-readiness.html` (the cross-paper mining this must facilitate).*

## The principle (Joe)

> "If we do **one** paper end-to-end we should already have metrics, and those metrics
> should **improve** if we do 10. If they are not improving we should be able to
> **pinpoint why.**"

So the run is designed **metrics-first**, not volume-first. "N papers completed" is
**not** a success signal — it hides whether the holistic, cross-paper claim (the entire
point of Phase 2) is actually true. The deliverable next time is a **rising,
decomposable curve**, not a pile of graphs.

## Scope

- **Small first: 10 (then 20) whole papers** — not 100+. Enough to *see the slope*
  between n=1 and n=10.
- **Everything turned on** — the corrected contract's full DAG, *both reasoning
  siblings* (④ IATC over all proofs **and** ⑤ expository over all regions), ② concepts
  corpus-fresh, the comprehension/rung layer, and the **cross-paper mining** (recurring
  holes/concepts, retrieval) that `proofcheck-readiness.html` is about. No thin slice.
- **Whole-paper unit** (object B), not one-proof-per-paper.

## The metric contract

Every headline metric must be: **(a) defined at n=1**, **(b) expected to rise n=1→10**
(because Phase 2 is cross-paper — more papers = richer substrate), and **(c) decomposed
per-stage** so a flat curve points at the responsible stage.

Candidate rising metrics (each already has a producing stage):

| metric | stage | n=1 baseline | why it should rise with n | if flat, suspect |
|--------|-------|--------------|---------------------------|------------------|
| **concept-coverage / G-coverage** | ② / R2d | coverage of paper-1's concepts vs substrate-of-1 | a held-out paper's terms are more often grounded as the substrate grows | ② substrate quality / SFC detector |
| **comprehension floor** (corpus-relative) | S5 | per-proof score vs corpus-of-1 | more papers ground more nouns + strategies | R2d (nouns) or STRAT-REC (strategy axis) |
| **recurring (type,concept) holes surfaced** | WARRANT-NORM / PASS3 | 0 (no recurrence at n=1) | cross-paper gaps repeat (df≥2) → conjecture/weak-proof map fills | the (type,concept) keying / hole normalization |
| **structure-retrieval discriminativeness** | ⑧ | n/a (no neighbours) | the proof-structure space populates → method-clusters-across-topics sharpen | embedding weighting / macro-vs-method (mark5 D1/D2) |
| **expository scope coverage** | ⑤ | scope-kinds hit on paper-1 | minted scopes cover more expository sentences (saturating curve, ~35% @193) | the expository vocab / hole-filling |
| **strategy-recognizer recall** | STRAT-REC | recall on paper-1's prose | co-learning on more CT prose grows the vocab | recognizer vocab growth |

## The diagnosability requirement

A flat or falling metric is a **finding, not a failure to hide** — the run must emit a
per-stage, per-metric breakdown so we can say *which* stage isn't contributing (e.g.
"comprehension flat because the strategy axis is flat because STRAT-REC isn't growing").
This is the mark5 lesson generalized: gates with teeth + attribution, so we never report
"done" over a stalled signal.

## Anti-patterns (explicitly out)

- "100 papers completed" as the headline. **Volume is not progress.**
- One reasoning sibling only (mark5 ran ④, skipped ⑤).
- One proof per paper.
- Reporting raw counts without the n=1→n=10 slope.

## What this implies we still need to build first (CPU, off the meter)

- **S3 all-proofs extraction** + **S4 expository GPU run** wired (contract §needs-build).
- **S6 paper-graph (B) assembler** (the one new component).
- **The metrics harness**: compute the table above at n=1 and n=10 and emit the slope +
  per-stage attribution. *This is the actual centerpiece of the next run* — without it
  we'd just be measuring throughput again.

*Cross-refs:* `linode-stepper-contract.md` (DAG + feature grid), `proofcheck-readiness.html`
(cross-paper mining), `mark5-ct100-results.md` (what throughput-without-metrics looked like).
