# RAW-CTL re-analysis — the control arm exists, and it is underpowered

**Card:** RAW-CTL ("70B-on-raw control arm"), currently `build` in
`proofcheck-readiness.html`.
**Finding:** the card is wrong twice. The run is not missing — it was executed
2026-06-18 and its artifacts are intact. But the comparison it reported is
invalid, and once corrected the experiment turns out to have **no power to
answer its own question**. The card should move to `partial — run exists,
underpowered`, not to `ready`.

## What the original report claimed

`data/exp-20260618/mark3-eval-summary-70b-raw.md`:

```
Kind: iatc (20 EDN artifact(s))    Papers referenced: 10
grounding-%:      12.50% (4 / 32 resolved warrant edges)
checker-PASS-%:  100.00% (20 / 20 structural items)
substance-PASS-%: 50.00% (10 / 20 items)
```

against an enriched arm reported at 21.4% — i.e. enrichment roughly doubling
warrant grounding. That is the number the readiness card rests on.

## Why that comparison is invalid

1. **Sidecars counted as artifacts.** "20 EDN artifact(s)" for 10 papers is 10
   proof graphs plus 10 `.rung2.edn` reports. The reports are not argument
   graphs, which is also why substance-PASS reads 50% (10/20): the ten
   "failures" are the ten reports being gated as graphs. Fixed generally in
   `run_artifacts.proof_graphs`; on finals only, **substance is 8/8 in both
   arms**.
2. **The arms had different paper sets.** Raw has 10 papers, enriched 8. The
   headline compared a 10-paper arm against an 8-paper arm.

## The corrected comparison

Finals only, paper-matched on the 8 papers present in both arms, using the eval
harness's own `warrant_resolution_counts` so the metric is identical to the one
originally reported (`scripts/rawctl2.py`):

| arm | files | resolved | inference edges | grounding |
|---|---:|---:|---:|---:|
| enriched | 8 | 3 | 25 | **12.0%** |
| raw | 8 | 4 | 26 | **15.4%** |

Delta **−3.4 points**, on a total of 51 inference edges across both arms.

## What this does and does not license

**It does not show that enrichment fails.** It shows the experiment cannot tell.
The entire difference between the arms is **one resolved warrant edge** (3 vs
4). At this size the measurement has no power, and the apparent doubling in the
original report was produced by the two accounting errors above rather than by
the model seeing enriched candidates.

**It does show the card cannot be marked ready.** The stated purpose is to
isolate the go-live confound — does the anatomy enrichment change what the model
recovers? Answering that needs enough warrant-bearing edges for a difference to
be visible. Ten papers yielding ~25 inference edges per arm is roughly an order
of magnitude short.

## What would answer it

The run is cheap to repeat at a size that could resolve it, and no longer needs
a 70B: the 16-paper corpus already produces **419 inference edges and 383
warrants** — about sixteen times the evidence per arm. The comparison wants:

- both arms over the same manifest, finals only, one model, one prompt version;
- the raw arm built by extracting candidates **without** `--all-proofs`
  enrichment inlining, everything else identical;
- the harness metric on both, plus the missing-warrant rate, which is the
  quantity the enrichment is supposed to move.

That is a bounded local run against the served endpoint — hours, not a window.

## Provenance

- Artifacts: `data/exp-20260618/loop-run-70b-raw/` (10 finals + 10 reports),
  `data/exp-20260618/loop-run-70b/` (8 finals + 8 reports); both intact.
- Re-analysis: `scripts/rawctl2.py`, reproducible; uses
  `mark3_eval_harness.warrant_resolution_counts` unchanged.
- The sidecar-selection defect this exposes is the same one that failed the S3
  stage gate on its own directory and put 98 spurious rows in S5's verdict
  distribution.
