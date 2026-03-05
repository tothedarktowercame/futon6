# Frontier Superpod Empirical Trial (2026-03-04)

Goal: move from abstract run checks to practical proof-facing retrieval trials for FrontierMath pilot problems.

## Command used

```bash
cd /home/joe/code/futon6
python3 scripts/run-frontier-superpod-trial.py \
  --outdir /home/joe/code/storage/math-processed-gpu \
  --frontier-dir data/first-proof/frontiermath-pilot \
  --out-review data/first-proof/frontiermath-pilot/superpod-frontier-trial-review.json \
  --out-summary data/first-proof/frontiermath-pilot/superpod-frontier-trial-summary.md
```

Runtime (cached compact entities): ~26s.

## What was added

- `scripts/run-frontier-superpod-trial.py`
  - builds empirical candidate sets per FM problem (`FM-001..003`)
  - combines lexical seed retrieval + structural-neighbor expansion
  - emits review JSON with `judgement` placeholders (`yes`/`no`/`unsure`)
  - emits summary markdown with accounting stats
  - supports scoring mode for completed judgements:

```bash
python3 scripts/run-frontier-superpod-trial.py \
  --score-judgements data/first-proof/frontiermath-pilot/superpod-frontier-trial-review.json
```

## Practical findings from this run

Per problem candidate totals after gating:

- `FM-001` (Ramsey books): 25 candidates (`12` lexical + `13` structural)
  - proxy likely/unclear/unlikely: `15/10/0`
  - structural filtered: `1` no-overlap, `6` missing metadata
- `FM-002` (Ramsey hypergraphs): 22 candidates (`12` lexical + `10` structural)
  - proxy likely/unclear/unlikely: `11/11/0`
  - structural filtered: `8` no-overlap, `2` missing metadata
- `FM-003` (Large Steiner systems): 20 candidates (`12` lexical + `8` structural)
  - proxy likely/unclear/unlikely: `12/8/0`
  - structural filtered: `12` no-overlap, `0` missing metadata

## Improvement made after first trial attempt

Observed failure mode: raw structural neighbours often returned generic “proof-shape” matches with weak topic relevance.

Applied fix in harness:

- require structural candidates to have at least one query-token overlap (`--struct-min-overlap 1` default)
- drop structural candidates with missing titles by default
- report filtered counts explicitly in accounting

This keeps the review set focused on practically actionable frontier threads.

## Artefacts produced (ignored by git)

- `data/first-proof/frontiermath-pilot/superpod-frontier-trial-review.json`
- `data/first-proof/frontiermath-pilot/superpod-frontier-trial-summary.md`

`data/` is gitignored, so these remain local run artefacts.

## Next validation step

Complete human judgements on the generated review JSON and score:

1. fill each candidate `judgement`: `yes`, `no`, or `unsure`
2. run score mode to compute strict precision, weighted score, P@k, MAP
3. compare across reruns when tuning retrieval knobs (`seed_k`, `n_anchors`, `struct_min_overlap`)
