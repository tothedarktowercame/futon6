# Frontier Superpod Empirical Trial (MO, 2026-03-04)

Goal: run the same practical FrontierMath retrieval trial loop on the smaller MathOverflow corpus.

## Command used

```bash
cd /home/joe/code/futon6
python3 scripts/run-frontier-superpod-trial.py \
  --outdir /home/joe/code/storage/mo-processed-gpu \
  --frontier-dir data/first-proof/frontiermath-pilot \
  --out-review data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-review.json \
  --out-summary data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-summary.md \
  --rebuild-compact
```

Runtime: ~5.1s.

## Artefacts produced (local, gitignored under `data/`)

- `data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-review.json`
- `data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-summary.md`

Compact cache created:

- `/home/joe/code/storage/mo-processed-gpu/entities.compact.jsonl` (~15MB)

## Practical findings

- `FM-001` (Ramsey books): 29 candidates (`12` lexical + `17` structural)
  - proxy likely/unclear/unlikely: `16/13/0`
  - structural filtered: `1` no-overlap, `2` missing-meta
- `FM-002` (Ramsey hypergraphs): 21 candidates (`12` lexical + `9` structural)
  - proxy likely/unclear/unlikely: `13/8/0`
  - structural filtered: `8` no-overlap, `3` missing-meta
- `FM-003` (Large Steiner systems): 20 candidates (`12` lexical + `8` structural)
  - proxy likely/unclear/unlikely: `12/8/0`
  - structural filtered: `8` no-overlap, `4` missing-meta

Observed quality: lexical top ranks are materially on-topic for all three FM problems (many direct Steiner/hypergraph/Ramsey prompts), consistent with expectation that MO is denser in research-grade combinatorics content.

## Next validation step

Perform human judgement pass on MO review JSON (`yes`/`no`/`unsure`) and score with:

```bash
python3 scripts/run-frontier-superpod-trial.py \
  --score-judgements data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-review.json
```
