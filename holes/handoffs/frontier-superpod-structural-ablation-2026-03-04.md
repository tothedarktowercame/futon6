# Frontier Superpod Structural Ablation Note (2026-03-04)

Context: concern that overlap gating (`--struct-min-overlap 1`) may collapse structural retrieval into lexical retrieval.

## Ablation runs

Ungated runs (allow pure structural novelty):

```bash
# MO
python3 scripts/run-frontier-superpod-trial.py \
  --outdir /home/joe/code/storage/mo-processed-gpu \
  --out-review data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-review-ungated.json \
  --out-summary data/first-proof/frontiermath-pilot/superpod-frontier-trial-mo-summary-ungated.md \
  --struct-min-overlap 0 --allow-blank-struct-titles

# Math.SE
python3 scripts/run-frontier-superpod-trial.py \
  --outdir /home/joe/code/storage/math-processed-gpu \
  --out-review data/first-proof/frontiermath-pilot/superpod-frontier-trial-review-ungated.json \
  --out-summary data/first-proof/frontiermath-pilot/superpod-frontier-trial-summary-ungated.md \
  --struct-min-overlap 0 --allow-blank-struct-titles
```

## What the ungated output shows

Structural candidates per problem are 20 (by config).  Zero-overlap counts:

- MO:
  - FM-001: 3/20
  - FM-002: 11/20
  - FM-003: 12/20
- Math.SE:
  - FM-001: 7/20
  - FM-002: 10/20
  - FM-003: 12/20

So for FM-002/003, roughly half of structural neighbors are token-disjoint from frontier query terms.

Observed sample titles in zero-overlap pool are often generic/off-topic (graph diameter, epi-convergence, induction meta, etc.), indicating substantial noise.

## Interpretation

The critique is valid: if we only keep overlap>0 structural neighbors, we reduce novelty risk but also bias toward lexical compatibility.

Empirically, ungated structural novelty currently has low precision proxy and high noise, especially for FM-003. Therefore we should not treat raw structural novelty as proven value yet.

## Mechanism now available in scorer

`run-frontier-superpod-trial.py --score-judgements ...` now reports candidate counts and metrics by source bucket:

- `lexical_seed`
- `structural_neighbor`
- `structural_neighbor_zero_overlap`
- `structural_neighbor_with_overlap`

This enables direct measurement of structural lift once judgements are filled.

## Recommended evaluation protocol (next pass)

1. Judge ungated files (not only gated), at least top 10 structural per FM problem.
2. Compute source-sliced metrics with scorer.
3. Define structural lift as:
   - yes-rate(`structural_neighbor_zero_overlap`) minus yes-rate(`lexical_seed`) at matched review depth
   - plus unique-yes count from structural pool not appearing in lexical seed set.
4. If zero-overlap structural lift is non-positive, keep overlap gate in production path and treat structural novelty as research mode only.
