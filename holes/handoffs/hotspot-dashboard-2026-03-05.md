# Hotspot Dashboard

Generated: `2026-03-05T00:33:29.117461+00:00`

## Gates

- Spec gate (all FM state files pass): `FAIL`
- Label gate (>= 30 labels/problem per review file): `FAIL`
- Stubborn-node gate (no persistent unresolved nodes): `FAIL`

## First-Proof Problem Status

| Problem | Rows | Verified | Plausible | Gap | Error | Parse | Unresolved % | Stubborn Nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P1 | 9 | 7 | 2 | 0 | 0 | 0 | 22.2% | - |
| P2 | 10 | 4 | 6 | 0 | 0 | 0 | 60.0% | - |
| P3 | 27 | 2 | 20 | 5 | 0 | 0 | 92.6% | p3-problem, p3-s1, p3-s2, p3-s3, p3-s4, p3-s5, p3-synthesis |
| P5 | 8 | 2 | 2 | 4 | 0 | 0 | 75.0% | - |
| P7 | 91 | 31 | 29 | 30 | 0 | 1 | 65.9% | p7-problem, p7-s6, p7-synthesis |
| P8 | 11 | 10 | 1 | 0 | 0 | 0 | 9.1% | - |
| P9 | 12 | 1 | 4 | 7 | 0 | 0 | 91.7% | - |
| P10 | 15 | 8 | 7 | 0 | 0 | 0 | 46.7% | - |

## Frontier Spec-Lock

| File | Spec Lock Status |
|---|---|
| FM-001-ramsey-book-graphs-state.md | fail |
| FM-002-ramsey-hypergraphs-state.md | fail |
| FM-003-large-steiner-systems-state.md | fail |

## Frontier Review Coverage

| Review File | Labeled | Total | Coverage | Gate Target |
|---|---:|---:|---:|---:|
| superpod-frontier-trial-mo-review-ungated.json | 0 | 96 | 0.0% | 90 |
| superpod-frontier-trial-mo-review.json | 0 | 70 | 0.0% | 90 |
| superpod-frontier-trial-review-ungated.json | 0 | 96 | 0.0% | 90 |
| superpod-frontier-trial-review.json | 0 | 67 | 0.0% | 90 |

## Immediate Focus

1. Complete Spec-Lock for all FM problems before new retrieval runs.
2. Label existing Frontier review files to satisfy label gate.
3. Run targeted proof deep-dives on stubborn nodes only.
