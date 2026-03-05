# Superpod Review Judgement Pass (2026-03-04)

## Scoring
- `yes`: relevant
- `no`: not relevant
- `unsure`: partial/uncertain relevance (0.5 for weighted metrics)

## math-processed-gpu
- pairs: 9
- yes/no/unsure: 5/3/1
- strict precision (yes/(yes+no)): 0.625
- weighted score ((yes+0.5*unsure)/N): 0.611
- P@5 yes: 0.600; P@10 yes: 0.556; P@20 yes: 0.556
- P@5 weighted: 0.600; P@10 weighted: 0.611; P@20 weighted: 0.611
- MAP (yes-only): 0.826

## mo-processed-gpu
- pairs: 50
- yes/no/unsure: 15/21/14
- strict precision (yes/(yes+no)): 0.417
- weighted score ((yes+0.5*unsure)/N): 0.440
- P@5 yes: 0.200; P@10 yes: 0.100; P@20 yes: 0.200
- P@5 weighted: 0.400; P@10 weighted: 0.250; P@20 weighted: 0.375
- MAP (yes-only): 0.295

## combined
- pairs: 59
- yes/no/unsure: 20/24/15
- strict precision (yes/(yes+no)): 0.455
- weighted score ((yes+0.5*unsure)/N): 0.466
- P@5 yes: 0.200; P@10 yes: 0.100; P@20 yes: 0.200
- P@5 weighted: 0.400; P@10 weighted: 0.250; P@20 weighted: 0.375
- MAP (yes-only): 0.308

