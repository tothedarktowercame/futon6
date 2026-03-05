# Proof-Polish Arm Comparison

Baseline: `claim-only-pilot5` (data/first-proof/problem7-codex-results-exp-mo-only-claimonly-pilot5.jsonl)
Candidate: `wired-pilot5` (data/first-proof/problem7-codex-results-exp-mo-only-wired-pilot5.jsonl)

## Baseline Summary

- rows: 5
- status: verified=2, plausible=2, gap=1, error=0, parse=0
- score: sum=6, avg=1.200
- retries/timeouts: attempts>1=0, timed_out=0
- elapsed seconds: mean=96.5, median=102.3, max=136.9
- references: total=17, mo=12 (70.6%), mse=5 (29.4%)
- missing assumptions: total=16, mean/row=3.20
- non-empty suggested improvements: 5

## Candidate Summary

- rows: 5
- status: verified=3, plausible=1, gap=1, error=0, parse=0
- score: sum=7, avg=1.400
- retries/timeouts: attempts>1=0, timed_out=0
- elapsed seconds: mean=114.4, median=93.9, max=196.0
- references: total=18, mo=16 (88.9%), mse=2 (11.1%)
- missing assumptions: total=17, mean/row=3.40
- non-empty suggested improvements: 5

## Pairwise Node Comparison

- common nodes: 5
- candidate better: 2
- candidate worse: 1
- tie: 2
- only in baseline: 0
- only in candidate: 0

## Node Status Changes

- p7-s2: plausible -> verified
- p7-s3: plausible -> verified
- p7-s3a: verified -> plausible

## Edge-Consistency Diagnostic

- evaluated edge pairs: baseline=4, candidate=4
- target stronger than source (lower is usually better): baseline=1, candidate=1
- hard jumps source<=gap to target=verified (lower is better): baseline=0, candidate=0
