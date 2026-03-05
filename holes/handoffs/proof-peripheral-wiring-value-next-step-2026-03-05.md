# Proof Peripheral: Wiring Value vs Speed (P7 MO Pilot, 2026-03-05)

Question addressed: are wiring diagrams improving proof-polish quality, or just slowing runs?

## What was run

Corpus/control:

- corpus: `se-data/mathoverflow.net` (MO-only)
- model: `gpt-5.3-codex`
- web search: `disabled`
- timeout/retries: `300s`, `max-retries=2`
- parallel workers: `3`

Ablation arms (same 5 nodes: `p7-problem, p7-s1, p7-s2, p7-s3, p7-s3a`):

1. claim-only context
2. wired context

Outputs:

- `data/first-proof/problem7-codex-results-exp-mo-only-claimonly-pilot5.jsonl`
- `data/first-proof/problem7-codex-results-exp-mo-only-wired-pilot5.jsonl`

Comparison report:

- `holes/handoffs/proof-peripheral-wiring-ablation-p7-pilot5-2026-03-05.md`

## Result

- claim-only: `verified=2, plausible=2, gap=1` (score avg `1.20`)
- wired: `verified=3, plausible=1, gap=1` (score avg `1.40`)
- pairwise (5 nodes): wired better `2`, worse `1`, tie `2`

Interpretation:

- On this pilot, wiring produced a small quality gain.
- Wiring also increased mean elapsed time (`114.4s` vs `96.5s`) and a larger max latency tail.
- So wiring is not "free"; it appears to trade speed for some diagnostic lift.

## Practical workflow to avoid low-value reruns

Use two-stage runs:

1. Fast triage pass (claim-only, high parallelism) across all nodes.
2. Targeted wired pass only on uncertain nodes (`plausible`, `gap`, `parse`, timed-out).

This is now supported in the runner:

- `--prompt-mode {wired,claim-only}`
- `--node-id` (repeatable, targeted reruns)
- `--resume-rerun-failures`
- `--resume-rerun-timed-out`

Example targeted rerun:

```bash
python3 scripts/run-proof-polish-codex-p7.py \
  --math-se-dir se-data/mathoverflow.net \
  --output data/first-proof/problem7-codex-results-targeted.jsonl \
  --prompt-mode wired \
  --node-id p7-s2 --node-id p7-s3 --node-id p7-s4 \
  --resume --resume-rerun-failures --resume-rerun-timed-out \
  --parallel 3 --web-search disabled --timeout-seconds 300 --max-retries 2
```

## FAISS note

If we add FAISS, it should be evaluated inside the same ablation harness (claim-only vs wired, fixed node set and replicates), not as an untracked speed tweak. Otherwise we still won't know if understanding improved.
