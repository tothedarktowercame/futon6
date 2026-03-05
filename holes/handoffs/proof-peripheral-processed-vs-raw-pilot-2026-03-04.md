# Proof Peripheral A/B Pilot: Processed vs Raw Corpus (2026-03-04)

Goal: test whether `Proof Peripheral` verification runs benefit from `processed` local corpus hints versus `raw` dump hints.

## Setup

Peripheral runner used:

- `scripts/run-proof-polish-codex-p3.py`

Fixed settings in both arms:

- model: `gpt-5.3-codex`
- reasoning: `medium`
- web search: `disabled`
- same wiring + solution + schema + prompt structure

Only variable changed:

- **processed arm**: `--math-se-dir se-data/math-processed`
- **raw arm**: `--math-se-dir se-data/math.stackexchange.com`

## Commands

```bash
cd /home/joe/code/futon6

# Processed
python3 scripts/run-proof-polish-codex-p3.py \
  --model gpt-5.3-codex \
  --reasoning-effort medium \
  --web-search disabled \
  --math-se-dir se-data/math-processed \
  --output data/first-proof/problem3-codex-results-exp-processed.jsonl \
  --prompts-out data/first-proof/problem3-codex-prompts-exp-processed.jsonl

# Raw
python3 scripts/run-proof-polish-codex-p3.py \
  --model gpt-5.3-codex \
  --reasoning-effort medium \
  --web-search disabled \
  --math-se-dir se-data/math.stackexchange.com \
  --output data/first-proof/problem3-codex-results-exp-raw.jsonl \
  --prompts-out data/first-proof/problem3-codex-prompts-exp-raw.jsonl
```

## Runtime

- processed arm: ~14m45s
- raw arm: ~10m30s

## Aggregate results

Status counts:

- processed: `plausible=7`, `gap=2`, `error=0`
- raw: `plausible=6`, `gap=3`, `error=0`

Per-node status comparison (processed vs raw):

- processed better on: `p3-s2`, `p3-s5`
- raw better on: `p3-s4`
- tie on: remaining 6 nodes

Confidence:

- processed: all `medium`
- raw: `8 medium`, `1 high`

Reference counts (`math_se_references` total):

- processed: `4`
- raw: `6`

Observed quality note:

- raw produced more references, but some were weak/background-only for the target node (e.g., generic MSE items for synthesis/composition)
- processed produced fewer references but somewhat more targeted links to primary-source framing in key nodes

## Interpretation (pilot only)

- There is **weak positive signal** for processed corpus hints on this task: fewer `gap` outcomes (2 vs 3) and slight node-level advantage (2 wins vs 1).
- The effect size is small and within plausible model stochastic variation for a single run.
- Benefit appears more in **focus/precision** than in raw reference count.

## Limitations

- Single problem (P3), single run per arm.
- No fixed random seed control for model outputs.
- Prompt text still says “processed data” even in raw arm; path changed but wording was not arm-specific.

## Next step to validate

Run the same A/B on `p7` and `p8`, plus one repeat per arm, then aggregate:

- gap rate
- per-node status win/loss
- reference relevance (human-judged)
- synthesis-node quality

This would make the conclusion statistically and behaviorally more reliable.
