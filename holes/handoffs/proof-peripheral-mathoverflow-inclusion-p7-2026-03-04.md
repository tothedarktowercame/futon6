# Proof Peripheral Experiment: Include MathOverflow (P7, 2026-03-04)

User request: re-run experiment including MathOverflow corpus, since research-heavy proofs likely benefit from MO.

## Runner and controls

Runner:

- `scripts/run-proof-polish-codex-p7.py`

Controls held fixed:

- model: `gpt-5.3-codex`
- reasoning effort: `medium`
- web search: `disabled`
- same wiring/solution/schema/prompt template

## Arms attempted

1. **MO-only**
   - `--math-se-dir se-data/mathoverflow.net`
   - output: `data/first-proof/problem7-codex-results-exp-mo-only.jsonl`
2. **Processed+MO mixed hint**
   - `--math-se-dir se-data`
   - output: `data/first-proof/problem7-codex-results-exp-processed-plus-mo.jsonl`
3. **Processed-only**
   - `--math-se-dir se-data/math-processed`
   - repeated severe latency/stall; no reliable completed run artifact in this session

## Completed-arm results

### MO-only

- rows: 9
- status: `verified=4`, `plausible=2`, `gap=3`, `error=0`
- references: 35
  - `mathoverflow.net`: 30
  - `math.stackexchange.com`: 5

### Processed+MO mixed

- rows: 9
- status: `verified=3`, `plausible=4`, `gap=2`, `error=0`
- references: 35
  - `mathoverflow.net`: 28
  - `math.stackexchange.com`: 7

## Node-level deltas (Processed+MO minus MO-only)

- better: `p7-problem`, `p7-s6`
- worse: `p7-s2`, `p7-s3a`
- tie: `p7-s1`, `p7-s3`, `p7-s4`, `p7-s5`, `p7-synthesis`

## Reading

- Including MO works operationally and yields high MO reference usage in both completed arms.
- Mixed corpus hint reduced total `gap` count (2 vs 3) but also reduced `verified` count (3 vs 4), so net quality is mixed rather than clearly dominant.
- Weighted status score (`verified=2, plausible=1, gap=0`) is tied across completed arms.

## Runner update applied

For cleaner experiments, p7 runner now supports explicit controls:

- `--reasoning-effort`
- `--web-search`

and uses neutral corpus wording in prompts:

- “Use local corpus data if available under: …”

## Recommended next step

To isolate the MO effect cleanly, add explicit multi-path corpus hints in prompt text (separate lines for processed and MO dirs) and rerun with 2 replicates per arm:

- processed-only
- MO-only
- processed+MO (explicit dual-path)

Then compare per-node majority status and reference relevance quality.
