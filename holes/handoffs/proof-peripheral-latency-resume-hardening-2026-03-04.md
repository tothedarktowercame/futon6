# Proof Peripheral Latency + Resumability Hardening (2026-03-04)

Problem observed: long/variable per-node latency in `codex exec` made full `p7` experiments brittle. If a run was interrupted and restarted, previous script behavior (`output` opened in write mode) could overwrite partial progress.

## What changed

Updated `scripts/run-proof-polish-codex-p7.py` with:

1. **Resumable execution**
- `--resume`
- reads existing output JSONL, collects completed `node_id`s, skips them
- appends new rows instead of truncating output

2. **Latency containment**
- `--timeout-seconds` (per-node hard timeout)
- `--max-retries` (retry count per node)
- timeout rows are recorded as `parse_error` with `timed_out=true`

3. **Durability of streamed results**
- per-row `flush()` + `os.fsync()` after write
- ensures completed rows survive crashes/stalls

4. **Node-level observability**
- output rows now include:
  - `attempts`
  - `elapsed_seconds`
  - `timed_out`
- stdout progress includes attempts/elapsed/timeout

5. **Dual-path corpus hints**
- added repeatable `--extra-corpus-dir`
- prompt now includes explicit multi-path corpus hints (not just a single path)

6. **Determinism controls already present**
- `--reasoning-effort`
- `--web-search`

## Smoke test (resumability)

- ran with `--limit 1 --timeout-seconds 1 --max-retries 1`
- got timeout row written with metadata
- reran with `--resume`; script detected completed node and exited with “Nothing to do.”

## Diagnosis of earlier “made progress but file missing rows” behavior

Most likely cause is restart/overwrite semantics, not inability to stream:

- old code always used `open(output, "w")`
- any restart on same output path truncates prior partial results
- with high-latency calls, this can look like “we made progress but output didn’t persist”

New resume+append+fsync logic directly addresses this.

## Recommended 3-arm rerun protocol

Use separate output files + resume + bounded timeout.

Example (single replicate):

```bash
cd /home/joe/code/futon6

# processed-only
python3 scripts/run-proof-polish-codex-p7.py \
  --model gpt-5.3-codex --reasoning-effort medium --web-search disabled \
  --math-se-dir se-data/math-processed \
  --output data/first-proof/problem7-codex-results-exp-processed.jsonl \
  --prompts-out data/first-proof/problem7-codex-prompts-exp-processed.jsonl \
  --timeout-seconds 300 --max-retries 2 --resume

# MO-only
python3 scripts/run-proof-polish-codex-p7.py \
  --model gpt-5.3-codex --reasoning-effort medium --web-search disabled \
  --math-se-dir se-data/mathoverflow.net \
  --output data/first-proof/problem7-codex-results-exp-mo-only.jsonl \
  --prompts-out data/first-proof/problem7-codex-prompts-exp-mo-only.jsonl \
  --timeout-seconds 300 --max-retries 2 --resume

# processed + MO explicit dual hint
python3 scripts/run-proof-polish-codex-p7.py \
  --model gpt-5.3-codex --reasoning-effort medium --web-search disabled \
  --math-se-dir se-data/math-processed \
  --extra-corpus-dir se-data/mathoverflow.net \
  --output data/first-proof/problem7-codex-results-exp-processed-plus-mo.jsonl \
  --prompts-out data/first-proof/problem7-codex-prompts-exp-processed-plus-mo.jsonl \
  --timeout-seconds 300 --max-retries 2 --resume
```

For replicates, write to `...-rep1.jsonl`, `...-rep2.jsonl`, etc.

## Note

Equivalent resumability hardening has **not yet** been ported to `run-proof-polish-codex-p3.py` / `p8.py` in this patch. If desired, mirror the same pattern there next.
