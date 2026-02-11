# Stage 6 Reverse Morphogenesis (First 50) — Reproduction Notes

Date: 2026-02-11
Repo: `futon6`

## Task
Process the first 50 lines from Stage 6 prompt JSONL, run each `prompt` through an LLM, parse response JSON, and append records as:

- valid parse: `{entity_id, question_id, ...response}`
- invalid parse: `{entity_id, question_id, raw, parse_error: true}`

Also report quality counts for `quality.form` (象), `quality.salience` (香), and `quality.arrow` (←).

## Input / Output
- Requested input path from task text: `futon6/moist-prompts/stage6-reverse-morphogenesis.jsonl`
- Actual existing input path used: `futon6/se-data/math-processed/moist-prompts/stage6-reverse-morphogenesis.jsonl`
- Gemini output file (repo): `futon6/se-data/math-processed/moist-prompts/stage6-results.jsonl`
- Codex output file (repo): `futon6/se-data/math-processed/moist-prompts/stage6-results-codex.jsonl`
- Output line count (each): `50`

## Model/Provider Reproduction

### Attempt 1 (failed)
- Provider/API: OpenAI `chat/completions`
- Model: `gpt-4.1-mini`
- Auth: `OPENAI_API_KEY` from environment
- Failure: HTTP `429` with `insufficient_quota`
- Result: run aborted before completion

### Attempt 2 (Gemini, initial successful run)
- Provider/API: Google Generative Language API (`generateContent`)
- Model: `gemini-2.0-flash`
- Auth: `GEMINI_API_KEY` from environment
- Generation settings:
  - `temperature: 0`
  - `response_mime_type: application/json`
- Retry behavior: exponential backoff for transient HTTP errors (`429/5xx`), max 6 attempts per prompt
- Execution mode: sequential over first 50 lines
- Result: produced output, but 19 entries were marked parse errors due to top-level shape mismatch (mostly arrays)

### Attempt 3 (Gemini rerun with strict object schema)
- Provider/API: Google Generative Language API (`generateContent`)
- Model: `gemini-2.0-flash`
- Auth: `GEMINI_API_KEY` from environment
- Generation settings:
  - `temperature: 0`
  - `response_mime_type: application/json`
  - `response_schema`: strict object schema requiring
    - `xiang_form`, `xiang_salience`, `arrow_constraint`, `quality`, `situation_S`, `roundtrip_check`
    - `quality.{form,salience,arrow}` in `good|weak|broken`
- Result: parse errors eliminated (`0/50`)

### Attempt 4 (Codex comparison run)
- Provider: Codex CLI (`codex exec`)
- Model: `gpt-5.3-codex` (from local Codex config default)
- Script: `scripts/run-stage6-codex.py`
- Key run behavior:
  - one Codex call per input prompt (first 50)
  - enforced JSON object output schema via `codex exec --output-schema`
  - output normalization identical to Gemini run
- Result: parse errors eliminated (`0/50`)

## Parser / Normalization Rules
- Parse each prompt response with `json.loads`.
- Accept only top-level JSON object responses as valid for expansion into output row.
- Any parse failure (or non-object JSON) recorded as:
  - `{"raw": "<original model text>", "parse_error": true}`
- Always preserve source IDs in output:
  - `entity_id`
  - `question_id`

## Commands Used

- Gemini schema rerun:
  - `python3 ...` (internal script execution with Gemini `generateContent`, `response_schema` enabled)
- Codex run:
  - `python3 scripts/run-stage6-codex.py --limit 50 --output se-data/math-processed/moist-prompts/stage6-results-codex.jsonl`

## Run Summary (first 50)

### Gemini (strict schema) — `stage6-results.jsonl`
- Processed: `50`
- Valid top-level JSON objects: `50`
- Parse errors: `0`
- 象 (`quality.form`): `good=49, weak=1, broken=0`
- 香 (`quality.salience`): `good=48, weak=2, broken=0`
- ← (`quality.arrow`): `good=44, weak=6, broken=0`

### Codex (`gpt-5.3-codex`) — `stage6-results-codex.jsonl`
- Processed: `50`
- Valid top-level JSON objects: `50`
- Parse errors: `0`
- 象 (`quality.form`): `good=21, weak=29, broken=0`
- 香 (`quality.salience`): `good=44, weak=6, broken=0`
- ← (`quality.arrow`): `good=21, weak=28, broken=1`

## Notes
- Earlier parse errors were primarily shape mismatches (arrays instead of a single object), not malformed JSON.
- Enforcing schema at generation time removed parse errors for both Gemini and Codex runs.
