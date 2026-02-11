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
- Output file (repo): `futon6/se-data/math-processed/moist-prompts/stage6-results.jsonl`
- Output line count: `50`

## Model/Provider Reproduction

### Attempt 1 (failed)
- Provider/API: OpenAI `chat/completions`
- Model: `gpt-4.1-mini`
- Auth: `OPENAI_API_KEY` from environment
- Failure: HTTP `429` with `insufficient_quota`
- Result: run aborted before completion

### Attempt 2 (successful)
- Provider/API: Google Generative Language API (`generateContent`)
- Model: `gemini-2.0-flash`
- Auth: `GEMINI_API_KEY` from environment
- Generation settings:
  - `temperature: 0`
  - `response_mime_type: application/json`
- Retry behavior: exponential backoff for transient HTTP errors (`429/5xx`), max 6 attempts per prompt
- Execution mode: sequential over first 50 lines

## Parser / Normalization Rules
- Parse each prompt response with `json.loads`.
- Accept only top-level JSON object responses as valid for expansion into output row.
- Any parse failure (or non-object JSON) recorded as:
  - `{"raw": "<original model text>", "parse_error": true}`
- Always preserve source IDs in output:
  - `entity_id`
  - `question_id`

## Run Summary (first 50)
- Processed: `50`
- Valid top-level JSON objects: `31`
- Parse errors: `19`

Quality counts from valid parsed objects:
- 象 (`quality.form`): `good=30, weak=1, broken=0`
- 香 (`quality.salience`): `good=30, weak=1, broken=0`
- ← (`quality.arrow`): `good=28, weak=3, broken=0`

## Notes on Parse Errors
Most parse errors came from model outputs that were valid JSON but with the wrong top-level shape (for example, an array of objects instead of a single object). Per task rules, these are stored in `raw` with `parse_error: true`.
