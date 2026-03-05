# Superpod Rerun Notes for Rob

**Date**: 2026-03-04
**Author**: Joe + Claude (session analysis)

## What Changed and Why

### 1. Stage 6 Prompt Tightened (build_reverse_morphogenesis_prompt)

**File**: `scripts/superpod-job.py`, line ~3029

The original prompt produced 9.6% parse success with local Llama-3-8B because
the model would write prose analysis *then* append JSON, often truncated.

The revised prompt:
- Puts "Return ONLY a valid JSON object" at the very top
- Lists schema keys inline with word budgets (≤40 words per field)
- Removes the verbose 4-step task description (quality criteria kept as bullets)
- Moves Q&A content to the end (after all instructions)

**Validation**: Tested on n=13 random samples via Claude Sonnet (subscription CLI):
- Original prompt: 69% parse success (9/13)
- Tightened prompt: **100% parse success** (13/13)
- Tightened also produced better quality ratings and more concise outputs

**What Rob needs to do**:
1. Regenerate JSONL prompts: run `superpod-job.py` with `--moist-run` to produce
   new `stage6-reverse-morphogenesis.jsonl` files (CPU-only, fast)
2. For the superpod run: use `--stage6-backend local-llm` with vLLM's
   `guided_json` or `--guided-json-schema` for schema-constrained decoding.
   The schema is `RESPONSE_SCHEMA` in `run-stage6-codex.py` (lines 43-71).
3. Health gate is now 80% (was 10%). If parse rate drops below 80%, the job
   will fail loudly rather than silently producing garbage.

### 2. Stage 6 Health Gate Raised

`--gate-stage6-parse-rate-min` default changed from 0.10 to 0.80.

### 3. Stage 6 Backend Flag

New `--stage6-backend` flag: `local-llm` (default), `codex`, `gemini`.
Recorded in run manifest for provenance.

### 4. Confidence Tier Baked into Stage 7

`thread_performatives.py:diagram_to_dict()` now emits `confidence_tier` on
each edge:
- Tier 1: IATC performatives (clarify, assert, query, etc.) — 66-72% aligned, usable
- Tier 2: Category-theoretic assertions — 3-5% consistent, experimental only

Downstream consumers should check `confidence_tier` before trusting an edge.

### 5. LaTeX Parser Fixes

`latex_sexp.py` now handles:
- `*` and `#` as postfix operators (dual/adjoint/sharp)
- `\rm`, `\bf`, `\it`, `\sf` old-style font switches
- `\,`, `\;`, `\:`, `\!`, `\ ` spacing commands
- `\frac12` shorthand (unbraced 2-digit numerator/denominator)

Expected to eliminate ~70% of the 1% parse failure rate.

### 6. Post-Stage-10 Review Pair Generation

New `--review-sample-size N` flag. When nonzero, samples retrieval validation
pairs from the FAISS index at high/medium/low similarity tiers after Stage 10.

### 7. Cascade After Stage 6 Fix

Once Stage 6 produces ≥80% parse success, Stages 7→9a→9b→10 should be
re-derived (they depend on Stage 6 structure). Stages 1-5, 8 are untouched.

## Test Status

265 passed, 24 skipped (skips = missing PlanetMath fixture data), 0 failures.

## Reference

Full details: `superpod-1a-technote.md`
Known gaps: `KNOWN_LIMITATIONS.md`
Validation script: `scripts/validate-stage6-local.py`
