# Superpod 1a: Post-Run Pipeline Improvements

**Date**: 2026-03-04
**Scope**: Improvements to the futon6 processing pipeline based on analysis of
the Feb 27, 2026 superpod run (Math.SE 805K QA pairs, MathOverflow 95K QA pairs).

These changes prepare the pipeline for a second run with higher quality gates,
fix known parser gaps, and add validation infrastructure that was missing from
the first run.

---

## 1. Test Infrastructure (A1)

**Problem**: Tests failed with `ModuleNotFoundError: No module named 'futon6'`.

**Fix**:
- Created `.venv/` with `pip install -e ".[dev]"`
- Fixed `test_graph_embed.py`: `train()` returns 3 values `(model, embeddings,
  stats)` — tests expected 2
- Added `tests/conftest.py` to skip PlanetMath-dependent tests when fixture
  data is absent (24 tests)

**Result**: 265 passed, 24 skipped, 0 failures.

---

## 2. LaTeX Parser Improvements (A7 → fix)

**File**: `src/futon6/latex_sexp.py`

**Problem**: 1% of LaTeX fragments fail to parse (332K/33.4M for Math.SE,
44K/5.1M for MO). Failure analysis showed most are trivial patterns:

| Cause | % of failures |
|-------|--------------|
| `*` / `#` as postfix (dual, adjoint, sharp) | ~62% (no known construct) |
| `\rm`, `\bf`, `\it` old-style font switches | ~8% |
| Multi-letter unrecognized commands | ~34% |
| `\frac12` shorthand (digits not braced) | scattered |

**Changes**:
1. **Postfix `*` and `#`**: Added `star` and `hash` token types to the lexer
   regex and postfix handling in `parse_unary()`. `R^*` → `(star R)`,
   `f^#` → `(sharp f)`.

2. **Old-style font commands** (`\rm`, `\bf`, `\it`, `\sf`): Added recognition
   in `parse_command()`. Handles both `{\rm text}` and `\rm text` forms.
   Content is emitted as a plain `Atom`.

3. **Spacing commands** (`\,`, `\;`, `\:`, `\!`, `\ `): Added
   `(?P<spacing_cmd>\\[,;:! ])` to the tokenizer regex. These are silently
   consumed (spacing is not semantically meaningful in s-expression output).

4. **`\frac12` shorthand**: When `\frac` is followed by a bare 2-digit number
   instead of braced groups, split into numerator/denominator:
   `\frac12` → `(/ 1 2)`.

**Expected impact**: Eliminates ~70% of parse failures. Remaining failures are
multi-letter commands that would need a command dictionary.

**Tests**: 34/34 LaTeX tests pass, no regressions.

---

## 3. Stage 6 Health Gate and Backend Selection

**File**: `scripts/superpod-job.py`

**Problem**: The first run's Stage 6 (reverse morphogenesis) achieved only 9.6%
parse success with local Llama-3-8B. The health gate was set at 10%, so it
barely passed. Schema-constrained decoding via API achieves 100% on a 50-row
sample.

**Changes**:
1. **Raised health gate**: `--gate-stage6-parse-rate-min` default 0.10 → 0.80.
   A second run must achieve ≥80% parse success or fail loudly.

2. **Added `--stage6-backend`** flag with choices `local-llm` (default),
   `codex`, `gemini`. Recorded in the run manifest. This makes it explicit
   when an API backend is used for schema-constrained decoding.

---

## 4. Confidence Tier in Stage 7 Output

**File**: `src/futon6/thread_performatives.py`

**Problem**: Stage 7 produces two kinds of edges — IATC performatives
(clarify, assert, query, etc.) at 66-72% alignment, and categorical assertions
at 3-5% consistency. Downstream consumers had no way to distinguish them.

**Change**: `diagram_to_dict()` now includes `confidence_tier` on each edge:
- **Tier 1**: IATC performatives and well-grounded relational edges
  (`assert`, `comment-on`, `responds-to`). Usable for decision-making.
- **Tier 2**: Category-theoretic structural assertions. Experimental only.

The tier is computed at serialization time via `_edge_confidence_tier()`, so
existing wiring logic is unaffected.

**Tests**: 46/46 wiring tests pass.

---

## 5. Post-Stage-10 Review Pair Generation

**File**: `scripts/superpod-job.py`

**Problem**: The first run produced no retrieval validation artifacts. The
review-50 pairs used in the original evaluation were sampled from a broader set
than what produced hypergraphs, so ~69% of thread IDs didn't exist in the
FAISS index.

**Change**: Added `--review-sample-size N` flag (default 0 = skip). When
nonzero, a post-Stage-10 step samples pairs from the FAISS structural
similarity index at three tiers:

| Tier | Source | Purpose |
|------|--------|---------|
| High | Top-5 neighbors | Should be semantically related |
| Medium | Rank 50-150 | Moderate similarity |
| Low | Random pairs | Negative baseline |

Pairs are enriched with entity metadata (tags, title, score) for downstream
judgment (human or LLM).

---

## 6. Analysis Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/audit-graph-embeddings.py` | Validates Stage 9b embeddings: collapse check, cosine distribution, per-dimension variance, nearest-neighbor analysis |
| `scripts/generate-review-pairs.py` | Samples tier-stratified pairs from FAISS index with tag/title heuristic judgments |
| `scripts/analyze-latex-failures.py` | Streams expression-surfaces.json, categorizes parse failures by construct family and length |
| `scripts/add-confidence-tiers.py` | Post-processes thread-wiring-ct.jsonl to add confidence tiers (standalone, for first-run data) |

---

## 7. Key Findings from First-Run Analysis

### Stage 9b embeddings are valid but the metric is inflated
- No embedding collapse: all 128 dimensions active, healthy per-dim variance
- Random-pair cosine similarity centered near 0 (median ≈ -0.001)
- Top-k neighbor similarities (0.62-0.79) well above random baseline
- 99.8% Acc@1 reflects easy contrastive negatives (graph dropout creates
  near-identical positive views), not genuinely perfect retrieval

### Retrieval precision is modest
Heuristic-judged review-100 pairs:

| Tier | Math.SE Strict P | MO Strict P |
|------|-----------------|-------------|
| High (top-5) | 36.4% | 27.3% |
| Medium (50-150) | 18.2% | 15.2% |
| Low (random) | 0.0% | 2.9% |

Tier separation confirms embeddings discriminate, but absolute precision is
modest. An LLM judgment pass reading full thread bodies would be more
informative than the tag/title heuristic.

### LaTeX failures are dominated by trivial patterns
85% of failures are <20 characters. The parser fixes above address the
dominant failure modes (`*`/`#` postfix, `\rm` font switches, spacing
commands). Remaining failures are mostly unrecognized multi-letter commands.

---

## What's Next

See `KNOWN_LIMITATIONS.md` for the full gap inventory. Key next steps:

1. **B1**: Full-corpus Stage 6 rerun with schema-constrained decoding
   (Gemini preferred — 49/50 vs 21/50 "good" on form dimension in 50-row
   sample). Needs cost estimate first.
2. **C3**: After Stage 6 is fixed, cascade reprocessing of Stages 7→9a→9b→10.
3. **A8**: O-0 classical baseline (Corneli 2014 learning-event detection).
4. LLM-judged retrieval evaluation on review-100 pairs.
