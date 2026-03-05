# Codex Handoff: ASE Pipeline (Artificial Stack Exchange)

Date: 2026-03-05
Agent: Claude → Codex
Type: Infrastructure integration + overnight generation

## What This Is

A pipeline that generates synthetic math QA threads in **hypergraph-native
format** — born with typed nodes and edges, not retro-fitted. The pipeline
retrieves relevant threads from the 900K Math.SE/MathOverflow corpus via
FAISS + BGE embeddings, uses them as exemplars and component sources, then
asks an LLM to generate new QA pairs that fill identified proof gaps.

The generated threads feed back into the retrieval corpus, creating a
growing synthetic knowledge base that improves proof-polish runs.

**Current state**: Pipeline tested end-to-end on P7 curriculum questions
(5/5 successful, valid hypergraph JSON output). Gap specs exist for P2,
P3, P7, P8. Tickle integration is wired. No overnight runs have been
executed yet.

## Architecture

```
                           ┌─────────────────────┐
                           │  Wiring Diagrams     │
                           │  problem{N}-wiring   │
                           └────────┬────────────┘
                                    │
┌──────────────────┐    ┌───────────▼───────────┐
│  Real Corpus     │    │  retrieve-proof-       │
│  math-processed  │───▶│  context.py            │
│  mo-processed    │    │  (BGE + FAISS 4-stage) │
│  ase store       │    └───────────┬────────────┘
└──────────────────┘                │
                                    ▼
                        ┌───────────────────────┐
                        │  generate-synthetic-   │
                        │  qa.py                 │
                        │  (gap analysis +       │
                        │   component extraction │
                        │   + prompt building)   │
                        └───────────┬────────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      ▼             ▼              ▼
               ┌────────────┐ ┌──────────┐  ┌──────────┐
               │ Direct API │ │  Tickle  │  │  Codex   │
               │ (test)     │ │  batch   │  │  CLI     │
               └─────┬──────┘ └────┬─────┘  └────┬─────┘
                     │             │              │
                     └─────────────┼──────────────┘
                                   ▼
                        ┌───────────────────────┐
                        │  ase-store.py          │
                        │  ingest → reindex      │
                        │  (BGE + GNN + FAISS)   │
                        └───────────┬────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  ASE corpus     │
                           │  ~/code/storage │
                           │  /ase/          │
                           └────────────────┘
```

## Files

### Scripts (futon6)

| File | Purpose | Status |
|------|---------|--------|
| `scripts/retrieve-proof-context.py` | 4-stage retrieval: tag filter → keyword → BGE rerank → FAISS structural expansion. Queries math, MO, and ASE corpora. | Working |
| `scripts/generate-synthetic-qa.py` | Gap analysis per proof node. Extracts question components from retrieved threads. Builds generation prompts. | Working (dry-run tested for P2, P3, P7, P8) |
| `scripts/prepare-ase-tickle-queue.py` | Converts generation prompts to Tickle work queue format (`data/ase-queue/`). | Working |
| `scripts/ase-store.py` | File-based ASE store: ingest, reindex (BGE + GNN + FAISS), stats, query. | Working (no data ingested yet) |
| `scripts/test-ase-generate.py` | End-to-end test without Tickle. Calls `claude -p` or `codex -q` via CLI. Includes P7 curriculum questions. `--backend claude\|codex` flag. | Working (5/5 passed) |
| `scripts/run-ase-hotspot-loop.py` | Chains: hotspot manifest → retrieval → generation → enriched stepper. | Written, untested |

### Clojure (futon3c)

| File | Purpose | Status |
|------|---------|--------|
| `src/futon3c/agents/ase_work_queue.clj` | ASE entity loader, prompt builder, evidence emission. Follows `tickle_work_queue.clj` pattern. | Written, not compiled |
| `dev/futon3c/dev.clj` (lines ~1823-2000) | REPL helpers: `ase-progress!`, `run-ase-entry!`, `run-ase-batch!`. Added to `status` output. | Written, not compiled |

### Data

| Path | Contents |
|------|----------|
| `data/first-proof/problem7-corpus-context.jsonl` | Pre-retrieved context: 8 nodes × 8 threads each (64 total) |
| `data/synthetic-qa/problem{N}-prompts.jsonl` | Generation prompts (P2: 8, P3: 8, P7: 16, P8: 8 = 40 total) |
| `data/ase-queue/entities.json` | 40 Tickle work items with prompts |
| `data/ase-queue/review-prompts.json` | Claude review prompts (5 criteria each) |
| `data/ase-queue/queue-manifest.json` | Metadata: problems, counts, estimated time |
| `data/ase-test/q{1-5}-result.json` | Test results from P7 curriculum run |
| `~/code/storage/ase/` | ASE store directory (empty, awaiting first ingest) |

## The Retrieval Pipeline (4 stages)

`retrieve-proof-context.py` finds relevant threads for each proof node:

1. **Tag filter**: Match proof node seed tags against thread pattern tags.
   Reduces 900K threads to ~10K candidates.

2. **Keyword match**: Score candidates by keyword overlap with gap
   description terms. Top 500 candidates advance.

3. **BGE embedding rerank**: Encode gap description with
   `BAAI/bge-large-en-v1.5` (1024-dim), compute cosine similarity against
   pre-computed thread embeddings (`storage/*/embeddings.npy`).
   Top 5 threads selected.

4. **FAISS structural expansion**: Map text-seed thread IDs into the GNN
   FAISS index (128-dim, 1M+ vectors at
   `storage/math-processed-gpu/structural-similarity-index.faiss`).
   Average their GNN vectors, search for structural neighbors.
   Top 3 structurally similar threads added.

Result: 8 threads per node (5 text + 3 structural), ~8 seconds total.

**Key detail**: The FAISS index is 128-dim GNN embeddings (Stage 9b),
NOT 1024-dim BGE text embeddings. You cannot query FAISS with a text
string directly. The pipeline uses text seeds to bridge into the GNN
space.

## Gap Specs (NODE_GAP_SPECS)

`generate-synthetic-qa.py` contains 20 gap specifications across 4
problems, derived from REVIEWER.md findings:

| Problem | Nodes | Gap Sources (from REVIEWER.md) |
|---------|-------|-------------------------------|
| P2 (Rankin-Selberg) | `p2-problem`, `p2-s3`, `p2-s3a`, `p2-s5` | Universal test vector unproved; restriction nonvanishing unjustified; ideal-to-monomial jump; conductor matching heuristic |
| P3 (Markov chains) | `p3-problem`, `p3-s1`, `p3-s4`, `p3-s6` | Star/non-star normalization bridge; t-geometric weighting justification; irreducibility claim; positivity of stationary weights |
| P7 (Manifold torsion) | `p7-problem` through `p7-s6` (8 nodes) | PD group formulation loose; normal map setup asserted; obstruction vanishing unsupported; Smith theory anti-obstruction only |
| P8 (Lagrangian smoothing) | `p8-problem`, `p8-s3`, `p8-s5`, `p8-s6` | Basis/nondegeneracy unjustified; surgery theorem outside smooth hypotheses; global patching not established |

Each gap spec contains:
- `topic`: mathematical area
- `gap`: specific deficiency to target
- `seed_tags`: for retrieval filtering

## Question Component Extraction

When FAISS-retrieved threads are available, `extract_question_components()`
pulls reusable fragments:

- **Sub-questions**: Sentences ending in `?`
- **LaTeX expressions**: `$...$` fragments
- **Proof tasks**: "show that" / "prove that" clauses
- **Conclusions**: "therefore" / "hence" sentences

These are formatted as "Question Components (from similar threads)" and
included in the generation prompt, so the LLM composes new questions by
recombining real fragments rather than generating from scratch.

## Test Results (P7 Curriculum)

Codex's 5-question gate for p7-problem was used as a test case. All 5
questions produced valid hypergraph-native JSON via gpt-4o:

**gpt-4o baseline** (initial test):

| Q# | Title | Nodes | Edges | Answer | Time |
|----|-------|-------|-------|--------|------|
| 1 | Quantifier lock | 10 | 10 | 935 chars | 8.7s |
| 2 | Implication vs equivalence lock | 13 | 13 | 482 chars | 7.8s |
| 3 | Trivial-case boundary | 10 | 10 | 753 chars | 7.6s |
| 4 | Obstruction stack declaration | 11 | 11 | 1660 chars | 10.2s |
| 5 | 2-torsion mechanism | 11 | 11 | 751 chars | 7.6s |

**Claude Opus 4.6** (Q1 rerun via `--backend claude`):

| Q# | Title | Nodes | Edges | Answer | Time |
|----|-------|-------|-------|--------|------|
| 1 | Quantifier lock | 16 | 23 | 3690 chars | 49.9s |

Claude produces significantly richer output: 8 term nodes (including
"Davis construction", "Smith theory", "surgery obstruction") vs 4 from
gpt-4o. Answer correctly identifies all three quantifier readings and
their implications. Use `--backend claude` (default) or `--backend codex`.

**Observations**:
- Hypergraph structure is consistently well-typed (post/term/expression/scope nodes; mention/surface/scope/discourse/iatc edges)
- LaTeX escape issue in JSON required `fix_latex_escapes()` workaround
- Results at `data/ase-test/q{1-5}-result.json`

## Hypergraph Output Schema

Each generated thread produces:

```json
{
  "thread_id": "ase-p7-curriculum-q1",
  "title": "...",
  "question": "... (LaTeX inline) ...",
  "answer": "... (rigorous, with LaTeX) ...",
  "tags": ["algebraic-topology", "manifolds", ...],
  "nodes": [
    {"id": "n1", "type": "post", "subtype": "question"},
    {"id": "n2", "type": "post", "subtype": "answer"},
    {"id": "n3", "type": "term", "subtype": "uniform lattice"},
    {"id": "n8", "type": "expression", "subtype": "latex"},
    {"id": "n10", "type": "scope", "subtype": "conditional"}
  ],
  "edges": [
    {"type": "mention", "ends": ["n1", "n3"]},
    {"type": "surface", "ends": ["n1", "n8"]},
    {"type": "scope", "ends": ["n1", "n10"]},
    {"type": "discourse", "ends": ["n1", "n2"]},
    {"type": "iatc", "ends": ["n1", "n2"], "attrs": {"performative": "clarify"}}
  ]
}
```

Node types: `post`, `term`, `expression`, `scope`
Edge types: `mention`, `surface`, `scope`, `discourse`, `iatc`, `categorical`

## What Codex Should Do

### Task 1: Run generation for all 4 problems (HIGH)

For each problem with corpus context, generate synthetic QA threads:

```bash
# Step 1: Retrieve corpus context (if not already present)
python3 scripts/retrieve-proof-context.py --problem 2
python3 scripts/retrieve-proof-context.py --problem 3
# P7 already has context
python3 scripts/retrieve-proof-context.py --problem 8

# Step 2: Generate prompts (dry-run to verify)
python3 scripts/generate-synthetic-qa.py --problem 2 --dry-run
python3 scripts/generate-synthetic-qa.py --problem 3 --dry-run
python3 scripts/generate-synthetic-qa.py --problem 8 --dry-run

# Step 3: Run generation via test script (or direct API)
.venv/bin/python3 scripts/test-ase-generate.py --problem 7 --question all
```

For problems without pre-built curriculum questions, modify
`test-ase-generate.py` to read from `data/synthetic-qa/problem{N}-prompts.jsonl`
and feed each prompt to the API.

**Success criteria**: ≥80% valid JSON parse rate. All outputs have
≥4 term nodes, ≥1 expression node, ≥1 scope node.

### Task 2: Ingest results into ASE store (MEDIUM)

After generation, convert results to JSONL and ingest:

```bash
# Convert test results to JSONL
python3 -c "
import json, glob
with open('data/ase-test/all-results.jsonl', 'w') as out:
    for f in sorted(glob.glob('data/ase-test/q*-result.json')):
        obj = json.load(open(f))
        if 'thread_id' in obj:
            out.write(json.dumps(obj) + '\n')
"

# Ingest
python3 scripts/ase-store.py ingest data/ase-test/all-results.jsonl

# Reindex (requires sentence-transformers + torch in .venv)
python3 scripts/ase-store.py reindex

# Verify
python3 scripts/ase-store.py stats
```

**Success criteria**: Store reports correct entity count. Embeddings
shape matches entity count. FAISS index built if GNN model available.

### Task 3: Extend gap specs to P1, P4-P6, P9-P10 (MEDIUM)

All 10 problems have wiring diagrams. Only P2, P3, P7, P8 have gap specs.
Add `NODE_GAP_SPECS` entries for the remaining problems using REVIEWER.md
findings:

- P1: measure equivalence (3 gaps: equivalence chain, Young's inequality, Wick expansion)
- P4: polynomial superadditivity (3 gaps: headline overstatement, derivative formulas, degree-2 convolution)
- P5: reverse implication (3 gaps: verbatim localization, subgroup reduction, filtration)
- P6: graph Laplacian (2 gaps: existential claim, concentration)
- P9: tensor rank detection (3 gaps: converse genericity, Hadamard rank, iff statement)
- P10: PCG convergence (2 gaps: preconditioner quality, asymptotic cost)

Source: `REVIEWER.md` gap descriptions + `data/first-proof/problem{N}-wiring.json` node IDs.

### Task 4: Add prompt-file execution mode to test script (HIGH)

`test-ase-generate.py` currently only has hardcoded P7 curriculum questions.
Add a `--prompts-file` flag that reads from a JSONL file (as produced by
`generate-synthetic-qa.py`) and runs each prompt through the API:

```bash
.venv/bin/python3 scripts/test-ase-generate.py \
    --prompts-file data/synthetic-qa/problem3-prompts.jsonl \
    --timeout 120
```

This bridges the gap between prompt generation and API execution without
requiring Tickle.

### Task 5: Validate hotspot stepper integration (LOW)

`run-ase-hotspot-loop.py` chains hotspot stepper → retrieval → generation.
Verify it works with existing P7 hotspot data at
`data/first-proof/stepper/problem7-hotspot-stepper-*.json`.

## Key Gotchas

1. **FAISS space mismatch**: The FAISS index is 128-dim GNN, not 1024-dim
   BGE. You cannot query it with text embeddings. The retrieval script
   handles this by using text seeds to bridge.

2. **LaTeX in JSON**: LLMs produce `\Gamma` not `\\Gamma` in JSON strings.
   `fix_latex_escapes()` in `test-ase-generate.py` handles this. Any new
   API integration must apply the same fix.

3. **Nested Claude sessions**: `claude -p` cannot run inside another Claude
   session (CLAUDECODE env var conflict). Use OpenAI API or run from a
   separate terminal.

4. **Problem-prefix filtering**: `generate-synthetic-qa.py` filters
   `NODE_GAP_SPECS` by problem prefix from the wiring file's `thread_id`
   (e.g., `first-proof-p3` → only `p3-*` specs). If you add gap specs for
   a new problem, the `thread_id` in the wiring JSON must match.

5. **Corpus context optional in dry-run**: `generate-synthetic-qa.py`
   proceeds without corpus context in `--dry-run` mode (prompts will lack
   exemplar threads). For real generation, run `retrieve-proof-context.py`
   first.

## Priority

Task 4 (prompt-file mode) >> Task 1 (run generation) >> Task 2 (ingest)
>> Task 3 (extend gaps) >> Task 5 (hotspot integration)

Task 4 unblocks Task 1 for all problems beyond P7.
