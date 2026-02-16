# Superpod Handoff (Literate Runbook)

Date: February 16, 2026
Owner: Joe Corneli
Operator: Rob

This document is the source of truth for the superpod handoff. It is a
literate program: the shell scripts that execute the run are embedded in the
prose below. Run `entangled tangle` at the repo root to extract them.

## 1. Run Location

Run on the Superpod execution host after cloning `futon6`.

```bash
git clone <futon6-repo-url>
cd futon6
```

## 2. Single Command (required path)

```bash
bash scripts/handoff-superpod-all.sh
```

What it does (with step-by-step progress and hard fail gates):
- bootstrap inputs
- sanity tests
- smoke run + verification
- CPU baseline runs + verification
- required GPU backfill runs + verification
- packaging

## 3. Fast Verification Mode

```bash
bash scripts/handoff-superpod-all.sh --smoke-only
```

## 4. The Orchestrator

The orchestrator is `scripts/handoff-superpod-all.sh`. It assembles all the
pieces below into a single sequential pipeline. Each section that follows
defines one named chunk; together they compose the complete script.

``` {.bash file=scripts/handoff-superpod-all.sh}
<<bash-strict>>

<<all-header-comment>>

<<all-root-and-args>>

<<all-utility-functions>>

<<all-verify-run-dir>>

<<all-run-smoke>>

<<all-package-outputs>>

<<all-orchestration>>
```

### 4.1 Strict mode

Every script in this handoff starts the same way. `set -euo pipefail` means:
any failing command aborts the script, undefined variables are errors, and
pipe failures propagate. For a multi-hour batch job this is non-negotiable —
silent failures waste the entire run.

``` {.bash #bash-strict}
#!/usr/bin/env bash
set -euo pipefail
```

### 4.2 Usage documentation

``` {.bash #all-header-comment}
# Single-command Superpod handoff runner.
# Default behavior:
#   1) bootstrap inputs
#   2) sanity tests
#   3) smoke run + verification
#   4) full CPU runs + verification
#   5) required GPU backfill + verification
#   6) package outputs
#
# Options:
#   --smoke-only       stop after smoke run + verification
#   --skip-bootstrap   do not run bootstrap script
#   --skip-tests       do not run pytest sanity checks
```

### 4.3 Argument parsing

The script locates the repo root relative to its own path (so it works
regardless of where you invoke it), then parses three optional flags.
`--smoke-only` is the fast confidence check. `--skip-bootstrap` and
`--skip-tests` exist for reruns where the data is already in place.

``` {.bash #all-root-and-args}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_ONLY=0
SKIP_BOOTSTRAP=0
SKIP_TESTS=0

for arg in "$@"; do
  case "$arg" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--smoke-only] [--skip-bootstrap] [--skip-tests]"
      exit 1
      ;;
  esac
done
```

### 4.4 Progress reporting and fail gates

Three small utilities. `step` prints a numbered progress banner so you can
watch the run and know where it is. `fail` prints to stderr and exits —
every verification failure routes through here. `assert_file` is the
simplest gate: does this file exist? If not, something went wrong upstream.

``` {.bash #all-utility-functions}
STEP=0
step() {
  STEP=$((STEP + 1))
  echo
  echo "==> [all][step ${STEP}] $*"
}

fail() {
  echo "[all] ERROR: $*" >&2
  exit 1
}

assert_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "missing file: $path"
}
```

### 4.5 Run directory verification

This is the most important safety mechanism. After each pipeline run
(smoke, CPU, GPU), we check that the output directory contains the three
required artifacts and that the manifest and verifier output are not vacuous.
The embedded Python script checks three invariants:

1. `stage7_stats.ct_backed` is true (the CT-backed wiring path was used)
2. `threads_processed > 0` (something was actually processed)
3. `edges_checked > 0` (the verifier found and checked real edges)

If any check fails, the entire run aborts. Better to fail loudly than
deliver empty tarballs.

``` {.bash #all-verify-run-dir}
verify_run_dir() {
  local dir="$1"
  assert_file "$dir/manifest.json"
  assert_file "$dir/thread-wiring-ct.json"
  assert_file "$dir/thread-wiring-ct-verification.json"

  python3 - "$dir" <<'PY'
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
manifest = json.loads((d / "manifest.json").read_text())
s7 = manifest.get("stage7_stats") or {}

if not s7.get("ct_backed", False):
    raise SystemExit(f"{d}: stage7_stats.ct_backed is false")
if int(s7.get("threads_processed", 0)) <= 0:
    raise SystemExit(f"{d}: stage7_stats.threads_processed <= 0")

ver = json.loads((d / "thread-wiring-ct-verification.json").read_text())
if isinstance(ver, dict):
    edges = int((ver.get("summary") or {}).get("edges_checked", 0))
elif isinstance(ver, list):
    edges = 0
    for item in ver:
        if isinstance(item, dict):
            edges += int((item.get("summary") or {}).get("edges_checked", 0))
else:
    edges = 0

if edges <= 0:
    raise SystemExit(f"{d}: verifier edges_checked <= 0")

print(f"{d}: ok (threads={s7.get('threads_processed')}, edges_checked={edges})")
PY
}
```

### 4.6 The smoke run

Before committing to the full corpus (hours of compute), we run the pipeline
on a 4-thread mini dataset bundled in the test fixtures. This exercises the
complete path — parse, term spotting, thread wiring, verification — in under a
minute. GPU stages are skipped (the smoke run tests the CPU pipeline only).

``` {.bash #all-run-smoke}
run_smoke() {
  local out_smoke="/tmp/superpod-rob-smoke-$(date +%s)"
  echo "[all] smoke output dir: $out_smoke"

  python3 scripts/superpod-job.py \
    tests/fixtures/se-mini/Posts.xml \
    --comments-xml tests/fixtures/se-mini/Comments.xml \
    --site math.stackexchange \
    --output-dir "$out_smoke" \
    --min-score 0 \
    --thread-limit 4 \
    --skip-embeddings \
    --skip-llm \
    --skip-clustering

  python3 scripts/ct-verifier.py verify \
    --wiring "$out_smoke/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$out_smoke/thread-wiring-ct-verification.json"

  verify_run_dir "$out_smoke"
}
```

### 4.7 Packaging deliverables

Four tarballs: CPU and GPU outputs for each of the two corpora. The CPU
tarballs list specific files (to avoid bundling intermediate artifacts);
the GPU tarballs take the whole directory.

``` {.bash #all-package-outputs}
package_outputs() {
  tar czf superpod-math-processed.tar.gz \
    math-processed/entities.json \
    math-processed/relations.json \
    math-processed/tags.json \
    math-processed/stats.json \
    math-processed/ner-terms.json \
    math-processed/scopes.json \
    math-processed/thread-wiring-ct.json \
    math-processed/thread-wiring-ct-verification.json \
    math-processed/manifest.json

  tar czf superpod-mo-processed.tar.gz \
    mo-processed/entities.json \
    mo-processed/relations.json \
    mo-processed/tags.json \
    mo-processed/stats.json \
    mo-processed/ner-terms.json \
    mo-processed/scopes.json \
    mo-processed/thread-wiring-ct.json \
    mo-processed/thread-wiring-ct-verification.json \
    mo-processed/manifest.json

  tar czf superpod-math-processed-gpu.tar.gz math-processed-gpu
  tar czf superpod-mo-processed-gpu.tar.gz mo-processed-gpu
}
```

### 4.8 Pipeline orchestration

This is the main flow. It runs top-to-bottom with hard fail gates between
stages. The sequence is: bootstrap, tests, smoke, CPU baseline (math.SE then
MO), GPU backfill, packaging. If `--smoke-only` was passed, it exits after
the smoke gate. Each CPU run is followed immediately by ct-verifier and
`verify_run_dir`.

``` {.bash #all-orchestration}
echo "[all] repo: $ROOT_DIR"
echo "[all] smoke_only=$SMOKE_ONLY skip_bootstrap=$SKIP_BOOTSTRAP skip_tests=$SKIP_TESTS"

if (( ! SKIP_BOOTSTRAP )); then
  step "bootstrap inputs"
  bash scripts/handoff-superpod-bootstrap.sh
else
  step "bootstrap skipped by flag"
fi

if (( ! SKIP_TESTS )); then
  step "sanity tests"
  PYTHONPATH=src pytest -q tests/test_superpod_job_smoke.py tests/test_ct_verifier.py
else
  step "sanity tests skipped by flag"
fi

step "smoke run + verification"
run_smoke

if (( SMOKE_ONLY )); then
  echo
  echo "[all] done (smoke-only)."
  exit 0
fi

step "CPU baseline run + verification (math.stackexchange)"
python3 scripts/superpod-job.py \
  ./se-data/math.stackexchange.com/Posts.xml \
  --comments-xml ./se-data/math.stackexchange.com/Comments.xml \
  --site math.stackexchange \
  --output-dir ./math-processed \
  --skip-embeddings \
  --skip-llm \
  --skip-clustering
python3 scripts/ct-verifier.py verify \
  --wiring ./math-processed/thread-wiring-ct.json \
  --reference data/nlab-ct-reference.json \
  --output ./math-processed/thread-wiring-ct-verification.json
verify_run_dir "./math-processed"

step "CPU baseline run + verification (mathoverflow)"
python3 scripts/superpod-job.py \
  ./se-data/mathoverflow.net/Posts.xml \
  --comments-xml ./se-data/mathoverflow.net/Comments.xml \
  --site mathoverflow.net \
  --output-dir ./mo-processed \
  --skip-embeddings \
  --skip-llm \
  --skip-clustering
python3 scripts/ct-verifier.py verify \
  --wiring ./mo-processed/thread-wiring-ct.json \
  --reference data/nlab-ct-reference.json \
  --output ./mo-processed/thread-wiring-ct-verification.json
verify_run_dir "./mo-processed"

step "required GPU backfill + verification"
bash scripts/handoff-superpod-gpu-backfill.sh both
verify_run_dir "./math-processed-gpu"
verify_run_dir "./mo-processed-gpu"

step "package outputs"
package_outputs

assert_file "superpod-math-processed.tar.gz"
assert_file "superpod-mo-processed.tar.gz"
assert_file "superpod-math-processed-gpu.tar.gz"
assert_file "superpod-mo-processed-gpu.tar.gz"

echo
echo "[all] complete. Deliver:"
echo "  superpod-math-processed.tar.gz"
echo "  superpod-mo-processed.tar.gz"
echo "  superpod-math-processed-gpu.tar.gz"
echo "  superpod-mo-processed-gpu.tar.gz"
```

## 5. Bootstrap: Getting the Data

The bootstrap script downloads the two StackExchange data dumps from the
Internet Archive and checks that all required input files are present. It
is idempotent — re-running it skips already-downloaded data.

``` {.bash file=scripts/handoff-superpod-bootstrap.sh}
<<bash-strict>>

<<bootstrap-setup>>

<<bootstrap-download>>

<<bootstrap-check>>
```

### 5.1 Setup and prerequisites

The bootstrap needs `python3` to invoke the downloader built into
`superpod-job.py`. We check for it immediately rather than failing
midway through a large download.

``` {.bash #bootstrap-setup}
# Bootstrap required inputs for the superpod handoff run.
# Intended to run on the target machine (e.g. Linode) after cloning futon6.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] repo: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[bootstrap] ERROR: python3 not found"
  exit 1
fi
```

### 5.2 Downloading SE data dumps

Two corpora: math.stackexchange (~3.4 GB compressed, ~567K threads) and
mathoverflow (~500 MB compressed, ~100K threads, research-level). The
`--download` flag in `superpod-job.py` handles wget + 7z extraction.

``` {.bash #bootstrap-download}
echo "[bootstrap] downloading StackExchange dumps (idempotent)..."
python3 scripts/superpod-job.py --download math --data-dir ./se-data
python3 scripts/superpod-job.py --download mathoverflow --data-dir ./se-data
```

### 5.3 Checking required files

After download, we verify that the six required input files exist: Posts.xml
and Comments.xml for each corpus (downloaded above), plus two reference files
that already ship with the repo and need no downloading — the category-theory
reference dictionary (`data/nlab-ct-reference.json`, built from 20K nLab
pages) and the math term dictionary (`data/ner-kernel/terms.tsv`, 19K terms).

``` {.bash #bootstrap-check}
echo "[bootstrap] checking required files..."
test -f se-data/math.stackexchange.com/Posts.xml
test -f se-data/math.stackexchange.com/Comments.xml
test -f se-data/mathoverflow.net/Posts.xml
test -f se-data/mathoverflow.net/Comments.xml
test -f data/nlab-ct-reference.json
test -f data/ner-kernel/terms.tsv

echo "[bootstrap] OK: all required inputs are present."
```

## 6. GPU Backfill

The CPU baseline (stages 1, 5, 7) gives us parsed threads, term/scope
spotting, and wiring diagrams. The GPU backfill re-runs the full 7-stage
pipeline with embeddings, LLM pattern tagging, clustering, and reverse
morphogenesis enabled. The two outputs are complementary: CPU is
deterministic and fast; GPU adds richer semantic signals.

``` {.bash file=scripts/handoff-superpod-gpu-backfill.sh}
<<bash-strict>>

<<gpu-preamble>>

<<gpu-env>>

<<gpu-run-site>>

<<gpu-dispatch>>
```

### 6.1 Target parsing and GPU check

The script takes a single argument: `math`, `mathoverflow`, or `both`
(default). It validates the argument and locates the repo root.

``` {.bash #gpu-preamble}
# GPU backfill for stages 2/3/4/6 after CPU baseline artifacts exist.
# This is a required handoff stage on Superpod.
# Usage:
#   bash scripts/handoff-superpod-gpu-backfill.sh [math|mathoverflow|both]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-both}"
case "$TARGET" in
  math|mathoverflow|both) ;;
  *)
    echo "Usage: $0 [math|mathoverflow|both]"
    exit 1
    ;;
esac
```

### 6.2 Model configuration

The embedding and LLM models can be overridden via environment variables.
Defaults are BGE-large for embeddings and Llama-3-8B-Instruct for pattern
tagging and reverse morphogenesis.

``` {.bash #gpu-env}
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu] WARNING: nvidia-smi not found. GPU stack may be unavailable."
fi

LLM_MODEL="${LLM_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-large-en-v1.5}"
```

### 6.3 The run_site function

Each site runs the full `superpod-job.py` pipeline with GPU flags enabled,
followed by ct-verifier to produce the verification artifact.

``` {.bash #gpu-run-site}
run_site() {
  local site="$1"
  local posts="$2"
  local comments="$3"
  local outdir="$4"

  echo "[gpu] running full 7-stage pipeline for $site ..."
  python3 scripts/superpod-job.py \
    "$posts" \
    --comments-xml "$comments" \
    --site "$site" \
    --output-dir "$outdir" \
    --embed-device cuda \
    --embed-model "$EMBED_MODEL" \
    --llm-model "$LLM_MODEL"

  python3 scripts/ct-verifier.py verify \
    --wiring "$outdir/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$outdir/thread-wiring-ct-verification.json"
}
```

### 6.4 Site dispatch

``` {.bash #gpu-dispatch}
if [[ "$TARGET" == "math" || "$TARGET" == "both" ]]; then
  run_site \
    "math.stackexchange" \
    "./se-data/math.stackexchange.com/Posts.xml" \
    "./se-data/math.stackexchange.com/Comments.xml" \
    "./math-processed-gpu"
fi

if [[ "$TARGET" == "mathoverflow" || "$TARGET" == "both" ]]; then
  run_site \
    "mathoverflow.net" \
    "./se-data/mathoverflow.net/Posts.xml" \
    "./se-data/mathoverflow.net/Comments.xml" \
    "./mo-processed-gpu"
fi

echo "[gpu] done."
```

## 7. The Python Pipeline

The heavy lifting happens in `scripts/superpod-job.py` (1,800 lines). It is
not embedded here — it is a library, not part of the handoff narrative. The
seven stages are:

| # | Stage | What it does | HW |
|---|-------|-------------|-----|
| 1 | Parse | Stream XML into structured QA pairs | CPU |
| 2 | Embed | Vector embedding of every post (for similarity search) | GPU |
| 3 | Tag | Classify which of 25 argument patterns each answer uses | GPU |
| 4 | Cluster | Group threads by topic (based on embeddings) | CPU |
| 5 | Term + scope spotting | Find known math terms (19K dictionary) and where authors introduce variables/assumptions ("Let X be...", "Assume...", "Define...") | CPU |
| 6 | Situation reconstruction | LLM reconstructs the logical skeleton of each QA pair | GPU |
| 7 | Thread wiring | Build a typed graph of each thread: who asserts/challenges/clarifies what, which math structures appear, how conclusions connect to assumptions | CPU |

CPU-only mode (`--skip-embeddings --skip-llm --skip-clustering`) runs
stages 1, 5, and 7. The critical output is stage 7.

### Planned stages 8-10: expression surfaces and structural embeddings

The current pipeline produces rich per-thread annotations but they remain
local — there is no global structure connecting threads by the *shape* of
their mathematical arguments. Three additional stages address this:

| # | Stage | What it does | HW |
|---|-------|-------------|-----|
| 8 | Expression surface parsing | Parse every `$...$` LaTeX expression into an s-expression tree. E.g. `$s(X(e))=X(s(e))$` → `(= (s (X e)) (X (s e)))`. Each expression becomes a typed surface for further wiring. | CPU |
| 9a | Thread hypergraph assembly | Combine all annotation layers (NER terms, scope bindings, wiring edges, categorical detections, s-exp surfaces) into a single typed hypergraph per thread — the thread's *structural signature*. | CPU |
| 9b | Hypergraph embedding | Embed each thread's typed hypergraph into a fixed-dimensional vector using a graph neural network (Graphormer, GPS, or HGNN+). Training signal is free: tag co-occurrence, SE "related questions" links, shared categorical detections. | GPU |
| 10 | Structural similarity index | Build a FAISS index over the embedding vectors. Enables structural search: "find threads with the same argument shape as this one." | CPU |

Stage 8 is deterministic and embarrassingly parallel across the 128 CPU
cores. Stage 9b is the only new GPU stage — it trains a graph embedding
model on the ~2M thread hypergraphs. The result is what we're calling
a **LWGM** (Large Wiring Graph Model): an embedding space over *argument
structure*, not token sequences.

A demo of stages 5+7+8 on a single thread is at
`data/first-proof/nnexus-glasses-demo.html` — open it in any browser to
see NER terms, scope maps, wiring edges, categorical badges, and
expression surfaces with s-exp tooltips all rendered on math.SE #633512.

These stages are not yet implemented in `superpod-job.py` — they are
defined as prototype P11 in the devmap. The current handoff run produces
the inputs they need (stages 1-7 output). A follow-up run would add
stages 8-10 once the code is ready.

## 8. What the output actually looks like

We've already run the pipeline at small scale on 200 StackExchange threads
(50 each from math.SE category-theory, math.SE mathematical-physics,
MathOverflow category-theory, MathOverflow mathematical-physics) and on
500 nLab wiki pages. Here's what it produces.

### A single thread, wired

Take math.SE thread #633512: *"What is the definition of a commutative
diagram?"* (score 18). The pipeline turns it into a typed graph with 16
nodes and 15 edges:

```text
Nodes:
  q-633512   (question, score=18)  terms: category, commutative diagram, functor, poset...
  a-633527   (answer,   score=17)  terms: category, diagram, morphism, identity...
                                   categorical: equivalence, limit, universal-property
  c-1335328  (comment,  score= 0)
  c-1335337  (comment,  score= 0)
  ...14 more comments

Edges (argument flow):
  a-633527   -> q-633512    [assert]      the answer asserts a response
  c-1335328  -> q-633512    [clarify]     a comment clarifies the question
  c-1335337  -> q-633512    [exemplify]   a comment gives an example
  c-1335340  -> q-633512    [reference]   a comment cites external material
  c-1335366  -> a-633527    [challenge]   a comment challenges the answer
  c-1335392  -> a-633527    [query]       a comment asks a follow-up
  c-1336112  -> a-633527    [agree]       a comment agrees
  ...8 more edges
```

The edge types (assert, clarify, exemplify, reference, challenge, query,
agree, reform, retract) are detected from discourse markers in the text.
The categorical annotations (equivalence, limit, universal-property) come
from matching against a reference dictionary built from 20,000 nLab pages.

### The signal is real: category theory vs. mathematical physics

Across the 200-thread pilot, category-theory threads light up the categorical
detector while mathematical-physics threads do not:

| Corpus | Threads | Nodes | Edges | Cat. detections | Cat./thread |
|--------|---------|-------|-------|-----------------|-------------|
| math.SE category-theory | 50 | 341 | 291 | 134 | 2.7 |
| math.SE math-physics | 50 | 280 | 230 | 6 | 0.1 |
| MO category-theory | 50 | 463 | 413 | 159 | 3.2 |
| MO math-physics | 50 | 557 | 507 | 20 | 0.4 |

Category-theory threads average 3 categorical detections per thread.
Mathematical-physics threads average 0.2. The pipeline is distinguishing
content, not just counting words.

The edge-type distribution also varies across corpora. MathOverflow has
proportionally more `reference` and `challenge` edges than math.SE —
consistent with research-level threads being more argumentatively dense.

### Term and scope spotting on nLab

On 500 nLab pages, stage 5 found 16,181 term hits (97.8% of pages had at
least one known term) and 856 scope openers. The scope types break down as:

| Scope type | Count | Example |
|-----------|-------|---------|
| Set membership (`$x \in X$`) | 498 | `$\forall X\in S$` |
| Universal quantifier ("for any...") | 131 | "for every morphism $f$" |
| Let-binding ("Let $X$ be...") | 114 | "Let $C$ be a category" |
| Where-binding ("where $X$ is...") | 82 | "where $F$ denotes the free functor" |
| Consider ("Consider a...") | 21 | "Consider a pullback square" |
| Assume ("Assume/Suppose...") | 10 | "Suppose $f$ is an epimorphism" |

These are the places where an author introduces a variable or sets up the
playing field for an argument. The superpod run will find the same structures
across 667K threads.

### Existing evidence in this repo

All of this is already in the repo if you want to poke at it:

- `data/thread-wiring/` — 200 wiring diagrams (4 files, 3.6 MB)
- `data/nlab-preview/` — 500 nLab pages processed (entities, terms, scopes)
- `data/ct-validation/` — 313 PlanetMath category-theory entries with wires,
  ports, golden test cases, and PlanetMath-to-nLab concept bridges
- `data/physics-se-classical/` — 114K physics.SE QA pairs (full stage 1+5 run)
- `data/stackexchange-samples/` — the 200 raw threads used for the pilot

## 9. Deliverables and return payload

Expected outputs:
- `superpod-math-processed.tar.gz`
- `superpod-mo-processed.tar.gz`
- `superpod-math-processed-gpu.tar.gz`
- `superpod-mo-processed-gpu.tar.gz`

Send back all 4 tarballs plus a short metric table from CPU and GPU
`manifest.json` files:
- `entity_count`
- `stage5_stats.total_ner_hits`
- `stage7_stats.threads_processed`
- `stage7_stats.total_nodes`
- `stage7_stats.total_edges`
- `stage7_stats.n_categorical`
- `stage7_stats.n_port_matches`

## 10. Mission wiring diagram

Intent: Convert raw public math Q/A corpora into verified typed wiring
artifacts that can be queried by downstream proof work.

Legend: `M*` mission control, `D*` data source, `P*` processing stage,
`V*` verification gate, `O*` output.

```text
Mission Control (single command)
  M0: bash scripts/handoff-superpod-all.sh
      |
      +--> M1 bootstrap: scripts/handoff-superpod-bootstrap.sh
      |       |
      |       +--> D1 math.stackexchange.com (Posts.xml, Comments.xml)
      |       +--> D2 mathoverflow.net       (Posts.xml, Comments.xml)
      |       +--> D3 local refs, already in repo (nlab-ct-reference.json, terms.tsv)
      |
      +--> M2 tests: pytest smoke + verifier checks
      |
      +--> M3 smoke gate (mini dataset, 4 threads)
      |       |
      |       +--> V1 verify required files + edges_checked > 0
      |
      +--> M4 CPU baseline run (math.SE + MO; stages 1/5/7)
      |       |
      |       +--> P1 parse XML posts/comments
      |       +--> P2 term + scope spotting
      |       +--> P3 thread wiring assembly (category-theory backed)
      |       +--> V2 manifest sanity + structural verifier
      |       +--> O1 math-processed/
      |       +--> O2 mo-processed/
      |
      +--> M5 required GPU backfill (math.SE + MO; full stages 1..7)
      |       |
      |       +--> P4 embeddings (GPU)
      |       +--> P5 LLM pattern tagging (GPU)
      |       +--> P6 clustering + situation reconstruction
      |       +--> V3 structural verifier refresh
      |       +--> O3 math-processed-gpu/
      |       +--> O4 mo-processed-gpu/
      |
      +--> M6 packaging
      |       |
      |       +--> O5 superpod-math-processed.tar.gz
      |       +--> O6 superpod-mo-processed.tar.gz
      |       +--> O7 superpod-math-processed-gpu.tar.gz
      |       +--> O8 superpod-mo-processed-gpu.tar.gz
      |
      +--> M7 [planned] expression surfaces + structural embeddings
              |
              +--> P7  Stage 8:  LaTeX → s-exp (CPU x128, deterministic)
              +--> P8  Stage 9a: thread hypergraph assembly (CPU)
              +--> P9  Stage 9b: hypergraph embedding (GPU, graph transformer)
              +--> P10 Stage 10: FAISS structural similarity index (CPU)
              +--> O9  structural-embeddings.tar.gz
```

Invariants enforced by the orchestrator:
- required artifacts exist: manifest, CT wiring output, and CT verifier output
- `stage7_stats.ct_backed=true`, `stage7_stats.threads_processed > 0`,
  and `edges_checked > 0`; otherwise the run fails hard

## 11. Why this work is interesting and valuable

This run is not just data collection. It produces a reusable evidence layer for
math reasoning work: each thread becomes a typed wiring object with explicit
nodes, edges, and port matches, not just text.

That is valuable for two reasons. First, retrieval quality improves: we can ask
for threads that match an input/output proof shape, not only threads that share
keywords. Second, verification quality improves: `ct-verifier` checks the
resulting structure and blocks empty artifacts.

The CPU and GPU outputs are complementary, not interchangeable. CPU gives a
deterministic baseline and immediate wiring products. GPU backfill adds richer
semantic signals (embeddings and LLM-derived structure) over the same corpus.
Keeping both lets us compare quality and track where extra compute changes
results.

At project level, this turns a one-off run into infrastructure. The same
pipeline can be rerun, audited, and diffed over time, so Rob can evaluate
whether later changes improve signal, regress quality, or break invariants.

The longer-term vision is stages 8-10: parse every LaTeX expression into a
typed s-expression tree, assemble all annotation layers into a hypergraph
per thread, and embed those hypergraphs via a graph neural network. The
result is a structural similarity index where "related questions" means
"structurally analogous mathematical reasoning" — not just shared keywords.
We're calling this a LWGM (Large Wiring Graph Model). The current run
produces the inputs it needs; stages 8-10 would run as a follow-up once
the code is ready.

## Notes

- `--skip-bootstrap` and `--skip-tests` are available for reruns only.
- `scripts/handoff-superpod-run.sh` is an earlier draft, superseded by
  `scripts/handoff-superpod-all.sh`.
- To regenerate the scripts from this document: `entangled tangle`
