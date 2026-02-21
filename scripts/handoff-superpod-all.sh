# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-all.sh>>[init]
#!/usr/bin/env bash
# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bash-strict>>[init]
set -euo pipefail
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-header-comment>>[init]
# Single-command Superpod handoff runner.
# Runs 11-stage pipeline (stages 1-7 + LWGM stages 8-10).
#
# Default behavior (NUM_SHARDS=1 or unset):
#   1) bootstrap inputs
#   2) sanity tests
#   3) smoke run + verification
#   4) full CPU runs + verification (stages 1/5/7/8/9a)
#   5) required GPU backfill + verification (stages 1-10 incl. LWGM)
#   6) package outputs
#
# Sharded / Block 1 (NUM_SHARDS>1):
#   1) bootstrap inputs
#   2) sanity tests
#   3) smoke run + verification
#   4) sharded GPU pipeline for both corpora
#   5) package outputs
#
# Block 2 (BLOCK=2, requires Block 1 output):
#   1) verify Block 1 output exists + GPU/LLM prereqs
#   2) sharded LLM on thread sample (both corpora)
#   3) compose LLM files into Block 1 output
#
# Invocations:
#   Block 1:  NUM_SHARDS=8 EXTRA_SHARD_ARGS="--skip-llm" bash scripts/handoff-superpod-all.sh
#   Block 2:  BLOCK=2 NUM_SHARDS=8 bash scripts/handoff-superpod-all.sh
#
# Options:
#   --smoke-only       stop after smoke run + verification
#   --skip-bootstrap   do not run bootstrap script
#   --skip-tests       do not run pytest sanity checks
#
# Environment:
#   NUM_SHARDS          number of parallel shards (default: 1 = unsharded)
#   EXTRA_SHARD_ARGS    extra args passed to each shard job (e.g. "--skip-llm")
#   BLOCK               1 (default) or 2; Block 2 adds LLM to Block 1 output
#   LLM_THREAD_LIMIT    threads per shard for Block 2 LLM (default: 5000)
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-root-and-args>>[init]
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_ONLY=0
SKIP_BOOTSTRAP=0
SKIP_TESTS=0
NUM_SHARDS="${NUM_SHARDS:-1}"
EXTRA_SHARD_ARGS="${EXTRA_SHARD_ARGS:-}"
BLOCK="${BLOCK:-1}"
LLM_THREAD_LIMIT="${LLM_THREAD_LIMIT:-5000}"

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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-utility-functions>>[init]
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-verify-run-dir>>[init]
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-run-smoke>>[init]
run_smoke() {
  local out_smoke="tmp/superpod-rob-smoke-$(date +%s)"
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-package-outputs>>[init]
package_outputs() {
  if [[ "$NUM_SHARDS" -le 1 ]]; then
    tar czf superpod-math-processed.tar.gz \
      math-processed/entities.json \
      math-processed/relations.json \
      math-processed/tags.json \
      math-processed/stats.json \
      math-processed/ner-terms.json \
      math-processed/scopes.json \
      math-processed/thread-wiring-ct.json \
      math-processed/thread-wiring-ct-verification.json \
      math-processed/expression-surfaces.json \
      math-processed/hypergraphs.json \
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
      mo-processed/expression-surfaces.json \
      mo-processed/hypergraphs.json \
      mo-processed/manifest.json
  fi

  # GPU tarballs include the whole directory: per-thread output AND
  # reusable model artifacts (graph-gnn-model.pt, structural-similarity-index.faiss,
  # hypergraph-embeddings.npy, etc.) for downstream hot re-embedding on CPU.
  tar czf superpod-math-processed-gpu.tar.gz math-processed-gpu
  tar czf superpod-mo-processed-gpu.tar.gz mo-processed-gpu
}
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-orchestration>>[init]
echo "[all] repo: $ROOT_DIR"
echo "[all] block=$BLOCK smoke_only=$SMOKE_ONLY skip_bootstrap=$SKIP_BOOTSTRAP skip_tests=$SKIP_TESTS num_shards=$NUM_SHARDS"

# ---- Block 2: LLM enrichment (compose onto Block 1 output) ----
if [[ "$BLOCK" == "2" ]]; then
  for d in ./math-processed-gpu ./mo-processed-gpu; do
    [[ -d "$d" ]] || fail "Block 1 output not found: $d (run Block 1 first)"
  done

  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || fail "PyTorch cannot see CUDA"

  LLM_MODEL="${LLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

  step "Block 2: LLM on ${LLM_THREAD_LIMIT}/shard sample (math.stackexchange)"
  python3 scripts/superpod-shard.py run \
    --posts-xml ./se-data/math.stackexchange.com/Posts.xml \
    --comments-xml ./se-data/math.stackexchange.com/Comments.xml \
    --site math.stackexchange \
    --num-shards "$NUM_SHARDS" \
    --output-dir ./math-processed-gpu-llm \
    --skip-post-merge \
    -- --thread-limit "$LLM_THREAD_LIMIT" --skip-embeddings \
    --llm-model "$LLM_MODEL" --embed-device cuda

  step "Block 2: LLM on ${LLM_THREAD_LIMIT}/shard sample (mathoverflow)"
  python3 scripts/superpod-shard.py run \
    --posts-xml ./se-data/mathoverflow.net/Posts.xml \
    --comments-xml ./se-data/mathoverflow.net/Comments.xml \
    --site mathoverflow.net \
    --num-shards "$NUM_SHARDS" \
    --output-dir ./mo-processed-gpu-llm \
    --skip-post-merge \
    -- --thread-limit "$LLM_THREAD_LIMIT" --skip-embeddings \
    --llm-model "$LLM_MODEL" --embed-device cuda

  step "compose LLM files into Block 1 output"
  for f in pattern-tags.json reverse-morphogenesis.json; do
    if [[ -f ./math-processed-gpu-llm/"$f" ]]; then
      cp ./math-processed-gpu-llm/"$f" ./math-processed-gpu/
      echo "  math-processed-gpu/$f"
    fi
    if [[ -f ./mo-processed-gpu-llm/"$f" ]]; then
      cp ./mo-processed-gpu-llm/"$f" ./mo-processed-gpu/
      echo "  mo-processed-gpu/$f"
    fi
  done

  echo
  echo "[all] Block 2 complete. LLM files composed into Block 1 output."
  exit 0
fi

# ---- Block 1 / default ----
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

if [[ "$NUM_SHARDS" -gt 1 ]]; then
  # Block 1: sharded pipeline supersedes separate CPU baseline + GPU backfill.
  # NUM_SHARDS and EXTRA_SHARD_ARGS are exported so gpu-backfill.sh picks them up.
  export NUM_SHARDS EXTRA_SHARD_ARGS

  step "sharded pipeline ($NUM_SHARDS shards) + verification"
  bash scripts/handoff-superpod-gpu-backfill.sh both
  verify_run_dir "./math-processed-gpu"
  verify_run_dir "./mo-processed-gpu"

  step "package outputs"
  package_outputs

  assert_file "superpod-math-processed-gpu.tar.gz"
  assert_file "superpod-mo-processed-gpu.tar.gz"

  echo
  echo "[all] complete (sharded / Block 1). Deliver:"
  echo "  superpod-math-processed-gpu.tar.gz"
  echo "  superpod-mo-processed-gpu.tar.gz"
else
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
fi
# ~/~ end
# ~/~ end
