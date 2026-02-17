# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-gpu-backfill.sh>>[init]
#!/usr/bin/env bash
# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bash-strict>>[init]
set -euo pipefail
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-preamble>>[init]
# GPU backfill: full 11-stage pipeline including LWGM (stages 9b+10).
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-env>>[init]
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu] WARNING: nvidia-smi not found. GPU stack may be unavailable."
fi

LLM_MODEL="${LLM_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-large-en-v1.5}"
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-run-site>>[init]
run_site() {
  local site="$1"
  local posts="$2"
  local comments="$3"
  local outdir="$4"

  echo "[gpu] running full 11-stage pipeline for $site ..."
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-dispatch>>[init]
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
# ~/~ end
# ~/~ end
