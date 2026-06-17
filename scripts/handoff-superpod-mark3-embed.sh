#!/usr/bin/env bash
set -euo pipefail

# H8 mark3 embedding handoff.
# Defaults assume one 8x A100 80GB node with 16 Slurm-exposed CPU procs, but
# every hardware value is an env knob.
#
# Environment:
#   NUM_GPUS              GPU count hint (default: 8)
#   EMBED_BATCH           BGE encode/train batch size (default: 256)
#   EMBED_WORKERS         BGE replica workers (default: 0 = auto all visible GPUs)
#   NUM_CPU_WORKERS       dataloader workers (default: Slurm/cgroup affinity, max 16)
#   NUM_SHARDS            corpus shard hint (default: 8)
#   BGE_MODEL             BGE model name/path (default: BAAI/bge-small-en-v1.5)
#   MARK3_EMBED_OUT       output dir (default: tmp/mark3-embed/ct-sample)
#   MARK3_CONCEPT_LIMIT   concept sample/full limit (default: 200)
#   MARK3_TERM_LIMIT      term-prior sample/full limit (default: 120)
#   MARK3_PAPER_LIMIT     paper sample/full limit (default: 24)
#
# Usage:
#   bash scripts/handoff-superpod-mark3-embed.sh sample
#   NUM_GPUS=8 EMBED_BATCH=512 bash scripts/handoff-superpod-mark3-embed.sh infer --space global

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

STAGE="${1:-sample}"
if [[ $# -gt 0 ]]; then
  shift
fi

NUM_GPUS="${NUM_GPUS:-8}"
EMBED_BATCH="${EMBED_BATCH:-256}"
EMBED_WORKERS="${EMBED_WORKERS:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BGE_MODEL="${BGE_MODEL:-BAAI/bge-small-en-v1.5}"
if [[ -z "${NUM_CPU_WORKERS:-}" ]]; then
  if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_PER_TASK > 0 )); then
    NUM_CPU_WORKERS="$SLURM_CPUS_PER_TASK"
  elif "$PYTHON" - <<'PY' >/dev/null 2>&1
import os
os.sched_getaffinity(0)
PY
  then
    NUM_CPU_WORKERS="$("$PYTHON" - <<'PY'
import os
print(min(16, len(os.sched_getaffinity(0))))
PY
)"
  else
    NUM_CPU_WORKERS=16
  fi
fi

export NUM_GPUS EMBED_BATCH EMBED_WORKERS NUM_CPU_WORKERS NUM_SHARDS BGE_MODEL

case "$STAGE" in
  sample|run-sample)
    "$PYTHON" scripts/mark3_embed.py run-sample "$@"
    ;;
  train|infer|eval)
    "$PYTHON" scripts/mark3_embed.py "$STAGE" "$@"
    ;;
  *)
    echo "Usage: $0 [sample|train|infer|eval] [mark3_embed.py args...]" >&2
    exit 1
    ;;
esac
