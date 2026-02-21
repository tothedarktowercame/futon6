#!/usr/bin/env bash
set -euo pipefail

# End-to-end superpod handoff run.
# Intended to run on the Superpod execution host after cloning futon6.
#
# Modes:
#   --smoke-only  : run tests + smoke run, skip full runs
#   --full-only   : skip smoke run, do full runs (CPU baseline + GPU backfill)
#   --skip-tests  : skip pytest sanity checks

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="all"
RUN_TESTS=1
for arg in "$@"; do
  case "$arg" in
    --smoke-only) MODE="smoke-only" ;;
    --full-only) MODE="full-only" ;;
    --skip-tests) RUN_TESTS=0 ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--smoke-only|--full-only] [--skip-tests]"
      exit 1
      ;;
  esac
done

echo "[run] repo: $ROOT_DIR"
echo "[run] mode: $MODE"

if (( RUN_TESTS )); then
  echo "[run] pytest sanity checks..."
  PYTHONPATH=src pytest -q tests/test_superpod_job_smoke.py tests/test_ct_verifier.py
fi

if [[ "$MODE" != "full-only" ]]; then
  echo "[run] smoke run..."
  OUT_SMOKE="tmp/superpod-rob-smoke-$(date +%s)"
  python3 scripts/superpod-job.py \
    tests/fixtures/se-mini/Posts.xml \
    --comments-xml tests/fixtures/se-mini/Comments.xml \
    --site math.stackexchange \
    --output-dir "$OUT_SMOKE" \
    --min-score 0 \
    --thread-limit 4 \
    --skip-embeddings \
    --skip-llm \
    --skip-clustering

  python3 scripts/ct-verifier.py verify \
    --wiring "$OUT_SMOKE/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$OUT_SMOKE/thread-wiring-ct-verification.json"

  echo "[run] smoke output: $OUT_SMOKE"
fi

if [[ "$MODE" == "smoke-only" ]]; then
  echo "[run] done (smoke-only)."
  exit 0
fi

echo "[run] full CPU baseline: math.stackexchange..."
OUT_MATH="./math-processed"
python3 scripts/superpod-job.py \
  ./se-data/math.stackexchange.com/Posts.xml \
  --comments-xml ./se-data/math.stackexchange.com/Comments.xml \
  --site math.stackexchange \
  --output-dir "$OUT_MATH" \
  --skip-embeddings \
  --skip-llm \
  --skip-clustering

python3 scripts/ct-verifier.py verify \
  --wiring "$OUT_MATH/thread-wiring-ct.json" \
  --reference data/nlab-ct-reference.json \
  --output "$OUT_MATH/thread-wiring-ct-verification.json"

echo "[run] full CPU baseline: mathoverflow..."
OUT_MO="./mo-processed"
python3 scripts/superpod-job.py \
  ./se-data/mathoverflow.net/Posts.xml \
  --comments-xml ./se-data/mathoverflow.net/Comments.xml \
  --site mathoverflow.net \
  --output-dir "$OUT_MO" \
  --skip-embeddings \
  --skip-llm \
  --skip-clustering

python3 scripts/ct-verifier.py verify \
  --wiring "$OUT_MO/thread-wiring-ct.json" \
  --reference data/nlab-ct-reference.json \
  --output "$OUT_MO/thread-wiring-ct-verification.json"

echo "[run] required GPU backfill (stages 2/3/4/6)..."
bash scripts/handoff-superpod-gpu-backfill.sh both

echo "[run] packaging..."
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

tar czf superpod-math-processed-gpu.tar.gz \
  math-processed-gpu

tar czf superpod-mo-processed-gpu.tar.gz \
  mo-processed-gpu

echo "[run] done."
echo "[run] outputs:"
echo "  superpod-math-processed.tar.gz"
echo "  superpod-mo-processed.tar.gz"
echo "  superpod-math-processed-gpu.tar.gz"
echo "  superpod-mo-processed-gpu.tar.gz"
