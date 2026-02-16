# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-all.sh>>[init]
#!/usr/bin/env bash
# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bash-strict>>[init]
set -euo pipefail
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-header-comment>>[init]
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-root-and-args>>[init]
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-package-outputs>>[init]
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
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-orchestration>>[init]
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
# ~/~ end
# ~/~ end
