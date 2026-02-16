# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-bootstrap.sh>>[init]
#!/usr/bin/env bash
# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bash-strict>>[init]
set -euo pipefail
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bootstrap-setup>>[init]
# Bootstrap required inputs for the superpod handoff run.
# Intended to run on the target machine (e.g. Linode) after cloning futon6.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] repo: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[bootstrap] ERROR: python3 not found"
  exit 1
fi
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bootstrap-download>>[init]
echo "[bootstrap] downloading StackExchange dumps (idempotent)..."
python3 scripts/superpod-job.py --download math --data-dir ./se-data
python3 scripts/superpod-job.py --download mathoverflow --data-dir ./se-data
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bootstrap-check>>[init]
echo "[bootstrap] checking required files..."
test -f se-data/math.stackexchange.com/Posts.xml
test -f se-data/math.stackexchange.com/Comments.xml
test -f se-data/mathoverflow.net/Posts.xml
test -f se-data/mathoverflow.net/Comments.xml
test -f data/nlab-ct-reference.json
test -f data/ner-kernel/terms.tsv

echo "[bootstrap] OK: all required inputs are present."
# ~/~ end
# ~/~ end
