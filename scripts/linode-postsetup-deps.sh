#!/usr/bin/env bash
# mark4 — post-setup pipeline dependencies for the Linode runner box.
#
# `linode-4gpu-setup.sh` brings up the vLLM 70B server. THIS script installs the
# CPU-side gate/checker dependencies that the staged IATC driver shells out to,
# so the runner doesn't die mid-run on a missing binary (learned live 2026-06-18:
# the box has no `bb`, and `mark4_iatc_concurrent.py` calls babashka for the
# per-paper repair + argument/semantic-check gates).
#
# Run AFTER rsyncing futon6 to the box and BEFORE the run:
#   cd ~/futon6
#   scripts/linode-postsetup-deps.sh     # this script (idempotent)
#   scripts/linode-4gpu-setup.sh         # vLLM serve
#   scripts/linode-4gpu-run.sh           # or the staged runner
#
# Idempotent — safe to re-run; installed deps are skipped.
#
# Deps:
#   - babashka (bb): runs iatc_argcheck.bb / iatc_repair.bb / iatc_semcheck.bb
#     (the driver's per-paper repair + argument/semantic-check gates). The .bb
#     scripts only (require '[clojure.edn]) — no sibling-repo reach, so bb alone
#     is enough.
#   (substance_gate.py is pure-stdlib python3 — nothing to install.)
set -euo pipefail

REPO="${REPO:-$HOME/futon6}"

echo "== babashka (bb) =="
if command -v bb >/dev/null 2>&1; then
  echo "bb already present: $(bb --version)"
else
  curl -sSL https://raw.githubusercontent.com/babashka/babashka/master/install | bash
  echo "installed: $(bb --version)"
fi

echo "== sanity: pipeline gate scripts reachable in $REPO =="
missing=0
for s in scripts/iatc_argcheck.bb scripts/iatc_repair.bb scripts/iatc_semcheck.bb scripts/substance_gate.py; do
  if [ -f "$REPO/$s" ]; then
    echo "  ok: $s"
  else
    echo "  MISSING: $s  (rsync futon6 to the box first)"
    missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo "FATAL: gate scripts missing — rsync futon6 before running."; exit 1; }

echo "== post-setup deps complete =="
