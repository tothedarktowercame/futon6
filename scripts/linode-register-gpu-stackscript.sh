#!/usr/bin/env bash
# Create or update the private Linode StackScript used for mark4 GPU boxes.
set -euo pipefail

LABEL="${LABEL:-mark4-ubuntu2404-gpu-bootstrap}"
IMAGE="${IMAGE:-linode/ubuntu24.04}"
SCRIPT="${SCRIPT:-scripts/linode-gpu-bootstrap-stackscript.sh}"
DESCRIPTION="${DESCRIPTION:-mark4 Ubuntu 24.04 GPU bootstrap: NVIDIA driver, optional CUDA toolkit, optional linode-cli}"
REV_NOTE="${REV_NOTE:-$(date -Is) update from futon6}"

command -v linode-cli >/dev/null 2>&1 || {
  echo "FATAL: linode-cli not found. Install with: pipx install linode-cli"
  exit 1
}

[ -f "$SCRIPT" ] || {
  echo "FATAL: StackScript body not found: $SCRIPT"
  exit 1
}

existing_id="$(
  linode-cli stackscripts list \
    --label "$LABEL" \
    --format 'id,label' \
    --text --no-headers 2>/dev/null |
  awk -v label="$LABEL" '$2 == label {print $1; exit}'
)"

if [ -n "$existing_id" ]; then
  echo "== update StackScript $existing_id ($LABEL) =="
  linode-cli stackscripts update "$existing_id" \
    --label "$LABEL" \
    --images "$IMAGE" \
    --description "$DESCRIPTION" \
    --rev_note "$REV_NOTE" \
    --script "$(cat "$SCRIPT")"
  echo "$existing_id"
else
  echo "== create StackScript $LABEL =="
  linode-cli stackscripts create \
    --label "$LABEL" \
    --images "$IMAGE" \
    --description "$DESCRIPTION" \
    --is_public false \
    --rev_note "$REV_NOTE" \
    --script "$(cat "$SCRIPT")"
fi
