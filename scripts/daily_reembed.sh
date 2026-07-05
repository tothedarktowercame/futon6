#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/joe/code
FUTON6="$ROOT/futon6"
PY="$FUTON6/.venv/bin/python"
MISSION_RECORDS="$ROOT/data/notions/mission_records.json"
EMBED_OUT="$ROOT/data/notions/bge_mission_embeddings.json"
EMBED_DIR="$(dirname "$EMBED_OUT")"
EMBED_MODEL="BAAI/bge-large-en-v1.5"
RAW_TMP="$(mktemp "$EMBED_DIR/.bge_mission_embeddings.raw.XXXXXX.json")"
FINAL_TMP="$(mktemp "$EMBED_DIR/.bge_mission_embeddings.XXXXXX.json")"

cleanup() {
  rm -f "$RAW_TMP" "$FINAL_TMP"
}
trap cleanup EXIT

echo "[daily-reembed] embedding missions with $EMBED_MODEL"
"$PY" "$ROOT/futon3a/scripts/embed_text.py" \
  --json \
  --model "$EMBED_MODEL" \
  < "$MISSION_RECORDS" \
  > "$RAW_TMP"

echo "[daily-reembed] adding embedding metadata and checking record parity"
"$PY" - "$MISSION_RECORDS" "$RAW_TMP" "$FINAL_TMP" "$EMBED_MODEL" <<'PY'
import json
import sys
from pathlib import Path

records_path = Path(sys.argv[1])
raw_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
model = sys.argv[4]

records = json.loads(records_path.read_text())
embedded = json.loads(raw_path.read_text())

if len(records) != len(embedded):
    raise SystemExit(
        f"record-count mismatch: mission_records={len(records)} embedded={len(embedded)}"
    )

for i, item in enumerate(embedded):
    vec = item.get("vector")
    if not isinstance(vec, list) or not vec:
        raise SystemExit(f"missing vector at record {i}")
    item["embed_model"] = model
    item["embed_dim"] = len(vec)

out_path.write_text(json.dumps(embedded, separators=(",", ":")) + "\n")
print(f"[daily-reembed] prepared {len(embedded)} records; dim={embedded[0]['embed_dim'] if embedded else 0}")
PY

mv "$FINAL_TMP" "$EMBED_OUT"
rm -f "$RAW_TMP"
trap - EXIT

echo "[daily-reembed] regenerated $EMBED_OUT"

cd "$FUTON6"
echo "[daily-reembed] refreshing pattern attestation (60d rolling, live evidence)"
scripts/refresh_pattern_attestation.sh \
  || echo "[daily-reembed] WARNING: attestation refresh failed; keeping previous dump"

echo "[daily-reembed] regenerating attestation-weighted pattern roads"
"$PY" scripts/mission_carpet.py --roads-only \
  || echo "[daily-reembed] WARNING: roads refresh failed; keeping previous roads"

echo "[daily-reembed] regenerating mission carpet variants"
"$PY" scripts/mission_carpet_variants.py

echo "[daily-reembed] refreshing per-scope districts from substrate-2"
"$PY" scripts/mission_efe_scope_dump.py

echo "[daily-reembed] regenerating embedded EFE field"
"$PY" scripts/mission_efe_field.py embed

echo "[daily-reembed] complete"
