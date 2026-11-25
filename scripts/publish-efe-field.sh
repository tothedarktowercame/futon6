#!/usr/bin/env bash
# Regenerate and publish the embedding-coordinate EFE field as a self-contained
# Zone snapshot.  The source renderer polls localhost; a public static page
# must instead contain the exact live response observed at publication time.
: "${FUTON_CODE_ROOT:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# june 2026-09-16: derived code root replaces hardcoded ${FUTON_CODE_ROOT} paths.
set -euo pipefail

ROOT="${EFE_CODE_ROOT:-${FUTON_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
F6="${EFE_FUTON6:-$ROOT/futon6}"
F2="${EFE_FUTON2:-$ROOT/futon2}"
BASE="${EFE_BASE:-http://localhost:7070}"
OUT="${EFE_OUT:-/var/www/zone.hyperreal.enterprises/wip/mission-efe-field.html}"
PY="${EFE_PYTHON:-$F6/.venv/bin/python}"
TRIES="${EFE_TRIES:-6}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fetch_required() {
  local path="$1" dest="$2"
  local i=1 code
  while [ "$i" -le "$TRIES" ]; do
    code=$(curl -sS -m 90 -o "$dest" -w '%{http_code}' "$BASE$path" || true)
    if [ "$code" = "200" ]; then
      echo "  $path -> HTTP 200 ($(stat -c%s "$dest") bytes, try $i)" >&2
      return 0
    fi
    echo "  $path -> HTTP ${code:-000} (try $i/$TRIES): $(head -c 120 "$dest" 2>/dev/null)" >&2
    i=$((i + 1))
    sleep $((i * 3))
  done
  echo "FATAL: $path never returned 200 in $TRIES tries; refusing to publish." >&2
  exit 1
}

echo "Refreshing the used embedding-field inputs ..." >&2
(
  cd "$F6"
  scripts/refresh_pattern_attestation.sh
  "$PY" scripts/mission_carpet.py --roads-only
  "$PY" scripts/mission_carpet_variants.py
  "$PY" scripts/mission_efe_scope_dump.py
  bb scripts/starmap_to_capability_graph.bb
)
(
  cd "$F2"
  clojure -M scripts/capability_zones_live_map.clj
)

echo "Fetching the live overlay from the running futon3c JVM ..." >&2
fetch_required "/api/alpha/live-efe-map" "$WORK/live-efe-map.json"

echo "Rendering the embedding-coordinate field ..." >&2
(
  cd "$F6"
  "$PY" scripts/mission_efe_field.py embed
)

SRC="$F6/data/mission-efe-field-embed.html"
test -s "$SRC" || { echo "FATAL: renderer did not produce $SRC" >&2; exit 1; }

"$PY" "$F6/scripts/inline_efe_snapshot.py" "$SRC" "$WORK/live-efe-map.json" "$OUT" "$ROOT"

# Archipelago prototypes (M-live-efe-map; brief ~/futon-archipelago.txt §11):
# separate pages built over the fresh embed page, published beside it as
# archipelago-<X>.html. Their data tables are refreshed first. A prototype that
# fails is reported and skipped; it never blocks the main page above.
echo "Publishing archipelago prototypes ..." >&2
OUTDIR="$(dirname "$OUT")"
(
  cd "$F6"
  "$PY" scripts/mission_dating.py >/dev/null || echo "WARN: mission_dating failed; prototypes use the previous table" >&2
  "$PY" scripts/pattern_roads.py >/dev/null || echo "WARN: pattern_roads failed; prototypes use the previous table" >&2
  "$PY" scripts/pattern_exposure_daily.py >/dev/null || echo "WARN: pattern_exposure_daily failed; prototypes use the previous table" >&2
  for spec in "archipelago_tide.py:archipelago-A-tide.html" \
              "archipelago_illumination.py:archipelago-B-illumination.html" \
              "archipelago_c.py:archipelago-C.html" \
              "archipelago_d.py:archipelago-D.html"; do
    script="${spec%%:*}" page="${spec##*:}"
    [ -f "scripts/$script" ] || continue
    if "$PY" "scripts/$script" >/dev/null \
       && "$PY" scripts/inline_efe_snapshot.py "data/$page" "$WORK/live-efe-map.json" "$OUTDIR/$page" "$ROOT"; then
      :
    else
      echo "WARN: prototype $script failed; $OUTDIR/$page left as it was" >&2
    fi
  done
)

# The Futon City visual essay (futon-city.html): pictures of the pages just
# published, in the configurations its deep links open. Non-fatal.
echo "Rebuilding the Futon City visual essay ..." >&2
(
  cd "$F6"
  SHOTS="$WORK/futon-city-shots"
  if "${EFE_NODE:-node}" scripts/futon_city_shots.js "file://$OUTDIR" "$SHOTS" >/dev/null \
     && "$PY" scripts/futon_city_essay.py "$SHOTS" "$OUTDIR"; then
    :
  else
    echo "WARN: futon-city essay not rebuilt; the previous page stays" >&2
  fi
)

echo "Updating the shared WIP navigation ..." >&2
"$ROOT"/p4ng/publish-wip-nav.sh

grep -q 'id="snapshot-provenance"' "$OUT"
grep -q 'futon6/scripts/publish-efe-field.sh' "$OUT"
if grep -qE 'localhost:7070|fetch\(' "$OUT"; then
  echo "FATAL: verification found a live-fetch call in $OUT" >&2
  exit 1
fi
echo "Published self-contained EFE field: $OUT" >&2
