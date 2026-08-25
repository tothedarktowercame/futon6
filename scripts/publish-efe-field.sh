#!/usr/bin/env bash
# Regenerate and publish the embedding-coordinate EFE field as a self-contained
# Zone snapshot.  The source renderer polls localhost; a public static page
# must instead contain the exact live response observed at publication time.
set -euo pipefail

ROOT="${EFE_CODE_ROOT:-/home/joe/code}"
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

"$PY" - "$SRC" "$WORK/live-efe-map.json" "$OUT" "$ROOT" <<'PY'
import datetime
import hashlib
import html
import json
import os
import pathlib
import sys
import tempfile

src, snapshot_path, out, root = map(pathlib.Path, sys.argv[1:])
page = src.read_text()
snapshot = json.loads(snapshot_path.read_text())
if snapshot.get("ok") is not True:
    raise SystemExit("FATAL: live EFE response did not contain ok=true")

endpoint = '  const ENDPOINT = "http://localhost:7070/api/alpha/live-efe-map";'
if endpoint not in page:
    raise SystemExit("FATAL: live endpoint anchor changed; refusing ambiguous inlining")
page = page.replace(endpoint, "  const EFE_SNAPSHOT = " + json.dumps(snapshot, separators=(",", ":")) + ";", 1)

start_marker = "  async function refresh() {"
end_marker = "\n\n  drawCapabilityZones({\"capability-zones\": STATIC_CAPABILITY_ZONES});"
start = page.find(start_marker)
end = page.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("FATAL: live refresh hook changed; refusing ambiguous inlining")
static_refresh = '''  function refresh() {
    const data = EFE_SNAPSHOT;
    draw(data);
    const agents = data.agents ? data.agents["with-placement"] : 0;
    const wm = data["war-machine"] ? data["war-machine"].count : 0;
    setBadge("live", `snapshot · ${agents} agents · ${wm} WM`);
  }'''
page = page[:start] + static_refresh + page[end:]
page = page.replace("  window.setInterval(refresh, REFRESH_MS);\n", "", 1)

stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
paths = [
    (root / "futon6/data/mission-carpet-pos-embed.json", "refreshed", "embedding coordinates used by this page"),
    (root / "futon6/data/efe-scopes.json", "refreshed", "live futon1b mission scopes"),
    (root / "futon6/data/capability-graph.json", "refreshed", "curated capability star-map projection"),
    (root / "futon6/data/mission-carpet-roads.json", "refreshed", "rolling live-attestation roads"),
    (root / "futon3c/resources/capability_zones/live-map-pca3-v1.json", "refreshed", "PCA-3 capability-zone projection"),
    (root / "futon6/data/mission-carpet-pos.json", "not refreshed", "canonical force layout; not read by the embed variant"),
]

rows = []
for path, status, note in paths:
    if not path.is_file():
        raise SystemExit(f"FATAL: provenance input absent: {path}")
    modified = datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(
        "<tr><td><code>" + html.escape(str(path.relative_to(root))) + "</code></td>"
        "<td>" + html.escape(status) + "</td><td>" + html.escape(modified) + "</td>"
        "<td><code>" + digest[:12] + "…</code></td><td>" + html.escape(note) + "</td></tr>"
    )

provenance = f'''<section id="snapshot-provenance" style="margin:8px 20px 14px;padding:10px 12px;border:1px solid #334155;border-radius:6px;background:#0b1220;color:#cbd5e1">
<strong>Static live-data snapshot</strong> generated {html.escape(stamp)}.
The live overlay was read successfully from the running futon3c service and inlined; this page performs no browser-time live fetch.
Regenerate: <code>futon6/scripts/publish-efe-field.sh</code>.
<details><summary>input provenance and honest holes</summary>
<table style="margin-top:6px;border-collapse:collapse"><thead><tr><th>input</th><th>status</th><th>file time</th><th>SHA-256</th><th>note</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p><b>Used-input holes:</b> none. The canonical force-layout coordinate file is shown for clarity but is not an input to this embedding variant and was deliberately not re-solved.</p>
</details></section>'''
# The renderer emits a minimal HTML5 document with an IMPLICIT body -- no
# <body> tag at all, which is valid HTML5 and is true of every
# mission-efe-field*.html on disk, including the 2026-07-04 static one. The
# <body>-only anchor therefore refused to publish every time. Anchor on <body>
# when present, else on the first <header>, which is where the visible page
# starts and is exactly where the provenance block was meant to land.
# (claude-13, 2026-08-25.)
body = page.find("<body")
if body >= 0:
    insert_at = page.find(">", body)
    if insert_at < 0:
        raise SystemExit("FATAL: malformed <body> tag")
    insert_at += 1
else:
    insert_at = page.find("<header")
    if insert_at < 0:
        raise SystemExit("FATAL: generated page has neither <body> nor <header>")
page = page[:insert_at] + provenance + page[insert_at:]

if "localhost:7070" in page or "fetch(" in page:
    raise SystemExit("FATAL: published page still contains a browser-time live fetch")

out.parent.mkdir(parents=True, exist_ok=True)
fd, tmp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
try:
    with os.fdopen(fd, "w") as handle:
        handle.write(page)
    os.replace(tmp_name, out)
    # mkstemp creates 0600 and os.replace preserves it, so the published page
    # was unreadable by nginx and served 403. Match the rest of the docroot.
    # (claude-13, 2026-08-25.)
    os.chmod(out, 0o644)
except BaseException:
    try:
        os.unlink(tmp_name)
    except FileNotFoundError:
        pass
    raise
print(f"wrote {out} ({len(page)} bytes), snapshot at {stamp}")
PY

echo "Updating the shared WIP navigation ..." >&2
/home/joe/code/p4ng/publish-wip-nav.sh

grep -q 'id="snapshot-provenance"' "$OUT"
grep -q 'futon6/scripts/publish-efe-field.sh' "$OUT"
if grep -qE 'localhost:7070|fetch\(' "$OUT"; then
  echo "FATAL: verification found a live-fetch call in $OUT" >&2
  exit 1
fi
echo "Published self-contained EFE field: $OUT" >&2
