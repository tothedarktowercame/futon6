#!/usr/bin/env bash
# Refresh futon6/data/pattern-attestation.json from the LIVE evidence store.
#
# Provenance: the original file was a one-off ad-hoc dump (2026-06-08) of
# `bb -m futon0.report.pattern-density 60 700` scraped through python. This
# script is that pipeline, checked in, atomic, and guarded: the report walks
# context-retrieval evidence (one entry per A->B turn; body :results = the
# patterns futon3a surfaced for that turn) via GET :7070/api/alpha/evidence.
# Signal semantics: pattern-SURFACED-by-retrieval, not PSR-confirmed
# application (see derive-pattern-activations in futon3c transport/http.clj).
#
# Window: 60 days rolling. top-n 5000 >> 1081 flexiargs, so nothing is
# silently dropped (the June dump's top-700 was a silent cap).
set -euo pipefail

OUT=/home/joe/code/futon6/data/pattern-attestation.json
TMP="$(mktemp "$(dirname "$OUT")/.pattern-attestation.XXXXXX.json")"
trap 'rm -f "$TMP"' EXIT

cd /home/joe/code/futon0
# Pin to the local serving JVM: an inherited FUTON3C_SERVER can silently point
# the report at a remote mesh host with a near-empty evidence store (found
# live 2026-07-05: 172.236.28.208 answered with 5 events vs localhost's 8k).
export FUTON3C_EVIDENCE_BASE="http://localhost:7070"
timeout 90 bb --classpath scripts -m futon0.report.pattern-density 60 5000 2>/dev/null \
  | python3 -c "
import sys, re, json
att = {}
for line in sys.stdin:
    m = re.match(r'\|\s*([a-z0-9][\w/-]+?)\s*\|\s*(\d+)\s*\|', line)
    if m:
        pid, c = m.group(1), int(m.group(2))
        att[pid] = max(att.get(pid, 0), c)
if len(att) < 50:
    sys.exit(f'refusing to overwrite: only {len(att)} patterns scraped (JVM down or report shape changed?)')
byname = {}
for pid, c in att.items():
    name = pid.split('/')[-1]
    byname[name] = max(byname.get(name, 0), c)
json.dump({'by_id': att, 'by_name': byname,
           'window_days': 60, 'source': 'futon0.report.pattern-density via refresh_pattern_attestation.sh'},
          open('$TMP', 'w'))
print(f'{len(att)} pattern ids, {len(byname)} names', file=sys.stderr)
"
mv "$TMP" "$OUT"
trap - EXIT
echo "[attestation] refreshed $OUT"
