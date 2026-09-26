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
: "${FUTON_CODE_ROOT:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# june 2026-09-16: derived code root replaces hardcoded ${FUTON_CODE_ROOT} paths.
set -euo pipefail

# Repo and code root derive from THIS script's location; override with the
# documented environment variables rather than editing a path in here.
REPO="${FUTON6_CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_ROOT="${FUTON_CODE_ROOT:-$(dirname "$REPO")}"
STORAGE_ROOT="${FUTON6_STORAGE_ROOT:-$CODE_ROOT/storage}"
OUT="$REPO/data/pattern-attestation.json"
TMP="$(mktemp "$(dirname "$OUT")/.pattern-attestation.XXXXXX.json")"
trap 'rm -f "$TMP"' EXIT

cd "${FUTON0_ROOT:-$CODE_ROOT/futon0}"
# Pin to the local serving JVM: an inherited FUTON3C_SERVER can silently point
# the report at a remote mesh host with a near-empty evidence store (found
# live 2026-07-05: 172.236.28.208 answered with 5 events vs localhost's 8k).
export FUTON3C_EVIDENCE_BASE="http://localhost:7070"
# Measured 2026-09-26 (E-kimi-task-54): the report asks for a limit=30000 page;
# the server applies its 48h broad-page window (~17k entries, 52MB) and spends
# ~5ms/entry assembling it, i.e. 90-174s total. The old 90s outer timeout and
# the report's own 90s HTTP budget both sat INSIDE that range, so the scrape
# saw 0 rows and the guard blamed a healthy JVM. Budgets: inner HTTP 240s so
# the report fails cleanly on a genuinely stuck server, outer kill 300s.
export FUTON3C_EVIDENCE_TIMEOUT_MS="${FUTON3C_EVIDENCE_TIMEOUT_MS:-240000}"
REPORT_TIMEOUT="${PATTERN_DENSITY_TIMEOUT:-300}"
REPORT_OUT="$(mktemp)"
REPORT_ERR="$(mktemp)"
trap 'rm -f "$TMP" "$REPORT_OUT" "$REPORT_ERR"' EXIT
set +e
timeout "$REPORT_TIMEOUT" bb --classpath scripts -m futon0.report.pattern-density 60 5000 \
  >"$REPORT_OUT" 2>"$REPORT_ERR"
report_rc=$?
set -e
if [ "$report_rc" -eq 124 ]; then
  echo "FATAL: pattern-density report TIMED OUT after ${REPORT_TIMEOUT}s (exit 124)." >&2
  echo "  This is a slow evidence page, not a dead JVM: GET :7070/api/alpha/evidence" >&2
  echo "  costs ~5ms/entry server-side and a 48h broad page is ~17k entries (~90-174s)." >&2
  echo "  Raise PATTERN_DENSITY_TIMEOUT or fix the server-side scan; check the JVM" >&2
  echo "  separately (curl :7070/api/alpha/agents) before blaming it." >&2
  exit 1
fi
if [ "$report_rc" -ne 0 ]; then
  echo "FATAL: pattern-density report exited $report_rc: $(tail -n 1 "$REPORT_ERR")" >&2
  exit 1
fi
python3 -c "
import sys, re, json
att = {}
for line in sys.stdin:
    m = re.match(r'\|\s*([a-z0-9][\w/-]+?)\s*\|\s*(\d+)\s*\|', line)
    if m:
        pid, c = m.group(1), int(m.group(2))
        att[pid] = max(att.get(pid, 0), c)
if len(att) < 50:
    sys.exit(f'refusing to overwrite: report completed but only {len(att)} patterns scraped (report shape changed or evidence window empty; a timeout exits earlier with its own message)')
byname = {}
for pid, c in att.items():
    name = pid.split('/')[-1]
    byname[name] = max(byname.get(name, 0), c)
json.dump({'by_id': att, 'by_name': byname,
           'window_days': 60, 'source': 'futon0.report.pattern-density via refresh_pattern_attestation.sh'},
          open('$TMP', 'w'))
print(f'{len(att)} pattern ids, {len(byname)} names', file=sys.stderr)
" <"$REPORT_OUT"
mv "$TMP" "$OUT"
rm -f "$REPORT_OUT" "$REPORT_ERR"
trap - EXIT
echo "[attestation] refreshed $OUT"
