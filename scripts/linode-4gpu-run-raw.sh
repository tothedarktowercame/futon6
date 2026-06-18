#!/usr/bin/env bash
# mark4 — 70B-on-RAW control arm (RAW-CTL). Companion to linode-4gpu-run.sh.
#
# PURPOSE. The go-live comparison (enriched-70B vs blind-8B) moved TWO variables at
# once: enrichment AND model size. This arm holds the model fixed (same 70B served
# by linode-4gpu-setup.sh) and the candidates fixed, and strips ONLY the
# deterministic anatomy (the `enrichment` array). Any delta vs loop-run-70b is then
# attributable to enrichment alone. It answers the cost-at-scale question: do we
# enrich before running the 70B over arXiv, or does raw suffice?
#
# DESIGN. The raw candidates are DERIVED from the EXACT enriched candidates the
# enriched arm used (data/iatc-candidates), emptying only `enrichment`. That is a
# tighter control than re-extracting: identical source windows, binders, and line
# ranges — the single changed variable is the inlined anatomy.
#
# GATE-COMPAT (why schema is retained). scripts/mark3_iatc_loop.py's precondition
# gate (require_enriched, ~L234) accepts a candidate iff
#   schema == "iatc-candidate/v2-enriched"  AND  an "enrichment" key is present.
# render_enrichment (~L86) already renders an empty array as
#   "(no deterministic anatomy detected in this window)".
# So the raw candidates RETAIN the schema string purely as a gate token, set
# `enrichment: []` (the real control variable), and carry an explicit
#   "_control_arm": "raw-no-enrichment"
# provenance marker. The arm is unambiguous via that marker + the dedicated
# data/iatc-candidates-raw input dir + the loop-run-70b-raw output dir — this runner
# needs ZERO source edits. (Cleaner-but-source-touching alternative: add an
# "iatc-candidate/v2-raw" schema to CANDIDATE_SCHEMAS in mark3_iatc_loop.py and
# relax the gate; deliberately avoided here so the handoff is a single script.)
#
# RUN ON the Linode AFTER linode-4gpu-setup.sh is READY and the enriched arm
# (linode-4gpu-run.sh) has produced data/iatc-candidates + loop-run-70b. The 70B
# must already be served (vLLM on :$PORT); this script does NOT re-serve it.
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-mark4-70b}"                 # --served-model-name from setup
REPO="${REPO:-$HOME/futon6}"
VENV="${VENV:-$HOME/mark4-venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
ENRICHED_CANDS="${ENRICHED_CANDS:-data/iatc-candidates}"            # source of truth
RAW_CANDS="${RAW_CANDS:-data/iatc-candidates-raw}"                  # derived here
ENRICHED_OUT="${ENRICHED_OUT:-data/iatc-argument-graphs/loop-run-70b}"
OUT="${OUT:-data/iatc-argument-graphs/loop-run-70b-raw}"
EVAL_REPORT="${EVAL_REPORT:-mark3-eval-report-70b-raw.json}"
EVAL_SUMMARY="${EVAL_SUMMARY:-mark3-eval-summary-70b-raw.md}"
RAW_CERT="${RAW_CERT:-/tmp/raw.cert.json}"
ENR_CERT="${ENR_CERT:-/tmp/enriched.cert.json}"
RUN_EVAL="${RUN_EVAL:-1}"

cd "$REPO" 2>/dev/null || { echo "FATAL: REPO=$REPO not found"; exit 1; }

echo "== wait for vLLM on :$PORT =="
for i in $(seq 1 60); do curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 5; done
curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not serving on :$PORT (run linode-4gpu-setup.sh first)"; exit 1; }
echo "server up: $(curl -s localhost:$PORT/v1/models | head -c 200)"

[ -x "$PYTHON" ] || { echo "FATAL: PYTHON=$PYTHON not executable. Run linode-4gpu-setup.sh, or set PYTHON=/path/to/python."; exit 1; }

# --- Derive RAW candidates from the enriched ones (strip enrichment only) ---
echo "== derive raw candidates: $ENRICHED_CANDS -> $RAW_CANDS (enrichment stripped) =="
"$PYTHON" - "$ENRICHED_CANDS" "$RAW_CANDS" <<'PY'
import json, sys, glob, os
src, dst = sys.argv[1], sys.argv[2]
cs = sorted(glob.glob(os.path.join(src, "*.candidate.json")))
if not cs:
    sys.exit(f"FATAL: no enriched candidates in {src}. Run the enriched arm "
             f"(linode-4gpu-run.sh) first, or rsync data/iatc-candidates to this box.")
os.makedirs(dst, exist_ok=True)
stripped = 0
for c in cs:
    d = json.load(open(c))
    stripped += len(d.get("enrichment") or [])
    d["enrichment"] = []                     # THE control variable: model sees no anatomy
    d["_control_arm"] = "raw-no-enrichment"  # explicit provenance (schema kept as gate token)
    d["_derived_from"] = os.path.basename(c)
    json.dump(d, open(os.path.join(dst, os.path.basename(c)), "w"), indent=2)
print(f"  wrote {len(cs)} raw candidates to {dst} (removed {stripped} enrichment marks total)")
PY

echo "== IATC reconstruction loop (70B) over RAW $RAW_CANDS -> $OUT =="
OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=x \
  "$PYTHON" scripts/mark3_iatc_loop.py \
    --candidates "$RAW_CANDS" \
    --out "$OUT" \
    --backend openai --model "$MODEL"

# --- auto-eval tail (NON-FATAL): mirrors linode-4gpu-run.sh. Graphs above are the
#     precious artifact and are already saved; eval is only reporting, so a hiccup
#     here must never abort or mask a completed GPU run. ---
if [ "$RUN_EVAL" = "1" ]; then
  echo; echo "== auto-eval over $OUT (non-fatal; graphs above already saved) =="
  EVAL_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mark4-eval-raw.XXXXXX")"
  trap 'rm -rf "$EVAL_STAGE"' EXIT
  n_final=0
  for f in "$OUT"/*.edn; do [ -e "$f" ] || continue; cp "$f" "$EVAL_STAGE/"; n_final=$((n_final+1)); done
  echo "  grading $n_final final graph(s) from $OUT (excluding $OUT/.attempts)"
  if [ "$n_final" -eq 0 ]; then
    echo "  WARN: no final graphs in $OUT — did the loop write here?"
  else
    if command -v bb >/dev/null 2>&1; then
      bb scripts/iatc_argcheck.bb "$EVAL_STAGE" || echo "  (argcheck exit $? — graphs flagged; a finding, not a tool error)"
    else
      echo "  WARN: bb not on PATH — per-graph argcheck ran in-loop; skipping dir summary."
    fi
    "$PYTHON" scripts/substance_gate.py "$EVAL_STAGE" --kind iatc \
      || echo "  (substance_gate exit $? — some graphs flagged; expected, not a tool error)"
    if "$PYTHON" scripts/mark3_eval_harness.py "$EVAL_STAGE" --out "$EVAL_REPORT" --summary-out "$EVAL_SUMMARY"; then
      echo "  wrote $EVAL_REPORT + $EVAL_SUMMARY"
    else
      echo "  WARN: eval harness exited non-zero (e.g. golden/prior absent on box) — graphs are safe; re-run on dev box."
    fi
  fi
fi

# --- enriched-vs-raw comparison artifacts (NON-FATAL) ---
echo; echo "== conformance certs for both arms (cas_cert) =="
"$PYTHON" scripts/cas_cert.py --graph-dir "$OUT" --out "$RAW_CERT" || echo "  (cas_cert raw exit $?)"
if [ -d "$ENRICHED_OUT" ]; then
  "$PYTHON" scripts/cas_cert.py --graph-dir "$ENRICHED_OUT" --out "$ENR_CERT" || echo "  (cas_cert enriched exit $?)"
  echo "  certs written: enriched=$ENR_CERT  raw=$RAW_CERT"
  echo "  -> diff aggregate gate + concept-grain + proof-grain between them (run-#2 enriched baseline: concept mean 0.867)."
else
  echo "  NOTE: enriched graphs ($ENRICHED_OUT) not on this box — generate its cert where they live and diff vs $RAW_CERT."
fi

echo; echo "== headline comparison: eval summaries side by side =="
[ -f "$EVAL_SUMMARY" ] && { echo "---- RAW: $EVAL_SUMMARY ----"; cat "$EVAL_SUMMARY"; echo; }
[ -f "mark3-eval-summary-70b.md" ] && { echo "---- ENRICHED: mark3-eval-summary-70b.md ----"; cat "mark3-eval-summary-70b.md"; echo; }

echo; echo "== RAW-CTL done. Artifacts:"
echo "  raw graphs : $OUT"
echo "  raw eval   : $EVAL_REPORT / $EVAL_SUMMARY"
echo "  certs      : raw=$RAW_CERT  enriched=$ENR_CERT"
echo
echo "  Owner read: if the RAW 70B matches the enriched arm on substance/grounding/concept-grain,"
echo "  enrichment is NOT needed before the arXiv 70B (the cost win). If raw degrades, enrichment earns its keep."
