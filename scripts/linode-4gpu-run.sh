#!/usr/bin/env bash
# mark4 — self-contained IATC run: ensure ENRICHED candidates (CPU stages 1+2),
# then run the GPU IATC reconstruction loop against the 70B served by
# linode-4gpu-setup.sh. Run ON the Linode after setup is READY.
#
# Why the extract step is here: the model stage must read the deterministic anatomy
# (symbol typings, scopes, proof-moves). A prior run fed the model raw source +
# binders only because nothing built/validated the enrichment — this script now
# closes that by (re)extracting when marks are local, or refusing early otherwise.
#
# Prereq: futon6 present at $REPO (rsync from dev box or git clone), and a python
# with the loop's HTTP deps ($PYTHON). The loop only needs an OpenAI-compatible
# client, not vLLM, so any small env works.
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-mark4-70b}"          # the --served-model-name from setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$HOME/mark4-venv}"     # same default as linode-4gpu-setup.sh
PYTHON="${PYTHON:-$VENV/bin/python}" # override if loop deps live elsewhere
if [[ ${FUTON6_PYTHON_CMD+x} ]]; then
  # Parse using the same stdlib configuration reader as the stepper. NUL
  # separators preserve flags and paths with spaces without eval or word splitting.
  mapfile -d '' -t PYTHON_ARGV < <("${FUTON6_PYTHON:-python3}" "$SCRIPT_DIR/futon6_config.py" --python-argv0)
else
  PYTHON_ARGV=("$PYTHON" -u)
fi
[[ ${#PYTHON_ARGV[@]} -gt 0 && -x ${PYTHON_ARGV[0]} ]] || {
  echo "FATAL: configured Python is not executable; set FUTON6_PYTHON_CMD."
  exit 1
}
cd "$REPO"
# Standalone legacy invocations must also pass their interpreter to bb gates.
FUTON6_PYTHON_CMD="$("${PYTHON_ARGV[@]}" -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "${PYTHON_ARGV[@]}")"
FUTON6_PYTHON_ARGV_JSON="$("${PYTHON_ARGV[@]}" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${PYTHON_ARGV[@]}")"
export FUTON6_PYTHON_CMD FUTON6_PYTHON_ARGV_JSON
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:$PORT/v1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL%/}"
export OPENAI_BASE_URL MODEL
OUT="${OUT:-data/iatc-argument-graphs/loop-run-70b}"
CANDIDATES="${CANDIDATES:-data/iatc-candidates}"
MARKS_DIR="${MARKS_DIR:-data/showcases/ct-anatomy/golden}"
EVAL_REPORT="${EVAL_REPORT:-mark3-eval-report-70b.json}"   # distinct from the blind-run mark3-eval-report.json
EVAL_SUMMARY="${EVAL_SUMMARY:-mark3-eval-summary-70b.md}"
RUN_EVAL="${RUN_EVAL:-1}"                                  # set 0 to skip the auto-eval tail (e.g. golden/prior absent on box)

echo "== wait for configured model endpoint =="
for i in $(seq 1 60); do curl -sf -H "Authorization: Bearer ${OPENAI_API_KEY:-x}" "$OPENAI_BASE_URL/models" >/dev/null 2>&1 && break; sleep 5; done
curl -sf -H "Authorization: Bearer ${OPENAI_API_KEY:-x}" "$OPENAI_BASE_URL/models" >/dev/null 2>&1 || { echo "FATAL: model endpoint not serving"; exit 1; }
echo "server up"

# --- Stages 1+2 (CPU): ensure the candidate dir carries inlined enrichment ---
# Pre-built candidates (e.g. the stepper's S3 extract, which threads the run's
# id manifest) take precedence — re-extracting here without --list/--papers
# would silently fall back to the 10-paper demo set and clobber them. Only
# extract when the dir is empty AND marks are local; refuse early otherwise.
# IDS_LIST (file of paper ids) / ALL_PROOFS=1 parameterize that fallback extract.
if compgen -G "$CANDIDATES/*.candidate.json" >/dev/null 2>&1; then
  echo "== pre-built candidates found -> verifying enrichment in $CANDIDATES =="
elif compgen -G "$MARKS_DIR/fable-*-dp-emacs.json" >/dev/null 2>&1; then
  echo "== no candidates, marks present -> extract enriched candidates into $CANDIDATES =="
  "${PYTHON_ARGV[@]}" scripts/mark3_extract_candidates.py --out "$CANDIDATES" \
    ${IDS_LIST:+--list "$IDS_LIST"} ${ALL_PROOFS:+--all-proofs}
else
  echo "== no candidates and no local marks =="
fi

# Verify whatever the branch above left in $CANDIDATES (pre-built, fresh, or nothing).
"${PYTHON_ARGV[@]}" - "$CANDIDATES" <<'PY'
import json, sys, glob
d = sys.argv[1]
cs = sorted(glob.glob(f"{d}/*.candidate.json"))
if not cs:
    sys.exit(f"FATAL: no candidates in {d} — extract on the dev box "
             f"(python scripts/mark3_extract_candidates.py) and rsync them here.")
# The schema and required inputs belong to the run contract; the loop enforces
# them (require_candidates). A second hardcoded copy here is how the two drift.
print(f"  found {len(cs)} candidate(s); the loop checks them against the run contract")
PY

echo "== IATC reconstruction loop (70B) over $CANDIDATES -> $OUT =="
OPENAI_API_KEY="${OPENAI_API_KEY:-x}" \
  "${PYTHON_ARGV[@]}" scripts/mark3_iatc_loop.py \
    --candidates "$CANDIDATES" \
    --out "$OUT" \
    --backend openai --model "$MODEL"

# --- auto-eval tail (NON-FATAL): self-report the graded result over $OUT. ---
# The graphs above are the precious artifact and are already saved; eval is only
# reporting, so a tooling hiccup here must NEVER abort or mask a completed GPU run.
# Each step is guarded so a non-zero exit is logged and the script still exits 0.
# This is the "replace the blind eval with real metrics" step: it produces the
# checker-% / substance-% / grounding-% / expository-% the manual block used to
# only *describe*. (Per-graph argcheck+repair+substance already run in-loop; this
# is the aggregate pass over the whole dir.)
if [ "$RUN_EVAL" = "1" ]; then
  echo
  echo "== auto-eval over $OUT (non-fatal; graphs above are already saved) =="

  # Grade ONLY the canonical final graphs. $OUT/.attempts holds retry intermediates
  # (including attempts for papers that never produced a final, e.g. a failed paper's
  # attempt1/attempt2). The graders disagree on recursion — substance_gate globs the
  # top level (correct), but iatc_argcheck (file-seq) and mark3_eval_harness (rglob)
  # recurse into .attempts and would double-count finals + score failed-paper attempts
  # as passes. So stage the top-level *.edn into a clean temp dir and grade THAT, so
  # all three agree on the same finals-only set. (Temp dir is outside $OUT — we never
  # write into the live output dir.)
  EVAL_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mark4-eval-final.XXXXXX")"
  trap 'rm -rf "$EVAL_STAGE"' EXIT
  n_final=0
  for f in "$OUT"/*.edn; do [ -e "$f" ] || continue; cp "$f" "$EVAL_STAGE/"; n_final=$((n_final+1)); done
  echo "  grading $n_final final graph(s) from $OUT (excluding $OUT/.attempts)"
  if [ "$n_final" -eq 0 ]; then
    echo "  WARN: no final graphs in $OUT — nothing to grade (did the loop write here?)."
  else
    echo "-- [1/3] structural gate: iatc_argcheck (finals only) --"
    if command -v bb >/dev/null 2>&1; then
      bb scripts/iatc_argcheck.bb "$EVAL_STAGE" || echo "  (argcheck exit $? — some graphs flagged; that is a finding, not a tool error)"
    else
      echo "  WARN: bb not on PATH — per-graph argcheck already ran in-loop; skipping dir summary."
    fi

    echo "-- [2/3] substance gate (finals only) --"
    "${PYTHON_ARGV[@]}" scripts/substance_gate.py "$EVAL_STAGE" --kind iatc \
      || echo "  (substance_gate exit $? — non-zero means some graphs flagged; expected, not a tool error)"

    echo "-- [3/3] eval harness -> real grounding / expository / prior-vs-posterior metrics --"
    if "${PYTHON_ARGV[@]}" scripts/mark3_eval_harness.py "$EVAL_STAGE" \
         --out "$EVAL_REPORT" --summary-out "$EVAL_SUMMARY"; then
      echo "  wrote $EVAL_REPORT + $EVAL_SUMMARY (graded $n_final finals from $OUT)"
      [ -f "$EVAL_SUMMARY" ] && { echo "  ---- $EVAL_SUMMARY ----"; cat "$EVAL_SUMMARY"; }
    else
      echo "  WARN: eval harness exited non-zero (e.g. golden/prior not on this box) —"
      echo "        the graphs are safe; re-run the harness on the dev box where golden/prior live."
    fi
  fi
fi

echo
echo "== done. Remaining owner review (the gates + metrics above now run automatically): =="
echo "  - faithfulness: spot-check >=3 graphs vs source at cited line anchors (still manual)."
echo "  - distribution: confirm node/edge/hole spread is non-uniform (no template collapse)."
echo "  - compare the 70B pass-rate above against the 8B baseline (8B auto-failed 6/10 substance)."
echo
echo "  apm-structure-match stage is CPU (scope extract + match); runs here or on dev box."
echo "  GPU only needed there if using the pgvector/embedding matcher (Rob's pattern)."
