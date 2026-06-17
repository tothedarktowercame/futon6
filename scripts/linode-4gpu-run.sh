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
REPO="${REPO:-$HOME/futon6}"
VENV="${VENV:-$HOME/mark4-venv}"     # same default as linode-4gpu-setup.sh
PYTHON="${PYTHON:-$VENV/bin/python}" # override if loop deps live elsewhere
OUT="${OUT:-data/iatc-argument-graphs/loop-run-70b}"
CANDIDATES="${CANDIDATES:-data/iatc-candidates}"
MARKS_DIR="${MARKS_DIR:-data/showcases/ct-anatomy/golden}"
EVAL_REPORT="${EVAL_REPORT:-mark3-eval-report-70b.json}"   # distinct from the blind-run mark3-eval-report.json
EVAL_SUMMARY="${EVAL_SUMMARY:-mark3-eval-summary-70b.md}"
RUN_EVAL="${RUN_EVAL:-1}"                                  # set 0 to skip the auto-eval tail (e.g. golden/prior absent on box)

echo "== wait for vLLM on :$PORT =="
for i in $(seq 1 60); do curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 5; done
curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not serving on :$PORT"; exit 1; }
echo "server up: $(curl -s localhost:$PORT/v1/models | head -c 200)"

cd "$REPO"

[ -x "$PYTHON" ] || {
  echo "FATAL: PYTHON=$PYTHON is not executable."
  echo "Run scripts/linode-4gpu-setup.sh first, or launch with PYTHON=/path/to/python."
  exit 1
}

# --- Stages 1+2 (CPU): ensure the candidate dir carries inlined enrichment ---
# If marks are local, (re)build enriched candidates so the run is self-contained.
# If not (marks not rsync'd to this GPU box), refuse early unless a pre-built
# enriched candidate dir is already present. The loop's own precondition gate is
# the final backstop, but failing here is earlier and clearer.
if compgen -G "$MARKS_DIR/fable-*-dp-emacs.json" >/dev/null 2>&1; then
  echo "== marks present -> (re)extract enriched candidates into $CANDIDATES =="
  "$PYTHON" scripts/mark3_extract_candidates.py --out "$CANDIDATES"
else
  echo "== no local marks -> verifying pre-built enriched candidates in $CANDIDATES =="
  "$PYTHON" - "$CANDIDATES" <<'PY'
import json, sys, glob
d = sys.argv[1]
cs = sorted(glob.glob(f"{d}/*.candidate.json"))
if not cs:
    sys.exit(f"FATAL: no candidates in {d} — extract on the dev box "
             f"(python scripts/mark3_extract_candidates.py) and rsync them here.")
stale = [c for c in cs if json.load(open(c)).get("schema") != "iatc-candidate/v2-enriched"]
if stale:
    sys.exit(f"FATAL: {len(stale)}/{len(cs)} candidates are pre-enrichment — the model "
             f"would never see the anatomy. Re-extract on the dev box and rsync.")
print(f"  ok: {len(cs)} enriched candidates")
PY
fi

echo "== IATC reconstruction loop (70B) over $CANDIDATES -> $OUT =="
OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=x \
  "$PYTHON" scripts/mark3_iatc_loop.py \
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
    "$PYTHON" scripts/substance_gate.py "$EVAL_STAGE" --kind iatc \
      || echo "  (substance_gate exit $? — non-zero means some graphs flagged; expected, not a tool error)"

    echo "-- [3/3] eval harness -> real grounding / expository / prior-vs-posterior metrics --"
    if "$PYTHON" scripts/mark3_eval_harness.py "$EVAL_STAGE" \
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
