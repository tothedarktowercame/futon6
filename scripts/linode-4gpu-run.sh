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

echo
echo "== done. Next (owner review, same as the 8B run but now graded on 70B): =="
echo "  1. gates:        scripts/iatc_argcheck.bb + scripts/substance_gate.py over $OUT"
echo "  2. faithfulness: spot-check >=3 graphs vs source at cited line anchors"
echo "  3. distribution: confirm node/edge/hole spread is non-uniform (no template collapse)"
echo "  Compare 70B pass-rate against the 8B baseline (8B auto-failed 6/10 substance)."
echo
echo "  apm-structure-match stage is CPU (scope extract + match); runs here or on dev box."
echo "  GPU only needed there if using the pgvector/embedding matcher (Rob's pattern)."
