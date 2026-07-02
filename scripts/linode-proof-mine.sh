#!/usr/bin/env bash
# PROOF-MINE GPU run — mirrors linode-meme-mine.sh. Mines per-mission discharge evidence (graded
# records) against the vLLM served by linode-4gpu-setup.sh, gold-anchored to the sealed A-next
# yardstick. Per futon6/holes/proof-mine-runner-spec.md.
#
# HOW TO RUN. Only the LLM pass needs the GPU; dossiers, gold corpus, canonical bridge and ALL
# writes stay on dev (D1 — tunnel-first, sync NOTHING). Run this script ON DEV and tunnel the box:
#     ssh -o ServerAliveInterval=30 -L 8000:localhost:8000 <box> &   # keepalive; the tunnel lesson
#     OPENAI_BASE_URL=http://localhost:8000/v1 RUNG=gold scripts/linode-proof-mine.sh
#   Kill the tunnel by EXACT pid (pgrep -x ssh), NEVER `pkill -f 8000` — it matches this script too.
#
# Rungs (RUNG env): smoke = stub/openai plumbing on a few missions · gold = the 10 A-next blind eval
#   with the D5 abort bands · full = the ~200-mission sweep (ONLY after gold PASSES).
# The box writes NOTHING to :7071 — landing is a separate gated CPU step (scripts/land_proof_mine.bb).
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-mark4-70b}"                 # the --served-model-name from linode-4gpu-setup.sh
REPO="${REPO:-$HOME/code/futon6}"
VENV="${VENV:-$HOME/code/futon6/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
LIMIT="${LIMIT:-0}"                          # 0 = all (full); a small N for smoke
RUNG="${RUNG:-smoke}"                        # smoke | gold | full
BACKEND="${BACKEND:-openai}"                 # smoke can use stub to skip the GPU entirely
OUT="${OUT:-$REPO/data/proof-mine}"

cd "$REPO"
[ -x "$PYTHON" ] || { echo "FATAL: PYTHON=$PYTHON not executable. Run scripts/linode-4gpu-setup.sh, or set PYTHON=/path/to/python."; exit 1; }

# --- gates-as-code BEFORE any spend (D2). Torch-free; catches an empty dossier / broken yardstick. ---
echo "== [D2] gates-as-code (local; no GPU) =="
"$PYTHON" scripts/check_proof_mine_gates.py || { echo "FATAL: gates FAILED — fix the dossier/yardstick before renting the box. See output above."; exit 1; }

# --- wait for vLLM (skipped for a pure-stub smoke, which needs no server) ---
if [ "$BACKEND" = "openai" ]; then
  echo "== wait for vLLM on :$PORT =="
  for _ in $(seq 1 60); do curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 5; done
  curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not serving on :$PORT — run scripts/linode-4gpu-setup.sh on the box, or SSH-tunnel the port (see header)."; exit 1; }
  echo "server up: $(curl -s "localhost:$PORT/v1/models" | head -c 200)"
  export OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=x
fi

case "$RUNG" in
  smoke)
    echo "== RUNG=smoke — plumbing (backend=$BACKEND, limit=${LIMIT:-3}) =="
    "$PYTHON" scripts/proof_mine.py --rung smoke --backend "$BACKEND" --limit "${LIMIT:-3}" --out "$OUT" --resume
    ;;
  gold)
    echo "== RUNG=gold — the 10 A-next BLIND eval with D5 abort bands (backend=$BACKEND) =="
    if ! "$PYTHON" scripts/proof_mine.py --rung gold --backend "$BACKEND" --model "$MODEL" --out "$OUT"; then
      echo "FATAL: GOLD BANDS FAILED — endpoint precision / grade agreement / witness rate out of band."
      echo "       Fix the prompt; DO NOT run the full sweep. See $OUT/proof-mine-gold-eval.json."
      exit 1
    fi
    echo "gold PASS — safe to run RUNG=full."
    ;;
  full)
    echo "== RUNG=full — the ~200-mission sweep (backend=$BACKEND) =="
    echo "   (gold must have PASSED first — the anti-cockup gate. Re-run RUNG=gold if unsure.)"
    "$PYTHON" scripts/proof_mine.py --rung full --backend "$BACKEND" --model "$MODEL" --limit "$LIMIT" --out "$OUT" --resume
    ;;
  *) echo "FATAL: unknown RUNG='$RUNG' (want smoke|gold|full)"; exit 1 ;;
esac

# --- D8: capture manifest + sha256 of every artifact BEFORE the box is decommissioned ---
echo "== [D8] manifest =="
"$PYTHON" scripts/proof_mine_manifest.py --out "$OUT" || echo "  (manifest step non-fatal: $?)"

cat <<'RUBRIC'

== Owner-review rubric (gate substance, not a PASS) — proof-mine-runner-spec.md §Owner-review ==
  - Grade split: :open-heavy on IDENTIFY missions is HEALTHY; :discharged-heavy overall = credulous
    prompt (check witnesses); >90% any single grade = degenerate.
  - Witness validity: spot-check 10 random witnesses ARE verbatim dossier spans (witness_verbatim=true).
  - Quarantine rate: a few % unresolvable is healthy; >20% = the vocabulary bridge or dossier assembly
    is broken — fix BEFORE landing anything (scripts/land_proof_mine.bb stays dry-run until reviewed).
  - Join delta: after landing, re-run §11 coverage — 119/189 should rise, the uncovered-70 shrink.
  - pair_unverified=true on records = the E-have-want pairs corpus wasn't on disk (the pilot's ⚠pair);
    grades are at mission-closure granularity. Point the runner at the pairs corpus to tighten them.
RUBRIC
echo "artifacts in $OUT: proof-mine.jsonl · proof-mine-quarantine.jsonl · proof-mine-status.json · manifest.json"
