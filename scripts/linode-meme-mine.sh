#!/usr/bin/env bash
# MEME-MINE GPU run — mirrors linode-4gpu-run.sh. Mines human→agent turns into (have,want) memes
# (M-operational-vocabulary): the GPU Layer-2 (extract+resolve+cite) against the vLLM served by
# linode-4gpu-setup.sh, then the CPU consume tail (endpoint-identity bridge → moves, floor/cert,
# concept-tag). Per holes/meme-mine-runner-spec.md.
#
# HOW TO RUN. Only Layer 2 (the LLM call) needs the GPU; everything else is local CPU on ~17 MB of data.
#   ► (a) TUNNEL — RECOMMENDED, sync NOTHING. Run this script ON DEV; tunnel the box's vLLM port:
#         ssh -L 8000:localhost:8000 <box> &
#         OPENAI_BASE_URL=http://localhost:8000/v1 scripts/linode-meme-mine.sh
#       The box only serves the model (it pulls weights from HF itself, per README-linode.md).
#       futon6 + futon3a + the transcripts all stay on dev. THIS IS ALMOST ALWAYS WHAT YOU WANT.
#   (b) ON-BOX — only if you must. Do NOT rsync futon6 (it's ~31 GB of corpora+venv this run never uses).
#       Copy ONLY: these scripts + data/{diffsub-scopes.json,diffsub-moves-mined.edn,capability-graph.json}
#       + ../futon3a/resources/notions/minilm_{pattern,mission}_embeddings.json + ~/.claude/projects/ (turns).
#       NB the .py scripts hardcode /home/joe/code paths, so on-box also needs those paths (or a path fix).
#
# Prereq: vLLM serving (scripts/linode-4gpu-setup.sh) + the turns present in TURNS_DIR.
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-mark4-70b}"            # the --served-model-name from setup
REPO="${REPO:-$HOME/futon6}"
VENV="${VENV:-$HOME/mark4-venv}"       # same default as linode-4gpu-setup.sh
PYTHON="${PYTHON:-$VENV/bin/python}"   # the runner only needs an OpenAI-compatible client + stdlib
TURNS_DIR="${TURNS_DIR:-$HOME/.claude/projects}"
LIMIT="${LIMIT:-0}"                    # 0 = all human→agent asks; set small for a smoke run
MEMES="${MEMES:-data/meme-mine/resolved-memes.openai.json}"
RUN_CONSUME="${RUN_CONSUME:-1}"        # set 0 to skip the CPU consume tail

echo "== wait for vLLM on :$PORT =="
for i in $(seq 1 60); do curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 5; done
curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not serving on :$PORT (run scripts/linode-4gpu-setup.sh, or SSH-tunnel the port from the box)"; exit 1; }
echo "server up: $(curl -s localhost:$PORT/v1/models | head -c 200)"

cd "$REPO"
[ -x "$PYTHON" ] || { echo "FATAL: PYTHON=$PYTHON is not executable. Run scripts/linode-4gpu-setup.sh first, or launch with PYTHON=/path/to/python."; exit 1; }
compgen -G "$TURNS_DIR/*/*.jsonl" >/dev/null 2>&1 || { echo "FATAL: no turn transcripts in $TURNS_DIR/*/*.jsonl — rsync ~/.claude/projects here, or set TURNS_DIR."; exit 1; }

# --- GPU JOINT reason: turn × retrieved candidate missions × patterns ---
#   grounds endpoints to real ids · characterizes pattern-applications · composes cascades · proposes new
#   patterns (R17). CPU retrieves candidates; the 70B does the joint reasoning. (meme_mine_runner.py is the
#   turns-only fallback if you want extraction without the mission/pattern context.)
echo "== MEME-MINE joint (vLLM $MODEL) over $TURNS_DIR =="
LIM_ARG=(); [ "$LIMIT" != "0" ] && LIM_ARG=(--limit "$LIMIT")
OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=x \
  "$PYTHON" scripts/meme_mine_joint.py --backend openai --model "$MODEL" "${LIM_ARG[@]}"
# flatten the joint records' .memes into the flat resolved-memes shape the consume tail bridges
"$PYTHON" - <<'PY'
import json
src, dst = "data/meme-mine/joint-memes.openai.json", "data/meme-mine/resolved-memes.openai.json"
try: j = json.load(open(src))
except FileNotFoundError: j = []
flat = [{"id": r["id"], "ask": r["ask"], "provenance": r.get("provenance", {}), "meme": m}
        for r in j for m in r.get("memes", [])]
json.dump(flat, open(dst, "w"), indent=2)
print(f"flattened {len(flat)} memes from {len(j)} joint records -> {dst}")
PY

# --- CPU consume tail (NON-FATAL): the resolved-memes are the precious artifact and already saved. ---
# Each step guarded so a tooling hiccup never masks a completed GPU mine; the script still exits 0.
if [ "$RUN_CONSUME" = "1" ]; then
  echo
  echo "== consume tail over $MEMES (non-fatal; memes above are already saved) =="
  [ -f "$MEMES" ] || { echo "  WARN: $MEMES not found — did the run write it? skipping consume."; exit 0; }

  echo "-- [1/3] endpoint-identity bridge: memes → meme-grounded rollout moves --"
  "$PYTHON" scripts/meme_consume.py --memes "$MEMES" \
    || echo "  (meme_consume exit $? — non-zero is a finding, not a tool abort)"

  echo "-- [2/3] actionability floor + per-mission action certificate --"
  "$PYTHON" scripts/meme_consume.py --floor \
    || echo "  (floor exit $? — non-zero is a finding)"

  echo "-- [3/3] concept-tag (noun axis + the turn→mission routing index) --"
  "$PYTHON" scripts/mission_concept_tag.py \
    || echo "  (concept-tag exit $? — non-zero is a finding)"
fi

echo
echo "== done. Artifacts in data/meme-mine/: joint-memes.openai.json (memes+pattern-apps+cascades+new-patterns) ·"
echo "   resolved-memes.openai.json · diffsub-moves-meme.edn · action-cert.json · concept-index.json. Owner review (vs prereg): =="
echo "  - P5 grounding: >=40% of turns ground to >=1 candidate mission (joint runner reports recall); <15% => retrieval/grounding too weak."
echo "  - P1 tiers: endpoint tiers ~ contextual 59% / named 23% / unsupported 18% (hand-sample); wildly off => tune meme_mine_joint.py INSTR."
echo "  - P6 (R17): spot-check >=10 NEW-pattern proposals — plausible, non-redundant heuristics, not hallucinations."
echo "  - spot-check >=3 meme-grounded moves vs what the ask said; the 82 ask-covered missions upgrade in action-cert.json."
echo "  - feed the meme-grounded move-set into the rollout/act-gate (futon2) to close R16 criterion (2) with real provenance."
