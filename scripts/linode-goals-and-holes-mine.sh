#!/usr/bin/env bash
# GOALS-AND-HOLES MINE (應-voice) GPU run — the BACKWARD dual of linode-meme-mine.sh.
# Where that mines human→agent turns into (have,want) MEMES (methods), this mines agent→human turns —
# the 應-voice — into C-ENTRIES (the belly / Friston's C-vector, R19) for M-goals-and-holes:
#   reach       — an unstated goal the AGENT orients toward, and
#   correction  — a human reply that OVERRIDES the agent (the cleanest C-signal),
# grounded against the same retrieved candidate missions/patterns. Per holes/goals-holes-readiness.html.
#
# HOW TO RUN. Only the LLM call needs the GPU; everything else is local CPU on ~17 MB of data.
#   ► (a) TUNNEL — RECOMMENDED, sync NOTHING. Run this script ON DEV; tunnel the box's vLLM port:
#         ssh -L 8000:localhost:8000 <box> &
#         OPENAI_BASE_URL=http://localhost:8000/v1 scripts/linode-goals-and-holes-mine.sh
#       The box only serves the model; futon6 + futon3a + the transcripts all stay on dev.
#       This HOT-SWAPS onto the box already serving the forward meme run — same model, no teardown.
#   (b) ON-BOX — only if you must. Do NOT rsync futon6 (~31 GB this run never uses). Copy ONLY these
#       scripts + ../futon3a/resources/notions/minilm_{pattern,mission}_embeddings.json + ~/.claude/projects/.
#       NB the .py scripts hardcode /home/joe/code paths, so on-box also needs those paths (or a path fix).
#
# Prereq: vLLM serving (scripts/linode-4gpu-setup.sh) + the turns present in TURNS_DIR.
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-mark4-70b}"             # the --served-model-name from setup (same box as the forward run)
# Resolve the repo from THIS script's location, so it is correct on dev (/home/joe/code/futon6) AND on-box.
DEFAULT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
REPO="${REPO:-${DEFAULT_REPO:-$HOME/futon6}}"
# Prefer the in-repo venv (dev/tunnel mode); fall back to the box venv. The runner only needs stdlib.
PYTHON="${PYTHON:-$([ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || echo "$HOME/mark4-venv/bin/python")}"
TURNS_DIR="${TURNS_DIR:-$HOME/.claude/projects}"
LIMIT="${LIMIT:-0}"                     # 0 = all (agent-turn, human-reply) pairs; set small for a smoke run
OUT="${OUT:-data/c-vector/c-entries.openai.json}"

echo "== wait for vLLM on :$PORT =="
for i in $(seq 1 60); do curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 5; done
curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not serving on :$PORT (run scripts/linode-4gpu-setup.sh, or SSH-tunnel the port from the box)"; exit 1; }
echo "server up: $(curl -s localhost:$PORT/v1/models | head -c 200)"

cd "$REPO"
[ -x "$PYTHON" ] || { echo "FATAL: PYTHON=$PYTHON is not executable. Set PYTHON=/path/to/python (dev: $REPO/.venv/bin/python; box: run scripts/linode-4gpu-setup.sh)."; exit 1; }
compgen -G "$TURNS_DIR/*/*.jsonl" >/dev/null 2>&1 || { echo "FATAL: no turn transcripts in $TURNS_DIR/*/*.jsonl — rsync ~/.claude/projects here, or set TURNS_DIR."; exit 1; }

# --- GPU JOINT reason: (agent turn, human reply) × retrieved candidate missions × patterns ---
#   classifies reach|correction|neither; grounds the goal to a real candidate id; cites a verbatim span.
#   CPU retrieves candidates + filters to OPERATOR replies (inter-agent bells dropped); the 70B reasons.
echo "== GOALS-AND-HOLES mine (應-voice, vLLM $MODEL) over $TURNS_DIR =="
LIM_ARG=(); [ "$LIMIT" != "0" ] && LIM_ARG=(--limit "$LIMIT")
OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=x \
  "$PYTHON" scripts/c_mine_joint.py --backend openai --model "$MODEL" "${LIM_ARG[@]}"

# c_mine_joint already emits flat C-entries in the shared shape (no flatten step needed). The CPU channels
# (c_vector.bb: stated/incompleteness/mess) run locally and need no GPU; folding the 應-voice C-entries into
# the assembled C-vector + the c-store overlay is a local CPU follow-on (see done-message).
echo
echo "== done. Artifact: $OUT (reach + correction C-entries, each with a verbatim span + grounded_ref). =="
echo "   Fold into the belly locally:  bb scripts/c_vector.bb   (CPU channels) then merge $OUT  [follow-on]."
echo
echo "Owner review (vs the backward-pass expectations — gate substance, not a PASS):"
echo "  - SPLIT: reach ≫ correction is expected (corrections are rarer); a flood of 'correction' => INSTR too loose."
echo "  - GROUNDING: a healthy fraction of reach entries name a candidate mission/pattern (grounded_ref non-null);"
echo "    near-zero => retrieval too weak for the backward corpus."
echo "  - CORRECTION PRECISION: spot-check ≥5 correction entries — each reply_span must be a REAL human override"
echo "    ('not only' / 'not that — this' / a redirect), NOT mere agreement. Agreement-as-correction => tighten INSTR."
echo "  - OPERATOR-ONLY: confirm no entry's provenance is an inter-agent bell (no Caller: claude-N/codex-N leaked"
echo "    past the read_pairs filter). The belly is the OPERATOR's preferences."
echo "  - PROVENANCE: every C-entry carries a verbatim span (I1); spans stay on dev — $OUT is gitignored (data/*)."
