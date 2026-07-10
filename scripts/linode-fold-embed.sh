#!/usr/bin/env bash
# E-fold-embed-pipeline Stage C+D — the GPU runner (per futon0/README-linode.md).
#
# PURPOSE. Run the fold phase-2 embedding experiment on a 4-GPU Linode: predict a
# mission's real substrate-2 sorry endpoints from its cascade. The actual experiment
# is the ABLATION — text-only (BGE) vs struct-only (GNN) vs hybrid — so we finally
# settle BGE-vs-structure FAIRLY on this task (the crux the 3 shallow numpy negatives
# could not decide). Each mode is scored vs popularity/degree + random baselines.
#
# DESIGN. A STANDALONE torch job — it does NOT serve vLLM and does NOT need the 70B
# (unlike the mark4 pipeline). The SAGE net over ~63k nodes is small: it uses ONE GPU,
# so it can share a box that is otherwise idle. Modelled on linode-bge-retrieval.sh
# (self-contained embedding job) + linode_stepper.py's capture-before-teardown.
#
# HOW TO RUN (per futon0/README-linode.md — this runs ON THE BOX, torch needs the GPU):
#   1. Provision:  StackScript 2142757 → g2-gpu-rtx4000a4-s (RTX4000 Ada ×4); nvidia-smi works.
#   2. Stage:      rsync ~/code/futon6 to the box (the data bundle data/fold-embed/ ships with it).
#   3. Run:        cd ~/futon6 && scripts/linode-fold-embed.sh
#   4. RETRIEVE:   pull the scorecards + embeddings to dev BEFORE teardown (printed at the end;
#                  capture-before-decommission — the box bills by the hour and stops==deletes).
#
# The three mode runs are INDEPENDENT and each writes its scorecard to disk the moment it
# finishes (train_fold_embed.py: scorecard-<mode>.json). A crash in mode 3 never loses modes
# 1–2 — the durability lesson from README-linode.md's meme-mine incident (no single end-of-run
# write that one bad record can throw away).
set -euo pipefail

REPO="${REPO:-$HOME/futon6}"
DATA="${DATA:-data/fold-embed}"                 # relative to REPO — the A+B bundle (nodes/edges/pairs)
MODES="${MODES:-text struct hybrid}"            # the full ablation; override for a single-mode smoke run
MODEL="${MODEL:-BAAI/bge-small-en-v1.5}"        # BGE node-text features (small = fast; -large if capacity-bound)
EPOCHS="${EPOCHS:-40}"
# Prefer a venv that already has CUDA torch (e.g. mark4-venv from a prior vLLM box); else a lean fold-venv.
VENV="${VENV:-$([ -x "$HOME/mark4-venv/bin/python" ] && echo "$HOME/mark4-venv" || echo "$HOME/fold-venv")}"
PYTHON="${PYTHON:-$VENV/bin/python}"

cd "$REPO" 2>/dev/null || { echo "FATAL: REPO=$REPO not found (rsync futon6 to the box first)"; exit 1; }
[ -f "$DATA/nodes.jsonl" ] && [ -f "$DATA/pairs.jsonl" ] \
  || { echo "FATAL: dataset bundle missing at $REPO/$DATA (build it on dev: scripts/fold_embed/mk_dataset.py)"; exit 1; }

echo "== ensure venv ($VENV): torch + sentence-transformers =="
if [ ! -x "$PYTHON" ]; then echo "   creating $VENV"; python3 -m venv "$VENV"; fi
"$PYTHON" -c 'import torch' 2>/dev/null \
  || "$PYTHON" -m pip install -q torch \
  || { echo "FATAL: could not import or install torch in $PYTHON"; exit 1; }
"$PYTHON" -c 'import sentence_transformers' 2>/dev/null \
  || "$PYTHON" -m pip install -q sentence-transformers \
  || { echo "FATAL: could not import or install sentence-transformers in $PYTHON"; exit 1; }

echo "== GPU check =="
"$PYTHON" -c 'import torch; print("   cuda:",torch.cuda.is_available(),"devices:",torch.cuda.device_count())'
"$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || echo "   WARNING: no CUDA — this will run on CPU (slow). Continuing (baselines still valid)."

echo "== fold phase-2 embedding ablation over $DATA (model=$MODEL, epochs=$EPOCHS) =="
for m in $MODES; do
  echo; echo "---- mode=$m ----"
  # scorecard-<m>.json is written by the script the instant this mode finishes → durable per-mode.
  "$PYTHON" scripts/fold_embed/train_fold_embed.py \
      --data "$DATA" --mode "$m" --model "$MODEL" --epochs "$EPOCHS" \
    || echo "  (mode=$m FAILED — see above; other modes' scorecards are already on disk)"
done

echo; echo "== ablation summary + verdict =="
"$PYTHON" - "$DATA" <<'PY'
import json, sys, os
d = sys.argv[1]
scores = {}
for m in ("text", "struct", "hybrid"):
    p = os.path.join(d, f"scorecard-{m}.json")
    if os.path.exists(p):
        scores[m] = json.load(open(p))
if not scores:
    print("  (no scorecards — all modes failed)"); raise SystemExit(0)
# popularity/random baselines are identical across modes; read from whichever ran.
any_s = next(iter(scores.values()))
pop = any_s.get("popularity", {}).get("recall@20", 0.0)
rnd = any_s.get("random", {}).get("recall@20", 0.0)
print(f"  {'model':10} {'recall@20':>10} {'MRR':>7}")
print(f"  {'popularity':10} {pop:>10} {any_s.get('popularity',{}).get('MRR',0):>7}   (degree baseline)")
print(f"  {'random':10} {rnd:>10} {any_s.get('random',{}).get('MRR',0):>7}")
best_m, best_r = None, -1.0
for m in ("text", "struct", "hybrid"):
    if m not in scores: continue
    own = scores[m].get(m, {})
    r = own.get("recall@20", 0.0)
    print(f"  {m:10} {r:>10} {own.get('MRR',0):>7}"
          + ("   (BGE-text baseline)" if m == "text" else ""))
    if r > best_r: best_m, best_r = m, r
text_r = scores.get("text", {}).get("text", {}).get("recall@20", 0.0)
print()
print(f"  winner: {best_m} (recall@20={best_r})")
# Success criterion (E-fold-embed-pipeline): the ablation winner beats popularity AND BGE-text-only.
beats_pop = best_r > pop
beats_text = best_m != "text" and best_r > text_r
if beats_pop and beats_text:
    print(f"  VERDICT: structure/hybrid ADDS signal — beats popularity ({pop}) AND BGE-text ({text_r}). "
          "Ansatz has legs → Stage E laptop export.")
elif best_m == "text" and text_r > pop:
    print(f"  VERDICT: TEXT wins (>{pop} popularity); structure adds nothing here on THIS task. "
          "Measured text-vs-structure ceiling — a real result, not a null hack.")
elif not beats_pop:
    print(f"  VERDICT: nothing beats popularity ({pop}) — the coarse used-var/ns ground truth may be "
          "popularity-flattered; re-check hard negatives + the gold empirical-sorry eval set.")
else:
    print("  VERDICT: mixed — inspect the per-mode numbers above against the gold eval set.")
PY

echo; echo "== CAPTURE BEFORE DECOMMISSION (README-linode.md — the box stops==deletes, bills hourly) =="
echo "   Pull these to dev BEFORE teardown, verify, THEN delete the box:"
echo "     $DATA/scorecard-{text,struct,hybrid}.json   (the fair BGE-vs-structure verdict)"
echo "     $DATA/embeddings-{struct,hybrid}.pt         (trained node vectors → Stage E laptop-index.npz)"
echo "   From dev:"
echo "     rsync -avz root@\$BOX:'futon6/$DATA/scorecard-*.json futon6/$DATA/embeddings-*.pt' \\"
echo "       ~/code/futon6/$DATA/    # verify the 3 scorecards land, THEN teardown"
echo
echo "== done. Next: Stage E — freeze the winner's embeddings → laptop dot-product fold (sub-second)."
