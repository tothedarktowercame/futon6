#!/usr/bin/env bash
# CAS-SEL-3b embedding-retrieval experiment on the Linode (the bigger box).
#
# PURPOSE. Run the bge-large recall test that OOM'd on the dev box (1.3GB load).
# It is DISCRIMINATING (final-checklist §6): does a strong text model recover the
# 3 zero-overlap steps that hotword retrieval can't reach?
#   - recovers them  -> the ceiling was MODEL SIZE; ship the embedding modality.
#   - recovers NONE  -> the ceiling is TEXT-vs-STRUCTURE (those are structural, not
#                       lexical, matches) -> evidence for the R-GCN / structure-first
#                       direction (§6). Either outcome is decisive.
#
# DESIGN. A standalone embedding job — it does NOT need the 70B and does NOT serve
# vLLM. Runs on CPU by default (DEVICE=cpu) so it never contends with a 70B that may
# be filling the GPUs at TP=4. The corpus is tiny (39 patterns + 22 fixture steps), so
# the only real cost is the model load + a ~1.3GB HF download for bge-large on first use.
# Compares bge-large vs bge-small vs MiniLM, across two pattern-text representations,
# with an embedding-collapse audit (cosine-to-mean std) on each.
#
# RUN ON the Linode (or any box with the RAM the dev box lacked), from the futon6
# checkout ($REPO). Independent of linode-4gpu-setup.sh; can run alongside it or alone.
set -euo pipefail

REPO="${REPO:-$HOME/futon6}"
VENV="${VENV:-$HOME/mark4-venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
DEVICE="${DEVICE:-cpu}"                       # cpu: don't fight vLLM for VRAM
MODELS="${MODELS:-BAAI/bge-large-en-v1.5 BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2}"
REPRS="${REPRS:-title+conclusion+hotwords full}"
OUTDIR="${OUTDIR:-/tmp/cas-sel-3b}"

cd "$REPO" 2>/dev/null || { echo "FATAL: REPO=$REPO not found"; exit 1; }
[ -x "$PYTHON" ] || { echo "FATAL: PYTHON=$PYTHON not executable. Set PYTHON=/path/to/python."; exit 1; }

echo "== ensure sentence-transformers in the venv =="
"$PYTHON" -c "import sentence_transformers" 2>/dev/null \
  || "$PYTHON" -m pip install -q sentence-transformers \
  || { echo "FATAL: could not import or install sentence-transformers in $PYTHON"; exit 1; }

# The experiment reads the committed pattern snapshot (data/cas-select/pattern-texts.json)
# when futon3 is absent, and the fixture steps from the futon6 checkout — so no futon3
# rsync is required on this box.
mkdir -p "$OUTDIR"
echo "== CAS-SEL-3b embedding recall experiment (device=$DEVICE) =="
echo "   NOTE: bge-large pulls ~1.3GB from HuggingFace on first use (needs network)."
for m in $MODELS; do
  for r in $REPRS; do
    tag="$(echo "$m" | tr '/' '_')__$r"
    echo; echo "---- model=$m repr=$r ----"
    "$PYTHON" scripts/cas_sel_3b_embed_experiment.py \
        --model "$m" --repr "$r" --device "$DEVICE" --out "$OUTDIR/$tag.json" 2>/dev/null \
      | grep -E "hotword|UNION|zero-overlap|collapse|VERDICT" \
      || echo "  (run failed for $m/$r — see $OUTDIR/$tag.json if written)"
  done
done

echo; echo "== headline table =="
"$PYTHON" - "$OUTDIR" <<'PY'
import json, sys, glob, os
rows = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
if not rows:
    print("  (no result files — all runs failed)"); raise SystemExit(0)
for f in rows:
    d = json.load(open(f)); rec = d["recall"]; acc = d["acceptance"]
    print(f"  {d['model']:40} {d['repr']:26} "
          f"hot={rec['hotword']} emb={rec['embed']} union={rec['union']} ceil={d['hotword_full_pool_ceiling']} "
          f"recovered={d['recovered_by_embed'] or 'NONE'} collapse={d['collapse_audit']['verdict']} "
          f"accept={'YES' if acc['all_zero_overlap_recovered'] else 'no'}")
PY

echo; echo "== how to read =="
echo "  bge-large accept=YES (all 3 zero-overlap recovered, union > ceiling):"
echo "    -> model size was the issue. Ship embedding CAS-SEL-3b; re-pin the honest-recall test up."
echo "  bge-large accept=no (recovers NONE even at full strength):"
echo "    -> the ceiling is text-vs-structure. Those 3 are STRUCTURAL matches. That is the"
echo "       empirical case for the R-GCN / structure-first direction (final-checklist §6)."
echo "  Also watch the collapse column: 'mild'/'COLLAPSE' on the pattern embeddings means the"
echo "  representation under-discriminates (try --repr full, or it is a model-capacity signal)."
echo
echo "== done. per-model JSON in $OUTDIR. Send-gated to Joe (box time)."
