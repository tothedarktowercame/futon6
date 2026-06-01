#!/usr/bin/env bash
#
# setup-ct-run.sh — one-command setup + invocation for the math.CT NER/structure run.
#
# Fetches the consolidated handoff dir from linode-chicago in ONE rsync
# (~/ct-handoff/ -> data/ + eprint bundle), verifies + extracts it, and runs the
# GPU Stage-5/6 pipeline on the math.CT slice.
#
# Verified against futon6 @ b9e149a (2026-05-31): every flag below exists in
# scripts/superpod-job.py argparse; the three stale flags from the earlier
# hand-written invocation (--msc-prior / --se-corpus-prior / --update-msc-prior)
# have been REMOVED from the script and are dropped here.
#
# Usage:
#   bash scripts/setup-ct-run.sh            # setup + run
#   bash scripts/setup-ct-run.sh --setup-only
#   bash scripts/setup-ct-run.sh --run-only
#
# Prereqs assumed: run from a checkout at ~/code/futon6 ; ssh host 'linode-chicago'
# configured (HostName 172.236.108.82, port 2222, user rob) ; nnexus at ~/code/nnexus.
set -euo pipefail

REPO="${REPO:-$HOME/code/futon6}"
STORE="${STORE:-$HOME/code/storage/futon6/data}"
REMOTE="${REMOTE:-linode-chicago}"
HANDOFF="arxiv-math-ct-handoff-2026-02-20.7z"

MODE="${1:-all}"

do_setup() {
  echo "== [1/4] fetch consolidated handoff (ONE rsync) =="
  cd "$REPO"
  mkdir -p "$STORE"
  # Everything Rob needs lives in ONE dir on the linode: ~/ct-handoff/, laid out
  # as data/... (repo-mirrored) + bundle/ (the eprint+metadata 7z) + MANIFEST.txt.
  # One transfer, resumable (--partial), only copies what changed.
  rsync -avh --partial "$REMOTE:ct-handoff/" "$STORE/handoff/"

  echo "== [2/4] place supplementary files into repo =="
  mkdir -p data/dictionary
  cp -v "$STORE/handoff/data/ner-kernel-clean.tsv"             data/ner-kernel-clean.tsv
  cp -v "$STORE/handoff/data/ct-term-prior.json"              data/ct-term-prior.json
  cp -v "$STORE/handoff/data/dictionary/entries-pm-seed.edn"   data/dictionary/entries-pm-seed.edn
  cp -v "$STORE/handoff/data/dictionary/entries-nlab-seed.edn" data/dictionary/entries-nlab-seed.edn

  echo "== [3/4] verify + extract eprint+metadata bundle =="
  ( cd "$STORE/handoff/bundle" && sha256sum -c "$HANDOFF.sha256" ) || {
    echo "   !! sha256 mismatch on $HANDOFF — aborting"; exit 1; }
  if [ -d "$STORE/arxiv-math-ct-eprints" ] && [ -e "$STORE/arxiv-math-ct-metadata.jsonl" ]; then
    echo "   already extracted (skip)"
  else
    7z x "$STORE/handoff/bundle/$HANDOFF" -o"$STORE"
  fi

  echo "== [4/4] sanity: nnexus present for ../nnexus/... resolution =="
  if [ -d "$HOME/code/nnexus" ]; then echo "   have ~/code/nnexus"; else
    echo "   !! ~/code/nnexus missing — clone it or edit the two --discover-terms-nnexus-* flags"
    echo "      git clone https://github.com/dginev/nnexus.git ~/code/nnexus"
  fi
}

do_run() {
  cd "$REPO"
  echo "== running math.CT GPU pipeline =="
  python3 scripts/superpod-job.py \
    --arxiv-jsonl "$STORE/arxiv-math-ct-metadata.jsonl" \
    --site arxiv.math-ct \
    --output-dir "$HOME/code/storage/arxiv-paper-hg-gpu-ct-5k" \
    --limit 5000 \
    --paper-eprint-dir "$STORE/arxiv-math-ct-eprints" \
    --ner-kernel data/ner-kernel-clean.tsv \
    --discover-terms \
    --discover-structures \
    --discover-terms-eprint-dir "$STORE/arxiv-math-ct-eprints" \
    --discover-terms-pm-seed data/dictionary/entries-pm-seed.edn \
    --discover-terms-nlab-seed data/dictionary/entries-nlab-seed.edn \
    --discover-terms-nnexus-stopwords ../nnexus/lib/NNexus/StopWordList.pm \
    --discover-terms-nnexus-snapshot ../nnexus/lib/NNexus/resources/database/snapshot-6-2014.sqlite \
    --discover-terms-collocation-prior data/ct-term-prior.json \
    --stage6-backend codex
  echo "== done. results in ~/code/storage/arxiv-paper-hg-gpu-ct-5k =="
  echo "   collocation gate: see candidate-new-terms-summary.json ->"
  echo "   collocation_gate_enabled / collocation_rejected_terms"
}

case "$MODE" in
  --setup-only) do_setup ;;
  --run-only)   do_run ;;
  all|"")       do_setup; do_run ;;
  *) echo "usage: $0 [--setup-only|--run-only]"; exit 2 ;;
esac
