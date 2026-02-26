#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Override via env if desired.
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/gh/scratch/darktower/futon6}"
DATA_DIR="${DATA_DIR:-$SCRATCH_ROOT/data}"
OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/data/ct-validation/arxiv-paper-hg-gpu}"
NUM_SHARDS="${NUM_SHARDS:-8}"

BUNDLE_URL="http://172.236.28.208/futon6/arxiv-math-ct-handoff-2026-02-20.7z"
SHA_URL="http://172.236.28.208/futon6/arxiv-math-ct-handoff-2026-02-20.7z.sha256"
BUNDLE_FILE="${BUNDLE_FILE:-arxiv-math-ct-handoff-2026-02-20.7z}"
SHA_FILE="${SHA_FILE:-arxiv-math-ct-handoff-2026-02-20.7z.sha256}"

step() {
  echo
  echo "==> $*"
}

download_bundle() {
  step "Download handoff bundle"
  if [[ -f "$BUNDLE_FILE" && -f "$SHA_FILE" ]]; then
    echo "[skip] bundle already present: $BUNDLE_FILE"
    sha256sum -c "$SHA_FILE"
    return 0
  fi
  curl -fL -o "$BUNDLE_FILE" "$BUNDLE_URL"
  curl -fL -o "$SHA_FILE" "$SHA_URL"
  sha256sum -c "$SHA_FILE"
}

unpack_bundle() {
  step "Unpack bundle into $DATA_DIR"
  mkdir -p "$DATA_DIR"
  7z x "$BUNDLE_FILE" -o"$DATA_DIR"
}

run_arxiv_pipeline() {
  step "Run arXiv math.CT pipeline"
  if [[ "$NUM_SHARDS" -gt 1 ]]; then
    echo "[note] arXiv pipeline runs unsharded; NUM_SHARDS=$NUM_SHARDS is ignored."
  fi
  local distinctor_flags=(--run-distinctor-mit --distinctor-eprint-dir data/arxiv-math-ct-eprints)
  if [[ ! -f "$ROOT_DIR/scripts/pilot-planetmath-distinctors.py" ]]; then
    echo "[warn] pilot-planetmath-distinctors.py not found; skipping --run-distinctor-mit"
    distinctor_flags=()
  fi
  (
    cd "$ROOT_DIR"
    python3 "$ROOT_DIR/scripts/superpod-job.py" \
      --arxiv-jsonl data/arxiv-math-ct-metadata.jsonl \
      --site arxiv.math-ct \
      --output-dir "$OUT_DIR" \
      --input-dir "$SCRATCH_ROOT" \
      --limit 10000 \
      --skip-llm --skip-clustering --skip-threads --skip-expressions \
      --paper-hg-eprint-dir data/arxiv-math-ct-eprints \
      --discover-terms \
      --discover-terms-eprint-dir data/arxiv-math-ct-eprints \
      "${distinctor_flags[@]}"
  )
}

run_se_pipeline() {
  # Optional: set POSTS_XML + SE_OUTDIR to enable.
  local posts_xml="${POSTS_XML:-}"
  local se_outdir="${SE_OUTDIR:-}"
  if [[ -z "$posts_xml" || -z "$se_outdir" ]]; then
    echo "[skip] Set POSTS_XML and SE_OUTDIR to run StackExchange pipeline."
    return 0
  fi

  step "Run StackExchange pipeline (new-term spotting)"
  if [[ "$NUM_SHARDS" -gt 1 ]]; then
    (
      cd "$ROOT_DIR"
      python3 "$ROOT_DIR/scripts/superpod-shard.py" run \
        --posts-xml "$posts_xml" \
        --site math.stackexchange \
        --num-shards "$NUM_SHARDS" \
        --output-dir "$se_outdir" \
        -- \
        --discover-terms
    )
  else
    (
      cd "$ROOT_DIR"
      python3 "$ROOT_DIR/scripts/superpod-job.py" "$posts_xml" \
        --output-dir "$se_outdir" \
        --discover-terms
    )
  fi
}

run_mo_pipeline() {
  # Optional: set MO_POSTS_XML + MO_OUTDIR to enable.
  local mo_posts_xml="${MO_POSTS_XML:-}"
  local mo_outdir="${MO_OUTDIR:-}"
  if [[ -z "$mo_posts_xml" || -z "$mo_outdir" ]]; then
    echo "[skip] Set MO_POSTS_XML and MO_OUTDIR to run MathOverflow pipeline."
    return 0
  fi

  step "Run MathOverflow pipeline (new-term spotting)"
  if [[ "$NUM_SHARDS" -gt 1 ]]; then
    (
      cd "$ROOT_DIR"
      python3 "$ROOT_DIR/scripts/superpod-shard.py" run \
        --posts-xml "$mo_posts_xml" \
        --site mathoverflow.net \
        --num-shards "$NUM_SHARDS" \
        --output-dir "$mo_outdir" \
        -- \
        --discover-terms
    )
  else
    (
      cd "$ROOT_DIR"
      python3 "$ROOT_DIR/scripts/superpod-job.py" "$mo_posts_xml" \
        --site mathoverflow.net \
        --output-dir "$mo_outdir" \
        --discover-terms
    )
  fi
}

#download_bundle
#unpack_bundle
#run_arxiv_pipeline
run_se_pipeline
run_mo_pipeline

echo
echo "Done."