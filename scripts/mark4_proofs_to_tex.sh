#!/usr/bin/env bash
# Convert the FROZEN mark4 APM proof markdown -> clean structural LaTeX, for
# keyword extraction. Same recipe as regenerate-full-tex-safe.sh (pandoc + the
# First Proof pandoc-mathify.lua filter), but NO proof-box / colour layer and
# driven by the frozen candidate list only. Sources are never mutated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILTER="${REPO_ROOT}/scripts/pandoc-mathify.lua"
NORMALIZER="${REPO_ROOT}/scripts/normalize-math-prose.py"
SRC_DIR="/home/joe/code/futon3c/data/apm-informal-proofs"
FROZEN="/home/joe/code/storage/apm/mark4-frozen-candidates.txt"
OUT_DIR="/home/joe/code/storage/apm/mark4-tex"

mkdir -p "${OUT_DIR}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

# Stage temp copies (never touch the source markdown), normalize on the copies.
while read -r id; do
  [ -z "${id}" ] && continue
  cp "${SRC_DIR}/apm-${id}.md" "${tmp_dir}/apm-${id}.md"
done < "${FROZEN}"

python3 "${NORMALIZER}" --write --allow-in-place "${tmp_dir}"/*.md >/dev/null

n=0
for src in "${tmp_dir}"/*.md; do
  base="$(basename "${src}" .md)"
  out="${OUT_DIR}/${base}.tex"
  pandoc "${src}" \
    -f gfm-superscript-subscript \
    -t latex \
    --wrap=preserve \
    --lua-filter "${FILTER}" \
    -o "${out}"
  # Same apostrophe-artifact cleanup as the canonical regen script.
  sed -i \
    -e "s/\\\\textquotesingle\\\\textquotesingle{}/''/g" \
    -e "s/\\\\textquotesingle\\\\textquotesingle/''/g" \
    "${out}"
  n=$((n + 1))
done
echo "Converted ${n} frozen proofs -> ${OUT_DIR}/apm-*.tex"
