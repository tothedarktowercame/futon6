#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_DIR="${REPO_ROOT}/data/first-proof"
OUT_DIR="${SRC_DIR}/latex/full"
NORMALIZER="${REPO_ROOT}/scripts/normalize-math-prose.py"
FILTER="${REPO_ROOT}/scripts/pandoc-mathify.lua"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}"
cp "${SRC_DIR}"/problem*-solution.md "${tmp_dir}/"

# Normalize only temporary copies, never the source markdown.
python3 "${NORMALIZER}" --write --allow-in-place "${tmp_dir}"/problem*-solution.md >/dev/null

for src in "${tmp_dir}"/problem*-solution.md; do
  base="$(basename "${src}" .md)"
  out="${OUT_DIR}/${base}-full.tex"
  pandoc "${src}" \
    -f gfm-superscript-subscript \
    -t latex \
    --wrap=preserve \
    --lua-filter "${FILTER}" \
    -o "${out}"
  # Normalize paired apostrophe artifacts from pandoc into TeX right-double-quotes.
  sed -i \
    -e "s/\\\\textquotesingle\\\\textquotesingle{}/''/g" \
    -e "s/\\\\textquotesingle\\\\textquotesingle/''/g" \
    -E -e "s/\\\\\\(([^)]*),''\\\\\\)/\\\\(\\1\\\\),''/g" \
    "${out}"
done

echo "Regenerated ${OUT_DIR}/problem*-solution-full.tex from temporary normalized copies."
