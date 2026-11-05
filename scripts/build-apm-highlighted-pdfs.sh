#!/usr/bin/env bash
set -euo pipefail

input_dir=${1:-/home/joe/code/futon3c/data/apm-informal-proofs}
output_dir=${2:-/tmp/apm-highlighted-pdfs}
jobs=${JOBS:-8}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
style_dir=${MATH_PROOFREAD_STYLE_DIR:-/home/joe/code/storage/futon6/data/first-proof/latex}
texlive_root=${TEXLIVE_ROOT:-/usr/local/texlive}

resolve_xelatex() {
  if [[ -n ${XELATEX:-} ]]; then
    if [[ -x $XELATEX ]]; then
      printf '%s\n' "$XELATEX"
      return 0
    fi
    printf 'XELATEX is not executable: %s\n' "$XELATEX" >&2
    return 1
  fi

  if command -v xelatex >/dev/null 2>&1; then
    command -v xelatex
    return 0
  fi

  local candidate
  candidate=$(find "$texlive_root" -type f -o -type l 2>/dev/null |
    grep '/bin/[^/]*/xelatex$' | sort -V | tail -n 1 || true)
  if [[ -n $candidate && -x $candidate ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  printf 'Cannot find XeLaTeX; set XELATEX or TEXLIVE_ROOT (searched %s)\n' \
    "$texlive_root" >&2
  return 1
}

xelatex=$(resolve_xelatex)

if [[ ! -f "$style_dir/math-proofread-style.sty" ]]; then
  printf 'Missing math-proofread-style.sty in %s\n' "$style_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/logs"
find "$input_dir" -maxdepth 1 -type f -name '*.md' ! -name '*.failed.*' -print0 |
  sort -z |
  xargs -0 -P "$jobs" -I{} bash -c '
    src=$1
    output_dir=$2
    script_dir=$3
    style_dir=$4
    xelatex=$5
    base=$(basename "$src" .md)
    export TEXINPUTS="$style_dir:${TEXINPUTS:-}"
    common=(--lua-filter "$script_dir/pandoc-mathify.lua" --standalone
      --include-in-header "$script_dir/apm-math-highlight-header.tex"
      --pdf-engine="$xelatex" -o "$output_dir/$base.pdf")
    if pandoc "$src" -f markdown+tex_math_dollars+superscript+subscript \
        "${common[@]}" >"$output_dir/logs/$base.log" 2>&1; then
      printf "markdown\t%s\n" "$base" >"$output_dir/logs/$base.reader"
    elif pandoc "$src" -f gfm-superscript-subscript \
        "${common[@]}" >"$output_dir/logs/$base.log" 2>&1; then
      printf "gfm\t%s\n" "$base" >"$output_dir/logs/$base.reader"
    else
      printf "%s\n" "$src" >"$output_dir/logs/$base.failed"
      exit 1
    fi
  ' _ {} "$output_dir" "$script_dir" "$style_dir" "$xelatex"

pdf_count=$(find "$output_dir" -maxdepth 1 -type f -name '*.pdf' | wc -l)
failure_count=$(find "$output_dir/logs" -type f -name '*.failed' | wc -l)
printf 'Built %s PDFs; %s failures; output: %s\n' "$pdf_count" "$failure_count" "$output_dir"
test "$failure_count" -eq 0
