# Annotated + Full LaTeX Monograph

This directory now contains a two-layer manuscript:

1. **Annotated Overview** (concise, citation-rich introductions)
2. **Full Proof Drafts** (expanded problem writeups converted from `problem*-solution.md`)

## Primary entrypoint
- `first-proof-monograph.tex`

## Supporting files
- `problemN-annotated.tex` (Part I)
- `full/problemN-solution-full.tex` (Part II, generated via `pandoc`)
- `annotated-writeups.tex` (legacy overview-only build)

## Build

```bash
cd data/first-proof/latex
# Regenerate boxed files first (standoff annotations):
python3 ../../../scripts/apply-proof-boxes.py
# Then build PDF (pdflatex or xelatex both work):
pdflatex -interaction=nonstopmode first-proof-monograph.tex
pdflatex -interaction=nonstopmode first-proof-monograph.tex
```

(If `latexmk` is available, use `latexmk -pdf first-proof-monograph.tex`.)

## Regenerate full-solution LaTeX from markdown (source-safe)

```bash
cd /home/joe/code/futon6
./scripts/regenerate-full-tex-safe.sh
```

This script normalizes temporary copies and renders from those copies, so
`data/first-proof/problem*-solution.md` is never modified during typesetting.

If you intentionally want to rewrite source markdown with the normalizer, both
flags are required:

```bash
python3 scripts/normalize-math-prose.py --write --allow-in-place data/first-proof/problem*-solution.md
```

## Project-status notes
- Problem 6 is explicitly marked as containing the remaining open/conditional step.
- Problem 7 is tracked in this manuscript revision as closed per the updated rotation-route theorem chain in `problem7-solution.md`.
