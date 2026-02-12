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
xelatex -interaction=nonstopmode first-proof-monograph.tex
xelatex -interaction=nonstopmode first-proof-monograph.tex
```

(If `latexmk` is available, use `latexmk -xelatex first-proof-monograph.tex`.)

## Regenerate full-solution LaTeX from markdown

```bash
cd /home/joe/code/futon6
mkdir -p data/first-proof/latex/full
for n in 1 2 3 4 5 6 7 8 9 10; do
  pandoc data/first-proof/problem${n}-solution.md -f gfm -t latex --wrap=preserve \
    -o data/first-proof/latex/full/problem${n}-solution-full.tex
done
```

## Project-status notes
- Problem 6 is explicitly marked as containing the remaining open/conditional step.
- Problem 7 is tracked in this manuscript revision as closed per the updated rotation-route theorem chain in `problem7-solution.md`.
