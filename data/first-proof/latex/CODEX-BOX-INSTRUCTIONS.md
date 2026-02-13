# How to Work with Proof-Layer Boxes (for Codex)

## The System

The monograph uses **coloured tcolorbox environments** to separate proof
content from process annotations (dead paths, revisions, open obligations,
process notes). These boxes are managed via a **standoff annotation layer**
so that you can freely edit the clean `.tex` files without worrying about
breaking box environments.

## Architecture

```
full/*.tex                  ← You edit these (clean, no boxes)
  + standoff-boxes.json     ← Annotation manifest (25 entries)
  → scripts/apply-proof-boxes.py
  → full-boxed/*.tex        ← Generated output (gitignored)
  → pdflatex builds from full-boxed/
```

**Key rule: never add `\begin{processnote}`, `\begin{deadpath}`,
`\begin{revision}`, or `\begin{openobligation}` directly to `full/*.tex`.**
The script handles all box insertion.

## The Four Box Types

| Environment | Color | Purpose |
|---|---|---|
| `processnote` | Blue | Confidence, status, corpus references |
| `deadpath` | Grey | Abandoned approaches, wrong turns |
| `revision` | Yellow | Corrections applied during sprint |
| `openobligation` | Red | Gaps that remain open |

## How Annotations Work

The manifest (`standoff-boxes.json`) has 25 entries. Each maps a location
in a clean `.tex` file to a box type. Two modes:

### Section mode
Wraps an entire `\subsection{...}` or `\subsubsection{...}`:

```json
{
  "file": "full/problem7-solution-full.tex",
  "box": "deadpath",
  "title": "Approach II: Equivariant Surgery (Reflection Route)",
  "mode": "section",
  "heading": "Approach II: Equivariant surgery"
}
```

The script finds `\subsection{...Approach II: Equivariant surgery...}`,
replaces the heading with `\begin{deadpath}[title={...}]`, and inserts
`\end{deadpath}` before the next heading at the same or higher level.

**Important:** The `heading` field is matched as a **substring** of the
`\sub*section{...}` argument. If you rename a section heading, the
annotation will silently fail. The script reports warnings for unfound
headings — check the output.

### Block mode
Wraps a text range between two markers:

```json
{
  "file": "full/problem4-solution-full.tex",
  "box": "openobligation",
  "title": "What Remains Open for $n \\ge 4$",
  "mode": "block",
  "start_marker": "\\textbf{What remains open for",
  "end_before": "\\subsubsection{8. Numerical evidence}"
}
```

The script finds the line containing `start_marker` and inserts
`\begin{openobligation}[title={...}]` before it, then finds the line
containing `end_before` and inserts `\end{openobligation}` before it.

## What You Need to Do After Editing

After editing any `full/*.tex` file, regenerate the boxed files:

```bash
python3 scripts/apply-proof-boxes.py
```

This outputs to `full-boxed/` which is what the monograph includes.
The script reports how many annotations were applied. **If the count
drops below 25, a heading or marker was broken by your edit.**

## If You Rename a Section Heading

If you need to rename a heading that's used as an annotation anchor:

1. Make the edit in `full/*.tex`
2. Update the corresponding `heading` or `start_marker`/`end_before`
   field in `standoff-boxes.json`
3. Re-run `python3 scripts/apply-proof-boxes.py`
4. Verify 25/25 annotations applied

## If You Want to Add a New Box

Add an entry to the `annotations` array in `standoff-boxes.json`:

```json
{
  "file": "full/problemN-solution-full.tex",
  "box": "processnote",
  "title": "Your Title Here",
  "mode": "section",
  "heading": "unique substring of the section heading"
}
```

Then re-run the script.

## Quick Reference

```bash
# Apply annotations (do this after every edit to full/*.tex)
python3 scripts/apply-proof-boxes.py

# Dry run (see what would be applied without writing)
python3 scripts/apply-proof-boxes.py --dry-run

# Verify balanced environments in output
for f in data/first-proof/latex/full-boxed/*.tex; do
  echo "$f:"
  grep -c 'begin{deadpath}\|begin{revision}\|begin{openobligation}\|begin{processnote}' "$f" || echo "  0"
  grep -c 'end{deadpath}\|end{revision}\|end{openobligation}\|end{processnote}' "$f" || echo "  0"
done
```

## Current Annotation Inventory (25 total)

| Problem | processnote | revision | openobligation | deadpath |
|---------|------------|----------|----------------|----------|
| P1 | 1 (corpus refs) | | | |
| P2 | 2 (confidence, refs) | 1 (universality gap) | | |
| P3 | 1 (confidence) | | | |
| P4 | 2 (numerical, refs) | | 1 (n>=4 open) | |
| P5 | 2 (confidence, research) | 1 (correction) | 1 (scope) | |
| P6 | 1 (status) | | 2 (dependency, conditional) | |
| P7 | 2 (status, parity) | | | 3 (approaches II, III, deprioritized) |
| P8 | 2 (confidence, refs) | 1 (corrections v1) | | |
| P9 | 1 (corpus refs) | | | |
| P10 | 1 (corpus refs) | | | |
| **Total** | **15** | **3** | **4** | **3** |
