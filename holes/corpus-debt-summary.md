# Corpus Debt Summary

Generated from `data/warp/corpus-debt.json` (`warp-corpus-debt-v2`) after the
WARP concordance and shuttle coverage pass.

## Payoff

The high-use concept frontier is not missing from external knowledge sources:
`external_debt_frontier` is empty. The useful debt is in-corpus definition debt:
papers repeatedly use these concepts without defining them inline, while nLab,
Lean, or PlanetMath already provide anchors for grounding.

## Top Corpus Definition Holes

These are concepts used across many papers and defined in no processed corpus
paper. The raw TeX/macros are provenance only; the frontier terms are concept
labels.

| # | concept | papers using | uses | in-corpus gap | external anchor |
|---:|---|---:|---:|---|---|
| 1 | homotopy colimit | 168 | 1363 | no corpus definition | nLab: `homotopy colimit` |
| 2 | 2 category | 115 | 2216 | no corpus definition | nLab: `nlab-300`; PlanetMath: `18E05-2category.tex` |
| 3 | homotopy limit | 101 | 664 | no corpus definition | nLab: `homotopy limit` |
| 4 | cocartesian fibration | 83 | 1156 | no corpus definition | nLab: `cocartesian fibration` |
| 5 | dg category | 76 | 1429 | no corpus definition | nLab: `dg-category` |
| 6 | Frobenius Perron dimension | 69 | 1682 | no corpus definition | nLab: `Frobenius-Perron dimension` |
| 7 | global dimension | 54 | 503 | no corpus definition | nLab: `global dimension`; PlanetMath: `13D05-GlobalDimension.tex` |
| 8 | oplax functor | 46 | 1089 | no corpus definition | Lean: `OplaxFunctor`; nLab: `oplax functor` |
| 9 | coend | 41 | 752 | no corpus definition | nLab: `coend` |
| 10 | filtered colimit | 37 | 438 | no corpus definition | nLab: `filtered colimit` |
| 11 | simplex category | 35 | 353 | no corpus definition | Lean: `SimplexCategory`; nLab: `simplex category` |
| 12 | module category | 32 | 566 | no corpus definition | nLab: `module category` |
| 13 | pretriangulated category | 29 | 528 | no corpus definition | nLab: `pretriangulated category` |
| 14 | right dual | 27 | 714 | no corpus definition | Lean: `HasRightDual` |
| 15 | cobar construction | 27 | 694 | no corpus definition | nLab: `cobar construction` |
| 16 | symmetric monoidal category | 26 | 458 | no corpus definition | nLab: `symmetric monoidal category` |
| 17 | inner product | 26 | 374 | no corpus definition | nLab: `inner product`; PlanetMath: `11E39-InnerProduct.tex` |
| 18 | operad | 17 | 142 | no corpus definition | nLab: `operad` |

## Well-Covered Core

These terms are both used and defined in the processed corpus, and also have
external grounding in at least one shuttle layer.

| concept | papers using | papers defining | external anchor |
|---|---:|---:|---|
| category | 17 | 16 | Lean: `Category`; nLab: `category`; PlanetMath: `18A05-Category.tex` |
| functor | 16 | 15 | Lean: `Functor`; nLab: `functor` |
| algebra | 10 | 9 | Lean: `Algebra`; nLab: `algebra`; PlanetMath: `20C99-Algebra.tex` |
| isomorphism | 9 | 8 | nLab: `isomorphism`; PlanetMath: `54A05-Isomorphism.tex` |
| object | 9 | 8 | nLab: `object` |
| space | 9 | 8 | nLab: `space` |
| structure | 9 | 7 | Lean: `Structure`; nLab: `structure`; PlanetMath: `03C07-Structure.tex` |
| morphism | 10 | 6 | Lean: `morphism`; nLab: `morphism` |
| monoidal | 7 | 6 | Lean: `Monoidal` |
| groupoid | 6 | 5 | Lean: `Groupoid`; nLab: `infinity-groupoid`; PlanetMath: `20N02-Groupoid.tex` |

## Spot Checks

- `homotopy colimit` comes from the raw concordance macro `\hocolim`, but the
  report resolves it to a concept and anchors it to nLab.
- `oplax functor` comes from `\Oplax` / `\oplax`, and is covered by both Lean
  (`OplaxFunctor`) and nLab.
- `simplex category` comes from `\Deltaop`; the concept is covered by Lean
  (`SimplexCategory`) and nLab.

## Method

Stream the WARP concordance term-by-term, filter raw TeX controls to concept
aliases, count used-but-not-defined terms by paper breadth, and cross-check each
candidate against exact Lean/mathlib, exact PlanetMath filename, and nLab page
coverage.
