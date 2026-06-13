# Corpus Debt Summary

Generated from `data/warp/corpus-debt.json` (`warp-corpus-debt-v2`) after refreshing `data/warp/concordance.json` against the current golden corpus.

## Snapshot

- Concordance terms scanned: `173109`
- Reportable terms: `3451`
- External-uncovered debt candidates after concept filtering: `0`
- Externally anchored corpus-undefined concept candidates: `33`

## Payoff

After concept filtering, the high-confidence external-uncovered frontier is empty. The useful frontier is corpus definition debt: concepts used repeatedly in the processed corpus without an in-corpus definition, but already anchored in Lean, nLab, or PlanetMath.

## Top Corpus Definition Holes

| # | concept | papers using | uses | in-corpus gap | external anchor |
|---:|---|---:|---:|---|---|
| 1 | homotopy colimit | 168 | 1363 | no corpus definition | nLab: `homotopy colimit` |
| 2 | 2 category | 113 | 2099 | no corpus definition | nLab: `nlab-300`; PlanetMath: `18E05-2category.tex` |
| 3 | homotopy limit | 101 | 664 | no corpus definition | nLab: `homotopy limit` |
| 4 | cocartesian fibration | 83 | 1156 | no corpus definition | nLab: `cocartesian fibration` |
| 5 | dg category | 76 | 1429 | no corpus definition | nLab: `dg-category` |
| 6 | Frobenius Perron dimension | 69 | 1682 | no corpus definition | nLab: `Frobenius-Perron dimension` |
| 7 | global dimension | 53 | 502 | no corpus definition | nLab: `global dimension`; PlanetMath: `13D05-GlobalDimension.tex` |
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
| 19 | symmetric group | 10 | 153 | no corpus definition | nLab: `symmetric group`; PlanetMath: `20B30-SymmetricGroup.tex` |
| 20 | classifying space | 8 | 92 | no corpus definition | nLab: `classifying space` |

## External-Uncovered Frontier

No high-confidence concept rows remain after filtering quantifier fragments, TeX script artifacts, and formula-shape terms.

## Well-Covered Core

| concept | papers using | papers defining | external anchor |
|---|---:|---:|---|
| category | 232 | 223 | Lean: `Category`; nLab: `category`; PlanetMath: `18A05-Category.tex` |
| functor | 200 | 172 | Lean: `Functor`; nLab: `functor` |
| isomorphism | 170 | 117 | nLab: `isomorphism`; PlanetMath: `54A05-Isomorphism.tex` |
| algebra | 122 | 104 | Lean: `Algebra`; nLab: `algebra`; PlanetMath: `20C99-Algebra.tex` |
| morphism | 165 | 83 | Lean: `morphism`; nLab: `morphism` |
| object | 205 | 82 | nLab: `object` |
| monoidal | 86 | 81 | Lean: `Monoidal` |
| space | 115 | 78 | nLab: `space` |
| equivalence | 83 | 78 | Lean: `Equivalence`; nLab: `equivalence` |
| module | 95 | 72 | Lean: `Module`; nLab: `∞-module` |

## Spot Checks

- `homotopy colimit`: `168` papers, `1363` uses, `defined_papers=0`; provenance `\hocolim`; anchor nLab: `homotopy colimit`.
- `dg category`: `76` papers, `1429` uses, `defined_papers=0`; provenance `\DGCat, \dgCat, \dgcat`; anchor nLab: `dg-category`.
- `cocartesian fibration`: `83` papers, `1156` uses, `defined_papers=0`; provenance `\Cocart, \coCart, \cocart`; anchor nLab: `cocartesian fibration`.

## Method

Stream the WARP concordance term-by-term, map known math notation macros to concept labels, filter raw TeX/layout/formula fragments, count used-but-not-defined terms by paper breadth, and cross-check each concept against exact Lean/mathlib, PlanetMath filename, and nLab page coverage.
