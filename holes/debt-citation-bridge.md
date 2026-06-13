# Debt-Citation Bridge

Input artifacts:

- `data/warp/corpus-debt.json` (`warp-corpus-debt-v2`)
- `data/warp/concordance.json`
- `data/warp/citations.json` (`30,426` edges, linkage `0.118838`)

Question: for each high-use corpus definition hole, does a paper that uses the
concept cite another corpus paper that defines it?

## Result

Positive bridges using corpus-debt provenance terms: `0`.
Positive bridges using normalized concept labels: `0`.

The corpus-debt v2 undefined status is provenance-term based: for example,
`\smcat` and `\inprod` have no `role=defined` rows. A stricter
concept-label scan finds in-corpus definitions for two labels
(`symmetric monoidal category`, `inner product`), but no using paper cites
those defining papers. The graph therefore still does not self-contain the
frontier holes.

Grounding strategy: use external nLab/Lean/PlanetMath anchors for all 18;
optionally also attach the two label-level in-corpus definers as secondary
anchors after the concordance role/provenance model is reconciled.

## Bridge Table

| concept | papers using | provenance-term definitions | concept-label definitions | citation bridges | verdict |
|---|---:|---:|---:|---:|---|
| homotopy colimit | 168 | 0 | 0 | 0 | no corpus concept-label definition |
| 2 category | 115 | 0 | 0 | 0 | no corpus concept-label definition |
| homotopy limit | 101 | 0 | 0 | 0 | no corpus concept-label definition |
| cocartesian fibration | 83 | 0 | 0 | 0 | no corpus concept-label definition |
| dg category | 76 | 0 | 0 | 0 | no corpus concept-label definition |
| Frobenius Perron dimension | 69 | 0 | 0 | 0 | no corpus concept-label definition |
| global dimension | 54 | 0 | 0 | 0 | no corpus concept-label definition |
| oplax functor | 46 | 0 | 0 | 0 | no corpus concept-label definition |
| coend | 41 | 0 | 0 | 0 | no corpus concept-label definition |
| filtered colimit | 37 | 0 | 0 | 0 | no corpus concept-label definition |
| simplex category | 35 | 0 | 0 | 0 | no corpus concept-label definition |
| module category | 32 | 0 | 0 | 0 | no corpus concept-label definition |
| pretriangulated category | 29 | 0 | 0 | 0 | no corpus concept-label definition |
| right dual | 27 | 0 | 0 | 0 | no corpus concept-label definition |
| cobar construction | 27 | 0 | 0 | 0 | no corpus concept-label definition |
| symmetric monoidal category | 26 | 0 | 2 (`0706.0711`, `0809.2517`) | 0 | label definition exists, but users do not cite it |
| inner product | 26 | 0 | 1 (`0706.0711`) | 0 | label definition exists, but users do not cite it |
| operad | 17 | 0 | 0 | 0 | no corpus concept-label definition |

## Spot Checks

There are no positive bridges to resolve. I checked the disagreement mode directly:

- `\smcat`: `26` used rows and `0` defined rows for the provenance term;
  the normalized label `symmetric monoidal category` has definitions in
  `0706.0711` and `0809.2517`, but none of the 26 users cite those papers.
- `\inprod`: `26` used rows and `0` defined rows for the provenance term;
  the normalized label `inner product` has a definition in `0706.0711`,
  but none of the 26 users cite it.
- `homotopy colimit`: `168` users via `\hocolim`; no provenance-term or
  normalized-label corpus definition exists, so no citation bridge can exist.

## Method

For each frontier concept, collect using papers from its provenance
`concordance_terms`, collect defining papers in two modes (exact provenance
term and normalized concept label), then intersect the using papers'
outgoing citation targets with each defining-paper set.
