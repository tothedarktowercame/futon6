# Debt-Citation Bridge

Input artifacts:

- `data/warp/corpus-debt.json` (`warp-corpus-debt-v2`)
- `data/warp/concordance.json`
- `data/warp/citations.json` (`30,426` edges, linkage `0.118838`)

Question: for each high-use corpus definition hole, does a paper that uses the
concept cite another corpus paper that defines it?

## Result

No positive citation bridges were found for the 18 corpus-undefined frontier
concepts. The holes are therefore not self-contained via the current citation
graph and concordance definitions. They should be grounded to their external
nLab/Lean/PlanetMath anchors rather than resolved by following corpus citations.

## Bridge Table

| concept | papers using | corpus defining papers | citation bridges found | verdict |
|---|---:|---:|---:|---|
| homotopy colimit | 168 | 0 | 0 | external dependency |
| 2 category | 115 | 0 | 0 | external dependency |
| homotopy limit | 101 | 0 | 0 | external dependency |
| cocartesian fibration | 83 | 0 | 0 | external dependency |
| dg category | 76 | 0 | 0 | external dependency |
| Frobenius Perron dimension | 69 | 0 | 0 | external dependency |
| global dimension | 54 | 0 | 0 | external dependency |
| oplax functor | 46 | 0 | 0 | external dependency |
| coend | 41 | 0 | 0 | external dependency |
| filtered colimit | 37 | 0 | 0 | external dependency |
| simplex category | 35 | 0 | 0 | external dependency |
| module category | 32 | 0 | 0 | external dependency |
| pretriangulated category | 29 | 0 | 0 | external dependency |
| right dual | 27 | 0 | 0 | external dependency |
| cobar construction | 27 | 0 | 0 | external dependency |
| symmetric monoidal category | 26 | 2 (`0706.0711`, `0809.2517`) | 0 | corpus definition exists, but is not cited by users |
| inner product | 26 | 1 (`0706.0711`) | 0 | corpus definition exists, but is not cited by users |
| operad | 17 | 0 | 0 | external dependency |

## Spot Checks

There were no bridges to resolve, so the positive-bridge gate is inapplicable.
I checked the negative result directly:

- `homotopy colimit`: `168` using papers via `\hocolim`; no concordance row
  defines `homotopy colimit`, so no cited corpus definition can exist.
- `oplax functor`: `46` using papers via `\Oplax` / `\oplax`; no concordance row
  defines `oplax functor`, so no cited corpus definition can exist.
- `symmetric monoidal category`: `26` using papers and two defining corpus
  papers (`0706.0711`, `0809.2517`), but none of the outgoing citation edges
  from the using papers target either defining paper.

## Method

For each frontier concept, collect using papers from its provenance
`concordance_terms`, collect defining papers by normalized concept label in the
concordance (`role=defined`), then intersect the using papers' outgoing
citation targets with the defining-paper set.
