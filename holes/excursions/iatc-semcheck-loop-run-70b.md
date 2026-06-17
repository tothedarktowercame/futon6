# IATC semcheck report — loop-run-70b

- Description-first: yes
- N/A != FAIL: yes
- Graphs: `10` (includes explicitly requested attempt graph)
- R2a rate label: lexical lower bound
- R2c warrant floor: `0.0`; loop-run-70b final spread includes 0.0; aggregate 6/28, so default is report-only until calibrated
- Overall verdict: **FAIL**

## Ground-truth Anchors

- `0706.1286`: clean anchor; R2a lexical lower bound `0.857`, R2b `:pass`.
- `0709.0248`: R2a-flagged at the proposition anchor; R2a reasons `4`.
- `0708.2185`: R2b-flagged attempt; closure status `:fail`.

## Check Summary

| paper | check | status | rate | reasons |
|---|---|---:|---:|---:|
| `0705.0452` | `:anchor-faithfulness` | `:pass` | 0.500 | 5 |
| `0705.0452` | `:closure` | `:fail` | 0.400 | 1 |
| `0705.0452` | `:warrant-resolution` | `:pass` | 0.000 | 0 |
| `0706.1286` | `:anchor-faithfulness` | `:pass` | 0.857 | 1 |
| `0706.1286` | `:closure` | `:pass` | 1.000 | 0 |
| `0706.1286` | `:warrant-resolution` | `:pass` | 0.200 | 0 |
| `0708.1921` | `:anchor-faithfulness` | `:pass` | 1.000 | 0 |
| `0708.1921` | `:closure` | `:fail` | 0.571 | 1 |
| `0708.1921` | `:warrant-resolution` | `:pass` | 0.000 | 0 |
| `0708.2067` | `:anchor-faithfulness` | `:pass` | 0.625 | 3 |
| `0708.2067` | `:closure` | `:fail` | 0.375 | 1 |
| `0708.2067` | `:warrant-resolution` | `:pass` | 0.000 | 0 |
| `0709.0248` | `:anchor-faithfulness` | `:pass` | 0.333 | 4 |
| `0709.0248` | `:closure` | `:pass` | 1.000 | 0 |
| `0709.0248` | `:warrant-resolution` | `:pass` | 0.333 | 0 |
| `0711.0473` | `:anchor-faithfulness` | `:pass` | 0.667 | 2 |
| `0711.0473` | `:closure` | `:pass` | 1.000 | 0 |
| `0711.0473` | `:warrant-resolution` | `:pass` | 0.000 | 0 |
| `0712.0724` | `:anchor-faithfulness` | `:pass` | 0.667 | 4 |
| `0712.0724` | `:closure` | `:fail` | 0.583 | 2 |
| `0712.0724` | `:warrant-resolution` | `:pass` | 0.000 | 0 |
| `0801.0199` | `:anchor-faithfulness` | `:pass` | 0.833 | 1 |
| `0801.0199` | `:closure` | `:pass` | 1.000 | 0 |
| `0801.0199` | `:warrant-resolution` | `:pass` | 0.600 | 0 |
| `0801.3843` | `:anchor-faithfulness` | `:pass` | 0.417 | 7 |
| `0801.3843` | `:closure` | `:pass` | 1.000 | 0 |
| `0801.3843` | `:warrant-resolution` | `:pass` | 0.500 | 0 |
| `0708.2185` | `:anchor-faithfulness` | `:pass` | 1.000 | 0 |
| `0708.2185` | `:closure` | `:fail` | 1.000 | 1 |
| `0708.2185` | `:warrant-resolution` | `:pass` | 0.000 | 0 |

## Paper-description profiles

### `0705.0452`

- file: `data/iatc-argument-graphs/loop-run-70b/0705.0452.edn`
- skeleton: nodes `10`, edges `3`, holes `3`, lines `[1290 1302]`
- imported terms: category, commutative, covering, descent, diagram, disjoint, equation, fold, funct, functor, groupoid, intersections, manifold, morphism, morphisms, object, objects, path, smooth, trans, ...
- reasoning edges: `3`

### `0706.1286`

- file: `data/iatc-argument-graphs/loop-run-70b/0706.1286.edn`
- skeleton: nodes `7`, edges `5`, holes `1`, lines `[333 341]`
- imported terms: arise, arose, bicategories, calmod, cat, describing, equivalence, fine, isomorphisms, issue, other, pht, problem, problems, ring, similar, situations
- reasoning edges: `5`

### `0708.1921`

- file: `data/iatc-argument-graphs/loop-run-70b/0708.1921.edn`
- skeleton: nodes `7`, edges `3`, holes `3`, lines `[679 683]`
- imported terms: inv, sigma
- reasoning edges: `3`

### `0708.2067`

- file: `data/iatc-argument-graphs/loop-run-70b/0708.2067.edn`
- skeleton: nodes `8`, edges `2`, holes `2`, lines `[389 397]`
- imported terms: beke, category, chosen, cofibrant, cofibrations, combinatorial, commutative, dense, domains, equivalences, generating, model, only, set, small, smith, square, tractable, trivial, weak
- reasoning edges: `2`

### `0709.0248`

- file: `data/iatc-argument-graphs/loop-run-70b/0709.0248.edn`
- skeleton: nodes `6`, edges `3`, holes `1`, lines `[1510 1519]`
- imported terms: cartesian, category, closed, equivalent, extensional, governing, identity, locally, parameterized, reflexivity, rules, standard, term, terms, type, types, versions
- reasoning edges: `3`

### `0711.0473`

- file: `data/iatc-argument-graphs/loop-run-70b/0711.0473.edn`
- skeleton: nodes `6`, edges `2`, holes `1`, lines `[1118 1122]`
- imported terms: arrangement, arrangements, assignment, assume, category, compatible, composable, composition, compositions, consider, consisting, double, generality, horizontal, last, loss, more, sequence, single, square, ...
- reasoning edges: `2`

### `0712.0724`

- file: `data/iatc-argument-graphs/loop-run-70b/0712.0724.edn`
- skeleton: nodes `12`, edges `3`, holes `3`, lines `[884 907]`
- imported terms: adjoint, evident, forgetful, functor, left, lifting, morphism, phi, pitchfork, thg
- reasoning edges: `3`

### `0801.0199`

- file: `data/iatc-argument-graphs/loop-run-70b/0801.0199.edn`
- skeleton: nodes `8`, edges `5`, holes `2`, lines `[386 392]`
- imported terms: cat, categories, consider, consisting, embedding, faithful, full, fully, implies, separated, subcategory, suffices, yoneda
- reasoning edges: `5`

### `0801.3843`

- file: `data/iatc-argument-graphs/loop-run-70b/0801.3843.edn`
- skeleton: nodes `12`, edges `2`, holes `1`, lines `[643 658]`
- imported terms: alpha, beta, composite, crossed, group, homomorphism, identity, maps, module, morphism, object, source, target, topological
- reasoning edges: `2`

### `0708.2185`

- file: `data/iatc-argument-graphs/loop-run-70b/.attempts/0708.2185.attempt2.edn`
- skeleton: nodes `9`, edges `6`, holes `1`, lines `[177 185]`
- imported terms: accessible, accessibly, card, cardinal, closed, cofinal, colimits, diagram, directed, embedded, embedding, full, inclusion, lambda, only, pullback, pure, regular, subcategory, subobjects, ...
- reasoning edges: `6`

