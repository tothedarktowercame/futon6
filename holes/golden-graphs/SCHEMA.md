# Golden anatomy graph schema — v1

Controlled vocabulary for the golden graphs (and, downstream, the
recognizer registry — recognizers = scopes, golden-walk-ledger.md).
Normalization is applied by `scripts/normalize-golden-graphs.bb`:
original author kinds are preserved in `:kind-original` when mapped.

## Edge kinds (controlled, 22)

### Expression layer (R1/R9/R14)
| kind | meaning | absorbs |
|---|---|---|
| `envelope` | $-span/display contains content | — |
| `parses` | expression decomposes into subterms / operator applications | `applies`, `applies-operator`, `subscript-parametrizes` |
| `built-from` | compound object assembled from parts | — |

### Binding layer (R2/R5–R8/R11)
| kind | meaning | absorbs |
|---|---|---|
| `macro-def` | lexical macro definition (R5) | — |
| `bind` | semantic binder introduces symbol w/ type+ambient (R2) | `exists`, `considers-as` |
| `defines` | definition site binds term/concept (`:=`, definition env, "given by", "denote") (R6/W1) | `display-definition`, `definition`, `denotes`, `designates`, `definition-condition`, `defines-on` |
| `convention` | convention-bind ("we write...") (R7) | — |
| `promises` | promissory-bind opens a debt region (R8) | — |
| `decorator-supplies-component` | decorator-bind (R11) | — |

### Concept layer (R9/R10)
| kind | meaning | absorbs |
|---|---|---|
| `parametrizes` | `$X$-term` decorator parameter (R9) | — |
| `equips` | with-clause capability (R10) | `requires` |
| `constrains` | constraint/relation between bound things | `constrain`, `membership`, `typed-arrow` |

### Statement/logic layer
| kind | meaning | absorbs |
|---|---|---|
| `states` | statement node asserts its content | `statement`, `claims-equation` |
| `quantifies` | quantifier scope (for any / for all) | — |
| `logical` | connective structure; `:connective` slot (iff, implies) | `iff`, `iff-expansion` |

### Proof discourse layer (R13)
| kind | meaning | absorbs |
|---|---|---|
| `goal` | explicit goal statement | — |
| `decomposes-into` | suffices-split into obligations | `obligation-statement` |
| `enters-subgoal` | "We proceed to prove i)" | — |
| `proof-step` | one step w/ `:justification` (stackrel, technique) (R12/R13) | `equational-step`, `proves-by-diagram-chain`, `monomorphism-cancel`, `weakens-local-hypothesis`, `concludes`, `supports` |
| `announces-dependency-map` | author states assumption⇢result partiality (R10.3) | — |

### Authored link layer (R12)
| kind | meaning | absorbs |
|---|---|---|
| `anchors` | `\label` anchor | — |
| `refers` | `\ref` intra-paper, typed by prose word before | `references`, `refers-to` |
| `cites` | `\cite[locus]` extern edge | — |

## Satiety (nodes) — recognizers = scopes

Every node carries `:satiety`. A scope is **full** when every slot is
bound within reach; **hungry** when a slot points outside, typed by
what feeds it:

- `:full` — bound/resolved in-region or by a cited in-paper site.
- `{:hungry-for :canon}` — concept/operator awaiting a canon link (R3 pointer).
- `{:hungry-for :payoff}` — promissory debt awaiting discharge (R8).
- `{:hungry-for :parse}` — parse-incomplete envelope (R7).
- `{:hungry-for :bundling}` — bundling-unresolved capability (R10).
- `{:hungry-for :role}` — UNKNOWN control sequence (R5/R15 amendment).

Assignment rules (mechanical, in the normalizer): `:canon :pointer` →
canon-hungry; `:bundling :unresolved` → bundling-hungry;
`:parse-incomplete true` → parse-hungry; promise-role end of a
`promises` edge → payoff-hungry; everything else with an in-region
`:via` → full. Hand-set satiety survives normalization.

The corpus-level hunger field (sum of hungry slots by concept) is the
priority queue of the concept-driven mining plan (THE KILLER IDEA,
golden-walk-ledger.md).
