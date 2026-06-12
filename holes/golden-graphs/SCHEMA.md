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

## Mined mission-triple kinds (miner v1 — PINNED from measurement, fable-2 2026-06-12)

What `scripts/mission_triple_miner.py` (`26564e7`) actually emits across all 80
`data/mission-triples/*.edn` (measured, not specified — the input contract for the
substrate-metric cascade adapter):

- **Cascade nodes** `{:id :role :satiety :form :via :rule}` — roles: `:scope` (the per-mission
  problem node, 80×) · `:concept` (patterns, 400×). Satiety values: `:full` (400×) ·
  `{:hungry-for :parse}` (79×) · `{:hungry-for :payoff}` (1×) — all within the taxonomy above.
- **Cascade hyperedges** `{:kind :ends :via :rule}` — kinds: **`:differentiates`** (400×) ·
  **`:states`** (42×). End roles: `:context` · `:pattern` · `:problem`.
  `:states` is in the controlled table; `:differentiates` is cascade-specific (a pattern
  differentiates a problem context) and not absorbed by any of the 22 — left as its own kind.
- **Wiring nodes** — role `:application` (checkpoints as witnessed applications), satiety `:full`.
  **Wiring hyperedges** — kind **`:composes`** with `:from`/`:to` ends in authored checkpoint
  order (also cascade/wiring-specific, not absorbed).
- **NOT emitted by miner v1: `:jointly-with`.** It is spec vocabulary only; no v1 file contains
  it. Adapters must not assume it. When it lands it is n-ary: clique-expand (symmetry-preserving),
  never chain (chaining imposes an order the semilattice forbids) — claude-3's ruling, the
  projection choice is the adapter owner's.
