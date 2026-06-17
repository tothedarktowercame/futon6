# SFC Concept Aggregate — Adjunction Fixture

## Run

- Fixture JSON: `/home/joe/code/futon6/data/warp/sfc-adjunction-fixture.json`
- Instances: 24
- Sources: PlanetMath=2, arxiv-def-snippets=14, nLab=8
- Genus: `adjunction F⊣G` (inferred-v0)

## GC Surface -> Core Retention

- `all functors` -> `None`; action=None; df=None; retained_papers=None
- `any two` -> `None`; action=None; df=None; retained_papers=None
- `each other` -> `relation`; action=fold; df=3445; retained_papers=3445

## Variant-Axes Schema

Schema `lean-family-v0`: keep a structure-like `genus`, retain every grounded `instance`, and represent divergent but equivalent definitions as a labelled family under `variant_axes[].variants`. Equivalence is recorded as explicit `iff-lemma` bridge holes, matching the Lean pattern of structures/classes with instances plus equivalence lemmas/defeq where available.

Recovered definition-framing variants:
- `contextual-use`: 17 instances; sources=arxiv-def-snippets, nLab
- `hom-set-natural-bijection`: 1 instances; sources=PlanetMath
- `unit-counit-triangle`: 5 instances; sources=PlanetMath, arxiv-def-snippets, nLab
- `universal-arrow`: 1 instances; sources=arxiv-def-snippets

## Remaining holes

- The reducer records equivalence bridges but does not prove them.
- Framing classification is keyword/classical-prose based; formula grounding is delegated to H-SFC2b.
- Genus inference falls back to the hand-recognised adjunction core when encyclopedia-v0 has only noisy genus data.
