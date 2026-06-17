# WARP-ORCH-3 Concept Phylogeny Artifact

Date: 2026-06-17

`scripts/warp_run.py --stage S6a` builds
`data/warp/concept-phylogeny.json` from:

- `data/showcases/ct-anatomy/golden`
- `data/concept-encyclopedia/ct`
- `data/warp/cite-resolution/`

This is the CAS-SEL genealogical-select descent input: a paper inherits its
imports'/citations' concept patterns.  The artifact records per-concept
trajectories over the resolved citation neighborhood, with each event typed as
`definition`, `first-use`, `cited-activation`, `uncited-activation`, or
`redefinition`.

The same descent relation is also the R2d-3 coupling candidate: cited concept
activation edges can be checked against proof/warrant inheritance in later
semantic-checker rungs.

## Built Artifact

- Path: `data/warp/concept-phylogeny.json`
- Schema: `futon6/warp/concept-phylogeny/v1`
- Papers scanned in citation neighborhood: 92
- Citation-neighborhood paper ids: 95
- Candidate concepts: 200
- Concepts with threads: 200
- Cited activations: 416
- Uncited activations: 2186
- Redefinitions: 1330
- Definitions: 68
- First uses: 132

## Worked Example

`abelian categories` reached paper `math__0111205` via
`cited-activation` from paper `math__0111204`.

This appears in the artifact as:

```json
{
  "concept": "abelian categories",
  "paper": "math__0111205",
  "via": "cited-activation",
  "from_paper": "math__0111204"
}
```

## Scope Note

The full golden directory is currently about 18 GB.  S6a therefore scans the
resolved citation neighborhood derived from `data/warp/cite-resolution/`, rather
than every golden paper, because the citation-descent relation is the substrate
CAS-SEL consumes.  This keeps WARP-ORCH-3 independent of `concept-usage.json`
and avoids touching guarded SFC-D3/SFC-AGG outputs.
