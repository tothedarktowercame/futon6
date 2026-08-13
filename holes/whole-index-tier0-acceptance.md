# Whole-index Tier-0 acceptance evidence

Measured 2026-08-13 with `scripts/cas_select.py`. The query set was frozen in
`tests/fixtures/cas-select/whole-index-provenance-queries.json` before running
retrieval. Queries use vocabulary from the patterns' provenance marks rather
than copying pattern names or constructing queries from index hotwords.

## Loader and compatibility checks

- Legacy `load_patterns()`: 45 patterns.
- Opt-in `load_all_patterns()`: 1,334 patterns.
- All six mining-campaign patterns: present.
- `python3 scripts/cas_select.py --backend stub` before/after SHA-256:
  `b92d33dbc8f55ab0d129d4f0f30c02caaefb9fd0ef6e01221eb44ef5db4c41ce`.
- Before/after serialized output: byte-identical (`cmp` exit 0).

## Provenance-vocabulary retrieval at k=5

Position `MISS` means the target did not occur in the top five. These are
reported as measured; no query was rewritten after seeing its result.

| # | Target | Position | Query |
|---:|---|:---:|---|
| 1 | `transport-across-an-instance-diamond` | MISS | The rational algebra structures on the same carrier are equal by uniqueness, so rewrite IsGalois across that equality. |
| 2 | `transport-across-an-instance-diamond` | MISS | After reinstalling the field structure the motive is not type correct; equate the two scalar actions before continuing. |
| 3 | `lift-prove-upstairs-reflect-by-injectivity` | MISS | Prove the intersection and generated compositum identities in the cyclotomic field first, then pull the equalities back to the degree-nine fixed field. |
| 4 | `lift-prove-upstairs-reflect-by-injectivity` | MISS | Send both sides into the larger intermediate field, simplify the infimum and supremum there, then use one-to-one-ness to return. |
| 5 | `close-bijectivity-by-counting-not-inverting` | MISS | A one-to-one map between finite types of equal size is onto. |
| 6 | `close-bijectivity-by-counting-not-inverting` | MISS | Show the kernel is trivial and compare the number of elements instead of constructing preimages. |
| 7 | `construct-through-a-finite-correspondence` | MISS | Build the degree-nine field from the product of the units modulo seven and nine, then read subfields from fixed groups. |
| 8 | `construct-through-a-finite-correspondence` | MISS | Prove the order-four and order-twelve group calculations first and obtain the field degrees from indices. |
| 9 | `probe-the-claimed-property-not-the-acceptance-proxy` | MISS | Zero sorries did not imply the main theorem was clean because an imported helper still depended on sorryAx. |
| 10 | `probe-the-claimed-property-not-the-acceptance-proxy` | MISS | The file compiled, yet the theorem still depended on compiler evaluation through an imported helper. |
| 11 | `replace-enumeration-with-structural-counting` | MISS | GL3 over integers modulo four was reduced modulo two; the fiber had 2^9 elements. |
| 12 | `replace-enumeration-with-structural-counting` | MISS | Replace the 86016-state computation by a surjection to GL3 over F2 and multiply image size by fiber size. |

This is a real hotword-retrieval finding, not a loader failure: the targets are
in the candidate pool, but generic whole-library candidates outscore them.
