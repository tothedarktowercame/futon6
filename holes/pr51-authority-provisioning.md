# PR #51 Stage 1a: ship and require concept authority

This batch repairs one confirmed dependency defect. It does not complete Stage
1's configuration sweep or qualify the full runner for a Linode allocation.

Both advertised bundles now include the existing full
`futon6/data/background-corpus-index.json`, without rebuilding or filtering it.
They retain every original regular-file payload byte-for-byte. The CT bundle
retains its smaller pattern selection; the legacy bundle retains its broader
selection. Both contain `futon6/data/mark7-substrate-manifest.json`, which records
every payload's size and SHA-256, the base archive's hash, and the authority's
existing source metadata. Original upstream input hashes are unavailable and
are explicitly not claimed. The authority remains schema 2, with 130,960 term
keys; its CT-prior portion was candidate-filtered when originally generated.

## Operator use

For the CT bundle, from the checkout:

```bash
(cd data && sha256sum -c mark7-ct-substrate.tgz.sha256)
tar -xzf data/mark7-ct-substrate.tgz -C /path/to/parent-of-checkouts
```

The archive retains the existing `futon6/` and `futon3/` layout. If the actual
checkout has another name, place the extracted futon6 data under that checkout,
or configure the authority path explicitly. This batch verifies that the
authority itself works in a renamed checkout; other runner path fixes remain.

Default authority: `data/background-corpus-index.json` relative to the actual
script checkout. Override: `FUTON6_BACKGROUND_CORPUS_INDEX`, resolved at use time
(relative overrides are relative to the process working directory). Prefer an
absolute override in a run configuration. Inspect it without model calls:

```bash
python3 scripts/concept_authority.py '\Hom' '\End' '\colim'
```

Preflight now checks authority readability, schema, nonempty and structurally
valid entries, and these three required resolutions. The same checks run when
the authority is constructed outside preflight. Missing/unusable data cannot
fall back to an empty authority. These probes establish dependency readiness,
not complete concept coverage for every corpus or mathematical correctness of
the upstream index. Normalization now handles a leading backslash for indexed
operator names, including `\End`, whose `end` entry previously went unused.

## Rebuild recipe

The builder accepts an existing base bundle and the materialized authority:

```bash
python3 scripts/build_mark7_substrate.py \
  --base /path/to/original/mark7-ct-substrate.tgz \
  --index /path/to/background-corpus-index.json \
  --output data/mark7-ct-substrate.tgz
```

Repeat with the legacy bundle if maintaining both. The same inputs produce the
same bytes. An updated bundle used as a new base records that new base's hash,
so preserve the original base to reproduce this exact build. No symlink payload
is accepted. The output gets a `.sha256` sidecar. It replaces an existing output
only after index validation and archive construction succeed.

## Validation on 2026-09-17

```bash
python3 tests/test_mark7_authority.py
python3 -m py_compile scripts/concept_authority.py scripts/preflight.py \
  scripts/build_mark7_substrate.py tests/test_mark7_authority.py
git diff --check
```

Six tests cover missing/malformed/empty/unsuitable data, configured lookups,
preflight refusal, preservation of an existing output on invalid input,
rejection of symlink payloads, deterministic packaging, inventory hashes, and
lookup from an extracted renamed checkout with no authority override.

An additional local full-bundle verification checked all inventory hashes,
compared the packaged authority to the source index, and confirmed every
original payload unchanged: 1,069 legacy files and 66 CT files. Each archive
was extracted into a temporary directory, its futon6 directory renamed, and
`preflight.check_concept_authority()` passed using only the extracted index.
These were authority checks, not the entire preflight or pipeline.

Resulting archive hashes:

- CT: `1dc82258aef7e364aff8f268825d6c061fbe5c998f4cb2dc42f19eb0b9d50456`
- Legacy: `1ea9a9b27608b1c40584c5506abd2305f793635049a7daf7ae0ebe99e5e99bb9`

Next: finish Stage 1's configured interpreter, sibling/dataset paths and
effective configuration record; then Stage 2 run identity/replay/retrieval and
Stage 3 rejection accounting. No new host has been provisioned or model run
started. Rob's raw run bundle remains needed for defect-specific comparison.
