# Mark7 run identity, resume, replay, and retrieval

Stage 2 of the [PR #51 response](../TN-PR51-response.md) makes a run directory
self-contained. The runner, replay, and retrieval read one `run-manifest.json`.
Existing unmanifested runs are historical evidence; the runner refuses to adopt
or overwrite their artifacts.

## Start and resume

Configure the host using [the shared settings](mark7-host-configuration.md).
After provisioning and staging have actually completed:

```bash
python3 scripts/linode_stepper.py --run --profile superpod \
  --run-id mark7-validation-01 --corpus-id math-ct-e2e-16 \
  --run-dir /scratch/mark7-validation-01 --ids holes/mark7-16.ids.txt \
  --from S1 --reuse S0 STAGE
```

`--run-id` / `RUN_ID` and `--corpus-id` / `CORPUS` must agree if both are set.
Missing identities and `adhoc` are refused. Without `--run-dir`, the destination
is the checkout's `data/runs/<run-id>`. Relative CLI paths are checkout-relative;
absolute run directories and paths containing spaces are supported. The default
corpus for a new run is `holes/mark7-16.ids.txt`; pass `--ids` for any other scope.

The runner freezes the exact corpus bytes as `corpus.ids.txt`, rejects empty or
duplicate IDs, and records its hash, paper list, code fingerprint, substrate
hashes, effective configuration, model name/revision, and selection policy.
All proofs and uncapped expository regions are the current selection. A nonzero
`FUTON6_EXPOSITORY_CAP_PER_PAPER` is refused until Stage 3 implements explicit
selection and deferred-item accounting. `FUTON6_MODEL_REVISION` records an
operator-supplied model revision/hash; if absent, the model name alone does not
prove immutable weights. The runner does not hash the eprint archive collection.

Code identity includes git HEAD, scripts/src executable sources, DAG contracts,
and EDN vocab/schema/contract files under holes/resources. Substrate identity
includes the five core concept files, configured authority, pattern index, and
both math-informal pattern families. This is not a complete environment image
or dependency lock. Configuration paths are frozen for execution, so relocating
a run permits replay/retrieval but does not silently authorize execution under
a different checkout or input root.

Resume with the same run and corpus IDs, configuration, and `--run-dir`, using
`--from S<n>` for the next stage. Omit `--ids` to use the frozen list. Changed
corpus bytes, code, substrate, model revision, or configuration require a new
run directory. A nonblocking process lock prevents concurrent runners or packing
while execution is active. Ledger and metric rows must belong to this run and
corpus; malformed or foreign rows refuse resume and replay.

Preflight and serving conformance remain mandatory on every execution. Their
success is not full-build acceptance. The operator still chooses the resume
stage after inspecting the halt; the runner does not automatically skip or
repair stages. Successfully ledgered stages cannot be rerun in place: resume at the next
stage or start a new run directory. Stage 3 must address richer attempt and
partial-failure accounting.

`--reuse S0 STAGE` acknowledges completed boot steps only; repeated `--reuse`
options accumulate. Every computational dependency needs a passing ledger row
in this run. S2 checks the frozen corpus against substrate and cannot be reused.
`--mark-done S0 STAGE` is a separate, terminal bookkeeping operation: it cannot
combine with execution/planning/range/reuse flags, and cannot mark S1–S12 done.
`--plan` is read-only and does not require identities or create a manifest.

## Output layout

| Run-relative path | Contents |
|---|---|
| `run-manifest.json`, `corpus.ids.txt` | Frozen identity and corpus |
| `host-config.jsonl` | Accepted invocation configuration, before entry gates |
| `phase-ledger.jsonl`, `metrics.jsonl` | Passing stages and emitted measurements |
| `logs/S<n>.command.log`, `logs/S<n>.gate.log` | Appended command/gate output; command exit status preserved |
| `artifacts/marks`, `artifacts/loss` | S1 marks and invariant dashboard |
| `artifacts/candidates`, `artifacts/graphs` | S3 candidates, graphs, retry report |
| `artifacts/expo-candidates`, `artifacts/expo` | S4 candidates and expository graphs |
| `artifacts/steps`, `artifacts/rung3` | S5 proof steps and technique maps |
| `artifacts/paper-graphs` | S6 paper graphs |
| `artifacts/clean`, `artifacts/demo` | S7 CLeans, embedding; S8 ingest exports |
| `render/`, run-root JSON/text files | Rendered pages, eval/anchor reports, harvested and learned outputs |

The runner installs output environment variables from the manifest. Standalone
scripts retain their documented legacy defaults; invoke the runner for this
manifest workflow. Copying only the metrics directory or only the embedding
omits required evidence.

## Replay and retrieval

```bash
python3 scripts/replay_e2e.py --run-dir /scratch/mark7-validation-01 --through S3
python3 scripts/retrieve_run.py pack --run-dir /scratch/mark7-validation-01 \
  --through S3 --output /scratch/mark7-validation-01-S3.tgz
# Transfer the archive to durable storage, then verify on the receiving host:
python3 scripts/retrieve_run.py verify /durable/mark7-validation-01-S3.tgz \
  --extract-to /durable/mark7-validation-01-S3
```

Omit `--through` only for a completed S12 run. The receiving host needs this
checkout and its Python dependencies, but no live model or original input roots.
Replay derives all paths and corpus identity from the manifest. Explicit path,
corpus, ID-file, or log overrides must agree. Missing/misconfigured targets return
2; failing checks or warnings return nonzero. Step conservation starts at S5,
which produces steps. A stage prefix is not an arbitrary subset of paper IDs.

Packing requires passing ledger evidence through the declared stage and nonempty
required artifacts, including CLeans and paper graphs when applicable. It includes
all run files with per-file SHA-256/size and artifact-directory counts. Both pack
and verify extract a temporary copy, validate inventory and identity, and run
replay there. Duplicate members, links, escaping paths, checksum disagreement,
missing required outputs, and replay failures prevent success. Existing archives
and extraction destinations are not overwritten. Verify the transferred copy
before teardown; verifying only the source archive is insufficient.

These checks establish identity, transport integrity, and the existing replay
invariants. Stage 3 still has to close semantic rejection/error accounting and
existing replay tolerances; neither a prefix fixture nor an archive report proves
a rejection-free S1–S12 build. Fresh-host acceptance remains Stage 4.
