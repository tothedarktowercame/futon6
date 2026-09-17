# Mark7 host configuration

`scripts/futon6_config.py` is the shared resolver for the Mark7 paths identified
in PR #51's Stage 0 inventory. The stepper, preflight, conformance, S3 shell
wrapper, semantic gate, and corresponding Python readers use this configuration.
Environment variables are the configuration interface; no operator-specific
absolute path is a default. CLI input/output flags on individual tools still
override those tools' defaults.

## Paths

| Setting | Default / meaning |
|---|---|
| Checkout | Derived from the actual code location; no requirement that its directory be named `futon6` |
| `FUTON_CODE_ROOT` | Checkout's parent; conventional location of sibling repositories |
| `FUTON3_ROOT` | `$FUTON_CODE_ROOT/futon3`; pattern index and libraries, checked by preflight |
| `FUTON3C_ROOT` | `$FUTON_CODE_ROOT/futon3c`; optional APM/staging and close-reading inputs |
| `FUTON6_STORAGE_ROOT` | `$FUTON_CODE_ROOT/storage`; external storage tree, not the checkout's output directory |
| `FUTON6_EPRINTS` | Explicit source directory, or the discovery order below |
| `FUTON6_ANATOMY` | `$FUTON6_STORAGE_ROOT/futon6/data/ct-anatomy-v0` |
| `FUTON6_BACKGROUND_CORPUS_INDEX` | Actual checkout's `data/background-corpus-index.json` |
| `FUTON6_SCOPES` | `$FUTON6_STORAGE_ROOT/mark2/ct-fresh-scopes` |
| `FUTON6_NER_TERMS` | `$FUTON6_STORAGE_ROOT/mark2/ct-handoff/output/ner-terms.json` |
| `MATHLIB4_ROOT`, `PLANETMATH_ROOT`, `NLAB_CONTENT_ROOT`, `NNEXUS_ROOT` | Corresponding sibling under `FUTON_CODE_ROOT`; used by optional readers/builders |

Relative path overrides resolve against the actual checkout, including relative
authority overrides (this standardizes Stage 1a's earlier cwd-relative behavior).
`~` is expanded. Checkout-owned outputs stay in that checkout even when
`FUTON_CODE_ROOT` points elsewhere. Required inputs are checked by preflight;
optional builder inputs are not invented or installed merely because their
locations are configured.

Without `FUTON6_EPRINTS`, every migrated consumer and preflight use the first
populated directory in this order:

1. `$FUTON6_STORAGE_ROOT/futon6/data/arxiv-math-ct-eprints`
2. `$HOME/data/arxiv-math-ct-eprints`
3. The actual checkout's `data/arxiv-math-ct-eprints`

An explicit missing directory never falls back to another source. If discovery
finds nothing, the conventional storage location is reported and preflight
refuses. The entry points normalize and export selected paths once so their
children cannot independently discover a different source midway through a run.

## Interpreter and serving configuration

`FUTON6_PYTHON_CMD` is a shell-quoted **argv**, such as
`'/path with spaces/bin/python' -u`. It does not support shell assignments,
pipes, or command substitution. The default is the interpreter that launched
the entry point, plus `-u`; a checkout-local `.venv` is not required. Executable
symlinks are preserved, since resolving a venv's Python to the system binary
would lose that environment.

The runner passes the selected argv to preflight and conformance and exports it
to the S3 wrapper and Babashka semantic gate. `FUTON6_PYTHON` and
`FUTON6_PYTHON_ARGV_JSON` are internal propagation values; set
`FUTON6_PYTHON_CMD` at the entry point rather than setting these separately.
Preflight imports its required Python modules using the selected interpreter,
even when preflight itself was launched using another one.

`OPENAI_BASE_URL` selects the OpenAI-compatible base URL (through `/v1`, without
`/chat/completions`); the default is `http://localhost:${PORT:-8000}/v1`.
`MODEL` selects the served model name and defaults to `mark4-70b`. All stages
receive that selection, including S3 and S7, which previously constructed their
own localhost URLs. `OPENAI_API_KEY` is inherited, not overwritten by the wrapper.
The server must still pass preflight and conformance; configuring it is not
evidence of readiness.

Standalone legacy `linode-4gpu-run.sh` use still accepts `PYTHON`/`VENV` when
`FUTON6_PYTHON_CMD` is absent. Under Mark7 the shared setting takes precedence.
Its default repository is now derived from the shell script's location; the
stepper explicitly sends its actual checkout through `REPO`.

## Inspect and run

```bash
export FUTON3_ROOT=/data/checkouts/futon3
export FUTON6_EPRINTS=/data/arxiv-math-eprints
export FUTON6_BACKGROUND_CORPUS_INDEX=/data/substrate/background-corpus-index.json
export FUTON6_PYTHON_CMD="'/opt/runner environment/bin/python' -u"
export OPENAI_BASE_URL=http://model-host:8000/v1
export MODEL=my-served-model
python3 scripts/futon6_config.py
python3 scripts/linode_stepper.py --plan --profile superpod
```

The stepper prints the resolved host configuration and appends it to
`<run-dir>/host-config.jsonl` after manifest identity validation and before entry gates. A conflicting
identity is refused without appending a configuration record. It records paths, interpreter argv, model and
endpoint, not the whole environment or API key. URL userinfo/query/fragment are
excluded from that record. Preflight prints the same configuration.

The [run manifest](mark7-run-manifest.md) freezes this configuration alongside
corpus, code, substrate, model, and output identity. Do not infer fresh-host
readiness from successful configuration tests.

## Provisioning boundary

No new Linode has been provisioned. Use `futon0/README-linode.md` only after
Stages 2–3 and the local acceptance prerequisites are resolved. Remote transfer
sources are independent of these local paths: never substitute a local
`FUTON_CODE_ROOT` into a remote rsync path. The unrelated `setup-ct-run.sh`
remote-path rewrite from PR #51 has not been imported; its separate repair is
still in Stage 5. Mark7's current STAGE step remains operator-driven.
