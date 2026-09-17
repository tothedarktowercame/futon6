# PR #51 Stage 2 validation

Date: 2026-09-17. Branch: `work/pr51-response`, isolated worktree
`/home/joe/code/futon6-pr51-response`. Builds on Stage 1b `e84e6f6`.

## Implemented

- Shared immutable manifest for runner, replay, and retrieval; exact frozen corpus
  hash, code/substrate/configuration/model identity, selection, relative artifacts.
- Explicit run/corpus identities and disagreement refusal; nonblocking run lock;
  resume validation; refuse existing artifacts without a manifest. Successful
  stages cannot be rerun in place over downstream evidence.
- All stage outputs collected under `--run-dir`; marks/loss readers consume the
  manifest paths. Stage logs persist while preserving command/gate exit status.
- Boot-only terminal `--mark-done`; repeated `--reuse` accumulates, restricted to
  S0/STAGE. S2 always requires measured evidence for this corpus.
- Replay derives inputs from the manifest, rejects inconsistent overrides, and
  returns nonzero for warnings. C1 step conservation starts at its S5 producer.
- Retrieval inventory covers every file, size, SHA-256, and artifact count.
  Packing and receiving-side verification both replay a fresh extracted copy.
  Missing required CLeans/paper graphs/learning outputs cannot be accepted merely
  because tar creation succeeded.
- Synchronized both operator handoffs/playbooks, reproduction guidance, and host
  configuration docs. Removed S2 bypass, mismatched output paths, and concurrent
  shards sharing one run directory. The normative operations guide is
  [mark7-run-manifest.md](../docs/mark7-run-manifest.md).

Disposition: implements the relevant Stage 2 ideas from PR #51 with a shared
manifest rather than importing conflicting path prose or broad `--mark-done` /
reuse overrides. No unrelated Stage 5 portability hunks were imported.

## Validation commands and evidence

```bash
PYTHONPATH=/tmp/futon6-pr51-test-deps /home/joe/code/futon6/.venv/bin/python -m pytest -q \
  tests/test_run_manifest.py tests/test_mark7_configuration.py \
  tests/test_mark7_authority.py tests/test_stepper_exit_status.py tests/test_warp_run.py
clj-kondo --lint scripts/iatc_semcheck.bb
clj-kondo --lint scripts/iatc_anchor_faithfulness.bb
emacs -Q --batch -l /home/joe/code/futon4/dev/check-parens.el \
  --eval '(arxana-check-parens-cli)' -- --no-defaults \
  scripts/iatc_semcheck.bb scripts/iatc_anchor_faithfulness.bb
git diff --check
```

Results: **45 tests passed, 6 subtests passed**. Each Babashka file has zero
clj-kondo errors/warnings; check-parens reports OK; diff whitespace check passes.
Pytest is isolated in `/tmp/futon6-pr51-test-deps`; no dependency change to Joe's
project environment was needed.

Tests exercise changed corpus/code/substrate refusal, conflicting CLI/environment
identity, locking, foreign metric records, duplicate corpus IDs, unsafe CLI
combinations, run paths containing spaces, shell expansion of two disjoint runs,
real command logging, failed command/gate exit status, successful-stage overwrite
refusal, unsupported selection-cap refusal, and replay override refusal.

An S1 fixture is packed, extracted, and replayed by the real retrieval/replay
entry points. A deliberately corrupted archive fails checksum verification;
missing CLeans prevent S7 packaging. The fixture mocks code/substrate identity
acquisition and supplies synthetic S1 marks/metrics. It proves workflow behavior,
not scientific correctness or a completed model run. No synthetic evidence is
installed into a production run directory.

## Outstanding acceptance work

Stage 3 remains responsible for per-item partial-failure/rejection accounting,
repair and retry semantics, capped/deferred selection, and existing replay
semantic tolerances (including logged-but-untyped CLeans and anchor/reference
thresholds). File presence and passing the current replay are insufficient proof
of zero final rejections. Substrate identity does not hash eprints; model revision
is optional operator metadata, not verified weights. Legacy unmanifested runs
are not silently migrated.

The two existing CAS failures documented in Stage 1b remain unresolved; no test
expectations were weakened. Rob's raw run bundle is still needed for the Stage 0
comparison. No model run, new Linode, full-build acceptance, merge, push, or message
to Rob/GitHub occurred. Fresh-host provisioning stays deferred to Stage 4.
