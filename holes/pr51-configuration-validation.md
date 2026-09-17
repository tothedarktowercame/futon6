# PR #51 Stage 1b: shared configuration and validation

2026-09-17. See [operator configuration](../docs/mark7-host-configuration.md).

The Stage 0 Mark7 path inventory has been migrated to `futon6_config.py`.
This covers checkout-local outputs, the actual authority location, configured
siblings and storage, and shared eprint discovery. Additional reachable
dependencies `log_loss.py` and `background_corpus_index.py` were repaired too.
No unrelated PR files were cherry-picked.

Interpreter selection now propagates through the stepper, both entry gates,
WARP subprocesses, the S3 shell wrapper and Babashka's R2d subprocess. Python
argv flags and paths with spaces survive each boundary. Module availability is
checked in the selected interpreter. The S3 wrapper no longer relocates itself
to `$HOME/futon6` or silently selects its own serving endpoint. S7 also uses the
configured endpoint; API credentials are inherited and absent from the host
record. Each run invocation records effective host configuration before gates.

## Checks

Pytest was installed into `/tmp/futon6-pr51-test-deps`; the existing interpreter
environment was not modified. From this worktree:

```bash
PYTHONPATH=/tmp/futon6-pr51-test-deps \
  "$PY" -m pytest -q \
  tests/test_mark7_configuration.py tests/test_mark7_authority.py \
  tests/test_stepper_exit_status.py tests/test_warp_run.py
clj-kondo --lint scripts/iatc_semcheck.bb
emacs -Q --batch -l "$FUTON_CODE_ROOT"/futon4/dev/check-parens.el \
  --eval '(arxana-check-parens-cli)' -- --no-defaults scripts/iatc_semcheck.bb
bash -n scripts/linode-4gpu-run.sh
git diff --check
```

The targeted tests passed (31 tests, six subtests at this checkpoint), including:

- A renamed checkout with no `.venv` and datasets/siblings elsewhere: preflight,
  anatomy, WARP and DP readers agree on paths, and stepper planning works.
- Explicit absent source configuration does not fall back to another store.
- Quoted interpreter paths, flags, and executable symlinks are preserved.
- The real Babashka subprocess boundary invokes the chosen interpreter argv.
- The real S3 shell wrapper validates candidates and invokes a stubbed model
  boundary with the selected checkout, Python flags, remote endpoint and key.
- Refused runs retain their effective host configuration without API keys.
- All Stage 1a authority tests and existing stepper/WARP tests pass.

All changed Python files parse; AST inspection found no executable absolute
Joe/Rob home-path strings in the changed Python files. The Babashka lint and
parenthesis checks pass. These are local configuration checks, not a GPU run.

## Existing acceptance failures retained

Including `tests/test_cas_select.py` exposed two failures. Both reproduce on the
unchanged master code using the same sibling pattern library:

1. `test_trigger_path_premint_pool_enqueues_exactly_three_minted_steps` queues
   six items rather than the expected three.
2. `test_tier0_retrieval_recall_is_honest` measures whole-index recall 12/22
   rather than the expected 15/22.

Baseline command, from the dev checkout:

```bash
PYTHONPATH=<test-deps>:<checkout>/scripts \
  .venv/bin/python -m pytest -q tests/test_cas_select.py
```

Result: two failed, six passed, matching the modified code. The script-directory
PYTHONPATH is needed for the baseline's `llm_json` import during collection.
Neither test expectation nor retrieval behavior was changed. These failures
remain to be investigated before claiming the fully valid build target.

Next is Stage 2's run manifest and replay/retrieval agreement, followed by
Stage 3's explicit partial-result accounting and defect repair. Rob's raw run
bundle remains unavailable. No cloud allocation or live model run was started.
