# TN: Process PR #51 in stages; establish a fully valid Mark7 run

Date: 2026-09-17. Requested by Joe for handoff to other agents.

PR: https://github.com/tothedarktowercame/futon6/pull/51

Reviewed head: `304beb6bb43f51ceae0f297cdb40a3741a8e1214`.
Reviewed master: `8dd08f49d351cc3808f9bb079db888847a1efb9a`.
An isolated review checkout exists at `/tmp/futon6-pr51-review`.

Implementation started on branch `work/pr51-response` in
`/home/joe/code/futon6-pr51-response`. See the
[Stage 0 evidence comparison](holes/pr51-baseline-comparison.md),
[146-file disposition](holes/pr51-file-disposition.md), and
[Stage 1a authority provisioning](holes/pr51-authority-provisioning.md).
The subsequent [Stage 1b configuration work](holes/pr51-configuration-validation.md)
and [operator settings](docs/mark7-host-configuration.md) cover the remaining
Mark7 host configuration. [Stage 2 validation](holes/pr51-stage2-validation.md)
and [run operations](docs/mark7-run-manifest.md) now cover immutable identity,
run-contained outputs, resume, replay, and verified retrieval.
[Stage 3 discovery](holes/pr51-stage3-discovery.md) and
[Stage 3 validation](holes/pr51-stage3-validation.md) cover per-item accounting,
retry history, the declared S4 cap, replay acceptance, and the causes of the
malformed paper graphs, dangling references and G7 cycles. Stage 0 still lacks
Rob's raw run bundle; fresh-host acceptance (Stage 4) remains outstanding.
Tests that already failed before this work still fail; their expectations have
not been weakened.

## Decision and intent

Do not merge the PR wholesale. It contains useful host-portability changes and
valuable observations from Rob's Superpod run, but also regressions and a
missing-data fallback that weakens the pipeline. GitHub reported a clean merge;
that is not evidence that the resulting runner meets its contract.

Joe's intended acceptance target is a **fully valid, rejection-free build over
the agreed validation corpus**. He previously attempted provisioning validation
with that target. Do not reinterpret that work as merely an installation check,
or relax the target to match Rob's observed failures. First recover what was
actually tested and explain the discrepancy. Rob also says some changes are
unrelated to immediate Mark7 needs; handle those separately.

The stages below are proposed implementation and review batches, not claims that
the fixes already exist. Produce a separately reviewable commit or PR per batch,
with its own evidence. Follow `../AGENTS.md`, especially the prohibition on
bypassing invariants. No changes to gates, thresholds, or corpus membership just
to obtain a passing report.

## Evidence and limits

The PR changes 146 files. Most edits replace absolute paths; its three declared
behavioral changes are an interpreter override, missing concept-authority
fallback, and optional per-paper expository cap. Most runner improvements in the
PR description are observations, not implementations.

Rob reports a 12-paper run on one A100-80GB with `llama3.1:70b`:

- Preflight 11/11 and conformance 6/6 passed.
- S4: 267/280 accepted; S6: 10/12 paper graphs well formed.
- S7: 40 typed, one G7 cycle rejection.
- Replay through S12: eight passes, one warning, two failures; 12/165 dangling
  premise/conclusion references and 52 records tagged `adhoc` are reported.

These are author-reported results, not independently reproduced GPU results.
The suggestion that missing concept authority worsened dangling references is
a hypothesis to test, not an established cause. The PR's summary reports 41/41
S3 passes, while its operational narrative describes an earlier 40/41 attempt;
recover attempt-specific evidence rather than combining those counts.

Local review verified that 136 changed Python files parse and six changed shell
files pass `bash -n`. All four standalone stepper exit-status tests passed with
`FUTON6_PYTHON_CMD` set to the existing interpreter. Without the override, the
isolated checkout has no `.venv`, and one test fails before the expected refusal
message. Pytest was unavailable; no full suite was run. A small cap probe kept
30 candidates from each of two papers and preserved the uncapped input. These
checks do not establish runtime portability or a valid Mark7 build.

## Stage 0 — Recover the baseline and divide the patch

1. Locate Joe's earlier provisioning/build evidence and Rob's exact run bundle:
   code revisions, wrapper scripts (including fixes outside this PR), corpus IDs
   and hashes, substrate checksums, commands, environment/configuration, model
   and server versions, ledgers, logs, replay reports, and output inventories.
2. Compare the actual acceptance criteria and inputs. Determine whether the
   earlier test ran all required stages, used the same corpus and model, relied
   on locally available data, or exercised only preflight/conformance. Do not
   presume which explanation is correct. Record unavailable evidence explicitly.
3. Partition changed files by the actual Mark7 execution/import path. Separate
   unrelated mining, mission/session tools, publishing, PlanetMath, and other
   workflows into a deferred portability batch. Do not infer relevance merely
   from a filename or blanket-apply the 146-file rewrite.

Deliverable: a comparison of the two runs and an explicit file/hunk disposition:
keep for Mark7, repair for Mark7, or defer. Freeze the validation corpus before
repair runs; failures must not disappear by removing their papers.

## Stage 1 — Ship concept authority and configure Mark7 dependencies

The concept index is a required S1 dependency but is absent from preflight's
six-file substrate check. The PR says it is absent from the shipped archive.
The local `data/background-corpus-index.json` exists (about 39 MB); inspect its
provenance and contents before packaging it. Inspect actual archive members too:
documentation uses both `mark7-substrate.tgz` and `mark7-ct-substrate.tgz`.

- Include the complete usable authority in the versioned substrate bundle, with
  checksum, schema/version information, and source provenance. A smaller subset
  is acceptable only if its required resolution coverage is demonstrated; do
  not silently drop terms to meet an old bundle-size estimate.
- Keep the repo-relative authority default and explicit
  `FUTON6_BACKGROUND_CORPUS_INDEX` override. Remove automatic empty-authority
  fallback. An absent, malformed, or unsuitable authority must fail preflight
  before model work and fail at use if preflight was not invoked.
- Test parsing and representative resolutions, including the operator aliases
  used by S1, rather than just file existence. Verify the bundle after extraction
  with no access to Joe's data tree and with symlinks properly dereferenced.
- Establish documented configuration for checkout root, sibling repositories,
  external datasets, interpreter, and remote staging source. Defaults for this
  repository should derive from the actual checkout, including a renamed
  checkout; sibling and external paths need explicit, validated configuration.
  `FUTON_CODE_ROOT` may supply a conventional sibling layout but must not imply
  that every local or remote resource shares that layout. Print the effective
  non-secret configuration and record it with the run.
- Keep `FUTON6_PYTHON_CMD` or an equivalent documented interpreter setting;
  ensure preflight, gates, and subprocesses consistently use it. Test a host
  without a checkout-local `.venv`.

Exit evidence: the shipped bundle alone supplies the required authority; known
lookups succeed; a missing authority causes an early refusal; paths work under
an arbitrary temporary checkout location without either operator's home tree.

## Stage 2 — Make execution, replay, and retrieval describe the same run

Introduce one persisted run manifest used by the runner, replay harness, and
retrieval process. Include corpus hash, run identity, code/substrate/model
identity, effective configuration, candidate selection, and artifact locations.
Validate command-line/environment identity disagreements instead of recording
one run ID while writing under another.

The inspected stepper writes `holes/clean/$RUN_ID` and
`data/showcases/$RUN_ID`. Existing playbook/retrieval prose instead names
`holes/clean-<run-id>` and `data/showcases/clean-<run-id>-demo`. The PR narrative
also disagrees with current showcase code. Resolve this from actual producers;
do not copy either prose description unquestioningly.

- Derive replay paths from the manifest associated with `--run-dir`, validating
  any explicit overrides. Distinguish a misconfigured/missing target from
  invalid artifacts, and return failure for either when evaluating a completed
  run. Remove the unrelated historical default paths from this workflow.
- Retrieve all required artifacts, including CLeans and paper graphs, with
  counts and checksums. Verify extraction and replay on the retrieved copy.
  Archive creation alone cannot establish completeness; unexpected empty output
  is a failure, and any legitimate zero must have explicit accounting.
- Make `--mark-done` clearly terminal and restrict it to legitimate boot-stage
  bookkeeping. Reject combinations that imply continuing execution; never use
  it to bless failed computational stages.
- Make repeated `--reuse` accumulate or reject repetition explicitly. Preserve
  the rule that S2 is corpus-fresh and cannot be accepted via `--reuse`.
- Correct the stale S2-stub advice and synchronize the duplicated handoffs and
  playbooks with the actual invocation and output paths.

Exit evidence: one invocation produces an unambiguous run record; replay and
retrieval select exactly that run; intentionally missing CLeans fail retrieval;
resume cannot claim a mismatched corpus or overwrite another run's artifacts.

## Stage 3 — Preserve partial evidence without declaring a failed build valid

The PR correctly identifies that S3/S4 return nonzero on partial acceptance and
the stepper records a passing ledger row only after command and gate success.
S6 aborts its paper loop at the first unsuccessful assembly. This can leave
useful artifacts without adequate stage accounting. It does **not** justify
turning rejection into success.

Persist attempted, accepted, rejected, errored, and deliberately deferred counts,
item identities, reasons, and artifact references even when a stage fails.
Separate evidence of completed computation from evidence that acceptance passed.
Retain failure status and dependency enforcement. Independent items may finish
where the contract permits, but downstream stages must not consume rejected
artifacts or treat incomplete prerequisites as satisfied. Resume must preserve
attempt history and avoid duplicate or stale outputs.

Keep the optional S4 per-paper cap as an explicit run parameter. Record the
selection algorithm and selected/deferred candidates; check that ordering and
sampling meet the intended coverage. A capped probe is not evidence of an
uncapped corpus build, and deferred work is not counted as accepted.

Investigate each reported defect: G7 cycle, malformed paper graphs, dangling
references, and `adhoc` provenance. Establish whether the source is extraction,
model output, repair behavior, missing substrate, or a gate defect. Fix causes
and retain reproductions. Do not remove guards, suppress rejection, add `|| true`,
or manually mark stages done to make a report green.

Exit evidence: focused reproductions pass after repair, valid candidates survive
other items' failures, and a deliberately rejected item still prevents a false
fully-valid result while leaving inspectable evidence.

## Stage 4 — Reproduce the intended acceptance test on a fresh host

Joe clarified the concrete test environment: provision a **new Linode using
`linode-cli`**, following the checklist in
[`futon0/README-linode.md`](../futon0/README-linode.md). **Defer provisioning until
the Mark7 fixes in Stages 1–3 are in place and their local checks pass.** Do not
allocate a host now merely to reproduce known failures. The unrelated Stage 5
batch need not delay this test unless it affects the selected provisioning path.

The checklist covers the GPU bootstrap StackScript, post-reboot `nvidia-smi`,
dependency installation, and model serving. It currently describes Mark4 scripts;
reconcile its invocation with the repaired Mark7 runner and acceptance manifest
before provisioning. Record the actual host and serving configuration. Passing
on this Linode establishes that configuration's result; it does not by itself
establish conformance of Rob's different Superpod/Ollama environment.

Treat these as distinct obligations, all required:

| Obligation | Required evidence |
|---|---|
| Provisioning | Declared bundle/config/dependencies suffice without Joe's tree |
| Serving conformance | Real endpoint passes schema and behavioral checks |
| Valid build | Frozen corpus completes required S1–S12 stages with zero final rejections/errors and all required artifacts valid |
| Provenance | Ledger, metrics, artifacts, and manifest agree; no accidental `adhoc` identities |
| Replay | All applicable acceptance checks pass; no unresolved warning excused as success |
| Retrieval | Retrieved artifacts match source inventory/checksums and pass replay after extraction |
| Resume | Interrupted work resumes with correct identities, dependencies, and counts |

Negative conformance tests must still reject deliberately invalid inputs. That
expected behavior is not a rejection in the positive validation corpus. Record
intermediate rejected attempts and repairs honestly even if the final build is
fully valid. Any intentional cap must be declared in the acceptance scope.

Record serving context length and GPU residency. Rob reports that limiting
Ollama context to 32,768 kept his 70B model on the A100; treat that as a measured
host configuration to verify, not a universal constant or an acceptance result.

Run cheap dependency and negative checks first, then the frozen positive corpus
on the actual serving stack. Compare this evidence directly with Stage 0. A
successful small corpus qualifies that configuration and scope; it does not
guarantee rejection-free extraction over all arXiv mathematics. If the agreed
target remains unmet, report it as unmet with item-level evidence.

## Stage 5 — Repair unrelated portability changes separately

Concrete regressions already found at the reviewed PR head:

- `scripts/c_vector.bb:32,35`, `magnet_probe_extract.bb`,
  `promote_to_proof_layer.bb`, and `starmap_to_capability_graph.bb` default to
  `/users/rjmeyers/darktower`. Replace these with documented configuration and
  suitable derived defaults, not another person's absolute path.
- `scripts/process-all-planetmath.sh:74–81` puts
  `Path('${FUTON_CODE_ROOT}/planetmath')` inside a quoted heredoc, leaving the
  variable literal. Pass configuration explicitly into Python and align it with
  the shell's clone directory, which still uses `$HOME/code/planetmath`.
- `scripts/setup-ct-run.sh:34` derives a remote path from local
  `FUTON_CODE_ROOT`, changing even Joe's default source from
  `/home/joe/ct-handoff` to `/home/joe/code/ct-handoff`. Configure the remote
  source independently. If this script is part of the chosen Mark7 provisioning
  route, move its repair into Stage 1 rather than waiting for this batch.

Audit the remaining mechanical replacements for literal shell variables,
quoting, assumed checkout names, remote/local confusion, and residual assumptions
such as splitting paths on `/code/`. AST parsing cannot catch these. Exercise
representative workflows with nonstandard roots; keep their review and release
separate from the immediate Mark7 acceptance work.

## Handoff requirements

For each stage, return commit(s), disposition of PR hunks, exact validation
commands, resulting evidence paths, and outstanding failures. Do not claim a
Superpod reproduction from syntax/unit checks. Do not publish messages or reviews
to Rob/GitHub without Joe's authorization.

The main checkout already has unrelated/uncommitted edits in handoffs,
readiness documentation, dependency setup, and `scripts/preflight.py`, plus an
untracked `docs/` directory. Preserve these and reconcile intentional overlap;
do not reset or sweep them into this work. The initial review/handoff task only
added this note; subsequent implementation lives on the branch identified above.
