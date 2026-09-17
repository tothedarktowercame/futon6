# PR #51 Stage 3 validation

Date: 2026-09-17. Branch `work/pr51-response` in `/home/joe/code/futon6-pr51-response`,
building on Stage 2 `eeac70a`. Implemented by claude-9. Discovery, decisions and
corrections are in [pr51-stage3-discovery.md](pr51-stage3-discovery.md); operator
behaviour is in [mark7-run-manifest.md](../docs/mark7-run-manifest.md).

## Commits

| Commit | Change |
|---|---|
| `7bdebf9` | Discovery note |
| `fca302d` | `stage_accounting.py`; stepper attempt rows and accounting-based pass; S3/S4/S6/S7 producers account every item; invocation-scoped attempts and acceptance provenance; S4 cap with deferred accounting; S6 evaluates every paper and attaches accepted S3/S4 outputs; S7 gate rejections fail the stage; S7/S10 metric stage labels; run-dir refuses `adhoc` ids in S6/S7 |
| `1b403f9` | Replay: `A1-item-accounting`; `C2` from S7 accounting; `I3` against accounted papers; `I2` fails `adhoc`/stageless records; parsed zero-tolerance `S2`/`S3` |
| `8dac169` | S1: anchored text-style proof starts; learned `\newtheorem`/`\newenvironment` names; `\let`/`\newcommand` statement and proof aliases |
| `bb91fc9` | S3 gate rejects premise→conclusion cycles and `:infer` edges without `:id`/`:conclusion`; prompt states both rules |
| `c95d9ed` | Loop stamps the passage `:source` from the candidate window, window check without slack; gate rejects inline-map premises/conclusions |
| this commit | Documentation: run guide, playbooks/handoffs, TN pointer, this note |

## PR #51 disposition for this stage

- `mark3_expository_loop.py` `_cap_per_paper` (even sample over filename-sorted
  candidates, dropped silently): **repaired**. The cap is now a manifest-pinned run
  parameter applied in candidate extraction. Selection is even spacing in source
  order, and deferred regions are recorded per item.
- PR runner observations about S3/S4 nonzero exit on partial acceptance and S6
  stopping early: **addressed** by per-item accounting. Rejection is not converted
  into success.
- Missing concept-authority fallback: already rejected in Stage 1a; unchanged.

## Validation commands

```bash
PY=/home/joe/code/futon6/.venv/bin/python
PYTHONPATH=/tmp/futon6-pr51-test-deps $PY -m pytest -q \
  tests/test_stage_accounting.py tests/test_run_manifest.py tests/test_mark7_configuration.py \
  tests/test_mark7_authority.py tests/test_stepper_exit_status.py tests/test_warp_run.py \
  tests/test_mark3_iatc_loop_rung2.py tests/test_build_proof_anatomy_viewer.py
clj-kondo --lint scripts/iatc_argcheck.bb
clj-kondo --lint scripts/iatc_repair.bb
emacs -Q --batch -l /home/joe/code/futon4/dev/check-parens.el \
  --eval '(arxana-check-parens-cli)' -- --no-defaults scripts/iatc_argcheck.bb scripts/iatc_repair.bb
git diff --check
```

Results: **71 passed, 6 subtests passed** in the focused set, including 22 in
`test_stage_accounting.py`. Each changed `.bb` file lints separately with zero
errors and zero warnings. Linting both in one invocation reports a spurious
duplicate-require warning across the two files. check-parens OK; diff check clean.

`test_stage_accounting.py` covers:
- **Accounting rules:** uniqueness, required reasons, unaccounted/extra/rejected/deferred items, missing artifacts.
- **Runner:**
  - a rejected item fails the stage (rc 3) and leaves an attempt row and accounting;
  - an unaccounted paper fails;
  - a retry creates `S6-a003` and the ledger records it;
  - a zero exit with no accounting fails;
  - downstream expected items are read from the ledgered upstream accounting.
- **S3 loop:**
  - a valid candidate is accepted while another is rejected;
  - on retry the accepted final is carried forward with its provenance, and both invocations' attempt files remain;
  - a final without provenance is errored.
- **S4:** the cap selection is deterministic; the manifest pins the cap and refuses a changed or negative one.
- **S6:** a malformed paper does not stop the next; both paper objects are written and the stage exits 1.
- **S7:** a real `clean_argcheck` G7 rejection exits 1 and records the G7 reason, while the other graph is accepted.
- **Replay:**
  - a fully accepted S3 prefix passes with key-order-independent parsing;
  - a rejected item fails `A1` even though a ledger row exists;
  - an inline premise and an out-of-passage anchor fail with zero tolerance.
- **S1 fixtures:** sentence-ending "proof." is ignored; TAC `\let` aliases; `\newcommand` aliases and learned theorem titles.
- **S3 gate:**
  - an equivalence written as two implications is rejected, and a single `:iff` edge passes;
  - a conclusion-less edge and an inline-map premise are rejected;
  - the passage stamp is applied and the window check has no slack.

Full suite, `pytest -q tests` (975 tests): the branch introduces no new failures.
Every test failing on the branch also fails on `eeac70a` run in a temporary
worktree (36 shared). The baseline copy had 8 further failures (`test_cas_select`,
`test_cas_checks`, `test_rung3_technique`), probably from running outside the
usual sibling layout. On the final code the full suite gives 36 failed, 897
passed, 44 skipped.
`test_expository_phase5.py` fails its 2 extractor tests on both because fixture
marks for `0710.2254` are absent from this machine.

## Evidence on historical local artifacts (no model run)

- **S1/S6, 16-paper corpus, committed detector vs this change (same eprints):**
  - `0708.1921`: malformed → well-formed; from 0 statements and 2 false proofs to 19 statements and 14 proofs.
  - `0708.2185`: malformed → well-formed; the false introduction "proof." is removed.
  - The other 14 papers stay well-formed. Their only kind changes are author environment names resolving to canonical kinds (for example `eg`→example, `protodefinition`→definition, `rmk`→remark).
  - All 11 text-proof starts removed across the corpus were sentence endings.
- **S3 gate on the historical 98-graph run:**
  - cycle gate: 4 graphs, exactly the four S7 G7 rejections;
  - missing `:conclusion`: 12 graphs, whose edges S7 had been dropping silently;
  - inline-map premises: 2 graphs.

  The three example seeds used in the prompt pass. Three `gh200` graphs, used only by the stub backend and when more than three examples are requested, contain cycles.
- **Replay on the same 98 graphs:**
  - parsed references: 2/956 unresolved (the regex had reported 19/683);
  - anchors: 1/1079 outside the passage, and 10 graphs have no passage `:source`.
- **`adhoc` metrics:** the historical records came from invocations outside the runner. Current runner paths thread ids, S6/S7 refuse `adhoc` with `--run-dir`, and replay fails on any `adhoc` record.

## Outstanding (not claimed)

- **No model run.** Nothing here shows that a real model build over the frozen
  corpus reaches zero rejections. The stricter S3 gate will raise first-pass
  rejection counts until the model follows the new rules; retries and attempt
  history exist for that. Stage 4 fresh-host acceptance remains.
- **S3 proof units vs S1 proof regions.** S3 extracts "proofs" from `proof-move`
  mark groups, not S1 proof environments. Only 42/98 historical graphs overlap an
  S1 proof region. S6 lists `unattached_iatc` per paper but does not fail on it.
  Whether S3 should extract from S1 proof regions, and whether unattached graphs
  should block a whole-paper object, is unresolved and affects what "fully valid"
  means for S6.
- **Item accounting covers S3, S4, S6 and S7.** S1, S2, S5 and S8–S12 remain
  stage-level: command status, existing gates, and replay `C1`/`P1`–`P3`.
- **The S3 command still runs `iatc_anchor_faithfulness.bb … || true`** as a
  documented known-red measurement. It is not an acceptance check. Whether
  acceptance must require it is undecided.
- **Stale finals.** A final without acceptance provenance (for example from code
  before this change) is recorded as `errored` and blocks the stage. Recovery is a
  new run directory. There is no adoption path, by design.
- **S7 retries** retype every accepted graph (temperature 0) instead of carrying
  forward earlier CLeans.
- **Stage 2 boundaries are unchanged:** model revision is operator metadata, and
  eprints are not hashed.
- Rob's raw bundle is still unavailable, so his G7 rejection and his 2 malformed
  graphs are not confirmed to be the same cases. No push, merge, Linode, or
  message to Rob or GitHub.

## S3 rebuilt on identified proofs (after review with Joe)

Joe's direction: S3 "proofs" must be proofs S1 identified, and the model should not
write a data format that then needs repair, retries and extra gates.

- `85c6faf`: S1 now detects French and German proof headings, qualified headings
  ("Proof of Theorem 2:"), macro-defined headings and end marks, and statements given
  as markup headings. Proofs up to 30000 characters are detected; the old 6000-character
  limit dropped longer ones silently. Of the six corpus papers that had no S1 proofs,
  `math/0409598` now has 18 and `math/9810017` has 1. The other four contain no formal
  proofs. All 16 paper objects are well-formed. The corpus now yields 324 proof
  candidates, each paired with a statement.
- `a6fcd08`:
  - one S3 candidate per outermost S1 proof, shown with its statement;
  - the model returns JSON under a strict schema (`iatc_json`) at temperature 0, one call per proof per invocation;
  - code checks references, that each conclusion is a claim and not its own premise, and step order (which rules out cycles), then writes the EDN graph;
  - a violation rejects the item with its reasons; truncated or non-JSON responses are errored;
  - no escape repair, `iatc_repair.bb` call or retry loop remains in S3;
  - the prompt rules and ±3-line changes from `bb91fc9`/`c95d9ed` are superseded. The `iatc_argcheck` cycle, shape and inline-premise gates remain as independent checks that code-written graphs satisfy;
  - `mark4_iatc_concurrent.py`, built on the removed EDN functions, is retired.
- Tests: `tests/test_iatc_json_contract.py` (7). A stub run over 15 real proof
  candidates writes graphs that pass `iatc_argcheck`. Full suite: 36 failed, 896 passed;
  every failure also fails on `eeac70a`.

S4 (`mark3_expository_loop.py`: freeform EDN, escape repair, three attempts) and S7
(`clean_box_typing.py`: unconstrained JSON found by regex, re-prompted when invalid)
still use the pattern S3 has dropped.

## S4 and S7 moved to the same contract

- **S4:**
  - `expository_json` defines the schema: kind enum from the vocabulary with out-of-scope kinds excluded, line numbers bounded to the region, and either `fill` or `held_reason`;
  - code checks that exactly one of those two is filled and that ranges are ordered, then writes the EDN scope graph with the slot name taken from the vocabulary;
  - the loop makes one call per region at temperature 0, with no escape repair or three-attempt retry.
- **Vocabulary loading:** Python `edn_format` misreads keywords with two slashes (`:rationale/telos/organization-roadmap` became `:rationale/telos` plus a stray symbol). It merged two kinds and gave one the other's slot, so the vocabulary is now read with babashka, which the gate also uses. All 16 kinds load.
- **S7:**
  - the typing request uses a schema whose required keys are exactly the graph's box ids, each an enum of vocabulary methods;
  - the model no longer returns `_macro`, which code already overwrote with the derived macro;
  - there is no regex JSON extraction and no re-prompting; a contract violation rejects the graph, and an endpoint failure errors it;
  - waiting for a restarting server (connection refused) is kept, because that is server state, not output format.
- **Tests:** `tests/test_model_json_contracts.py` (6):
  - S4: nested-kind slots; contract checks; code-written EDN passes `expository_argcheck` including escaped held reasons; loop outcomes and retrying only failures; a schema request at temperature 0;
  - S7: schema keys match box ids and a valid typing is accepted; a contract violation is rejected and an endpoint failure errored.
- **Full suite:** 36 failed, 902 passed. Every failure also fails on `eeac70a`; the 2 extractor tests in `test_expository_phase5.py` fail for want of fixture marks.
