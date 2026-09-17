# PR #51 Stage 3 discovery: accounting gaps and defect causes

Date: 2026-09-17. Branch `work/pr51-response` at `eeac70a`. Author: claude-9,
taking over from codex-14. This note covers discovery only. Nothing in it has been
implemented yet. Local evidence comes from Joe's historical `mark7z` artifacts in
`/home/joe/code/futon6`. Rob's raw bundle is still unavailable, so any link to his
reported counts is a hypothesis until his artifacts are compared.

## 1. Places where a stage can pass without accounting for every item

| Stage | Code | What happens |
|---|---|---|
| S3 extract | `mark3_extract_candidates.py:259-262` | A paper with no marks or no selectable proof prints `skip` and is omitted; exit 0. The frozen corpus silently shrinks. |
| S3 loop | `mark3_iatc_loop.py:315, 343-346` | Attempts are scoped by run id but not by invocation; retrying in the same run overwrites `<pid>.attemptN.edn`. An existing final counts as `pass (resumed)` without recording which attempt produced it. Returns 1 on any failure, but no per-item record survives apart from stdout. |
| S4 extract | `mark3_extract_expository_candidates.py:100-105` | Any exception, including a text mismatch, prints `skip` and exits 0. |
| S4 loop | `mark3_expository_loop.py:255-339` | `.attempts` is not run-scoped and there is no resume. A re-invocation re-samples every candidate at temperature 0.2/0.5 and overwrites finals that had already been accepted. |
| S6 | stepper `OPS["S6"]` | `|| exit 1` stops at the first malformed paper, so later papers get no evaluation or artifact. |
| S7 | `clean_box_typing.py:153-163, 206` | A CLean that fails `clean_argcheck` (G1-G8, including G7) is deleted and the script **exits 0**: "typed or cleanly rejected" is treated as success. A G7 rejection therefore still gets a passing S7 ledger row. |
| S7 loader | `iatc_to_clean.py` `load_graph` | Infer edges missing `:id`/`:conclusion` are dropped with no record (2 historical cases, see `TN-cycle-detector-reconciliation.md` §5). |
| Stepper | `linode_stepper.py:470-482` | Only passes are recorded. Failed attempts leave nothing but appended logs: no attempt row, rc, or item counts. |

## 2. Causes of the reported defects (local reproductions)

### Malformed paper graphs (S6), caused by S1 anatomy extraction

Historical `data/iatc-paper-graphs/mark7z` has 2 of 16 papers not well-formed:
`0708.1921` (0 statements, 2 proofs) and `0708.2185` (one proof precedes every
statement). That is the same count as Rob's 10/12, but whether they are the same
papers is unconfirmed.

- `dp_paper_view.py` `_TEXT_PROOF_START_RE` matches `Proof\.` case-insensitively
  and without a sentence-start anchor. Sentence endings get marked as proof
  starts:
  - `0708.2185` at offset 5251: "We will provide the mi\-ssing **proof.** Our argument…" (introduction).
  - `0708.1921` at offsets 47472 and 71304: "…this completes the **proof.**\n\frp\n\thm…"
    The region runs from the end of one proof across the following theorem statement.
- `0708.1921` delimits statements with author macros (`\thm … \eth`) and proofs
  with `\prf … \frp`. `_PROOF_MACRO_PAIRS` knows `prf/eprf`, not `prf/frp`, and
  there is no statement-macro detector, so the paper has zero statements. All of
  its real proofs are missing from the anatomy, not just unattached.
- S3 `--all-proofs` consumes the same marks. The bogus regions also become IATC
  candidates, which is a plausible upstream source of bad anchors and graphs in S3.

The fix belongs in S1 detection, with these two papers as fixtures. The S6
well-formedness rule stays as it is.

S6 has two further issues:
- IATC attachment (`paper_graph_assemble.py:77-84`) attaches graphs only when
  exactly one graph matches the paper by substring. With `--all-proofs` a paper
  has several graphs, so every proof gets `iatc: None`.
- The assembler never reads S4 output, although the DAG contract calls the S6→S4
  dependency required ("the whole-paper object is proofs AND exposition AND
  concepts").

### Dangling references: mostly a replay-check defect

Replay `S2-refs-resolve` finds node ids with the regex
`\{:id :([a-zA-Z0-9-]+), :kind :(?:object|claim|ref)`, so it depends on key order
and id character set. On the historical 98-graph `iatc-argument-graphs/run`:

- regex: **19/683** dangling;
- parsed with the pipeline's EDN reader: **2** unresolved, and both are inline
  claim maps used as premises (`0705.0102__p0`, `0708.2185__p0`), not missing ids.

Most of the reported dangling references are the checker failing to recognise
node ids. The two inline-map premises are a real schema question: either the
graph gate should refuse them or the reader should accept them explicitly. The
5% tolerance hides both problems. `S3-anchors-in-passage` uses the same regex
style and also needs to be checked against the parsed graphs.

### G7 cycles: equivalence proofs flattened into two implications

`TN-cycle-detector-reconciliation.md` (2026-08-08) established that each genuine
G7 cycle in the 98-graph run (4 of 4) is an iff proved as two implication edges
sharing nodes. The vector-`:conclusion` serialisation bug that produced six false
G7-labelled rejections was fixed in `eb423f4` (H36). Rob's single G7 rejection
cannot be classified without his CLean. The open decision is how a biconditional
is represented (see §4); G7 itself should not be relaxed.

### `adhoc` provenance

In historical `mark7z` metrics, `adhoc` records come from S5 (1,927), S7 CLean
discharge (88), S6 (28) and S10 lexicon (2). The stepper now threads ids into all
of these. Stage 2's `validate_records` already rejects any record whose
run/corpus differs from the manifest, so these records came from manual
out-of-runner invocations. Two labelling defects remain:
- `clean_box_typing.py` emits `stage="S4"` for S7 metrics;
- `iatc_lexicon_harvest.py` emits `stage="S3"` for S10 metrics.

The replay provenance check should compare `stage` against the producing stage,
and standalone `--run-id/--corpus-id` defaults of `adhoc` should refuse when
`--run-dir` is given.

## 3. Replay tolerances to replace

- `C2-clean-accounting`: a graph counts as accounted for if its id appears
  anywhere in a log. It should read the S7 item accounting and require every
  graph to be typed. Rejected or errored items mean the build is not fully valid.
- `S2-refs-resolve` / `S3-anchors-in-passage`: replace the regex and the 5%
  threshold with parsed checks. Acceptance requires zero; any mid-run abort
  threshold must be reported separately from acceptance.
- `I3-id-families`: requires both old- and new-style ids to be present. That is
  a property of one historical corpus. It should compare the parsed families
  against the frozen corpus, and fail only on collapse or mismatch.
- `I2-metrics-tagged`: `adhoc` is a warning here. Given Stage 2 record
  validation it can no longer be reached, so replace it with the stage-label check.

## 4. Proposed implementation slices (each separately reviewable)

1. **3a Stage attempt record.** The stepper appends an attempt row for every
   execution (stage, invocation id, command/gate rc, outcome, accounting path) to
   `stage-attempts.jsonl`. It writes the ledger pass only when rc is 0 **and**
   the stage's item accounting exists, covers the expected inputs, and contains no
   rejected or errored items. A shared `stage_accounting.py` writer defines the
   item schema: id, paper, status ∈ accepted/rejected/errored/deferred, reason,
   artifacts, attempt refs.
2. **3b S3.** The extractor records per-paper failures instead of skipping them.
   The loop scopes attempts by invocation, records the accepting attempt for each
   final, retries only non-accepted items on re-invocation, and writes accounting.
3. **3c S4.** The same as S3, plus the per-paper cap: a declared deterministic
   selection algorithm, with selected and deferred ids in accounting and the
   manifest. Then lift the Stage 2 refusal.
4. **3d S6.** Evaluate every paper and still fail the stage if any is malformed.
   Attach IATC graphs by proof id and window lines. Whether to consume S4 needs
   Joe's decision.
5. **3e S7.** Gate rejections and dropped edges become rejected items; the stage
   fails. Fix the stage labels.
6. **3f S1 detection.** Anchor text-proof starts to sentence or paragraph
   boundaries. Learn author statement and proof macros from the preamble (or
   `\def`), with 0708.1921 and 0708.2185 as fixtures.
7. **3g Replay.** Make the replacements in §3.
8. **3h Biconditionals.** Depends on the representation decision below.

## 5. Decisions needed from Joe

- **Retry in place.** May a failed S3/S4/S7 be re-invoked in the same run
  directory, keeping accepted items and resampling only rejected ones, with the
  full attempt history preserved? Or must any rejection start a fresh run?
  Sampling is non-deterministic (temperature 0.2/0.5), so this determines what
  "zero final rejections" means.
- **Biconditional representation.** Should the extraction schema gain an
  explicit equivalence edge (for example `:relation :iff` with a two-directional
  conclusion) that the CLean renders as an equivalence box, so G7 stays strict?
  Or is an iff flattened into a cycle an extraction error to reject and retry?
- **S6 and S4.** Should the paper graph attach expository scopes now, as the DAG
  contract says it must, or is that edge removed from the contract for this
  acceptance scope?
- **Cap selection rule.** Proposed: per paper, in source order, stratified
  round-robin by region type up to N; the rest deferred.
- **Dispatch.** CLAUDE.md defaults to belling Codex for each slice, with claude-9
  reviewing. Confirm, or say "no bells or whistles".
