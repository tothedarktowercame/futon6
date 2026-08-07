# Adversarial review of the capability proof against the runnable pipeline

**Review date:** 2026-08-07  
**Reviewer:** codex-5  
**Artifact reviewed:** `futon3c/holes/labs/M-diagramprover/capability-proof-arxiv.tex`
and its three input files and built PDF  
**Runtime evidence:** Zone checkout `~/code/futon6`, especially
`data/runs/mark7z/`

## Verdict

**The capability proof is not yet safe to hand to a third party as a proof of an
executed end-to-end capability.** Most of its object-level census is real and
reproducible: the 320,337 marks, 98 argument graphs, 883 nodes, 419 edges, 410
holes, 280 expository scopes, 358 typed boxes, 198 carried holes, 58% missing-
warrant rate, 88/10/0 typing accounting, and 0.02 entropy-gate cosine all check
out. The paper's strongest claim does not. The run labelled
`math-ct-e2e-16` has ledger evidence for only **four of twelve** stages, and the
repository's own replay refuses the claimed completed run. More seriously, the
current stepper can report a refusal or gate failure while returning process
exit code 0; its dependency graph describes an older stage numbering; S2 checks
a pre-existing substrate instead of producing a corpus-fresh one; S3's directory
gate currently mistakes rung-2 reports for proof graphs; S6 writes to a path the
retrieval manifest does not retrieve and has only 12 of 16 paper objects, two
not well formed; and S9 still masks the first sub-command's failure with `;`.
The paper should be presented as a strong **component capability census plus a
partially witnessed integration claim**, not as `12/12` executed end to end,
until one clean, run-isolated replay produces a complete ledger and passes the
replay harness.

## Method and evidence grades

I use the paper's warrant vocabulary:

- **Mechanical:** I ran the checker or recomputed the value from the artifact.
- **Artifact audit:** the value is directly present and internally consistent,
  but I did not reproduce the producer.
- **Designed:** the contract or code describes the behavior, without a clean run
  witnessing it.
- **Refused / not verified:** the necessary artifact is absent, ambiguous, or
  contradicted by a stronger check.

Before every Zone command I ran:

```text
pgrep -af "mark3_iatc_loop|linode_stepper"
```

Only the checking shell itself matched. I did not restart or modify any service,
and did not write to the run corpus.

## Q1. Is the capability proof accurate?

### Q1.1 The requested headline counts

| Claim | Result | Warrant and derivation |
|---|---:|---|
| S1 marks | **320,337 verified** | **Mechanical.** I read the 16 IDs from `holes/mark7z-e2e16.ids.txt`, opened exactly `fable-<id>-dp-emacs.json` for each, and summed `len(marks)`. All 16 resolved. The kind counts also reproduce exactly: classified 89,348; symbol-grounded 68,484; math 50,607; concept 46,001; symbol 12,499; bind/typed 7,795; constrain/relation 6,758; let-binder 4,465; definiendum 4,466. |
| S3 nodes / edges / holes | **883 / 419 / 410 verified** | **Mechanical.** `scripts/census16.py` over the 98 canonical final graphs reports 719 claim + 145 object + 19 ref nodes, 419 inference edges, and 410 top-level holes. |
| S4 scopes / kinds | **280 / 20 verified** | **Mechanical.** The same census finds 280 scope graphs and 20 distinct scope kinds. |
| S7 typed boxes / methods | **358 / 8 verified** | **Mechanical.** The 88 CLean files contain 358 boxes in the eight distributions printed in the paper (204, 100, 16, 13, 11, 7, 4, 1). |
| S7 sorry holes | **198 verified** | **Mechanical.** Recomputed from the CLean corpus by `census16.py`. |
| Missing-warrant rate | **222 / 383 = 57.96%, hence 58% verified** | **Mechanical.** Warrant kinds are 222 missing-warrant, 108 claim, and 53 citation. |
| Typing accounting | **88 typed / 10 cyclic / 0 failed of 98 verified as an artifact census** | **Artifact audit.** There are 88 typed outputs and the recorded census partitions the remaining 10 as cyclic. This is complete object accounting. It is **not** evidence that S7 ran under the `math-ct-e2e-16` stage ledger, which has no S7 entry. |
| Entropy gate | **0.02 verified** | **Mechanical.** `clean_entropy_gate.py --embed data/showcases/clean-run-demo/clean-embed.json` reports 88 proofs, normalized macro entropy 0.53, mean off-diagonal structure cosine 0.02, and PASS against ceiling 0.85. |

The S1 recomputation was deliberately manifest-filtered:

```text
FOUND 16 MISSING [] MARKS 320337
classified 89348
symbol-grounded 68484
math 50607
concept 46001
symbol 12499
bind/typed 7795
constrain/relation 6758
let-binder 4465
definiendum 4466
```

That qualification matters. Running the checked-in `scripts/census16.py`
unchanged today does **not** reproduce the S1 line; it scans the mutable whole
`golden/` directory and reports 122 papers and 2,357,597 marks. The published
number is correct, but its purported census script is not a byte-stable
derivation of it unless supplied the frozen 16-ID manifest.

### Q1.2 Warrant-table audit

| Row | Review verdict |
|---|---|
| **A1** | **Count partly stale.** The artifact census is 16/16 marks files, while the warrant still says 12/12. The sixteen-paper ledger has S1, so the stage is evidenced, but the “twelve complete / four partial” partition has no machine-readable field I could re-derive. |
| **A2** | **Not fully verified.** The sixteen-paper ledger has S2, but OPS checks a pre-existing substrate rather than rebuilding one. I did not find a frozen artifact deriving the separate 4,508/4,616 full-scale claim. H1 itself says the raw-stream/corpus-fresh tier remains open. |
| **A3** | **Core verified; retry rate mismatches.** Selecting the 98 canonical finals explicitly gives argcheck exit 0 and `substance-gate: 98 file(s), 0 failure line(s) — PASS`. Under the natural definition “a final has an attempt numbered at least 1,” 45/98 = 45.9% retried, not 48%; the paper needs to state its denominator/definition or correct the rate. |
| **A3'** | **Not reproducible in this environment.** The checked-in anchor script failed because its hardcoded eprint root did not resolve the first paper. The 41%/median-3 figures may be right, but the warrant is not portable as shipped. The stated post-fix sample is correctly labelled only n=3. |
| **A4** | **Verified for finals.** Passing the 280 top-level files explicitly to `expository_argcheck.bb` gives 280/280 PASS. Passing the directory instead recurses into 465 attempts and gives 180 failures, so the stage must use finals-only selection. I did not independently reconstruct the “238 then 42 all one defect” history. |
| **A5** | **Artifact census verified, stage warrant absent.** The valid proof rows give 202 grounded, 564 thin, 52 ungrounded and verdicts 6/82/10. The shared output also has 98 spurious `no-structure` rows from rung-2 reports. There is no S5 executable criterion and no full CAS certificate in OPS. |
| **A6** | **Refuted.** Only 12 paper objects exist; two have `wellformed: false`; 33 orphan statements are recorded; the output is not in the retrieval path. |
| **A7** | **Component verified.** 88/10/0 and entropy 0.02 reproduce. It is not ledgered as S7 for `math-ct-e2e-16`. |
| **A8** | **Partial.** The run export contains exactly 446 nodes and 286 edges. “95 theorems” is not represented in that graph (it has 88 Proof and 358 Step nodes), and the separate XTDB census is not present in the run directory, so those subclaims are not verified here. |
| **A9** | **Artifact count verified, provenance weaker than stated.** The run lexicon has 732 entries; the census reports 737 moves, 62 relation types, and mean confidence 0.4328. The metrics file is heavily mixed (`adhoc` dominates), and S10 is absent from the sixteen-paper ledger. |
| **A10** | **Refuted as a paper-level result.** Five proof shapes are real, but legacy paper IDs collapse to `math`; the 10 signatures and twin similarity are computed over the wrong paper partition. |
| **A11** | **Artifact verified, run completion not verified.** The nine-point curve and +0.154 rise exist. S12 is absent from the named ledger and has no DAG dependencies or executable gate. |
| **A12** | **Refuted.** The replay reports four of twelve stages, not twelve of twelve. Hazard close count is not mechanically derivable. |
| **A12'** | **Looks like a static catalogue, not a run warrant.** I did not execute all 12 query forms against a frozen query endpoint; treat this as designed/demo evidence unless a query transcript is attached. |
| **A13** | **Correctly labelled designed.** No scale witness is claimed. |
| **A14** | **Correctly labelled registered.** No model-independence conclusion is yet warranted. |

### Q1.3 Claims that do not survive the artifact check

#### 1. `S1--S12 ledgered 12/12` is false for the named corpus

The `math-ct-e2e-16` entries in
`data/runs/mark7z/phase-ledger.jsonl` cover only:

```text
S1, S2, S5, S11
```

The repository's own command confirms the failure:

```text
.venv/bin/python scripts/replay_e2e.py --through S12 \
  --graphs data/iatc-argument-graphs/run \
  --steps data/cas-select-steps/run \
  --ids holes/mark7z-e2e16.ids.txt \
  --corpus-id math-ct-e2e-16
```

It passes the object census and identity checks, but reports:

```text
FAIL ledger: only 4/12 stages for e2e-16;
missing S3,S4,S6,S7,S8,S9,S10,S12
9/11 checks pass, 1 warn, 1 fail — ABORT RECOMMENDED
```

There is an older `math-ct-e2e-12` ledger with most stage names. It cannot be
used as evidence for `math-ct-e2e-16` without violating the contract's own
same-corpus rule. This invalidates A12 as written and removes the claimed
mechanical warrant from A1/A2/A6/A7 where those rows rely on the sixteen-paper
ledger.

#### 2. The paper's current source and its PDF are different artifacts

The PDF was built at 15:10, while `capability-proof-arxiv.tex` was modified at
15:38 and is dirty relative to commit `6251305`. The source changes include the
corpus-provenance paragraph and the S11 warrant. Thus “the built PDF beside it”
is not a build of the reviewed TeX source. A third-party handoff must rebuild and
record the source/PDF hashes.

#### 3. A6 (“paper objects ledgered; orphan check clean”) is contradicted

`paper_graph_assemble.py` defaults to `data/paper-graphs`, while the stepper's
RETRIEVE contract promises `data/iatc-paper-graphs/$RUN_ID`. On Zone:

```text
data/paper-graphs/*.B.json:       12 files
data/iatc-paper-graphs/**:         0 files
wellformed:                       10 true, 2 false
orphan statements:               33
```

The assembler also says expository edges attach “in a later pass”; no such pass
is in S6. A6 is therefore not merely unledgered: its “orphan check clean” claim
is false under the producer's own `wellformed` field.

#### 4. A10's “10 paper signatures” contains an ID-collision artifact

`structural-canon.json` says `n_papers: 10`, but one signature is for a paper
literally named `math`. The producer uses:

```python
bypaper[i.split("__")[0]].append(b)
```

so all legacy arXiv IDs such as `math__0310337` collapse to the same paper.
This is exactly the ID-family class the hazard ledger says was centralized and
fixed elsewhere, but this producer bypasses the shared parser. The five shapes
are a valid proof-level census; the 10-paper signature count and 0.7487 twin
similarity are not reliable paper-level evidence.

#### 5. The hazard close count is not derivable from its ledger

The paper says “H1--H21; 21 found, 20 closed.” The hazard document is not a
state table with 21 uniquely enumerable rows. It includes sub-hazards H11b,
H12b, H19b and H19c, and its visible status headings still say:

- H1 tier 2 **OPEN**;
- H10 **PARTIALLY FIXED**;
- H13 **OPEN** (although the code was subsequently wired);
- H16 “S12 fixed, S11 **OPEN**” (although later code addresses it);
- H21 **FIXED (validating)**.

H22, named in stepper comments and the review request, is not a numbered entry
in `E-superpod-hardening.md`. “20 closed” may be an author's later synthesis,
but it is not mechanically re-derivable from the ledger it cites.

#### 6. The exported “95 theorems” is not present in the run export

The current S8 `clean-graph.json` mechanically gives the claimed 446 nodes and
286 edges, but the nodes are 88 `Proof` + 358 `Step`. I found no run artifact
from which “95 theorems” can be derived, and no XTDB run artifact from which I
could independently reproduce 772 nodes / 419 edges. Those two subclaims should
be cited to their separate benchmark artifact or marked **not verified here**.

#### 7. “222 filled slots” omits a category from its own composition

The listed composition is 83 + 34 + 22 + 13 + 11 = 163. The census reaches 222
only by adding 59 records whose type is the generic `slot`; that category is
omitted from the table. Either list `slot: 59`, or call 163 the typed filled
slots and explain what the generic records mean.

### Q1.4 Claims that are accurate but need narrower wording

- The 98 final argument graphs do pass the graph-level checks when finals are
  selected correctly. However, running the current **stage directory gate** on
  `data/iatc-argument-graphs/run` sees 196 EDN files: 98 graphs and 98
  `.rung2.edn` reports. `substance_gate.py` treats the latter as malformed graph
  artifacts and returns FAIL. The component result is sound; the runnable stage
  integration is not.
- S5's 818 moves and 6/82/10 substantive verdict distribution are present, but
  its shared output also contains 98 `no-structure` rows from processing the
  rung-2 reports as graphs. The paper silently selects the right half.
- The S12 curve artifact exists and reports nine checkpoints and +0.154 rise.
  It is component evidence, not evidence for a ledgered S12 execution on the
  named corpus.
- The paper honestly labels A13 designed and A14 registered. Those boundaries
  should be preserved; they are stronger than the integration warrant currently
  given to A12.

## Q2. Is the runnable pipeline complete relative to the written spec and built code?

No. The missing pieces are predominantly wiring and failure propagation, not
missing algorithms. The most important mismatches follow.

### Q2.1 The machine DAG and the runnable stages describe different pipelines

`linode-stepper-contract.md` and the current OPS table use the corrected
semantics:

```text
S4 expository; S5 comprehension; S6 paper graph; S7 CLean; S8 export
```

But `load_deps()` reads `superpod-dag-contract.md`, whose machine block still
uses the old semantics:

```text
S4 clean; S5 strategy; S6 comprehension; S7 embedding
```

Consequences in the actual `DEPS` map include:

- current S5 consumes S3 graphs but depends only on S1;
- current S6 is allowed after S2/S3/S5 but omits its specified S4 expository
  dependency;
- current S7 consumes S3 and produces CLean but depends only on old S4;
- current S8 depends on old S4 and S7;
- injected S10--S12 have no dependencies at all.

The completeness guard therefore proves completion of a different graph from
the one OPS executes.

### Q2.2 The stepper does not propagate refusal or failure to its caller

Every refusal path in `run()` prints and returns from the Python function;
`main()` then exits normally. A safe dry probe demonstrated:

```text
✗ S1 BLOCKED — upstream S0 has no passing ledger entry for corpus 'audit'
PROCESS_EXIT= 0
```

The same shape is used for command and gate failures. An outer scheduler can
therefore record a successful process while the stepper says it stopped. This
is a release blocker for an unattended run.

### Q2.3 The ledger is weaker than the contract

The contract requires output, fingerprint, structured gate result, reuse flag,
timestamp, and run/corpus identity. `ledger_record` writes only:

```json
{"stage":"S5","corpus_id":"...","run_id":"...","gate":"pass"}
```

It writes that record after command exit even when the criterion is only
printed for a human and never executed. `--mark-done` can create the same pass
record without an output. This explains how “ledgered” drifted into a stronger
claim than the ledger can warrant.

### Q2.4 Stage-by-stage gap table

| Stage | Written specification | What OPS actually does | Gap |
|---|---|---|---|
| S1 | Emit marks and gate them | `emit_marks` + invariant gate | Mostly wired. Output remains in shared `golden/`, so the census needs the manifest to remain run-scoped. |
| S2 | **Build a corpus-fresh** term prior, encyclopedia, concept index, and coverage curve | `warp_substrate_check` + `coverage_inline` over committed `data/warp/concept-usage.json` | **Producer missing from stage.** `build_term_prior.py`, `build_concept_encyclopedia.py`, `sfc_concept_index.py` and the raw-stream instrument are not invoked. H1 tier 2 is genuinely open. |
| S3 | Produce final proof graphs; finals-only argcheck and substance gate | Wrapper produces graphs and rung-2 reports in one directory; post-gate scans `*.edn` | **Current post-gate fails** because 98 rung-2 reports are treated as proof graphs. The wrapper's eval tail repeats the same “all top-level EDN” selection. |
| S4 | All expository regions, bounded at archive scale, explicit gate | Extract + OpenAI loop; region cap remains out-of-band | **Cap unwired** (H10). No stage-level explicit gate command; correctness depends on loop behavior. At full scale this is a cost and completeness risk. |
| S5 | Rung-2, R2d, strategy recognition, CAS select/certificate, symbol grounding, comprehension gate | `cas_segment`, deterministic `rung3_technique`, then `clean_comprehension` | H13's two producers are now wired, but `iatc_semcheck.bb`, `cas_select.py`, `cas_checks.py`, `cas_cert.py`, `sfc_symbol_grounding.py`, `rung3_residue_llm.py`, and `warrant_normalize.py` are absent as executable sub-stages. The criterion is printed, not checked. |
| S6 | Whole-paper object combining proofs, exposition and concepts; orphans flagged/gated | `paper_graph_assemble` with only IATC input, default output `data/paper-graphs` | Does not consume exposition or concepts, does not output to the retrieved run path, returns 0 even when `wellformed` is false, and has no executable gate. Current artifacts are 12/16, with two false `wellformed` values. |
| S7 | CLean producer, clean argcheck, vocab/cyclic/entropy gates | `clean_box_typing` + embedding; vocab + entropy gate | Omits `clean_argcheck.bb`; does not pass run-specific output directories; its inputs include rung-2 reports unless separately filtered. |
| S8 | Graph/pgvector **and Lean** export, syntax/load smoke tests | `clean_graph_export.py` only | `clean_to_lean.py` and export validation/load smoke are not invoked. No gate. |
| S9 | APM coverage plus scoped pass-3 harvest | `mark4_apm_structure_coverage.py ; clean_hole_harvest.py` | The semicolon still masks failure of the first command. Harvest defaults to the global graph tree and writes a shared demo path, not the run corpus. |
| S10 | Harvest and persist the run's lexicon; measure reground lift | Harvest gets run IDs; reground commands mostly print | Corrected corpus parameters are present, but the lift criterion is not executable and the two reground outputs are not persisted as run artifacts. |
| S11 | Extract formulae, definition canon, paper signatures | Three commands joined with `&&` | H22's missing producer and semicolon are fixed. However `sfc_struct_canon` can emit a refusal artifact and exit successfully, and the paper-signature producer still has the legacy-ID collision. No gate distinguishes measured from refused. |
| S12 | Run-scoped accretion checkpoints | `accretion_curves.py` | Producer exists, but S12 is injected outside the machine DAG, has no dependencies, and its `rising` criterion is printed rather than enforced. |
| render/retrieve | Render all papers and retrieve every durable product | Render scripts absent from OPS; RETRIEVE is a boot note | `render_run.py` and `build_proofcheck_demo.py` exist but no stage invokes them. S6's actual output path is absent from the retrieval manifest. |

### Q2.5 Built scripts that are not wired to their claimed stage

The following are not merely “scripts somewhere in a large repository”; the
written pipeline or readiness cards name them as products of this run, while
OPS does not invoke them:

- S2: `build_term_prior.py`, `build_concept_encyclopedia.py`,
  `sfc_concept_index.py` (and the corpus-fresh substrate build generally);
- S5/S6: `sfc_symbol_grounding.py`, `cas_select.py`, `cas_checks.py`,
  `cas_cert.py`, `warrant_normalize.py`, `rung3_residue_llm.py`, and the
  standalone rung-2 harness;
- S7/S8: `clean_argcheck.bb`, `clean_to_lean.py`;
- render tail: `render_run.py`, `build_proofcheck_demo.py`;
- validation/training follow-ups: `herald_validate.py`,
  `herald_ct_endtoend.py` (appropriately out of the core run, but not evidence
  generated by the Superpod stage).

`strategy_recognizer.py` is an important exception: it is not named in the OPS
string, but `clean_comprehension.py` imports and executes it. That one is wired.

### Q2.6 Current Zone readiness

I ran preflight without `--fix` against the 16-ID manifest and the shared local
endpoint. It reported 7/9:

```text
[FAIL] eprints:resolvable  FUTON6_EPRINTS unset
[FAIL] model:endpoint     endpoint reachable; requested model name not in /models
```

This does not show that the data or server are absent; it shows the exact launch
environment I tested is not run-ready. The preflight is doing the right thing.
The launch playbook should record the actual served model name and export the
eprint root before treating the host as ready.

## Q3. Which partial/build cards can become ready before the Superpod?

### Q3.1 Classification

| Card | Actual class | Decision |
|---|---|---|
| **CAS-SEL** | **(b) needs wiring** | The selector, registry, checks, segmenter and certificate exist. OPS runs only the segmenter and uses the pattern library indirectly; it never runs `cas_select`, selected checks, or `cas_cert`. Tier-1 model verification is then a small evidence pass, not a new capability. “build” is stale. |
| **RAW-CTL** | **(a) needs evidence reconciliation; the run already exists** | The dashboard card says build, but `data/exp-20260618/loop-run-70b-raw` and its report exist: ten papers, 12.5% warrant grounding versus 21.4% enriched. The old report accidentally counted graph and rung-2 EDN together (20 items, substance 10/20), so rerun the modern finals-only analytic before marking READY. No model run is required unless a modern replication is desired. |
| **SFC2b** | **(b) needs wiring, then (a) evidence** | `sfc_symbol_grounding.py` works for one formula/context and has an OpenAI backend. There is no corpus driver in S2/S5 and no run-scoped output contract. Build the thin batch adapter, then run the 16-paper sample locally. |
| **rung-3** | **(a) needs evidence** | Deterministic `rung3_technique.py` is now wired and produced the 818-move census. The remaining card claim is the bounded LLM-on-residue pass; `rung3_residue_llm.py` exists. If its questions are intended as a standard run product it also needs a small OPS hook, but the capability itself needs a run, not a build. |
| **RENDER** | **(b) needs wiring** | Both renderers exist and `render_run.py --all` already expresses the mechanical corpus loop, but no render-tail stage invokes it and its directories are hardcoded to legacy runs. Parameterize the run paths and add a post-S8 stage. |
| **STRAT-REC** | **(a) needs evidence / calibration** | It is executed inside `clean_comprehension`; the core recognizer exists. The remaining work is to run and review the math.CT misses, grow the vocabulary, and freeze a measured recall/error report. |
| **WARRANT-NORM** | **(b) needs wiring** | `warrant_normalize.py` exists but no stage invokes it. Its default input is the global graph tree and its default output is a shared demo path. Give it the S5/S6 run corpus and persist its `(type, concept)` vocabulary under the run directory. |
| **PASS3-HARVEST** | **(b) needs correct wiring** | It appears in S9, but after `;`, reads the global graph tree, and writes a shared path. Thus “invoked” is not equivalent to “a run-scoped stage product.” Fix command propagation and pass explicit run paths; then it is CPU-ready. |
| **LEAN-NL** | **(c) needs build for the remaining claim; core should be reclassified READY** | The card itself says the core validation is done at 0.71 recall. The outstanding per-step attribution and attachment of the hidden Lean layer into CLean are new integration capability, not merely another measurement. Split those into a new build card rather than leaving the already-witnessed core partial. |

### Q3.2 What can be completed cheaply before the window

Everything in (a) and (b) is small-scale CPU or bounded local-model work. The
served GLM-4.5-Air endpoint is sufficient once the correct model name is taken
from `/v1/models`; none of these needs an eight-GPU Superpod allocation.

## Workplan, ordered by value to the Superpod run divided by cost

| Priority | Work item | Cards / stages | Cost | Completion evidence |
|---:|---|---|---|---|
| 1 | **Make failure loud:** return nonzero on dependency refusal, command failure, or gate failure; replace S9 `;` with `&&`; add regression tests for shell exit status. | Core runner, PASS3 | Very low | A refused dry run has nonzero process exit; deliberately failed first sub-command prevents ledger write. |
| 2 | **Reconcile the DAG source of truth** with current S4--S9 semantics and add S10--S12 dependencies. Add real `inputs` and run-isolated output paths. | All | Low | `--plan` dependency audit matches `linode-stepper-contract.md`; unit tests refuse S5 without S2/S3/S4 and S12 without its producers. |
| 3 | **Fix artifact selection and S6 durability:** exclude `*.rung2.edn` everywhere proof graphs are globbed; make S6 consume the specified inputs, write `data/iatc-paper-graphs/$RUN_ID`, and fail its gate on unattached proofs. | S3, S5, S6 | Low | Current 16-paper replay no longer sees 196 “graphs”; S6 output appears in RETRIEVE path and every false `wellformed` is a gate finding. |
| 4 | **Re-run the already-existing RAW-CTL analytic** with finals-only readers and its registered comparator. Update the stale readiness card rather than spend model tokens repeating the run. | RAW-CTL | Very low | Frozen report with identical paper set, model, 10 finals per arm, and modern gate outputs. |
| 5 | **Make S9 run-scoped:** wire `warrant_normalize` and `clean_hole_harvest` with explicit graph/output paths; preserve both artifacts under `$RUN`. | WARRANT-NORM, PASS3 | Low, CPU | 16-paper normalized-hole vocabulary and pass-3 map; rerun byte-identical. |
| 6 | **Wire the deterministic CAS chain** (`cas_select` -> selected checks -> `cas_cert`) after segmentation and before comprehension/paper assembly. | CAS-SEL | Low, CPU | Per-proof select and certificate artifacts for all 98 finals; every selected check recorded; no offered-but-unused pseudo-output. |
| 7 | **Add a manifest-driven SFC2b batch adapter** and run it over the 12 complete / 16 declared papers against the local endpoint. | SFC2b | Medium, bounded LLM | Run-scoped symbol files, evidence support/unsupported rates, resumable manifest, no global defaults. |
| 8 | **Run rung-3 residue only**, two-way, with a registered cap and persist questions; separately score strategy-recognizer misses and update the vocabulary. | rung-3, STRAT-REC | Medium, bounded LLM + CPU | One call per residue asserted; question artifact and reviewed sample; before/after recognizer recall. |
| 9 | **Parameterize and invoke `render_run --all` as a post-S8 tail.** | RENDER | Low-to-medium, CPU | Render count equals eligible paper count; skips have structured reasons; artifacts included in RETRIEVE. |
| 10 | **Split LEAN-NL status:** mark the 0.71-recall core READY; register per-step attribution / hidden-layer attachment as a separate build. | LEAN-NL | Medium build, not window-blocking | New card has explicit input/output and one small end-to-end CLean example before scale-up. |
| 11 | **Only after 1--10, execute a clean 12--16 paper rehearsal in a fresh run namespace.** Rebuild S2 rather than check it; run replay through S12; rebuild the PDF from committed TeX. | Integration warrant | Hours on CPU/local model | `replay_e2e --through S12` 11/11 PASS, 12/12 same-corpus ledger entries with hashes and gates, no `adhoc` metrics, clean RETRIEVE, source/PDF hashes recorded. |

The decisive acceptance criterion is not another component count. It is one
fresh run for which the stepper's process status, stage ledger, output paths,
replay harness, and paper all refer to the same corpus and agree. Once that
exists, the already-strong census can support the paper's capability-proof
framing without asking the reader to bridge the integration gaps by trust.
