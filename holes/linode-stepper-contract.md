# Linode pipeline stepper — stage contract

*claude-1 + Joe, 2026-06-22. The contract the Linode "stepper" runs against: the
stage DAG for the adapted proof-structure pipeline (Phase ⑧ of
`pre-superpod-pipeline-readiness.html` + the proofcheck comprehension/strategy
layer), what each stage consumes/produces, its gates, where it HALTS for
inspection, and the go/no-go checks. Goal: a CT batch runs end-to-end on Linode
(~$3/hr) so Phase 2 has proof-structure to reason over — and we **see breakages
live** instead of after, the mark2/mark3 lesson.*

## Stepper semantics (the disciplines, learned the hard way)

1. **Supervised step-through, not fire-and-forget.** The stepper IS the executor
   *and* the gate (mark3 lesson: a planner that isn't the executor is a trap).
   It runs stage N, checks the postcondition, **halts at designated points**,
   writes that stage's metrics report, and awaits operator `go` before N+1.
2. **Precondition gate per stage** — refuse (exit nonzero, print why) if the input
   is missing or stale, *before* any expensive work (the `require_enriched`
   lesson). No silent run on degraded input.
3. **Postcondition gate per stage** — verify the output is well-formed before it
   counts as done (argcheck / substance / clean_argcheck / vocab-gate).
4. **Non-fatal eval tail** — the precious artifacts (IATC graphs, CLean) are never
   aborted by an eval hiccup; eval is `RUN_EVAL=0`-skippable and never fatal.
5. **Finals-only counting** — never recurse into `.attempts/`; an `--include-attempts`
   escape hatch only (the inflated-metrics lesson).
6. **Idempotent / resumable** — re-running a stage reuses prior stages' outputs;
   a stage can be re-run in isolation against a fixed input.

## Stage DAG

```
S0 provision ─► S1 anatomy ─► S2 concept-substrate ─┐
                   │                                  ├─► S6 comprehension ─► (S9 pass-3 mining)
                   └─► S3 IATC-extract ─► S4 CLean ───┤
                                          │           └─► S7 structure-embed ─► S8 export→Rob
                                          └─► S5 strategy-recognition ─────────┘
```
Two GPU/LLM stages (S3 extract, S4 box-typing); everything else CPU. S2 is a
corpus-aggregate barrier (needs the whole batch's marks before it can build the
substrate); the rest pipeline per-paper.

## Per-stage contract

| id | consumes | produces | compute | precondition (refuse if…) | postcondition gate | HALT | go/no-go |
|----|----------|----------|---------|---------------------------|--------------------|:----:|----------|
| **S0 provision** | StackScript, model weights | served 70B (vLLM), repo+venv | gpu | bootstrap not done / GPU absent | `/health` 200; smoke generate | ✔ | model serves at expected flags |
| **S1 anatomy** | `paper.tex` (CT batch) | marks (scopes, symbols, proof-moves, expository regions) | cpu | no `.tex` | `check_invariants` **wf=0**; tagged/math coverage | ✔ | wf=0 holds across the batch sample |
| **S2 concept-substrate** | all marks (barrier) | term-prior, concept-encyclopedia, concept-index | cpu | anatomy not done for the whole batch | substrate non-empty; re-ground pass run | ✔ | **G-coverage**: coverage-vs-corpus-fraction curve *rises* (not flat) |
| **S3 IATC-extract** | marks → enriched candidates (w/ `source-window`) | IATC argument-graph `.edn`/proof | gpu-llm (70B) | candidate lacks v2 schema / source-window (`require_enriched`) | `iatc_argcheck` PASS + `substance_gate` PASS (finals-only) | ✔ | **G-substance**: checker-% / substance-% above floor; structure entropy sane |
| **S4 CLean** | IATC graph | `*.clean.edn` (LLaMA box-typed) | gpu-llm + cpu | IATC graph FAIL | `clean_argcheck` PASS | ✔ | **G-method-vocab**: LLaMA `:method` ∈ controlled vocab; **G-cyclic**: cyclic-equiv rejections *logged*, not silently dropped |
| **S5 strategy-recognition** | `source-window` prose (+ formal trace where paired) | per-proof two-layer profile (discursive + hidden) | cpu | no source-window | recognizer ran | — | discursive recall ≈ Herald-CT baseline (0.71) on a check set |
| **S6 comprehension** | IATC graph (R2d) + recognizer profile + substrate | per-proof comprehension score + verdict | cpu | substrate or recognizer missing | scores computed | ✔ | **G-comprehension**: verdict gate separates weak-extraction from weak-proof; comp *rises* vs earlier corpus fraction |
| **S7 structure-embed** | `*.clean.edn` (PASS) | 33-d structure vectors (+ text baseline) | cpu/gpu-benefit | CLean not gated | vectors L2-normed | ✔ | **G-entropy**: embedding entropy / cluster discriminativeness above threshold (no low-entropy collapse) |
| **S8 export→Rob** | CLean graph + structure vectors | neo4j cypher + pgvector SQL | cpu | embeddings/gate missing | cypher/SQL valid; load smoke-test | ✔ | row counts match; sample ANN query returns sane structural neighbors |
| **S9 pass-3 mining** *(opt)* | CLean corpus + comprehension | conjecture/open-problem map + weak-proof candidates | cpu | comprehension missing | harvest ran | — | recurring gaps surface once (type,concept)-keyed |

## Go/no-go gate registry (the pre-mortem checks, wired to stages)

Each is a **halt-and-decide** the operator confirms before proceeding; a red gate
means stop and fix, not push on.

- **G-coverage (S2)** — does grounding a held-out paper against 10%/50%/100% of the
  corpus *rise* then saturate? If flat → the substrate isn't helping; if it
  saturates at 10% → know the ceiling. Tests Joe's "improves as we run."
- **G-substance (S3)** — checker-% / substance-% above floor on a random
  (non-keyword) sample; catches shell-gaming and enrichment-blind runs.
- **G-method-vocab (S4)** — every LLaMA `:method` is in `clean-method-vocab.edn`;
  off-vocab tags silently pollute the embedding (un-gated today — must add).
- **G-cyclic (S4)** — cyclic-equivalence proofs are *counted and logged* when the
  DAG gate rejects them, never silently dropped (silent truncation reads as
  "covered everything").
- **G-comprehension (S6)** — the floor separates weak-EXTRACTION ("study more")
  from weak-PROOF, and comp rises with corpus. The Phase-2-readiness signal.
- **G-entropy (S7)** — structure embeddings carry discriminative signal (method-tag
  / macro entropy above threshold), not collapsed onto one cluster — else
  retrieval is useless while every card stays green.

## What exists vs. what the stepper still needs

- **Stages with code:** S1 (`render_gh200`/detector+checker), S3
  (`mark3_iatc_loop` + `iatc_argcheck` + `substance_gate`), S4
  (`iatc_to_clean` + `clean_argcheck`), S5 (`strategy_recognizer`), S6
  (`clean_comprehension`), S7 (`clean_structure_embed`), S8
  (`clean_graph_export`), S9 (`clean_hole_harvest`). S0/S2 from the warp + linode
  scripts.
- **The stepper runner — BUILT** (`scripts/linode_stepper.py`): reads the EDN
  contract below, enforces pre/postconditions, runs the gate at each ✔ point,
  halts for inspection, flags host-only GPU stages, and resumes (`--from`/`--to`).
  Verified locally on the CPU tail (S7→S8 with the G-entropy gate; S3 host-halt).
- **Gates built (2026-06-22):** **G-method-vocab** (`clean_vocab_gate.bb`, S4 —
  every `:method`/`:macro` ∈ controlled vocab; 7/7 conformant, un-typed skeleton
  correctly fails). **G-coverage diagnostic** (`coverage_curve.py`, S2) — which
  established a key finding: **G-coverage cannot be a post-hoc gate.** All existing
  concept artifacts are already HAPAX-filtered (term-prior df=1 drop), so a
  post-hoc curve reads ~1.0 flat; the rise happened upstream. The real G-coverage
  must run **INLINE at S2 on RAW per-paper concepts (pre-drop)** as the substrate
  grows — an instrument inside the detector/substrate stage, not a standalone check.
- **Built (2026-06-22):** **G-entropy** (`clean_entropy_gate.py`, S7 — macro-entropy
  + mean off-diagonal cosine; PASS on the demo: entropy 0.98, sim 0.76) and the
  **stepper runner** (`linode_stepper.py`) — reads this contract's EDN, drives
  stages with precondition→cmd→gate→halt, flags host-only GPU stages and resumes
  (`--plan` / `--run --from … --to … [--no-halt]`). Verified locally: S7→S8 run
  with the G-entropy gate firing; S3 halts as host-only.
- **Still to build:** the inline S2 raw-coverage instrument (raw pre-HAPAX
  concepts during the run); wiring the GPU stages' real commands on the host.

## Machine-readable contract (the stepper reads this)

```edn
{:pipeline :adapted-proof-structure
 :discipline {:executor-is-gate true :precondition-refuse true :finals-only true
              :non-fatal-eval true :resumable true :supervised-halts true}
 :stages
 [{:id :S0 :name "provision"        :compute :gpu      :halt true
   :consumes [:stackscript :weights] :produces [:served-70b :venv]
   :pre :bootstrap-done :post :health-200 :go-no-go [:serves-at-flags]}
  {:id :S1 :name "anatomy"          :compute :cpu      :halt true
   :consumes [:paper-tex] :produces [:marks]
   :pre :tex-present :post :checker-wf0 :go-no-go [:wf0-on-batch]}
  {:id :S2 :name "concept-substrate" :compute :cpu :barrier true :halt true
   :consumes [:marks-all] :produces [:term-prior :encyclopedia :concept-index]
   :pre :anatomy-complete :post :substrate-nonempty :go-no-go [:G-coverage]}
  {:id :S3 :name "iatc-extract"     :compute :gpu-llm  :halt true
   :consumes [:marks :enriched-candidates] :produces [:iatc-graph]
   :pre :require-enriched :post [:iatc-argcheck :substance-gate]
   :go-no-go [:G-substance]}
  {:id :S4 :name "clean"            :compute :gpu-llm  :halt true
   :consumes [:iatc-graph] :produces [:clean-edn]
   :pre :iatc-pass :post :clean-argcheck :go-no-go [:G-method-vocab :G-cyclic]}
  {:id :S5 :name "strategy-recognition" :compute :cpu  :halt false
   :consumes [:source-window :formal-trace?] :produces [:strategy-profile]
   :pre :source-window-present :post :recognizer-ran
   :go-no-go [:discursive-recall-baseline]}
  {:id :S6 :name "comprehension"    :compute :cpu      :halt true
   :consumes [:iatc-graph :strategy-profile :concept-index] :produces [:comprehension]
   :pre :substrate-and-recognizer :post :scores-computed :go-no-go [:G-comprehension]}
  {:id :S7 :name "structure-embed"  :compute :cpu      :halt true
   :consumes [:clean-edn] :produces [:structure-vectors :text-baseline]
   :pre :clean-gated :post :vectors-normed :go-no-go [:G-entropy]}
  {:id :S8 :name "export"           :compute :cpu      :halt true
   :consumes [:clean-graph :structure-vectors] :produces [:neo4j-cypher :pgvector-sql]
   :pre :embeddings-present :post :load-smoke-test :go-no-go [:rowcounts :ann-sane]}
  {:id :S9 :name "pass3-mining"     :compute :cpu      :halt false :optional true
   :consumes [:clean-edn :comprehension] :produces [:conjecture-map :weak-proof-map]
   :pre :comprehension-present :post :harvest-ran :go-no-go [:recurring-gaps]}]
 :gate-registry
 {:G-coverage      {:at :S2 :status :inline-only :check "raw pre-HAPAX concept coverage rises with corpus-fraction; MUST run inline at S2 (post-hoc reads flat — artifacts already df=1-filtered; diagnostic coverage_curve.py)"}
  :G-substance     {:at :S3 :check "checker-% / substance-% above floor on random sample"}
  :G-method-vocab  {:at :S4 :check "every :method/:macro in clean-method-vocab.edn (clean_vocab_gate.bb)" :status :built}
  :G-cyclic        {:at :S4 :check "cyclic-equiv rejections logged, not dropped"}
  :G-comprehension {:at :S6 :check "separates weak-extraction/weak-proof; rises with corpus"}
  :G-entropy       {:at :S7 :check "structure-embedding macro-entropy + off-diag cosine discriminative (clean_entropy_gate.py)" :status :built}}}
```

*Cross-refs:* `pre-superpod-pipeline-readiness.html` (Phase ⑧),
`proofcheck-readiness.html` (comprehension/strategy cards),
`E-comprehension-foundation.md`, `E-strategy-recognizer.md`, `E-clean.md`,
`clean_pipeline.sh` (the local end-to-end precursor to the stepper).
