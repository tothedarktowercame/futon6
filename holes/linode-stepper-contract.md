# Linode/superpod pipeline stepper — stage contract

*claude-1 + Joe, 2026-06-22 (corrected). The contract the stepper runs against. This
revision **realigns the DAG to the pipeline we actually built** — the phases in
`pre-superpod-pipeline-readiness.html` (① anatomy → ② concepts → ④ IATC ∥ ⑤
expository → ⑥ APM, ⑦ orch, ⑧ proof-structure embedding) and the proof-check/
comprehension layer in `proofcheck-readiness.html`. It **supersedes the mark5 thin
slice**, which ran only ④ (one proof passage per paper) → ⑧ and skipped ② concepts,
the entire ⑤ expository sibling, and the comprehension layer — see
`mark5-ct100-results.md` D8 and the macro-flow note below.*

## The pipeline this exercises (macro-flow, from the readiness docs)

> ① Weft anatomy (CPU, **whole paper incl. exposition**) → ② Concept/term substrate
> (CPU) → **two GPU reasoning siblings over the same anatomy: ④ IATC = the formal/
> illative argument DAG, and ⑤ Expository-region reasoning = the informal reasoning
> (motivation, heuristics, strategy)** → comprehension/rung ladder grounds both → the
> per-paper graph (B) integrates them → ⑧ CLean structure-embedding ships proof
> structure to Rob. ⑥ APM-match + pass-3 mining are CPU tails.

**Unit discipline (the mark5 correction):** *Phase 1* is per-paper — each paper →
a unified paper-level graph **(object B)**: definitions/theorem-statements as nodes,
**every** proof as an IATC substructure off its statement, expository reasoning as the
connective tissue, concepts grounded. *Phase 2* is **cross-paper/holistic** — the
concept substrate, comprehension, and structure embeddings operate over the whole
corpus; the per-paper B graphs are its input nodes. "Per paper" is only a Phase-1 word.

## Stepper semantics (disciplines, learned the hard way)

1. **Supervised step-through; the stepper IS the executor and the gate** (mark3 lesson).
2. **Precondition gate per stage** — refuse before expensive work on missing/stale input
   (`require_enriched`); and **DAG-completeness** — refuse a stage whose upstream phase
   has no passing ledger entry *for this corpus* (the S2-skip lesson; see
   `superpod-dag-contract.md`).
3. **Postcondition gate per stage** — output well-formed before it counts (argcheck /
   substance / expository-argcheck / clean-argcheck / vocab-gate).
4. **Non-fatal eval tail**; **finals-only counting** (never recurse `.attempts/`).
5. **Idempotent / resumable**; **error-tolerant** at scale (HTTP-skip + resume —
   mark5 crashed on one vLLM 400).
6. **Both siblings or neither.** ④ IATC and ⑤ expository are co-required for whole-paper
   mining; running ④ alone is the mark5 mistake, not a valid run.

## Stage DAG (corrected)

```
S0 provision
  └► S1 anatomy ① (whole paper) ──┬─► S2 concepts ② (corpus barrier) ─┐
                                  ├─► S3 IATC ④ (ALL proofs) ─────────┤
                                  └─► S4 expository ⑤ (ALL regions) ──┤
                                                                      ▼
                              S5 comprehension/rung-ladder  ◄── needs {S2, S3, S4}
                                       │
                                       ▼
                              S6 paper-graph (B)  ◄── integrates {S3 proofs, S4 expository, S2 concepts}
                                       ├─► S7 CLean structure-embed ⑧ ─► S8 export→Rob ⑧
                                       └─► S9 APM-match ⑥ / pass-3 mining (opt)
```
GPU/LLM: S3, S4 (the two siblings), and S7's box-typing. Everything else CPU. S2 is a
corpus barrier (needs all anatomy); S5/S7/S8 are cross-paper (Phase 2).

## Per-stage contract

| id | phase | consumes | produces | scripts (built) | compute | gate | HALT |
|----|-------|----------|----------|-----------------|---------|------|:----:|
| **S0 provision** | infra | StackScript/cluster, weights | served LLaMA/70B, venv | `linode-4gpu-setup.sh` (+ cluster alloc on superpod) | gpu | `/v1/models` 200; smoke gen | ✔ |
| **S1 anatomy** | ① | `paper.tex` | whole-paper marks: scopes, symbols, proof-moves, claim/inference, **exposition regions**, definitions | `dp_paper_view.py`·`check_invariants.py`·`render_gh200.py`·`emit_marks.py` | cpu | `check_invariants` wf=0 | ✔ |
| **S2 concept-substrate** | ② | all marks (barrier) | term-prior, encyclopedia, **concept-index** (corpus-fresh) | `build_golden_paper.py`·`build_term_prior.py`·`build_concept_encyclopedia.py`·`sfc_concept_index.py` | cpu | **G-coverage** (inline, raw concepts) | ✔ |
| **S3 IATC-formal** | ④ | marks → **all** enriched proof candidates | IATC argument-graph `.edn` per proof | `mark3_extract_candidates.py` *(→ all proofs, see needs-build)*·`mark3_iatc_loop.py`·`iatc_repair.bb`·`iatc_argcheck.bb`·`substance_gate.py` | gpu-llm | `iatc_argcheck`+`substance_gate` (finals-only); **G-substance** | ✔ |
| **S4 expository** | ⑤ | marks → **all** expository-region candidates | expository-reasoning graph (filled typed holes) | `mark3_extract_expository_candidates.py`·`mark3_expository_loop.py`·`expository_argcheck.bb`·`expository-superpod-vocab.edn` | gpu-llm | `expository_argcheck` PASS | ✔ |
| **S5 comprehension/rung** | C/D + ⑧.5 | IATC + expository + concept-index | rung-2 profile, R2d coverage, per-proof comprehension + verdict | `iatc_semcheck.bb`·`r2d_concept_coverage.py`·`strategy_recognizer.py`·`clean_comprehension.py`·`cas_select.py`/`cas_cert.py` | cpu | **G-comprehension** (verdict separates weak-extraction from weak-proof) | ✔ |
| **S6 paper-graph (B)** | synth | {proofs S3, expository S4, concepts S2} | unified paper-level graph: statements/defs as nodes, proofs as substructures, exposition as edges | **needs-build** (`paper_graph_assemble.py`) | cpu | B well-formed: every statement has its proof substructure or a flagged hole | ✔ |
| **S7 CLean structure-embed** | ⑧ | IATC graphs (and B) | `*.clean.edn` (box-typed) + 33-d structure vectors | `iatc_to_clean.py`·`clean_box_typing.py`·`clean_argcheck.bb`·`clean_vocab_gate.bb`·`clean_structure_embed.py` | gpu-llm + cpu | `clean_argcheck`+**G-method-vocab**+**G-cyclic**; **G-entropy** | ✔ |
| **S8 export→Rob** | ⑧ | CLean graph + vectors | neo4j cypher + pgvector SQL (+ DarkTower Lean) | `clean_graph_export.py`·`clean_to_lean.py` | cpu | cypher/SQL valid; load smoke-test | ✔ |
| **S9 APM-match / pass-3** | ⑥/opt | B + comprehension | APM scope-coverage; conjecture/weak-proof map | `mark4_apm_structure_coverage.py`·`clean_hole_harvest.py` | cpu | APM gate / recurring-gaps keyed | — |

## Go/no-go gate registry

- **G-coverage (S2)** — raw pre-HAPAX concept coverage *rises* with corpus-fraction
  (inline; `coverage_inline.py`). Built.
- **G-substance (S3)** — checker-% / substance-% above floor on a random sample;
  catches shell-gaming. Built (`substance_gate.py`).
- **expository-argcheck (S4)** — the expository analogue of `iatc_argcheck` (3 golden
  PASS / 6 negatives FIRE). Built (`expository_argcheck.bb`, cfec4f9) — **GPU run never
  exercised** (mark5 skipped ⑤).
- **G-method-vocab (S4→S7)** — every `:method`/`:macro` ∈ `clean-method-vocab.edn`.
  Built (`clean_vocab_gate.bb`).
- **G-cyclic (S7)** — cyclic-equivalence proofs logged + set aside, never dropped. Built.
- **G-comprehension (S5)** — verdict separates weak-EXTRACTION ("study more") from
  weak-PROOF; corpus-relative. Built (`clean_comprehension.py`).
- **G-entropy (S7)** — structure embeddings discriminative, not collapsed.
  **mark5 finding: FAILS at scale** — the 5 macro-shapes collapse (98/102 one macro);
  signal is method-level. Built (`clean_entropy_gate.py`); the *vocabulary* needs rework.

## What's built vs. what this run still needs

- **Built and ready to exercise** (most of it): S0, S1, S2, S3 spine, S5, S7, S8, S9
  scripts all exist and are individually validated (readiness cards 1.x–8.x).
- **Built but NEVER exercised on GPU — the conspicuous mark5 gap:** **S4 expository**
  (`mark3_expository_loop.py` + `expository_argcheck.bb`, cfec4f9, stub-tested only).
  The whole point of the corrected run is to exercise ⑤ alongside ④.
- **Needs a small change:** **S3 candidate extraction must yield ALL proof passages
  per paper**, not the single `choose_passage` window (the mark3-demo one-proof
  bottleneck — `mark5-ct100-results.md` D8).
- **Needs build:** **S6 paper-graph (B) assembler** — the one genuinely-new component;
  the rest is composition of built stages.
- **Needs rework (not just a run):** the **macro vocabulary / embedding weighting**
  (G-entropy collapse, mark5 D1/D2) before S7's embeddings are useful for retrieval.
- **Needs re-sync:** `linode_stepper.py`'s `OPS` command-map still carries the mark5
  stage semantics (S4=CLean, S5=strategy, S6=comprehension…); it must be re-keyed to
  these corrected stage IDs (S4=expository, S5=comprehension, S6=paper-graph-B,
  S7=clean-embed) before `--run` drives them. `--plan`'s stage list/gates already read
  correct from this contract; only the per-stage commands lag.

## Machine-readable contract (the stepper reads this)

```edn
{:pipeline :proof-structure-full
 :supersedes "mark5 thin slice (④→⑧, one proof/paper, no ②/⑤/comprehension)"
 :units {:p1 :per-paper :p2 :corpus}
 :discipline {:executor-is-gate true :precondition-refuse true :dag-completeness true
              :finals-only true :non-fatal-eval true :resumable true :error-tolerant true
              :both-siblings-or-neither true}
 :stages
 [{:id :S0 :name "provision"          :phase :infra :compute :gpu     :depends-on []            :halt true}
  {:id :S1 :name "anatomy"            :phase :p1    :compute :cpu     :depends-on [:S0]         :halt true
   :note "WHOLE paper: proofs + definitions + theorem statements + expository regions (dp_paper_view, all flags)"}
  {:id :S2 :name "concept-substrate"  :phase :p2    :compute :cpu     :depends-on [:S1] :barrier true :must-be-corpus-fresh true :halt true
   :go-no-go [:G-coverage]}
  {:id :S3 :name "iatc-formal"        :phase :p1    :compute :gpu-llm :depends-on [:S0 :S1]     :halt true
   :go-no-go [:G-substance] :note "extract + reconstruct EVERY proof in the paper (not one passage)"}
  {:id :S4 :name "expository"         :phase :p1    :compute :gpu-llm :depends-on [:S0 :S1]     :halt true
   :go-no-go [:expository-argcheck] :note "THE restored sibling: fill the 16 typed-hole expository scopes over ALL regions; built (cfec4f9) but never GPU-run"}
  {:id :S5 :name "comprehension"      :phase :p2    :compute :cpu     :depends-on [:S2 :S3 :S4] :halt true
   :go-no-go [:G-comprehension] :note "rung-ladder + R2d + strategy recognizer; needs BOTH siblings + concepts"}
  {:id :S6 :name "paper-graph-B"      :phase :p1    :compute :cpu     :depends-on [:S2 :S3 :S4] :halt true :needs-build true
   :note "unified paper-level graph: statements/defs nodes, proofs as substructures, exposition as edges"}
  {:id :S7 :name "clean-embed"        :phase :p2    :compute :gpu-llm :depends-on [:S3 :S6]     :halt true
   :go-no-go [:G-method-vocab :G-cyclic :G-entropy] :note "macro-vocab/embedding-weighting rework needed (mark5 G-entropy collapse)"}
  {:id :S8 :name "export"             :phase :p2    :compute :cpu     :depends-on [:S7]         :halt true}
  {:id :S9 :name "apm-match+mining"   :phase :p2    :compute :cpu     :depends-on [:S5 :S6]     :halt false :optional true}]
 :gate-registry
 {:G-coverage          {:at :S2 :status :built-inline}
  :G-substance         {:at :S3 :status :built}
  :expository-argcheck {:at :S4 :status :built-stub :note "GPU run never exercised — the mark5 gap"}
  :G-method-vocab      {:at :S7 :status :built}
  :G-cyclic            {:at :S7 :status :built}
  :G-comprehension     {:at :S5 :status :built}
  :G-entropy           {:at :S7 :status :built :finding "FAILS at scale (macro collapse) — vocab rework needed"}}}
```

*Cross-refs:* `pre-superpod-pipeline-readiness.html` (phases ①–⑧),
`proofcheck-readiness.html` (rung ladder + comprehension), `superpod-dag-contract.md`
(phase-completeness enforcement, inherits this DAG), `mark5-ct100-results.md` (what the
thin slice showed + why), `E-clean.md`, `E-comprehension-foundation.md`,
`E-strategy-recognizer.md`.
