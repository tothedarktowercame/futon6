# mark4 — Linode 4-GPU go-live preregistration

*Author: claude-1. Written **2026-06-17, before the run** — predictions committed in
advance so the go-live can confirm or disconfirm them, not be rationalised after the
fact. Baselines cited are real artifacts already on disk; predicted columns are
hypotheses.*

---

## 0. Why preregister

The prior 70B eval was misleading because of a **wiring** flaw, not a model flaw: the
IATC stage (the GPU LLM) ran **first, on raw source**, with none of the upstream CPU
enrichment it is designed to consume. So a poor result could not distinguish "the
model is bad at this" from "we fed it the wrong thing." This go-live fixes the wiring
(`linode-4gpu-run.sh` now rebuilds/validates enriched candidates before the loop, and
`mark4_pipeline_runner.py --plan` proves the GPU stage runs **last**). Because we are
re-testing after changing the setup, we commit the predictions first.

## 1. Headline hypothesis (H-main)

**Feeding the 70B the full upstream enrichment (typed symbols, scopes, proof-moves —
schema `iatc-candidate/v2-enriched`) produces materially better IATC argument graphs
than the blind raw-input run did, clearing the substance bar the blind run failed.**

The sharpest single comparison we have is the 8B run, which **auto-failed 6/10 on the
substance gate** (`linode-4gpu-run.sh` header). H-main predicts the enriched 70B is
on the right side of that line.

### Known confound (stated up front)
This run changes **two** variables at once vs the 8B baseline: model size (8B→70B)
*and* input (raw→enriched). A good result therefore supports "enriched-70B works"
but does **not** by itself isolate how much is enrichment vs size. The clean
disentangling arm — a **70B-on-raw** control — is *out of scope today* (costs a second
GPU run); flagged here so we don't over-claim "enrichment did it" from this run alone.
If the result is ambiguous, that control is the first follow-up.

## 2. Setup under test (frozen before the run)

| Item | Value |
|---|---|
| Box | Linode 4×GPU, Ubuntu 24.04, re-provisioned **with CUDA toolkit (`nvcc`) on the image** if available |
| Setup | `linode-4gpu-setup.sh`: installs Ubuntu base packages, creates `$HOME/mark4-venv`, optionally installs `linode-cli` via `pipx`, then serves vLLM |
| Model | Llama-3.1-70B-Instruct-AWQ-INT4, TP=4, `--enforce-eager`, `--max-model-len 16384`, `--gpu-memory-utilization 0.95` |
| Inputs | `data/iatc-candidates/` — **10/10** candidates, all `iatc-candidate/v2-enriched` (verified on dev box) |
| Run | `linode-4gpu-run.sh`: uses `$HOME/mark4-venv/bin/python` by default; (re)build enriched candidates → wait for vLLM → IATC loop → owner-review gates |
| Order proof | `mark4_pipeline_runner.py --plan`: GPU IATC stage is **last** (position N/N); every prior stage is CPU enrichment |
| Gates after | `iatc_argcheck.bb` + `substance_gate.py` over the output dir |

**Out of scope this run:** the expository ⑤.4 GPU reconstruction (lane C, built+
reviewed on stub, GPU run pending); the APM embedding/pgvector matcher; the 70B-raw
control arm above.

## 3. Predictions (committed; each with a threshold + a falsifier)

| # | Metric | Baseline | Predicted (enriched 70B) | Falsified if |
|---|---|---|---|---|
| P1 | **argcheck** pass rate | 8B/blind: well-formedness frequently broke | **≥ 8/10** graphs pass `iatc_argcheck` after mechanical repair | < 8/10 pass, or repair has to paper over structural (not cosmetic) defects |
| P2 | **substance gate** pass rate | 8B: **4/10** pass (6/10 auto-fail) | **≥ 8/10** pass | ≤ 6/10 (no better than 8B) |
| P3 | **faithfulness** (≥3 graphs spot-checked vs source at cited line anchors) | blind run: anchors often didn't support the claim | cited `:source {:lines}` actually contain the claimed premise/conclusion in **≥ 3/3** spot-checks | any spot-check cites lines that don't support the node |
| P4 | **distribution** (node/edge/hole spread) | risk: template collapse (every graph identical shape) | **non-uniform** across the 10 — varied node counts, real `:holes`/`missing-warrant` where the proof genuinely gaps | near-identical graphs ⇒ the model is templating, not reading |
| P5 | **APM ⑥ coverage** on the live pool | frozen CPU set: `type_multichar` mean 0.257 / median 0.136, 13 proofs ≥80% (gate: mean≥.20, median≥.10, tail≥10) | gate **holds** on the live run (within noise of the frozen numbers) | gate fails on the live pool ⇒ the frozen numbers were not representative |

## 4. What we learn regardless of outcome

- **P2 high, P3/P4 also high** → H-main confirmed; the wiring fix was the story;
  proceed to scale + the side-by-side demo refresh (H6) on real graphs.
- **P2 high but P4 shows collapse** → the model passes gates by templating; the gates
  are too loose, not the model good. Tighten substance/distribution checks before
  trusting the pass rate.
- **P2 high but P3 (faithfulness) fails** → graphs are well-formed and substantive but
  *not grounded in the source*; the enrichment improved form, not reading — points at
  the prompt's source-citation contract, not the model.
- **P2 no better than 8B** → either enrichment isn't reaching the model (re-inspect
  the candidate→`build_prompt` path) or the task is genuinely hard at 70B; the 70B-raw
  control (§1) becomes necessary to tell which.
- **P5 fails** → the APM scope-coverage signal is sample-dependent; couples to
  codex-1's D arm (non-keyword random pool) for whether scope adds anything over
  keywords.

## 5. Stop / failure conditions

- **vLLM won't serve** (KV-cache OOM etc.) → the run aborts at the wait-loop;
  not a pipeline result. Re-check `--max-model-len` / utilization, don't grade it.
- **A substance-gate catch may be a true positive**, not a model failure — e.g. the
  GH200 `1308.1804` case (`X-implies-X` vacuous edge) was a *real* catch (H5). Classify
  each failure real-vs-fixture before counting it against P2.
- **One ambiguous "fail"** (cf. `0711.1761`) is not a verdict; the side-by-side CPU-vs-
  GPU demo (`build_iatc_goldens.py`) is how we adjudicate borderline graphs.

## 6. Artifacts this run should leave behind

- `data/iatc-argument-graphs/loop-run-70b/` — the 10 graphs (+ `.attempts/`)
- argcheck + substance-gate reports over that dir, with per-paper pass/fail + reasons
- the refreshed side-by-side demo on the **real** 70B graphs (H6)
- a results note scoring P1–P5 against the predictions above — **filled in after**, so
  the preregistration stays honest.
