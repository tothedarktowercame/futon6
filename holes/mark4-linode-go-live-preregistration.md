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
| Provisioning | Private StackScript `mark4-ubuntu2404-gpu-bootstrap` (`2142757`) on `linode/ubuntu24.04`: installs NVIDIA drivers, optional CUDA toolkit, optional `linode-cli`, then reboots |
| Setup | after the StackScript reboot, `linode-4gpu-setup.sh`: verifies `nvidia-smi`, creates `$HOME/mark4-venv`, then serves vLLM |
| Model | Llama-3.1-70B-Instruct-AWQ-INT4, TP=4, `--enforce-eager`, `--max-model-len 16384`, `--gpu-memory-utilization 0.95` |
| Inputs | `data/iatc-candidates/` — **10/10** candidates, all `iatc-candidate/v2-enriched` (verified on dev box) |
| Run | `linode-4gpu-run.sh`: uses `$HOME/mark4-venv/bin/python` by default; (re)build enriched candidates → wait for vLLM → IATC loop → owner-review gates |
| Order proof | `mark4_pipeline_runner.py --plan`: GPU IATC stage is **last** (position N/N); every prior stage is CPU enrichment |
| Gates after | `iatc_argcheck.bb` + `substance_gate.py` over the output dir |

**In scope — added 2026-06-17 (Joe):** the **expository ⑤.4 GPU reconstruction**
(lane C) — its **first real-GPU run** (it had only been stub-validated). Run on a
**bounded 10-candidate sample** (1 inflight region per paper) against the same vLLM
70B; the full carve is **2058 regions**, a separate scaling job. Scored as **P6**.

**Out of scope this run:** the APM embedding/pgvector matcher; the 70B-raw control
arm above; **full-coverage** expository over all 2058 regions (sample only this run).

Provisioning template:

```bash
linode-cli linodes create \
  --region us-ord \
  --type g2-gpu-rtx4000a4-s \
  --image linode/ubuntu24.04 \
  --stackscript_id 2142757 \
  --label mark4-70b-$(date +%Y%m%d) \
  --root_pass 'REPLACE-ME' \
  --authorized_keys "$(cat ~/.ssh/id_ed25519.pub)"
```

## 3. Predictions (committed; each with a threshold + a falsifier)

| # | Metric | Baseline | Predicted (enriched 70B) | Falsified if |
|---|---|---|---|---|
| P1 | **argcheck** pass rate | 8B/blind: well-formedness frequently broke | **≥ 8/10** graphs pass `iatc_argcheck` after mechanical repair | < 8/10 pass, or repair has to paper over structural (not cosmetic) defects |
| P2 | **substance gate** pass rate | 8B: **4/10** pass (6/10 auto-fail) | **≥ 8/10** pass | ≤ 6/10 (no better than 8B) |
| P3 | **faithfulness** (≥3 graphs spot-checked vs source at cited line anchors) | blind run: anchors often didn't support the claim | cited `:source {:lines}` actually contain the claimed premise/conclusion in **≥ 3/3** spot-checks | any spot-check cites lines that don't support the node |
| P4 | **distribution** (node/edge/hole spread) | risk: template collapse (every graph identical shape) | **non-uniform** across the 10 — varied node counts, real `:holes`/`missing-warrant` where the proof genuinely gaps | near-identical graphs ⇒ the model is templating, not reading |
| P5 | **APM ⑥ coverage** on the live pool | frozen CPU set: `type_multichar` mean 0.257 / median 0.136, 13 proofs ≥80% (gate: mean≥.20, median≥.10, tail≥10) | gate **holds** on the live run (within noise of the frozen numbers) | gate fails on the live pool ⇒ the frozen numbers were not representative |
| P6 | **expository ⑤.4** first real-GPU run (stub-only before) — *exploratory* | stub: 1/1 gated PASS (plumbing only); no real-GPU baseline | **≥ 6/10** sample candidates yield an `expository_argcheck`-passing graph, with non-uniform output | < 6/10 pass, or template collapse across the 10 (⇒ the phase isn't ready for real input) |

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

## 7. RESULTS — filled in after the run (2026-06-17, claude-1 driving)

Box `172.232.13.6` (4× RTX 4000 Ada, driver 580.159.03, driver-only path with
`--enforce-eager` + FlashInfer sampler off). vLLM 70B-AWQ-INT4 TP=4 served; IATC loop
over the 10 enriched candidates, then the expository ⑤.4 sample (10, 1 region/paper).

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | argcheck ≥8/10 after repair | **9/9 final graphs PASS (100%)**; the failed paper's best attempt is *also* well-formed | **PASS** |
| P2 | substance ≥8/10 | **9/10 as-run → 8/10 under the sharpened gate** (`249aa83` reads *all* premise tokens, so `0712.0724`'s `[:F-functor :F-pitchfork]→:F-pitchfork` self-loop is now caught — the old first-premise-only check missed it); ≥8 → **PASS** | **PASS** |
| P3 | cited lines support the node, 3/3 | content-faithful (terms/claims drawn from the right region) but **line-anchor precision loose** on ~1/3 of spot-checked nodes (e.g. `0709.0248` cites `\begin{proposition}` rather than the statement; one off-by-one) | **PARTIAL** |
| P4 | non-uniform distribution | nodes **6–12**, edges 2–5, holes 1–3; real `:holes` throughout; no template collapse | **PASS** |
| P5 | APM ⑥ coverage gate holds | `type_multichar` mean **0.257** / median **0.136**, tail≥80% = **13**, **`gate_pass: true`** (135 proofs) | **PASS** |
| P6 | expository ⑤.4 first real-GPU run, ≥6/10 gated + non-uniform | **10/10 gated PASS**; 4 distinct scope kinds (example-source, literature-gap, computes-invariant, difficulty-assessment) with real slot-fills | **PASS** (exceeds) |

**Headline (H-main): supported.** Enriched 70B clears the substance bar the blind 8B
run failed (8/10 under the sharpened gate, vs 4/10). The confound stands — size *and* input both changed — so the
70B-on-raw control remains the clean follow-up before claiming "enrichment did it."

**Self-loop catches — true positives, not fixture bugs:** substance rejected `0708.2185`
for *"self-loops (`:premise == :conclusion`) — vacuous X⊢X"* (same class as the prereg's
GH200 `1308.1804`). And post-run, the **sharpened** substance gate (`249aa83`, all premise
tokens) also flags `0712.0724` (conclusion among a *later* premise token) — which **R2b
closure independently flags too**, so two checks agree. The gate is discriminating;
the honest rate is **8/10 under the improved gate** (9/10 as originally run), still ≥8 → P2 PASS.

**P3 is the actionable finding:** the model reads the right region but its line anchors
are imprecise — points at the prompt's source-citation contract (matches §4's "form, not
reading" branch), not a model-capability ceiling. Tightening the citation contract is the
next IATC-prompt iteration.

**Grading note — finals-only (FIXED).** The shared graders originally recursed into
`.attempts/` (`mark3_eval_harness` rglob, `iatc_argcheck` file-seq), grading **20**
artifacts (9 finals + 11 retries) and scoring the failed paper's attempts as passes;
`substance_gate` (top-level glob) was already correct. Fixed on `mark4-held-steps`
(codex-2 authored, claude-loop reviewed, claude-1 re-verified): both now default to
**finals-only** with an opt-in `--include-attempts`. Re-run confirms **9 artifacts,
checker 9/9, substance 9/9**. The P-table above is finals-only.

**`grounding-%` REDEFINED — now real and discriminating.** The old metric
(`#graph files / #layer-a marks`) divided count-by-count, never inspected groundings,
and sat at ~0.01% regardless of enrichment. Redefined (commit `cc808d4`) as
**warrant-resolution** = resolved-warrant edges / total inference edges (an edge with a
real `:warrant` that is not `:missing-warrant`). On the 9 finals: **6/28 = 21.4%**,
per-graph spread **0/3 … 3/5** — it moves and discriminates. `expository-coverage`
0.32→0.59% and `prior-vs-posterior` 12.70% also moved. (Independently re-run by
claude-1: 21.43% grounding, 9/9 checker, 9/9 substance.)

**Deferred / follow-up:** H6 side-by-side render still points at the old dp-demo 5 IDs —
needs repointing to `loop-run-70b` + these 10 IDs (CPU, no GPU needed). Full 2058-region
expository coverage and the 70B-raw control remain separate jobs.
