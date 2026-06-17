# Pre-superpod run #2 — invocation + content (exercises the full checker spine)

*Author: claude-1, 2026-06-17. Successor to `mark4-linode-go-live-preregistration.md` (run #1,
which validated the **producer** half — anatomy→IATC). Run #2 exercises the **checker spine** the
session built: rungs 0–2 → R2d → CAS-SEL → CAS-CERT, with the residue measured. v1 draft —
revise after the seam-6 adapter + a live endpoint land. Pre-flight gate: `pipeline_witness.py`.*

## 0. Purpose (what run #2 answers)

Run #1's parent question was *"are we recovering the semantics?"* — answered *partially* (producer
wired; P3 faithfulness PARTIAL). Run #2 asks the **checker** question: **over real papers, does the
spine produce honest CAS-CERT certificates** — conformance-by-grain + a residual-sorry map — and
**what is the empirical LLM-fraction at scale?** It is NOT "are the proofs correct" (no gold; the
cert asserts well-formed-wiring, not verified — the honesty boundary).

## 1. Content — two tracks (run them as separate populations; do NOT conflate)

| track | population | what it exercises | grains live |
|---|---|---|---|
| **arXiv** | the 9 `loop-run-70b` finals (+ the run-#1 candidate pool if re-IATC'd) | producer → rungs 0–2 → R2d → CAS-CERT | **proof + concept**; technique N/A (seam-6), symbol N/A (SFC2b) |
| **APM** | the 4 CAS-0 worked proofs (a93J05, a96J01, b97J01, a96J04) + more APM step-fixtures as authored | CAS-SEL (select+registry) → CAS-CERT | **+ technique** (step fixtures exist) |

Witness for the pre-flight gate: **`0706.1286`** (clean) + one of each shape from run #1's spread
(orphan `0708.2067`, anchor-flag `0709.0248`, self-loop `0708.2185`).

**Why two tracks:** the technique grain can't run on arXiv until **seam-6** (the IATC-graph→steps
segmenter ≈ rung-3 move-extraction) lands. APM proofs already have hand-authored step fixtures, so
the technique grain is live there. When the seam-6 adapter lands, the tracks merge.

## 2. Pre-flight gate (run BEFORE provisioning — it's CPU/local)

```
python3 scripts/pipeline_witness.py --plan                 # DAG + seam contracts
python3 scripts/pipeline_witness.py --witness 0706.1286    # PASS producer; surfaces seam gaps
```
Run `--witness` on each candidate; a candidate is run-eligible if stages 1–4 are **PASS** (producer
artifacts conform). MISS at stage 5 is fine (semcheck materializes on-demand). The standing GAP is
seam-6 (expected; arXiv technique grain stays N/A this run).

## 3. Invocation — ordered, CPU/GPU marked, with in-between seam checks

**GPU is needed for exactly one stage** — the 70B IATC call (③). Everything else is CPU. (SFC2b
symbol grounding + rung-3-3 LLM-on-residue would also be GPU, but their grains are N/A this run.)

### arXiv track (per paper or batch)
```
# ① anatomy (CPU)        marks = the enrichment IATC must read
python3 scripts/mark4_pipeline_runner.py --evidence <pid>          # proves stage-1 enrichment exists
# ② candidates (CPU)
python3 scripts/mark3_extract_candidates.py <pid>                  # FIX: carry grounding/scopes/expository, not just binders
#   ↳ SEAM ②→③ check: candidate carries the enrichment (run #1's gap — verify, don't assume)
# ③ IATC (GPU, 70B-AWQ TP=4, --enforce-eager)   ← the only GPU stage
python3 scripts/mark3_iatc_loop.py --rung2-gate <candidates>       # emits graph + .rung2.edn sidecar
#   ↳ SEAM ③ conformance: graph is {:nodes :edges :holes}, typed edges w/ :warrant + :source
# ④ repair + rung-0/1 gate (CPU)
bb scripts/iatc_repair.bb <graph> ; bb scripts/iatc_argcheck.bb <graph> ; python3 scripts/substance_gate.py <graph>
# ⑤ render (CPU)
python3 scripts/build_iatc_goldens.py
# ⑥ semcheck — rungs 0–2 + R2d (CPU)
bb scripts/iatc_semcheck.bb --out <profile.edn> data/iatc-argument-graphs/<run>/
# ⑨ CAS-CERT — the certificate (CPU, deterministic, no model)
python3 scripts/cas_cert.py --graph-dir data/iatc-argument-graphs/<run>/ --out <run>.cert.json [--gate]
```

### APM track (the 4 worked proofs + more; CPU; technique grain LIVE)
```
# CAS-SEL select (Tier-0 deterministic + Tier-1 verify)
python3 scripts/cas_select.py --backend openai --model <served-model>   # or --backend stub for a dry, no-endpoint pass
#   ↳ report Tier-1 match-rate vs ground truth (the LLM-fraction; rung-3-1 measured 27.3% deterministic-residue)
# CAS-SEL-2 registry (executes the selected checks)
python3 -c "import scripts.cas_checks ..."                              # per cas_checks API
# CAS-CERT with the technique grain fed
python3 scripts/cas_cert.py --graph-dir <dir> --cas-select <cas_select.json> --out <apm>.cert.json
# residue measurement (already run; re-run if the pool changed)
python3 scripts/rung3_residue_spike.py --json-out <run>-residue.json
```

## 4. Preregistered predictions (the scorecard to score against)

State these BEFORE the run so we can't rationalize after (run #1 discipline):
- **C1 — CAS-CERT runs deterministically over all arXiv candidates**; per-paper conformance vector
  by grain emitted; aggregate gate FAIL iff any mis-wire (run #1's `loop-run-70b` already shows 6/9
  FAIL on miswires — re-IATC'd papers may differ).
- **C2 — proof-grain rates spread low–high** (run #1 baseline: warrant ≈ 6/28 aggregate; ~4/9 have
  orphans). Expect re-IATC with full enrichment (the run-#1 fix) to *raise* warrant-resolution.
- **C3 — concept-grain coverage ≈ 0.5–1.0** (R2d baseline mean 0.867; bounded by extraction noise).
- **C4 — APM technique grain**: CAS-SEL reproduces the hand-classification; residue ≈ 27% (the
  rung-3-1 number) — confirm it holds on any *new* APM proofs added.
- **C5 — the size-vs-enrichment confound** (run #1's open caveat): does full-enrichment IATC
  recover *more* IATC marks than the raw-source run? (RAW-CTL is the control arm for this — separate.)
- **C6 — honesty holds**: no certificate claims "verified"; every FAIL traces to a real mis-wire;
  every empty port is a genuine open question (spot-check ≥3, as in the CAS-CERT review).

## 5. Known gaps going in (stated, not hidden)
- **seam-6** — no arXiv→steps segmenter ⇒ arXiv technique grain N/A. (Adapter ≈ rung-3 move-
  extraction; build before merging the tracks.)
- **symbol grain** — SFC2b not wired ⇒ N/A both tracks.
- **arXiv verifier** — no oracle-backed Tier-1 verifier for arXiv ⇒ rung-3 strict residue = 100% on
  arXiv (retrieval reach reported separately, per rung-3-1). The real LLM-residue number comes from
  the APM track (27%).
- **Tier-0 recall ceiling** — hotword retrieval recall@4 = 16/22 on CAS-0 (CAS-SEL-3b embedding
  retrieval would lift it). Affects how many APM moves reach Tier-1 vs false-trigger induce.

## 6. Box + decommission (per run #1)
- Provision via `scripts/linode-4gpu-setup.sh` + the register/bootstrap StackScripts (~15 min up).
  Only ③ needs the GPU; the whole checker spine is CPU and can run locally without the box.
- **Send-gate = Joe** (the GPU spend is a Joe decision; this doc does not auto-provision).
- Decommission via `linode-cli` when ③ is done — pull the graphs back, run the CPU checker spine
  locally (it doesn't need the box).

## 7. Open / to confirm before running (revise this doc)
- Exact `mark3_extract_candidates.py` / `mark3_iatc_loop.py` flags (confirm against the scripts;
  some are from run #1's notes, not re-verified here).
- Whether to re-IATC the run-#1 candidate pool with full enrichment (tests C2/C5) or run CAS-CERT
  over the existing `loop-run-70b` finals only (cheaper, no GPU — answers C1/C3/C6 today).
- The `cas_checks` CLI/entry surface (codex-4's `cas_checks.py` — confirm invocation).
- **Cheapest first pass needs no GPU at all:** CAS-CERT over the existing `loop-run-70b` finals +
  the APM track answers C1/C3/C4/C6 immediately. The GPU run (re-IATC, C2/C5) is the second step.
