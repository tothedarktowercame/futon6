# E-informal-proof-checking — final checklist

*Distilled 2026-06-18 (claude-1) from a full sweep of the readiness card, the excursion + specs
(`cas-cert-spec`, `rung-3-spec`), every handoff/breakdown, and the run/infra docs. One place for the
caveats, gaps, deferred refinements, and the questions to answer **when GPUs are next available**.*

**Status going in:** the **deterministic build of the checker spine is complete** — all four CAS-CERT
grains report and are tested (suite 843): SFC→symbol, R2d→concept, R2a/b/c→proof, CAS-SEL-3 topology +
CAS-SEL-2/4 registry+select-dispatch, rung-3-2 technique fill + rung-3-3 residue questions, all wired
into CAS-CERT. What's left is (1) the LLM/GPU-run measurements, (2) deferred refinements, (3) honesty
caveats to **preserve** (not fix), (4) verification gaps to close on the run.

Legend:  ☐ open · ✅ done this session · 🔒 honesty boundary (do NOT "fix") · ⏳ run-time/GPU-gated

---

## 0. Already done this session (so the checklist below is the *remainder*)
- ✅ seam-6 IATC-graph→proof-steps segmenter (`cas_segment.py`, 659e905; review PASS)
- ✅ SFC2b→CAS-CERT symbol wiring (`cas_cert --symbols`, 211fcf2; review PASS)
- ✅ symbol-extraction sharpening — junk tokens (`and`/`share`) gone, real symbols (944fb68; review PASS)
- ✅ rung-3-2 deterministic technique detector + `cas_cert --rung3` (e380e7e; review PASS)
- ✅ rung-3-3 LLM-on-residue → ArSE questions, `cas_cert --questions` report-only (a97e790; self-reviewed)
- ✅ CAS-SEL-4 v0 select-dispatch — verified already built in `cas_checks.py` (`run_selected_checks`)

---

## 1. GPU-run questions — answer these when we have GPUs (the headline)

**Preregistration to score against** (predictions — confirm/refute, don't just run):
- ☐ **C1** CAS-CERT runs deterministically over all arXiv candidates; gate FAIL **iff** a mis-wire.
- ☐ **C2 / L2** proof-grain: re-IATC with **full enrichment** *raises* warrant-resolution above the run-#1 baseline (warrant ≈ 6/28; ~4/9 orphans).
- ☐ **C3** concept-grain coverage ≈ 0.5–1.0 (R2d baseline mean 0.867; bounded by extraction noise).
- ☐ **C4** APM technique grain: CAS-SEL reproduces the hand-classification; residue ≈ **27%** (rung-3-1) holds on any *new* APM proofs.
- ☐ **C5** the size-vs-enrichment confound: does full-enrichment IATC recover *more* IATC marks than the raw-source run? (**RAW-CTL** is the control arm — send-gated, ~15min up + ~1hr.)
- ☐ **C6 / L4** honesty holds on fresh graphs: no cert claims "verified"; every FAIL traces to a real mis-wire; every empty port is a genuine open question — **spot-check ≥3** (closes the PARTIAL P3 faithfulness check).
- ☐ **L1** end-to-end 10→30 papers traverse producer→IATC→gate→checker→CAS-CERT with **no stage erroring**; certs emit `verdict` + `confidence{level,limiting_factors}`.
- ☐ **L3** batching: with `--concurrency M`, GPU util stays high, **no KV OOM**; report wall-clock/paper. **What is the KV-cache-safe M?** (the one genuinely-unknown Linode number.)
- ☐ **L5** checker determinism: re-running the checker over the same graphs is **byte-identical** (the 70B isn't deterministic; the checker must be).

**LLM-coupled measurements that need the served model** (these make the *placeholder* grains real):
- ⏳ **Symbol** — run the `openai` grounding pass → real per-paper bindings (today's stub numbers are noise); re-score the symbol rate. Confirm the verbatim-evidence `check()` still gates under the real model.
- ⏳ **Technique** — run cas_select's **Tier-1 verify** (`openai`) on arXiv → fills technique ports (deterministic path is all-thin/empty on arXiv); then rung-3-3's `openai` **novel-vs-gap** split + the actual ArSE questions.
- ⏳ **LLM-share** — measure the deterministic residue on **real arXiv moves** (the rung-3 value gate; CAS-0 gave 27% on the verified APM set / 31.8% strict).
- ☐ **Confidence may rise** — it was capped at *medium* while symbol/technique were N/A; now they're wired, re-check the confidence level after the openai passes (still bounded by no-gold).

---

## 2. Deferred refinements (real future builds — not blockers for the run)
- ☐ **CAS-SEL `.flexiarg`-pattern dispatch** (Q1's fuller select; v0 registry-predicate dispatch is built).
- ☐ **CAS-SEL-5 genealogical select** — a proof inherits its citations'/imports' patterns; **deferred on WARP-ORCH being live**.
- ☐ **CAS-SEL-3b embedding retrieval** — lift the hotword recall ceiling (recall@4 = 16/22; 3 zero-overlap steps are unreachable by hotword alone).
- ☐ **R2a-v2 de-noising** — light LaTeX normalization + N/A for nodes with <K scoreable terms (R2a is a conservative lexical *lower bound*; over-flags macro-dense nodes).
- ☐ **seam-6 Tier-1 prose cleanup** — bounded IATC-graph→clean-prose call (the deterministic resolve-node-text version is the seam-closer; this is the quality follow-on).
- ☐ **WARP-ORCH wiring** — `warp_concept_usage` / `def-snippets` / `concept-encyclopedia` / `mark3_thread_tapestry` are live on disk but in no runner; the full rebuild **timed out on codex's 30-min limit** → re-approach (bg-supervised vs lighter audit). *Blocks CAS-SEL-5.*
- ☐ **Superpod parallel runner** — claude-2 adds `--eprints/--out` to `warp_def_snippets` + `warp_concept_usage`; claude-1 builds the scheduler + the **`def_snippets` stable-sort merge** (the trap: naive merge silently breaks the drift-hash equivalence). Incremental-vs-full substrate is a follow-on (v1 = full per batch).
- ☐ **rung-3-3 → ArSE typed-bells** — the artifact carries the `:query`/`:ref` shape; *actually opening* typed-bells is downstream (FUTON3C_TYPED_BELLS is off).
- ☐ **Pattern-seeding loop closure** — an answered gap-question mints a new *verifiable* pattern (typed heuristic-vs-verifiable).
- ☐ **SFC-AGG hardening** — genus is hand-recognized for the fixture; encyclopedia seed is noisy; framing is keyword-based; Iff-bridges are recorded-not-proved.
- ☐ **RENDER generalization** — extend the inline renderer across all papers (mechanical); Haskell rewrite optional, not a blocker.

---

## 3. Honesty caveats to PRESERVE 🔒 (load-bearing — do not "fix" or let anyone misread)
- 🔒 **The certificate asserts "well-formed wiring — every port filled or flagged," NOT "this proof is correct."** No gold exists (the parent question). This is the spine's reason-to-exist.
- 🔒 **N/A ≠ FAIL.** Gate FAILs **only** on a mis-wire. Symbol + technique grains are **report-only** — nothing maps to `miswired`, neither ever changes the gate verdict.
- 🔒 A **PASS with proof_rate 0.000** (all ports empty, zero miswired — e.g. `0708.1921`) is *correct*, not a bug. It's the load-bearing "well-wired-or-flagged ≠ verified" distinction.
- 🔒 **Today's deterministic technique/symbol numbers are placeholders.** arXiv technique is all-empty/all-thin until the Tier-1 LLM verify; stub symbols are noise. A 0.0 technique rate means **"LLM verify not run,"** not "no grounded techniques."
- 🔒 **"defined" (R2d/SFC1) = evidence exists, not a usable structured definition.**
- 🔒 **R2c warrant-floor stays report-only (0.0).** ~half the 70B graphs have 0 resolved warrants — a generation *style*, not a quality fail; missing-warrant *is* the residual-sorry signal. Hard-failing would conflate style with faithfulness.
- 🔒 **Heuristic vs verifiable:** a cascade chains heuristics but must bottom out in verifiable leaves; "thin" = a load-bearing step bottoming out at a heuristic where a verifiable step is required.
- 🔒 **Don't fix the LLM fraction a priori — measure it.** The deterministic residue on real moves *is* the LLM's share.

---

## 4. Verification gaps to close on the run
- ☐ **P3 faithfulness was hand-spot-checked only → PARTIAL** (the 70B reads the right region but anchors imprecisely). Close via the C6/L4 spot-check ≥3 on fresh graphs.
- ☐ **4/9 loop-run-70b finals have orphan/dangling nodes** (`0708.1921`, `0708.2067`, `0712.0724`) — declared-but-unwired nodes. Re-IATC with enrichment should reduce these (C2).
- ☐ **No oracle-backed verifier for arXiv moves** → strict rung-3 residue = 100% on arXiv (candidate *reach* ≠ correctness; the real 27% LLM-residue is APM-only). Decide whether to build an arXiv verifier or keep APM as the calibration anchor.
- ☐ **Test-suite live-substrate coupling** — `test_r2d_concept_coverage` + `test_sfc_concept_aggregate` read **live `data/warp`** (against the "fixtures, not live data" rule) → they drift on every substrate rebuild (already re-pinned once). **Freeze a small fixture substrate** for them.
- ☐ **rung-3-3 was self-reviewed** (Codex out of quota until Jul 18; no independent reviewer) — get a second pair of eyes when Codex returns or via another agent.
- ☐ **Confirm exact producer flags** — `mark3_extract_candidates.py` / `mark3_iatc_loop.py` / `cas_checks.py` invocations against the scripts (some are from run-#1 notes, not re-verified).

---

## 5. Infra & cost (all GPU spend send-gated by Joe)
- ☐ **Nothing auto-provisions.** Linode via `linode-4gpu-setup.sh`; decommission when GPU work is done. GPU spend = Joe's call.
- ☐ **Linode shape:** 4×RTX 4000 Ada, 70B-AWQ-INT4 **TP=4 ~18.8 GB/card, util 0.95 → one server**, no device-sharding; only parallelism lever = vLLM continuous-batching (the `--concurrency M` from L3).
- ☐ **Superpod shape:** device-sharding via Rob's `superpod-shard.py`; **`def_snippets` stable-sort** is the load-bearing merge-correctness property; copies-per-GPU is a *measured, tunable* knob (LLM inference is bandwidth-bound — not a guaranteed 7×).
- ☐ **Cost:** arXiv on a hosted API ≈ **$13–16K** (big) / **$2K** (small); **local 70B = GPU-time** and is the default (`OPENAI_BASE_URL`). Pin **temp=0 + fixed seed** so the model is a stable instrument (a hosted API can silently change versions).
