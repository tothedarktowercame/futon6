# Mark7 (rh-grh-top100) — read-only EDA: improving WEFT/per-paper rounds at constant model-call cost

Author: zai-4 (read-only analysis; no code or run artifacts modified)
Run: `/home/joe/mark7-audit.BB0iVE/run` — `mark7rhgrh100-20260928`, corpus `rh-grh-top100`, 92 papers referenced, 7029 accepted proof graphs, checker 14058/14058 PASS, substance 7029/7029 PASS, grounding 42.99%, expository coverage 30.24% (line basis) / mean 0.24 (paper basis, metrics.jsonl).

Sections below separate **findings** (directly computed from the artifacts; command or source named) from **hypotheses** (opportunities that need one experiment to confirm).

## 1. The expository bottleneck is a flat selection cap, not model quality

**Finding.** S4 `select` accounting (`accounting/S4/S4-a001/S4.select.json`, 99458 items): 96708 deferred vs 2750 accepted (2.8%). Every top deferred reason is the same policy string — `cap 30 of N regions for this paper; not selected by exposition-first-scaled-cap/v3` — with N ranging into the thousands (2102.13459: cap 30 of 12661; math__0206203: 30 of 11122; 0910.0114: 30 of 6892). Accept rate by paper falls to 0.2–0.7% exactly for the region-rich papers.

**Finding.** Consequence in metrics: corr(log₁₀ regions-per-paper, S4 expository-coverage) = **−0.856** (n=100). Expository coverage mean 0.24, median 0.113, p90 0.789 — the distribution is a long tail of under-covered big papers and a handful of fully covered small ones. (98 of 100 papers had ≥1 accepted region; only 1 S6 assemble rejection.)

**Hypothesis (cheap, deterministic).** The cap is already the deterministic step; the *level* is the problem. A proof-count- or region-count-scaled cap (e.g. 30 → max(30, k·√regions) with the same global budget of 2750 acceptances, reallocated from papers whose 30-cap is not binding — 55+ papers accepted <30) should raise median expository coverage at *zero* extra model calls, because select is pre-model filtering. The accounting file already contains everything needed to re-run selection offline.

## 2. A large fraction of S3/S4 losses are pre-checkable or retryable — no new model spend needed

**Finding** (`accounting/S3/S3-a001/S3.loop.json`, 7362 items; S4.loop: 28 errored / 11 rejected):

- 266 S3 rejections: 39 × "no clause spans in the proof, required by mark7-v4"; the rest dominated by step-graph **2-cycles** ("contract: the steps derive node 7 -> node 8 -> node 7", "5 -> 6 -> 5", "3 -> 4 -> 3", …). Both conditions are detectable from the region text / the proposed step list *before* the expensive node/step calls.
- 67 S3 + 28 S4 errors: endpoint returned **non-JSON with "Invalid control character"** (17 + 25), **TimeoutError** (34 across nodes/steps), **max_tokens=8192 truncation** (10), HTTP 400 context overflow (3).
- Concentration: rejections cluster in a few papers — math__0409584 (75 rejected + 14 errored), 1204.6277 (23), 2206.02022 (15), 2010.01906 (12).

**Hypotheses (same budget, recovered slots).** (a) Deterministic pre-checks (clause-span presence; step-DAG acyclicity on the *proposal* before enactment) convert the 266 rejections into either early-skips or fixed inputs — each avoided rejection is a full round of calls saved that can be spent on the deferred region tail of §1. (b) A JSON-sanitizing retry shim (escape raw control characters, single deterministic retry) addresses the "Invalid control character" family — these are parse failures of otherwise-completed completions, so the completion was already paid for. (c) Timeout/truncation retries should route to smaller scoped prompts (fewer nodes per call) rather than same-prompt retries.

## 3. Learned vocabulary has collapsed: one canonical absorbs 42% of all slugs

**Finding** (`hole-vocabulary.json`: 17326 slugs → 6799 canonicals at threshold 0.72):

- The single canonical `dimension-shift-through-a-short-exact-sequence` holds **7365 of 17326 variants (42.5%)**. The next clusters are 111 (`definition-of-sigma`), 45, 40, 32… while **5566 of 6799 canonicals are singletons**.
- The collapse is lexical, not semantic: e.g. `definition-of-sigma`'s variant list is entirely `construction-of-gamma-*` / `definition-of-lambda-*` style slugs sharing template tokens; a SequenceMatcher sanity check found only ~54% of sampled (canonical, variant) pairs above 0.6 string similarity, and similarity ≠ synonymy here.
- Downstream distortion is visible in `pass3-holes.json` `recurring_gaps` (1221 entries): the top recurring wanted-term is the mega-canonical, "wanted" across dozens of unrelated papers — i.e. the recurrence signal that should drive targeted background retrieval is largely an artifact of the collapse.

**Finding** (`inference-lexicon.json`: 59031 entries, 84142 total usages): 48458 entries (82%) are single-use; top entries include noise (`text`, `the text`, `$x\in x$` with mean_conf 0.0, `theorem statement`); 9823 entries have mean_conf < 0.3 covering 15894 usages; only 17087 entries reach confidence ≥ 0.7.

**Hypotheses (pure postprocessing).** Re-cluster the slug→canonical map offline with a deterministic algorithm (exact-slug and lemma-key grouping first, cluster only residual slugs, cap cluster radius, quarantine singletons instead of forcing them under a canonical). Even a conservative re-run fixes the 42% mega-cluster, which in turn makes `recurring_gaps` usable as a *deterministic routing signal*: papers genuinely sharing a recurring gap can be batched into one shared background-context prompt instead of N duplicate lookups — a call-budget saving, not a cost.

## 4. Outcomes: what actually tracks comprehension

**Findings** (`comprehension.json`, `quote-agreement.json`, 7029 proofs):

- Verdicts: partial-comprehension 3954, **weak-extraction 2844 (40%)**, well-formed only 208 (3%), weak-proof 22, open-problem-bearing 1.
- Move grounding: 82387 moves total — thin 59333 (72%), grounded 9050 (11%), conjecture 37; ungrounded moves 13967. Noun grounding is the healthy half: named-concept grounding mean 0.943 vs proof-move grounding mean 0.49.
- Quote agreement (82028 nodes): agrees 32.7%, **unclear 46% (37700)**, uncheckable 18182, re-anchor 5292 (6.4%); sequential-only 10.4%. Proof-level agrees-share: median 0.25; **1745 proofs below 10%**.
- Correlations (proof level, n=7029 unless noted): comprehension vs strategy_rung3 **r=0.90**; vs noun-score r=0.57 (n=4579; 2450 proofs have noun=None — itself a gap); vs quote agrees-share r=0.36; comprehension vs S1 markup-coverage **r=−0.02** (paper level, n=82) — upstream markup coverage is *not* the bottleneck.
- Concentration/long tail (paper level, 83 papers with ≥1 accepted proof; caveat: old-style `math/NNNNNNNN` ids group under one "math" key — treat its 1592-proof figure as ~10+ papers): median 36 proofs/paper; top-10 papers hold ~3937/7029 proofs (56%). **1204.6277 is the worst combination**: 550 proofs, paper-mean comprehension 0.34 (bottom-5), 23 S3 rejections, hundreds of deferred regions under the cap. Small papers (≤20 proofs) have *higher* weak-extraction share (0.62) than big papers (0.39).
- S6 statement-proof attachment: median 0.705, min 0 (rows are duplicated per paper). S7 clean-discharge-rate: mean 0.244, median 0.2, two papers at 0.0 — discharge quality is uniformly low, not tail-driven.

**Hypotheses (routing/allocation, same budget).** (a) Since strategy_rung3 tracks comprehension at r=0.90 and both are computed post-hoc, the *pre-call* signals (region clause-span count, candidate count under cap, quote-agreement of the paper's earlier proofs) could route the fixed per-paper call budget: proofs from papers whose first proofs land <10% agrees-share are unlikely to improve on re-extraction, while thin-move-heavy proofs from high-agrees papers are the best marginal spend. (b) The 2450 noun=None proofs are a deterministic backfill candidate (noun resolution from the already-built encyclopedia index) before spending model calls. (c) The 37700 `unclear` quote nodes never receive proposals (proposals exist only for re-anchor verdicts); a deterministic re-anchor-style span search over the paper text for a sample would show whether "unclear" hides recoverable matches — if even 20% recover, agrees-share rises ~10 points with no model calls.

## 5. Compact opportunity list (all reuse the existing call budget)

| # | Lever | Type | Evidence |
|---|---|---|---|
| 1 | Re-run S4 `select` with a size-scaled cap, same 2750-acceptance budget | deterministic selection | §1, r=−0.856 |
| 2 | Pre-call checks: clause-span presence, step-DAG acyclicity | deterministic routing | §2, 266 rejections |
| 3 | JSON-control-char sanitize + single retry; scoped-down retry for timeout/truncation | postprocessing | §2, 42+34+10 errors |
| 4 | Re-cluster slug→canonical vocabulary; quarantine singletons; rebuild `recurring_gaps` | postprocessing | §3, 42% mega-cluster |
| 5 | Use repaired `recurring_gaps` to batch shared background context across papers | prompt-context allocation | §3 |
| 6 | Budget routing keyed on early agrees-share / thin-move density per paper | routing | §4 |
| 7 | Deterministic noun backfill for the 2450 noun=None proofs | postprocessing | §4 |
| 8 | Deterministic span-proposal trial for `unclear` quote nodes | postprocessing | §4, 46% unclear |

## Caveats

- Paper-level joins key on `pid.split('__')[0]`; old-style `math/` ids collide (the "math" paper). Counts labelled with this caveat above.
- S6 metrics rows are duplicated per paper (200 rows / 100 papers, identical pairs).
- This analysis is entirely read-only; nothing in the run or repo was modified except this report file.
