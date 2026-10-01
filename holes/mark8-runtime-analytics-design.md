# Mark8 runtime-analytics design — call/attention allocation at constant model-call budget

Author: zai-4. Status: design (read-only; nothing implemented yet). Grounded in the Mark7 rh-grh-top100 EDA (`holes/mark7-rh-grh100-zai-eda.md`) and the futon6 code paths named below.

Goal: a runtime analytics layer for Mark8 that (a) spends the *same* per-run model-call budget better, (b) points operator attention at the few places where human intervention pays, and (c) is auditable against gaming. Everything here is either pre-call (usable to route/filter before spending a call) or post-call (usable to reallocate future slots or postprocess outputs) — the distinction is load-bearing and kept explicit.

## 1. Charts and signals

All charts render from existing run artifacts; the renderer lives next to `render_run.py` (no pipeline coupling).

| Chart | Source artifacts | Type | Decision it drives |
|---|---|---|---|
| C1 Region-cap waterfall | `accounting/S4/S4-a001/S4.select.json` | per-paper accepted vs deferred-by-cap bars, sorted by regions | whether a paper's tail deserves slots (§3 A1) |
| C2 Budget-flow Sankey | stage accounting S3/S4/S5 (`stage_accounting.py` docs) | expected → accepted/rejected/errored/deferred | where recoverable losses concentrate (pre-check ROI) |
| C3 Recovery ledger | `S3.loop.json`, `S4.loop.json` reason strings | stacked bars by reason class (no-clause-spans, 2-cycle, non-JSON, timeout, truncation) | which deterministic fix to build next |
| C4 Wooliness vs outcome scatter | wooliness index (§2) vs `comprehension.json` verdict + quote-agreement agrees-share | 2-D hexbin | whether wooliness predicts extraction failure (validation, §5) |
| C5 Marginal-value curve per paper | agrees-share of proofs N-1 vs N within paper | line per paper, top-20 papers | early-stop / re-spend routing (§3 A3) |
| C6 Attention top-K board | C1∩C4 residuals | table: paper, proofs, cap-deficit, wooliness, comprehension, recoverable errors | the operator's one screen |
| C7 Vocabulary cluster health | `hole-vocabulary.json`, `inference-lexicon.json` | cluster-size log-log plot + singleton share | when the canonicalization needs the offline re-cluster |

## 2. Wooliness index (computable)

Definition: for a passage *p* (proof or expository region), wooliness W(p) ∈ [0,1] measures "hard to follow because terms are undefined locally and/or definitions live behind unresolved citations." Text-only, deterministic, pre-call.

Components (each kept separately in the record for auditability):

- **U(p) local-undefined share.** Terms (noun slugs, from the same slugger that feeds `hole-vocabulary.json`) used in *p* that are not defined in *p*, not defined earlier in the same paper (definition spans from `mark3_extract_candidates.py` output), and not in the concept encyclopedia (`concept-encyclopedia-rh-grh.json` / `background-corpus-index.json`). Reuses, does not re-implement, Mark7's noun machinery (`clean_comprehension.py`, `symbol_grounding.py`).
- **C(p) unresolved-citation share.** `\cite`/reference keys in *p* (or in the definitions *p* depends on) that do not resolve in the background corpus index. Purely a bib/manifest lookup.
- **D(p) definition distance.** For each term undefined locally, the citation-graph hop distance to the nearest *resolvable* definition: 0 if in-paper, 1 if behind a resolved citation, ∞ (clipped to 3) if behind unresolved citations only.

W(p) = clip( 0.45·U + 0.25·C + 0.30·min(D̄/3, 1) ), with weights frozen in config and reported alongside. All three components are pre-call: they depend only on the corpus text, the citation manifest, and the encyclopedia — never on model output.

Readings: W ≥ 0.6 → do not spend a full extraction call; queue for background-context backfill or route to the shared-context batch. W in [0.3, 0.6) → extract but attach the top-k missing definitions to the prompt context (prompt-context allocation, same call). W < 0.3 → normal path.

## 3. Pre-call vs post-call features

**Pre-call (route/filter before spending):**
- clause-span count in region (kills the 39× "no clause spans" rejections),
- step-proposal DAG acyclicity (kills the 2-cycle contract rejections),
- prompt token estimate vs context window (kills HTTP-400 class),
- W(p) and its components U, C, D,
- paper-level priors: regions count (cap pressure), proof count.

**Post-call (reallocate or repair after a paid call):**
- JSON control-character sanitize + single deterministic re-parse (the "Invalid control character" family — completion already paid for),
- truncation detection (finish_reason/max_tokens) → scoped-down retry eligibility flag, not auto-retry,
- comprehension verdict, strategy_rung3, noun buckets, thin/grounded move counts,
- quote-agreement verdict mix (agrees / unclear / uncheckable / re-anchor), agrees-share,
- per-paper agrees-share trajectory (C5) → stop/spend signal for remaining slots.

**Allocator (A-series, all zero-sum within the fixed budget):**
- A1 cap reallocation: `exposition-first-scaled-cap/v3` → v4 with cap(paper) = max(30, round(k·√regions)) normalized to the same global acceptance total (2750 in Mark7 terms). Offline re-run against the existing `S4.select.json` items before any live use.
- A2 pre-check gate: items failing a pre-call check skip their node/step calls; the freed slots feed A1's tail.
- A3 trajectory stop-rule: if a paper's last 3 proofs all have agrees-share < 0.1 and W ≥ 0.6, reallocate its remaining slots to papers with cap-deficit and W < 0.6.

## 4. Validation and anti-gaming

- **Determinism gate:** every index and allocator decision must be bit-reproducible from run artifacts (`replay_e2e.py` harness): same inputs → same allocations. CI test replays Mark7 and diffs.
- **No-output-as-input:** W and pre-call features are forbidden (by test — scan the feature builder's import graph) from reading anything under `artifacts/` produced by model calls. Otherwise a model could lower its own wooliness by emitting definitions, or the allocator could chase post-hoc artifacts.
- **Zero-sum assertion:** allocator reports Σ(slots in) = Σ(slots out) per stage; a mismatch fails the run gate. Prevents "creating" budget.
- **Monotonicity checks (unit):** W is non-decreasing in each component holding others fixed; adding a resolvable citation or an in-paper definition strictly lowers W.
- **Predictive validation (offline, before any live use):** on Mark7, W must correlate with weak-extraction verdicts and low agrees-share at least as well as the current best pre-call proxy (region count). Threshold: AUC ≥ 0.65 for W predicting weak-extraction; report calibration deciles. If it fails, W ships as a *reported* signal only, not an allocator input.
- **Gaming of metrics:** agrees-share recomputation is checked against a 1% manual anchor sample (`anchor-faithfulness.txt` style) so postprocessing "repairs" cannot silently inflate agreement.
- **Anti-concentration:** allocator emits a Gini over per-paper slot share; operator board flags if one paper would hold >30% of a stage's slots.

## 5. Integration points (exact)

- Select policy + reason strings: `scripts/run_manifest.py` (search `exposition-first-scaled-cap`) — A1 lands here as v4 alongside v3, selected by config.
- Stage accounting: `scripts/stage_accounting.py` (`record`, `findings`) — pre-call rejects get a distinct status (`precheck-skipped`) so C2/C3 stay honest; do not overload `rejected`.
- Metrics: `scripts/metric_harness.py` — new metrics `wooliness`, `wooliness/U|C|D`, `slot-gini`, `recovered-parse-retry`, same schema as `metrics.jsonl`.
- Loop/stepper (clause spans, step contract): `scripts/linode_stepper.py` — pre-checks A2 gate calls here.
- Vocabulary/canonicalization: `scripts/iatc_lexicon_harvest.py` + hole-vocabulary builder in `run_manifest.py` — C7 reads only; re-cluster stays a separate offline experiment (per codex-11, changes recurrence semantics).
- Post-call parsing: the nodes/steps client path shared by S3/S4 loops — sanitize-retry wrapper at the JSON decode site, one retry max, accounted as `recovered-parse-retry`.
- Charts: extend `render_run.py` (or a sibling `render_run_analytics.py`) reading only run artifacts; viewers under `viewers/`.

## 6. Smallest implementation slice (one PR, no behavior change by default)

1. `scripts/wooliness_index.py` — computes W/U/C/D per passage from corpus text + citation manifest + encyclopedia; CLI over a run dir; writes `wooliness.json`. (~150 lines; the only new "engine".)
2. Offline A1 re-run script over an existing `S4.select.json` with cap v4, emitting a *diff report only* (no pipeline change).
3. C4 scatter + C6 board in the renderer, reading `wooliness.json` + `comprehension.json` + accounting.
4. Validation: determinism replay test + predictive AUC check against Mark7 weak-extraction labels.

Everything allocator-shaped (A2 gates, stop-rules, live cap switch) is deferred until W passes the AUC gate and the A1 diff shows coverage gain at equal acceptances — then the second slice flips config flags, not code.
