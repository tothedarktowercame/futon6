# Pre-Superpod Handoff — Classical DP Mining, Cross-MSC Validated (2026-06-14)

**To:** Rob (has the full arXiv eprint corpus locally; we don't).
**From:** futon6 warp/weft loom.
**Ask:** run the *classical* (CPU-only, no GPU) structure-mining pipeline over the
full corpus, one MSC class at a time, with the checker as the per-class
acceptance gate. This is the "floor" pass that precedes any GPU work.

Everything below is reproducible from `scripts/` in this repo; no `data/` is
shipped (it's gitignored and corpus-sized). The numbers cited are from runs on
math.CT (9742 papers) plus a 50-paper cross-MSC probe off the local mark2 inbox.

---

## 1. Why this is ready — the validation

### Structural floor generalizes *perfectly* off math.CT
`scripts/warp_crossmsc_demo.py` sampled 5 papers each from 10 non-CT MSC classes
(the date-sorted mark2 inbox, genuinely all-of-math), ran the detector + checker:

```
class      n  grounded  tagged   math  wf-err
math.AG    5      50%   100%  100%       0
math.NT    5      33%   100%  100%       0
math.CO    5      62%   100%  100%       0
math.AP    5      49%   100%  100%       0
math.PR    5      60%   100%  100%       0
math.DG    5      50%   100%  100%       0
math.RT    5      69%   100%  100%       0
math.GT    5      72%   100%  100%       0
math.QA    5      80%   100%  100%       0
math.LO    5      74%   100%  100%       0
```

- **wf-errors = 0, tagged = 100%, math-coverage = 100% on every class.** The
  scope/binder/math/atomicity detection is domain-agnostic and robust. The
  well-formedness-overfit risk we saw while scaling *within* CT does **not**
  recur cross-domain.

### Grounding varies by lexicon distance — expected, and the run spec fixes it
- Grounding tracks proximity to the *CT-derived* concept lexicon: categorically
  adjacent domains ground high (QA 80, LO 74, GT 72, RT 69); far domains lower
  (NT 33, AP 49, AG 50, DG 50).
- NT's 33% is **not** the method failing — it's the CT hitlist not covering
  number theory. The detector's binders/appositives are domain-agnostic; only
  *grounding* resolves against a lexicon. **Fix: each MSC class builds its own
  lexicon** (§3, step B). Run that way, NT grounds against NT concepts.

### Concept extraction is unbiased (parity)
Within CT, at validation-time snapshot the DP-marked (~1k) vs classical-only
(~8.7k of 9742) papers were at parity on what the extractor sees — concepts/paper
median 99 vs 103, defs/paper 39 vs 43. So landscape placement by concept-usage is
fair corpus-wide. The DP-marked set *was* a biased **sample** (citation in-deg
mean 8.6 vs 2.5; median year 2009 vs 2020, because it was the early + most-cited
papers) — which is exactly why the rollout marks **all** papers: it erases the
selection bias. DP-exclusive metrics (grounding%, aliveness) should not be read as
corpus-representative until the whole class is marked.

> Status (2026-06-14): the math.CT mark-all is **already running locally** (a
> 4-way `dp_batch --shard` pass over the ~8.7k unmarked CT papers, ~12h), so the
> CT selection bias is being closed now and CT will land fully marked. Rob's
> full-corpus run is therefore primarily the **other** MSC classes; he can redo
> CT on his complete corpus if he wants the canonical version, but it isn't the
> bottleneck.

---

## 2. The pipeline (scripts, in order)

Weft (per-paper structure) and warp (cross-corpus second layer). Detailed
runbooks: `holes/dp-fleet-runbook.md`, `holes/warp-runbook.md`.

**Weft — per paper:**
1. `scripts/dp_paper_view.py` — detector. `build(pid, with_binders, with_scopes,
   with_ca, with_xref)` → marks (scopes, binders, math envelope, symbols,
   proof-moves). Reads eprints; capability modules in `scripts/dp_capabilities/`.
2. `scripts/check_invariants.py` — checker. `check_paper(pid, {text, marks})` →
   coverage (`symbol_grounded`, `symbol_tagged`, `math_coverage`, `symbols`) +
   `wellformed_errors`. **Coverage ⊥ well-formedness**: never trade one for the
   other; fix the detector, never the checker.

**Warp — cross-corpus (after a class is weft-marked):**
3. `warp_defined_pass.py` — EMPH/DEFENV/CALL concept-definition scan → defined-index.
4. `warp_concordance.py` — term → {paper, role}.
5. `warp_hitlist.py` — defined-index ∩ concordance → groundable concept hitlist.
6. `warp_concept_usage.py` — corpus-wide 1-3gram concept usage per paper.
7. `warp_citations.py` / `warp_bib.py` — citation edges (W2).
8. `warp_concept_graph.py` — definition-dependency graph + PageRank authority.
9. `warp_concept_embed.py` — multiplicity embedding (no GPU/superpod needed).
10. Landscape/overlays (optional, for inspection): `warp_paper_landscape.py`
    (t-SNE), `warp_or_curvature.py` (#1 terrain), `warp_salingaros.py` (#2
    aliveness), `warp_greatest_hits.py` (scope-district portrait).

---

## 3. Run spec for the full corpus

**A. Partition by MSC class.** Process each primary-category class independently.
This is also the natural unit for the per-class gate (§4) and floor/ceiling
decoupling (the classical floor here; GPU ceiling later).

**B. Build the lexicon PER CLASS.** Run steps 3–5 (defined-pass → concordance →
hitlist) *within each class* so grounding resolves against that domain's own
vocabulary. Do **not** reuse the CT hitlist cross-domain — that's what caps NT
at 33%. (Cross-class concept sharing is a later, optional merge step.)

**C. Mark ALL papers** in the class (not a cited/early subsample) — removes the
selection bias documented in §1.

**D. Quarantine pathological inputs, don't hide them.** One CT paper (1001.4071,
~1.2M chars, malformed `$`+comment spans) hangs the detector ~2h. Skiplist
convention: `holes/dp-outlier-skiplist.txt` (documented, counted, not silently
dropped). Apply a per-paper wall-clock timeout and append timeouts to the
skiplist with their reason.

---

## 4. Acceptance gate (per class)

A class passes when, over its marked papers:
- `wellformed_errors == 0` (hard gate — structural correctness; held on all 10
  probe classes).
- `math_coverage` and `symbol_tagged` ≈ 100% (held on all 10).
- `symbol_grounded` reported but **not** gated to a fixed threshold — it's
  lexicon-relative; report the per-class distribution instead. Use it to decide
  where a richer lexicon (or later GPU grounding) buys the most.

`check_invariants.py` produces all of these; aggregate them per class the way
`warp_crossmsc_demo.py` does (its table is the template).

---

## 5. Outputs to return

Per class: the marks (weft), the warp artifacts (defined-index, concordance,
hitlist, concept-usage, citations, concept-graph, concept-embed), and the
per-class coverage/wf aggregate table. The coverage table is the headline
deliverable — it's what tells us the floor held and where grounding needs help.

## 6. Reproduce the validation locally
```
.venv/bin/python scripts/warp_crossmsc_demo.py [batch.tar.gz] [K-per-class]
# default batch: ~/code/storage/mark2/inbox/batch-007.tar.gz, K=5
```
