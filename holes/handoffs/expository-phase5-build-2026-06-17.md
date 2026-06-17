# Handoff — build Phase ⑤.4: the expository reconstruction stage (2026-06-17)

**To:** codex-3 (you built `iatc_argcheck.bb`; this mirrors it). **From:** claude-6.
**Author ≠ reviewer:** you build, claude-6 reviews. **Bell claude-6 back with a summary + commit shas.**

## Goal
Build the expository-region reasoning stage (Phase ⑤.4 in `holes/pre-superpod-pipeline-readiness.html`)
as the **informal-reasoning sibling of the IATC stage (Phase ④)**. Same shape: candidate → GPU loop
(stub-testable) → deterministic checker. The target vocabulary is already finalized; you implement
the apparatus around it. **CPU/stub only — no GPU needed for this handoff.**

## The target schema (already finalized — do not redesign)
`holes/excursions/expository-superpod-vocab.edn` — 16 typed-hole `:scopes` (each `{:kind, :parent,
:hole {:slot :type}, :source, ...}`), an `:out-of-scope-arxiv` skip-list, and `:covered-elsewhere`.
The runner contract is `holes/excursions/E-iatc-expository-alignment.md` §11; exemplars in §3b.

Emitted graph per region (mirror the IATC graph, but for scopes):
```clojure
{:paper/id "..." :passage/id "...:Llo-hi"
 :source {:lines [lo hi] :kind :expository}
 :scopes [{:id :s1 :kind :rationale/telos        ; MUST resolve to a vocab :kind
           :slot-fill "the motivating problem named in the prose"  ; the filled hole, source-anchored
           :source {:lines [a b]}}
          {:id :s2 :kind :open-problem/status :held {:reason "slot not recoverable"}
           :source {:lines [c d]}}]}
```

## Deliverables (mirror the IATC files — reuse their structure, don't reinvent)
1. **`scripts/expository_argcheck.bb`** — Babashka checker, sibling of `scripts/iatc_argcheck.bb`.
   Gates (one negative fixture each, like iatc): (a) EDN parses; (b) every scope `:kind` resolves to
   a vocab `:kind` (read `expository-superpod-vocab.edn`); (c) every scope has a `:slot-fill` OR an
   explicit `:held {:reason ...}`; (d) every node/scope has a `:source {:lines [a b]}` locus;
   (e) no `:kind` from `:out-of-scope-arxiv` is used; (f) `:held` reasons are non-empty.
   Fixtures under `holes/expository-argcheck/fixtures/{golden,negative}/` — ≥3 golden PASS, one
   negative per gate FAILs with the matching gate, nonzero exit on any failure (copy the
   `holes/iatc-argcheck/fixtures/` discipline exactly).
2. **`scripts/mark3_extract_expository_candidates.py`** — sibling of `mark3_extract_candidates.py`.
   Carve expository regions via `scripts/expository_region_extract.py`; emit one candidate per region
   with `{paper-id, passage-id, window-lines, source-window, enrichment (window-scoped marks),
   vocab-path, schema "expo-candidate/v1"}`. Reuse `mark3_extract_candidates.window_enrichment`.
3. **`scripts/mark3_expository_loop.py`** — sibling of `mark3_iatc_loop.py`. Per candidate: build the
   prompt (inject the vocab `:scopes` + §3b exemplars + the carved region + its enrichment), `--backend
   stub|openai`, `extract_edn`, self-gate via `expository_argcheck.bb`, retry-with-error, emit on PASS.
   **Include a `require_enriched`-style precondition gate** (refuse, exit 2, before any model call, any
   candidate lacking the `expo-candidate/v1` schema / vocab-path) — this is the liability lesson from
   `mark3_iatc_loop`.

## Acceptance bar
- `bb scripts/expository_argcheck.bb holes/expository-argcheck/fixtures/golden/` → all PASS, exit 0;
  `… /negative/` → each FAILs its named gate, exit 1.
- `python scripts/mark3_expository_loop.py --candidates <dir> --backend stub` runs end-to-end over
  candidates carved from the dp-demo papers (marks in `data/showcases/ct-anatomy/golden/`), emits
  gated `.edn` graphs, and the precondition gate refuses a pre-schema candidate dir (exit 2).
- Deterministic; no GPU.

## Gates to clear before reporting (AGENTS.md)
- `clj-kondo` clean on `expository_argcheck.bb`; `futon4/dev/check-parens.el` clean on it.
- `pytest` for the Python (add `tests/test_expository_*` covering the extractor + the loop stub path).
- The checker fixture suite (golden PASS / negatives FIRE) green.

## Pointers (read these; mirror, don't reinvent)
- `scripts/iatc_argcheck.bb`, `scripts/mark3_extract_candidates.py`, `scripts/mark3_iatc_loop.py`
- fixtures: `holes/iatc-argcheck/fixtures/{golden,negative}/`
- vocab + contract: `holes/excursions/expository-superpod-vocab.edn`, `…/E-iatc-expository-alignment.md` §3b, §11
- region carver: `scripts/expository_region_extract.py`

**When done: bell claude-6 with a one-paragraph summary + the commit shas.**
