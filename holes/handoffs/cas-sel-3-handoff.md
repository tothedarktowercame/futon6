# CAS-SEL-3 — the topology extractor (Tier-0 retrieve + Tier-1 verify + assemble)

*Codex handoff. Owner/reviewer: claude-1 (author ≠ reviewer). Implements the SELECT path of
`../excursions/cas-sel-1-spec.md` §3, grounded on the 4-proof corpus
`../excursions/cas0-worked-{a93J05,a96J01,b97J01,a96J04}.md`. Part of
[[E-informal-proof-checking]] / the CAS-SEL breakdown.*

## Goal

Build `scripts/cas_select.py` — given a proof **already segmented into steps**, compute its
**topology** (the sequence of matched reasoning patterns) and from that **assemble its wiring
+ sorry list deterministically**. This is the cheap, common-case path; it must NOT do pattern
*induction* (Tier 2) — when no pattern matches a step it emits a `:none` trigger and a `thin`
sorry, and moves on.

Three tiers, per the spec — keep them as separate, separately-testable functions:

- **Tier 0 (deterministic, no model):** candidate retrieval by hotword; wiring + sorry assembly.
- **Tier 1 (bounded LLM, pluggable):** per-step match-verify. Stub + real, like
  `scripts/sfc_symbol_grounding.py`'s `call_stub`/`call_openai` split — reuse that shape.
- **Tier 2 trigger only:** on `:none`, append to an induce queue (a list in the output); do
  **not** mint anything.

## Inputs / data (all on disk)

- Pattern pool: `~/code/futon3/library/math-informal/*.flexiarg` (39 patterns). Each carries
  `@title`, `! conclusion:` (the `THEN`/conclusion text), and a `+ HOWEVER:` block.
- Hotword index: `~/code/futon3/resources/sigils/patterns-index.tsv` — columns
  `pattern \t tokipona \t truth \t rationale \t hotwords` (comma-separated hotwords). The 39
  math-informal rows are the retrieval corpus. (Same hotword principle as the P0 spotter
  `tag-patterns.bb`; a light Python overlap scorer over this TSV is fine — cite the P0 lineage.)
- Step fixtures (build these from the worked-example docs, schema below): one JSON per proof in
  `tests/fixtures/cas-select/<id>.steps.json`.

## Contract

```python
# step: {"id": "s3", "text": "...the prose of one reasoning step..."}
# returns, per proof:
{ "paper_id": "a93J05",
  "topology": ["construct-auxiliary-object","reduce-to-known-result", ...],   # ordered, matched only
  "matches": [ {"step":"s1","pattern":"construct-auxiliary-object","slot":null,"score":0.7,
                "tier1":"verified"}, ... ],
  "wiring":  [ {"step":"s1","conclusion":"<THEN text of matched pattern>"}, ... ],  # the DAG spine
  "sorry":   [ {"step":"s3","pattern":"quotient-by-irrelevance","obligation":"<HOWEVER text>",
                "kind":"declared"}, ... ],
  "induce_queue": [ {"step":"s4","candidates":[...],"reason":"no candidate verified"} ],  # Tier-2 triggers
  "checks":  [ {"step":"s2","pattern":"reduce-to-known-result","fires":["R2c-warrant"]} ] }  # §3.6 menu
```

### Tier 0 — retrieve  `retrieve(step_text, k=4) -> [(pattern, score)]`
Hotword overlap between `step_text` (lowercased, tokenized) and each pattern's `hotwords`
(plus `@title` words). Return top-k. Deterministic; no model.

### Tier 1 — verify  `verify(step, candidates) -> {pattern|None, slot}`
One prompt per step listing the k candidates: *"Which one (if any) does this step instantiate?
Return the pattern name + its slot fill (the cited theorem for reduce-to-known-result; the
decomposition; the ε-budget) or NONE."* Two backends, env-selected like SFC2b:
- `call_stub`: deterministic, reads `tests/fixtures/cas-select/<id>.oracle.json` (the
  ground-truth matches) — used by tests, no network.
- `call_openai`: the real LLaMA-70B via the vLLM `OPENAI_BASE_URL` (same endpoint the IATC
  loop uses). Small prompt, JSON out.
A candidate below a confidence/score floor, or an LLM `NONE`, ⇒ no match ⇒ Tier-2 trigger.

### Tier 0 — assemble  `assemble(matches) -> {wiring, sorry, checks}`
Pure function over the matched patterns + their parsed `.flexiarg` fields:
- **wiring**: for each matched step in order, emit `{step, conclusion=<THEN text>}`.
- **sorry**: for each matched pattern, emit `{step, pattern, obligation=<HOWEVER text>,
  kind:"declared"}`; for each `induce_queue` step, emit `{step, kind:"thin"}` (claude-loop's
  rung-3 typology: declared = matched pattern's HOWEVER; thin = unmatched).
- **checks** (the CAS-SEL menu, §3.6): a small static map matched-pattern → which rung-2 check
  its obligation implies, e.g. `reduce-to-known-result→["R2c-warrant"]`,
  `separate-into-independent-pieces→["R2b-disjointness"]`, `local-to-global→["R2b-closure"]`,
  `count-over-a-decomposition→["decomposition-exhaustive"]`, `epsilon-of-room→["forall-eps-structure"]`,
  `construct-auxiliary-object→[]`, `unfold-the-definition→[]`. (Stub map; CAS-SEL-2 makes it executable.)

## Ground truth (acceptance) — reproduce the hand-derived decompositions

Build `<id>.steps.json` from the worked-example prose, and `<id>.oracle.json` from these tables
(the stub backend returns these; the test asserts the assembled output matches).

**a93J05** (5 steps): `construct-auxiliary-object` · `reduce-to-known-result`(slot: EVT) ·
`quotient-by-irrelevance` · `local-to-global` · `reduce-to-known-result`(slot: Liouville).
Sorries from HOWEVERs of: construct-aux (P compact), quotient-by-irrelevance (lattice tiles ℂ),
local-to-global (f(z)=f(z₀)).

**a96J01** (5 roles): `construct-an-explicit-witness` · `construct-auxiliary-object` ·
`separate-into-independent-pieces` · `reduce-to-known-result`(slot: harmonic) ·
`estimate-by-bounding`. Sorries: construct-aux (telescoping), separate-into-independent-pieces
(disjointness), construct-an-explicit-witness (tent exists).

**b97J01** (7 roles): `construct-an-explicit-witness` · `split-into-cases` ·
`reduce-to-known-result`(class equation) · `count-over-a-decomposition` ·
`estimate-by-bounding` · `construct-auxiliary-object` · `induction-and-well-ordering`.
Sorries: count-over-a-decomposition (non-central classes ≡0 mod p),
induction-and-well-ordering (G/Zᵢ smaller p-group), reduce-to-known-result (class equation derivation).

**a96J04** (6 steps): `unfold-the-definition`(AC) · `unfold-the-definition`(monotone) ·
`unfold-the-definition`(null-set) · `estimate-by-bounding` · `epsilon-of-room`.
Sorries: epsilon-of-room (δ fits cover), estimate-by-bounding (countable subadditivity),
unfold-the-definition (which characterisation of AC).

## Acceptance bar

1. **Happy path (stub backend = full 39-pattern pool):** for all 4 proofs, the assembled
   `topology` + `sorry` (pattern + obligation per step) **equal** the ground-truth tables.
   Deterministic.
2. **Trigger path (stub backend simulating the pre-mint 36-pattern pool — i.e. remove
   `separate-into-independent-pieces`, `count-over-a-decomposition`, `epsilon-of-room`):**
   `induce_queue` must contain **exactly** the disjoint-support step (a96J01), the
   class-equation-divisibility step (b97J01), and the ε-arbitrary step (a96J04) — and nothing
   else. This proves the Tier-1-NONE trigger is the right signal (it reproduces the 3 mints).
3. **Real backend (`call_openai`, LLaMA-70B):** runnable end-to-end on the 4 proofs; **report
   the Tier-1 match rate vs ground truth** (this is the Tier-1-reliability / LLM-fraction
   measurement that feeds rung-3-1 — a number, not a pass/fail). Set the confidence floor from
   the observed spread, don't hard-code.
4. **Tier 0 is model-free** (retrieve + assemble import no LLM); only `verify`'s `call_openai`
   touches the network.

## Gates
PY: `python3 -m py_compile scripts/cas_select.py` + `pytest -q tests/test_cas_select.py`
(cover happy-path + trigger-path + assemble-from-flexiarg). No network in tests (stub backend).
Then: **bell claude-1 back with a summary + commit shas; append findings to this doc.**

## Out of scope (explicit)
- Tier-2 induce (a separate deliverable — this only *enqueues*).
- Proof segmentation from raw prose (Tier-1; here steps are given via fixtures). Note it as the
  one remaining Tier-1 call to wire later.
- Making `checks` executable (that's CAS-SEL-2).

## Findings — CAS-SEL-3 (codex-?)
*(append here.)*

## Findings — CAS-SEL-3 (codex-1)

Implemented `scripts/cas_select.py` with the three requested tiers kept as separate,
testable functions:

- Tier 0: `retrieve` is deterministic hotword/title/keyword overlap over
  `patterns-index.tsv` plus the math-informal `.flexiarg` `@keywords`; `assemble`
  reads `+ THEN:` and `+ HOWEVER:` directly from `.flexiarg` files.
- Tier 1: `verify` supports `backend=stub` via per-proof oracle fixtures and
  `backend=openai` via the same OpenAI-compatible `OPENAI_BASE_URL`/`OPENAI_API_KEY`
  shape used by the IATC loop.
- Tier 2: unmatched steps only append `induce_queue` rows and produce `thin`
  sorries; no pattern minting is done.

Fixtures committed under `tests/fixtures/cas-select/`:

- `a93J05`: 5 steps.
- `a96J01`: 5 steps.
- `b97J01`: 7 steps.
- `a96J04`: 5 matched reasoning roles. The worked doc prose has 6 proof lines,
  but the handoff ground-truth topology has 5 matched roles; the image-cover line
  is folded into the estimate-by-bounding role.

Acceptance numbers:

- Full 39-pattern stub path: `22/22` oracle matches, rate `1.000`.
- Tier-0 retrieval alone contains the oracle pattern in top-4 for all `22/22`
  fixture steps.
- Pre-mint 36-pattern trigger path enqueues exactly:
  `a96J01/s3` (`separate-into-independent-pieces`),
  `b97J01/s4` (`count-over-a-decomposition`),
  `a96J04/s5` (`epsilon-of-room`).
- Flexiarg assembly test verifies conclusions come from `+ THEN:` and obligations
  from `+ HOWEVER:`.

Real backend:

- `call_openai` is wired and runnable through the CLI, but this environment had no
  local vLLM server available: default `http://localhost:8000/v1` returned
  connection refused. I did not fabricate a Tier-1 LLM match-rate or floor. The
  command to rerun when the endpoint is up is:
  `python3 scripts/cas_select.py --backend openai --model <served-model>`.

Gates passed:

- `python3 -m py_compile scripts/cas_select.py`
- `pytest -q tests/test_cas_select.py` (`5` passed)
