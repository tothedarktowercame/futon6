# apm-structure-match — design (mark4 final stage)

Date: 2026-06-16 · Joe + claude-1 · Status: DESIGN (pilot-scoped, pre-build)
Feeds: the next Linode 4-GPU run testing the **entire mark4 pipeline** end-to-end.

## IDENTIFY — the gap

We have keyword retrieval working (`mark4_proof_keyword_retrieval.py` →
`mark4-retrieval-top200.tsv`): APM proof keywords → arXiv papers ranked by hits.
That is *vocabulary* overlap. The mark4 hypothesis is stronger:

> **APM proofs can be turned into queries over a graph database**, and matched to
> eprints that **share meaningful scopes — not just keywords**.

A "scope" here is the typed binder/quantifier structure the anatomy engine already
extracts (see MAP). The stage tests whether a prelim *proof's* structure is
**found in the literature** — structural, not lexical, retrieval.

(Scope note: this stage processes **proofs**, not problem statements. Going
problem-statement → proof via graph query is a later phase; not now.)

## MAP — the pieces that already exist

- **Scope representation** (verified from `storage/math-processed-gpu/scopes.json`):
  per entity, a list of typed scope-hyperedges:
  ```
  {"hx/id":"…:scope-001","hx/type":"bind/integral","hx/parent":null,
   "hx/ends":[{role:entity},{role:binder,latex:"\\int"},{role:symbol,latex:"x"}],
   "hx/content":{match:"\\int_a^b f(x)…dx",position:…},"hx/labels":["scope","integral"]}
  ```
  Types seen: `quant/universal`, `bind/integral`, `assume/explicit`, … with
  `hx/parent` nesting and roled endpoints. **This is "meaningful scope."**
- **Shared extractor**: `detect_scopes(text)` (`scripts/build-golden-50.py`) is run
  over *both* arXiv (`audit-scope-binders.py:audit_arxiv_jsonl`) and SE
  (`audit_se_entities`). Running it on the APM proof `.tex` = "process the same way
  we do eprints." **Input is ready**: the clean `storage/apm/mark4-tex/apm-*.tex`
  (259 frozen proofs, code/colour-stripped) built this session.
- **Scale-safe access**: `ct_anatomy_slice.py` slices the 19 G `scopes.json`
  per-paper (the pattern that avoids loading whole multi-GB files — the OOM we hit).
- **Already-built corpus**: the keyword top-200 (`data/mark4-retrieval-top200.tsv`)
  — the ≈200-paper pool to process and the baseline to disagree with.
- **Optional embedding**: `hypergraph-embeddings.npy` + `graph-gnn-model.pt` exist
  (trained on SE-math). Usable if convenient; not required for v1.
- Prior art to reuse/extend: `apm_proof_audit.py`, `audit-scope-binders.py`.

## DERIVE — the stage

1. **Extract** scopes for both sides with the *same* `detect_scopes`:
   - ≈200 representative APM proofs (`mark4-tex/*.tex`) → APM scope-hyperedges.
   - the ≈200-paper eprint pool (batch-007/008 ∩ keyword top-200) → eprint scopes.
2. **Load eprint scopes into a real graph DB** (datalog/datascript in-process for the
   pilot — already in the stack; XTDB/futon1a if we want durable). Hyperedges →
   entities with roled refs; `hx/type`, `hx/labels`, endpoint symbol-classes indexed.
3. **Query**: each APM proof's scope-set becomes a **subhypergraph query pattern**.
   Match eprints sharing scopes by `hx/type` + endpoint symbol-class compatibility
   (the easy, interpretable matcher; embedding-NN over the GNN vectors is a drop-in
   alternative per fork 2).
4. **Rank / score** eprints per proof by count/quality of shared meaningful scopes.

## ARGUE

- **IF** we want evidence that APM proofs are structural objects retrievable from the
  literature (not just keyword bags),
- **HOWEVER** keyword retrieval can't distinguish shared *vocabulary* from shared
  *mathematical structure*,
- **THEN** extract typed scopes from proofs and papers with the same detector, load
  paper-scopes into a graph DB, and run each proof's scopes as a query pattern,
- **BECAUSE** the scope-hyperedge schema already encodes binders/quantifiers/
  assumptions as typed, queryable structure — so "shares a meaningful scope" is a
  literal graph match, and a proof-as-query is a literal subgraph query.

## VERIFY — what counts as confirmatory evidence

Two metrics, both small-sample-honest (pilot ≈200 × ≈200):

1. **Scope coverage %** (Joe's): per APM proof, the fraction of its scopes that are
   matched by ≥1 literature scope. Full coverage = the proof is structurally
   assembled from known pieces (strong confirmation; unlikely at 200 papers, so the
   **% is the signal**, and its distribution across proofs is the result).
2. **Scope-vs-keyword disagreement** (the diagnostic): papers that scope-match
   surfaces but keyword-match *misses* — shared structure without shared vocabulary —
   are the strongest evidence APM proofs work as structural queries. (combining-
   methods-as-diagnostic: the disagreement *is* the signal.)

**Falsification**: if scope-match ≈ keyword-match (same papers, same ranks) and
coverage % is ~uniformly near-zero, the structural claim adds nothing over keywords.

## INSTANTIATE — pilot plan

- **Inputs**: ≈200 APM proofs from `mark4-tex/` (pick the representative set; the 145
  Lean-free proofs are the cleanest candidates) × the ≈200 eprint pool.
- **Scale**: 200 × 200 — small; in-memory datalog is fine, no full-arXiv concern yet.
- **Graph DB**: datascript (in-process datalog, zero infra) for the pilot; revisit
  XTDB/futon1a for durability when scaling past the pilot.
- **Outputs**: per-proof matched-scope records, coverage-% distribution, and the
  scope-vs-keyword disagreement set.
- **Runs on the Linode**: this stage slots after extraction in the full mark4 pipeline
  test; GPU only needed if we use the GNN embedding matcher.

### First concrete step (before any Linode run)
Verify `detect_scopes` runs on one `mark4-tex/*.tex` proof and emits the `hx/` schema
— i.e. confirm the shared-extractor claim holds for APM proofs locally, on one file,
before scaling to 200. (Cheap, local, no GPU.)

## Open / deferred
- Exact "scope compatibility" rule (type-only vs type+symbol-class vs nesting-aware).
- Whether to also process the 114 Lean-carrying proofs (richer binders) as a 2nd arm.
- Embedding matcher (GNN) — only if the typed-overlap matcher under-discriminates.

## Update 2026-06-17 — random control arm + disagreement metric

The keyword top-200 eprint pool is selected by vocabulary overlap, so it cannot by
itself test the diagnostic "scope retrieves structure that keyword missed." The
control arm is now:

- draw `N=200` eprints uniformly from batch-007/008 metadata with fixed seed
  `20260617`;
- do not apply any keyword filter;
- extract scopes with the same `nlab-wiring.detect_scopes` eprint-source reader;
- compare the random-pool scope signal against the keyword-selected pool.

Formal disagreement metric, per APM proof `p`:

- `K_p`: keyword-retrieved eprints from `mark4-batch-keyword-hits.json`, restricted
  to `mark4-retrieval-top200.tsv`, where `p` is in `source_proofs`;
- `S_p(tau)`: eprints in the union of keyword-pool scopes and random-pool scopes
  whose individual `type_multichar` coverage of proof `p` is at least `tau`
  (`tau=0.05` in the pilot);
- `scope_not_keyword_p = S_p - K_p`;
- `keyword_not_scope_p = K_p - S_p`;
- `Jaccard_p = |S_p ∩ K_p| / |S_p ∪ K_p|`.

Null hypothesis: random-pool `type_multichar` coverage is approximately the same
as keyword-pool coverage and the mean `scope_not_keyword` rate is low. In that
case scope matching is tracking keyword selection/common structure rather than
providing an independent structural retrieval signal.

The implementation is `scripts/mark4_apm_random_scope_disagreement.py`. It writes
the random scopes and report under `storage/apm/` with the seed and sample IDs in
the provenance block.

Pilot run (`seed=20260617`, `N=200`, `tau=0.05`) over batch-007/008 sampled from
10,000 metadata records:

- random scopes: `storage/apm/mark4-random-eprint-scopes-seed20260617-n200.json`;
- report:
  `storage/apm/mark4-random-scope-disagreement-seed20260617-n200.json`;
- missing sampled eprint source: 18/200;
- keyword-pool `type_multichar`: mean `0.256994642147068`, median
  `0.13636363636363635`, tail ≥80% `13`;
- random-pool `type_multichar`: mean `0.2556957558481817`, median
  `0.13333333333333333`, tail ≥80% `13`;
- disagreement over 135 scored proofs: mean Jaccard `0.04822104360816591`,
  mean `scope_not_keyword_rate` `0.884298203214691`, mean
  `keyword_not_scope_rate` `0.6824612160935244`, and 88 proofs with at least one
  scope-not-keyword eprint.

Verdict from this first metric: **signal**, because scope retrieval surfaces a
large scope-not-keyword arm. The coverage-control part is null-like (random and
keyword-pool aggregate coverage are nearly identical), so the next validation
must inspect whether the scope-not-keyword examples are meaningful structure or
common multichar-symbol false positives.

## Update 2026-06-16 — decisions, Rob's pattern, pilot run 1

**Decisions (Joe):** local pilot; **145 Lean-free proofs**; easy-default match first
("later we'll need something much better"); spin up a real graph DB for the phase.

**Rob's pattern (proven prior art — our target architecture already exists).** Rob
built, out of necessity, exactly this shape over a *Lean* corpus: **neo4j** for
*structural* proof queries + **pgvector** for *semantic* search, refreshed in
**realtime on every Lean build**. So fork 3 (real graph DB) → **neo4j**, fork 2
(embedding) → **pgvector**. Our local pilot is the POC; the **fully scaled version is
a Rob collaboration**. (Rob's side-insight, relevant to the Lean-proof arm and the
later problem→proof phase: *defining Lean types up front* matters a lot — a mistake
was not doing so early.)

**Corpora built (local, CPU, `nlab-wiring.detect_scopes`):**
- proofs: 145 Lean-free → **1,450 scopes** (`storage/apm/apm-proof-scopes.json`);
  10 proofs yield 0 scopes. Heavy `bind/integral` (777), `bind/summation` (251).
- eprints: 200-paper keyword pool → **153,879 scopes** over 178 papers
  (`storage/apm/eprint-scopes.json`); 22 papers have no source in the batch.
  *(Reader fix: arXiv eprints are single-gzipped `.tex` OR multi-file tarballs — must
  try both; tar-only missed 78% of them.)*

**Pilot run 1 — coverage:** type-only and type+single-symbol matches **saturate**
(mean 1.00 / 0.86) — confirming the easy default is too loose (common binders +
single-letter vars). **type + multi-char symbol** is the first discriminative cut:
**mean 0.26, median 0.14**, distribution over [<20,20-40,40-60,60-80,80-100]% =
[73, 30, 12, 7, 13]. **13 proofs are ≥80% structurally covered** by the 200-paper
pool — the confirmatory tail.

**Next:**
- The "much better" matcher: **symbol-class / role-typed** overlap, or **pgvector
  embedding** of scopes (Rob's pattern) — single-letter overlap is the saturation
  artifact to kill.
- **Disagreement metric needs a broader pool**: the current eprint pool *is* the
  keyword top-200, so we can't yet test "scope finds what keyword missed." Add a
  random batch-007/008 sample as a second arm.
- Stand up **neo4j** (Rob's pattern) for the graph-DB phase; pilot used in-memory.
