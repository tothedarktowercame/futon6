# Mission: Canon fingerprint store — Billey-Tenner instantiation for symbol grounding

**Date:** 2026-05-23
**Status:** IDENTIFY → DERIVE — design proposal
**Owner:** Joe (frames it) / claude-7 (drafted)
**Predecessor:** [M-bayesian-structure-learning.md](M-bayesian-structure-learning.md)
**Source pointer:** Billey, S. C. & Tenner, B. E. (2013).
*Fingerprint Databases for Theorems.* arXiv:1304.3866
(`~/Downloads/1304.3866v1.pdf`)

## 1. Why this mission exists

Two findings from the Bayesian-grounding work converge on the same gap:

1. The Channel-2 self-improvement experiment (commit `94bd227`) showed
   that within a single batch the posteriors stabilise but don't *grow*.
   Knowledge accumulated in one Stage 5 run does not carry forward
   into the next.
2. Joe (2026-05-23) observed: "the structure of the batches which are,
   I think, temporally selected not subject-specific, we need to be
   MAP-REDUCE-ing into a knowledge store that we can reuse as we go."

We're missing the persistent layer. Each Stage 5 run produces a
`learned-newcommand-vocab.json` snapshot, but there's no schema for
the broader question: across all batches we've ever run, what
canon(s) has symbol X been bound to, by which strategies, in which
papers, with what reliability?

Billey & Tenner's framing — "fingerprint databases for theorems" —
gives the right shape:

> A fingerprint database of theorems should be a searchable,
> collaborative database of citable mathematical results indexed
> by small, language-independent and canonical data.

They use OEIS (Online Encyclopedia of Integer Sequences) as the
prototype: every integer-sequence theorem gets the sequence itself
as a unique-enough fingerprint that future mathematicians can query
against. The lesson they emphasize, in Rod Brooks's phrase: *"fast,
cheap, and out of control"* — start collecting now with an
imperfect but efficient fingerprint, rather than waiting for the
perfect mechanism.

For symbol grounding, the fingerprint *per binding* is the
`(symbol, canon, paper_id, strategy, position)` record. The
*query* shape is "show me canon distribution for `\sigma(T)` across
all batches ever run." The *useful action* is using that
distribution as a prior in Bayesian arbitration (§3.2 of
M-bayesian-structure-learning.md).

## 2. The fingerprint, schemed

Each binding the engine emits becomes a fingerprint record:

```python
@dataclass(frozen=True)
class CanonFingerprint:
    symbol: str            # e.g. "\\sigma(T)"
    canon: str | None      # e.g. "Spectrum" or None (un-canonical)
    paper_id: str          # e.g. "arxiv-2305.01234v2" or "pm:Spectrum"
    strategy: str          # e.g. "let-binding"
    confidence: str        # "high" | "medium" | "low"
    constructor: str       # "single" | "comma-list" | "relation-chain" | ...
    timestamp: str         # ISO-8601 when emitted
```

A `(symbol, canon)` pair fingerprint aggregates across records:

```python
@dataclass
class CanonAggregate:
    symbol: str
    canon: str
    n_occurrences: int                  # how many times this pair appeared
    source_paper_ids: list[str]         # which papers
    strategy_breakdown: dict[str, int]  # how many per strategy
    first_seen: str                     # ISO timestamp
    last_seen: str                      # ISO timestamp
```

The store, queried by `symbol`, returns the *canon distribution*:

```python
def canon_distribution(symbol: str) -> dict[str, CanonAggregate]:
    """For symbol X, return all canons it's been bound to and the
    aggregate evidence for each."""
```

This is the OEIS-shaped query: "what does `\sigma(T)` mean across
the literature we've seen so far?"

## 3. MAP-REDUCE shape

The store is **append-only** in the MAP phase. Each Stage 5 run
appends new CanonFingerprint records to a per-batch JSONL file.
This is the MAP step.

The REDUCE phase runs offline, aggregating across batches:

```
batches/batch-008.fingerprints.jsonl    [per-binding records]
batches/batch-009.fingerprints.jsonl
...
                       ↓ REDUCE
canon-store.json                        [aggregated CanonAggregate]
```

REDUCE is incremental: when batch N+1 arrives, the aggregate updates
without re-scanning batches 1..N (state-merge against the existing
aggregate). This keeps the operation cheap as the store grows.

The store is *queryable* by symbol (the primary key for inference)
and by canon (secondary — for the literature-lifted reduction in §5).

## 4. How (1) per-binding posterior consumes the store

The per-binding canon posterior (§3.2 of the bayesian mission)
becomes:

```
P(canon = c | symbol X, evidence E in current paper)
  ∝ P(c | X) prior                               -- from canon-store
  × ∏_s [voted-for-c_s ⇒ rel_s, else (1-rel_s)]  -- engine evidence
```

`P(c | X) prior` is the canon distribution for X across all prior
batches, normalised. The store IS the prior.

The very first batch sees a uniform prior (nothing in the store).
After batch N, the prior on common symbols is informative; on rare
ones it's still flat. This is the explicit MAP-REDUCE Joe asked
for — each batch contributes evidence that improves arbitration on
all future batches.

## 5. Literature-lifted strategy reduction (the (3) lift)

Joe's reframing of Bayesian model reduction: the literature already
contains semantic unifications. PM `\pmrelated` says
`{TopologicalGroup, Group}` are linked. ProofWiki namespaces say
`Definition:Set` and `Theorem:CardinalityOfSet` are conceptually
adjacent. nLab `[[parent]]` links encode the same.

When two strategies emit canons that are LINKED in the literature
graph, that's evidence the strategies are capturing the SAME
underlying concept under different framings. Specifically:

> If strategies A and B systematically emit canons (c_A, c_B) where
> c_A and c_B are linked in the literature graph for the same
> symbols, A and B are candidates for merging.

This is the Friston "Bayesian model reduction" mechanism with the
literature as the regulariser. We don't have to discover that two
concepts are related from scratch — we ask the literature.

Concretely: walk the canon-store. For each symbol X, look at the
top-k canons emitted by each strategy. If strategy A's top canon is
graph-adjacent to strategy B's top canon for many symbols, propose
the merge.

Validation: held-out gold — does the merged strategy retain the
combined precision of the parts, or does it regress? Apply the
merge only on no-regression.

This is the path from "longshot" to "tractable" — the literature
gives us the prior over which merges are plausible.

## 6. Concrete implementation slices

In order of cheapest-first:

### Slice F1 — schema + writer (1 day)
- `src/futon6/canon_store.py` with the dataclasses
- `write_batch_fingerprints(records, batch_id) -> path`
- Wire into Stage 5 + the QC viewer's `detect_grounded_symbols`
- Tests: round-trip, append-only correctness

### Slice F2 — reducer + query interface (1 day)
- `aggregate_canon_store(batch_files, prior_store) -> canon_store.json`
- `query_canon_distribution(store, symbol) -> dict[canon, aggregate]`
- Incremental update: state-merge a new batch against existing store
- Tests: idempotent reduce, correct aggregation

### Slice F3 — per-binding posterior consumes the store (1-2 days)
- Implements §3.2 of the bayesian mission
- `CanonPosterior` dataclass; reliability-weighted vote combination
  using the store's distribution as prior
- New return type from `detect_grounded_symbols`: per-binding
  posterior instead of single canon
- Held-out precision measurement: does it lift?

### Slice F4 — literature-lifted strategy merge proposer (2-3 days)
- Build extended literature graph (PM `\pmrelated` ∪ ProofWiki
  namespace structure ∪ Wikipedia categories, when available)
- `propose_strategy_merges(canon_store, literature_graph) -> [merge_proposals]`
- Validate each on held-out gold; accept on no-regression
- Tests: known synonymy cases produce expected proposals

### Slice F5 — Stage 5 closes the loop (1 day)
- Each Stage 5 run reads canon_store.json on startup, uses it as
  prior for that batch's grounding
- At end of run, MAP-step appends new fingerprints, then
  incremental REDUCE
- The system literally improves with each batch

Total: 6-9 days. F1 + F2 + F3 are the minimum viable loop
(persistent store + posterior-using-store); F4 + F5 close the
self-improvement gap.

## 7. What this is NOT

- Not a full theorem-fingerprint system (Billey-Tenner's broader
  vision). Just symbol-grounding fingerprints — narrower scope,
  same shape.
- Not a replacement for the gold extractors. Gold remains the
  supervised signal that initialises strategy reliability priors.
  The fingerprint store is the *aggregated evidence prior*; gold is
  the *strategy calibration*.
- Not an immediate accuracy win. The first batch with empty store
  performs identically to today's engine. Lift comes from batch
  N+1 onward as the prior accumulates.

## 8. Decision asked of Joe

(a) Is the schema in §2 the right granularity, or should
    fingerprints be coarser (per-paper-per-symbol with canon list)
    or finer (with position offsets for re-extraction)?
(b) Should F1-F2 run BEFORE or ALONGSIDE the §3.2 per-binding
    posterior (Slice F3)? F3 doesn't strictly need F1-F2 to work;
    it just doesn't *learn* without them.
(c) Persistence format: SQLite (queryable), JSONL+aggregate
    (cheap), or DuckDB (columnar, fast for canon-distribution
    queries)? JSONL+aggregate matches existing infrastructure
    (`learned-newcommand-vocab.json` is JSON); SQLite would let
    Stage 5 query during the run.

My recommendation: schema as written (a), F1-F2 then F3 (b),
JSONL+aggregate for v1 (c). Iterate if scale demands.
