# Mission: Canon fingerprint store — Billey-Tenner instantiation for symbol grounding

**Date:** 2026-05-23
**Status:** DERIVE → INSTANTIATE — decisions resolved + holes articulated (Joe 2026-06-08): scope bindings (§2.1) + frequency-ordered MAP-REDUCE (§3.1) + SQLite; F1 schema delta in §8, holes in §9
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

### 2.1 Scope bindings — from symbol fingerprints to *theorem* fingerprints (Joe, 2026-06-08)

The §2 schema fingerprints **symbol** bindings — `(symbol, canon, …)`. Necessary, but not
sufficient for a *theorem* fingerprint. A theorem is not a set of bound symbols; it is a
**scoped relation** — bindings under *binders* that assign each symbol a **role**.

Pythagoras is the test. Knowing `a, b, c : Number` (symbol bindings) does *not* fingerprint the
theorem. What fingerprints it is the **scope**:
- **inputs (binders):** `a, b` are the *legs* and `c` the *hypotenuse* of a *right triangle* —
  numbers *mapped to* side-lengths under that geometric scope;
- **output (relation):** `a² + b² = c²`.

So §2 is taken one step further: each record carries not just `(symbol → canon)` but the **scope
binding** — the binder/role the symbol plays (input-with-role) and the output relation the theorem
asserts. A theorem fingerprint is `{scoped-inputs → output-relation}` — language-independent and
canonical in the Billey-Tenner sense, because the *roles + the relation* are the invariant, not the
surface symbols.

**Anchor, not position (Joe).** The locator is a **strategy in the futon4 / Arxana sense**, NOT a
character offset. "The third lemma in the paper" is semantically meaningful and stable; a position
offset is neither. Re-extraction anchors to the strategy (the structural locator), not the byte
position. → **resolves §8(a): drop `position`; the locator is an Arxana-style strategy.**

**Statements now; proofs later (a different deliverable).** This mission fingerprints theorem
*statements*. A **proof** fingerprint is separate: since a proof is a *composition of known
theorems*, a proof fingerprint is itself **OEIS-shaped** — an ordered list of proof steps, each
step citing a *statement* fingerprint. We will likely want both, but they are different kinds of
object, and the statement fingerprint is the prerequisite (a proof step references statement
fingerprints). Scoped here to statements; **proof-as-OEIS-of-steps** noted as the follow-on.

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

### 3.1 Build order — most-cited-first (the "mathematical genome"; Joe + Rob, 2026-06-08)

The MAP-REDUCE is **frequency-ordered**, not subject-agnostic-temporal. Build the reduced store
**most-cited-first**: a bibliographic pass orders papers (and the mathematical objects they use)
by citation count, and the REDUCE assimilates the **most-cited papers + most-common mathematical
objects first**, so later batches *reuse* already-reduced structure instead of rediscovering it.
MSC / arXiv codes are processed in most-cited-first order. The fingerprint database then grows like
a **mathematical genome** — the high-frequency core (the canon everyone cites) is laid down first
and reused; the long tail accretes against it. This is the explicit reuse-as-we-go MAP-REDUCE
(per Rob): we MAP-REDUCE *while* building, seeding the store with the genome's conserved regions.

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

## 8. Decisions — RESOLVED (Joe, 2026-06-08; supersedes claude-7's v1 recommendation)

**(a) Granularity → SCOPE bindings (per-theorem), not just symbol bindings; strategy-anchor, not
position.** Expand the schema to carry **scope bindings** — the binder/role of each input + the
output relation — so a *theorem* can be fingerprinted, not merely a set of bound symbols (see
§2.1, the Pythagoras test). **Drop `position`**; the locator is an Arxana-style **strategy** ("the
third lemma in the paper"). Theorem *statements* now; **proof fingerprints (OEIS-of-steps)** are a
separate later deliverable.

**(b) ALONGSIDE.** F3 (the §3.2 per-binding posterior) runs *alongside* F1-F2; build **F1+F2 first**
only to dissolve the chicken-and-egg (F3 doesn't *learn* without the store). AND the MAP-REDUCE is
**frequency-ordered / most-cited-first** (§3.1, the "mathematical genome") so structure is reused
as we go (per Rob): a bibliographic most-cited-first ordering of papers + MSC/arXiv codes + objects.

**(c) SQLite** (or any in-run-queryable store) — Stage 5 queries the store *during* the run, not
just offline. Drops JSONL+aggregate-v1; **in-run queryability is the requirement.**

### Consequent schema delta (for F1)
- `CanonFingerprint`: replace `position: str` with `strategy_anchor: str` (Arxana-style locator);
  add scope fields — `role: str` (the binder role, e.g. "hypotenuse-of-right-triangle"),
  `scope: str | None` (the binding scope), and lift the per-theorem grouping so a theorem's
  `{scoped-inputs → output-relation}` is recoverable.
- Persistence: **SQLite** (`canon_store.db`), schema mirroring the dataclasses, indexed by `symbol`
  (primary inference key) and `canon` (secondary); Stage 5 opens it read/write per run.
- REDUCE: seed most-cited-first (§3.1) before the long tail.

## 9. Next holes — INSTANTIATE (articulated 2026-06-08, per the §8 decisions)

- [ ] **F1** — SQLite `canon_store.db`: schema with **scope bindings** (`role`, `scope`,
      `strategy_anchor` replacing `position`) per the §8 delta; `write_batch_fingerprints`; wire into
      Stage 5 + the QC viewer's `detect_grounded_symbols`. Tests: round-trip, append-only.
- [ ] **F2** — reducer + **in-run** query over SQLite; incremental state-merge; **most-cited-first
      seed** (§3.1, the mathematical genome). Tests: idempotent reduce, correct aggregation, ordering.
- [ ] **F3** (alongside, per §8b) — per-binding canon posterior consuming the store as prior (§3.2 of
      M-bayesian-structure-learning); held-out precision lift.
- [ ] **F4** — literature-lifted strategy-merge proposer (§5); validate on held-out gold, no-regression.
- [ ] **F5** — Stage 5 closes the loop: read store on startup → prior → MAP-append → incremental REDUCE.
- [ ] **(follow-on, separate deliverable)** proof fingerprints = OEIS-of-steps (§2.1) — defer until
      statement fingerprints land.
