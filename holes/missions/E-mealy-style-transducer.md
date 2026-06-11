# Excursion: E-mealy-style-transducer — attribute the mission's output tape

**Date:** 2026-06-11
**Type:** E-prefix excursion (bounded scope-out, single owner end-to-end).
**Spawned:** from E-mission-head, via the Skolem audit (E-scope-audit session 3,
Joe + Fable). Joe: a mission is "also, and maybe primarily, a transducer — this
is why we need the links into 'code' objects in substrate-2 to provide a solid
understanding of what a mission is doing."
**Status:** :greenfield — HEAD + design; INSTANTIATE is a Codex handoff, to be
belled when Joe says go.

## HEAD (Joe + Fable, 2026-06-11)

A mission is a **Mealy-style transducer**: the document is its *input tape plus
receipts* — the statement of intent, the context MAP binds, the records of what
was decided — and the **code graph is its output tape**: the commits, files,
and store writes the mission actually emitted. Phases are states; transitions
consume bound context items and emit code objects.

The satisfaction condition: **for any mission, "what did this mission do?" is
answerable from substrate-2 by following edges** — mission → its commits → the
files those commits edit. Then the Skolem audit's verdict upgrades: a MAP-bound
item is *discharged* if the mission's own output tape touched it, and
"confirmed unused" means undischarged on all three tapes (doc ends, doc text,
code edges). A document-grain fold (`mission_scope_bindings.py` today) is the
degenerate case where the output tape happens to be the document itself — which
is why a self-documenting excursion (E-mission-head) audits cleaner than a
build mission (M-war-machine-pilot, 46 confirmed-unused bindings at doc grain).

## 1. IDENTIFY — the gap (verified against the live store, 2026-06-11)

The output tape exists but is **unattributed**:

1. `code/v05/edits` (commit → file) is live in substrate-2 — the output tape —
   but **no commit→mission edge type exists**, so the tape belongs to no one.
2. `code/v05/file→mission` looks like the link we want but is not: its
   semantics is `mission/mentions-file` — minted from the mission *doc* citing
   the file. Input-tape grain; using it as discharge evidence is circular with
   the audit's doc-text channel.
3. The attribution chain is **derivable today, just never reified**:
   `commit_ingest.clj/resolve-session-for-commit` (timestamp-window heuristic
   over evidence events) gives commit→session; the Agency registry carries
   `mission-id` per agent (e.g. claude-3 → M-differentiable-substrate, live);
   so commit→session→agent→mission is computable from existing stores.
4. Prior art for declared attribution: the `Block:` commit trailer
   (`parse-block-trailer`, mana crediting) — the same trailer mechanism can
   carry a mission ident exactly.

## 2. MAP — bound context (each item is used by name in DERIVE/INSTANTIATE)

### Inventory: the build site

`futon3c/src/futon3c/watcher/commit_ingest.clj` — edge emission, trailer
parsing (`parse-block-trailer`, the `Block:` footer that templates provenance
(a)), and `resolve-session-for-commit` (provenance (b)) all live here.

### Inventory: conventions to follow

`futon3c/src/futon3c/watcher/file_ingest.clj` — the `code/v05/*` edge-type
registry and `post-hyperedge!` conventions the new edge must follow.

### Inventory: the tapes and the chain

The `code/v05/edits` hyperedges (futon1a:7071) are the output tape the new
edge attributes. The Agency registry's `mission-id`
(futon3c:7070 `/api/alpha/agents`) completes the backfill chain
session→agent→mission.

### Inventory: the consumer

`futon6/scripts/mission_scope_bindings.py` grows the third (code) channel;
E-scope-audit W14 is the worklist hook this excursion discharges.

## 3. DERIVE — the design

1. **New edge type `code/v05/commit→mission`** (commit sha, mission ident),
   emitted by `commit_ingest.clj` alongside `code/v05/edits`, following
   `file_ingest.clj`'s `post-hyperedge!` conventions. Props carry
   `:relation/provenance` ∈ {`trailer`, `session-heuristic`} so declared and
   inferred edges are always distinguishable.
2. **Provenance (a) — declared:** a `Mission: <ident>` commit trailer, parsed
   like `parse-block-trailer`. Starts working the day agents adopt it;
   near-zero cost; exact.
3. **Provenance (b) — backfill:** `resolve-session-for-commit` → session →
   agent (registry) → `mission-id`. Heuristic (30-min window), so tagged as
   such; never overwrites a trailer edge.
4. **Audit channel 3:** `mission_scope_bindings.py` queries
   commit→mission edges for the mission, joins `code/v05/edits` on the commit
   shas, and marks a MAP-bound file *code-discharged* if any attributed commit
   edits it. Verdict vocabulary: `doc-used` / `code-discharged` /
   `confirmed-unused` (undischarged on all three tapes).

## Scope

### Scope in
1. The `code/v05/commit→mission` edge type, both provenances, flag-gated dark.
2. The audit's code channel (+ tests, store mocked).
3. The `Mission:` trailer convention documented where agents commit.

### Scope out
1. Scope-grain attribution (which *scope* a commit discharges) — that is
   E-scope-audit W8's territory, after this lands.
2. Retroactive trailer rewriting of git history.
3. Any change to the existing `file→mission` (mentions) lane.

## 4. INSTANTIATE — Codex handoff (PENDING — bell when Joe says go)

Handoff spec (scope-bounded, per the futon3c protocol):
- **Goal:** items 1–3 of DERIVE in `commit_ingest.clj`; item 4 in
  `mission_scope_bindings.py` + `tests/test_mission_scope_bindings.py`.
- **:in (read-only):** `file_ingest.clj` (conventions), `registry.clj`
  (mission-id lookup), this doc.
- **:out:** `commit_ingest.clj` (extended), `mission_scope_bindings.py`
  (extended), tests both sides.
- **Gates:** clj-kondo clean on Clojure; `futon4/dev/check-parens.el` clean;
  `clojure -X:test` (futon3c) + `pytest tests/` (futon6) pass; new edge
  emission flag-gated OFF by default (store-write discipline); bell back with
  summary + commit shas.
- **Acceptance:** on a repo with a `Mission:`-trailered test commit, the edge
  appears with `provenance trailer`; audit reports a bound file as
  `code-discharged`; with the flag off, store writes are byte-for-byte absent.

## Relation to E-mission-head

E-mission-head types the HEAD (the ∃ — satisfaction conditions); this excursion
types the *discharge* (the output tape the Skolem function writes to). Together
they close the loop the Anatomy paper's §5 calls coupling: the organism reading
gains a motor record to go with its senses.
