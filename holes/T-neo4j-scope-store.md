# T-neo4j-scope-store — a Neo4j store for superpod scope results

**Type:** T-prefix ticket. **Raised:** 2026-09-26 by Joe, after the SQLite/Neo4j sidecar
comparison: "the Neo4j scope store is worth building for superpod results, but we could
come back to that later". **Status:** open, parked; the SQLite sidecar is built first for
the current need.

## What it is for

The mark7 superpod runs write per-paper artifacts as files: expository passages with
scopes (kind, line span, slot-fills), proof graphs whose nodes cite S1 clause units, and S1
marks placed by source offset (`futon6/scripts/render_scope_margin.py` reads all three for
pages like `/wip/mark7-math_9906038-margin.html`). At the planned 5,000 papers
(`holes/mark7-ct-run-plan-5000.md`) that is roughly 500k scopes, 400k passages and 760k
proof nodes (estimated, ×600 from the 8-paper run mark7master-20260921).

A graph store would hold these as connected nodes, so that questions running across
papers can be asked as paths: which proof steps rest on a construction another paper
introduces, how a scope kind recurs along a chain of definitions, what a re-run changed
in the graph of one paper.

## What is already known

- `futon1b/holes/PROTO-neo4j-sidecar-2026-09-26.md` (kimi-6, E-kimi-task-47; review
  section by claude-12): Neo4j Community 5.26.0 from the tarball on Java 21, loopback
  only, 4 GB heap + 4 GB page cache. Loaded 13 papers, 399 passages, 857 scopes, 1,274
  proof nodes plus 160k v05 edits in 6.7 s over HTTP; 24 MB store; 1.4–1.6 GB server RSS.
  Warm queries 12–25 ms; two-hop 14.5 ms. The loader and the server were throwaway, in
  `/tmp/neo4j-proto`.
- On the queries measured so far SQLite was faster (two-hop 2.7 ms with the right
  indexes). Neither prototype ran 3+-hop or variable-length path queries, which is the
  case this ticket exists for.
- Neo4j has no as-of reads, so it holds the current reading of each run, with run id,
  model and generator as properties; history stays in the run directories.

## Open when this is picked up

1. The path queries that motivate it, written down with a sample paper set, before any
   build. A store is only worth running if at least one of them is slow or awkward in the
   SQLite scope index (`futon1b/holes/DESIGN-hyperedge-scope-sidecar-2026-09-26.md`, P5–P6).
2. Source of truth: the run files, as in the SQLite design; Neo4j is rebuilt from them.
   A re-run of one paper replaces that paper's subgraph in one transaction.
3. Where the server runs and who starts it: a second long-running JVM beside futon3c and
   futon1b, or started on demand for analysis sessions.
4. Proof-graph model: kimi-6 found `graphs/*.rung2.edn` are semcheck reports that name
   node ids, not graph definitions; the node and edge sources need confirming.

## Not in scope

v05 commit/var data (the design note recommends it leave futon1b for git) and anything
needing bitemporal reads.
