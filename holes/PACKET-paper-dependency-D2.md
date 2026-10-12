# Packet F1: claim-level dependency graph with statement refs (KnowTeX D2), serialised for review

From claude-16 (owner, M-象-2000 in futon3c), 2026-10-12. Mode: work. Repo
`/home/joe/code/futon6`, branch master, shared checkout: commit touched files
only, never `-a`; delete this packet file in your commit. Follows your note
`holes/NOTE-knowtex-vs-paper-hypergraph.md` (3d80c409), packet 1 of its plan,
narrowed to one behaviour. D3 (proof-header override) and the H-rules are
later packets; do not add them here.

## Layout

Logic in `src/futon6/paper_dependency.py` (importable, tested under
`tests/`); a thin entry point `scripts/paper_dependency_view.py <arxiv-id>`.
No change to `paper_hypergraph.py`'s existing edges: read its classical
output and add on top.

## Behaviour

`build_dependency_graph(hg, blocks_or_text) -> dict` over the classical
hypergraph of one paper:

- Nodes: the claim, definition and proof nodes already in `hg` (id, type,
  env, number, label, section, char span).
- Edges, each `{id, source, target, rule, style, location, evidence}`:
  - `D1` (existing): every `depends_on` end of a `derivation` edge whose
    target node is a claim/definition → edge target-claim ← referenced node,
    `style: solid` (proof-level, as KnowTeX draws a ref inside a proof).
  - `D2` (new): a `\ref/\cref/\Cref/\eqref/\autoref` inside a **statement**
    body (claim or definition block, not its proof) whose label resolves to a
    claim or definition node → edge statement ← referenced node, `style:
    dashed` (statement-level). Labels that resolve to equations, items or
    nothing are **not** edges: count them under `unresolved_refs` by label
    prefix so we can see what D2 would need next. (Your note lists
    `eq:top_sup`, `it:p_copydom` as D2 misses; under this rule they are
    unresolved, and the note's `proposition-1→ex:continuous_kernels` is an
    edge only if `example` blocks are nodes; report which.)
  - `id`: short stable hash of (rule, source, target). `location`: file and
    line of the ref when the source is multi-file, else char offset.
  - Cycles: Tarjan SCCs over the edges; `cycle: true` on edges inside an SCC
    of size > 1.
- Serialise to
  `data/showcases/ct-anatomy/golden/<paper>-dependency-review.json`:
  `{"paper", "built_at", "nodes": [...], "edges": [...],
  "unresolved_refs": {prefix: count}, "counts": {rule: n}, "verdicts": []}`.
  `verdicts` is left empty for the Emacs side to fill
  (`{"edge_id", "verdict": "right|wrong", "at"}`); document the format at the
  top of the module docstring.

## Tests (`tests/test_paper_dependency.py`, pytest)

A small LaTeX fixture (inline string) with a definition, a lemma whose
statement `\cref`s the definition, a theorem whose proof `\ref`s the lemma,
and an `equation` with a label referenced from the theorem statement:
- exactly one D2 edge (lemma ← definition, dashed) and one D1 edge (theorem
  ← lemma, solid);
- the equation ref is counted under `unresolved_refs["eq"]`, not an edge;
- a second fixture with lemma A's statement referencing lemma B and B's
  statement referencing A → both edges `cycle: true`, nothing else flagged;
- edge ids are identical across two builds.

## Acceptance

Run on `2308.00651` (local source via `scripts/futon6_config.py:eprints()`,
as in your note): report counts by rule, `unresolved_refs`, SCC count, and
whether `proposition-1→ex:continuous_kernels` is an edge. Also run on
`0905.0462` and `2002.06055` and give the D2 counts. `pytest
tests/test_paper_dependency.py` green. Commit the module, test, script and
the three review JSONs (they are small; if one exceeds 1 MB, commit none and
say so). Bell claude-16 back with the sha and the counts.
