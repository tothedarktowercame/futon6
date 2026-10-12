# Packet: what KnowTeX's dependency rules would add to futon6's paper hypergraph (discovery, read-only)

From claude-16 (owner, M-象-2000 in futon3c), 2026-10-12. Mode: work. Repo:
`/home/joe/code/futon6`. **No code changes**: the deliverable is a note,
`futon6/holes/NOTE-knowtex-vs-paper-hypergraph.md`, committed alone (touched
files only, never `-a`). Delete this packet file in the same commit.

## Why

Joe asked (2026-10-12) for front-end work on futon6 inspired by KnowTeX
(Uskuplu, Moss, de Paiva, arXiv:2601.15294, CC BY 4.0; no public repository
found). KnowTeX builds a theorem/definition/proof dependency graph from LaTeX
and shows it with an edge review table. Its rules, from the paper:

- Manual: `\uses{labels}` (dashed inside a statement = conceptual; solid
  inside a proof = logical), `\proves{label}`; node labels from `\label`,
  then emphasised terms, then `\index`, then a number.
- Deterministic: D1 `\ref/\cref/\eqref` in a proof → edge from the referenced
  node to the proof's statement; D2 the same in a statement; D3 an explicit
  proof header (`\begin{proof}[Proof of Theorem~\ref{x}]`) overrides the
  default proof association; D4 terms emphasised or indexed in
  definition-like environments are stemmed and matched in later statements
  (multi-word terms contiguous).
- Heuristic: H1 a proof defaults to the nearest preceding theorem-like
  statement; H2 a corollary with no refs and no incoming edge links to the
  nearest preceding theorem/proposition; H3 a lemma links to the next
  theorem/proposition within a gap (default 3 nodes); H4 index terms matched
  longest-first with `|see{}` aliases.
- Output: DOT/TikZ/PNG; solid (proof-level), dashed (statement-level), dotted
  (heuristic); cycles by Tarjan drawn red; optional transitive reduction;
  edge review table (source, target, type, location, rule) with add/delete;
  corrected `\uses{}` exported back to the source. Multi-file via
  `\input/\include/\import/\subfile`, `\includeonly` honoured.
- Reported: group-theory notes F1 0.818 (path-based precision 0.971); PFR
  blueprint infer-mode F1 0.826; MathGloss D4-only F1 0.671, all rules 0.885.

## Questions to answer, with file:line citations

1. **Rule coverage.** For each of D1–D4, H1–H4, say whether
   `src/futon6/paper_hypergraph.py` (classical arm,
   `extract_paper_hypergraph_classical`; the `derivation` edge at ~l.335–370
   already does H1 via `prev_claim` and D1 via `_REF_PATTERN` in the proof
   body), `src/futon6/theorem_extraction.py` (`refs`, `proof_refs`),
   `src/futon6/tex_env_scopes.py` (`\newtheorem` resolution) or
   `src/futon6/latex_terms.py` already implements it, partly implements it,
   or lacks it. One table.
2. **Measured on three papers.** Pick three papers with local `.tex` sources
   that also have golden marks under `data/showcases/ct-anatomy/golden/`
   (the eprint directory is `FUTON6_EPRINTS`; if unset, find where the
   corpus lives from `REPRODUCING.md` Tier 2 and say which path you used; if
   no local source exists, say so and stop at question 1 and 3). Run the
   classical arm on each; report node counts by type, `derivation` edges,
   `\ref`-based `depends_on` ends, definition-use edges, and whether the graph
   has cycles (compute SCCs over claim→claim dependencies). Then list, for
   each paper, up to ten dependencies you can see in the text that the
   classical arm did not produce, tagged with the KnowTeX rule that would have
   produced them (D2, D3, D4, H2, H3) or `none`.
3. **Front end.** What `scripts/build-proof-anatomy-viewer.py`,
   `scripts/dp_paper_view.py` and `futon3c/emacs/paper-anatomy.el` show today
   for one paper (node and edge kinds, interaction), and the smallest change
   that would add a dependency view with KnowTeX's three edge styles, red
   cycle edges, and an edge review table whose verdicts are saved as a file
   under the golden dir. Name the files it would touch.
4. **Plan.** Two or three packets, each one file / one behaviour / one
   acceptance test, in order, with the premise each depends on.

Under 120 lines. Bell claude-16 back with the sha and the rule-coverage table.
