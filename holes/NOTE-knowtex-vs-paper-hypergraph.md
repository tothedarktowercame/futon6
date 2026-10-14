# KnowTeX rules versus futon6 paper hypergraph

Discovery, 2026-10-12.  The local corpus came from
`scripts/futon6_config.py:eprints()` (FUTON6_EPRINTS unset):
`/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints`.  I used the
matching `.tar.gz` eprints and golden marks under
`data/showcases/ct-anatomy/golden/`.

## Rule coverage

| rule | current coverage | evidence |
|---|---|---|
| D1 proof `\ref` | partial | Classical `_REF_PATTERN` scans proof bodies and adds `depends_on` ends, `paper_hypergraph.py:354-360`; extraction stores `proof_refs`, `theorem_extraction.py:211-223`. It does not model manual `\uses`/`\proves`. |
| D2 statement `\ref` | partial harvest, no graph edge | `theorem_extraction.py:209-210` stores `refs`, but classical edge construction only scans proof bodies (`paper_hypergraph.py:334-370`). |
| D3 proof-header override | implemented (futon6 `b610ef5`, 2026-10-14, codex-47) | `paper_dependency.py` reads the proof block's bracketed title; a `\ref`/`\cref` there that resolves to a claim replaces the positional target (rule D3, attrs `displaced` = the positional claim), a header agreeing with position stays D1 with `header-confirmed`, an unresolved header counts under `unresolved_refs["proof-header"]`. A proof with no positional edge (it follows another proof) gets D3 edges from its body refs. 2002.06055: one D3 edge, `lemma-90 → theorem-24` (whitehead.tex:1679). `theorem_extraction.py` still searches proof content only. |
| D4 definition-term matching | partial literal infrastructure, rule absent | `latex_terms.py:16-29,44-55` extracts emphasis terms; classical definition-use scans supplied terms literally (`paper_hypergraph.py:372-413`). No stemming, `\index`, contiguous longest-match, or later-statement rule. |
| H1 nearest preceding statement | partial/implemented | `prev_claim` associates a proof with the preceding recognized claim (`paper_hypergraph.py:334-370`); extraction also pairs only an immediately following proof (`theorem_extraction.py:211-223`). |
| H2 corollary fallback | absent | No corollary-without-ref/incoming-edge fallback in the four examined modules. |
| H3 lemma-to-next-claim | absent | No forward-gap heuristic in the four modules. |
| H4 index aliases/longest-first | absent | `latex_terms.py:58-70` handles only PlanetMath xrefs; `tex_env_scopes.py:38-85` resolves environment names, not index aliases. |

`tex_env_scopes.py:19-31,38-85` supplies canonical environment/newtheorem
resolution and spans, but it creates scope records, not dependency edges.

## Three-paper measurement

For each paper I passed `latex_terms.extract_terms(text)` as the classical
concept vocabulary (the extractor API requires the vocabulary to be supplied)
and no techniques. Node counts are `type=count`; edge counts are
`type=count`. `\ref` ends count only `depends_on` ends in derivation edges.

| paper | nodes | edges | derivation | `\ref` depends_on ends | definition-use | claim cycles |
|---|---|---|---:|---:|---:|---|
| 2002.06055 | section 90, definition 187, equation 104, claim 196, proof 168, concept 682, term-use 19,652 | derivation 167, definition-use 3,381, structural-cooccurrence 545 | 167 | 198 | 3,381 | no |
| 0905.0462 | section 42, definition 50, claim 183, proof 166, concept 5, citation 6 | derivation 151, citation-grounding 7 | 151 | 195 | 0 | no |
| 2308.00651 | section 27, definition 24, equation 54, claim 79, proof 76, concept 78, term-use 476, citation 1 | derivation 75, definition-use 69, structural-cooccurrence 156, citation-grounding 1 | 75 | 134 | 69 | no |

Missed dependencies visible in the source (up to ten each; these are
candidate additions, not gold labels):

- **2002.06055:** D2 `lemma-23→sigma-modification`,
  `lemma-23→sigmap-modification`, `proposition-20→duskin-cocycle`,
  `lemma-27→eij`, `lemma-28→vhwgxfy`, `lemma-29→foneyf-zetayinv`,
  `proposition-27→foneyf-zetayinv`, `proposition-28→foneyf-zetayinv`,
  `lemma-33→ps-falg-coherence-i`, `lemma-34→ps-falg-coherence-ii`.
- **0905.0462:** D2 `theorem-1→camper`, `theorem-2→camper`,
  `theorem-2→camperr`, `theorem-2→ilmut`, `theorem-2→spade`,
  `theorem-2→curt2`, `proposition-3→amplecrown`, `proposition-5→disty`,
  `corollary-2→disty`, `proposition-7→tubal`.
- **2308.00651:** D2 `proposition-1→ex:continuous_kernels`,
  `proposition-1→eq:top_sup`, `lemma-10→def:abs_bicont`,
  `theorem-3→it:p_copydom`, `theorem-3→it:p_atomic`,
  `theorem-3→it:p_supported`, `theorem-3→it:p_smallest`,
  `theorem-6→eq:beh_object`, `theorem-6→eq:beh_morphism`,
  `proposition-18→eq:strong_idempotent`.

## Front end and smallest extension

`build-proof-anatomy-viewer.py:282-328,331-422` currently renders First Proof
writeup pages: expression and scope overlays, expression/grade tables, free
symbols, vacuous scopes, and optional gold-diff tables. It has no graph edges.
`dp_paper_view.py:814-859,1263-1307` emits a golden JSON text plus positional
marks for `dp` classifications (`label`, `ref`, `cite`, `claim`, `concept`,
`exposition`, symbols, proof moves) and `scope` bind/constrain/quant/env
marks; it does not emit paper-hypergraph edges. `paper-anatomy.el:131-198`
opens that JSON and overlays marks; `:311-347` supplies navigation/tag keys,
and `:459-500,531-627` supplies capability toggles and a side panel.

Smallest useful change: add `scripts/paper_dependency_view.py` to serialize
the classical graph plus style (`solid` D1, `dashed` D2/D4, `dotted` H*),
Tarjan cycle flags, and `data/showcases/ct-anatomy/golden/<paper>-dependency-review.json`;
add a dependency side-buffer and verdict writer to
`futon3c/emacs/paper-anatomy.el`. Leave the existing mark generator intact.

## Ordered packets

1. **Parser/graph packet:** `scripts/paper_dependency_view.py`; implement D2,
   D3, and stable edge-review records. Depends on this note and the classical
   graph API. Acceptance: fixture text yields each known statement/header ref
   with source location and a review JSON under golden.
2. **Heuristics packet:** same file; implement D4, H2, H3, H4 and SCC styling.
   Depends on packet 1's edge schema. Acceptance: focused fixture covers one
   positive and one negative case per rule, with no false cycle on the three
   measured papers.
3. **Emacs packet:** `futon3c/emacs/paper-anatomy.el`; render three styles,
   red cycle edges, and add/delete verdict actions persisted to the golden
   review file. Depends on packet 1's JSON. Acceptance: opening one golden
   review shows rows and a verdict survives reload.
