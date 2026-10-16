# M-informal-lean-alignment — tie informal proof steps to the Lean that proves them

**Type:** mission (design note and prototype). **Raised:** 2026-10-16 by Joe,
after the first mark8 run over the APM informal solutions.
**Status:** prototype built (`scripts/align_informal_lean.py`); first results below.

## The question this answers

What is the mined markup *for*? On the APM corpus the run's S4 expository
scopes are mostly empty. At least 17% of 8,677 fills are the slot description
copied back ("the structure/theory connected to"). An S3 proof graph
reproduces roughly what an agent gets by reading the solution itself. Neither
helps an agent writing a Lean blueprint for an unsolved problem.

What such an agent cannot do today is find the work already done elsewhere.
t98A01 (is a regular 5-sheeted cover Σ → Σ' with Σ ≅ Σ' possible; compute
χ(K#T#T#P#K#T)) still has a sorry. Its outline names the remaining pieces:
promote the circle cover x ↦ 5x to the torus map (x,y) ↦ (5x,y), and prove
Euler characteristic multiplicativity for finite covers. Elsewhere in the same
repository:

- t01A04 proves circle and torus covering maps
  (`apm_t01a04_torusCoverMap_surjective`);
- t97J07 has `t97J07_eulerCharacteristic_of_homologyData`;
- t91A02's blueprint routed around χ altogether, via π₁ and b₁.

22 informal solutions use "Euler characteristic" together with covers; 7 of
them have complete Lean proofs and 15 do not. The lemma index
(`LEMMA-INDEX.md`, 2,694 proved lemmas) can only be searched by Lean name and
signature. The informal claims are searchable by meaning, but are tied to
nothing formal.

**Use case to prove out:** for an informal claim in an unsolved problem, find
the Lean declarations in solved problems that prove the same or a similar
claim.

## The stage

Given a paper's S3 proof graph and its Lean source, align the two.

**Inputs**
- S3 graph: nodes (claim text, gloss, source lines), edges (premises,
  conclusion, warrant, `:missing-warrant` holes).
- Lean: the declarations of the paper's files (name, docstring, statement,
  body), and which local declarations each body uses.
- For retrieval: an index of proved declarations across all papers and the
  shared library (`LEMMA-INDEX.md` rows, or the same parse run over every
  file).

**Outputs** (per paper)
1. Aligned pairs: informal node ↔ Lean declaration, each with a score and the
   evidence that matched.
2. Informal nodes with no Lean counterpart: steps the formal proof did not
   take, or took differently.
3. Lean declarations with no informal counterpart: helper lemmas, which are
   places the informal proof skipped a step.
4. Retrieval: for each node, the best-matching declarations from other papers.

## The test

The stage, and the S3 mining under it, earn their place only if this holds:

> S3's `:missing-warrant` holes coincide with Lean helper lemmas. Where the
> informal proof skips a reason, the formal proof needed an extra
> declaration.

t98A01 shows the shape. S3 flags "χ = 0 ⇒ genus 1" as missing a warrant, and
the Lean file has `t98A01_orientable_genus_arithmetic` for exactly that step.
Measure over the papers that have both an S3 graph and a complete Lean proof:

- **coverage:** the share of S3 nodes aligned to a declaration above a score
  threshold, with precision judged by reading a sample;
- **hole agreement:** the share of S3 holes whose conclusion is matched by an
  otherwise unaligned helper declaration, against the same rate for nodes
  that are not holes (the baseline);
- **retrieval:** for the unsolved papers in the evaluation set, whether the
  top hits include the declarations a person would point to (t98A01 → t01A04,
  t97J07).

## First evaluation set

The 22 "Euler characteristic with covers" problems: t00A02 t00J01 t00J06
t00J08 t01A04 t01J05 t03J02 t03J04 t91A02 t91J05 t92J06 t94J04 t95J03 t96A01
t96J02 t97A01 t97J02 t97J07 t98A01 t99A03 t99J01 t99J03. Seven have complete
Lean proofs (t00J08 t01A04 t91A02 t96J02 t97A01 t97J07 t99J01).

## Prototype method (classical first)

Text on both sides is reduced to tokens: LaTeX commands and Unicode symbols to
words (χ and `\chi` to "chi"), Lean names split at case and underscore
boundaries, and the problem prefix removed. Pairs are scored by TF-IDF cosine
over the node's text and gloss against the declaration's name (weighted),
docstring and statement. A model pass that judges the candidate pairs is the
next step if the classical scores leave too much undecided. The prototype
reads the existing run (apm-501-xi-r1). Its S3 graphs were made from the
badly converted sources and without the problem statements, so they are a
floor, not the result.

## Caveats known now

- S3 glosses are sometimes attached to the wrong node (in t98A01, "the map is
  regular" sits on "5-sheeted covering"). That lowers alignment precision, and
  a guided tour for a human would inherit the error.
- Every S3 step is labelled `:by-definition`, and many warrants restate their
  conclusion, so relation labels carry no signal yet.
- Lean docstrings vary a great deal: t01A04 has 67, t98A01 has 4.
  Declaration names carry most of the signal where docstrings are thin.

## First results (2026-10-16, prototype over apm-501-xi-r1)

The prototype ran over all 501 papers in about two seconds; 487 have both an S3
graph and Lean declarations. Run: `scripts/align_informal_lean.py --run-dir
<run> --lean-root <apm-lean> --ids <ids> --out <dir>`.

**Alignment works only when lemmas are matched to steps.** Matching each claim
to its single best declaration let broad definitions absorb everything: in
t98A01, `t98A01IsRegularNfoldCover` took 8 claims. A Lean lemma proves a step
(hypotheses ⇒ conclusion), so lemmas are now matched to S3 edges, each edge
described by its premises, conclusion and warrant. A lemma serves one step.
Claims prefer theorems to definitions. With this, t98A01's flagged step e4
("χ = 0 ⇒ genus 1", premises n3, n4, n5) matches
`t98A01_orientable_genus_arithmetic` (score 0.47), which is the lemma a person
would name.

**Precision is moderate.** In a random sample of 14 matched step–lemma pairs,
about 9 are plausible (Sylow counting →
`apm_b96a02_sylow_nonidentity_injective`; "every u has a primitive" →
`apm_m02j01_normalizedPrimitive_add`; Rouché's "same number of zeros" →
`apm_a97j08_exists_compact_norm_margin`), 3 are wrong, and 2 are unclear.
Coverage over the 348 papers with complete Lean proofs: 29% of S3 nodes align
above the threshold.

**The hole test fails, for now.** Over the 348 complete papers, 97 of 422 S3
`:missing-warrant` steps (23%) have a lemma, against 732 of 2,901 other steps
(25%). S3's holes do not pick out the steps the formal proof needed a lemma
for. The cause may lie on either side. The matcher is about two-thirds
precise. The holes come from S3 runs made without the problem statements and
on badly converted sources, and their warrants are weak: many repeat the
step, some are boilerplate ("definition of …"), and one is a span id
(`L112-73a4`). Re-test on a run over the repaired sources before drawing a
conclusion about S3 itself.

**Retrieval looks useful today.** t98A01's steps queried against 16,344 proved
declarations (every problem's Lean files plus `ConstructionTargets/`):

- "z ↦ z⁵ is a 5-to-1 covering of S¹" → `squareCover_isCoveringMap` in
  `ConstructionTargets.TorusSquareDeckAction`. That module gives the
  ℤ/2 × ℤ/2 deck action for (z,w) ↦ (z², w²) on the torus, built on
  `ProductEvenlyCoveredTrivialization`, and is already used by t01A02. It is
  a direct template for t98A01's (z,w) ↦ (z⁵, w) with deck group ℤ/5, the
  first item in its proof outline.
- "every closed surface is classified by χ and orientability" →
  `apm_t99J03_orientable_genus_eq_of_eulerCharacteristic_eq`, which bears on
  its last item (homeomorphic presentations have equal genus).
- "χ of a connected sum" → `eulerCharacteristic_connectedSum`
  (`ConstructionTargets.Surfaces`).
- "χ multiplies under a finite cover" → only χ definitions and
  `t91A02_hasEulerCharacteristic_unique`. That is the honest answer: no
  problem has proved multiplicativity, and t91A02's blueprint routed around
  it.

A person can find each of these with effort. Retrieval found them by
mathematical meaning, from the informal claims, without knowing the Lean
names.

## Next

1. ~~Expose retrieval as a query tool~~ — done: `scripts/lean_retrieval.py`.
   `index --root <repo> --glob 'problems/*/lean/*.lean' --glob
   'ConstructionTargets/*.lean' --out <index.json>` indexes proved theorems
   and lemmas (no `sorry`): 16,233 for apm-lean, in 22 s. `query --index
   <index.json> "<informal claim>"` answers in about 0.3 s, with file and line;
   `query --run-dir <run> --paper <id>` answers each S3 step of a paper,
   leaving out the paper's own declarations. For t98A01 it also surfaced
   `apm_t98a03_regular_threeFoldCover` (a regular cover built from a torus
   rotation, complete). Nothing in it is APM-specific beyond the default
   `problems/<id>/` owner rule, so it can be shipped to anyone with scopes and
   Lean side by side.
2. Re-run the hole test on the S3 output of a run over the repaired sources
   (apm-lean `34c53beba`), with problem statements included (futon6
   `6d1db4f`).
3. Add a model pass that judges the top three candidates per step, to raise
   precision above two-thirds, and measure it on a hand-labelled set (t98A01,
   t99J03, t01A04).
