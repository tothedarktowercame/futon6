# Library Research Findings (MO/MSE dump, local)

Date: 2026-02-11
Scope: Targeted checks for First Proof Problems 3 and 5 using local Stack Exchange dumps:
- `se-data/mathoverflow.net/Posts.xml` (snapshot 2024-04-06)
- `se-data/math.stackexchange.com/Posts.xml` (snapshot 2024-04-07)

This file follows the requested format from `library-research-brief.md`:
- title, URL/ID, author, date
- key claim/definition
- relation to the question (confirm / contradict / extend)

## Problem 5 (O-slice / transfer systems / N_infty / geometric fixed points)

1. Homotopical Combinatorics (answer)
- URL/ID: https://mathoverflow.net/questions/85124 (post id 467605)
- Author/date: David White, 2024-03-23
- Key claim: gives explicit definition of transfer systems, explains indexing systems, and states equivalence (up to homotopy) between indexing systems and `N_infty` operads.
- Relation: confirm. Strong evidence that transfer/indexing systems are the standard language behind incomplete equivariant structure; supports Q5.1/Q5.2 framing.

2. Categories on which one can determine all model structures? (answer)
- URL/ID: https://mathoverflow.net/questions/462305 (post id 462316)
- Author/date: David White, 2024-01-16
- Key claim: points to transfer-system literature (including "Self-duality of the lattice of transfer systems via weak factorization systems") and a 2024 AMS survey by Blumberg/Hill/Ormsby/Osorno/Roitzheim.
- Relation: extend. High-value bibliography for the exact transfer-system side of Q5.1/Q5.4.

3. Is this class of groups already in the literature or specified by standard conditions? (question)
- URL/ID: https://mathoverflow.net/questions/430939 (post id 430939)
- Author/date: kyleormsby, 2022-09-21
- Key claim: cites "Lifting N_infty operads from conjugacy data" and introduces a group-theoretic condition ("lossless") in that context.
- Relation: confirm. Direct evidence that modern `N_infty` work is organized around conjugacy/indexing data, matching the brief's hypothesis.

4. "abstract" description of geometric fixed points functor (question + answer)
- URL/ID: https://mathoverflow.net/questions/164210 (post ids 164210, 164350)
- Author/date: Tom Bachmann (question, 2014-04-24), Peter May (answer, 2014-04-25)
- Key claim: gives an abstract model-independent construction of geometric fixed points via isotropy separation (`X /\tilde E P`-style description), not tied to one point-set model.
- Relation: confirm. Supports Q5.3 that geometric fixed point characterizations are genuinely structural and not merely notational tautology.

5. "abstract" description of geometric fixed points functor (second answer)
- URL/ID: https://mathoverflow.net/questions/164210 (post id 164363)
- Author/date: Tyler Lawson, 2014-04-25
- Key claim: in equivariant symmetric spectra, geometric fixed points are computed levelwise via `H`-fixed points with representation-sphere compatibility `(S^{rho_G})^H = S^{rho_H}`.
- Relation: extend. Clarifies the "regular representation sphere" mechanism in the complete case, useful for testing any incomplete analogue.

6. The slice filtration does not arise from a t-structure (question)
- URL/ID: https://mathoverflow.net/questions/427280 (post id 427280)
- Author/date: desiigner, 2022-07-25
- Key claim: cites Hill's remark that desuspension does not shift slice-connective pieces in the way a t-structure would.
- Relation: extend. Indicates slice filtration behavior is subtle/nonformal, consistent with the brief's concern that naive restatements may miss depth.

## Problem 3 (ASEP / interpolation / Macdonald-Hall-Littlewood / Hecke exchange)

1. Yang-Baxter equation for the asymmetric simple exclusion process (ASEP) (question + answer)
- URL/ID: https://mathoverflow.net/questions/255022 (post ids 255022, 255075)
- Author/date: Y. Pei (question, 2016-11-18), Leonid Petrov (answer, 2016-11-19)
- Key claim: identifies ASEP S-matrix form and connects it to Yang-Baxter through stochastic six-vertex model limits; provides concrete references.
- Relation: confirm. Directly supports Q3.2 (exchange relations / integrable structure linkage).

2. Algebra/Algebraic geometry in statistical mechanics (answer)
- URL/ID: https://mathoverflow.net/questions/466723 (post id 466780)
- Author/date: Jules Lamers, 2024-03-10
- Key claim: summarizes ASEP as tied to XXZ/six-vertex and representation theory of Hecke/Temperley-Lieb and quantum groups.
- Relation: extend. Supports the Hecke-algebra framing, but does not give the q=1 interpolation-polynomial coefficient formulas.

3. First order approximation of the current in ASEP (question)
- URL/ID: https://mathoverflow.net/questions/112589 (post id 112589)
- Author/date: Guillaume, 2012-11-16
- Key claim: highlights known ASEP asymptotics/phase behavior (including step-Bernoulli regime split) and requests elementary interpretation.
- Relation: extend. Useful ASEP baseline, but not specific to interpolation ASEP polynomials.

4. When should we expect Tracy-Widom? (question)
- URL/ID: https://mathoverflow.net/questions/71306 (post id 71306)
- Author/date: Adrien Hardy, 2011-07-26
- Key claim: places ASEP/TASEP in Tracy-Widom universality context.
- Relation: extend (weak). Background only; not specific to Problem 3's polynomial stationary measure claim.

5. What do Macdonald polynomials hint about Rep(S_infty)? (question)
- URL/ID: https://mathoverflow.net/questions/410764 (post id 410764)
- Author/date: Student, 2021-12-14
- Key claim: asks for representation-theoretic meaning of Macdonald deformations and explicitly notes ASEP as one combinatorial interpretation path.
- Relation: extend (weak). Confirms community awareness of Macdonald<->ASEP links, but no direct interpolation-ASEP Markov-chain theorem in-thread.

## Negative findings (important)

1. No exact local hits for:
- "interpolation ASEP polynomials"
- "Corteel-Mandelshtam-Williams" (as a phrase)
- direct "q=1 exchange coefficient formulas" for interpolation ASEP polynomials

2. MSE dump signal for these terms is mostly low-quality/noisy compared with MO.

3. We did not find a local MO/MSE post that directly states the exact Problem 3 claim
`stationary distribution = F*_mu(q=1,t) / P*_lambda(q=1,t)` with explicit transition rates and a nontriviality discussion.

## Practical conclusion

- Problem 5: local corpus gives solid support that transfer/indexing system + `N_infty` is the right framework and points to the right papers/surveys; this likely helps validate or revise O-slice definitions.
- Problem 3: local corpus supports adjacent integrable/Hecke/ASEP structure, but does not locally settle interpolation-ASEP-at-q=1 exchange formulas or nontriviality.

Best next server-side step is to pull the cited arXiv papers from these MO posts and extract exact theorem statements for:
- transfer/indexing systems vs `N_infty`
- interpolation ASEP exchange relations and q=1 specialization.
