# Excursion: E-anatomy-of-a-proof — the scope reading of A First Proof Sprint

**Date:** 2026-06-11
**Type:** E-prefix excursion (bounded scope-out, single owner end-to-end).
**Spawned:** from the futon6 prepwork line (Joe + Fable), sibling of
E-mission-head: same two-ascent shape, one floor down — the Anatomy paper
follows the unit of work; this follows a unit of *proof*.
**Status:** :greenfield — HEAD + IDENTIFY + MAP; DERIVE direction sketched.

## HEAD (Joe + Fable, 2026-06-11)

A proof, like a mission, admits two readings that must cohere:

The **scope reading** (anatomy): a proof is a nested system of binders and
their bodies — "let", "assume", "case", "pick", "witness" (Lamport's
structured-proof keywords are the Skolem vocabulary verbatim) — with typed
symbolic expressions inside them, and typed holes (gap notes) where
construction is still owed. The Skolem rules apply unchanged: a symbol used
without a binder is suspect; a binder whose body never uses it is suspect; a
stated ∃ owes a witness.

The **process reading** (physiology): the same proof is the residue of a
*sprint* — IATC moves (Assert, Suggest, Judge, strategy), worker dispatches
and returned certificates, wiring diagrams that grew through versions. The
expository register of the writeup is the sprint's HEAD-trace: what it
promises, the body (and the certificates) must discharge.

Satisfaction condition: for a First Proof problem, both readings are
computable from the artifacts on disk, and their agreements are where we
point and say "this part of the proof is real" — reading-agreement as
peradam-site, exactly as in the Anatomy paper §5.

## Design conditions carried in from the E-mission-head external review (§8)

Not post-hoc fixes — birth constraints. Each §8 critique generalizes:

1. **Scope condition on signals (8.1):** heuristic judgements in the
   expository register ("is small", "should be straightforward") are
   **observation-only**; only externally-discharged steps — a certificate, a
   verified computation, an independent check — are **targetable**
   (load-bearing for the proof's status). The anatomy must type which is
   which; a proof's health may never be computed from its own confidence
   prose.
2. **No laundering readouts (8.2):** never render a coverage number beside an
   unvalidated quality number. The workup's own idiom is the model — red gap
   boxes, green status boxes: the strong half solid, the noise half visibly
   provisional. The F5-analog is named at birth: *scope coverage has no
   predictive validity over proof correctness until a worked-proofs-with-
   outcomes corpus exists.*
3. **Heartbeat, not easy-mode (8.3):** scope markup that only exists in
   curated demos is M-aif-head's death by non-wiring. The test: does the
   markup fire for the next proof written by someone who isn't thinking about
   it? (Converges with E-scope-audit W12 — the ceremony belongs in the
   watcher.)
4. **Structure ≠ semantics (8.4):** scope coverage is structural. A gap note
   must carry its **discharge obligation** — the testable condition under
   which it closes — or the anatomy drifts toward a rhetoric simulator with
   colored boxes.
5. **Payload or it didn't happen (8.5):** a proof-step claim without its
   certificate payload is constructed-without-construction. The sprint
   already has the payloads (PHC certifications, SOS certificates, numerical
   verification JSONs); the anatomy binds step → payload explicitly.
6. **The Real Book (8.6):** proof-grain anchors accumulate only by playing
   more tunes through to their outcomes — the math.CT corpus and the prelim
   problems are the tune book. One worked example (this excursion) mints the
   format, not the idiom.
7. **Author ≠ reviewer caution (8.7):** Fable auditing instruments Fable
   codified — standing caution inherited verbatim.

## 1. IDENTIFY — the gap (evidence pass, 2026-06-11)

1. **Our own proofs read ZERO on all instruments.** `detect_scopes` and the
   nLab Skolem audit return nothing on `problem{1,4,6}-writeup.md`: the
   writeups use ASCII math (`Phi_n(p)`, `>=`) in indented code blocks, never
   `$...$`, and prose binders ("For a monic polynomial p(x) of degree n…")
   the regexes don't know. The proofs we care most about are the least
   instrumented — while 20,653 nLab pages are fully marked up.
2. **The md→tex workup already solved expression typing, mechanically.**
   `math-proofread-style.sty` types symbols by *renewing* LaTeX commands at
   render time (Greek/operator/relation/arrow/delimiter/number/…) — the
   typing is lexical, not semantic annotation. The CPU-side port already
   exists (`classify_expr` in `nlab_skolem_audit.py`); the workup and the
   audit are the same mechanism on two surfaces.
3. **The process trace exists at full grain but is unjoined.** Per problem:
   codex prompts/results `.jsonl` (the dialogue), wiring `v1..v6` `.mmd`
   (the diagram's growth — the piano roll), certificates (`.json` payloads),
   gap/status notes in the tex. Nothing links a writeup passage to its
   certificate or its wiring node.
4. **The corpus baselines exist to compare against.** nLab: 18.3% floating
   expressions, 33.4% HEAD discharge, 797 vacuous environments. A proof
   should beat the wiki on binding discipline — measurably.

## 2. MAP — bound context (each item used by name in DERIVE)

### Inventory: the sprint corpus

`/home/joe/code/storage/futon6/data/first-proof/` — per problem N:
`problemN-writeup.md` (the text), `problemN-wiring.json` + `problemN-v*.mmd`
(the process), `problemN-codex-prompts.jsonl` / `problemN-codex-results.jsonl`
(the dialogue), certificates (e.g. `problem4-case3c-phc-certified.json`).

### Inventory: the typed-tex workup

`latex/math-proofread-style.sty` (the lexical type system),
`latex/problemN-annotated.tex` + `latex/full/problemN-solution-full.tex`
(LaTeX register), `gapnote`/`statusnote` (the sorry/discharge idiom).

### Inventory: the instruments

`scripts/nlab_skolem_audit.py` (scope grades, expression typer, HEAD
discharge), `scripts/mission_scope_bindings.py` (binding flow),
`scripts/nlab-wiring.py` `detect_scopes` (binder regexes to extend).

### Inventory: the lessons

`holes/missions/E-mission-head.md` §8 (the review — design conditions above),
`holes/anatomy-of-a-futonic-mission.md` (the two-ascent format this joins).

## 3. DERIVE — direction (sketch, not yet built)

1. **Proof scope spine:** Lamport binders (`assume/let/case/pick/witness/
   suffices/prove`) + the prose forms in `problemN-writeup.md` ("For a … p",
   "where X = Y", "set a = b", "WLOG") as binder types; proof sections
   (Problem Statement / Answer / Proof Step k / Certification) as the phase
   spine — Statement+Answer is the HEAD that the steps must discharge, the
   same coupling `anatomy-of-a-futonic-mission.md` §5 makes for missions.
2. **Detector extension for the house register:** ASCII math and
   indented-block display math become first-class expressions
   (`classify_expr` in `nlab_skolem_audit.py` already classifies both once
   tokenized); prose binders join `SCOPE_REGEXES` in `nlab-wiring.py`.
3. **The join:** writeup passage ↔ wiring node (`problemN-wiring.json`,
   the `v*.mmd` growth) ↔ certificate payload
   (`problem4-case3c-phc-certified.json` is the exemplar) — step claims
   bound to their payloads (condition 5), gap notes bound to their discharge
   obligations (condition 4).
4. **The audit:** Skolem classes over the proof scope tree (free symbols,
   vacuous scopes, undischarged Statement promises) via the
   `mission_scope_bindings.py` flow analysis, reported in the gap/status
   idiom (condition 2), runnable from the watcher (condition 3), with the
   `E-mission-head.md` §8 conditions checked as standing obligations.

## Scope

### Scope in
1. One problem worked end-to-end as the running example — **Problem 4**
   (richest trail: analytic steps + algebraic elimination + computational
   certification + the de Bruijn discovery).
2. The detector extension for the proof register (ASCII math, prose binders).
3. Baseline Skolem numbers for all ten writeups, against the nLab corpus.

### Scope out
1. Any change to the GPU mining pipeline (this is its audit surface, not it).
2. Re-rendering the monograph.
3. Formalization (Lean etc.) — the anatomy types informal proofs.

## 4. First results (tasks 1+2 landed 2026-06-11; codex-1 build `fd5d377`, fable-2 review)

Review fixes applied (`proof_scope_audit.py`): display-block tokenization
double-counted every block expression at drifted offsets (761→585 records
after dedup at true offsets); weak-grading was span-grain while the printed
nLab baseline is paragraph-grain — now matched, so the comparison is honest.

**Baseline, ten writeups: 585 expressions, 77.8% floating vs nLab 18.3%.**
Per-writeup spread: problem3 25.0% (the disciplined one) → problem9 98.4%
(one detected binder in the whole writeup). Free symbols 24–65 per writeup;
7 vacuous scopes.

**The 77.8% is an upper bound, not a verdict** (the two-channel lesson):
problem9's text is full of binders the register vocabulary misses. Labeled
miss-classes for the next detector round, E-scope-audit style:
- `Let A^(1), ..., A^(n) in R^{3x4} be ...` — superscripted symbol LISTS;
- `For alpha, beta, gamma, delta in [n]` — greek-WORD lists (for-list-binding
  only takes single letters);
- `Fix camera-row pairs (gamma, k)` — Fix with prose object, no `$`;
- `Take lambda_{abgd} = 1` — Take/Suppose forms absent entirely;
- `Define P(A^(1),...,A^(n)) = det[...]` — Define with argument lists.

What survives after those land is the real indiscipline measure — and the
residue is expected to stay well above nLab: these writeups are *answer
summaries* in the expository register, not scope-disciplined proof texts.
Which is the mini-mission reading again: a writeup is mostly HEAD, and its
discharge lives in the certificates, not in its own prose (condition 5).

## First cut tasks

1. Tokenizer for the writeup register (ASCII math + indented display blocks)
   feeding `classify_expr`. [Unblocks everything.]
2. Prose-binder regexes ("For a … p", "where", "set", "WLOG") → scope
   records; run the Skolem audit on all ten writeups.
3. Problem 4 join by hand: steps ↔ certificates ↔ wiring versions — the
   worked example that mints the format.
4. Compare: do the proofs beat the nLab on binding discipline?
