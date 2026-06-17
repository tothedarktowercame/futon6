# CAS-SEL breakdown — select-per-topology (each proof its own cascade)

*Breakdown of the `CAS-SEL` "needs breakdown" card (`holes/proofcheck-readiness.html`,
Rank D). Author: claude-1 (Joe asked me to break it down; he'll help on the open
questions — it's complex). Owner: claude-1 (dispatch + review). Part of
[[E-informal-proof-checking]]; consumes the rung-2 stack + the structure-first substrate.*

## The idea (what CAS-SEL is)

Today rung-2 applies `{R2a, R2b, R2c}` **uniformly** to every graph. But an induction
proof, a diagram chase, and a proof-by-contradiction need *different* checks. CAS-SEL
makes the cascade **per-proof**: match a check-**menu** to *this* proof's **topology**.

It is a **port of the existing prototype** `futon3c/holes/excursions/pipeline-pattern-cascade.html`
(Moran-style), which already runs this over the stack's own missions+patterns:
*basins* = pattern-clusters, *cited patterns warrant basins*, *hollow nodes = residual
sorries*, and `select` = the **inductive attachment rule**: start from attested nodes →
attach the next item by **concrete cited-pattern overlap (before embedding proximity)** →
**split a basin when it is mixed**. Swap `(missions, library-patterns) → (proofs,
reasoning-patterns)` and it transfers.

## CAS-0 — empirical prerequisite (Joe, 2026-06-17): patterns before matching

The open questions below must be settled **empirically, not by fiat** — you cannot match
on patterns you do not have, and the pool is thin today: **36 math-informal, 0 math-formal,
+ a 12+8 survey** (the MO·math.SE RM corpora — ~900K entries / 3.4 GB — are *un-mined*; the
full embed→cluster→name pass never ran). So Rank D **leads with CAS-0**: work an example
**APM proof** (before arXiv) end-to-end — induce its **sorry + wiring**, and iteratively
**find** (in math-informal / the un-mined RM corpora) or **write** the reasoning patterns
its steps need. The topology vocabulary (Q3), static-vs-adaptive (Q4), and
deterministic-vs-judge (Q5) **emerge** from these worked examples. The brainstormed
match-axes below become a source of *candidate patterns to look for*, not a fixed schema.
CAS-SEL-1's spec-spike runs **on** the worked-example output, not ahead of it.

## Decomposition (car-of-sequence)

- **CAS-SEL-1 · spec spike (FIRST; no build).** On ~4 real proofs (the loop-run-70b
  graphs), hand-define two things by example: (a) the **topology-feature vocabulary** —
  what `select` matches on; (b) the **select-table** — topology-features → which checks
  fire. Output: a short spec grounded in real proofs. This resolves the underspecification
  before any code. *(Mirrors the SFC-AGG spec-spike pattern.)*
- **CAS-SEL-2 · the check REGISTRY.** Wrap the built rung-2 checks (R2a/R2b/R2c) — and
  stubs for proof-shape-specific ones — as registry entries, each with an
  **applicability predicate** over topology features. (Q1: the "registry executes" side.)
- **CAS-SEL-3 · the topology EXTRACTOR.** From the coarse frame (the IATC graph + the
  document skeleton + the concept profile + the sorry-topology), compute a proof's
  feature set. (Q2: coarse-frame source.)
- **CAS-SEL-4 · `select`.** Match features → the check set (a `.flexiarg`-pattern dispatch
  per Q1, or a registry-predicate dispatch for v0); run only the selected checks; emit
  the per-proof cascade + verdict.
- **CAS-SEL-5 · genealogical select (DEFERRED — WARP-ORCH dep).** A proof inherits its
  imports'/citations' patterns via the citation-descent (`warp_citations`/`bib` +
  `mark3_thread_tapestry` phylogeny — **shares WARP-ORCH-3's artifact** with R2d-3). This
  is the prototype's "attach by cited-pattern overlap." Waits on WARP-ORCH being live.

## Open questions (Joe to help — these gate CAS-SEL-1)

- **Q1 (recorded leaning) — where does the menu live?** Split: **checks = a registry**
  (executable, what's built), **select-rule = `.flexiarg` patterns** over the topology.
  *Confirm?*
- **Q2 (recorded leaning) — where does `select` read topology?** The **coarse frame** —
  making the resolution axis (coarse→fine) the control flow (coarse → select → fill →
  re-render). *Confirm the build-order?*
- **Q3 (NEW — the crux of CAS-SEL-1) — the topology-feature vocabulary.** What does
  `select` actually match on? Candidate axes: **proof-shape** {induction, diagram-chase,
  contradiction, construction, case-split, computation}; **concept-profile** (which
  concepts/MSC); **sorry-topology** {orphan nodes, missing-warrant edges, undefined
  terms}. Which axes are v0, and at what granularity?
- **Q4 (NEW) — static vs adaptive `select`.** Is v0 a *static* feature→check map, with the
  prototype's **pattern-seeding loop** (repeated held-sorry shapes → new menu patterns)
  deferred to v2? Or is the seeding loop core?
- **Q5 (NEW) — the menu beyond R2a/R2b/R2c.** The proof-shape-specific checks
  (induction-schema, diagram-chase commutativity, …) — drawn from the seed inventories
  (Pólya / RM survey / expository taxonomy). Are they **deterministic** (like rung-2) or
  **judge-based** (rung-3)? This sets whether CAS-SEL stays CPU or pulls in the LLM.

## Ready-to-dispatch when

CAS-SEL-1 (the spec spike) is dispatchable once Q3 + Q4 are settled (Q1/Q2 confirmed).
CAS-SEL-2/3/4 follow from the spike's output. CAS-SEL-5 waits on WARP-ORCH-3.

## Remaining gaps / notes
*(append findings + commit shas here.)*

### CAS-SEL-1 spec spike DRAFTED (claude-1, 2026-06-17) → `../excursions/cas-sel-1-spec.md`
Written on the 4-proof CAS-0 corpus. Resolves Q1/Q2/Q3 empirically (menu = the flexiarg pool;
topology = matched-pattern sequence; retrieval deterministic, checks = rung-2 registry) and
recasts Q5 as a **cost gradient** prompted by Joe's "LLaMA is just an LLM" concern. Core move:
**factor the agentic loop I ran by hand into three tiers** —
- **Tier 0 deterministic** (no model): hotword candidate retrieval (reuse P0 spotter), wiring
  assembly + sorry extraction (reads matched patterns' `THEN`/`HOWEVER` fields), rung-0/1/2.
- **Tier 1 bounded-LLM** (~10 small LLaMA calls/proof): step segmentation + per-step match-verify.
- **Tier 2 agentic induce** (rare, gated, async bell, author≠reviewer): mint a new pattern only
  when Tier-1 finds NONE — the seeding loop; this is where "runs are long/expensive" lives.
Induce-rate is 0,1,1,1 and falling, so steady-state runs are cheap Tier-0/1; expensive bursts
are bounded by # genuinely-new shapes. CAS-SEL-3 = Tier-0+1; CAS-SEL-4 = the §3.6 dispatch;
CAS-SEL-2 predicates key on matched-pattern; induce promoted to a first-class deliverable.

### CAS-SEL-2 check registry BUILT (codex-4, 2026-06-17)
Added `scripts/cas_checks.py` as the executable check registry without editing
`scripts/cas_select.py`. The registry imports the existing `cas_select.CHECK_MENU`,
keys predicates on matched-pattern `fires` labels, executes built R2a/R2b/R2c/R2d
checks through `scripts/iatc_semcheck.bb`, and leaves proof-shape-specific checks
(`R2b-disjointness`, `decomposition-exhaustive`, `forall-eps-structure`,
`cases-exhaustive`, `well-defined-on-quotient`) wired as N/A stubs, preserving
N/A != FAIL.

Evidence: `tests/test_cas_checks.py` pins that registry selection reproduces
`cas_select`'s emitted `checks` field on the four worked fixtures (`a93J05`,
`a96J01`, `b97J01`, `a96J04`) and runs real semcheck-backed `R2c-warrant`,
`R2b-closure`, and `R2d-concept-coverage` on
`data/iatc-argument-graphs/loop-run-70b/0706.1286.edn` with rates `0.2`, `1.0`,
and `0.5` respectively. Gates run: `python3 -m py_compile scripts/cas_checks.py
tests/test_cas_checks.py`; `pytest -q tests/test_cas_checks.py` (`4 passed`).
