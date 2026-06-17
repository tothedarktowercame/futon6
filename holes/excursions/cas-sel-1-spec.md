# CAS-SEL-1 — spec spike (select-per-topology), grounded on the 4-proof CAS-0 corpus

*The "spec, no build" first step of the CAS-SEL breakdown
(`holes/handoffs/cas-sel-breakdown.md`). Written **on** the worked examples
`cas0-worked-{a93J05,a96J01,b97J01,a96J04}.md`. Author: claude-1, 2026-06-17.*

## 0. Joe's framing concern (the thing this spec is built around)

> "You're an LLM-based coding *agent*. Poor old LLaMA is just an LLM. If we expect LLaMA
> to select patterns *and* induce new ones like you've been doing, we may need an agentic
> coding *wrapper* and recognise runs can be long/expensive."

Correct, and load-bearing. My hand-process across the 4 proofs was an **agentic loop**
(segment → retrieve → read-and-compare → judge → generate-if-no-fit → register), not one
inference. The spec's whole job is to **factor that loop by cost**, so the common case is
cheap and deterministic and only the rare case needs the agentic wrapper. The mistake to
avoid is making *selection* agentic; selection is mostly retrieval + bounded classification.

## 1. Decompose what I actually did, and price each sub-operation

| # | sub-operation (what I did by hand) | kind | cost tier |
|---|---|---|---|
| a | break the proof prose into reasoning steps | parse | **bounded-LLM** (1 call) |
| b | for each step, find *candidate* patterns | retrieval | **deterministic** (hotword/NER) |
| c | judge "does step S instantiate pattern P? which slot?" | classify | **bounded-LLM** (1 call/step) |
| d | rule out near-misses / decide "no pattern fits" | compare | falls out of (c) |
| e | **mint a new pattern** when nothing fits | generate | **agentic** (rare, gated) |
| f | assemble the **wiring** (chain matched patterns' conclusions) | extract | **deterministic** |
| g | assemble the **sorry list** (matched patterns' undischarged HOWEVERs) | extract | **deterministic** |

The key empirical facts from CAS-0 that make this tractable:
- (f)+(g) are **fully deterministic** — the patterns already carry their conclusion (`THEN`)
  and obligation (`HOWEVER`) text in the `.flexiarg`; assembling wiring+sorry is field
  extraction over the matched set, **no LLM at all**. This is the single biggest cost win and
  it's *why* the "sorry = undischarged HOWEVER" finding matters operationally, not just
  conceptually.
- (b) is deterministic and **already built**: the P0 classical hotword spotter
  (`tag-patterns.bb` / `spot-terms.bb`) matches text → patterns via `patterns-index.tsv`
  hotwords ("all 25 patterns fire, 27–291 hits each"). LLaMA is **not** needed to generate
  candidates — only to *adjudicate* them.
- (e) is **rare and decreasing**: discovery rate over the corpus was **0,1,1,1**, and #1's
  zero came precisely because it reduced to a named theorem the pool already had. As the pool
  grows, (e) fires less. So the expensive path is both gated *and* amortising toward zero.

## 2. The cost gradient → the architecture (three tiers)

```
TIER 0  DETERMINISTIC          no model. candidate retrieval (hotword), wiring assembly,
        (cheap, always-on)     sorry extraction, all rung-0/1/2 checks. Runs on every proof.

TIER 1  BOUNDED-LLM            LLaMA-70B, single classification calls, parallel + cacheable.
        (per-proof, cheap)     step segmentation (1) + per-step match-verify (~1/step).
                               ≈ (1 + #steps) calls/proof — a 5-step proof ≈ 6 calls.

TIER 2  AGENTIC INDUCE         the agentic wrapper Joe names. Fires ONLY when Tier-1 finds no
        (rare, gated, async)   pattern for a step. Multi-call, tool-using, long/expensive.
                               Dispatched as a BELL (like the Codex handoffs), NOT inline.
                               Output (a new .flexiarg) is GATED: author ≠ reviewer before it
                               enters the pool. This is the prototype's "the cascade needs an
                               adviser, not a static roadmap" hole, made first-class.
```

**Mapping to existing infrastructure (composition over reinvention):**
- Tier 0 retrieval = the P0 hotword spotter, unchanged.
- Tier 0 checks = the built rung-0/1/2 `bb` stack (`iatc_argcheck`, `substance_gate`,
  `iatc_semcheck`), unchanged.
- Tier 1 = LLaMA-70B served exactly as the IATC loop already serves it (vLLM, the same box),
  but with *tiny* bounded prompts (a classification, not a generation).
- Tier 2 = the Agency mesh's bell-dispatch + author≠reviewer protocol, the same machinery
  these CAS-0 patterns were *themselves* written under. The induce wrapper is "a coding agent
  belled a pattern-authoring job," and Joe's "runs can be long/expensive" is acknowledged by
  making it async + budgeted, never on the select critical path.

## 3. The SELECT path (Tier 0 + Tier 1), concretely — answers Q1/Q2/Q3

For a proof `P` with steps `s₁…sₙ`:
1. **(T1) segment** `P` → steps with their text spans. *(1 call.)*
2. **(T0) retrieve** per step: hotword-match `sᵢ` against `patterns-index.tsv` → top-k
   candidate patterns. *(no model.)*
3. **(T1) verify** per step: one prompt — "which of {candidates} does `sᵢ` instantiate, and
   what is its slot fill (the `:cites` theorem / the decomposition / the ε-budget)? or NONE?"
   → a matched pattern + parameters, or NONE. *(~1 call/step; batched over candidates.)*
4. **(T0) assemble wiring**: order matched patterns by step; chain their `THEN`/conclusions
   into the argument DAG. *(no model.)*
5. **(T0) assemble sorry list**: for each matched pattern, emit its `HOWEVER` as a residual
   obligation, slot-filled. *(no model.)*
6. **select the check menu** (the original CAS-SEL goal): each matched pattern's obligation
   names which rung-2 check fires — e.g. `reduce-to-known-result` → warrant-resolution (R2c)
   on the cited step; `separate-into-independent-pieces` → a disjointness/closure check (R2b);
   `epsilon-of-room` → an "∀ε structure present" check; `count-over-a-decomposition` →
   "decomposition exhaustive+disjoint". *(T0 registry dispatch.)*
7. **escape to Tier 2** only if step `sᵢ` returns NONE at (3) with no candidate clearing
   threshold → enqueue an induce job; meanwhile mark `sᵢ`'s sorry as **thin** (undeclared-
   unfilled — claude-loop's rung-3 typology) and continue.

This resolves the recorded open questions empirically:
- **Q1 (menu location):** confirmed split. The **menu = the 39 `.flexiarg` patterns**
  (growable); **retrieval = deterministic hotword**; **the checks = the rung-2 registry**,
  dispatched by which patterns matched. flexiarg for the shape→cascade map, registry for what
  executes — each used for what it's good at.
- **Q2 (where select reads topology):** the matched-pattern set *is* the topology; it's read
  off the proof text by T0 retrieve + T1 verify (the coarse frame). Coarse → select → fill →
  re-render holds.
- **Q3 (topology-feature vocabulary):** **the vocabulary is the pattern pool itself** — a
  proof's topology is its sequence of matched patterns (+ slots). No separate hand-authored
  taxonomy; this was the empirical finding from all 4 worked examples.
- **Q5 (deterministic vs judge):** answered as a *gradient*, not a binary — Tier 0 deterministic,
  Tier 1 bounded judge, Tier 2 agentic. The LLM-fraction is exactly the Tier-1 + Tier-2 share,
  and it is **measurable** (dovetails with claude-loop's rung-3-1 residue spike).

## 4. The INDUCE path (Tier 2) — the agentic wrapper, scoped and gated

Fires only from 3.7. It is the one place that *needs* an agent, because minting a pattern is
what I did by hand: read several near-miss patterns, argue why none fit, write a new one in
house style, name it, check sigil/index collisions, register. Spec:
- **Trigger:** a step with no verified match (Tier-1 NONE, all candidates below threshold).
- **Shape:** a belled coding job (async; job-id; bell-back) — *not* an inline blocking call,
  because Joe's right that it can run long. It has tool access (read the pattern library,
  grep near-misses, write the `.flexiarg`, append to `patterns-index.tsv`, validate the TSV).
- **Gate (non-negotiable):** author ≠ reviewer. The induced pattern does **not** enter the
  pool until a second agent (or human) reviews it — the same discipline these 3 new patterns
  went through. A bad pattern pollutes every future select.
- **Budget:** each induce job carries a token/step ceiling; the orchestrator caps concurrent
  induce jobs. The select path never waits on induce — the proof's thin sorry is reported and
  the pattern lands on a later pass.
- **Amortisation:** because discovery is 0,1,1,1 and falling, steady-state induce-rate → low;
  most proofs over a mature pool are **pure Tier-0+1** (cheap).

## 5. Budget / runtime (making "long/expensive" quantitative)

- **Per-proof SELECT cost** ≈ `(1 segment + n verify)` bounded LLaMA calls, all small-prompt,
  parallelisable across steps, cacheable by (step-text, candidate-set). For the corpus:
  a93J05 ≈ 6, a96J01 ≈ 6, b97J01 ≈ 8 (multi-part), a96J04 ≈ 7. **Order ~10 small calls/proof.**
- **Per-proof INDUCE cost** = one agentic session *iff* a novel shape appears — easily 10²–10³
  calls (a full coding agent). But gated, async, rare, and reviewed.
- **Implication:** batch-checking a corpus is cheap (Tier 0/1 dominate); the expensive tail is
  bounded by how many *genuinely new shapes* the corpus contains, which CAS-0 shows is small
  and shrinking. So: **plan for cheap steady-state runs + occasional expensive induce bursts**,
  budget the bursts explicitly, and never block selection on them.

## 6. What each tier would have done on the corpus (grounding)

| proof | T0 retrieve+assemble | T1 verify | T2 induce |
|---|---|---|---|
| a93J05 | 5 candidate sets; wiring+sorry from matched | confirm 5 matches (EVT/Liouville slots) | none |
| a96J01 | candidates for 4 steps; disjointness step → weak | 4 matches; step "disjoint supports" NONE | **fire** → `separate-into-independent-pieces` |
| b97J01 | candidates incl. induction/class-eqn | matches; "class-equation divisibility" NONE | **fire** → `count-over-a-decomposition` |
| a96J04 | unfold/bound candidates strong; closer weak | matches; "ε arbitrary ⇒ 0" NONE | **fire** → `epsilon-of-room` |

The induce-fires line up exactly with the 3 patterns I minted — evidence the trigger
(Tier-1 NONE) is the right signal, and that Tier 0/1 would have carried everything else.

## 7. Hand-off to CAS-SEL-2..5
- **CAS-SEL-2 (check registry):** wrap rung-2 checks with applicability predicates keyed by
  *matched pattern* (§3.6). The predicate input is the Tier-0/1 select output.
- **CAS-SEL-3 (topology extractor):** = Tier-0 retrieve + Tier-1 verify (this spec defines it).
- **CAS-SEL-4 (`select`):** = §3.6 dispatch.
- **CAS-SEL-5 (genealogical):** unchanged — inherit imports'/citations' patterns via
  WARP-ORCH-3 tapestry; reduces Tier-1 candidate sets (cheaper verify) by priming likely patterns.
- **Induce (new, was implicit):** Tier-2 is a first-class deliverable, not a footnote — it's
  the seeding loop, and it's where Joe's agentic-wrapper + budget concerns live.

**Open (small):** the Tier-1 verify reliability on LLaMA-70B is itself to-be-measured (the
rung-3-1 residue spike covers it); set the NONE/threshold from that measurement, not a priori.
