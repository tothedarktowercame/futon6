# E-bge-retrieval-cas-sel-3b

Author: claude-1, 2026-06-18. Bounded Linode experiment, modelled on
`E-70B-on-raw-control-arm.md`. Owns the runner `scripts/linode-bge-retrieval.sh`
and the payload `scripts/cas_sel_3b_embed_experiment.py`.

## Why this experiment exists

CAS-SEL-3's Tier-0 retrieval is classical **hotword overlap**, with a measured
ceiling: recall@4 = **15/22**, full-pool ceiling **19/22** — three fixture steps have
**zero lexical overlap** with their correct pattern (`a93J05/s3` "z=z₀+mω₁+nω₂" →
`quotient-by-irrelevance`; `a96J01/s2` → `construct-auxiliary-object`; `b97J01/s6`).
CAS-SEL-3b proposed an embedding modality to lift this. On the dev box, **bge-small**
gave union recall **17/22** and recovered **none** of the 3 zero-overlap steps, and
**bge-large — the spec's model — was killed loading** (1.3 GB; dev-box memory; the
OOM lesson in miniature). So the question is unresolved on the dev box.

**This experiment resolves it on the bigger box, and it is discriminating:**
- **bge-large recovers the 3 zero-overlap steps** → the ceiling was **model size**;
  ship the embedding modality for CAS-SEL-3b and re-pin the honest-recall test up.
- **bge-large recovers none** → the ceiling is **text-vs-structure**: those matches
  are *structural*, not lexical or semantic-text (a step's prose shares nothing with
  its pattern; the link is the argument shape). That is the empirical case for the
  **R-GCN / structure-first** direction (final-checklist §6) — text retrieval, at any
  size, plateaus there.

Either outcome is decisive and cheap to get.

## Design

- **Standalone embedding job.** It does **not** serve or need the 70B. Runs on **CPU**
  by default (`DEVICE=cpu`) so it never contends with a vLLM 70B that may be filling
  the GPUs at TP=4. The corpus is tiny (39 patterns + 22 steps) — the only real cost is
  the model load + a one-time ~1.3 GB HF download for bge-large. (Small-data BGE is the
  README-embeddings "cheap, safe" case — *not* the big-data recompute that OOM'd.)
- **Portable, no futon3 needed on the box.** The payload reads the committed pattern
  snapshot `data/cas-select/pattern-texts.json` (39 patterns) when futon3 (cas_select's
  live source) is absent, and the fixture steps from the futon6 checkout. On the dev box
  it uses the live patterns and refreshes the snapshot.
- **Asymmetric BGE retrieval** (the documented mistake to avoid): pattern texts embed as
  passages; step texts embed as queries with the instruction prefix
  `"Represent this sentence for searching relevant passages: "`. (No prefix for MiniLM.)
- **Compares** bge-large vs bge-small vs MiniLM × two pattern-text representations
  (`title+conclusion+hotwords`, `full`), each with an **embedding-collapse audit**
  (cosine-to-mean std; `<0.01` collapse, `<0.05` mild — `audit-graph-embeddings.py`).
  The collapse to avoid is **R-GCN-specific**; BGE-text is the validated escape — the
  audit here just confirms the pattern vectors discriminate.

## What needs to run

On the provisioned box, from the futon6 checkout (`$REPO`):

1. `bash scripts/linode-bge-retrieval.sh` — ensures `sentence-transformers` in the venv,
   then runs `cas_sel_3b_embed_experiment.py` for each model × representation, prints a
   headline table + the discriminating verdict, writes per-model JSON to `$OUTDIR`
   (default `/tmp/cas-sel-3b`). Env overrides: `REPO/VENV/PYTHON/DEVICE/MODELS/REPRS/OUTDIR`.

## Artifacts

| what | where |
|---|---|
| per-model results | `/tmp/cas-sel-3b/<model>__<repr>.json` (recall, zero-overlap recovery, collapse audit) |
| pattern snapshot (committed) | `data/cas-select/pattern-texts.json` |
| dev-box baseline | bge-small: hot 15/22 · embed 12/22 · union 17/22 · recovered NONE · collapse mild (0.0226) |

## How to read the result

- **bge-large `accept=YES`** (all 3 recovered, union > 19/22) → model size; ship embedding
  CAS-SEL-3b, re-pin `test_tier0_retrieval_recall_is_honest` upward, note the embedding
  modality is model-free *of the generative LLM* (an embedding model is a separate tier).
- **bge-large `accept=no`** (recovers NONE) → text-vs-structure ceiling → the R-GCN /
  structure-first direction (§6) is the real path for those steps; record it as the
  decisive evidence and keep CAS-SEL-3b on hotword-only for now.
- **`collapse=mild/COLLAPSE`** on the pattern vectors → the representation
  under-discriminates; `--repr full` is the first lever.

## Result — RAN 2026-06-18 (box `mark4-70b-20260618-exp`): decisive, text-vs-structure

bge-large (the spec model that OOM'd on the dev box) **recovers NONE of the 3
zero-overlap steps**, and barely beats bge-small on union:

| model | repr | hotword | embed | union | recovered | accept |
|---|---|---|---|---|---|---|
| **bge-large** | full | 13/22 | 11/22 | 16/22 | **NONE** | no |
| bge-large | title+concl+hot | 13/22 | 11/22 | 16/22 | NONE | no |
| bge-small | full | 13/22 | 12/22 | 16/22 | NONE | no |
| MiniLM | full | 13/22 | 9/22 | 15/22 | NONE | no |

Small→large bought **nothing** on the steps that matter, and *both* pattern
representations (`title+conclusion+hotwords`, `full`) recovered none — so it is not a
model-capacity nor a representation artefact. **The ceiling is text-vs-structure.**

## By-hand analysis of the 3 zero-overlap steps

All three are the **same failure type: an abstract-method pattern vs. a concrete,
domain-specific instantiation** — the step's prose and the pattern's prose share no
surface, so any text model misses by construction.

| step | says (concrete) | should match (abstract) |
|---|---|---|
| `a93J05/s3` | "congruent modulo the period lattice" (number theory) | `quotient-by-irrelevance` |
| `a96J01/s2` | "partition [0,1] into intervals, telescoping" (analysis) | `construct-auxiliary-object` |
| `b97J01/s6` | "build the upper central series" (group theory) | `construct-auxiliary-object` |

The disambiguating signal is the step's **role in the proof flow** — `construct-*` emits
an object that *later* steps consume; `quotient` sits between "construct a domain" and
"extend by symmetry". That is a **neighbourhood / sequential-role** signal, **not** a
joint-multi-premise *hyperedge* signal. (So HyperGCN's specific advantage is orthogonal
to what these 3 need — park it for the genuinely hyperedge cases: citation/genealogy,
CAS-SEL-5.)

## EXP-3b — does proof-flow *context* recover them? No (text-context dilutes)

Added `--context N` to `cas_sel_3b_embed_experiment.py`: the embedding query carries the
step ± N proof neighbours (hotword baseline and the zero-overlap definition stay on the
isolated step, so any gain is attributable to context). Sweep (laptop, CPU):

| model | ctx | embed | recovered |
|---|---|---|---|
| bge-small | 0 | 12/22 | NONE  *(reproduces EXP-3 — control ✓)* |
| bge-small | 1 | 10/22 | NONE |
| bge-small | 2 | 8/22 | NONE |
| MiniLM | 0→2 | 8→5/22 | NONE (one noise hit at ctx2) |

Embed recall **degrades monotonically** — concatenating neighbour text pours in more
domain vocabulary and *dilutes* the query. The by-hand read was half right: the step's
pattern *is* fixed by its neighbours' **roles**, but that role is not in their raw
**text**. Flattening structure to text destroys it.

## Where it led: proof-similarity is functorial; the object is a Comb (Poly + holes)

Both cheap text routes are now eliminated — **EXP-3** (bigger model) and **EXP-3b**
(text-context). The deep reason: **proof-similarity is a *compositional* property** (how
method-interfaces chain), and text embeddings are non-compositional. The right unit is the
**whole-proof shape**, and that shape is exactly a **comb**: a wiring of method-boxes with
**holes at the sorry/obligation positions**. Read from our 4 fixtures:

- **Holes cluster at the creative moves, not the mechanical ones** — ○ sits on
  `construct-*`/`quotient`/`local-to-global`/`epsilon-of-room`, never on
  `reduce-to-known`/`estimate`/`unfold`. The hole-pattern marks *where the proof does real
  work*; it is a structural signature.
- **Shape-families recur across domains with no shared vocabulary** —
  `a96J04` = "unfold→unfold→unfold→bound→ε-of-room" (the generic ε-δ shape);
  `a93J05` = "construct a domain → exploit a symmetry → discharge to a known theorem".
  Those macro-shapes are invisible to text and visible to comb-matching.

**This is not greenfield — the objects already exist across three layers:**
1. **Lean (formal): `~/code/mathlib4/DarkTower/`** — `Comb.lean` defines a comb as a
   **morphism of polynomial functors = dependent lens** (`onPos` forward, `onDir`
   backward), with identity + composition + **associativity proven** (so combs form a
   category — the functoriality we argued for is already there); grounded in Niu–Spivak
   Poly (arXiv:2312.00990) and Roman's comb diagrams (arXiv:2004.07353). `TypedHole.lean`
   adds a `PFunctor` + **satiety grading** of positions (`parse/payoff/canon/bundling/
   role`) — the formal version of "holes typed by what obligation they expose".
   `Fill.lean` / `Discharge.lean` are the plug-a-hole / close-to-known moves. *(The n-hole
   comb proper — `⟨A;-;B;-;C⟩`, the open-diagram/coend layer — is flagged "not built yet"
   in `Comb.lean`.)*
2. **The 64 CT-concept ↔ iching pattern library: `~/code/futon3/library/iching/`** —
   Comb & Poly are among the 64 basic CT concepts, each a hexagram flexiarg with a
   `@ct-interpretation`. This is the bridge from the formal object to the
   pattern/xenotype/sigil system.
3. **futon5 wiring DSL: `src/futon5/wiring/` (compose, features, runtime) + `ct/dsl.clj`**
   — the executable wiring-diagram substrate.

## Next direction (EXP-3c) — shape-matching, not per-step

Reframed ladder: EXP-3 (size) ❌ → EXP-3b (text-context) ❌ → **lesson: similarity is
functorial over the comb** → next probe is **comb-shape matching**. Per-step
LLM-abstraction ("what method is this step instantiating?") is **not** a rival approach —
it becomes the **box-typing feeder** that labels each comb node; the matching then happens
on the *composite*.

Concrete, laptop-doable:
1. **Type the ~12 method interfaces** (consumes/produces) — small hand table, expressible
   as `PFunctor` positions/directions à la `DarkTower/TypedHole.lean`.
2. **Represent each proof as a typed comb** (boxes + wires + hole-positions/satiety) and
   test whether **comb-similarity clusters the proofs** where step-bags and text do not.
   Build on the futon5 wiring DSL / the DarkTower types rather than a new encoder.
3. Only *then* pick an architecture; any learned encoder must be **functorial** (composite
   of proofs ↦ composite of representations), which neither text embeddings nor a vanilla
   pooled GNN guarantee.

**Honest caveats 🔒** — this is **4 proofs**: the shape-families are suggestive, not
established (wants tens of proofs). Mapping IATC graphs to genuine Poly objects needs the
method-interfaces *defined* (design work, ~12 methods). And we have **eliminated text**;
we have **not** positively shown comb-shape recovers the 3 — EXP-3c is the first probe
that must prove a *positive*.

## Status

EXP-3 + EXP-3b **complete** (results above). Runner/payload + the `--context` extension
shipped. EXP-3c (comb-shape matching) is the named follow-on; the Comb/Poly/TypedHole
substrate it builds on already exists in `DarkTower/` + `futon3/library/iching/` +
`futon5/wiring`. Artifacts: `data/exp-20260618/bge-cas-sel-3b/` (EXP-3),
`data/exp-20260618/exp3b-context/` (EXP-3b).
