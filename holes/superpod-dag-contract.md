# Superpod DAG contract — phase-completeness for the scaled run

*claude-1 + Joe, 2026-06-22. The superpod is the scale-up target for the adapted
proof-structure pipeline: **LLaMA-only** (no Claude/Codex in the loop), many GPUs,
a whole-archive batch instead of a 200-paper slice. This contract exists for ONE
reason the Linode run made concrete — **so we don't miss phases.** Per-stage detail
(consumes/produces/gates/halts) lives in [`linode-stepper-contract.md`](linode-stepper-contract.md);
this doc adds the DAG-integrity layer on top of it and the superpod compute notes.*

## Why this contract (the mark5 lesson)

**IF** the run's output is to be corpus-faithful, every required phase must run for
*this* corpus — above all **S2 concept-substrate** (the corpus-aggregate barrier
that models the noun concepts S6 grounds against).

**HOWEVER** the stepper runs whatever `--from/--to` it's handed and never verifies
upstream phases happened. On the 2026-06-22 mark5 run we went S1 → S3 directly on
the box and **silently skipped S2** — reaching the point where S6 would ground
against a *stale, prior-corpus* `concept-index.json` (3,623 concepts, built Jun 18),
with **no coverage number taken** and nothing flagging it. A missed phase reads as
success.

**THEN** make the DAG enforceable: explicit `:depends-on` edges, a per-run **phase
ledger** that records each stage's completion + gated output + which corpus it
pertains to, and a stepper precondition that **refuses a stage whose upstream phase
has no ledger entry for this run** — the concept substrate (S2) **required
corpus-fresh** (never substituted from a prior/pre-planned vocabulary), and reuse of
any other artifact valid **only within the same corpus**.

**BECAUSE** silent skipping degrades the output invisibly (concept model not
corpus-faithful; no G-coverage) — the exact failure mode the gates exist to prevent.
"Good enough to reuse" must be a recorded decision, not an accident.

## Phase 1 (per-paper) vs Phase 2 (cross-paper) — the unit correction

A second mark5 lesson (Joe, 2026-06-22): the run was **per-paper-isolated end to end**,
which is wrong twice over. The architecture has two phases with *different units*:

- **Phase 1 — per-paper extraction.** Each paper → a **unified paper-level graph
  (target object "B")**: definitions and theorem/lemma statements are nodes, each
  proof is a substructure hanging off the statement it proves, and expository prose is
  the connective tissue/edges. Mine the **whole paper, including expository sections** —
  not one proof. This **replaces** the one-passage `mark3_extract_candidates` selector
  (a mark3 *demo* artifact: 1 candidate ≡ 1 paper ≡ 1 proof, which threw the paper
  away — see `mark5-ct100-results.md` D8). "Per paper" is *only* a Phase-1 word.

- **Phase 2 — cross-paper, holistic (the actual target).** The concept substrate spans
  the whole corpus; structure embeddings, retrieval, and comprehension operate **across
  all papers** (e.g. clustering proofs by method across topics). The per-paper Phase-1
  graphs are merely the **input nodes** to this cross-paper structure. Phase 2 never
  reasons about a single paper in isolation.

Consequence for the stages below: S1 produces the **whole-paper graph (B)**, not a
single-proof candidate; S3 reconstructs **every** proof substructure in the paper; S2,
S6, S7 are **Phase-2 cross-paper** stages whose unit is the corpus, not the paper.

## The DAG (explicit dependencies)

```
S0 provision
  └► S1 anatomy
        ├► S2 concept-substrate ──────────────┐         (BARRIER: needs all marks)
        ├► S3 iatc-extract ─► S4 clean ─┬─► S7 embed ─► S8 export
        │                               └─► (S9 pass-3)
        └► S5 strategy-recognition ─────────────┤
                                                ▼
                              S6 comprehension  ◄── depends-on {S2, S3, S5}
                                                └► (S9 pass-3)
```

The load-bearing edge is **S6 `:depends-on` S2** — comprehension cannot run without
the corpus substrate. Enforcing it is what makes the mark5 skip impossible.

## Phase ledger (the no-missed-phase mechanism)

Each run has a `:run-id` and a `:corpus-id` (the manifest hash / batch name). On a
stage's gate-PASS the stepper appends a ledger entry:

```edn
{:stage :S2 :run-id "mark6-mathCT-full" :corpus-id "math-ct-4616@2026-06-23"
 :output "data/warp/concept-index.json" :fingerprint "sha256:…" :n 4616
 :gate {:G-coverage :pass :curve [0.41 0.78 0.93]} :reused false :ts "…"}
```

Ledger lives at `data/runs/<run-id>/phase-ledger.edn`. It is the single source of
truth for "what has actually run for this corpus."

**Completeness check (stepper precondition).** Before stage *N*, for every `dep` in
`N.:depends-on`: require a ledger entry with the **same `:corpus-id`** and a passing
gate. If absent → **refuse** (exit nonzero, name the missing phase). This is the
`precondition-refuse` discipline extended across the DAG, not just per-stage inputs.

**Reuse = same-corpus only; emergent concepts are never substituted.** The
completeness check passes a dep if its artifact was built for **this `:corpus-id`**
(any run-id — that covers a genuine resume after a crash). What is *not* allowed is
substituting an artifact built from a **different** corpus. For **S2 specifically
this is a hard refusal, not a declarable option**: the concept substrate is
*discovered from the corpus being processed*, so a prior/pre-planned concept-index
is simply the wrong concepts — `--reuse` cannot launder it in. The mark5 reuse of
the Jun-18 index was a slice shortcut and is **explicitly invalid at superpod
scale**. (`--reuse <stage>=<path>` exists only to point at a same-`corpus-id`
artifact in a non-standard location, e.g. a recovered ledger — never to cross
corpora.)

### Emergent concepts vs. the controlled lens (don't conflate them)

Two "vocabularies" live in the pipeline and they are opposite in kind:

- **Concept substrate (S2) — EMERGENT, corpus-intrinsic.** The nouns/objects are
  *discovered* from this corpus. Must be built fresh per run; never reused from a
  pre-planned set. This is the data.
- **Method/macro vocabulary (`clean-method-vocab.edn`, S4 G-method-vocab) —
  CONTROLLED, pre-planned by design.** The 12 iching-derived method tags + 5
  macro-shapes are the *lens* we classify proofs through; reusing them across runs
  is correct (a fixed tagset keeps embeddings comparable). If the lens needs to
  grow for a new domain, that is a **deliberate, reviewed vocabulary-evolution
  step** — also never a silent reuse-vs-skip.

## Superpod compute specifics (differs from Linode)

| stage | Linode (mark5) | Superpod |
|-------|----------------|----------|
| **S0 provision** | 1× StackScript box, vLLM 70B-AWQ on 4 GPUs | cluster alloc (SLURM/queue), LLaMA served across many GPUs; bootstrap is the cluster job, not a StackScript |
| **S2 substrate** | reused prior index (the skip) | **must run fresh at archive scale** — the barrier most at risk of being cut for time; G-coverage inline is mandatory here |
| **S3 iatc-extract** | 70B-AWQ, ~16 s/paper, 4 GPUs | LLaMA at batch concurrency; per-paper loop unchanged but parallelised across workers; **HTTP-error skip + resume are load-bearing at scale** (mark5 crashed on one 400) |
| **S4 clean** | `clean_box_typing.py` served-70B | same producer, LLaMA endpoint; **fully automatic — no human/Claude in the loop**, which is why the controlled-vocab gate + cyclic-reject matter more |
| **S5–S9** | CPU on the box | CPU workers; S2/S6 are the corpus-aggregate stages that must see the whole batch |

LLaMA-only means S3/S4 cannot lean on Claude/Codex to repair bad output — the
gates (G-substance, G-method-vocab, G-cyclic) and the loop's error-tolerance are
the only safety net. They are now load-bearing, not advisory.

## Machine-readable DAG (the stepper reads this for completeness)

```edn
{:pipeline :superpod-proof-structure
 :inherits "linode-stepper-contract.md"          ; per-stage consumes/produces/gates
 :discipline {:executor-is-gate true :precondition-refuse true :finals-only true
              :non-fatal-eval true :resumable true :supervised-halts true
              :dag-completeness true :reuse-same-corpus-only true
              :emergent-stages-never-substituted [:S2]}
 :phase-ledger {:path "data/runs/<run-id>/phase-ledger.edn"
                :key [:stage :corpus-id]
                :entry [:stage :run-id :corpus-id :output :fingerprint :gate :reused :ts]
                :rule "before stage N, every dep in N.:depends-on must have a passing
                       ledger entry for the same :corpus-id, else REFUSE (or --reuse)"}
 :units {:p1 :per-paper :p2 :corpus}   ; Phase 1 mines each paper; Phase 2 reasons across all papers
 :dag
 ;; STAGE SEMANTICS MUST MATCH linode_stepper.OPS. They did not until 2026-08-07:
 ;; this block still carried the pre-2026-06-23 numbering (S4 clean, S5 strategy,
 ;; S6 comprehension, S7 embedding) while the runner used the corrected one, so
 ;; load_deps() proved completeness of a DAG the runner never executes — S5
 ;; depended only on S1, S6 omitted its expository input, and S10-S12 had no
 ;; dependencies at all. Current semantics: S3 IATC | S4 expository |
 ;; S5 comprehension | S6 paper-graph | S7 CLean+embed | S8 export | S9 APM |
 ;; S10 lexicon+reground | S11 structural canon | S12 accretion sweep.
 [{:id :S0  :depends-on []             :required true  :phase :p1}
  {:id :S1  :depends-on [:S0]          :required true  :phase :p1
   :note "deterministic anatomy: marks over the LaTeX source. Every later per-paper stage reads these."}
  {:id :S2  :depends-on [:S1]          :required true :barrier true :must-be-corpus-fresh true :phase :p2
   :note "concept-substrate — EMERGENT from this corpus; NEVER reused from a prior/pre-planned vocabulary (mark5 wrongly reused a stale index). No --reuse for S2: a foreign corpus-id is a hard refusal."}
  {:id :S3  :depends-on [:S1]          :required true :compute :gpu-llm :phase :p1
   :note "reconstruct EVERY proof substructure in the paper (not one selected passage). Finals only: *.rung2.edn are reports, not graphs."}
  {:id :S4  :depends-on [:S1]          :required true :compute :gpu-llm :phase :p1
   :note "expository scopes over the same anatomy; parallel to S3, not downstream of it."}
  {:id :S5  :depends-on [:S2 :S3]      :required true  :phase :p2
   :note "comprehension = R2d (needs the S2 substrate) (+) rung-3 (built from S3 graphs by cas_segment -> rung3_technique). CROSS-PAPER: corpus-relative."}
  {:id :S6  :depends-on [:S2 :S3 :S4]  :required true  :phase :p2
   :note "LOAD-BEARING EDGE on S4: the whole-paper object is proofs AND exposition AND concepts. Output belongs in the RETRIEVE path (data/iatc-paper-graphs/<run-id>)."}
  {:id :S7  :depends-on [:S3]          :required true :compute :gpu-llm :phase :p2
   :note "CLean typing then structure embedding; retrieval is CROSS-PAPER (cluster proofs by method across topics)."}
  {:id :S8  :depends-on [:S7]          :required true  :phase :p2}
  {:id :S9  :depends-on [:S3 :S7]      :required false :optional true :phase :p2
   :note "APM coverage + pass-3 hole harvest; both must be run-scoped, not global-tree."}
  {:id :S10 :depends-on [:S3 :S4]      :required true  :phase :p2
   :note "move lexicon from S3 graphs; expository reground over S4 scopes. Persist the lexicon — it is the answer to learning goal #2."}
  {:id :S11 :depends-on [:S2 :S7]      :required true  :phase :p2
   :note "definition canon from the S2 substrate's snippets; paper signatures from the S7 embedding."}
  {:id :S12 :depends-on [:S3 :S10]     :required true  :phase :p2
   :note "accretion sweep over S3 graphs using S10's harvested cues; the curve is the sweep's product and must land in the run dir."}]
 :gate-registry :inherited            ; see linode-stepper-contract.md (G-coverage … G-entropy)
 :superpod {:compute :llama-only :no-claude-codex true
            :s2-fresh-required true
            :s3-loop {:http-error-skip true :resume-from-existing true}
            :provision :cluster-alloc}}
```

## Status / next

- **Contract: set up (this doc).** The DAG, the phase-ledger spec, the
  completeness rule, and the explicit-reuse rule are defined.
- **Enforcement WIRED (2026-06-23):** `linode_stepper.py` is now the scale-aware runner
  (`--profile linode|superpod`): phase-ledger (`--run-dir`, keyed by `--corpus-id`),
  completeness refusal on missing upstream, `--reuse` (never S2), `--mark-done` for
  cluster stages. Verified: superpod `--plan` + S6-blocked-without-S2 + S2-reuse-refused.
  *(superseded next-step note:)* (1) emit phase-ledger entries on
  each gate-PASS keyed by `:corpus-id`; (2) a `--verify-dag` / precondition that
  reads `:depends-on` + the ledger and refuses on a missing upstream phase;
  (3) the `--reuse <stage>=<provenance>` flag. Until wired, the contract is the
  checklist the operator runs by hand (and the mark5 skip is documented so it
  isn't repeated).

*Cross-refs:* [`linode-stepper-contract.md`](linode-stepper-contract.md) (per-stage
detail + gate registry), `mark5-run-playbook.md` (the validated Linode sequence),
`E-comprehension-foundation.md` (why S2 grounds S6).
