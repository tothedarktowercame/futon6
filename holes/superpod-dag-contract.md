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
has no ledger entry for this run** — with reuse permitted only when *declared*
(recorded with provenance), never silent.

**BECAUSE** silent skipping degrades the output invisibly (concept model not
corpus-faithful; no G-coverage) — the exact failure mode the gates exist to prevent.
"Good enough to reuse" must be a recorded decision, not an accident.

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

**Explicit reuse (not silent skip).** To carry a prior artifact into a new run
(what we *did* with the substrate), pass `--reuse S2=<prior-run-id|path>`. The
stepper writes a ledger entry with `:reused true` + `:provenance` + a
`:corpus-faithful false` flag. Downstream still proceeds, but the output is *marked*
as not corpus-fresh, and reports say so. Reuse becomes a decision on the record.

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
              :dag-completeness true :reuse-must-be-declared true}
 :phase-ledger {:path "data/runs/<run-id>/phase-ledger.edn"
                :key [:stage :corpus-id]
                :entry [:stage :run-id :corpus-id :output :fingerprint :gate :reused :ts]
                :rule "before stage N, every dep in N.:depends-on must have a passing
                       ledger entry for the same :corpus-id, else REFUSE (or --reuse)"}
 :dag
 [{:id :S0 :depends-on []            :required true}
  {:id :S1 :depends-on [:S0]         :required true}
  {:id :S2 :depends-on [:S1]         :required true :barrier true :reuse-allowed true
   :note "concept-substrate — the phase mark5 skipped; reuse only via --reuse with provenance"}
  {:id :S3 :depends-on [:S0 :S1]     :required true :compute :gpu-llm}
  {:id :S4 :depends-on [:S0 :S3]     :required true :compute :gpu-llm}
  {:id :S5 :depends-on [:S1]         :required true}
  {:id :S6 :depends-on [:S2 :S3 :S5] :required true
   :note "LOAD-BEARING EDGE: depends-on S2 — cannot ground without the corpus substrate"}
  {:id :S7 :depends-on [:S4]         :required true}
  {:id :S8 :depends-on [:S4 :S7]     :required true}
  {:id :S9 :depends-on [:S4 :S6]     :required false :optional true}]
 :gate-registry :inherited            ; see linode-stepper-contract.md (G-coverage … G-entropy)
 :superpod {:compute :llama-only :no-claude-codex true
            :s2-fresh-required true
            :s3-loop {:http-error-skip true :resume-from-existing true}
            :provision :cluster-alloc}}
```

## Status / next

- **Contract: set up (this doc).** The DAG, the phase-ledger spec, the
  completeness rule, and the explicit-reuse rule are defined.
- **To wire into the stepper (next code step):** (1) emit phase-ledger entries on
  each gate-PASS keyed by `:corpus-id`; (2) a `--verify-dag` / precondition that
  reads `:depends-on` + the ledger and refuses on a missing upstream phase;
  (3) the `--reuse <stage>=<provenance>` flag. Until wired, the contract is the
  checklist the operator runs by hand (and the mark5 skip is documented so it
  isn't repeated).

*Cross-refs:* [`linode-stepper-contract.md`](linode-stepper-contract.md) (per-stage
detail + gate registry), `mark5-run-playbook.md` (the validated Linode sequence),
`E-comprehension-foundation.md` (why S2 grounds S6).
