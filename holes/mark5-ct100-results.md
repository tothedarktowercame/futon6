# mark5 — CLean/IATC structure-embedding run on ~100 math.CT proofs

*claude-1 + Joe, 2026-06-22. Tech note: the first real end-to-end run of the adapted
proof-structure pipeline (IATC → CLean → structure embeddings) on a live GPU box, over
a 100-proof slice of math.CT. Written to record what ran, what the artifacts show, and
— per Joe — **what the deficiencies are**. Evidence-first: every claim below is a count
from the run. Artifacts: `data/mark5-ct100-run/` (gitignored). Playbook:
[`mark5-run-playbook.md`](mark5-run-playbook.md). DAG: [`superpod-dag-contract.md`](superpod-dag-contract.md).*

## 1. What we ran

- **Box:** one Linode `g2-gpu-rtx4000a4-s` (4× RTX 4000 Ada, 80 GB aggregate, 32 vCPU),
  us-ord, StackScript 2142757, vLLM serving `Meta-Llama-3.1-70B-Instruct-AWQ-INT4`
  (TP=4, ~19.5 GB/card) as `mark4-70b`. ~1.5 h wall, ~$5. Claude drove `linode-cli`
  create→run→delete via `pass` (token never in transcript).
- **Corpus:** an unbiased even-spread-by-date draw of **200 primary math.CT papers**
  (`holes/math-ct-200.ids.txt`), pre-staged on dev to **199 v2-enriched candidates** —
  `mark3_extract_candidates` selects **one proof passage per paper** (1 candidate ≡ 1
  paper ≡ 1 proof; see §3 D8). The run was **capped at ~100** by operator choice
  ("enough to explore the slice").
- **Stages executed:** S1 anatomy (detector → marks → enriched candidates, on dev) →
  **S3** IATC reconstruction loop (70B) → **S4** CLean box-typing (70B) → **S7**
  structure embedding → **S8** export (neo4j cypher + pgvector SQL).
- **Stages NOT run:** S2 concept-substrate (reused a prior index — see §6), S5/S6
  recognition+comprehension, S9 mining.

## 2. Results

**S3 — IATC reconstruction (70B).** 106 proof graphs produced — **one proof per paper**,
so 106 graphs = 106 distinct papers (see §3 D8). First-batch yield
**28 pass / 2 fail (93%)**; pass effort 23 first-try / 3 / 2 (the retry loop helped 5).
The 2 fails were legit (one unparseable EDN, one substance-gate fail), not crashes.

**Structural distribution (43-graph mid-run slice).** nodes 3–14 (median 9), infer-edges
1–9 (median 4), holes 0–9 (mean 1.5) — real spread on every axis. **33/43 distinct
(nodes,edges) shapes**; most-common shape 4×. Spot-checked node texts are genuine
paper-specific math (accessible-categories / λ-pure-subobjects; nerve cohomology
`Ner_n S`, `C^n(S,D)`), not boilerplate. → **graphs are distinct reconstructions, not
templated shells** (the anti-gaming check passes).

**S4 — CLean box-typing (70B).** **102 typed / 4 cyclic-rejected (logged) / 0 failed**;
G-method-vocab PASS (all `:method`/`:macro` in controlled vocab).

**S8 — export.** 552 nodes, 285 edges, 75 theorems → `clean-graph.json`, `load.cypher`,
`pgvector.sql`.

## 3. Deficiencies (the honest part)

**D1 — Macro-vocabulary collapse (the headline; G-entropy RED).**
At 102 proofs the 5 macro-shapes collapsed: **98/102 = `construct-exploit-discharge`**
(3 contradiction-reduce, 1 induct-tower). macro-entropy 0.17 (floor 0.5); mean
off-diagonal structure cosine 0.82 (ceiling 0.85). The G-entropy gate **failed**.

**The cause is OPEN — we have not looked at a single proof's detail.** What we observe
is a per-proof macro collapse in *this* run; the *why* has at least two candidate
explanations, both consistent with prior design discussions and neither yet checked:

1. **Wrong unit (paper-level lens applied per-proof).** The 5 macro-shapes may be
   *whole-paper* abstractions; applied to a single proof (mark5's one-proof-per-paper
   error, D8) every proof reads the same, while a paper's macro arc (multiple proofs +
   the expository sweep) would not. If so the collapse is a **framing artifact**, not a
   vocabulary defect. *(Do NOT conclude "iching shapes are too coarse for CT" — unsupported.)*
2. **Fixed vocab, not data-driven.** The embedding + box-typing run against the
   *hand-authored* `clean-method-vocab.edn`. The **data-driven vocabulary machinery**
   the adapted (lavender) layer specifies — STRAT-REC co-learning on math.CT, LEAN-NL/
   Herald, WARRANT-NORM's (type,concept) keying — is **not wired into this stage**. A
   weak fixed vocab here, where a co-learned one was intended, would look exactly like this.

The 9-proof demo (macro-entropy 0.77, PASS) **oversold** regardless — too few proofs,
and hand-lifted to be macro-diverse.

**Diagnostic 1 (2026-06-22, CPU on the 102 in hand): vocab bottleneck, not proof-sameness.**
Unsupervised clustering of the 102 **method-bags** (the signal independent of the
collapsed macro): **28 distinct method-sets**; the 98 proofs the fixed macro lumped as
one class **split into method-distinct groups** (compute-invariant-led · construct-
auxiliary-*pure* · transport-along-symmetry-heavy · reduce-to-known-led). So the fixed
macro **discards real distinctions at the proof level** → candidate (2) [fixed vocab] is
supported over "proofs are identical." **Caveat:** clusters are *weak* (silhouette ~0.30,
peak 0.32 @k=8) — a graded continuum, not crisp classes; so the actionable signal is the
**method-composition vector itself (reweighted up)**, and a *discrete* data-driven macro
vocab may not buy much over using the composition directly. **Candidate (1) [paper-level
lens] remains untestable here** — mark5 is one-proof-per-paper; needs the whole-paper run.

**Diagnostic 2 (2026-06-23, CPU on the 102): the macro is OVER-TAGGED, and the vocab has
gaps.** Cross-tab of macro × dominant-method: the macro is `construct-exploit-discharge`
*independent of method* — construct-auxiliary 48/48, reduce-to-known 40/44, transport
6/6, compute-invariant 4/4. Annotated proofs confirm it: `1806.08645` is **all
transport-along-symmetry** yet tagged construct-exploit-discharge (no construction in it).
So the 70B's macro judgment is a near-constant **default** (over-tagging). Prototype fix —
derive the macro from the method composition instead of the model: with the *existing* 5
shapes it stays low (no transport/compute shape; construct-aux + reduce + transport all →
construct-exploit-discharge), but **adding transport/compute shapes lifts macro-entropy
0.17 → 0.42**. → **Two causes:** (1) over-tagging (the model defaults), (2) vocab gap (the
5 shapes don't span the common CT methods). **Fix (pre-run, CPU):** derive macro from
method-composition + **grow the macro vocab data-drivenly** (transport-symmetry, compute)
from the method distribution; re-derive macros on the 102 and re-check G-entropy (the
EXP-3-at-scale re-test). The methods discriminate; the macro *assignment + vocab* are the
fixable problem — NOT "iching shapes too coarse."

**FIX LANDED (2026-06-23).** Grew the macro vocab +3 shapes (`transport-symmetry`,
`reduce-to-known`, `local-to-global-glue`); `derive_macro()` (method-composition → macro,
`clean_macro_fix.py`) wired into `clean_box_typing` so the macro is **derived, not
70B-tagged**. On the 102: **macro-entropy 0.08 → 0.73** (clears the 0.5 floor), 56/94
re-macro'd, balanced 6-macro distribution. Side-by-side old-vs-new at
`data/showcases/macro-fix-comparison.html`. The structure-embedding's macro layer now
discriminates; G-entropy collapse resolved before the live run.

**D2 — Discrimination is method-level, not macro-level (and the embedding is mis-weighted).**
The 12-tag *method spine* stayed diverse — 10 tags fire: reduce-to-known-result 170,
construct-auxiliary-object 163, transport-along-symmetry 69, compute-invariant 20,
local-to-global 13, quotient-by-irrelevance 6, argue-by-contradiction 5,
induct-up-a-tower 2, cover-and-estimate 1, count-by-decomposition 1 (avg 4.4/proof).
So the 70B *does* discriminate — at the method layer — but the 33-d vector is dominated
by the (collapsed) macro + comb scalars, so overall discriminativeness is weak
(off-diag 0.82). The embedding should **reweight toward the method spine**.

**D3 — Method distribution is top-heavy.** Two tags (reduce-to-known 170,
construct-auxiliary 163) are ~74% of method occurrences; the tail (cover-and-estimate,
count-by-decomposition) fires once. Real but skewed — retrieval will be dominated by
the two head methods.

**D4 — Concept modelling (S2) was skipped; comprehension (S6) untested.**
The run reused a **stale prior `concept-index.json`** (3,623 concepts, built Jun 18 from
a different corpus), took **no G-coverage measurement**, and never ran S6. So the
"comprehension floor" (noun-grounding ⊕ strategy) is **unvalidated on this corpus**. Per
[`superpod-dag-contract.md`](superpod-dag-contract.md) this reuse is invalid at scale —
S2 concepts must be discovered from the corpus being processed.

**D5 — rung2 reasoning gate is uncalibrated.** Nearly every accepted graph logged
"rung2-soft-fail": the semcheck (anchor-faithfulness floor 0.3, warrant-resolution
floor 0.0) runs **report-only with uncalibrated thresholds** ("report-only until
calibrated"). It currently certifies nothing.

**D6 — 70B output fragility.** Three malformed-EDN classes broke the strict Python
loader mid-run (each passes the lenient `bb` reader): apostrophe-in-keyword (`:phi'`),
bare-keyword `:warrant`, infer-edge missing `:conclusion`. Plus a loop crash on an
uncaught vLLM HTTP 400. All fixed defensively, but it means **the producers need
hardening against LLM output before the superpod run** (where there's no Claude/Codex
to repair).

**D7 — Slice, not corpus.** 100/199 of a 200-paper *sample*; not full math.CT
(~4,616 papers). Findings are indicative, not archive-scale.

**D8 — One proof per paper.** `mark3_extract_candidates` selects a *single* proof
passage per paper (the chosen proof-move window), so each paper is represented by
exactly one argument graph: 199 candidates = 199 papers = 199 proofs (1:1), and the
106 graphs are 106 distinct papers. A CT paper's *other* proofs and lemmas are **not
captured**. "Proof graph" and "paper" are interchangeable counts in this run *only*
because of this one-per-paper selection — at scale, capturing more proofs per paper
is a real extraction-coverage axis we haven't touched.

## 4. What is solid

- **The pipeline runs end-to-end at scale** (S1→S4→S7→S8) on a real GPU box.
- **Per-proof yield is high** (93%) and **structures are distinct + source-faithful**
  (33/43 shapes; real CT content) — not gamed shells.
- **The method spine is diverse** (10/12 tags) — the recognition layer captures real
  method variety.
- **The gates have teeth:** G-entropy caught the macro collapse; argcheck, G-method-vocab,
  and G-cyclic all fired correctly. The run failed *loudly* where it should.
- **Resumability + error-tolerance proven** under live failure (the loop crashed and
  resumed from existing graphs).

## 5. Bugs found + fixed (live, committed)

| commit | fix |
|--------|-----|
| `cfdaeb5` | `iatc_to_clean`: parse EDN with apostrophe-keywords (CT primes) |
| `1b1040f` | `mark3_iatc_loop`: survive vLLM HTTP errors + resume from existing graphs |
| `9bf22ef` | `iatc_to_clean`: tolerate bare-keyword `:warrant` |
| `80a3f4d` | slash-safe ids (old-style `math/NNNN`) — recovered 4 papers |
| `b0f3d85` | skip malformed infer-edges (missing `:conclusion`) + per-graph error isolation |
| (S4 glob) | `clean_box_typing`: skip `.rung2.edn` sidecar reports |

## 6. Recommendations / next steps

1. **Diagnose the macro collapse before "fixing" it (D1):** look at actual per-proof
   structures — (a) do the macro-shapes vary at the *paper* level (collapse = a per-proof
   framing artifact, not coarseness)? (b) what would a **data-driven** vocabulary give
   here — i.e. wire the adapted-layer machinery (STRAT-REC co-learning on math.CT,
   LEAN-NL, WARRANT-NORM keying) into the typing/embedding instead of the fixed
   `clean-method-vocab.edn`? Only after looking decide whether to reweight toward the
   method spine and/or grow the vocab. **Do not assume the iching shapes are "too coarse."**
2. **Run S2 corpus-fresh + S6** on a real slice to validate the comprehension floor and
   take the G-coverage curve (the thing this run skipped).
3. **Harden the producers** (D6) against LLM output ahead of the LLaMA-only superpod run.
4. **Calibrate rung2** (D5) so it certifies rather than reports.
5. **Then scale** per [`superpod-dag-contract.md`](superpod-dag-contract.md) (phase-
   completeness enforced) once the macro/embedding fix lands.

*Bottom line: the machinery works and the gates are honest — and at the proof level the
discriminative signal is **method-level, not macro-level**. WHY the macro layer collapses
is still open (paper-level lens applied per-proof? fixed vocab where a data-driven one was
intended?) — it needs *investigation of the details*, not an assumed "too coarse" verdict,
before the embeddings can be trusted for retrieval. Good to learn at 100 proofs and ~$5
rather than at archive scale.*
