# E-mining-qual-loop — in-run quality surveys, a CT qualifying exam, and the bridge to memory

**Opened:** 2026-08-05 (Fable session; HEAD = Joe, same day). Status: design
note — nothing dispatched. Siblings: `E-superpod-hardening.md` (runner
defects), `futon1b/holes/M-xtdb-22x-benchmarking.md` (the store this feeds),
`futon3c/holes/labs/M-diagramprover/capability-proof-apm.tex` (the warrant
apparatus this borrows).

## HEAD (Joe, 2026-08-05)

Zone makes the full mining run possible without Rob; if it takes ~2 weeks,
so be it — "we've sat waiting longer than that, and what he would have run
would have been flawed anyway." QA needn't wait for the end — better if it
doesn't: **an agent-in-the-loop that every ~100 papers (escalated: every 10
on the way to 100) does a quality survey and *improves behaviour***, where
quality means things like "what kinds of questions can I now answer easily
that I couldn't before." The papers + datamined contents are starting to
look like an **unassimilated memory-bank**. There are no prelim problems in
category theory — so **write our own qualifying-exam problems from the
top-100 papers** and see whether a Zai agent who has NOT read the papers
can answer & formalise them from the datamined contents alone, as in APM.
"The bridge between datamined contents and memory system warrants some
reflection."

## 1. The capability metric is an accretion curve — S12 already has the slot

"What can I answer now that I couldn't before" is exactly the shape of the
mark7 accretion sweep (S12 checkpoints every tier metric at log-spaced n).
Joe's proposal upgrades the *metric family*: from coverage/compression
(internal) to **question-answering capability (external)**. The escalating
cadence (every 10 → every 100) is just log-spaced checkpointing said aloud.
So the survey slots into the existing design rather than beside it: an
exam-battery evaluation at each S12 checkpoint, curve = exam score vs
papers-mined.

## 2. The one hard constraint: version the behaviour, don't blur the curve

"Survey and improve behaviour" mid-run creates corpus heterogeneity: papers
1–10 mined under behaviour v0, 11–20 under v1, … Then the accretion curve
confounds corpus growth with behaviour drift, and pass-discipline (full
pass per stage before refinement) is silently violated. The reconcile —
straight from APM's N8 (lessons travel via *versioned packet templates*):

- lessons from a survey are **banked at the checkpoint and applied from the
  next batch**, never mid-batch;
- every artifact is **stamped with the behaviour version** that produced it
  (a `:behaviour/version` field in the graph EDN → an ingested column);
- curves are reported **segmented by version**; a version bump is a labeled
  regime marker, not noise.

With that, the in-loop improvement is honest DERIVE-time refinement, and
the eventual "official" pass can rerun early papers under the final
behaviour if the deltas matter (resumable loop makes that cheap: delete the
finals you want re-mined).

## 3. The CT qualifying exam (the external metric)

No CT prelims exist, so we author them. Design, with the load-bearing
control:

- **Authoring:** exam problems drafted from the top-100 papers (by an agent
  or Joe with source access). Leakage is fine *on the authoring side*; the
  discipline is that the **answering agent (Zai seat) never sees the
  papers** — only what it can retrieve.
- **Three arms, or the result is uninterpretable:**
  1. **mined-contents arm** — answer/formalise from the datamined store
     (graphs, marks, scopes, concept spine) via retrieval;
  2. **raw-text arm** — same agent, retrieval over raw eprint text
     (the control that tests whether mining *adds* anything);
  3. **closed-book arm** — no retrieval (floor; the model knows textbook CT).
  The claim worth having is arm-1 > arm-2 > arm-3 with the gap between 1
  and 2 attributable to the mining — an identified contrast, not a vibe.
- **Grading:** formalisation goes through the existing APM machinery (Lean
  gate: compile, axiom audit, statement hash — mechanical warrants);
  informal answers need a rubric + judge panel, warrant-typed accordingly.
- **Double duty:** the retrieval queries the answering agent issues against
  the bench store ARE the benchmark's query ladder under real load —
  M-xtdb-22x-benchmarking gets its workload trace for free.

## 4. The bridge: datamined contents ≠ memory, and APM already named why

"Unassimilated memory-bank" is exactly right, and APM's N5 is the warning
label: agents reliably *ask* a store; whether the store *answers* is a
separate property that failed there through a four-layer anatomy
(propensity / framing / affordance / index-reach). Dumping graphs into
XTDB gives us index-reach only. Assimilation = giving mined artifacts the
affordances the memory system's retrieval stage expects: tags carrying the
asker's vocabulary (APM's hunger-audit lesson), typed entries, recency/
authorship structure, ranked text recall (the #5637 thread!). The
full-extraction ingest (mission next-step 3) is the substrate move; the
exam's arm-1 failures then localise *which* layer of assimilation is
missing — that is the reflection Joe asked for, made operational.

## 5. Timeline honesty (Zone, CPU, 2-way shard)

S3 ≈ 6.5–7 days; S4 expository (capped ~30 regions/paper) and S7
box-typing are the other LLM legs — order another 4–7 days combined at
~6.5 t/s aggregate; CPU stages are noise. **So "~2 weeks to 'I've tested
it and it's ready'" is the right expectation**, with exam checkpoints
(every 10 papers early) producing evidence continuously from ~day 1 —
the point of §1 is that we don't wait two weeks to learn something.

## Next actions (when picked up)

1. Add `:behaviour/version` stamping to the loop output (one-line change,
   before the first survey checkpoint so v0 is explicit).
2. Draft exam-battery v0: ~10 problems from the first 10 mined papers
   (they're the most-cited; good exam material by construction), with the
   three-arm protocol and grading rubric written BEFORE any arm runs
   (preregistration discipline).
3. Wire the first survey checkpoint at 10 papers-complete; output = exam
   scores per arm + a lessons list → behaviour v1 candidate (applied from
   paper 11+ only).
4. Full-extraction ingest (mission next-step 3) so arm-1 retrieval has the
   marks/scopes/enrichment layers, not just S3 graphs.

## 6. The strange loop: mine the causal-CT literature alongside (Joe, same day)

M-diagramprover's causal layer is DAG-Pearl today; its next layers are
**category-theoretic causal reasoning** — and that literature is itself
category theory on arXiv. So we mine it with the same pipeline: corpus
**`ct-causal-v0`, 22 papers** (`holes/mark7z-ctcausal.ids.txt`), assembled
2026-08-05 by manifest-abstract search + API-verification (no id taken
from model memory unverified): Fong 1301.6201 (Causal Theories); Cho–Jacobs
1709.00322; Jacobs–Kissinger–Zanasi 1811.08338 (string-diagram surgery);
Fritz 1908.07021 (Markov categories) + the Markov-categories line
(1912.02769, 2010.07416, 2105.02639, 2204.02284, 2204.04920, 2207.05740
(d-separation), 2211.02507, 2303.14049, 2308.00651, 2401.14669 (Bayes
filter), 2404.02017 (combs/causality), 2411.12840, 2312.09666 excluded →
kept 2501.18404 (causal intervention diagrams), 2512.24417 (causal Markov
category with Kolmogorov products)); plus 1406.6030, 1701.02547
(quasi-Borel), 2004.09999, 2201.08963. Spacetime-causality homonyms
(Lorentzian "causal" papers) explicitly excluded.

The mined output is a **roadmap of the domain the causal engine will
model** — and since the causal engine is (Joe) a superstructure for the
memory system, this corpus is first-class exam material for §3: its
questions are the ones the *system itself* will need answered. Ops:
eprints staged (20 were already local; 2 fetched + verified), S1+extract
running niced on Zone, IATC queued to auto-start when shard-a drains
(`run-ctcausal-when-free.sh` → `data/iatc-argument-graphs/run-ctcausal`,
own corpus-id, own out dir — no stream added, no bookkeeping blur).

## 7. Preview/think-through (Joe's questions, 2026-08-05 evening)

- **Memory store vs superpod = process vs product.** The store mines agent
  turns (informal, MiniPolymath/MathOverflow-like); the superpod mines
  polished papers. The measured 67% missing-warrant rate in papers is the
  quantified Lakatos point: publication deletes scaffolding; turns contain
  it pre-deletion. Holding instruments for BOTH halves is the edge.
- **Scribe upgrade (mark6-inspired):** emit IATC-typed records from turns
  (moves with transcript anchors, warrant-explicit inferences, retraction
  events via the γ lexicon, holes as demand records) → turns and papers
  share one schema, one store, one query ladder, one R6-style QA surface.
  Preregisterable prediction: warrant-capture on turn-mines ≫ 33%; if not,
  the Scribe is dropping exactly the process content that justifies it.
- **Fully-mined arXiv.CT value-prop (Rob):** first corpus-scale empirical
  account of how a field argues — move-lexicon size/convergence (S10),
  archetype census (S11), warrant-omission profile (= map of the field's
  shared implicit background), backbone saturation. CT is 1/124 of math.
- **What the 22 causal-CT papers give M-diagramprover:** concept spine
  (roadmap); gap map (their mined holes/missing warrants = what a
  formalization layer must supply first — no Mathlib for Markov cats);
  self-referential exam material.
- **Causal-CT beyond Pearl** (for capability proofs): on one finite DAG,
  nothing — d-sep/do() have faithful categorical reconstructions
  (Fritz–Klingler; JKZ surgery). The gains: (1) **compositional
  identification** — proofs about systems-of-systems from interface
  properties; (2) **higher-order/feedback** — the mining-qual loop itself
  (behaviour v(k+1) ← survey of v(k) outputs while the corpus grows) is a
  **comb**-shaped identification claim (2404.02017), the first of our
  capability proofs to structurally outgrow the APM DAG apparatus;
  (3) **transport as comparison of experiments** (2010.07416) — the
  theorem-shaped home of APM's N6; (4) typed variables + native
  deterministic/stochastic mixture (quasi-Borel) = causal-engine target-3
  semantics. One line: Pearl suffices for any single experiment; the
  architecture needs the categorical layer.

## Log

- 2026-08-05 — opened from Joe's framing; no code yet. Shard state at
  writing: S3 2-way, ~10 finals done.
- 2026-08-05 (later) — §6 added: ct-causal-v0 corpus assembled, staged,
  queued behind shard-a.
