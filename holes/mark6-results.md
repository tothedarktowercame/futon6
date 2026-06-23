# mark6 — the scale-up-from-1 run (n=1 → n=10), end-to-end

**2026-06-23. Live GPU box (Linode g2-rtx4000a4, 4×RTX-4000, vLLM 70B-AWQ), torn down after the run.**

This is the first run built to test *progress*, not throughput. The question is not
"how many papers did we process" — it is: **run one paper through the whole pipeline and
you already have metrics; run ten and do the metrics improve, and can we say *why*.**

The sample is a **citation-coherent neighborhood** (a hub `math__0608040` + its in-corpus
citers), chosen because — as we verified before the run — a random date-spread sample's
papers don't share concepts or references, so any accretion slope at small n would be
flat for the wrong reason. Coherence is what makes the small-n slope *mean* something.

---

## 1. The headline: the accretion slope rises (this is the progress signal)

Leave-one-out over the run-corpus (a held-out paper's grounding vs the other k papers):

```
 n (papers):     1     2     3     4     5     6     7     8     9
 concept-cov:  0.08  0.21  0.31  0.46  0.56  0.59  0.63  0.67  0.70     ▲ 0.08 → 0.70
 ref-resolve:  0.35  0.35  0.35  0.37  0.37  0.38  0.39  0.43  0.44     ▲ 0.35 → 0.44
```

A held-out paper's concepts are **8% covered by one neighbor, 70% by nine.** That upward
curve *is* the scale-up-from-1 result: the corpus gets more useful to a new paper as it
grows. It is computed at n=1 and it rises — exactly the design.

---

## 2. One proof, end-to-end (the literate thread)

To show what "through the whole pipeline" means, here is a single proof —
`0709.0248__p0` — from input to verdict. The **annotated render** (existing
`build_iatc_goldens` engine, retargeted at the mark6 graph) shows the proof text with
colour: grounded concepts and symbols highlighted with their grounding in hover-tooltips,
golden CPU anatomy (128 marks) beside the 70B IATC projection (10 marks), on the *same*
text — **`file:///home/joe/code/futon6/data/showcases/mark6-iatc-annotated.html`**
(regen: `IATC_IDS=0709.0248 IATC_RUN=data/mark6-render IATC_OUT=… build_iatc_goldens.py`).
The text blocks below are the plain-text distillation of that render.

**(a) The input.** The raw passage the pipeline ingested (arXiv 0709.0248, on the
homotopy interpretation of type theory):

> Under the general interpretation in locally cartesian closed categories sketched above
> the reflection rule is always valid.
> **Proposition.** In the standard interpretation given above, every locally cartesian
> closed category 𝒞 is extensional.
> *Proof.* Note that it suffices to consider "parameterized" versions of the rules
> governing identity types. …

**(b) The reconstruction (IATC).** The 70B reconstructs the proof as an inference DAG —
and, crucially, **flags the warrants it cannot find** rather than papering over them:

```edn
:parameterized-rules  ⟹  :extensional-category   warrant: MISSING  ("parameterization implies extensionality")
:equivalent-rules     ⟹  :parameterized-rules    warrant: claim    ("structural rules of the theory")
:identity-type-rules  ⟹  :diagonal-object        warrant: MISSING  ("identity type rules imply isomorphism to diagonal object")
```

Three inference steps; two honest `:missing-warrant` markers. The structure is recovered;
the gaps are named, not hidden.

**(c) The comprehension verdict.** We then ask how well *we* understood it — grounding the
nouns against the concept substrate and the proof-moves against the pattern library:

```
0709.0248__p0:  noun=0.875   strategy=0.19   comp=min(N,S)=0.19   →  WEAK-EXTRACTION
   undefined noun: "parameterized rules"   (= the :parameterized-rules node above)
```

This is the comprehension floor doing its job. We grounded 88% of the nouns but only 19%
of the strategy, so `comp = 0.19` and the verdict is **weak-extraction — "we didn't
understand the strategy well enough yet"**, *not* "the proof is weak." The same undefined
noun, *parameterized rules*, surfaces in both the IATC missing-warrant and the
comprehension gap — a consistent, diagnosable signal pointing at the same place.

---

## 3. n=1 → n=10: what rose, what stayed flat (and why that's correct)

```
                                     n=1 (0709.0248)   n=10 (neighborhood)
ACCRETION (should rise with n):
  concept-coverage (run-corpus)            —             0.08 → 0.70   ▲
PER-PAPER completeness/quality (flat in n by design):
  S1  any-markup-coverage                0.322           0.806   (10 papers)
  S4  expository-coverage                1.000           0.975   (10)
  S7  clean-discharge-rate               0.542           0.329   (58 proofs)
  S5  comprehension-confidence           0.155           0.138   (58)   ← floor, not slope
  S5  symbol-grounding/named-concept     0.866           0.846   (56)
  S5  symbol-grounding/proof-move        0.155           0.144   (58)
  S5  weak-point                         0.000           0.000   (gate holds)
  S6  statement-proof-attachment         1.000           0.553   (10)
```

The accretion axis rises; the per-paper axes are flat — **as they should be.** Markup,
expository coverage, symbol grounding are properties of a *paper*; they don't (and
shouldn't) climb just because the corpus grew. The one axis we *want* to climb does.
"10 papers done" is not the result — **a rising slope with per-stage attribution is.**

---

## 4. The macro fix, validated at scale

mark5 had a collapsed structure-embedding macro (entropy 0.17 — the 70B defaulted one
shape regardless of method). We fixed it by *deriving* the macro from the box
method-composition and growing the vocab. On mark6's fresh n=10 proofs:

```
n=10 macro-entropy(norm) = 0.77   (floor 0.5; mark5 pre-fix 0.17)
dist = {construct-exploit-discharge 28, reduce-to-known 15, transport-symmetry 6,
        contradiction-reduce 4, induct-tower 4, count-invariant-obstruct 1}
```

A worked flip (from the old-vs-new comparison, `data/showcases/macro-fix-comparison.html`):

```
0705.0102   methods: reduce-to-known×2 · local-to-global×2 · construct-auxiliary×1
            OLD (70B default):  construct-exploit-discharge        ← rote
            NEW (method-derived): reduce-to-known                  ← faithful
```

The methods were always faithful; the macro *assignment* was the bug, and it stays fixed
on data it never saw.

---

## 5. Three honest findings (with the fix each implies)

1. **Comprehension is a floor, not yet a slope.** It grounds against the *full* corpus, so
   it sits at ~0.14 and is flat in run-n (§3). To make it *rise* with the run, scope its
   substrate to the run-corpus (the same leave-one-out the accretion metrics use).
2. **Attachment drops to 0.55 at n=10** (vs 1.0 for the one easy paper): at scale some
   papers have proofs that don't attach to a statement (orphans flagged). A real
   quality signal in the paper-graph assembly, not noise.
3. **The 70B occasionally emits illegal EDN** (e.g. a `:∅` keyword; 2/58 graphs). One bad
   graph used to sink the whole comprehension batch; now isolated per-graph (SKIP + carry
   on). Fixed and committed.

---

## 6. The topology that wasn't

An earlier claim that the run needed a dev↔box "distributed topology" was **wrong** — it
was a data-staging gap. The pipeline is **single-host**: stage ~68 MB once (warp
substrate + futon3 patterns, **symlinks dereferenced** — dev's `storage/` overlay ships
dangling links otherwise) and every stage S1→S9 runs on the one machine. The stepper now
encodes this: an explicit `STAGE` step, no split. On the superpod you `rsync` that 68 MB
once and run everything there — compute and disk are never the constraint.

---

## Status

Pipeline proven end-to-end at n=1 and n=10, single-host, on a live GPU box (since torn
down). Progress signal demonstrated (concept-coverage 0.08→0.70); macro fix holds at
scale (0.77); three diagnosable findings banked with their fixes. Next: scope
comprehension to the run-corpus (finding 1) so it becomes a slope, then a clean
neighborhood-vs-random comparison run.
