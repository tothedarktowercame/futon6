# W6 — the CAS chain is not a wiring task

**Recommendation: do not wire `cas_select` / `cas_checks` / `cas_cert` into S5.**
The review classified CAS-SEL as "(b) needs wiring — the selector, registry,
checks, segmenter and certificate exist; OPS runs only the segmenter". Tested
against the 16-paper corpus, that classification does not hold: the components
exist but two of the three produce nothing on this data, and the causal model
does not include the layer they would add.

## What was tested

`cas_segment` is already wired (it produces the proof steps S5's rung-3 half
consumes). The question was the remaining three.

**`cas_select --backend stub`, over the run's 98 proof steps:**

| | |
|---|---:|
| papers processed | 16 |
| verified pattern matches | **0** |
| static checks emitted | **0** |
| topology entries | **0** |
| induce-queue entries | **136** (every one `"no candidate verified"`) |

The deterministic path is an oracle *stub*: it retrieves candidate patterns and
then verifies none of them. Every step lands in the induce queue. Wiring this as
a stage would add a stage that provably emits nothing but a list of its own
failures. Real verification requires the OpenAI backend — bounded (136 steps)
but a genuine model cost, and therefore a decision rather than a wiring change.

**`iatc_semcheck.bb`, the other input `cas_cert` reads** — fails on the run's
graphs:

```
FAIL 0705.0102__p0.edn
  R2? semcheck-load   FAIL rate=n/a
    - R2d concept coverage failed
  profile: terms=0 nodes=null edges=null holes=null
```

This is an integration defect, not a data problem: **R2d alone succeeds on the
same graph** (coverage 1.000, 3 concepts defined, 0 undefined). The composer
fails to hand the graph to a sub-check that works when invoked directly.

`cas_cert` "reads emitted rung outputs (`iatc_semcheck` and optionally
`cas_select`)". With one failing and the other empty, a certificate assembled
now would certify nothing.

## Why this is a decision, not a gap

The causal model (Fig. 1) terminates S5 in comprehension, and comprehension is
witnessed: verdicts discriminate (well-formed 6 / partial 82 / weak-extraction
10) since its rung-3 half was wired. CAS-CERT would add a *conformance
certificate* layer the model does not currently claim. Adding it is a model
extension — and if the model does not mention it, it is not required for the
pipeline to do what the model says it does.

So the options are honest ones, and both are cheap to state:

1. **Leave it out.** S5's product is comprehension; the certificate layer waits
   until the model claims it. Reclassify CAS-SEL as *needs work + a scope
   decision*, not *needs wiring*.
2. **Bring it in deliberately** — wire `cas_select` with the model backend
   (~136 verification calls, hours on the local endpoint), fix the semcheck
   composition defect, then extend the causal model to include the certificate
   edge and give it a gate.

The workplan's own ordering argues for (1) before the window: it is the only
item in W1–W10 that requires a new model pass *and* a model-level change, and
nothing downstream of S5 consumes a certificate today.

## Defect recorded regardless of the decision

`iatc_semcheck.bb` fails to compose `r2d_concept_coverage.py`, which works
standalone. This is worth fixing whichever way the scope decision goes, because
rung-2 semantic checking is cited by S5's written spec and currently returns
FAIL for every graph in the corpus.
