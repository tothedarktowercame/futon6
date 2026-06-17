# rung-3-1 residue spike

This is the empirical verb-side counterpart to R2d: measure how often a proof move is covered by the current CAS pattern menu before asking a model to judge or invent the residue.

## Inputs and method

- Pattern pool: `/home/joe/code/futon3/library/math-informal` plus `/home/joe/code/futon3/resources/sigils/patterns-index.tsv`; loaded `39` math-informal patterns.
- CAS-0 worked proofs: `tests/fixtures/cas-select/{a93J05,a96J01,b97J01,a96J04}.steps.json` with their oracle files.
- Loop-run sample: final EDN graphs in `data/iatc-argument-graphs/loop-run-70b` only.
- Selector reuse: direct `cas_select.retrieve(..., k=4)` followed by `cas_select.verify(..., backend="stub", oracle=...)` for CAS-0. This deliberately avoids `select_proof`, whose test-only stub path injects oracle patterns after retrieval misses.
- Loop-run edges have no oracle, so their CAS rows are retrieval-only candidates, not verified matches. They are useful for estimating how much of the 70B edge vocabulary the current menu can even touch, but not for correctness.
- Question menu source on disk: `holes/handoffs/question-asking-pattern-mining-from-mo-rm-2026-03-06.md` and `holes/excursions/E-informal-proof-checking.md`. The referenced `data/question-patterns/question-asking-pattern-language.md` is not present in this checkout.

## Residue measurements

| Sample | Moves | Deterministic covered | Residue | Residue rate | Interpretation |
|---|---:|---:|---:|---:|---|
| CAS-0 strict verified | 22 | 16 | 6 | 27.3% | Current committed selector's honest strict share; this is the measured LLM/verifier residue on worked proof moves. |
| loop-run-70b strict verified | 28 | 0 | 28 | 100.0% | No oracle-backed verifier exists for these graph edges, so every edge remains strict residue. |
| loop-run-70b retrieval-only | 28 | 28 | 0 | 0.0% | Candidate coverage only; this measures menu reach, not correctness. |
| combined candidate surface | 50 | 44 | 6 | 12.0% | Upper-bound deterministic menu reach if loop retrieval candidates are later verified. |

Strict CAS-0 residue is therefore **6/22 = 27.3%**. That number is the empirical LLM share for the current CAS-0 verified setting.

## Buckets

CAS-0 strict buckets:

```json
{
  "grounded": 14,
  "thin": 2,
  "ungrounded": 6
}
```

loop-run-70b retrieval buckets:

```json
{
  "conjecture": 1,
  "grounded-provisional": 2,
  "thin": 25
}
```

Pattern-type counts:

```json
{
  "cas0": {
    "heuristic": 2,
    "none": 6,
    "verifiable": 14
  },
  "loop_run_70b": {
    "heuristic": 26,
    "verifiable": 2
  }
}
```

## Heuristic vs verifiable typing

For this spike, a matched pattern is typed `verifiable` only when it can plausibly license an inference leaf by a checkable object, definition, theorem application, calculation, induction, case split, or bound. Other retrieved CAS patterns are typed `heuristic`: they may justify a strategy, but a load-bearing proof edge still needs a lower verifiable discharge.

Verifiable pattern set used in the measurement:

```
construct-an-explicit-witness, count-over-a-decomposition, epsilon-of-room, estimate-by-bounding, induction-and-well-ordering, quotient-by-irrelevance, reduce-to-known-result, split-into-cases, unfold-the-definition, verify-universal-property
```

CAS-0 matched-pattern distribution:

```json
{
  "construct-an-explicit-witness": 2,
  "count-over-a-decomposition": 1,
  "epsilon-of-room": 1,
  "estimate-by-bounding": 3,
  "induction-and-well-ordering": 1,
  "local-to-global": 1,
  "reduce-to-known-result": 3,
  "separate-into-independent-pieces": 1,
  "split-into-cases": 1,
  "unfold-the-definition": 2
}
```

loop-run-70b top-candidate distribution:

```json
{
  "count-over-a-decomposition": 1,
  "pass-to-a-subsequence": 2,
  "structural-characterization": 1,
  "structural-equivalence": 23,
  "verify-universal-property": 1
}
```

## Conjecture recognition

Author-declared gaps are credited rather than flagged as thin: a sentence or edge text matching `conjecture`, `open problem`, `problem of`, `ought to`, `unknown`, or `we do not know` goes to the `conjecture` bucket. In this sample no CAS-0 fixture step is author-declared. The loop-run sample contains retrieval rows with phrases such as `ought-to-include`, and those are credited as author-declared/open-status rather than as hidden failures.

## Gap to ArSE question mapping

| Gap bucket | RM question pattern | ArSE question template |
|---|---|---|
| `thin` / heuristic leaf | `STRUCTURAL PROBE` | What verifiable inference discharges the heuristic step `<pattern>` here? |
| `ungrounded` / no match | `THEOREM APPLICABILITY` or `TECHNIQUE LANDSCAPE` | Which known theorem or proof technique, if any, licenses this move from `<premise>` to `<conclusion>`? |
| missing or unresolved warrant | `KERNEL IDENTIFICATION` | What is the one lemma/computation needed to turn this edge into a resolved inference? |
| author-declared gap/conjecture | `EXISTENCE_WONDER` / `CONJECTURE_TESTING` | Is the stated extension/problem known under the hypotheses used in the passage? |
| obstruction-like residual | `OBSTRUCTION_IDENTIFICATION` | What obstruction prevents the intended inference or generalization? |

## CAS-0 per-move evidence

| Paper | Move | Expected | Verified match | Type | Bucket | Top-4 candidates |
|---|---|---|---|---|---|---|
| `a93J05` | `s1` | `construct-auxiliary-object` | `NONE` | `none` | `ungrounded` | `the-diagonal-argument, use-probabilistic-method, structural-inclusion, monotone-approximation` |
| `a93J05` | `s2` | `reduce-to-known-result` | `reduce-to-known-result` | `verifiable` | `grounded` | `reduce-to-known-result, check-the-extreme-cases, count-over-a-decomposition, structural-obstruction-as-theorem` |
| `a93J05` | `s3` | `quotient-by-irrelevance` | `NONE` | `none` | `ungrounded` | `pass-to-a-subsequence, the-diagonal-argument, unfold-the-definition, use-probabilistic-method` |
| `a93J05` | `s4` | `local-to-global` | `local-to-global` | `heuristic` | `thin` | `local-to-global, estimate-by-bounding, reduce-to-known-result, epsilon-of-room` |
| `a93J05` | `s5` | `reduce-to-known-result` | `reduce-to-known-result` | `verifiable` | `grounded` | `reduce-to-known-result, count-over-a-decomposition, hybrid-certification, structural-obstruction-as-theorem` |
| `a96J01` | `s1` | `construct-an-explicit-witness` | `construct-an-explicit-witness` | `verifiable` | `grounded` | `use-probabilistic-method, epsilon-of-room, construct-an-explicit-witness, argue-by-contradiction` |
| `a96J01` | `s2` | `construct-auxiliary-object` | `NONE` | `none` | `ungrounded` | `separate-into-independent-pieces, split-into-cases, count-over-a-decomposition, the-diagonal-argument` |
| `a96J01` | `s3` | `separate-into-independent-pieces` | `separate-into-independent-pieces` | `heuristic` | `thin` | `separate-into-independent-pieces, epsilon-of-room, pass-to-a-subsequence, the-diagonal-argument` |
| `a96J01` | `s4` | `reduce-to-known-result` | `NONE` | `none` | `ungrounded` | `epsilon-of-room, argue-by-contradiction, quotient-by-irrelevance, the-diagonal-argument` |
| `a96J01` | `s5` | `estimate-by-bounding` | `estimate-by-bounding` | `verifiable` | `grounded` | `quotient-by-irrelevance, the-diagonal-argument, estimate-by-bounding, encode-as-algebra` |
| `a96J04` | `s1` | `unfold-the-definition` | `unfold-the-definition` | `verifiable` | `grounded` | `separate-into-independent-pieces, unfold-the-definition, split-into-cases, the-diagonal-argument` |
| `a96J04` | `s2` | `unfold-the-definition` | `NONE` | `none` | `ungrounded` | `separate-into-independent-pieces, structural-inclusion, count-over-a-decomposition, epsilon-of-room` |
| `a96J04` | `s3` | `unfold-the-definition` | `unfold-the-definition` | `verifiable` | `grounded` | `unfold-the-definition, separate-into-independent-pieces, argue-by-contradiction, quotient-by-irrelevance` |
| `a96J04` | `s4` | `estimate-by-bounding` | `estimate-by-bounding` | `verifiable` | `grounded` | `estimate-by-bounding, quotient-by-irrelevance, the-diagonal-argument, unfold-the-definition` |
| `a96J04` | `s5` | `epsilon-of-room` | `epsilon-of-room` | `verifiable` | `grounded` | `epsilon-of-room, structural-inclusion, pass-to-a-subsequence, induction-and-well-ordering` |
| `b97J01` | `s1` | `construct-an-explicit-witness` | `construct-an-explicit-witness` | `verifiable` | `grounded` | `construct-an-explicit-witness, the-diagonal-argument, use-probabilistic-method, count-over-a-decomposition` |
| `b97J01` | `s2` | `split-into-cases` | `split-into-cases` | `verifiable` | `grounded` | `split-into-cases, check-the-extreme-cases, pass-to-a-subsequence, find-the-right-abstraction` |
| `b97J01` | `s3` | `reduce-to-known-result` | `reduce-to-known-result` | `verifiable` | `grounded` | `reduce-to-known-result, exhaustion-as-theorem, structural-obstruction-as-theorem, encode-as-algebra` |
| `b97J01` | `s4` | `count-over-a-decomposition` | `count-over-a-decomposition` | `verifiable` | `grounded` | `count-over-a-decomposition, argue-by-contradiction, quotient-by-irrelevance, the-diagonal-argument` |
| `b97J01` | `s5` | `estimate-by-bounding` | `estimate-by-bounding` | `verifiable` | `grounded` | `estimate-by-bounding, argue-by-contradiction, quotient-by-irrelevance, the-diagonal-argument` |
| `b97J01` | `s6` | `construct-auxiliary-object` | `NONE` | `none` | `ungrounded` | `epsilon-of-room, construct-an-explicit-witness, pass-to-a-subsequence, the-diagonal-argument` |
| `b97J01` | `s7` | `induction-and-well-ordering` | `induction-and-well-ordering` | `verifiable` | `grounded` | `induction-and-well-ordering, use-probabilistic-method, monotone-approximation, the-diagonal-argument` |

## loop-run-70b per-edge sample

| Paper | Move | Lines | Top candidate | Type | Bucket | Top-4 candidates |
|---|---|---:|---|---|---|---|
| `0705.0452` | `edge-1` | `1290-1293` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, unfold-the-definition` |
| `0705.0452` | `edge-2` | `1290-1298` | `structural-characterization` | `heuristic` | `thin` | `structural-characterization, structural-equivalence, encode-as-algebra, pass-to-a-subsequence` |
| `0705.0452` | `edge-3` | `1298-1302` | `pass-to-a-subsequence` | `heuristic` | `thin` | `pass-to-a-subsequence, reduce-to-known-result, structural-equivalence, count-over-a-decomposition` |
| `0706.1286` | `edge-1` | `333-335` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0706.1286` | `edge-2` | `335-336` | `structural-equivalence` | `heuristic` | `conjecture` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0706.1286` | `edge-3` | `336-337` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0706.1286` | `edge-4` | `337-338` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0706.1286` | `edge-5` | `339-341` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0708.1921` | `edge-1` | `680-680` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0708.1921` | `edge-2` | `681-683` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0708.1921` | `edge-3` | `681-683` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0708.2067` | `edge-1` | `392-393` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, construct-an-explicit-witness, pass-to-a-subsequence` |
| `0708.2067` | `edge-2` | `395-397` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, local-to-global` |
| `0709.0248` | `edge-1` | `1515-1519` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0709.0248` | `edge-2` | `1516-1517` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, structural-characterization, structural-inclusion, epsilon-of-room` |
| `0709.0248` | `edge-3` | `1517-1519` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0711.0473` | `edge-1` | `1118-1119` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, local-to-global, reduce-to-known-result` |
| `0711.0473` | `edge-2` | `1119-1122` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, pass-to-a-subsequence, split-into-cases, local-to-global` |
| `0712.0724` | `edge-1` | `884-887` | `verify-universal-property` | `verifiable` | `grounded-provisional` | `verify-universal-property, structural-equivalence, pass-to-a-subsequence, the-diagonal-argument` |
| `0712.0724` | `edge-2` | `894-895` | `count-over-a-decomposition` | `verifiable` | `grounded-provisional` | `count-over-a-decomposition, pass-to-a-subsequence, reduce-to-known-result, structural-equivalence` |
| `0712.0724` | `edge-3` | `903-907` | `pass-to-a-subsequence` | `heuristic` | `thin` | `pass-to-a-subsequence, reduce-to-known-result, structural-equivalence, count-over-a-decomposition` |
| `0801.0199` | `edge-1` | `386-386` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, structural-inclusion, pass-to-a-subsequence, verify-universal-property` |
| `0801.0199` | `edge-2` | `390-390` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, quotient-by-irrelevance` |
| `0801.0199` | `edge-3` | `390-390` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, unfold-the-definition` |
| `0801.0199` | `edge-4` | `390-390` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, local-to-global` |
| `0801.0199` | `edge-5` | `392-392` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, local-to-global` |
| `0801.3843` | `edge-1` | `652-656` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, pass-to-a-subsequence, local-to-global` |
| `0801.3843` | `edge-2` | `658-658` | `structural-equivalence` | `heuristic` | `thin` | `structural-equivalence, epsilon-of-room, encode-as-algebra, pass-to-a-subsequence` |

## Interpretation

The deterministic menu is already useful but not sufficient. On CAS-0 it strictly verifies 16/22 moves and leaves 6/22 for a semantic retriever/verifier or a new pattern. On loop-run-70b it can attach candidates to most edges, but that is not a proof of match: rung-3-3 still needs a model or richer verifier for the residual judgement `does this edge instantiate this pattern?`.
