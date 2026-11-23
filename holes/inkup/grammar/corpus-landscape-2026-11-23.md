# InkUp grammar corpus landscape — 2026-11-23

Status: read-only inventory for sampling; no corpus-quality claim.

## Local CT scope store

`mark2/ct-fresh-scopes` under the storage home contains 9,916 JSON files totalling
11,754,518,979 bytes.  A deterministic 100-file stratified sample (evenly
spaced over the sorted filenames) parsed without errors:

- 98 of 100 files had nonzero scopes;
- 493,319 scopes occurred in the sample;
- the median was 3,846.5 scopes per sampled file;
- the most frequent types were low-level mathematical and binding structures
  (`math/subscript`, `math/macro-arg`, `math/macro-call`,
  `math/superscript`, `constrain/relation`, and `bind/typed`).

This store is broad enough for distributional work, but its raw scope count is
not a count of expository choices: much of it is syntax-level structure.
Sampling for the positive grammar must stratify by layer and scope type rather
than treating every scope as one observation of writing quality.

## Rob's retained Mark7 runs on metameso

Read-only SSH inspection of Rob's home directory on `metameso` found an empty
`mark2` directory there and these retained result archives beside it:

| Archive | Compressed bytes | Uncompressed tar bytes |
|---|---:|---:|
| `results-mark7-mark7master-20260921.tar.gz` | 7,887,244 | 110,397,440 |
| `results-mark7-mark7rhgrh12v4-20260927.tar.gz` | 48,337,119 | 758,988,800 |
| `results-mark7-mark7rhgrh100-20260928.tar.gz` | 1,208,338,712 | 14,960,988,160 |

Nothing was extracted, transferred, or modified.

The large archive identifies itself in `run/accretion-curve.json` as run
`mark7rhgrh100-20260928`, corpus `rh-grh-top100`.  Its S4 extraction accounting
lists 100 expected papers, all accepted and none rejected, errored, deferred,
or unaccounted.  Its proof-move grounding curve reaches 7,029 proof graphs,
3,365 cues, and grounding 0.465.

Therefore the evidence presently supports **100 retained papers and 7,029
proof graphs**, not 1,000 retained papers.  The remembered larger figure may
describe an upstream complex-analysis collection or a run not present under
Rob's home; it must not be substituted for the manifest count.

## Sampling consequence

Use three evidence strata:

1. **APM alternate tellings:** controlled comparisons over one formal proof,
   suitable for testing reader profiles and realization actions.
2. **Local CT/Mark7 scopes:** broad mined vocabulary, suitable for estimating
   prevalence and discovering candidate functions after separating syntax
   layers from expository layers.
3. **Rob's `rh-grh` run:** professional complex-analysis papers, suitable for
   transfer testing only after a small manual precision audit of anchors,
   scope labels, and duplicate/near-duplicate passages.

The first audit sample should cross paper length, number of proof graphs, S4
scope density, and publication date.  It should preserve rejected and
contested annotations.  Aggregate prevalence is not evidence of quality until
that audit estimates precision.
