# Pre-superpod run #2 — local CPU first pass report

Author: codex-1, 2026-06-17. This is the no-GPU local first pass requested from
`holes/proofcheck-run-invocation.md`: pre-flight witness checks, arXiv CAS-CERT
certificates over the 9 `loop-run-70b` finals, and the APM CAS-SEL stub track.

## Commands run

```bash
python3 scripts/pipeline_witness.py --plan
python3 scripts/pipeline_witness.py --witness 0706.1286
python3 scripts/pipeline_witness.py --witness 0708.2067
python3 scripts/pipeline_witness.py --witness 0709.0248
python3 scripts/pipeline_witness.py --witness 0708.2185

python3 scripts/cas_cert.py --graph-dir data/iatc-argument-graphs/loop-run-70b --out /tmp/run2-arxiv.cert.json
python3 scripts/cas_cert.py --graph-dir data/iatc-argument-graphs/loop-run-70b --gate >/tmp/run2-arxiv-gate.cert.json

python3 scripts/cas_select.py --backend stub > /tmp/run2-cas-select-stub.json
python3 scripts/rung3_residue_spike.py --json-out /tmp/run2-rung3-residue.json
```

I also smoke-fed the CAS-SEL JSON into `cas_cert --cas-select` with a temporary
APM semcheck shell in `/tmp/run2-apm-semcheck-shell.edn`. That demonstrates the
technique grain populates for APM IDs, but it is not a real proof/concept
certificate because no APM semcheck graph population exists in this run.

## Pre-flight witness gate

The DAG plan reports stage 6 `cas_select` as the expected arXiv seam GAP: it
needs `proof-steps`, and no upstream arXiv stage produces them yet.

| witness | s1 anatomy | s2 candidates | s3 IATC | s4 repair+gate | s5 semcheck | s6 CAS-SEL | prereg verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 0706.1286 | PASS | PASS | PASS | PASS | MISS | GAP | matched |
| 0708.2067 | PASS | PASS | PASS | PASS | MISS | GAP | matched |
| 0709.0248 | PASS | PASS | PASS | PASS | MISS | GAP | matched |
| 0708.2185 | PASS | PASS | MISS | MISS | MISS | GAP | miss |

Miss against P-A: `0708.2185` has stages 1-2 only; local artifacts for IATC,
repair+gate, and semcheck are not materialized. The expected seam-6 GAP holds
for all four witnesses.

## ArXiv CAS-CERT scorecard

`cas_cert.py` emitted 9 deterministic certificates. Aggregate gate is `FAIL`;
`--gate` exits `1` because miswired ports exist. Symbol and technique grains are
N/A across the arXiv track, as preregistered.

| paper | gate | miswires | residual sorries | concept vector | proof vector |
|---|---:|---:|---:|---|---|
| 0705.0452 | FAIL | 4 | 9 | filled=6 empty=0 rate=1.000 | filled=4 empty=9 miswired=4 rate=0.235 |
| 0706.1286 | PASS | 0 | 6 | filled=2 empty=2 rate=0.500 | filled=9 empty=4 miswired=0 rate=0.692 |
| 0708.1921 | PASS | 0 | 7 | filled=1 empty=1 rate=0.500 | filled=0 empty=6 miswired=0 rate=0.000 |
| 0708.2067 | FAIL | 2 | 7 | filled=8 empty=0 rate=1.000 | filled=5 empty=7 miswired=2 rate=0.357 |
| 0709.0248 | FAIL | 2 | 4 | filled=8 empty=2 rate=0.800 | filled=6 empty=2 miswired=2 rate=0.600 |
| 0711.0473 | FAIL | 1 | 2 | filled=2 empty=0 rate=1.000 | filled=6 empty=2 miswired=1 rate=0.667 |
| 0712.0724 | FAIL | 1 | 8 | filled=3 empty=0 rate=1.000 | filled=3 empty=8 miswired=1 rate=0.250 |
| 0801.0199 | PASS | 0 | 2 | filled=4 empty=0 rate=1.000 | filled=9 empty=2 miswired=0 rate=0.818 |
| 0801.3843 | FAIL | 1 | 1 | filled=6 empty=0 rate=1.000 | filled=8 empty=1 miswired=1 rate=0.800 |

Headline counts: 6/9 `FAIL`, 3/9 `PASS`. The PASS set is exactly the predicted
`0706.1286`, `0708.1921`, `0801.0199`.

Concept grain actual: mean `0.867`, spread `0.500-1.000`. This matches P-C.

Proof warrant actual: R2c filled `6/28 = 0.214`; empty `22/28`. This matches
P-D. Papers with orphan residuals: 4/9 (`0705.0452`, `0708.1921`,
`0708.2067`, `0712.0724`), also matching P-D.

Aggregate proof ports across the 9 certs: filled `50`, empty `41`, miswired
`11`, N/A `27`. Residual-sorry kinds: proof/orphan `19`,
proof/missing-warrant `22`, concept/undefined `5`.

## Residual-sorry map examples

These are the open-question outputs, not gate failures:

- `0706.1286`: concept `calmod like bicategory` undefined; warrants
  `e-equiv-cat-like`, `e-equiv-calmod-like`, `e-equiv-problem` missing.
- `0708.1921`: concept `mu inv` undefined; orphan nodes `mu-inv`, `sigma`,
  `sigma-S`; warrants `e-sigma-eq`, `e-rS-sigma-eq` missing.
- `0709.0248`: concepts `parameterized rules`, `standard rules` undefined;
  warrants `e-extensional-category`, `e-parameterized-rules` missing.
- `0801.0199`: warrants `e-f-cong-g`, `e-z-e` missing, but no miswires.

## Honesty spot-check

P-E holds on the checked cases. No certificate claims "verified"; certificates
report conformance-by-grain plus residual ports.

- `0709.0248` FAIL traces to real R2a anchor miswires:
  `anchor::extensional-category`, `anchor::reflexivity-term`.
- `0712.0724` FAIL traces to the real R2b `cycle` miswire.
- `0708.1921` PASS despite proof rate `0.000`: its proof ports are empty
  orphan/missing-warrant gaps, not miswired ports.
- `0801.0199` PASS with two residual warrants: empty ports are carried in the
  residual map and do not fail the gate.

I did not find a port state in these checks that could not be justified against
the underlying rung verdict shape.

## APM CAS-SEL track

`python3 scripts/cas_select.py --backend stub` reproduced all four worked-proof
topologies against the oracle:

| proof | topology slots | correct | rate | induce queue | declared sorries | thin sorries |
|---|---:|---:|---:|---:|---:|---:|
| a93J05 | 5 | 5 | 1.000 | 0 | 3 | 0 |
| a96J01 | 5 | 5 | 1.000 | 0 | 3 | 0 |
| a96J04 | 5 | 5 | 1.000 | 0 | 3 | 0 |
| b97J01 | 7 | 7 | 1.000 | 0 | 3 | 0 |

Overall: `22/22`, rate `1.000`.

The rung-3 residue spike confirms the preregistered residue: CAS-0 strict
residue `6/22 = 27.3%`, with `16/22 = 72.7%` grounded. Bucket counts:
grounded `14`, thin `2`, ungrounded `6`.

Technique-grain smoke test: feeding the CAS-SEL JSON into CAS-CERT with a
temporary APM semcheck shell populated CAS-SEL technique ports. Each APM proof
got `filled=3`, `empty=0`, `miswired=0`, `rate=1.000` for the partial
technique grain, plus the full rung-3 technique port remained N/A. This is a
shape check only until real APM semcheck graphs exist.

## Preregistration scorecard

| prediction | actual | score |
|---|---|---|
| P-A: stages 1-4 PASS for all 4 witnesses; stage 5 MISS; seam-6 GAP | 3/4 matched. `0708.2185` misses stages 3-4 as well as 5. seam-6 GAP all 4. | partial miss |
| P-B: CAS-CERT deterministic over all 9; aggregate FAIL; ~6/9 FAIL and predicted PASS set | 9/9 emitted; aggregate FAIL; 6/9 FAIL; PASS set exactly predicted. | hit |
| P-C: concept coverage mean ~0.87, spread 0.5-1.0 | mean 0.867, spread 0.500-1.000. | hit |
| P-D: warrant-resolution aggregate ~6/28; ~4/9 have orphans | R2c filled 6/28; 4/9 have orphan residuals. | hit |
| P-E: honesty holds | Spot checks justify FAILs as real miswires and empty ports as gaps; no verified overclaim. | hit |
| P-F: APM stub reproduces all 4 topologies; residue 27.3% | CAS-SEL stub 22/22; residue spike 6/22 = 27.3%. | hit |

## Determinism checks

- Re-running `cas_cert.py --graph-dir data/iatc-argument-graphs/loop-run-70b`
  produced byte-identical JSON (`cmp` exit `0`).
- Re-running `cas_select.py --backend stub` produced byte-identical JSON
  (`cmp` exit `0`).

## Verdict

Yes: the checker spine is producing honest certificates on this local CPU pass.
The main preregistered claim holds: CAS-CERT exposes real miswires as gate
failures, preserves empty ports as residual open questions, and avoids claiming
proof verification. The only scorecard miss is producer materialization for
`0708.2185`, where local IATC/repair/semcheck artifacts are absent.
