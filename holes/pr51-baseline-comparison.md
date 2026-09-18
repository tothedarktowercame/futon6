# PR #51 Stage 0: recovered validation evidence

2026-09-17; master `8dd08f4`, PR head `304beb6`. This is a comparison of
available records, not a claim to have reproduced either model run.

## What the earlier tests establish

`HANDOFF-superpod.md` §2b records preflight/conformance on three hosts:
linode-chicago 3/11 and 3/6; Zone 10/11 and 3/6; dev 11/11 and 6/6.
It explicitly states: **“None of the three ran the pipeline.”** The dev
conformance observation used Ollama/qwen2.5-coder, not Rob's llama3.1:70b.

The locally available `data/runs/mark7-rehearsal-20260808/phase-ledger.jsonl`
contains only S0, STAGE, S1, S2 passing entries (`corpus_id=mark7-16`). It
does not demonstrate S3–S12 or a rejection-free build.

`REPRODUCING.md` records a different reference observation: GLM-4.5-Air Q4
under llama.cpp on Zone, 16 papers, 98 proof graphs, 280/280 expository passes,
94 typed and **four cycle-rejected** proofs. Its acceptance text permits
11/11 **non-FAIL**, including warnings, and calls anchor-faithfulness red
expected while H38 remains open. The document says model hash and server build
were not captured. This acceptance definition is weaker than Joe's current
fully valid, rejection-free target; do not carry those exceptions forward.

The available `data/runs/mark7z/phase-ledger.jsonl` has 33 rows mixing
`math-ct-top100`, `math-ct-e2e-12`, and `math-ct-e2e-16`. The hazard ledger H35
and run-path discussion document shared directories admitting 58 graphs from
four undeclared papers. These records cannot certify a single clean fresh run.

This explains ambiguity in the published evidence. It does not prove Joe
never performed a later or separate validation; no such complete record was
located in this inspection.

## Rob's observation and missing evidence

PR #51 reports a 12-paper A100/Ollama llama3.1:70b run, preflight 11/11,
conformance 6/6, but partial S4/S6, one S7 G7 rejection, two replay failures
and one warning. This is a different corpus/model/serving configuration from
the documented reference. PR comments are empty at inspection time; the PR
contains one code commit and no run bundle. Raw logs, wrapper changes,
attempt-specific counts, exact model/server identities, and Rob's actual
manifest hash remain unavailable. No messages were sent requesting them.

Do not infer that the missing authority caused every reference failure.
Preserve that as a hypothesis until the original artifacts and corrected run
can be compared. The fully valid build target remains unmet by the recovered
evidence; it is not replaced by the older non-FAIL criterion.

## Frozen local corpus candidates

Keep both manifests byte-for-byte; no repair runs have started:

| Manifest | Papers | SHA-256 |
|---|---:|---|
| `holes/mark7-16.ids.txt` | 16 | `8cfb8461353f55aeeccf1b5bf8812d3e33f80207cf6cf81c89fc1b441e045a9d` |
| `holes/mark7z-e2e.ids.txt` | 12 | `7d274ec954ffa113b7c8e1169b0e7a4eec08266a74cc5455e67c80608be873bc` |

The first is the existing documented reference corpus; the second is a local
12-paper candidate, **not yet verified as Rob's exact corpus**. Use the 16-paper
manifest for the documented reference acceptance scope unless Joe establishes
another scope. Reproduce Rob's defects only against his confirmed inputs.

## Substrate inspection

Both existing archives contain only five futon6 data files (four WARP JSONs
and the CT concept encyclopedia); neither includes background-corpus-index:

| Original archive | Members | SHA-256 before repair |
|---|---:|---|
| `data/mark7-substrate.tgz` | 1147 | `1f9f9545c03500630fc415f84a380e3b15f48fe3ed2d93ed26f4b62b56384c7a` |
| `data/mark7-ct-substrate.tgz` | 69 | `db319b8a33851163ba92333e0b607f20a705555454ef78acfe7e3d605a285b2f` |

The local authority reports schema 2, generation time
`2026-06-11T12:54:10.643513+00:00`, 130,960 term keys, 80,586 NNexus rows,
20,653 nLab names, and 1,678 CT-prior entries. Its metadata says
`candidate-filtered?=true`: the full existing authority can be shipped, but
must not be described as a complete unfiltered CT prior. Original input hashes
are not recorded in the index; don't invent retrospective provenance.

## Implementation boundary

See [the complete 146-file disposition](pr51-file-disposition.md). Static
dependency tracing reaches 35 changed files from the chosen runner and setup
entry points. Review each before adoption. Begin Stage 1 with authority
provisioning and validation; keep the broader configuration sweep separate
within that stage. Stage 0 remains evidence-incomplete until Rob's raw bundle
and any additional earlier validation record are available. That does not
prevent repairing independently demonstrated dependency defects.
