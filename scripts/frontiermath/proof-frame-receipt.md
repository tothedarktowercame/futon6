# Proof Frame Receipt

Date: 2026-03-20
Status: seed

## Purpose

Define the `futon6`-owned receipt for one replayable proof frame.

Here, a frame is a bounded metamathematical working context:
- scratch work
- exploratory computation
- failed routes
- local artifacts
- runtime/tooling context

This receipt does not replace the proof graph in `futon3c`.

## Graph Discipline

Primary graph:
- the proof obligation DAG in `futon3c`
- nodes are blocker/lemma/claim obligations
- edges are mathematical dependency edges

Secondary graph:
- the proof frame trace graph in `futon6`
- nodes are replayable proof frames
- edges are trace/handoff relations between execution boundaries

Invariant:
- proof frame receipts may attach to a proof obligation node
- proof frame receipts may refer to earlier proof frame receipts
- proof frame receipts must not silently introduce or redefine proof
  dependency edges

So the answer to "are proofs already DAGs?" is yes:
- the obligation DAG is already the proof DAG
- the new receipt graph is only the navigable trace of how work happened on
  those DAG nodes

## Required Anchors

Every proof frame receipt should name:
- `proof/problem-id`
- `frame/id`
- at least one graph reference back into proof space

When known, it should also name:
- `proof/cycle-id`
- `proof/blocker-id`
- an algorithm owner for the execution boundary

## Output Location

Default location:
- `futon6/.state/proof-frames/<problem-id>/<frame-id>.json`

This keeps local proof-frame receipts owned by `futon6` instead of
scattering them across ad hoc working directories.

## Mapping Forward

The eventual bridge adapter should map this receipt into the
`futon3c` execute-phase payload shape:
- receipt `frame-boundary` -> execute `:step-boundary`
- receipt graph refs -> execute `:graph-refs`
- receipt artifacts -> execute `:artifacts`

Current bridge note:
- `futon3c` still uses the older `:step-boundary` term internally
- the adapter should translate `frame` terminology at the bridge edge rather
  than forcing `futon6` to keep the older name

That adapter must preserve the graph discipline above:
- proof dependency stays in the obligation DAG
- execution trace stays in the receipt graph

## Current EAL Link

The current container-owner seed lives in the separate `futon3` EAL branch:
- `eal/algorithms/create-container.md`

For now, `futon6` receipts may cite that algorithm owner while still keeping
the concrete local execution receipt in the correct repo.
