# Proof Frame Workspace

Date: 2026-03-31
Status: seed

## Purpose

Receipts record what happened in one proof frame. They do not provide
workspace isolation by themselves.

This document defines the owner-side workspace convention for one proof frame:
- frame-local Lean scratch space
- frame-local proof-plan / changelog / writeup artifacts
- explicit promotion into a shared local extension layer

The goal is to avoid collisions between parallel proof attempts while still
allowing reusable local Mathlib extensions to accumulate hygienically.

## Ownership Boundary

- `futon6` owns frame-local workspaces and proof-frame receipts
- `apm-lean` owns the Lean project that typechecks the artifacts
- `futon3c` owns proof obligations, cycle state, and proof DAG semantics

Invariant:
- exploratory or per-problem scratch work must land in the frame workspace
- reusable lemmas must be promoted explicitly into the shared extension layer
- the bridge from frame workspace to `futon3c` must remain at the execute edge

## Layout

Frame-local state:
- `futon6/.state/proof-frames/<problem-id>/<frame-id>/`

Required frame-local artifacts:
- `workspace.json`
- `proof-plan.edn`
- `changelog.edn`
- `execute.md`
- `README.md`

Frame-local Lean module space:
- `apm-lean/ApmCanaries/Frames/<ProblemSegment>/<FrameSegment>/Main.lean`
- `apm-lean/ApmCanaries/Frames/<ProblemSegment>/<FrameSegment>/Scratch.lean`

Shared local extension space:
- `apm-lean/ApmCanaries/Local/`

The frame-local Lean module is where bounded local work happens.
The shared local extension space is where reusable lemmas go after promotion.

## Promotion Discipline

Promotion must be explicit.

Allowed:
- copy or move a stabilized lemma file from one frame workspace into
  `ApmCanaries.Local.*`
- record provenance back to the originating frame

Forbidden:
- writing exploratory scratch work directly into `ApmCanaries.Local`
- treating shared extension files as the scratch area for active frames

## Receipt Link

When a proof-frame receipt is emitted for a workspace-backed frame, it should
carry a `frame/workspace` map naming:
- workspace root
- Lean module path
- Lean source files
- proof-plan path
- changelog path
- execute notes path
- shared extension root

This keeps the execution trace graph replayable without collapsing the frame
workspace into the proof DAG itself.
