# FrontierMath Frame Wiring Note For Rob

Date: 2026-03-20

## Context

We split the mixed `futon3c` PR so the merged branch only carried the IRC
bridge/channel-preservation work.

The FrontierMath-local and Windows/dev-lane work was kept out of that PR on a
separate branch and then moved toward `futon6` ownership instead of being
folded back into `futon3c` as a FrontierMath-specific special case.

## What Changed

- `futon6` now owns the local FrontierMath bring-up wrappers for Windows and
  Ubuntu GNU/Linux.
- `futon6` now also owns a new metamathematical receipt layer for local proof
  work:
  - `proof frame` is the term for one bounded working context
  - frames include scratch work, exploratory computation, local artifacts,
    and runtime/tooling context
- `futon3c` remains the owner of the mathematical proof graph:
  - the obligation DAG is still the proof DAG

## Why This Extension

The useful extension beyond the original PR direction is that we now make the
split between:

- mathematical dependency
- metamathematical working context

explicit in the artifacts.

That gives us a place to enforce "work happens in the suitable location"
without pretending that local scratch work is itself a formal proof step.

## Current Wiring

- `scripts/frontiermath/emit-proof-frame-receipt.py`
  writes a `futon6`-owned proof-frame receipt
- `scripts/frontiermath/advance-proof-cycle-from-frame-receipt.py`
  converts that receipt into the execute-phase payload expected by the current
  `futon3c` proof bridge
- the adapter deliberately translates `frame` terminology back to the current
  `futon3c` `:step-boundary` / `:proof-step` wire contract at the bridge edge
  so we can improve the local terminology without breaking the existing proof
  peripheral

## Direction

This is meant as a useful extension of the local-work abstraction, not a
repudiation of it:

- keep `futon3c` generic
- keep FrontierMath-local ownership in `futon6`
- make local proof-work contexts replayable and traceable
- avoid scattering state across whichever repo happened to launch the runtime
