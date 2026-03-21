FrontierMath Local Ops

Purpose:
- keep FrontierMath-specific local bring-up owned by `futon6`
- treat `futon3c` as the generic runtime/IRC substrate, not the owner of
  FrontierMath-specific launch policy

Current wrapper:
- `scripts/frontiermath/local-futon3c-windows.bat`
  - starts a local `futon3c` dev lane for FrontierMath work
  - keeps `#futon` as the primary room and adds `#math` through
    `futon3c`'s supported `--math-irc` switch
  - isolates Codex continuity to `futon6/.state/codex-frontiermath-local/`
  - defaults to a codex-only lane and disables Claude relay/register lanes
  - defaults `FUTON3C_PROOF_STATE_ROOT` to
    `mfuton/data/frontiermath-local/FM-001/active` via `MFUTON_ROOT`
    without changing `futon3c`'s generic proof-tool surface
- `scripts/frontiermath/local-futon3c-linux.sh`
  - starts local `futon3c` runtime plus `ngircd_bridge.py` under one shell
    wrapper
  - keeps `#futon` primary and joins `#math` via supported bridge env vars
  - isolates Codex continuity to `futon6/.state/codex-frontiermath-local/`
  - defaults `CODEX_CWD` to the `futon6` repo root so work lands in the
    intended owner neighborhood by default

Ownership boundary:
- `futon6` owns FrontierMath session continuity, room selection, and local
  mission bring-up policy
- `futon3c` owns the generic dev runtime, IRC bridge, and proof tools
- wrapper-level `CODEX_CWD` defaults are the current enforcement hook for
  "work should happen in the suitable location"

Proof-frame receipts:
- `scripts/frontiermath/proof-frame-receipt.md`
  - defines the local receipt owner for one replayable proof frame
  - keeps the graph distinction explicit:
    - `futon3c` owns the proof obligation DAG
    - `futon6` owns the execution-trace graph for how a frame was worked
- `scripts/frontiermath/emit-proof-frame-receipt.py`
  - writes receipts under `futon6/.state/proof-frames/`
  - emits graph refs back to proof problem/cycle/blocker/frame space
  - is the current local seed for "actual frame builders should emit a receipt"
- `scripts/run-proof-stepper.py`
  - can now emit one of these receipts when invoked with
    `--proof-problem-id` and optional cycle/blocker anchors
  - this keeps proof-stepper experiment runs attachable to proof space without
    pretending that the experiment run itself rewrites proof dependencies
- `scripts/frontiermath/advance-proof-cycle-from-frame-receipt.py`
  - reads one futon6 proof frame receipt
  - converts JSON graph-ref strings into the EDN keyword shape expected by
    `futon3c`
  - can print the execute-phase payload, print the `/eval` form, or submit
    `pb/cycle-advance!` to a running local `futon3c`
- `scripts/frontiermath/advance-proof-cycle-from-local-run.py`
  - projects one existing `mfuton/data/frontiermath-local/<problem>/runs/<run-id>/`
    bundle into the proof-frame seam
  - requires explicit `--cycle-id` and optional `--blocker-id` until the
    current run bundles carry those graph anchors natively
  - emits a proof-frame receipt first, then hands that receipt to the existing
    cycle-advance adapter
- `scripts/frontiermath/advance-proof-cycle-from-local-run-windows.bat`
  - Windows wrapper for the same owner-side local-run seam

Open design issue:
- proof-state-root and repo-layout assumptions are still unresolved
  cross-repo abstractions
- do not reintroduce FrontierMath-specific path assumptions into `futon3c`
  as a shortcut
- the current Windows wrapper default to
  `mfuton/data/frontiermath-local/FM-001/active` is an owner-side local lane
  binding, not a claim that the broader cross-repo abstraction is solved
- the current local-run bridge still needs explicit cycle/blocker anchors at
  invocation time; that is an honest temporary requirement until the run
  bundles themselves carry proof-graph ids
- a future solution should make proof-frame execution container-friendly rather
  than binding it to one repo's local filesystem layout
- when the bridge adapter is added later, it should map these receipts into
  `futon3c` execute-phase payloads without creating a second proof DAG
  - seed adapter now exists in
    `scripts/frontiermath/advance-proof-cycle-from-frame-receipt.py`
