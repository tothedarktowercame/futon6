FrontierMath Local Ops

Purpose:
- keep FrontierMath-specific local bring-up owned by `futon6`
- treat `futon3c` as the generic runtime/IRC substrate, not the owner of
  FrontierMath-specific launch policy

Current wrapper:
- `scripts/frontiermath/local-futon3c-windows.bat`
  - starts a local `futon3c` dev lane for FrontierMath work
  - prefers a sibling `futon3c-mfuton-overlay` checkout when present, then
    falls back to the sibling `futon3c` checkout
  - keeps `#futon` as the primary room and adds the configured FrontierMath
    room (default `#math`) through
    `futon3c`'s supported `--math-irc` switch
  - isolates Codex continuity to `futon6/.state/codex-frontiermath-local/`
  - defaults to a local FM lane with `codex` as the invoke bot and `tickle`
    available as a bridge-visible system sender for room assignment pages
  - disables Claude relay/register lanes by default
  - defaults to an isolated local rehearsal port quartet:
    - `FUTON1A_PORT=7271`
    - `FUTON3C_PORT=7270`
    - `FUTON3C_IRC_PORT=7667`
    - `FUTON3C_DRAWBRIDGE_PORT=7768`
  - isolates bridge-local runtime files and `/say` HTTP under the
    FrontierMath lane defaults:
    - `XDG_RUNTIME_DIR=mfuton/data/tmp/frontiermath-local/runtime`
    - `BRIDGE_HTTP_PORT=7769`
    - `INVOKE_BASE=http://127.0.0.1:7270`
  - defaults `FUTON3C_PROOF_STATE_ROOT` to
    `mfuton/data/frontiermath-local/FM-001/active` via `MFUTON_ROOT`
    without changing `futon3c`'s generic proof-tool surface
- `scripts/frontiermath/local-futon3c-linux.sh`
  - starts local `futon3c` runtime plus `ngircd_bridge.py` under one shell
    wrapper
  - keeps `#futon` primary and joins the configured FrontierMath room
    (default `#math`) via supported bridge env vars
  - isolates Codex continuity to `futon6/.state/codex-frontiermath-local/`
  - defaults `CODEX_CWD` to the `futon6` repo root so work lands in the
    intended owner neighborhood by default

Key FrontierMath env surface:
- shared runtime / bridge seam
  - `FUTON3C_FRONTIERMATH_ROOM`
    - configured FrontierMath IRC room
    - default `#math`
    - consumed directly by the `futon3c` bridge/runtime surfaces, not just by
      wrapper prose
  - `MATH_IRC`
    - enables the dedicated FrontierMath room-sensitive behavior in the
      `mfuton`-specific runtime seam
    - default `true` in the local FrontierMath wrapper
  - `IRC_CHANNELS`
    - extra IRC rooms joined by the bridge
    - defaults to the configured `FUTON3C_FRONTIERMATH_ROOM` in the local
      FrontierMath wrapper
  - `IRC_COMMAND_OWNER_AGENT_MAP`
    - optional bare-`!` room ownership map
    - defaults to `#futon:codex-1,<frontiermath-room>:codex-1` in the local
      FrontierMath wrapper
- local FrontierMath lane defaults
  - `FUTON3C_FM_CONDUCTOR_ROTATION`
    - default `codex-1`
  - `FUTON3C_FM_CONDUCTOR_AUTOSTART`
    - default `true`
  - `FUTON3C_DIRECT_INVOKE_TIMEOUT_SECONDS`
    - default `10`
  - `CODEX_SESSION_FILE`
    - defaults to the wrapper-owned FrontierMath continuity lane
  - `CODEX_CWD`
    - defaults to the `futon6` repo root for this lane
- wrapper-only override
  - `FUTON6_FRONTIERMATH_LOCAL_CONFIG`
    - Windows wrapper override for the typed local config JSON
    - useful for bounded live repro configs without mutating the checked-in
      default config

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

Proof-frame workspaces:
- `scripts/frontiermath/proof-frame-workspace.md`
  - defines the owner-side workspace convention that receipts alone do not provide
  - keeps per-frame scratch separate from reusable local extension work
- `scripts/frontiermath/init-proof-frame-workspace.py`
  - creates one frame-local workspace rooted in `futon6/.state/proof-frames/`
  - also creates per-frame Lean files under `apm-lean/ApmCanaries/Frames/...`
  - seeds `proof-plan.edn`, `changelog.edn`, and `execute.md`
- `scripts/frontiermath/promote-proof-frame-lean.py`
  - copies stabilized Lean material from one frame workspace into
    `apm-lean/ApmCanaries/Local/...`
  - keeps promotion explicit instead of using the shared local extension area as
    scratch space

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
- frame-local workspaces are now the intended owner-side answer to parallel
  scratch isolation; receipts continue to record provenance rather than
  replacing workspace boundaries
