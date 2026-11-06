# T-codeserver-arxana-demo — a code-server client for Arxana scopes, alongside Emacs

**Type:** T-prefix ticket (demo + client-side requirements, no implementation
yet). **Raised:** 2026-11-07 by Joe, with Zai-1. **Status:** open.
**Companions:** `T-arxana-ts.md` (futon6 `f5e5461`), `E-native-lean-scopes.md`
(futon6 `26512d2`), `futon4/holes/labs/M-jvm-arxana/` D1–D7 DERIVE artifacts
(baseline `84f999d`) and `A1-design-fitness-argument.md` (ARGUE, `37ba0dc`).

## 0. What is being asked

Stand up a [code-server](https://github.com/coder/code-server) (open-source
VS Code in the browser) demo that displays and consumes Arxana scopes
alongside Emacs — the first VS Code-class client P3's census found missing —
and, from that experience, write the client-side requirements to hand back to
codex-21 as consumer input to M-jvm-arxana's remaining design/INSTANTIATE
work.

Why code-server specifically: T-arxana-ts's phase 3 targets a local VS Code
extension; code-server gives the same extension host in a browser, so the
demo is shareable (Joe and others can open it by URL) without installing
anything, and it exercises exactly the D5 VS Code client surface (UTF-16 →
codepoint conversion at the edge, decorations carrying only response IDs, no
extension-local identity authority).

## 1. What already exists and converges

T-arxana-ts (futon6 `f5e5461`) was written 2026-11-05, one day before
M-jvm-arxana's DERIVE D1–D7 landed. Reading them together, the convergence is
real, not coincidental — both derive from the same evidence:

| T-arxana-ts concept | M-jvm-arxana DERIVE counterpart | Status |
|---|---|---|
| `Segment.hash` (sha256, checked before `exact`) | D1 artifact revision, content-addressed (`arxana:revision:sha256:…`), immutable | Same discipline; TS needs to *name* revisions by ID, not just hash-check |
| `ResilientAnchor` (content-hash, length, context before/after, hint offset/line) | D2 anchor finder v1 input fields — identical list | Direct port; D2's stage semantics and ambiguity algebra apply |
| `Resolution` kinds `exact`/`located`/`unanchored` | D2 statuses `resolved`/`resolved-unverified`(+`located` evidence)/`missing`(+`stale-input`) | Map 1:1-ish; see §3 R2 |
| `offset_unit: unicode-codepoints-zero-based-end-exclusive` | D3 canonical location vocabulary | Identical |
| `Store` (ID-addressed reads, typed writes) | P6/D3 waist (`entity/get`, `graph/query`, `assertion/promote`, …) | Same waist; transport differs — see §3 R1 |
| `Hyperedge` ends with roles | D1 `:hx/ends` roles, `:hx/endpoints` index | Same shape |

So the demo should NOT invent a new protocol. It consumes the T-arxana-ts
core (`@arxana/core` outline) as the client-side model, and the requirements
document (§3) is about aligning that client with the arxana.v1 authority
contract D3/D5 specified.

## 2. Demo plan (incremental, each step demoable)

1. **Substrate on frozen data.** Run T-arxana-ts phase 1 (a00J02 parity) as
   the fixture: one AnnotationSet (493 marks, 520 spans, 9 edges), the
   embedded `m7-source` as the segment. Open the same `.tex` in code-server;
   the `substrate-vscode` adapter paints decorations and hovers from the
   sibling JSON. Resolution is `exact` (hash check) — no service needed yet.
2. **Code-server packaging.** The extension runs unmodified inside
   code-server; add a webview pane rendering the margin/scopes view
   (`view-web`) next to the editor. Demo URL shows: decorated `.tex`, layer
   toggles, legend from `Layer.defined`, resolution histogram ("unanchored is
   a result, not an absence").
3. **Anchor resilience.** Edit the `.tex` above a span in code-server; the
   span re-finds its text via `ResilientAnchor` (T-arxana-ts phase 4's
   check). This is D2's anchor cascade running client-side as a *preview* —
   per D7-DR2, promotable evidence still requires authority replay, which we
   cannot do until the service exists. The demo must label this distinction.
4. **Emacs alongside.** The same AnnotationSet JSON loaded by the Emacs
   Arxana client (or, minimally, the same scopes consumed by the existing
   Emacs link machinery), showing the same spans over the same frozen
   source. The M-jvm-arxana first demonstration's "same answer" check
   (canonical response digest) is the target form; for the demo the JSON
   digest is the stand-in.
5. **(Later, when INSTANTIATE lands the service.)** Swap the JSON `Store`
   adapter for an `arxana.v1` adapter (`scope/resolve`,
   `correspondence/query`, `revision/ingest`) and replay the demo against
   the authority — at that point this demo becomes D5's VS Code proof
   client's dress rehearsal.

No step touches futon1b, the DERIVE artifacts, or Emacs code beyond reading.

## 3. Client-side requirements for codex-21 (the deliverable this demo drafts)

Each item below records the confirmed/corrected form **as dispositioned by
codex-21 on 2026-11-07** (job `invoke-1793996309343-14300-e2a9d5fd`).
Baseline note: DERIVE closed at futon4 `f42b39f`, client proof-presentation
requirements at `0fd5ad1`, final ARGUE synthesis
(`A2-final-design-fitness-argument.md`) at `f19eafe`; this ticket's
`84f999d`/`37ba0dc` references are historical baselines and the obligations
named below are no longer open.

- **R1 (→ D3/D7) — confirmed with correction.** TS `Store` network transport
  targets `arxana.v1`, never direct futon1 writes. The AnnotationSet remains
  the client/domain payload inside a versioned envelope, but not wholly
  unchanged: `revision/ingest` carries canonical segment bytes/digest,
  artifact identity and expected head; the AnnotationSet must name the
  resulting artifact revision ID; `assertion/promote` requires the
  authority-produced D2 result/evidence digest; unverified/ambiguous
  promotion additionally requires attributable `resolution-acceptance`.
  Frozen JSON is fixture input, not an authority message.
- **R2 (→ D2) — confirmed with two corrections.** `exact` → `resolved`
  (`verified-hash`). `located` → `resolved-unverified` *only* when exactly
  one explicitly permitted fuzzy/advisory candidate remains; multiple
  candidates → `ambiguous`; disallowed fuzzy recovery stays `missing` with
  diagnostic evidence. `unanchored` → `missing` with typed reason. Named
  revision bytes failing their SHA-256 identity → `stale-input`; but a span
  content-hash mismatch inside otherwise correct revision bytes is *not*
  `stale-input` — it drives later anchor stages toward `missing`,
  `ambiguous` or `resolved-unverified`.
- **R3 (→ D1/D4) — confirmed.** `Segment` carries `artifact/revision-id`,
  not merely a checking hash. Preview IDs use an unmistakably local
  namespace (`client-preview:*`), never `arxana:manifestation:*`, never
  persisted or presented as registry identity; semantic spans reference
  authority-minted manifestation IDs once the service exists.
- **R4 (→ D5) — confirmed exactly.** Conversion uses digest-matched revision
  bytes, UTF-16 → Unicode-codepoint half-open ranges, rejecting positions
  inside surrogate pairs; decorations carry response/candidate IDs only;
  local caches are disposable projections.
- **R5 (→ D7-DR2) — confirmed with promotion boundary.** Client anchor
  recovery is labelled `preview`; promotion requires authority replay
  against immutable bytes returning a promotable result digest; the client
  may request promotion but cannot promote its own preview evidence.
  Attributable acceptance applies to ambiguous/unverified evidence, not
  ordinary uniquely verified resolution.
- **R6 (→ D5/O-D7d) — already settled.** The canonical semantic projection
  includes protocol/semantics version, operation, domain status, revision
  IDs, ordered candidate IDs and canonical locations, verification states
  and semantic evidence digests; excludes request/message ID, principal,
  timestamps, duration, transport/trace metadata, presentation strings and
  operational diagnostics. The pre-service JSON digest is a fixture
  stand-in, labelled as such, not evidence of D5 semantic equality.
- **R7 (→ O-D7f) — confirmed.** The pre-service demo performs no
  futon1/futon1b writes; it reads frozen JSON and local source only. Direct
  futon1 adapters may support compatibility experiments but must not write
  `arxana/*` records (now reserved at futon1b for authenticated conditional
  writes with atomic uniqueness validation).

## 4. What this demo is not

Not the M-jvm-arxana first demonstration (that needs the headless service
and the repaired draft8 fixture); not a VS Code product decision; not a
second ARGUE or DERIVE artifact. It is consumer-side evidence: the first
VS Code-class client, feeding requirements back into the design it consumes.

## 5. Evidence to record when work starts

- a00J02 parity numbers (520 spans, 9 edges, 493 marks) reproduced through
  `substrate-vscode` in code-server;
- anchor-recovery results after scripted edits (per T-arxana-ts phase 4);
- the R1–R7 confirmations or corrections from codex-21 as they land.
