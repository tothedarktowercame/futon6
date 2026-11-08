# M-jvm-arxana code-server demo report (T-codeserver-arxana-demo, implementation slice)

> **Closed 2026-11-08 (already done).** This is a delivered implementation report, not open work: the VS Code / code-server extension is in `tools/arxana-vscode/` and its node tests pass. Evidence: futon6 e72a973, 5ee1c6a, 8e7e770 (2026-11-07); `node --test test/*.test.js` in `tools/arxana-vscode` on 2026-11-08: 35 tests, 35 pass, 0 fail. Checked by misc in the closure pass (claude-6, FIX/WONTFIX sheet).

Date: 2026-11-07. Agent: zai-10 (dispatched by codex-21). Status: delivered; done, closed 2026-11-08, see the closure note.

## Placement decision

The extension lives additively in **futon6 `tools/arxana-vscode/`**. No
clearer owned repo exists: futon4a owns the Emacs client, futon3c owns the
service, and futon6 hosts `T-codeserver-arxana-demo.md` and the draft8 paper
sources. Nothing in futon6's existing trees is touched; deleting the one
directory rolls the demo back.

## What was built

A plain-JS VS Code extension (no npm dependencies; loads in code-server
4.145.0 and desktop VS Code alike) plus pure lib modules and node tests:

- `lib/client.js` — arxana.v1 envelope client; typed client refusals
  (`client/unavailable`, `client/http-status`, `client/unparseable-response`);
  authority `error` envelopes pass through unchanged.
- `lib/utf16.js` — UTF-16 ↔ Unicode-codepoint half-open ranges; a coordinate
  inside a surrogate pair is refused (`arxana.client/mid-surrogate`), never
  rounded.
- `lib/digest.js` — digest-matches an open document's exact bytes to the
  fixture's committed revision ids; unmatched bytes are a typed refusal,
  never a guessed revision.
- `lib/view.js` — pure view models + webview HTML. All candidates displayed;
  ambiguity sets `requiresChoice`; no `selected`/`chosen` field exists in the
  model. Object-qualified refusal labels (`resolution: refused`).
- `extension.js` — commands `arxana.showCapabilities`, `arxana.resolveScope`
  (digest-match → selection conversion → scope/resolve → panel + candidate
  decorations carrying candidate IDs only), `arxana.correspondence`,
  `arxana.openCorrespondenceTarget` (reveals the rendered target line),
  `arxana.goToCandidate` (explicit user choice only).
- `fixtures/draft8-v1.json` — fixture stand-in config (R1): artifact
  revisions copied from the futon5 gate `17dc53a` manifest; strategy revision
  copied from the committed futon4a demo transcript; provenance recorded
  inline.

## Executed evidence

- Unit tests: **18 tests, 18 pass, 0 fail** (`node --test test/*.test.js`),
  covering: capabilities advertisement + stale-capability refusal; endpoint
  unavailable → typed refusal; recorded authority envelopes round-trip
  (resolve semantic digest `0397248a…`, stale-ingest recovery choices);
  ambiguity fixture → both candidates rendered, none selected; astral
  Unicode conversion + mid-surrogate refusal; real fixture bytes digest-match
  their revisions (and edited bytes refuse); the authority range 792–830
  round-trips and slices to exactly "The object of study is a finite family".
- Live acceptance (`node scripts/live-demo.js`) against the isolated JVM
  service (`futon3c.arxana.demo` on `127.0.0.1:7084`, started only after
  `check-demo-ports.sh` said AVAILABLE): **ALL LIVE CHECKS PASS** —
  scope/resolve and correspondence/query advertised; resolution matches the
  fixture revision with candidate range 792–830, evidence stages and semantic
  digest equal to the committed transcript; correspondence is directional
  (`source-to-target`), `partial`, method quote+source-map, producer
  `latexml_oxide 0.7.5` build `arxana-build:sha256:7ae3821ffcf2265d`, with
  navigation projections (source codepoint 792 → `intro-generated.tex`;
  target line 10 → gated `draft8.html`). Rendered views captured in
  `tools/arxana-vscode/demo-captures/*.html` (regenerable via the script).
- code-server load proof: a **private** code-server instance
  (`--extensions-dir`/`--user-data-dir` under `/tmp`, port 8097 fresh-censused,
  loopback, auth none) enumerated the extension as
  `futon6.arxana-codeserver-demo` (`code-server --list-extensions`). The
  shared code-server on :8090 (nREPL/Marimo setup) was never restarted or
  modified. The instance was stopped and its data deleted afterwards.

## Principal findings

1. **The live JVM and the committed transcript agree byte-for-byte on
   semantic digests through an independent TS/JS client** — the D5
   cross-client equality property held for a third client with zero shared
   code (resolve `0397248a…`, correspondence `9c91f35e…`).
2. **`page/draft8.html` has drifted from the gated revision**: its sha256 is
   `ceab935f…`, not `7ae3821f…`; the authority ingests
   `arxana-fixture/draft8.html`. Any client navigating the rendered target
   must read the gated copy or it will digest-mismatch. Recorded in the
   fixture config and README.
3. **code-server vs desktop extension hosts need no different authority
   behavior** (expected: no). The extension uses only `fetch` to loopback and
   the webview/decoration APIs; nothing is code-server-specific.

## Limitations

- Browser-level interaction (clicking commands, seeing decorations) was
  proven only by extension enumeration + headless execution of the same code
  paths; no screenshots. The webview HTML itself is a pure function of the
  view model and is captured in `demo-captures/`.
- `scope/resolve` resolves the fixture's declared subject
  (`draft8/introduction`); the editor selection is converted (UTF-16 →
  codepoints, against digest-matched bytes) and marks containing candidates
  but does not mint a new subject. Selection→new-scope declaration is future
  authority work (R5 promotion boundary), out of this slice.
- Target navigation lands on the source-map line; the fixture's
  `:source-map-coordinate` locator carries file-index/line only, so highlight
  granularity is per-line, as far as the fixture permits.
- No local anchor-recovery preview (ticket §2 step 3) was implemented —
  `client-preview:*` anchoring stays out until needed; nothing here is
  promotable evidence.

## Runtime state left behind

None. JVM service (7084) and the private code-server (8097) were stopped and
verified gone (`check-demo-ports.sh`: AVAILABLE arxana/store/drawbridge);
`/tmp/arxana-cs-demo` deleted; shared code-server on :8090 untouched and
still running. `/tmp/futon4a-demo-jvm.log`-style demo logs were written to
`/tmp/arxana-codeserver-demo-jvm.log` (harmless).

## Commit

See `git log` for the exact SHA (committed on futon6 master together with
this report and the `T-codeserver-arxana-demo.md` status update).
