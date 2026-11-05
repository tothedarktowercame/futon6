# T-arxana-ts — a TypeScript Arxana over mark8 scopes

**Type:** T-prefix ticket (design outline, no code yet). **Raised:** 2026-11-05 by Joe,
with Claude. **Status:** open. **Companions:** `T-neo4j-scope-store.md` (store side),
`M-informal-lean-alignment.md` (the Lean substrate this is meant to serve),
`futon4/README-essays.md` (the file-role convention this keeps).

## 0. What is being asked

Mark up any paper with offset annotations, the way futon4's Arxana does with
hyperedges and link strategies, and show the result alongside or on top of any
rendering of that paper: a LaTeXML page (as `/wip/apm-mark8/a00J02/a00J02-margin.html`
does today), a page from latexml-oxide, a page produced from Lean via LaTeX
(the mfuton-share blueprint lane), or the source itself open in VS Code. Do it in
TypeScript so the same model runs in a browser, in a VS Code extension, and in a
CLI, and so that adapters can be written by people who do not use Emacs.

## 1. What exists, as found in the three repos

### 1.1 futon6: the mark8 margin page is already three offset layers plus a bake

`scripts/render_scope_margin.py` reads, per paper, three artifacts a run wrote:

| Layer | Artifact | Addressing | What it is |
|---|---|---|---|
| S1 marks | `artifacts/marks/fable-<paper>-dp-emacs.json` | `[start, end)` codepoint offsets into `marks.text` | symbols, binders, quantifiers, environments, concepts (`MARK_KINDS`, ~45 kinds) |
| S3 proof graphs | `artifacts/graphs/<paper>__p*.edn` + `candidates/*.candidate.json` | nodes cite S1 clause units by id; graph has `source.lines` | claim / object / definition / ref nodes; edges with warrant stated / cited / missing |
| S4 scopes | `artifacts/expo/*`, audited by `expository_scope_audit.py` | line spans, plus a located `span` when the fill is quoted verbatim | expository scope kinds (`connection/...`, `rationale/...`) with slot fills and mechanical flags |

The page is built by: checking the typeset `.tex` is byte-identical to the run's
`marks.text` (the source of truth for offsets); converting lines to offsets with a
`starts` table; and embedding three JSON blocks into the LaTeXML tufte page:

- `m7-data`: summary, notes (proof / region / refused), marks as `[start,end,kind,grounded,tip]`, glossary;
- `m7-source`: the source text itself;
- `m7-scopes`: the output of `scope_margin_scopes.py`, a **renderer-neutral** form:
  `segments` (id, role, text), `layers` (id, name, ink), `spans` (id, seg, start, end,
  layer, kind, weight, note), `edges` (from, to, relation, label), `legend`,
  `offset_unit: unicode-codepoints-zero-based-end-exclusive`, and a `spot_check`
  that counts drawn versus summarised items.

Three browser scripts then do the work at view time: `margin.js` (margin notes by
block, every label resolved through the glossary or flagged UNDEFINED), `plates.js`
(places every rendered word and formula at a source offset using `data-sourcepos`
plus a bounded word search in the source, then inks cyan / magenta / yellow plates
that mix on overlap), and `futon3c/web/scopes/scopes.js` (the shared scope view,
consuming `m7-scopes`; not in this checkout).

The live a00J02 page, fetched 2026-11-05, has all three blocks: 493 S1 marks,
520 spans, 9 edges, 35 legend entries, 77 `data-sourcepos` elements, source 7,769
codepoints. Its one segment is still named `mark7-0708.2185`: `scope_margin_scopes.py`
hard-codes the segment id. Harmless on a one-paper page, fatal once segments are
shared across papers; it is the first thing the TS importer must not copy.

Two invariants in this code are worth carrying over verbatim:

1. **Every label on the page has a definition** (`MARK_KINDS`, glossary; an undefined
   kind stops the build, an undefined pill is shown as undefined, not hidden).
2. **Unanchored is a result, not an absence** (`view.js` says "Unanchored: no
   rendered element starts within these source lines"; refused and deferred
   passages get notes where the passage is, not a blank margin).

### 1.2 futon4: the Arxana model is hyperedges, tiers of links, and resilient anchors

`dev/arxana-links.el` states the model in its header:

- link tiers: **ephemeral** (computed by finders, not persisted), **strategy-bound**
  (the finder rules persist, links are recomputed), **anchored** (exact regions
  persist, for immutable content);
- entities persisted to futon1 (XTDB) over HTTP: `arxana/link-strategy`
  (`:scope` + `:finders`), `arxana/voiced-link`, `arxana/surface-form`,
  `arxana/scholium`, `arxana/embedding-cache`;
- a **resilient anchor** is `content-hash + content-length + context-before +
  context-after + hint-offset + hint-line`, re-found by exact offset, then
  context pattern, then fuzzy context (`arxana-links-find-anchor`).

Annotation hyperedges have endpoints with a role, an entity id and a passage
(`arxana-browser-annotations.el`); passages may carry `line N:` / `lines A-B:` prefixes.
The essays convention (`README-essays.md`) is the discipline this design keeps:
the source is canonical and read-only, annotations live in a separate data file
anchored by verbatim passage, and an annotated edition is a render, never a baked file.

The WebArxana surface (`dev/web/webarxana`, ClojureScript + shadow-cljs, served by
the futon3c JVM) is a graph and card UI over the store; it does not do in-text
overlays. So the TS work is not a rewrite of WebArxana; it is the in-text layer
WebArxana lacks, and it should talk to the same store.

### 1.3 mfuton-share: the Lean side gives declarations with source lines and a proof graph

- `src/mfuton/lean/entity_surface/source_extraction/` recovers declaration
  boundaries and source lines per `.lean` file (`declarations_declaration_boundaries.py`,
  `lean_source_declarations_source_lines.py`, byte masks for the indexer).
- `src/mfuton/lean/proof_graph/graph_core/` writes a typed graph artifact
  (node-link JSON via the networkx adapter; `declaration_node_id`), lifted onto a
  generic typed `Graph` spine.
- `src/mfuton/lean/blueprint_render/uses.py` computes the blueprint `\uses` graph
  and `\leanok` from the neutral `Expr` surface.

So a Lean project already yields: segments (`.lean` files), spans (declaration
line ranges), and hyperedges (`\uses`, proof-graph edges). A Lean-to-LaTeX page
built from the blueprint is a second rendering of the same segments.

## 2. The argument

```
IF:       one annotation set, authored once, should display over any rendering
          of the paper (LaTeXML, latexml-oxide, Lean blueprint page, VS Code buffer)
          and be authored from any of them.
HOWEVER:  today the anchoring is done at build time in Python for one rendering
          (plates.js is specific to LaTeXML's data-sourcepos and tufte layout);
          the annotation model lives in Emacs Lisp against XTDB; and the two
          data shapes (m7-scopes spans, Arxana scholia with resilient anchors)
          are not the same type.
THEN:     build a small TypeScript core whose only primitive is "a span of a
          segment's text, in codepoint offsets, carrying a kind and a layer", with
          hyperedges over spans; put everything rendering-specific behind a
          Substrate interface that answers one question, offset -> DOM Range (or
          offset -> editor Range); put everything storage-specific behind a Store
          interface; and make m7-scopes the first wire format the core reads.
BECAUSE:  the one thing all of these surfaces share is the source text and
          positions in it. The margin page already proves this: it embeds the
          source, checks the typeset .tex byte-for-byte against it, and derives
          every pixel from offsets. Everything else (margin notes, ink plates,
          cards, decorations) is a projection of spans and edges and can differ
          per surface without changing the data.
```

Verification against evidence: the a00J02 page already runs end to end on
`m7-scopes`, so a TS renderer that reproduces that page from the same block is a
parity test with known numbers (`spot_check`). The resilient anchor in futon4 is
what makes the same data survive an edited `.tex` or `.lean` in VS Code; the m7
pages never needed it because their source is frozen by the run.

## 3. Core model

Written as TypeScript so the outline is also the first file of `@arxana/core`.

```ts
// Offsets are Unicode codepoints, zero-based, end-exclusive — the unit m7-scopes
// already declares. JS strings index UTF-16 code units, so the core carries an
// index per segment and never lets a UTF-16 index leak into stored data.
export type OffsetUnit = "unicode-codepoints-zero-based-end-exclusive";

export interface Segment {
  id: string;                 // stable, unique per corpus: "apm-mark8/a00J02/tex"
  role: "paper" | "lean" | "markdown" | "note" | string;
  label: string;
  text: string;               // canonical source; read-only once published
  hash: string;               // sha256 of text; a substrate must match or re-anchor
  lines?: number[];           // start offset of each line (computed, cached)
}

export interface Layer {
  id: string;                 // "s1-term", "s3", "s4", "scholia"
  name: string;
  ink?: "c" | "m" | "y" | "k" | string;   // plates: ink separation colour
  defined: Record<string, string>;        // kind -> meaning; every kind a span uses MUST be here
}

export interface Span {
  id: string;
  seg: string;                // Segment.id
  start: number; end: number; // codepoints into Segment.text
  layer: string;              // Layer.id
  kind: string;               // must be a key of Layer.defined
  weight?: number;            // 0..1, how strongly to ink
  note?: string;
  anchor?: ResilientAnchor;   // optional: lets the span survive edits to the text
  meta?: Record<string, unknown>;   // S3 gloss/verdict, S4 fill/flags/mock, etc.
}

// Port of arxana-links-make-anchor / find-anchor.
export interface ResilientAnchor {
  strategy: "content-hash";
  contentHash: string; contentLength: number;
  contextBefore: string; contextAfter: string;      // ~30 chars each
  hintOffset: number; hintLine: number;
}

// Hyperedge: roles over spans (or segments, or other edges), as in futon4's
// annotation hyperedges. An S3 inference step is one of these: premises -> conclusion
// with a warrant. A scholium is one: {target: span, body: segment("note")}.
export interface Hyperedge {
  id: string;
  type: string;               // "iatc/step", "arxana/scholium", "blueprint/uses", ...
  ends: Array<{ role: string; ref: string }>;        // ref = Span.id | Segment.id | Hyperedge.id
  label?: string;             // e.g. warrant kind: stated | cited | missing
  meta?: Record<string, unknown>;
}

export interface AnnotationSet {
  offset_unit: OffsetUnit;
  record: string;             // run id / manifest id this came from
  segments: Segment[]; layers: Layer[]; spans: Span[]; edges: Hyperedge[];
  spot_check?: Record<string, Record<string, number>>;
}

// Where a span lands in a given rendering. Unresolved is a value, not an exception.
export type Resolution =
  | { kind: "exact";    ranges: Range[] }               // text hash matched; offsets used directly
  | { kind: "located";  ranges: Range[]; via: "sourcepos" | "anchor" | "align"; confidence: number }
  | { kind: "unanchored"; reason: string };

export interface Substrate {
  id: string;
  segmentFor(id: string): Promise<Segment | undefined>;  // the text this surface shows
  resolve(span: Span, seg: Segment): Resolution;
  paint(spans: Array<[Span, Resolution]>, layers: Layer[]): void;   // decorations / highlights
  onSelect?(cb: (range: Range) => void): void;          // for authoring
}

export interface Store {
  load(query: { segment?: string; record?: string; layer?: string }): Promise<AnnotationSet>;
  put(items: Array<Span | Hyperedge | Layer | Segment>): Promise<void>;
  // strategies persist here too (futon4 tier 1): finders that recompute ephemeral spans
}

export interface Strategy {
  id: string;
  scope: { segments?: string[]; roles?: string[] };
  finders: Array<{ type: string; autoLink: boolean; [k: string]: unknown }>;
}
```

Mapping of today's data onto this: an `m7-scopes` block is an `AnnotationSet`
with one segment and six layers; the glossary becomes `Layer.defined`; S3 `edges`
(from, to, relation, label) become `Hyperedge{type:"iatc/step", ends:[{premise},{conclusion}], label: warrant}`;
futon4 `annotations.edn` entries become `Span{layer:"review"}` plus a scholium
hyperedge whose body is a `Segment{role:"note"}`; a Lean proof-graph artifact becomes
segments per `.lean` file, spans per declaration, hyperedges per edge.

## 4. Anchoring: the one hard part, and where each piece already exists

Resolution runs in order and stops at the first success; each level is a port of
code that exists.

| Level | When | Port of | Output |
|---|---|---|---|
| exact | substrate exposes the same text (hash equal), e.g. a VS Code buffer of the `.tex`, or a page that embeds `m7-source` | `render_scope_margin.build`'s byte check | `exact` |
| sourcepos | rendered HTML carries `data-sourcepos` (LaTeXML; check whether latexml-oxide does) | `plates.js`: block start from sourcepos, word placement by bounded search, formula placement by the enclosing S1 `math` mark or `alttext` similarity | `located/sourcepos` |
| anchor | text drifted (edited buffer, different TeX revision) and the span has a `ResilientAnchor` | `arxana-links-find-anchor`: hint offset, then context pattern, then fuzzy | `located/anchor` |
| align | no positions at all (any HTML, PDF text layer, latexml-oxide without sourcepos) | new: align rendered text to `Segment.text` with a diff (diff-match-patch style), then map offsets through the alignment | `located/align` |
| none | | `view.js` | `unanchored` with reason, drawn in the margin |

Two implementation notes for the web substrate:

- Prefer the CSS Custom Highlight API (`CSS.highlights`, `Highlight` of `Range`s)
  over wrapping words in `<span class="m7-u">` as `plates.js` does. It leaves the
  DOM untouched, which matters once a page is also being edited or re-rendered,
  and it is supported in current Chromium, Safari and recent Firefox. Keep the
  span-wrapping path as a fallback.
- Keep the plates idea (ink separations, overlap mixes) as one `paint` style
  among several: plates, margin notes, inline underlines by strategy. The view
  should not know which layer is "S1"; it knows ink colour and `defined`.

## 5. Packages

A pnpm workspace, strict TypeScript, no framework in core.

```
arxana-ts/
  packages/
    core/            types above; codepoint<->UTF-16 index; line table; validate()
                     (every span inside its segment; every kind defined; every edge
                     end resolvable); resilient anchors; alignment resolver
    formats/         read: m7-scopes JSON, futon6 marks JSON, futon4 annotations.edn,
                     mfuton proof-graph node-link JSON, blueprint uses;
                     write: AnnotationSet JSON, futon1 entity payloads
    substrate-dom/   Substrate over any HTML: sourcepos resolver (port of plates.js),
                     align resolver, CSS Highlight painter, span-wrap painter
    substrate-vscode/ Substrate over vscode.TextDocument: exact + anchor resolvers,
                     TextEditorDecorationType painter
    view-web/        the page UI: layer toggles, legend from Layer.defined, margin
                     notes, plates; a single <script type=module> drop-in that
                     replaces margin.js + plates.js + scopes.js on an existing page
    vscode/          the extension: decorations, hover with span.note and glossary,
                     CodeLens per S3 proof / S4 region, a webview showing the proof
                     graph, "create scholium from selection" (makes a ResilientAnchor)
    store/           Store over JSON files; Store over futon1 HTTP (/entity, X-Penholder,
                     X-Profile, as arxana-store.el does); later Neo4j (T-neo4j-scope-store)
    cli/             arxana import <m7-page.html|run-dir>; arxana render <page> <set>;
                     arxana check <set> (spot_check parity; validate()); arxana anchor
                     (add ResilientAnchors to a set against its segment text)
```

The Python side keeps producing the run artifacts; only `scope_margin_scopes.py`'s
role moves to TS (`formats`), and `render_scope_margin.py` shrinks to "embed the
AnnotationSet and the module script". Graph artifacts are EDN; either Python emits
a JSON twin beside each `.edn` (preferred, one line in the writer) or `formats`
carries a small EDN reader. Do not reimplement the audits (`expository_scope_audit`,
`iatc_quote_check`) in TS; their verdicts arrive in `Span.meta`.

## 6. Phases, each with a check that can fail

1. **Parity.** `core` + `formats/m7` + `substrate-dom` + `view-web`. Feed the
   a00J02 page its own `m7-scopes` block through the TS module and reproduce the
   plates and margin. Check: drawn counts equal `spot_check.drawn` (520 spans,
   9 edges, 493 marks); word-placement rate not below `plates.js`'s on the same
   page; every legend label defined. Fix the hard-coded segment id on import.
2. **Second rendering, same data.** Run the same set over a latexml-oxide page of
   the same `.tex` (sourcepos resolver if it emits positions, else align). Check:
   the share of spans resolved `exact|located` on the second page, reported, not
   assumed.
3. **VS Code.** `substrate-vscode` + `vscode`: open `a00J02.tex`, load the set from
   a sibling JSON, see decorations and hovers; CodeLens on each proof opens the
   graph webview. Check: the extension marks every S3 node's quoted range.
4. **Authoring.** Select a range in VS Code or on the web page, create a scholium
   (span + hyperedge + note segment) with a `ResilientAnchor`; save to JSON; edit
   the `.tex` above it; reload; the scholium re-finds its text. Then the futon1
   store adapter, so the same scholium is visible in Emacs Arxana.
5. **Lean substrate.** `formats/mfuton`: segments per `.lean`, spans per declaration,
   hyperedges per `\uses` and proof-graph edge; overlay on a blueprint-derived
   LaTeX page and on the `.lean` in VS Code. This is where mark8 scopes over an
   informal proof and Lean declarations of the same result sit on one page
   (`M-informal-lean-alignment`). Check: a span pair (informal passage, Lean
   declaration) linked by a hyperedge resolves on both substrates.
6. **Strategies.** Port two futon4 finders (symbol-as-term, surface-form) as
   ephemeral spans computed in TS; persist the strategy, not the spans.

## 7. Decisions to take now

- **Offsets are codepoints** in all stored data, as `m7-scopes` already says.
  `core` owns the conversion; substrates never see UTF-16 indices.
- **The source text travels with the set** (as `m7-source` does today) or is
  fetched by `Segment.id`; either way `Segment.hash` is checked before `exact`.
- **No baking.** Annotations are never written into the `.tex`, the HTML body, or
  the `.lean`; a page is source rendering + one embedded or fetched AnnotationSet
  + one module script. Same rule as `README-essays.md`.
- **Unanchored is drawn.** Every phase reports its resolution histogram.
- **Store last.** JSON files first; futon1 when authoring needs to round-trip to
  Emacs; Neo4j only for the cross-paper path queries `T-neo4j-scope-store` names.

## 8. Not in scope

Replacing WebArxana's graph and card UI; re-running any S1/S3/S4 stage in TS;
PDF annotation (a PDF text layer is just another `align` substrate later);
Emacs changes.

## 9. Evidence to record when work starts

The devmap for this lives in `../futon3/holes/futon6.devmap` (not in this
checkout). Add evidence lines of the form
`evidence[a00J02 parity: 520/520 spans drawn, 9/9 edges, words placed N/M]` as
phase 1 lands; the numbers above are the baseline measured from the live page on
2026-11-05.
