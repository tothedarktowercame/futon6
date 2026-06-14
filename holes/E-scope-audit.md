# E-scope-audit — joint mission-mode scope audits (labeled misses → detector improvements)

**Type:** E-prefix excursion. **Spawned:** 2026-06-10 (Joe + Fable), from the
capability-map review shortlist. **Method:** Joe opens a mission in
`mission-mode`; we jointly inspect whether all relevant scopes are ascertained;
every miss is classified (*binder-miss / anchor-drift / parent-break /
vocabulary-gap*) and becomes a labeled example for improving
`futon6/scripts/mission_scope_detect.py` + the ingest.
**Pipeline:** `mission_scope_detect.py` (md→tree) → `mission_scope_ingest.clj`
(tree→futon1a:7071) → `mission_scope_view.clj` → `mission-mode.el` overlays.

## Session 1 — M-memes-arrows-patterns-diagrams (2026-06-10)

### Fixed this session
1. **Stale anchors after doc growth** (26 scopes ingested mid-life; doc then
   grew 3×). 7/26 anchors no longer matched; several *mis-anchored* — verbatim
   search has no uniqueness guard, so edited-away passages re-match elsewhere
   (an ARGUE phase scope resolved onto the `Status:` header line; a §7 scope
   onto §2.1). → re-detect + re-ingest; all 26 verbatim-fresh.
2. **PSR/PUR regex too narrow** (Joe's line-482 catch). The discriminator
   `([a-z0-9][a-z0-9-]{6,})` rejected backtick-quoted and namespaced idents —
   the house style (`` `structure/two-projections-of-one-quantity` ``). Fixed
   (`` `? `` + `/` in class) → psr=2, pur=1 detected (one PSR + one PUR we
   didn't know about).
3. **psr/pur scopes ingested anchor-less.** They skipped every `stable-*`
   enrichment (binder dispatch fell through to bare `(vec scopes)`) so no
   `:anchor` was ever computed; also `anchor-for-scope` had no branch for the
   detector's `anchor-line` key. Added `stable-record-scopes` + a
   `("psr" "pur")` anchor branch (record line = anchor). Verified live: psr
   chip renders at the `Pattern chosen:` line, nested in section + phase.
4. **mission-mode UI rebuilt** to Arxana/showcase conventions (was: bare
   `@scope` text markers): depth-tinted nested region overlays (batch-008
   palette), per-binder colored label chips, cursor-over posframe
   (cached require; echo one-liner fallback), `C-c m i` inspect side window,
   `C-c m o` linked overview panel (point motion scrolls the mission). Keymap
   moved off `C-c C-m` (= `C-c RET`).

### Session 1 addendum (later 2026-06-10)
5. **Phase scopes mis-anchored to the Status line** (Joe's "Scratch-Map is
   missing Argue/Document" catch). Ingest's `heading-passage` re-derived
   anchors from title text with a `str/includes?` fallback — the Status line
   names every phase and first-occurrence wins, so IDENTIFY/ARGUE/VERIFY all
   anchored to line 4 (and deduped into one overlay). **Fix:**
   `content-position-passage` — the ingest now trusts the detector's
   `hx/content :position` (line-at-offset) and only falls back to heading
   search. All 7 phases verified at true headings. (This closes most of W5's
   practical surface; the uniqueness guard for *re-resolution* remains.)
6. **Downstream lane added to the overview panel** (`↓ code` from
   `code/v05/file→mission`, `↔ missions` from cross-refs; RET visits; JSON via
   the :3100 proxy — futon1a speaks EDN; remember `decode-coding-string` on
   url-retrieve bodies or `→` in type names won't match).
7. **Code↔scope cross-linkage (interaction, not ink):** each code line carries
   `scope-refs` (scope regions whose text mentions the file basename) — point
   on a code line box-highlights the referencing scope blocks in the panel and
   scrolls the doc to the first mention. Mission-grain approximation pending W8.
8. **Ghost lines:** canon phases absent from the doc render as `∅ PHASE`
   ghosts interleaved in canon order (here: DOCUMENT).

### Session 1 addendum 2 (2026-06-10, late)
9. **W3 DONE — `###` sections bind** (`mission_scope_detect.py` level ≤ 3):
   loose-sections 9 → 58 on this doc; handoffs/VERIFY hooks/ARGUE rounds all
   scoped with parent links. Side-regression to chase: capability-scope 3→1,
   map-item 2→0 (sub-binder context shifted under ###); old canonical rows
   still serve the good data.
10. **In-passing phase closures detected** (Joe: DOCUMENT closed at L1567, not
   as a section). New `INLINE_PHASE_CLOSURE_RE` (`**PHASE phase:** satisfied/
   closed/done`) emits an eightfold scope anchored at the closure line.
   Ghost-lines now only render for genuinely absent phases.
11. **Panel folding**: phase blocks fold subsections (default folded, 105→45
   visible lines); point-on-parent peeks open; TAB pins; `t` toggles all.
12. **HEAD concept lane wired**: panel shows the detector's concept terms
   under HEAD (`⊙ agents · patterns · …`). Quality is kernel-unigram only —
   misses "meme graph", "pattern cascades", "futon3b" — see W10.

### Open findings (worklist, in priority order)
- **W12 — watcher-integrated scope lane.** The scope pipeline is manual
  (detect.py + ingest); the file watcher only does the mission-doc lane. Pace
  fixed for now via Drawbridge: `scripts/mission-scope-reingest.sh` (~1.5s) +
  `scripts/mission-scope-view-fast.sh` (~1.8s; calls view inner fns — its
  `-main` System/exit's, fatal through Drawbridge) + `C-c m R` in mission-mode.
  Real fix: watcher triggers the scope lane on mission-doc change (debounced).
  Greenfield demo: `futon6/holes/missions/E-mission-head.md` (E-prefix now
  recognized by mission-mode).
- **W10 — HEAD concept extraction is unigram-noise; upgrade to
  Interest-Network-grade** (multi-word entities, repo names, in-stack
  vocabulary). The display lane exists; better extraction drops straight in.
- **W11 — HEAD → AIF terminal vocabulary mapping** (the formal half of
  [[project_head_dual_reading]]): mission = virtual peripheral, HEAD should map
  to priors/preferences/observations so "satisfying the mission" is typed.
  Direction, not yet a build.
- **W2′ (SHARPENED) — plain full-run ingest skips ALL `stable-*` enrichments**
  (not just psr/pur): it writes raw `scope-NNN` rows with no anchors and no
  stable ids; only `--binder <b>` runs enrich. Until fixed, refresh missions
  binder-by-binder. Root cause in the full-run dispatch path; belongs with W1.
- **W8 — scope-grain downstream.** `file→mission` edges are mission-grain; the
  panel's basename-mention refs are an approximation. Real fix: file ends on
  ###-level scopes (handoffs), so code nests *inside* the blocks that own it.
  Depends on W3.
- **W1 — store accumulates id-generations; no GC.** Canonical-id rows (earlier
  ingest era) + raw `scope-NNN` rows coexist (67 rows for ~29 real scopes);
  plain re-ingest never retracts superseded rows (`--binder` legacy-retract is
  a narrower migration path). Mitigated client-side (mission-mode dedupes by
  type+passage); **proper fix = ingest-side retraction of rows not in the
  current ingest set.** Belongs with the endpoint-identity/rewording seam.
- **W2 — full-run vs `--binder` dispatch discrepancy.** The new
  `stable-record-scopes` enrichment fires via `--binder psr|pur` runs but did
  NOT via the plain full-mission run (old raw-id rows were upserted instead).
  Understand before trusting plain re-ingests for psr/pur. (Same mystery may
  explain W1's generation split.)
- **W3 — `###`-level structure invisible** (~47/57 headings unbound on this
  doc): H1–H6 INSTANTIATE handoffs (natural `mission-scope-out` objects —
  that binder fired 0), VERIFY hooks 11.1–11.5, ARGUE 10.x incl. the
  adversarial check. Detector currently binds `##` only.
- **W4 — `relates-to` / `source-material` fired 0** despite the `Related:`
  header, `[[memory]]` links, and cited artifacts (BHK-research.md, sorrys.edn,
  filament.clj). The binder exists; emission/anchoring path untested.
- **W5 — anchor uniqueness.** Verbatim-search should carry a disambiguator
  (heading-path or occurrence index + fingerprint check) so re-matches after
  edits can't silently land on the wrong line (the `Status:`-line mis-anchor
  class).
- **W6 — view/store/map snapshot skew.** The futon6 JSON dump, the live store,
  and the rendered district map were three different snapshots (map said 37
  scopes; dump 26; store 67). Re-render the map after ingest fixes, or stamp
  artifacts with ingest generation.
- **W7 — concept-end quality.** `concept` ends look like keyword-grabs
  ("globe", "argue", "phase") — same generic-grab failure mode as
  M-prior-mathematics flagged for NER. Low priority; noisy not harmful.

### Session 3 — the Skolem audit (2026-06-11, Joe's "empty scope is suspect")

13. **W13 NEW INSTRUMENT — `mission_scope_bindings.py` (Skolem audit).** Joe's
    observation: an empty scope is `∀x:` with no body; and if the body never
    uses what MAP binds, the map was decoration ("vibe coding"). Three checkable
    classes over any scope tree: *vacuous-binder* (scope with no content ends),
    *unused-binding* (item bound in HEAD/IDENTIFY/MAP, untouched by any body
    phase), *free-variable* (item used in a body phase, never introduced). Two
    channels — detector ends vs literal text in phase regions — and the
    disagreement IS the diagnostic: both-channels = real binding failure in the
    mission; ends-only = detector blindness (a W-class finding). Tests in
    `tests/test_mission_scope_bindings.py`.
14. **Fleet baseline (204 trees, pre-W3-fix detector):** 611 vacuous scopes,
    1,446 confirmed unused bindings, 78 confirmed free variables, 83 missions
    with no binder/body spine. The typical mission binds dozens of items in MAP
    and re-uses ~0–3 in the document's own body. Caveat: "use" often happens in
    code/commits rather than re-citation, so an unused binding at doc grain is
    an *undischarged citation*, not automatically a vice — but the near-zero
    used-counts say MAP currently functions as inventory, not as a binder.
    Spot-verified by hand (M-war-machine-pilot: `arxana-vsatarcs-belief.el`
    bound at MAP L584, absent from the entire body).
15. **W3 side-regression FIXED** (the addendum-9 chase item, quantified by the
    audit as E-mission-head MAP binding *nothing*): `###` sections binding as
    loose-sections masked the eightfold phase above them, so
    map-item/capability/psr-pur sub-binders saw phase "loose" and never fired.
    New `current_phase_context` (nearest non-loose phase on the stack) restores
    them: E-mission-head map-item 0→3, capability 0→1, MAP bound 0→12 items.
    Only E-mission-head re-detected; **fleet re-detect + re-ingest pending**
    (W1/W6 store-generation discipline — don't refresh trees the store still
    anchors to without retraction).
16. **E-mission-head Skolem findings (fresh tree):** (a) the three M-aif-head
    autopsy artifacts (`mission_head.clj`, `observe.clj`, `invariant.clj`) are
    confirmed-unused — MAP exhumed them and DERIVE built the sigil path without
    wiring them: the revive-vs-replace risk IDENTIFY itself flagged, now caught
    mechanically; (b) 6 confirmed free patterns, including
    `two-projections-of-one-quantity` — the audit independently rediscovers the
    Anatomy paper's §5.3 admission (leaned on in ARGUE, never introduced).

### Open findings (session 3 additions)
- **W14 — the transducer channel (mission output-tape attribution).** A mission
  is a Mealy-style transducer (Joe, 2026-06-11): doc = input tape + receipts,
  code graph = output tape. The output tape (`code/v05/edits`) is live but
  UNATTRIBUTED — no commit→mission edge exists; `file→mission` is
  `mission/mentions-file` (doc-grain, circular as discharge evidence). Design +
  Codex handoff spec: `futon6/holes/missions/E-mealy-style-transducer.md`
  (new `code/v05/commit→mission` edge, trailer + session-heuristic provenances,
  audit channel 3 → verdicts doc-used / code-discharged / confirmed-unused).
  Dispatch pending Joe's go.

### Audit protocol notes
- Classify every miss; only *vocabulary-gap* and *binder-miss* are detector
  work — drift/mis-anchor are lifecycle (re-ingest cadence + W5).
- The detector's blindness is uniform across missions, so fixes here pay out
  on all ~200 scope-trees; re-detect+re-ingest the ensemble after W3/W4.
- Next sessions (from the map shortlist): M-interim-director-proxy-metric-inventory
  (46 scopes, biggest mess district), M-pudding-peradams, M-pattern-application-diagnostic.

### Session 2 note (2026-06-10, E-mission-head greenfield)
- **12th binder minted: `plain-argument`** — the plain-language ARGUE statement
  is a *defined sub-scope* (Joe). Wired end-to-end: detector regex
  (`mission_scope_detect.py`), ingest enrichment via `stable-heading-scopes`
  (both dispatch sites), view fetch list, verified rendering in the panel.
- Reload discipline reminder: the Drawbridge-loaded ingest/view namespaces have
  `when-not find-ns` guards in the fast scripts — after editing either, force
  `load-file` via proof-eval or the JVM keeps serving the old code.

### Session 3 addendum — W15 (found live on M-smart-emacs-cursor, 2026-06-11)

- **W15 — produced-witness vs consumed-context.** The binding-flow audit
  flags any body-phase item with no binder-phase introduction as a free
  variable — but artifacts BORN in a body phase (the Reazon gate file and
  its certificate, cited by the VERIFY that produced them) are not unbound
  context; they are the existential's witnesses, the transducer's OUTPUTS.
  The free class must split: consumed-without-introduction (the vice) vs
  produced-in-phase (the construction). Discriminator candidates: the item
  first appears in the same section that claims to create it; or the
  commit→mission output tape (E-mealy-style-transducer) shows the mission
  itself wrote the file. Found when the instrument audited the mission that
  was being built around it, minutes after a spoken detector fix (the
  source-material extension, 69d4ab4) made the artifacts visible at all.
- **13th binder minted: `verify-gate`** (Joe, spoken, 2026-06-11) — the
  section binding the actual verification (executable gate + certificate)
  is a defined VERIFY sub-scope, not a loose section. Wired end-to-end live
  (detector regex, ingest both dispatch sites, view fetch list, Drawbridge
  reload, rendered on M-smart-emacs-cursor §5.1 within minutes of the
  spoken assertion). Same minting path as plain-argument, one phase over.
