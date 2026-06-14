# anatomy-v0 loss-surface → detector backlog (task #14, 2026-06-13)

Triage of the two overnight loss surfaces, ranked DP-style (clear the most
loss per unit work first). **Headline: the "707K unknown control sequences"
is inflated ~3-5× by THREE false-unknown classes; the genuine unknown
residue is far smaller and the top fixes are cheap.** Evidence is from a
2000-paper sample of `storage/futon6/data/ct-anatomy-v0/*.json`.

## CT lane — unknown-cseq loss is mostly false-unknown

Sampled 2000 papers → 141,487 unknown occurrences, 25,920 distinct cseqs.
Three reclassification fixes, in priority order:

**C1 — join the per-paper macro table to the token classifier (THE BUG).**
In 0809.2517, **73 cseqs are in BOTH the paper's `symbol-table` AND its
`unknown-list`** — including `\C`, recorded by the sweep as
`\newcommand{\C}{{\mathcal C}}`. The classifier built the macro table but
doesn't consult it when classifying tokens (`\ot` resolves, but `\C \F \G
\Hom \Set …` don't). The whole category-theory author-macro tail
(`\id \Hom \op \C \Set \Cat \N \Z \End \Mod \cat …`, hundreds of papers
each) is false-unknown for this reason. Highest value, single-site fix
(classifier consults the symbol-table before declaring unknown).

**C2 — extend the standard-vocab role table (~250 entries).** A small static
table clears ~30% outright; the top misses are embarrassingly basic:
- Greek letters: **18.2%** of all unknowns alone (`\alpha \pi \beta \eta
  \sigma \lambda \Delta …`) — role ID/atom.
- math alphabets `\mathcal \mathbb \mathrm \mathbf` (4.4%); delimiters
  `\{ \} \langle \rangle \left \right` (2.8%); spacing `\, \; \! \quad`
  (2.6%, layout-class per R15); structural `\begin \text \\ \ldots` (2.3%).
- Tail also hides more standard ops the lexicon misses: `\frac \le \ge
  \xrightarrow \overset \stackrel \ell \colim \big \mbox \rm \bf`.
Source the roles from the LaTeXML role lexicon (already mined to
`golden-graphs/latexml-math-roles.tsv`) — the gap is coverage, not method.

**C3 — package profiles for diagram families.** `\xymatrix \ar \ar@` (xy-pic
commutative diagrams) recur across many CT papers — a whole package family,
recognizable, addable as one profile (the way GrCalc was handled per-paper).
After C1+C2, profile the residual tail to find the next package cluster.

Expected effect: C1+C2 together likely reclassify well over half the 707K;
re-run the sweep (local, ~16min, free) to measure the new genuine-unknown
floor. That floor is the real detector frontier.

## Mission lane — from `mission-triples/_summary.json` + the hitlist audit (F1)

(Owned by `E-mission-mining.md`; listed here for the unified ranking.)

**M1 — status-aware advance-typing + CLOSED-detection + reopen-marker.**
Root cause of hitlist Finding F1 (0/4 checkpoint-only clean). Read the
canonical `Status:` line; detect `CLOSED`; honor the latest reopen marker.
Fixes the dry-basin `:checkpoint-only` reliability AND keeps already-closed
missions (e.g. M-explore-aiqa) out of the dry-basin set entirely.

**M2 — reconstruction pass for the 58 `:missing-derive`.** These predate the
derive gate; push `:unminable → :reconstructed-thin` via the HEAD/IDENTIFY/
MAP fallback, honestly tiered (never fabricate).

**M3 — triage `:zero-pattern-cites` (38) and `:unverifiable-artifacts` (53).**
Detector gap vs genuinely absent — sample to decide per class.

## Recommended dispatch order

1. **C1** (classifier-consults-symbol-table) — small, single-site, biggest
   false-unknown class. Codex-shaped, gated on re-running 0809.2517 and
   confirming the 73-overlap drops to ~0.
2. **C2** (standard-vocab table) — static data + lexicon join; cheap, ~30%+.
3. **M1** (status-aware typing) — independent, unblocks reliable closeable-
   mission surfacing.
4. Re-run both sweeps to measure the new floors; then **C3 / M2 / M3** against
   the residue.

## C4 — prose-concept definitions are dark (proofread-sourced, 2026-06-13)

Operator proofread tag (Joe, "incomplete", 0809.2517 Galois2.tex:152):
"an $H$-Galois object is defined to be an $H$-comodule algebra $A$ such
that …" → detect_scopes returns ZERO scopes. Two gaps:

1. **Cheap:** the `is-called` regex alternation is `(called|defined as)`
   — add `defined to be` / `to be`. Catches the bare-$symbol$ variants.
2. **Real:** the definiendum here is a PROSE CONCEPT PHRASE ("$H$-Galois
   object"), not a bare $-symbol. Needs a prose-definiendum regex:
   `(an?|the)\s+(<concept-phrase incl. $math$>)\s+is\s+(defined to be|
   called|said to be)\s+(<def>)` → emit bind/define with the concept as
   definiendum (the golden-walk emphasis-definiendum lesson, R-rules).
   This is the same class the golden builder caught via {\em …}; the
   superpod detector doesn't have it.
3. The sentence's `\cite[Definition 3.1]{Sch1}` is authored-layer
   (label/ref/cite harvest) — already the dashboard's `·` not-yet row.

Note: nlab-wiring.py is the SHARED superpod detector (nLab/papers/
missions). (1) is strictly additive/safe; (2) needs a corpus re-check
(could over-fire on "is defined" prose). Dispatch (2) with the
golden-walk specimens as the test set.

## C5 — the dense-definition blindness (codiagonal passage, proofread 2026-06-13)

Specimen (Joe): "If $M$ is a right $A$-module, then it is not difficult to
check that [big eqnarray of GrCalc \gbeg/\got/\gcl... ] := [...] equips
$M\ot H$ with a structure of right $A$-module, usually referred to as the
{\em codiagonal} one." — "a mess in the current markup regime." Multiple
distinct scope problems in ONE passage:

1. **GrCalc layout noise (R15 not applied in dp_paper_view).** \gbeg \got
   \gcl \grm \gmu \gbr \gcn \gob \gnl \gvac \scalebox are diagram/layout
   macros; they resolve as author-defined but role-UNKNOWN → role-gap noise
   filling the display. Fix: classify GrCalc (and \newcommand layout macros)
   as a `layout` class, excluded from role-gap/symbol accounting (the R15
   semantic-class column, ported into the sweep).
2. **`:=` in a display defines a structure (R6).** The eqnarray defines the
   codiagonal right-$A$-module structure on $M\ot H$; not recognised as a
   definition (the := display-binder). Should emit a bind/define scope whose
   definiendum is the constructed structure.
3. **`{\em codiagonal}` emphasis definiendum (C4).** The named structure is
   introduced by emphasis ("referred to as the {\em codiagonal} one") — the
   prose-definiendum gap, still open.
4. **Conditional binder ("If $M$ is a right $A$-module").** Now grounds $M$
   via the assume scope (C5 fix landed: manifest-grounding) — but the type
   extraction from the assume phrase is crude ("is a X" regex).

Grounding status after manifest-grounding: 72% (was 66%); 3272 symbols
still ungrounded — genuine debt (indices, ad-hoc vars, reference-bound, and
symbols inside GrCalc displays that are layout-noise not real symbols — (1)
would remove many). Order: (1) GrCalc/layout quarantine clears the display
noise AND drops the false ungrounded-symbol count; then (2)/(3) the
:=-definition + emphasis-definiendum; (4) assume-type quality.
