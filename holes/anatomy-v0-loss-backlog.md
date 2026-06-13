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
