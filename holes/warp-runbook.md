# WARP runbook — the corpus second layer (bibliography · citations · concordance)

**Status: LIVE 2026-06-13.** Orchestrated by claude-1, for the Codex pool. The
loom: the Claude fleet weaves the **weft** (per-paper, invariants ⊥ scopes —
`dp-fleet-runbook.md`); Codex lays the **warp** — the long threads running
through the *whole corpus*. Together they make the cloth. If you are a Codex
agent belled here, this is your standing task.

## What the warp is

A "second layer of meaning" ON TOP of the papers (Joe): concepts and papers as
*cross-paper* objects, not just intra-paper markup. Three composable
deliverables; the per-paper weave keys on the SAME concept vocabulary, so warp
and weft meet — "comodule algebra is a formalisation candidate (weft)" + "and
it's used in these 40 papers, reached via these citations (warp)".

## Corpus (the raw material — already confirmed present)

- **9742 eprints**: `/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints/<id>.tar.gz`
  (arXiv math.CT). Each tarball holds `.tex`/`.bbl` with `\title`, `\author`,
  `\bibitem{key}`, `\cite{key}`. Refs are often CLASSICAL (author + title +
  journal), NOT arXiv-ids — so citation linkage is partly fuzzy author/title
  matching, not pure id-join. (0809.2517: 46 bibitems, 29 cite-keys.)
- Per-paper DP markup (where the weft has run): `data/showcases/ct-anatomy/golden/fable-<id>-dp-emacs.json`
  (`marks[]` with kind definiendum / let-binder / classified / concept-typed).
- The sweep tokenizer: `scripts/anatomy_v0_sweep.py` (`read_eprint_files`,
  `classify_cseq`) — reuse it; do not re-tokenize from scratch.

## Deliverables — CLAIM one in `data/warp/claims.jsonl` before starting

(Same claim protocol as the fleet: append `{"agent","claim","at","state":"open"}`,
read first, pick another if taken. data/ is gitignored.)

**W1 — bibliography extraction** (the prerequisite layer). `scripts/warp_bib.py`:
over all 9742 eprints, emit per paper `{paper_id, title, authors,
bibitems:[{key, raw, author?, title?, year?, arxiv_id?}], cites:[key…]}` →
`data/warp/bib/<id>.json` + a merged `data/warp/bib-index.json`. Skip
unreadable tarballs, LOG the count covered/skipped (no silent caps). Gate:
runs over the full corpus; spot-check 0809.2517 = 46 bibitems.

**W2 — citation graph** (depends on W1; build the corpus identity index +
linkage algorithm first, wire to W1's output as it lands). `scripts/warp_citations.py`:
link papers by (a) arXiv-id when a bibitem carries one, (b) fuzzy normalized
author+title against the corpus paper-identity set (each paper's own
`\title`/`\author`). Emit `data/warp/citations.json` `{edges:[{from,to,via}],
stats}` + reverse cited-by. Gate: spot-checked edges resolve to real corpus
papers; report the linkage rate honestly (fuzzy matching will be partial).

**W3 — concordance** (independent of W1/W2 — needs only processed papers).
`scripts/warp_concordance.py`: the cross-paper term index. For each processed
paper, record `term → [{paper, count, role}]` where role = **defined** (term
appears as a definiendum / let-binder subject) vs **used** (otherwise). Source
terms from the DP markup where present, else the sweep's classified terms.
Emit `data/warp/concordance.json`. Start with a tractable batch (the papers
with DP markup + a first ~500), build to scale, LOG coverage. Gate:
spot-check a known concept ("Hopf algebra", "comodule") indexes to the right
papers with correct defined/used roles.

## The loop (bounded, never silent)

CLAIM → BUILD (idiomatic Python; reuse the sweep, don't reinvent) → VERIFY
(`python3 -m py_compile`; run on a sample + report counts; spot-check the gate)
→ RECORD (claim `state:"done"` + stats; commit the SCRIPT only — data/ is
gitignored; Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>) →
CHECKPOINT bell claude-1 with {deliverable, stats, sha}. Stop and bell on any
judgment call or the `WARP-OFF:` sentinel.

## Hard constraints

Never restart the futon3c JVM. Never commit `data/`. Self-contained JSON/EDN
stores in `data/warp/` for v1 (ingestible into futon3a/futon1a LATER — do not
couple now). Report linkage/coverage rates honestly; "fuzzy match" is not "all
linked". Bell claude-1 back — that is what lets the warp meet the weft.
