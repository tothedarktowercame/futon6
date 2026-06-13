# M-distributed-proofreaders — structure-first recognition + QA over the math corpus

**Date:** 2026-06-13 · owner: claude-1 seat (this session) · paired (Joe + Opus)
**Status:** DERIVE (working PoC in hand; derive gate minted below)
**Capability:** `:distributed-proofreaders` (region t3, Futon City) — the
solo-FUTON, CPU-local planet in the `full-arxiv-mining` ↔
`math-ct-prior-substrate` system.

## HEAD

The "killer idea" (golden-walk-ledger.md §THE KILLER IDEA), made operational:
mine the math corpus **structure-first** — concepts are the index, papers are
occurrence sites — and run a **Distributed-Proofreaders** loop over it:
mine → measure loss → fix the worst class → re-mine, anytime, never perfect.
Named for the volunteer book-digitisation model: systematic per-type passes,
each pass clearing one defect class, quality a property of the revision
process rather than any single snapshot (the Moran-1971 *because*-clause
lesson, futon5a/essays/moran-1971-agent-cascades).

## 1. IDENTIFY — the tension

`full-arxiv-mining` is parked (`:held`, frontier) because it was scoped as a
multi-party capability — Joe + Rob + MFUTON + a superpod + sysadmins. But the
PoC today showed the load-bearing work is **deterministic CPU**: macro
sweeps, role resolution, the recognizer registry, the loss census. None of it
is GPU. So the parked frontier has a **solo-doable core that runs on a
laptop** — and that core is most of the value. The tension: the capability we
want is filed as blocked-on-coordination when its substance isn't.

## 2. MAP — what we already have (the assets Joe named)

- **The substrate:** `math-ct-prior-substrate` (satisfied) — 9,795 math.CT
  eprints, the term prior, the superpod stage-1..11 output.
- **anatomy-v0 sweep** (`scripts/anatomy_v0_sweep.py`): per-paper extraction —
  symbol tables (1.44M macro-defs), authored layer (\label/\ref/\cite-with-
  locus: 210K locus-cites), token census. The deterministic substrate.
- **LaTeXML role lexicon** (`golden-graphs/latexml-math-roles.tsv`, 967 cseqs)
  + the **TeX-primitive inventory** + the **standard-vocab table** (added
  today, C2). Plus latexml itself for per-fragment shell-out.
- **NNexus** (`/home/joe/code/nnexus/archive/snapshot-1-2014.sqlite`) +
  `background-corpus-index.json` (nLab + CT-term-prior) — the concept authority
  for role-gap resolution.
- **The golden graphs + SCHEMA.md** — the target shape for recognized output.

## 3. DERIVE — the capability, and the gate

**The recognizer registry (parse once, recognize forever):** a concept/notation
defined or used across N papers is a corpus-wide recognizer. Demonstrated live
today: 191,927 distinct author-defined macros over 9,795 papers; **4,487
defined in ≥50 papers** (the shared-notation core — `\id` 5169, `\Hom` 4571,
`\C` 3831, ...). Mint a recognizer once → it classifies occurrences corpus-wide.

**The loss surface drives the passes (Distributed-Proofreaders proper):** each
re-mine emits a loss census; the dominant class is the next pass. Today's
first turn:
- measured: "707K unknown control sequences" — looked catastrophic;
- diagnosed: ~88% **false**-unknown (registry seed + classifier audit);
- fixed (commit on `scripts/anatomy_v0_sweep.py`): (A) alphabet-wrapper atoms,
  (B) class-unknown ≠ role-unknown, (C2) standard-vocab seeding;
- re-measured on 0809.2517: distinct unknowns **130 → 15**, symbol∩unknown
  **73 → 0**, role-gaps (10,911) now tracked apart. Corpus re-sweep running.

### Derive gate (per `derive-exits-on-a-minted-sorry`)

Derive gate verdict: PASS — typed hole minted:
- **have:** per-paper extraction with a large false-unknown loss and no
  cross-paper recognizer registry (extraction repeated per paper, never reused).
- **want:** a trustworthy corpus-wide recognizer registry + a hunger-field-
  driven Distributed-Proofreaders loop that lowers genuine loss monotonically.
- **kind:** `:construction` / `:open` — construction named (the mine→fix→
  re-mine loop; today's A/B/C2 is its first discharged step). Closes-by-process,
  not by one artifact.

## Backlog (the next passes, loss-ranked)

1. **Structural mop-up:** `\begin \label \ref \\` read unknown in the census
   though the authored-layer harvest already knows them — share the vocabulary.
2. **Standard-vocab top-up:** `\vert \Box \#` and the residue tail.
3. **Role-gap typing (the real frontier):** the 10,911 recognized-but-untyped
   tokens — resolve operator-name macros (`\Hom \End \colim`) against NNexus /
   latexml-fragment shell-out rather than flattening to atoms. This is where
   the concept authority earns its place.
4. **Stand up the registry** from the corrected macro tables → "concepts are
   the index" becomes a few lines on top.

## Relationship to the star-map

- **requires** `:math-ct-prior-substrate` (the corpus it mines).
- **precursor-of** `:full-arxiv-mining` — delivers the structure-first
  recognition core locally; de-risks and partially satisfies the parked
  frontier without the multi-party dependency. (When the superpod/MFUTON
  coordination lands, this becomes the engine it scales.)
- siblings: `E-mission-mining` (same loop, mission corpus not math corpus);
  the golden-walk ledger (the rules); the anatomy paper §8.

## Scope

In: the deterministic CPU extraction/recognition/QA loop over the local math
corpus; the recognizer registry; loss-driven passes. Out: the multi-party
superpod-scale arXiv-wide run (that stays `full-arxiv-mining`); GPU embedding
work (single-GPU Linode, separate lane).

## Checkpoint 1 — 2026-06-13 (turn-1 of the DP loop; CT demo end-to-end)

**What was done (capability chain demonstrated on math.CT):**
- Re-sweep (corrected extraction, A/B/C2) over 9,795 papers, 9.7 min:
  **unknown-cseqs 707,183 → 204,126 (71% reduction)**; spans-fully-classified
  5.98M → 12.8M. ~503K false-unknowns cleared.
- Recognizer registry built (`scripts/build_recognizer_registry.py` →
  `data/ct-recognizer-registry.json`, regenerable/gitignored): **21,811
  recognizers** (≥10 papers) of 191,927 macros; **80% role-resolved**.
- NNexus + LaTeXML brought online as reuse capabilities (concept-authority
  130,960 terms via `concept_authority.py`; latexmlmath fragment parse).
  Star-map: `:concept-authority`, `:latexml-fragment-parse` (t3, satisfied).
- Loop closed: **22/25 top role-gaps resolve** against the concept authority
  (\id→identity morphism, \Hom→hom, \op→opposite category, \End→end, \Set→set).

**Next (loss-ranked):**
1. **Role-gap concept precision:** single-letter surfaces mis-resolve
   (\C→"c" not ℂ). Prefer the RHS-derived concept (\C := \mathbb{C} →
   complex numbers) over the bare surface; guard single-letter lookups.
2. Structural mop-up (\begin/\label/\ref shared with authored-layer vocab).
3. Stand up the registry as a persistent recognizer index consumed at
   classification time (true "parse once, recognize forever").

**Per-MSC replication:** the whole chain is `--eprints <dir>`-parameterised;
the superpod blast re-points DEFAULT_EPRINTS per MSC class. Demo complete.
