# Codex handoff: seed-dictionary-from-pm.py — bulk PM → OED-shape dictionary entries

**Author:** claude-13
**Date:** 2026-05-19
**Target agent:** codex (any of codex-1, codex-2, codex-10)
**Form:** R11 scope-bounded-handoff per `~/code/futon3c/CLAUDE.md` §"Codex Handoff Protocol"
**Trigger:** Joe (2026-05-19): *"Let's do the Stage 1, please set this up as a handoff file that I can pass to Codex."*

## Title

`seed-dictionary-from-pm.py — bulk-convert PlanetMath corpus to OED-shape dictionary entries (Stage 1)`

## Goal

Produce a canonical PlanetMath-seeded math dictionary at OED-shape entry schema. ~19K PM articles → ~19K dictionary entries with definitions, usage-example contexts, and metadata. Plus a hand-seeded stopwords file (~50 entries) to bootstrap the noise-filter. **Stage 1 of the `E-discover-terms-as-dictionary-construction.md` excursion.** Fully local; no GPU; no superpod; no arxiv access required.

This step is **the prerequisite for everything downstream** (extractor pilot, feedback loop, kernel evolution) — it produces the canonical reference corpus that arxiv-discovered candidates will be compared against.

## Read first (canonical references)

In order:

1. `~/code/futon6/holes/excursions/E-discover-terms-as-dictionary-construction.md` (parent excursion; the design spec)
2. `~/code/futon6/holes/excursions/E-discover-terms-as-dictionary-construction.sample-entries.edn` (two worked entries demonstrating the schema; `ringed-space` is the canonical-from-PM exemplar)
3. `~/code/futon6/scripts/build-ner-kernel.bb` (existing PM enumeration — same corpus walk you'll mirror; do not modify)
4. `~/code/futon7/holes/M-interim-director-proxy-metric-inventory.md` §2.A.2.20, §2.A.2.43, §2.A.2.44 (context on the noise problem this Stage solves)

## :in (READ-ONLY)

| Path | What it provides |
|---|---|
| `~/code/futon6/holes/excursions/E-discover-terms-as-dictionary-construction.md` | OED-shape schema (excursion §2); stopword schema (§3); PM seed strategy (§4) |
| `~/code/futon6/holes/excursions/E-discover-terms-as-dictionary-construction.sample-entries.edn` | Worked entries — schema reference data |
| `~/code/futon6/scripts/build-ner-kernel.bb` | Existing PM-enumeration shape; produces the kernel TSV from PM |
| `~/code/planetmath/` | PM corpus on disk. Layout: `<NN_Subject-name>/<MSC-code>-<ArticleTitle>.tex` files (e.g. `43_Abstract_harmonic_analysis/43A07-OrnsteinWeissLemma.tex`). Per-directory `.edn` companion files appear to be index/metadata — inspect to confirm |
| `~/code/storage/futon6/data/ner-kernel/terms.tsv` | 19,236 kernel terms. Rows where `source = "pm-title"` give the canonical-ID lookup `(term_lower, canon_id)` pairs — use this to populate `:term/id` consistently with what the kernel already calls things |

## :out (NEW FILES)

| Path | Content |
|---|---|
| `~/code/futon6/scripts/seed-dictionary-from-pm.py` | Loader script. Python 3.10+. Idempotent. Self-contained (no new external deps beyond what's already in `pyproject.toml`) |
| `~/code/futon6/holes/excursions/dictionary-schema.edn` | Schema reference (separate from sample-entries; placed under `holes/` to avoid `data/*` gitignore) |
| `~/code/futon6/data/dictionary/entries-pm-seed.edn` | Output entries (~19K). Gitignored by default per `data/*` rule. Large file; OK |
| `~/code/futon6/data/dictionary/stopwords.edn` | ~50 hand-seeded stopwords (see §"Stopword seed list" below). Gitignored under `data/*`; OK |
| `~/code/futon6/data/dictionary/audit-sample.json` | Random sample of 100 entries for operator audit. Deterministic seed=13 |
| `~/code/futon6/data/dictionary/run-stats.json` | Per-run statistics (count succeeded / count skipped + reasons / elapsed seconds) |
| `~/code/futon6/tests/test_seed_dictionary_from_pm.py` | Pytest unit tests with fixtures at `tests/fixtures/planetmath-mini/` |

## Function signatures (suggested; refine if better)

```python
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, List, Dict


class PMArticle(NamedTuple):
    """One PlanetMath article ready for dictionary conversion."""
    canon_id: str         # e.g. "ringed-space" — lowercased canonical handle from filename + kernel lookup
    headword: str         # e.g. "Ringed space" — display form from filename or \title
    msc_code: str         # e.g. "14A15" — math-subject-classification from filename prefix
    subject_area: str     # e.g. "14_Algebraic_geometry" — parent dir name
    tex_path: Path        # absolute path
    body_text: str        # extracted body (no \begin{document} wrapper, no preamble)
    raw_tex: str          # full file content as fallback


def find_pm_articles(planetmath_root: Path) -> Iterator[PMArticle]:
    """Walk the PM corpus; yield one PMArticle per article.

    Resilient: skip with a warning (recorded to run-stats.json :skipped) if a .tex
    file is malformed (e.g. unreadable encoding, missing body, only-references).
    Do NOT crash on any single article."""


def extract_definition_from_pm(article: PMArticle) -> Optional[str]:
    """Extract the first definitional sentence or paragraph from a PM article body.

    Tiered:
    - If body has a \\begin{definition} block, return its content (stripped of LaTeX
      structural markers, but preserve inline math like $X$ and operators).
    - Else, return the first paragraph (first sentence + up to one more sentence).
    - Else, None.

    Treat numeric-ID-titled articles (e.g. files where the headword is "0" or
    "123") as no-definition cases — return None and record in skipped reasons."""


def pm_article_to_entry(article: PMArticle,
                        kernel_row: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """Convert one PM article to a dictionary entry per the OED-shape schema.

    Schema reference: holes/excursions/dictionary-schema.edn

    Use kernel_row (a dict with keys term_lower / term / source / canon_or_count
    from the kernel TSV) to populate :term/id consistently with what the kernel
    calls things. If kernel_row is None, derive :term/id from the filename via
    PM-canonicalisation rules (lowercase, hyphenate camelcase, strip MSC prefix).

    Returns None if the article should be skipped entirely (numeric-ID,
    body-too-short, etc.). Caller records skip reason."""


def hand_seeded_stopwords() -> List[Dict]:
    """Return the hand-seeded stopword list — see §'Stopword seed list' in the
    handoff. ~50 entries covering: common-emphasis (unique, asymmetric, etc.),
    bibliography-journal-name patterns, reference-marker patterns.

    Each entry conforms to the stopword schema in excursion §3."""


def audit_sample(entries: List[Dict], n: int = 100, seed: int = 13) -> List[Dict]:
    """Deterministic random sample for operator review."""


def main():
    """End-to-end:
      1. Walk PM (find_pm_articles)
      2. Load kernel TSV (build a {term_lower: row} dict for kernel_row lookup)
      3. For each PMArticle, call pm_article_to_entry; collect succeeded + skipped
      4. Write entries-pm-seed.edn (valid EDN — pprint with reasonable indent)
      5. Write hand_seeded_stopwords() → stopwords.edn
      6. Write audit_sample(entries) → audit-sample.json
      7. Write run-stats.json with counts + skipped-reason breakdown + elapsed
    """
```

## Shapes reference

### Dictionary entry shape (from `holes/excursions/E-discover-terms-as-dictionary-construction.md` §2)

Mandatory fields for the PM seed (since each comes from a known canonical source):

```clojure
{:term/id              "ringed-space"            ; lowercased canonical handle
 :term/headword        "Ringed space"
 :term/lower           "ringed space"
 :term/part            :noun                     ; default for PM seed; refine only if obvious
 :term/aliases         []                        ; PM seed typically empty
 :term/etymology
 {:first-source        "planetmath:RingedSpace"  ; "planetmath:" + canon-id-CamelCase
  :first-source-date   nil                       ; PM doesn't carry dates
  :first-extractor     :pm-seed-loader/v1
  :note                "PlanetMath canonical entry; ..."}
 :term/definitions
 [{:def/id              "ringed-space-d1"
   :def/text            "<definition extracted>"
   :def/extracted-from  "planetmath:RingedSpace"
   :def/source-context  "PM article body — full text"
   :def/extraction-method :pm-seed
   :def/extracted-at    #inst "2026-05-19T...Z"  ; from `python3 -c "from datetime import...`
   :def/confidence      1.0                      ; canonical for PM seed
   :def/status          :canonical}]
 :term/usage-examples
 [{:example/paper       "planetmath:RingedSpace"
   :example/role        :canonical-source
   :example/context     "(see PM article body)"
   :example/seen-at     nil}]
 :term/status           :canonical
 :term/canon-source     :planetmath-seed
 :term/first-seen       nil
 :term/last-seen        nil
 :term/occurrence-count 1
 :term/cross-refs       []                       ; populate if PM article has obvious \\ref{...}
 :term/review-notes     ["Seeded from PM 2026-05-19."]
 :term/graduated-at     #inst "2026-05-19T...Z"}
```

For articles where `extract_definition_from_pm` returns None:

```clojure
{:term/id              "<handle>"
 :term/headword        "<headword>"
 :term/lower           "<lower>"
 :term/part            :noun
 :term/definitions     []                        ; explicit empty
 :term/usage-examples  []
 :term/status          :canonical-no-definition  ; new status enum value
 :term/canon-source    :planetmath-seed
 :term/review-notes    ["PM article had no extractable definition. Body length: N chars. Reason: <reason>"]
 ...}
```

This explicit-no-definition marker is important so audit can flag PM coverage gaps without confusing them with "we don't know about this term yet."

### Stopword shape (from excursion §3)

```clojure
{:stopword/id          "unique"
 :stopword/lower       "unique"
 :stopword/reason      :generic-emphasis        ; or :bibliography-journal-name, :reference-marker, :proper-noun-not-concept
 :stopword/first-flagged-at #inst "2026-05-19T...Z"
 :stopword/example-context "(none — hand-seeded)"
 :stopword/source-paper "(none — hand-seeded)"
 :stopword/flag-method :hand-seed}
```

## Stopword seed list (for `hand_seeded_stopwords()`)

These are the ~50 stopwords to land in stopwords.edn. Grouped by reason:

**`:generic-emphasis` (~25):**
unique, asymmetric, complete, important, simple, complex, special, basic, key, main, new, novel, fundamental, classical, modern, recent, obvious, trivial, natural, standard, common, general, particular, specific, certain

**`:bibliography-journal-name` (~10):**
acta universitatis apulensis, adv in math, advances in mathematics, j math anal, j math phys, comm math phys, ann math, proc amer math soc, bull amer math soc, math z

**`:reference-marker` (~5):**
ibid, op cit, loc cit, et al, cf

**`:proper-noun-not-concept` (~5):**
let, suppose, define, assume, consider

**`:section-marker-fragment` (~5):**
section, chapter, appendix, theorem, lemma

Codex may extend the list (with rationale per entry) but should NOT shrink it without operator review. The 50 are seed-grade — wrong ones can be reverted in subsequent audit.

## Test expectations

`pytest tests/test_seed_dictionary_from_pm.py` passes. Fixtures at `tests/fixtures/planetmath-mini/` with 5 hand-crafted PM article files:

| Fixture | Tests |
|---|---|
| `14_Algebraic_geometry/14A15-RingedSpace.tex` (well-formed, has `\begin{definition}`) | `extract_definition_from_pm` returns the definition block content |
| `00_General/00X00-MoscowMathematicalPapyrus.tex` (historical/cultural, no definition block, no first-paragraph definition) | returns None; entry produced with `:term/status :canonical-no-definition` |
| `54_General_topology/54E40-MalformedTexExample.tex` (broken — unmatched braces) | `find_pm_articles` yields nothing for it, records skip in run-stats with reason `:malformed-tex` |
| `06_Order_lattices/06A06-SimpleDefinitionalParagraph.tex` (no `\begin{definition}`, but first paragraph is definitional) | `extract_definition_from_pm` returns the first paragraph |
| `00_General/00-NumericIdOnly.tex` (filename like `00-0.tex`) | `pm_article_to_entry` returns None; recorded as `:numeric-id-skip` |

Tests cover: definition-extraction-happy-path, malformed-tex-graceful-skip, no-definition-canonical-no-definition-marker, numeric-id-skip, idempotency (running twice produces byte-identical entries-pm-seed.edn except timestamps).

## Criteria checklist

- [ ] `python3 scripts/seed-dictionary-from-pm.py` runs without errors against the real `~/code/planetmath/` corpus
- [ ] Output `entries-pm-seed.edn` contains **at least 18,000 entries** (allowing for ~5% skip rate on malformed/numeric-ID/non-conformant articles)
- [ ] Entries validate against the OED-shape schema (every entry has `:term/id` + `:term/headword` + status flag set correctly)
- [ ] `audit-sample.json` has **exactly 100 entries** with deterministic `seed=13`
- [ ] `stopwords.edn` has **≥ 50 entries** covering at least: `:generic-emphasis`, `:bibliography-journal-name`, `:reference-marker`, `:proper-noun-not-concept`, `:section-marker-fragment` reason buckets
- [ ] `run-stats.json` reports: succeeded count + skipped-reason breakdown + elapsed seconds
- [ ] `pytest tests/test_seed_dictionary_from_pm.py` passes (no skips, no xfails)
- [ ] **No modifications** to existing kernel TSV (`~/code/storage/futon6/data/ner-kernel/terms.tsv`) or `build-ner-kernel.bb` (read-only)
- [ ] **Idempotent**: running twice produces byte-identical `entries-pm-seed.edn` except `:def/extracted-at` and `:term/graduated-at` instants (which should use a stable seed-time captured at script start, not per-entry)
- [ ] EDN output uses **clojure/edn-readable form** (round-tripable via `clojure.edn/read-string`); validate post-write with a small Clojure or `edn-format` Python lib check
- [ ] Script writes a one-line `print()` per ~500 articles processed so operator can monitor progress; final summary line at end

## Out of scope (named for foreclosure)

- T3 LLM-assisted extraction (Stage 2 work; defer)
- Graduation tool / kernel TSV regeneration from dictionary (Stage 3 work; defer)
- Actually running discover_terms or processing arxiv batches (gated on Rob)
- Cross-reference population beyond what's trivially extractable from `\\ref{...}` macros (Stage 2 work; the PM seed populates `:term/cross-refs []` empty initially)
- Multi-definition entries (if PM article has multiple definition blocks, take only the first; Stage 2 can extend)
- Pretty / human-readable EDN formatting beyond what `clojure.edn`-compatible writers do by default
- Normalising LaTeX math markup in `:def/text` (preserve `$X$` and `\\mathbb{N}` etc. as-is; downstream rendering will handle)
- Modifying the existing kernel TSV (read-only; the dictionary projects to the kernel at Stage 3, not Stage 1)

## Estimated effort

~1 codex-shift (6-8 hours). The work is mechanical Python over a known corpus; the only judgment calls are in `extract_definition_from_pm` heuristics. If the heuristic for "first definitional paragraph" turns out to need tuning after audit, that's a follow-on tweak, not a blocker.

## Coordination protocol

- Bell back to claude-13 (`agent-id: claude-13`) when:
  1. Script ships with passing tests — *"Stage 1 done: N entries written, M skipped (breakdown), audit sample ready"*
  2. Tests fail or schema-validation finds an issue that needs design clarification

- Joe-side audit (manual, after codex ships): sample 100 entries from `audit-sample.json`; flag any that look wrong (extraction caught wrong text, headword mis-cased, etc.). Findings inform Stage 2 (extractor T1+T2 design).

## After this lands

Stage 2 of the excursion becomes actionable: the extractor T1+T2 implementation can run over the §2.A.2.43 demo's 64 candidates as its pilot, comparing against the now-populated PM seed dictionary to filter out terms that are already canonical.

The §2.A.2.20 "stack-side feedback loop missing" gap is one step closer to closing — the canonical reference corpus now exists in dictionary form.
