# Excursion: discover-terms as OED-shape dictionary construction

**Date:** 2026-05-19
**Owner:** Joe + claude-13 (this draft)
**Parent mission:** `M-hyperreal-dictionary-planning.md` (futon6; status IDENTIFY since 2026-04-29; this excursion scopes the §6.B "paper → concept bridge" buildable-now probe under Joe's 2026-05-19 OED-framing)
**Trigger:** Joe (2026-05-19): *"The ideal setup might be not only doing term-spotting, but actually automatically writing entries into a dictionary. This would help with the noisy candidates ... 'acta universitatis apulensis' — no definition would be found; 'unique' either would be defined already (it is a common term, maybe a definition in terms of set theory is plausible), or added to a stopword list (more likely) ... The logic of the dictionary construction would be similar to that used by the OED — we'd have examples of usage (based on where the terms are found) as well as definitions (when those can be ascertained). We could seed it with PlanetMath (which we have locally and are already using as a source of 'known' terms)."*
**Cross-ref:** `~/code/futon7/holes/M-interim-director-proxy-metric-inventory.md` §2.A.2.20 (discover_terms findings), §2.A.2.43 (sanity-check demo + noise problem)

## 1. The shape Joe named

Today's pipeline (per §2.A.2.20 + §2.A.2.43):

```
arxiv .tex source ─► extract_open_ner_candidates()
                    ─► 6 source-contexts (latex-emph / called-as /
                       is-called / defined-as / definition-of /
                       definition-block-subject)
                    ─► candidate-new-terms.jsonl
                    ─► (nothing reads this)
```

Joe's proposal:

```
arxiv .tex source ─► candidate extraction (as today)
                    ─► definition-extractor (NEW)
                       ├── definition found in context → dictionary entry
                       │   (term + def + usage example + provenance)
                       ├── definition not found, term has known type → 
                       │   provisional entry (e.g. "common, set-theoretic")
                       └── definition not found AND noise-pattern matched →
                           stopword list (e.g. journal-name, generic emphasis)
                    ─► dictionary store (NEW)
                    ─► kernel TSV update on periodic graduation
                    ─► next batch sees enlarged kernel
```

**The noise filter is structural, not heuristic.** A latex-emph hit only becomes a dictionary entry if a definitional sentence can be retrieved for it. `acta universitatis apulensis` → no definition retrievable → drops (or graduates to stopword list). `unique` → no specific definition retrievable in this context → stopword list. `meta-set` → definition retrievable in surrounding text → dictionary entry. **Solves the §2.A.2.43 noise problem by construction.**

**Worked sample available.** Two illustrative entries (one PM-seeded canonical, one arxiv-discovered provisional) + two sample stopword records live at `E-discover-terms-as-dictionary-construction.sample-entries.edn` adjacent to this file. The schema fields below are demonstrated in that file with real-ish data drawn from the §2.A.2.43 demo output.

## 2. OED-shape entry schema (sketched)

The OED's load-bearing fields are: headword, part of speech, etymology, **definition(s)** (often multiple senses, numbered), **quotation evidence** (usage examples, dated, with source citation), cross-references. Adapted for mathematics:

```clojure
{:term/id           "meta-set"                ; lowercased canonical handle
 :term/headword     "meta-set"                ; display form
 :term/lower        "meta-set"                ; lookup form
 :term/part         :noun                     ; math: noun/operator/predicate (lighter than OED)
 :term/aliases      ["super-set"              ; per first-source: "may also be called a 'super-set'"
                     "meta-class"]
 :term/etymology
 {:first-source     "arxiv:0903.1234"         ; the paper that first introduced/used
  :first-source-date #inst "2009-03-15"
  :first-extractor  :superpod-job/v1
  :note             "First emphasis-defined in Baianu, 2009 (category-theory + meta-systems)"}
 :term/definitions
 [{:def/id          "meta-set-d1"
   :def/text        "A class that is not itself a set — e.g. the collection of all sets [59], [176]-[177]; may also be called a super-set."
   :def/extracted-from "arxiv:0903.1234"
   :def/source-context "context window from candidate-new-terms.jsonl"
   :def/extraction-method :paragraph-definitional-regex   ; or :llm-assisted, :pm-seed, :operator-added
   :def/extracted-at #inst "2026-05-19"
   :def/confidence  0.78                       ; if extraction is heuristic-only
   :def/status      :provisional}]             ; :provisional, :reviewed, :canonical
 :term/usage-examples
 [{:example/paper   "arxiv:0903.1234"
   :example/role    :first-introduction
   :example/context "a collection of sets may be a \\emph{class} ... may also be called a 'super-set', or a \\emph{meta-set}"
   :example/seen-at #inst "2009-03-15"}
  {:example/paper   "arxiv:1502.5678"
   :example/role    :passing-mention
   :example/context "Following Baianu's meta-set framework, we consider..."
   :example/seen-at #inst "2015-02-10"}]
 :term/status       :provisional               ; :provisional / :reviewed / :canonical / :stopword / :alias-of
 :term/canon-source :arxiv-discovery           ; :arxiv-discovery / :planetmath-seed / :operator-added
 :term/first-seen   #inst "2009-03-15"
 :term/last-seen    #inst "2026-04-22"
 :term/occurrence-count 17
 :term/cross-refs   [{:rel :related-to    :target "meta-theorem"}
                     {:rel :parent-concept :target "class-set-theory"}]
 :term/review-notes []                          ; operator/agent annotations during graduation review
 :term/graduated-at nil}                        ; #inst when promoted to :canonical
```

Three fields handle the noise/quality story:
- `:def/confidence` — extractor's self-assessment
- `:term/status` `:provisional` → `:reviewed` → `:canonical` (or `:stopword` / `:alias-of`)
- `:term/review-notes` — annotations made during graduation review

## 3. Stopword-shape entries

For terms surfaced by the regex layer that are NOT real dictionary entries:

```clojure
{:stopword/id      "unique"
 :stopword/lower   "unique"
 :stopword/reason  :generic-emphasis            ; or :bibliography-journal-name, :reference-marker, :proper-noun-not-concept
 :stopword/first-flagged-at #inst "2026-05-19"
 :stopword/example-context "a fully functional mind to observe and understand the human mind"
 :stopword/source-paper "arxiv:0903.1234"
 :stopword/flag-method :latex-emph-without-definitional-context}
```

Stopwords are *not* dropped silently — they're recorded as known non-entries so future review can audit the noise-filter's discipline. This matches Joe's "added to a stopword list" framing exactly.

## 4. PlanetMath seed strategy

PM is already on disk + already the kernel source. Each PM .tex file IS a definitional artefact (the article body defines the article subject). Convert each PM article to one dictionary entry:

- `:term/id` from PM article filename
- `:term/headword` from PM `\title{}`
- `:term/definitions[0]` from the PM article body (first paragraph or `\begin{definition}` block)
- `:term/usage-examples[0]` from PM body context
- `:term/canon-source :planetmath-seed`
- `:term/status :canonical` (PM articles are already-vetted)
- `:term/first-seen` from PM article timestamp if available
- `:term/etymology` if PM has historical-note

Seeding effort: ~19,234 PM article-titles already in the kernel TSV; each could become a dictionary entry. Bulk conversion is mechanical Python over PM .tex files. **This bootstraps the dictionary with ~19K canonical entries before any arxiv discovery runs**, giving the discovery-side noise-filter (definition-found check) something to compare against.

## 5. Definition-extractor: how it works

Given a candidate term + its context window (Stage 5 already captures both per the `discovery_example` map), find the definitional sentence(s) if any. Tiered approach:

| Tier | Method | Coverage | Cost |
|---|---|---|---|
| **T1** | Pattern-regex over context window — same 6 patterns the candidate-extractor used, but **inverted** (given the candidate, find the *definiens*, not the *definiendum*) | High where the candidate was extracted via a definitional pattern; lower where via emph alone | ~free (CPU regex) |
| **T2** | Adjacent-sentence inspection — definition often follows or precedes the emph-marked term within the same paragraph | Medium; catches "We define a *foo* to be ..." cases that T1 misses | ~free |
| **T3** | LLM-assisted extraction — pass paragraph + candidate, ask "is this term being defined here? If yes, what's the definition?" | High but slow; useful for hard cases | Token-cost; should run on T1+T2 misses only |
| **T4** | Cross-paper definition search — if the term appears in 5+ papers but no definition found in any single one, search all contexts | Medium; useful for canonical math terms with definitions taken-as-given | Compute-cost |

Tier-1 + T2 should run automatically; T3 should be batched (review-time, not extraction-time); T4 is research-stretch.

## 6. The feedback loop (closing the §2.A.2.20 gap)

```
Batch N processed with --discover-terms ON
    └─► candidate-new-terms.jsonl
    └─► definition-extractor (T1+T2)
        ├── provisional dictionary entries
        └── stopword candidates
            └─► operator/agent review
                ├── promote :provisional → :canonical (kernel TSV gets it)
                ├── promote :stopword candidate → :stopword (kernel ignores these forever)
                └── flag :uncertain (revisit)
            └─► kernel version-bump
                └─► batch N+1 sees enlarged kernel
```

**Review cadence** is the open operator-decision. Options: per-batch (high overhead, fastest learning), weekly (manageable), batch-of-batches (lowest overhead, slowest learning). The discover_terms candidate-new-terms file per batch is small (Joe's 2000-candidate-max default × ~1KB/entry = few MB max) — reviewable in an hour or two.

## 7. Buildable-now scope (the probe — what to ship)

**Stage 1 — week 1 (no arxiv compute needed):**

1. **Schema commit.** Land the EDN schema for dictionary entries + stopwords as `~/code/futon6/data/dictionary/schema.edn`.
2. **PM seed loader.** Python script `scripts/seed-dictionary-from-pm.py` that converts the existing PlanetMath corpus into dictionary entries. Output: `~/code/futon6/data/dictionary/entries-pm-seed.edn` (~19K entries).
3. **PM seed audit.** Sample 100 entries; verify schema fields populate sensibly; flag any PM articles that resist conversion (numeric-ID titles, malformed .tex, etc.).
4. **Stopword seed.** Hand-seed ~50 common stopwords (the kind §2.A.2.43 surfaced: "unique", "asymmetric", journal-name patterns) at `~/code/futon6/data/dictionary/stopwords.edn`.

**Stage 2 — week 2 (small arxiv pilot):**

5. **Definition-extractor T1+T2 implementation.** Python module `src/futon6/dictionary/extract.py`. Takes (term, context, paper-id, date) → returns dictionary-entry-shape or stopword-candidate.
6. **Pilot run.** Apply extractor to the 64 candidates surfaced by the 65KB PM .tex demo in §2.A.2.43. Classify: how many get T1/T2 definitions? How many drop to stopword? How many remain uncertain (T3 candidates)?
7. **Audit the pilot output.** Joe-side: do the auto-extracted definitions read as sane? Where does T1/T2 miss? Inform T3 LLM-prompt design.

**Stage 3 — weeks 3-4 (feedback loop alive):**

8. **Graduation tool.** Small CLI/Emacs-mode for reviewing provisional entries; promote :provisional → :canonical; promote :stopword candidate → :stopword.
9. **Kernel version-bump.** Script `scripts/dictionary-to-kernel-tsv.py` that emits an updated terms.tsv from the dictionary + stopwords. Test on PM seed first (should produce the same kernel we have today).
10. **First real cycle.** Run discover_terms on the next arxiv batch dispatched (per §2.A.2.43 flag-flip), feed candidates through the pipeline, review, version-bump.

## 8. Out of scope (named for foreclosure)

- Full corpus hypergraph (parent mission §6 Outcome A) — that's the *concept-layer-over-corpus* work; this excursion is *just the dictionary*
- Tutoring / pathway navigation (parent mission §6 Outcome C) — downstream consumer
- Self-play / formalisation (parent mission §6 lanes) — much further downstream
- Automated semantic drift detection — possible but speculative; defer
- Cross-language entries (e.g. German math terminology) — defer

## 9. Connection to existing futon6 prototypes (per devmap)

| Prototype | Relationship to this excursion |
|---|---|
| **P0 — Informal Argument Support** | Adjacent; P0 tags PM entries by reasoning pattern; dictionary entries get `:term/cross-refs :tagged-with-pattern` linkage when tagged |
| **P7 — StackExchange Import** | Adjacent; SE bodies → dictionary entries via the same extractor path |
| **P1-P6, P8-P9** | Greenfield; dictionary may enable several once it's running |
| **(implicit) Existing kernel TSV** | The dictionary is the structural successor; kernel TSV becomes a *projection* of dictionary entries + stopwords |

## 10. Sequencing relative to the 300K queue

The discover_terms flag flip (per §2.A.2.43) is the **load-bearing first step** — without it, no candidates flow, no dictionary populates. So:

| Order | Item |
|---|---|
| 1 | Rob enables `--discover-terms` on next batch dispatch (single-line; gated on Rob's return next week) |
| 2 | PM seed loader runs (Stage 1; doesn't need Rob's batch — fully local) |
| 3 | Stopword seed (Stage 1; fully local) |
| 4 | First arxiv batch with discover_terms returns; extractor runs over it (Stage 2) |
| 5 | Review cycle (Stage 3); kernel version-bump |
| 6 | Subsequent batches see the enlarged kernel; dictionary grows |

Steps 2-3 are pure local work and can start immediately. Step 4+ needs the flag flip to land.

## 11. Effort estimate

| Stage | Effort | Owner |
|---|---|---|
| Stage 1 (schema + PM seed + stopword seed) | ~1 week claude / codex-shift | claude-13 or codex |
| Stage 2 (extractor T1+T2 + pilot run + audit) | ~1-2 weeks | claude (extractor design) + codex (impl) + Joe (audit) |
| Stage 3 (graduation tool + kernel version-bump + first real cycle) | ~1-2 weeks | codex (tool impl) + Joe (review cadence + first cycle) |
| **Total to first-cycle-complete** | **~4-5 weeks** | mixed |

These are honest "weeks of focused effort" estimates; calendar-time will be longer given parallel work streams.

## 12. Success criteria for this excursion (per parent §8 "Deliverable standard")

- A working PM seed loader producing ~19K canonical dictionary entries from existing PM .tex (probe-level evidence)
- A working definition-extractor that produces auto-extracted entries for the §2.A.2.43 demo's 64 candidates with explicit classification (definition-found / stopword / uncertain)
- A graduation tool that allows operator review of provisional entries
- A kernel-TSV-from-dictionary roundtrip that produces an equivalent or larger kernel than today
- Honest reality-check on T3 (LLM-assisted) and T4 (cross-paper) — what they cost, when they're worth running

If those land, the **§2.A.2.20 feedback-loop gap closes** structurally: candidate-new-terms.jsonl becomes a *consumed* artefact, and the kernel evolves as the corpus grows.
