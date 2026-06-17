# Excursion: E-iatc-expository-alignment — do IATC's "ignored" categories show up in expository prose?

**Date:** 2026-06-17 · owner: Joe + claude-6 (paired) · **Status:** EXPERIMENT DESIGN (pre-registered; not yet run)
**Repo:** futon6 (scripts) + futon3c (close-reading artifacts).
**Prior art:** Corneli, Martin, Murray-Rust, Rino Nesin & Pease, *Argumentation Theory for
Mathematical Argument* (Argumentation 2019; arXiv:1803.06500) — the IATC spec (Joe is an author).
**Siblings:** `holes/excursions/E-iatc-model.md` (the IATC reasoning layer + its honest
"~2 of 5 categories implemented" census), `futon3c/holes/excursions/close-reading/`
(the empirical expository taxonomy: `expository-scope-hierarchy.edn`, the agent vote in
`proposals/`, `consolidation-report.{json,md}`).

## 1. The question

IATC was built mainly to model **proofs-in-progress** — dialogue about a developing
proof, where **performatives** (`Assert/Agree/Challenge/Retract/Define/Suggest/Judge/
Query`), **value** judgments (`easy/plausible/beautiful/useful`) and **meta** moves
(`goal/strategy/auxiliary/analogy/implements/generalise`) are relevant "in the flow."

In **published** papers those features are largely stripped from the *proof writeups*.
Joe's conjecture: a decent share of them resurface in the **expository sections** — the
narrative glue (motivation, connections, roadmap) around the formal proofs.

We now have a second, **independently derived** vocabulary for exactly those expository
sections: the close-reading-seeded `expository-scope-hierarchy.edn`, grown by a 4-agent
vote over ~193 papers. It was *not* seeded from IATC. So the alignment is a genuine
empirical question, with three acceptable outcomes (Joe): IATC's ignored categories
**slot into** the expository framework, **duplicate** it, or **diverge** from it.

**The alignment is a homecoming, not a stretch.** The IATC paper's explicit subject is the
*expository register* (FARM'17 §2): a register that "makes use of loose, heuristic
judgements of plausibility," adjectives like "deep"/"superficial," difficulty, strategies,
and analogies — exactly the layer Phase ⑤ targets. So IATC (top-down) and the expository
taxonomy (bottom-up from 193 papers) are two passes at the **same target**. And the paper's
§7 Future Work proposes *this experiment*: "statistical methods on the relevant corpora …
frequency of … schematic usages like 'let X be a Y' … expanding on Kaliszyk et al." This
excursion is that future work, generalised from "let X be a Y" to the full IATC inventory.

> **IATC vocabulary** is now **complete**, transcribed from the journal version
> (`s10503-018-9474-x`, Corneli et al. 2018) **Tables 1 & 2**: **9 performatives** + **25
> intermediate relations** (9 inferential `rel[…]` + 4 `value[…]` + 6 `meta[…]` + 6
> `struct[…]`). The FARM'17 "15 relations" was the earlier, smaller tag set; the journal
> co-authors later extended it (e.g. `analogy`, `sums`). `E-iatc-model.md` §2 matches this
> Table 2 exactly. Methodology: close content analysis of MiniPolymath 1 & 3 by 4 co-authors.

## 2. Pre-registered crosswalk (the hypothesis to test)

Reasoned prediction (claude-6, 2026-06-16). The point of the experiment is to confirm or
falsify these *before* measuring.

| IATC feature | Predicted outcome | Maps to expository kind |
|---|---|---|
| `meta[goal]` | **slot-in** | `rationale/telos` |
| `meta[strategy]` | **slot-in** | `rationale/telos/organization-roadmap` |
| `value[useful]` | **slot-in** | `rationale/telos` / `connection/application-domain` |
| `perf[Suggest]` | **slot-in / duplicate** | `connection/transfer` |
| `meta[analogy]` | **duplicate** | `connection/transfer` |
| `struct[used_in]` | **duplicate** | `connection` |
| `struct[instantiates]` | **duplicate** | `connection/example-source` |
| `perf[Query]/[QueryE]` | **duplicate** | `open-problem/status` |
| `perf[Agree]`, `perf[Retract]` | **diverge (IATC-only)** | — dialogical; no interlocutor in a monologue |
| dialogical `perf[Challenge]` | **diverge (IATC-only)** | — (content-challenge ≈ `obstruction`, but the *act* is absent) |
| `value[beautiful]`, `[easy]`, `[plausible]` | **diverge (IATC-only)** | — aesthetic/epistemic; stripped or collapsed into `telos` |
| `meta[auxiliary]`, `[generalise]`, `[implements]` | **diverge (IATC-only)** | — proof-development tactics |
| `rel[*]`, `perf[Assert]`, `perf[Define]` | **out of scope** | the FORMAL layer = Phase ④ illative-only IATC |
| (no IATC source) | **expository-only** | `universal-property/characterizes`, `connection/literature-gap(+/terminology-origin)`, `obstruction`, `computes-invariant/calculation` |

**One-line thesis:** the expository taxonomy ≈ IATC's reasoning layer **minus** the
dialogical/aesthetic features **plus** published-genre positioning moves. The seam is
*what survives the transition from proof-in-progress dialogue to published monologue.*

## 3. Data

- **`data/warp/gh200.txt`** — 200 cross-discipline arXiv papers (mixed `math__*` legacy +
  modern numeric ids) selected for the most recent run. 182 rendered; ~96% of a sampled 50
  carry golden DP marks (`data/showcases/ct-anatomy/golden/fable-<id>-dp-emacs.json`), so
  **expository-region extraction runs CPU-side on the laptop.**
- **Existing agent vote** — `futon3c/.../close-reading/proposals/{codex-1,codex-2}.proposals.jsonl`
  (+ the report counts 4 agents, 47,843 proposals over 193 papers). Row schema:
  `{paper, region_id, region_type, line, quote, kind, confidence, source_class, new_subkind}`.

## 3b. The exemplar bank — "what we're looking for"

Without concrete usage examples we cannot recognise a category's *alternative phrasings* in
arXiv — and "finding alternative phrasings of known scopes" is the mark4 GPU theme. So the
experiment is anchored by a tiered bank of **(category → exemplar phrasing)** pairs.

### The IATC inventory to find exemplars for (journal Tables 1 & 2 — complete)

**Performatives `perf[…]` (9):** `Assert(s[,a])` assert s true (opt. because a) ·
`Agree(s[,a])` agree with prior s · `Challenge(s[,a])` assert s false · `Retract(s[,a])`
retract prior s · `Define(o,p)` define o via p · `Suggest(s)` suggest a strategy s ·
`Judge(s,v)` apply value judgement v to s · `Query(s)` ask truth-value of s ·
`QueryE({pᵢ(X)})` ask for the class of X where all pᵢ hold.

**Intermediate relations (journal Table 2 — complete, 25):**
- **Inferential `rel[…]` (9):** `implies(s,t)` · `equivalent(s,t)` · `not(s)` ·
  `conjunction(s,t,…)` · `has_property(o,p)` · `instance_of(o,m)` · `indep_of(o,d)` ·
  `case_split(s,{sᵢ})` · `wlog(s,t)` (t equiv s but easier to prove)
- **Value `value[…]` (4):** `easy(s[,t])` · `plausible(s)` · `beautiful(s)` ·
  `useful(s)` (can be used in an eventual proof)
- **Meta / reasoning tactics `meta[…]` (6):** `goal(s)` (with `Suggest`, to direct others) ·
  `strategy(m,s)` · `auxiliary(s,a)` · `analogy(s,t)` · `implements(s,m)` · `generalise(m,n)`
- **Content-structural `struct[…]` (6):** `used_in(o,s)` · `reform(s,t)` ·
  `instantiates(s,t)` · `expands(x,y)` · `sums(x,y)` · `cont_summand(x,y)`

### Tier 1 — canonical (the papers)
FARM'17 Fig 2 + Listings 1–2: Gowers's walkthrough of *"the 500th digit of (√2+√3)²⁰¹²"*,
every utterance IATC-tagged. The **seed exemplar**, e.g.:
- `Suggest(strategy …)` — *"the trick might be: it is close to something we can compute"*
- `Judge` / `beautiful` / `is-small` — *"(√3−√2)²⁰¹² is a very small number. Maybe the answer is 9?"*
- `analogy` / `generalise` — *"Can we do this for x+y? For e? Rationals with small denominator?"*

And the journal's Fig 4 (MiniPolymath) shows the value/meta combination directly:
*"The following reformulation of the problem may be useful: show that…"* →
`perf[Assert](rel[equivalent])` + `perf[Judge](value[useful])` + `perf[Suggest](meta[goal])`
+ `struct[used_in]`. The journal's worked corpus is **MiniPolymath 1 & 3** (close content
analysis by the co-authors) — the canonical dialogue exemplar source.

These are **dialogue**, not published prose — which is precisely why arXiv presence is the
open question.

### Tier 2 — natural dialogue (MathOverflow / math.SE)
Processed, on the laptop: `~/code/storage/futon6/se-data/{mathoverflow.net,math.stackexchange.com}`
+ ready samples e.g. `futon5/data/stackexchange-samples/mathoverflow.net__category-theory.jsonl`.
IATC's **native habitat** (and an original IATC data source). The dialogical performatives
(`Agree/Challenge/Retract`) and value judgements should be densest here. This tier is the
**baseline**: a category common in MO/SE but absent in arXiv expository prose *is* the
"stripped on publication" signal — the experiment's headline measurement.

### Tier 3 — synthetic bridge (APM)
APM prelim proofs already carry an informal **"why is this hard"** component
(`data/apm-crossdisc-pool`, `apm-proof-scope-audit.json`). Two uses: (a) mine the existing
why-hard prose for value/meta exemplars; (b) reconstruct a handful of APM proofs as
**imagined dialogues** that exhibit the categories — semi-formal exemplars bridging
dialogue → published prose, and a way to *exhibit* a category before hunting alternative
phrasings of it.

### How the bank is used
Each exemplar is a `(category, phrasing)` pair. Pass B shows the agents these exemplars and
measures whether *alternative phrasings* of each category appear in arXiv expository regions;
the mint outcome per category is the arXiv-presence verdict.

## 4. Method — two CPU-side passes

### Pass A — mine the existing vote (immediate, pure CPU, uses data already collected)

The agents already did an **open-ended** classification of the gh200 expository regions
(they could propose any `new_subkind` with a definition + rationale). So we can ask, with
**zero new model calls**:

1. **Enumerate the distinct agent-proposed sub-kinds** and bin each against IATC
   `perf/value/meta`. *(Preliminary scan, 2026-06-17, 12,513 proposals from codex-1/2: only
   **9** distinct sub-kinds, **all** in the glue families — `terminology-origin`,
   `naming-convention`, `example-source`, `application-domain`, `organization-roadmap`,
   `counterexample-status`, `terminology/convention`. **Zero** aesthetic, dialogical, or
   proof-tactic proposals.)*
2. **Extend `expository-scope-hierarchy.edn`** with the IATC `perf/value/meta` features as
   candidate kinds (each with synonyms + definition + parent), and re-run
   `consolidate_scope_votes.py`. Read which IATC kinds attract support via synonym
   resolution vs receive none.

**What Pass A can and cannot show.** It can show **evidence of absence** — that, given
freedom to coin categories, agents never reached for `beautiful/Agree/Retract/generalise`
— and that the slot-in/duplicate IATC kinds are already represented by existing kinds. It
**cannot** show presence of an IATC category the agents were never offered: the vote was
seeded with the expository hierarchy, not IATC, so a null for an IATC kind in Pass A is
**confounded** (not-offered vs not-present). Hence Pass B.

### Pass B — seeded re-vote (the controlled test)

Add the IATC `perf/value/meta` features as **first-class candidate bins** (with the §3b
exemplars) to the hierarchy, then re-run the agent classification over a gh200
subset so agents can bin into IATC kinds directly; consolidate with the existing
**mint-threshold** (≥5 papers ∧ ≥2 agents, else resolve-to-parent). The mint outcome is the
measurement:

- an IATC kind that **mints** → it genuinely occurs in published expository prose
  (slot-in/duplicate **confirmed**);
- one that **resolves-to-parent or never fires** → stripped/absent (**diverge confirmed**).

The vote is the only model step; it is small (a gh200 subset, a handful of agents) and is
the same Codex-pool + handoff-review method that produced the original taxonomy. Region
extraction, hierarchy editing, consolidation, and the discovery/saturation curve are all
CPU-side.

## 5. Outcomes & how to read them

For each IATC feature, the experiment yields one of: **mint** (slot-in/duplicate), or
**resolve/absent** (diverge). Compare the measured column against the Pre-registered column
in §2. Three headline checks:

1. Do `meta[goal/strategy/analogy]`, `value[useful]`, `perf[Query/Suggest]` mint? (predict yes)
2. Do `value[beautiful/easy/plausible]` and `perf[Agree/Retract]` stay absent? (predict yes)
3. Does the discovery curve **saturate** without the IATC additions changing the existing
   minted set much? (predict yes — they mostly slot into existing kinds)

## 6. Threats to validity

- **IATC list provenance** — from `E-iatc-model.md` §2, not re-read from the paper; Joe to confirm.
- **Pass A confound** — agents were primed with the expository hierarchy; a null is
  not-offered, not necessarily not-present (Pass B addresses this).
- **Synonym matching is approximate** — `consolidate_scope_votes` resolves lexically; IATC
  kinds need carefully chosen synonyms or matches will be missed.
- **Corpus skew** — gh200 is arXiv- and CT-heavy; "absent" means "absent in this corpus,"
  not "never occurs." Aesthetic judgments may simply be rare.
- **Collapse, not absence** — `value[useful]` already collapses into `rationale/telos`
  (anchor: *"Frobenius algebras are interesting"*); a mint failure may mean *subsumed*, not
  *missing*. Report subsumption separately from true absence.

## 7. Relation to prior work

- **Extends the IATC paper**: an empirical map of which IATC categories survive into
  published mathematical prose, and where the expository layer adds genre-specific moves
  (`literature-gap`, `universal-property`, `obstruction`) IATC did not enumerate.
- **Typed-holes**: each expository kind is already a typed hole (`:hole {:slot :type}` +
  `:fill`), so a confirmed IATC↔expository alignment extends the same `meme → arrow → sorry`
  model used for the formal layer (see the cascade/sorry excursion + `M-typed-holes`).
- **Feeds Phase ⑤** (`pre-superpod-pipeline-readiness.html`): the aligned, IATC-augmented
  hierarchy becomes the typed target schema for the unbuilt GPU hole-filling stage (5.4).

## 8. First concrete step

Pass A, step 1 is essentially done (the 9-sub-kind scan above). The next CPU action is to
draft the IATC `perf/value/meta` seed block for `expository-scope-hierarchy.edn` (synonyms +
definitions + parents per §2), then re-run `consolidate_scope_votes.py` over the existing
proposals and tabulate support per IATC kind. Pass B follows if Pass A motivates the
controlled re-vote.

**Update (2026-06-17): Pass A is run — see §9 (executed as a direct cue-scan over the 47,843
quotes, a cleaner variant of the re-consolidation route).**

## 9. Pass A — first results (2026-06-17)

`scripts/iatc_alignment_passA.py` cue-scans all 47,843 proposal quotes (193 papers, 4 agents)
for IATC `perf/value/meta` phrasings (struct/rel excluded — the formal-scope-covered content
side, per Joe). Cues are lexical and tunable; the table uses cues tightened after a
false-positive audit (below). `%papers` = distinct papers with ≥1 cue hit.

### Logged gap — struct relations may live in proofs, not exposition (IATC / Phase ④)
Joe (2026-06-17): the content-structural moves `expands`, `used_in`, `reform`, `instantiates`
(and `sums`, `cont_summand`) *may* appear, but more likely **inside proofs** than in expository
sections — so they belong to the IATC/formal layer (Phase ④), not here. Excluded from this
expository scan; **logged as a Phase ④ gap, to measure against proof regions later.**

| family | category | predicted | %papers | verdict |
|---|---|---|---|---|
| perf | `Agree` | diverge | 1.0% | rare ✓ |
| perf | `Challenge` | diverge | 1.6% | rare ✓ |
| perf | `Retract` | diverge | 0.0% | absent ✓ |
| perf | `Suggest` | slot-in | 21.2% | PRESENT ✓ |
| perf | `Query` | duplicate | 16.6% | PRESENT ✓ |
| value | `easy` | diverge | 51.3% | PRESENT ✗ refuted |
| value | `plausible` | diverge | 13.0% | PRESENT ✗ refuted |
| value | `beautiful` | diverge | 7.8% | rare ✓ |
| value | `useful` | slot-in | 54.9% | PRESENT ✓ |
| meta | `goal` | slot-in | 39.4% | PRESENT ✓ |
| meta | `strategy` | slot-in | 21.2% | PRESENT ✓ |
| meta | `auxiliary` | diverge | 26.9% | PRESENT ✗ refuted |
| meta | `analogy` | duplicate | 52.3% | PRESENT ✓ |
| meta | `implements` | diverge | 3.1% | rare ✓ |
| meta | `generalise` | duplicate | 72.5% | PRESENT ✓ |

### Cue-contamination audit (why three cues were tightened)
Raw cues over-counted three categories via math-polysemy:
- `Retract` 10.9% → **0%**: bare "retract" is the *math* term (section-retraction); only the
  speech act counts.
- `easy` 77.7% → **51.3%**: dropped `trivial/elementary/immediate/routine` (trivial group,
  elementary topos, immediate consequence); kept the genuine hedge "easy to see/show",
  "straightforward", "clear/obvious that".
- `beautiful` 33.2% → **7.8%**: dropped broad `nice/striking/surprising/remarkable`; true
  aesthetic `beautiful/elegant` is rare (even "elegant" has FPs — "elegant Reedy category").
- `analogy`/`generalise`/`useful` (2nd round): dropped `extends to`/`can be extended`
  (content — "the functor extends to …"), `in the same way` (definitional), `mirrors`/`parallels`
  (mirror symmetry), `helpful`/`invaluable` (acknowledgements). **Verdicts unchanged** — the audit
  removed inflation, not signal: `analogy` 60.6→52.3%, `generalise` 76.2→72.5%, `useful` 59.1→54.9%.

### The corrected seam (the finding)
The pre-registration was right about the **dialogical** and **aesthetic** features but wrong
about heuristic **evaluation/tactics**:

- **Confirmed stripped (rare/absent in arXiv expository prose):** the interpersonal dialogue
  acts `Agree` (1%), `Challenge` (1.6%), `Retract` (0%); the `implements`-a-suggested-strategy
  act (3%); and pure-aesthetic `beautiful` (8%). These need an interlocutor or are flourish —
  what published monologue drops.
- **Refuted — survives (predicted diverge, measured PRESENT):** `easy` (51%, "it is easy to
  see"), `plausible` (13%, "intuitively/we expect"), `auxiliary` (27%, "the following lemma").
  The author's own heuristic hedging is alive in published prose.
- **Confirmed present (slot-in/duplicate):** `Suggest`, `Query`, `useful`, `goal`, `strategy`,
  `analogy`, `generalise`.

So the seam is sharper than "dialogue/aesthetic stripped": **what's stripped is specifically
multi-agent dialogue + aesthetic flourish; the author's heuristic reasoning and evaluation
(`goal/strategy/auxiliary/analogy/generalise/useful/easy/plausible`) survives into the
expository register.** This tells Phase ⑤ which IATC categories to fold in for arXiv (the
surviving set) vs reserve for the MO/SE dialogue tier (the stripped set).

### Caveats / next
All 15 cues have now had a false-positive audit and verdicts are stable under tightening, but
lexical matching on agent-selected quotes remains a first filter. **Pass B**
is the confirmatory test: offer the IATC categories (with §3b exemplars) to agents as bins, and
use the **MO/SE dialogue tier as the baseline** to confirm the stripped categories are common
*there* — i.e. "stripped on publication," not "nonexistent."

## 10. MO/SE dialogue baseline + alternative-phrasing harvest (2026-06-17)

`scripts/iatc_mose_scan.py` scans 200 MathOverflow / math.SE sample threads
(`futon5/data/stackexchange-samples/*.jsonl`; q=200, answers=313, comments=1224) — IATC's
native habitat — with the SAME Pass-A cues, plus an informal-dialogical probe and an example harvest.

### Headline: the dialogical performatives are stripped on publication
With register-appropriate (informal) cues, the dialogue acts are common in MO/SE but rare/absent
in arXiv expository prose:

| category | MO/SE %threads (informal) | arXiv %papers | verdict |
|---|---|---|---|
| `Agree` | 29.0% | 1.0% | **STRIPPED ON PUBLICATION** |
| `Challenge` | 7.5% | 1.6% | stripped |
| `Retract` | 5.5% | 0.0% | stripped |

This is the missing half of the Pass A claim: "rare in arXiv" + "common in dialogue" =
**stripped on publication, not nonexistent.**

### The arXiv-tuned cues undercount dialogue ~14× (why the harvest matters)
The strict (Pass A) cues fire on only 2.0% of MO/SE threads for `Agree`, vs 29.0% with informal
cues — a ~14× undercount (`Retract` ~11×). MO/SE phrases these moves informally ("I agree",
"good point", "you're right", "oops", "on second thought", "edit:") — exactly the **alternative
phrasings** the lexicon needs. Consequence: the strict `%threads` column is NOT a valid rarity
comparison for the non-dialogical categories either (a thread ≪ a paper in text; register
differs). The valid signal is the informal-vs-arXiv gap above.

### Harvested alternative phrasings (the LLM seed + Tier-2 exemplars)
Real dialogue instances pulled per category (full set via `--dump`):
- `Agree`: *"As Rasmus points out, monomorphism is a purely categorial notion."*
- `Challenge`: *"But I believe this is false."*
- `Retract`: *"@TomLeinster On second thought, I think you're okay: just replace 'colimit' by 'filtered colimit'…"*
- `Suggest`: *"Somehow the trick is to pick out the morphisms…"*
- `goal`: *"Our aim is to show that λ is the cocone of a colimit."*

### Next — the LLM harvest
The informal probe is a hand-built first pass at alternative-phrasing collection. The LLM step
(Joe: "feasible for an LLM"): read MO/SE posts + the IATC category defs + current cues, propose
and validate alternative phrasings per category, fold the survivors back into the cue lexicon
(improves Pass A recall) and the §3b exemplar bank (Tier 2). Then re-run Pass A with the expanded
cues for a register-robust arXiv table.

**Done (2026-06-17): the LLM harvest is in `iatc-mose-alt-phrasings.md`** — alternative phrasings
per category with verbatim MO/SE examples + proposed cue additions. Biggest recall fix: the
`plausible`/value hedge is dominantly "I think / I believe / I'm pretty sure" (cues missed it);
the dialogical Agree/Challenge/Retract are confirmed common-and-informal in comments, absent in arXiv.

## 11. Wrap-up — the arXiv expository vocabulary for the superpod runner (2026-06-17)

Focus is arXiv mining, so the alignment work is consolidated into a runner-ready target schema:
**`holes/excursions/expository-superpod-vocab.edn`** — the finalized expository-scope vocabulary
the superpod expository stage (Phase ⑤.4) classifies + fills. It is the empirical mint taxonomy
∪ the IATC categories that survive into arXiv ∖ the categories stripped on publication.

**The alignment's payoff — 4 new scope kinds** the IATC-blind vote never minted but Pass A shows
are common in arXiv expository prose, now added as typed holes:
- `:difficulty-assessment` (← `value/easy`, 51% papers) — "it is easy to see", "straightforward"
- `:heuristic-plausibility` (← `value/plausible`, 13%) — "intuitively", "we expect", "presumably"
- `:auxiliary-construction` (← `meta/auxiliary`, 27%) — "the following lemma", "the key is"
- `:generalisation` (← `meta/generalise`, 72%) — "more generally", "still applies if/when"

**The skip-list for arXiv** (`:out-of-scope-arxiv`) — don't hunt these in the superpod arXiv run;
they're stripped on publication (reserve for the MO/SE round): `perf/Agree`, `perf/Challenge`,
`perf/Retract`, `meta/implements`, `value/beautiful`.

**Covered elsewhere:** `rel[*]` = the Phase ④ formal IATC argument layer; `struct[*]`
(expands/used_in/reform/instantiates/sums/cont_summand) = content moves likely in PROOFS, logged
as the Phase ④ gap (§9).

### How the superpod runner consumes it (the Phase ⑤.4 contract)
Mirror the IATC stage (Phase ④) with this vocab as the target:
1. **carve** expository regions (`expository_region_extract.py`, Phase ⑤.1) → candidate regions;
2. **classify + fill** (GPU): assign each region a `:kind` from `:scopes` and fill its `:hole`
   `:slot` with source-anchored text — skipping `:out-of-scope-arxiv`;
3. **gate**: an expository checker (an `iatc_argcheck` sibling) validates every scope has a
   resolved `:kind`, a filled-or-explicitly-held `:slot`, and a source locus;
4. few-shot from the §3b Tier-1/Tier-3 exemplars (Tier-2 MO/SE exemplars deferred with the
   cross-corpus round).

This finalizes Phase ⑤ on the arXiv side: the taxonomy is closed under what we now know
survives publication, the GPU stage has a concrete typed-hole target, and the dialogue-only
categories are explicitly out of scope rather than silently missing.
