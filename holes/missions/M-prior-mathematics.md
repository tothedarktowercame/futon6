# Mission: Prior Mathematics — a corpus base-rate prior for Stage 5 NER

**Date:** 2026-05-31
**Status:** step-1 CT prior BUILT (full corpus, 2026-05-31); step-2 posterior-vs-prior test next; Joe framed it, claude-2 drafted
**Owner:** Joe (frames) / claude-2 (drafted)
**Predecessor:** [M-bayesian-structure-learning.md](M-bayesian-structure-learning.md)
  (this is one concrete, shippable instance of "accumulate posteriors, not counters")
**Timing constraint:** Rob is going live with the superpod runner on the CT
  dataset soon. A base-rate prior that suppresses the junk tail is worth having
  BEFORE that run. Scope this mission to the **CT dataset we hold locally**, not
  all of arXiv. **Scope note (Joe's correction, 2026-05-31):** building the prior
  ON CT vocabulary is the POINT, not a limitation — CT vocab then *defines what is
  normal*, so legitimate CT terms (functor, morphism, pretopos) sit at the mode
  of the distribution and CANNOT be flagged by the over-detection-surprise signal.
  codex-2's audit worried CT-only would "over-penalise legitimate CT vocabulary";
  that is wrong for the surprise mechanism (corrected). The only genuine residual
  is the *general*-arXiv claim: a CT-fit prior is a CT-runner guardrail, not a
  universal arXiv prior — and NEW/emerging CT terms (rare in the fit corpus) can
  still look anomalous, which is the trending-vs-hallucination hole (§2.3), a
  different and already-named concern.

## 1. Why this mission exists

Joe's idea (2026-05-31, verbatim sense): Zipf / Pareto / power laws hold across
arXiv word distributions. We can build a **prior** for the Stage 5 NER routines
just from those distributions — no labels needed. The motivating failure: in one
pilot, "Stable Marriage Problem" was being detected *everywhere*. With due
respect to real spousal difficulties, that is a base-rate impossibility, and we
should be able to catch it automatically.

### The failure, observed in real data (verified 2026-05-31, not assumed)

`storage/arxiv-paper-hg-gpu/candidate-new-terms.jsonl` (the CT corpus). Top
"new terms" by candidate_count:

| term | candidate_count | dominant source |
|---|---|---|
| `objects` | 404 | `latex-emph` (361 / 404) |
| `left` | 401 | `latex-emph` (352 / 401) |
| `cartesian` | 305 | `latex-emph` (295 / 305) |

`objects` and `left` are generic words the extractor latched onto because they
sat inside `\emph{...}`. (`cartesian` is NOT junk — it is a real CT property;
included here only because it tops the count list, and it illustrates why
latex-emph alone cannot separate junk from real terms — see §2.) This is the
same failure as "stable marriage everywhere" (stable marriage
itself: 0 hits in THIS corpus, as expected — it's category theory). The summary
(`candidate-new-terms-summary.json`) shows the scale: **193,098 total
extractions, 147,231 "unknown", 75,431 unique unknown terms.** A massive
over-detection tail.

### The current mitigation is hand-rolled whack-a-mole

Stage 5 already tries to suppress this with **hand-curated blocklists** in
`futon6/scripts/superpod-job.py` (~lines 601-620): `DISCOVERY_STOPWORDS`,
`DISCOVERY_GENERIC`, `DISCOVERY_QUALITY_SINGLE_WORDS`, `DISCOVERY_QUALITY_BAN_TOKENS`,
`DISCOVERY_ADJECTIVAL_SUFFIXES`. **Tell-tale evidence this is reactive patching:
`cartesian`, `left`, and `objects` — the exact three junk terms topping the
candidate file above — already appear in `DISCOVERY_QUALITY_SINGLE_WORDS`.**
Someone saw the junk and hand-added those words. (Caveat to confirm: the
candidate file may predate the blocklist additions, or the list may apply at a
different stage — verify the ordering before asserting causation.)

The mission's claim: **replace the hand-maintained blocklist with a principled,
label-free corpus base-rate prior** that generalises to junk terms nobody
thought to ban.

## 2. The prior (proposal — to be pinned down in DERIVE)

**Crucial framing (Joe, 2026-05-31): there are TWO distinct failure modes with
OPPOSITE relationships to base rate. They need different prior terms; conflating
them is the error that produced the bad "latex-emph separates junk" and
"over-penalises CT vocab" claims.**

| failure | example | flag condition | risk |
|---|---|---|---|
| over-detection of a rare term | "stable marriage everywhere" | detection-rate ≫ its base rate | low — rare-but-real terms only flagged if genuinely over-stamped |
| generic word grabbed as a concept | `\emph{objects}`, `\emph{left}` | base rate too high / too ubiquitous to be a special concept | HIGH — a naive genericness penalty also hits common-but-legitimate CT vocab (functor, morphism) |

The base rate is fit ON the CT corpus, so CT vocabulary IS the norm. That makes
the **over-detection signal safe for legitimate CT terms** — they define the mode.
The genericness case is the dangerous one and is NOT purely lexical: "objects" is
ITSELF a legitimate CT concept (objects of a category); the failure is that *this
extraction* — `\emph{objects}` in "the objects of 𝒞" — is not a new-term
definition. So the objects/left failure is **contextual, not lexical**, and
frequency alone (either direction) cannot fix it.

The prior, three terms:

1. **Marginal base-rate surprise (primary, safe).** Fit the rank-frequency
   (Zipf) / tail (Pareto) distribution of accepted concepts across the CT corpus
   → prior `P(concept)`. Flag extractions whose local assignment rate sits far
   ABOVE the prior. Catches "stable marriage everywhere." Does NOT penalise
   legitimate CT vocab (they are the distribution). Does NOT by itself catch
   "objects/left" (those are high-base-rate, not over-detected) — that is the job
   of context features, not a genericness hammer that would clobber "morphism".

2. **Source-profile conditioning (a junk-RISK feature, NOT a clean
   discriminator — corrected after codex-2 audit 2026-05-31).** The junk terms
   are ~90% `latex-emph` (objects 89.4%, left 87.8%, cartesian 96.7%, verified).
   The tempting claim was "high count + high latex-emph SEPARATES junk from
   genuine concepts." **codex-2 falsified that:** genuine CT terms also have very
   high latex-emph fractions — `pretopos` 94.4%, `lextensive` 100%, `bicategory`
   68.8%, `operad` 61.5% — and `cartesian` is mathematically real in its own
   context ("called cartesian if Q preserves pullbacks"), not junk. So latex-emph
   is a **junk-risk prior**, not a separator. It must be COMBINED with the
   base-rate term (1) and/or collocation (3) to discriminate; on its own it
   over-penalises legitimate CT vocabulary. Still computable from the `sources`
   field, still useful — but demoted from "the term to lean on" to "one feature
   requiring validation."

3. **(Conjecture, lower priority) Collocation coherence.** A real "stable
   marriage" mention drags in Gale–Shapley, blocking pair, preference list; a
   hallucinated one shows up *naked*. Missing-collocates is a stronger
   confabulation signal than raw count. Marked conjecture — verify it adds lift
   over (1)+(2) before building it.

## 2b. Architecture: hierarchical (per-category) priors (Joe, 2026-05-31)

The prior is **not single** — it is **one base rate per arXiv category**, with a
global prior as the hyperprior each category shrinks toward. This is hierarchical
empirical Bayes: `P(term | category)` with partial pooling to `P(term | global)`
for categories/terms with thin counts. CT is just the first category — small
enough (~5000 papers) to fit a near-complete prior locally; the others seed from
Rob's broader harvest.

**Why per-category matters (resolves the earlier confusion):** a term's
"normal-ness" is category-relative. "stable marriage" is mode-ish in math.OC /
econ but tail in math.CT — so a CT-fit prior flags it, correctly. "sheaf" is
normal in math.AG/CT but anomalous in math.PR. A single global prior would blur
these; per-category priors are what make the over-detection signal sharp. The
global prior only backs off thin cells.

**What's actually on the laptop (verified 2026-05-31):**
`storage/arxiv-manifest/arxiv_manifest.sqlite` (785MB) — Rob's harvest:
- **570,209 papers**, **30 distinct primary categories**, all `math.*`
  (math.AP 56,499; math.CO 52,177; math.PR 43,389; math.AG 39,581; … ;
  **math.CT 4,616**, confirming the ~5000 figure). NOT yet all of arXiv — no
  hep-th/cs.* — but a full math-wide per-category corpus.
- **Important asymmetry:** `local_path` is empty for all rows → the manifest holds
  **metadata only (title + abstract)**, not full eprints. CT full-text IS local
  (~5,913 eprints, per the candidate-new-terms summary). So:
  - CT prior → can be fit from **full text** (rich, near-complete).
  - other-30-categories priors → fit from **abstracts only** (thinner, but ample
    for a term base rate).
  - global prior → pooled across all 30 abstract-level category rates.
  This asymmetry is fine for a base rate but must be stated: CT and non-CT priors
  are fit on different text granularities; do not compare their absolute
  frequencies naively, compare each term to ITS OWN category's distribution.

**Staging vs Rob's run:** ship the CT (full-text) prior first as the guardrail
for the imminent CT runner; seed the other 29 math categories from manifest
abstracts in parallel; the global prior is the partial-pooling backstop and the
start of the eventual all-arXiv prior once Rob's harvest broadens past math.

## 3. Local data surfaces (verified to exist 2026-05-31)

- `storage/arxiv-manifest/arxiv_manifest.sqlite` (785MB) — 570,209 papers,
  30 math primary categories, metadata (title+abstract) only; `papers` table,
  columns `primary_category`, `categories_json`, `title`, `abstract`. The
  per-category prior corpus.
- `futon6/data/arxiv-coherence-mathct-50.json` — CT coherence sample.
- `storage/math-ct-proofs-elided-starter/` — CT proofs starter set.
- `storage/math-processed-gpu/` — processed CT embeddings (`embeddings.npy` 3.1G,
  hypergraph + faiss).
- `storage/arxiv-paper-hg-gpu/candidate-new-terms.jsonl` (2000 rows) +
  `-summary.json` — the over-detection evidence and the `sources` field the
  source-profile prior runs on.
- Existing Stage 5 code: `futon6/scripts/superpod-job.py`
  (`extract_open_ner_candidates`, `_normalize_discovered_term`, the blocklists).

## 4. Relevance to M-differentiable-code (tangential, recorded not scoped)

Per Joe: keyword-frequency analysis is weaker for code than for prose. But Zipf
holds *structurally* in source (a few hub namespaces, many leaf defns;
heavy-tailed node-degree; stable edge-type frequencies). The homolog of
"latex-emph dominance" is **edge-type provenance concentration** — a code node
whose edges are all one weak type (string co-occurrence, no real
`:requires`/`:defines`) is the same naked-detection smell. There the power law
becomes a **prior/regulariser on the soft-adjacency tensor `A[n,r,target]`** in
M-differentiable-code: a proposal that flattens the degree distribution is a
priori implausible. Same math, different substrate. Not in scope here.

## 5. First concrete probe (when picked up)

Label-free and cheap, runs on the local candidate file:
1. From `candidate-new-terms.jsonl`, compute per-term `latex-emph` fraction.
2. Rank by (high candidate_count AND high latex-emph fraction).
3. Check the SEPARATION directly (this is the make-or-break test, per codex-2):
   does the ranking put known junk (objects, left) ABOVE genuine CT terms
   (pretopos, lextensive, bicategory, operad, cartesian)? codex-2 already showed
   those genuine terms ALSO have high latex-emph fractions (94.4% / 100% / 68.8%
   / 61.5%), so latex-emph alone will NOT separate them. The real test is whether
   adding the base-rate term (1) — genuine CT terms are *rare-but-real*, generic
   words are *common-everywhere* — recovers the separation latex-emph cannot.
4. Compare the combined catch-set against the hand-curated blocklist: terms it
   catches that the list misses = generalisation win; genuine terms it wrongly
   flags (false positives like cartesian/pretopos) = where the prior is still
   over-penalising and needs collocation (3) or a held-out-CT calibration.

## Step 1 result: CT prior built and validated (2026-05-31)

**Prior = posterior-vs-prior framing (Joe, 2026-05-31).** The over-detection
signal IS divergence of our extractor's posterior from the corpus's true prior:
- **Prior** `P(term | CT corpus)` — built from RAW eprint .tex, extractor-independent.
- **Posterior** `P(term | our extractor)` — the mark2 `ner-terms/*.json` output.
- **Flag** = posterior ≫ prior. "stable marriage everywhere" = posterior diverging
  up from a near-zero prior. So the prior MUST come from raw text, never ner-terms;
  ner-terms is what we TEST against it (step 2).

**Builder:** `futon6/scripts/build_ct_prior.py` — reads every math.CT eprint,
light LaTeX strip, computes **document-frequency** (papers-containing-term, the
right denominator for over-detection) for unigrams + bigrams. Output
`futon6/data/ct-term-prior.json` = `{n_docs, unigram_df, bigram_df}`.

**Data format finding (verified):** the eprints in
`storage/futon6/data/arxiv-math-ct-eprints/` (9,798 files) carry a `.tar.gz`
extension but are **single gzip-compressed .tex files, NOT tarballs** — read via
`gzip.open`, not `tarfile`. No per-file meta; the directory is the math.CT filter.

**Full-corpus prior (MEASURED, full build 2026-05-31): n_docs=9,742, n_err=0,
unigram vocab 1,978,452, bigram vocab (df≥3) 1,076,058, output 50MB.** (56 of the
9,798 files yielded no decodable .tex and were skipped at read time — 9,742/9,798
= 99.4% coverage; the un-decodable 56 are a known gap, not silently dropped.)

| term | P(term\|CT) full | class it confirms |
|---|---|---|
| category | 0.945 | ubiquitous (field name) |
| objects | 0.897 | **high base rate → generic-grab risk (NOT over-detection)** |
| left | 0.879 | high base rate → generic |
| functor / morphism | 0.875 / 0.841 | legit CT vocab, at the mode |
| cartesian | 0.395 | real-but-common; vindicates "not junk" |
| sheaf | 0.190 | real, moderate |
| bicategory | 0.135 | real, specialised |
| operad | 0.121 | real, rare |
| pretopos | 0.0154 | real, very rare |
| lextensive | 0.0114 | real, very rare |
| **stable marriage** | **0.0000** | **absent → over-detection fires on any extraction** |

(The 200-doc validation sample gave near-identical figures — objects 0.925, left
0.88, stable-marriage 0.0 — so the estimate had already converged; full-corpus
values above supersede it.)

This **empirically confirms the two-mechanism split (§2):** generic-grab terms
(objects/left) have HIGH prior → caught by a high-base-rate test; "stable
marriage" has ~0 prior → caught by over-detection; genuine rare CT terms
(operad 0.121, pretopos 0.0154, lextensive 0.0114) sit between the two junk
classes — the separation codex-2 worried latex-emph could not provide, the base
rate provides.

## Inline collocation-coherence gate (designed 2026-05-31, claude-2 + codex-2)

A SECOND use of the prior, distinct from step 2: run it **inline during the CT
superpod NER pass** as a guardrail against runaway junk like "stable marriage
problem" — no hand blocklist, and (critically) without rejecting genuinely novel
CT terms. This sidesteps the staleness problem (the gate runs during the fresh
run; it needs no posterior).

**Why collocation, not frequency (measured against `ct-term-prior.json`):**
"stable" alone is common and legitimate in CT (P=0.414); the junk is the *phrase*
"stable marriage" (bigram df=0). "marriage" is alien (P=0.0023) and is NOT among
"stable"'s 593 licensed completions, whereas `category` (df 728) and `homotopy`
(df 833) are. So the discriminator is the **bigram**, not the unigram.

**Retracted along the way (discipline):** (a) a scalar "hunger" metric
`sum(right-bigram-df)/unigram-df` — broken: `functor` scored highest (10.72) yet
is the most standalone term; it tracks frequency+grammar, not modifier-hunger.
(b) claude-2's claim that a literal membership gate spares novel terms —
**codex-2 falsified it**: `lextensive completion` has bigram df=0 AND `completion`
is not in `lextensive`'s 6 licensed completions, so a raw "absent ⇒ reject" gate
would kill the novel term. The gate MUST be conditional on head-prior.

**The gate (three-way; verified on real numbers 2026-05-31):** for candidate
bigram `A B`:
- **PASS** if `bigram_df[A B] ≥ 3` (corpus-licensed collocation).
- **ABSTAIN** (allow, defer to definition/source/posterior evidence) if
  `P(A) < 0.05` — low-prior head ⇒ corpus can't reliably know its completions ⇒
  **do not reject; this is the anti-novelty-kill switch.**
- **REJECT/QUARANTINE** only if ALL: `bigram_df[A B] < 3` AND `P(A) ≥ 0.05`
  (common head, completions well-known) AND `P(B) ≤ 0.01` (alien tail) AND no
  strong local definitional context (`defined-as`/`called-as`/`definition-of`
  should downgrade hard-reject to quarantine).

Verified verdicts (computed against `ct-term-prior.json`, n_docs=9742):

| candidate | verdict | evidence |
|---|---|---|
| `stable marriage` | REJECT/quarantine | head 0.414, tail 0.0023, bigram df 0 — the target |
| `cartesian marriage` | REJECT/quarantine | head 0.395, tail 0.0023, df 0 |
| `stable category` | PASS | bigram df 728 |
| `stable homotopy` | PASS | bigram df 833 |
| `abelian group` | PASS | bigram df 2105 |
| `lextensive completion` | ABSTAIN | low-prior head 0.0114 → novelty survives |

Threshold rationale (real numbers): `P(A)≥0.05` separates high-evidence CT heads
(stable .414, cartesian .395, abelian .496, functor .874) from emerging heads
(lextensive .0114, pretopos .0154); `P(B)≤0.01` catches alien tails (marriage
.0023, shapley .0006, blocking .0021) without touching common CT tails (category
.945, homotopy .527, completion .279).

**Distributional "hungry modifier" signal — DIAGNOSTIC ONLY, not a gate
(codex-2):** content-mass of the completion set distinguishes modifier-like
stable/cartesian/abelian (top-12 ~10-11/12 content nouns) from standalone
functor/morphism (~5/12; completions are `between`/`preserves`/`given`). But it
is NOT clean enough to decide validity — `left` scores high content-mass (0.93)
yet is a generic junk risk. Use it to flag candidates for extra scrutiny, never
to accept/reject alone.

**Status:** IMPLEMENTED, default-OFF (2026-05-31). `scripts/superpod-job.py`:
`_load_collocation_prior` + `_collocation_incoherent` + `_discovery_keep_multiword_term`
extended with optional `collocation_prior`, threaded through `run_stage5_ner_scopes`
to CLI `--discover-terms-collocation-prior PATH`. Gate runs AFTER the seed-known
bypass (never judges known vocab) and is a no-op unless the flag is passed —
verified: argparse default None → prior None → inert, so Rob's runs are unchanged
unless he opts in. Summary now reports `collocation_gate_enabled` +
`collocation_rejected_terms`. Tests: `tests/test_collocation_gate.py` 8/8 incl.
against the real 50MB prior.

**To enable:** add `--discover-terms-collocation-prior data/ct-term-prior.json`
to the Stage 5 invocation.

**One unverified assumption (validate on a live run, do NOT trust blind):** the
tests prove the GATE LOGIC, not that the upstream extractor hands it spans
containing the `stable→marriage` adjacency. The candidate span is the normalized
`\emph{}`-derived term (≤4 tokens, `term = " ".join(toks)` at superpod-job.py:676);
whether real "stable marriage problem" extractions arrive as that adjacency must
be confirmed against a run's `candidate-new-terms.jsonl` before the gate is trusted.

## Step 2 status: BLOCKED on a fresh NER run (Joe, 2026-05-31)

Step 2 (load the mark2 `ner-terms/*.json` posterior, compute posterior-vs-prior
divergence against `ct-term-prior.json`, flag the over-detected terms) is
**deliberately deferred.** The existing `ner-terms/*.json` was produced BEFORE
Joe shipped the Stage 5 improvements, so it is a **stale posterior**. Testing the
prior against it would measure the OLD extractor's over-detection — divergences
it surfaces may already be fixed, so we'd risk "finding" dead problems and drawing
false conclusions about the prior's value.

**Unblock condition:** a fresh, post-improvement CT NER run (Rob's next CT runner
pass is the natural source). When that posterior exists, step 2 is: for each
extracted term, compare its posterior assignment rate to `P(term|CT)` from
`ct-term-prior.json`; flag posterior ≫ prior (over-detection) and high-prior grabs
separately (§2 two-mechanism split). Do NOT run it against the current stale
ner-terms.

The step-1 prior (`ct-term-prior.json`) is the durable artifact and is ready to
receive that comparison whenever the better posterior lands.

## Relations

- M-bayesian-structure-learning — parent; this is a shippable instance, and the
  hierarchical per-category prior (§2b) is a concrete realisation of its
  "accumulate posteriors with partial pooling" goal.
- M-paper-reverse-morphogenesis / M-superpod-mark3 — the Stage 5 pipeline this
  prior plugs into.
- M-differentiable-code (futon5) — structural analog of the same prior (§4).
- External: Rob / superpod CT runner — the consumer; timing driver.
