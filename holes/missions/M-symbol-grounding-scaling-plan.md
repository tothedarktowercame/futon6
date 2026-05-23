# Symbol grounding: scaling plan — preparation phase + expected value at 5K / 50K / 500K papers

**Date:** 2026-05-22 (rev 2026-05-23 with explicit preparation phase
+ Wikipedia gold source)
**Status:** DERIVE — preparation gates must pass before sending Rob's batch
**Owner:** Joe (decides) / claude-7 (drafted)
**Predecessor:** [M-symbol-grounding.md](M-symbol-grounding.md)

## 0. Why this rewrite

First-draft (2026-05-22) treated "tune PM precision and ship to Rob"
as a brief paragraph in §5. Joe (2026-05-23) pushed back: we have
shipped underperforming pipelines twice before, it is embarrassing,
and shipping a half-baked 13% precision engine to a 5000-paper run
risks repeating the pattern. He also pointed at a Zenodo
Wikipedia-math dump (record 15107679, ~1.6 GB tar of multi-lingual
MediaWiki math subsets at ~/Downloads/math.tar) as an independent
gold source.

This revision turns §5 into an explicit **preparation phase with
named gates**. Each gate has a measurable pass/fail criterion. Rob's
batch is not requested until all gates pass.

## 1. What we have measured so far

On the PlanetMath labeled-gold corpus (409 entries, 457 (symbol, canon) gold pairs):

| metric | strict match | loose (substring) match |
|---|---:|---:|
| overall recall | 14.4% | 26.7% |
| overall precision | 6.1% | 12.8% |
| overall F1 | 8.5% | 17.3% |

Per-strategy precision (loose), sorted by their emit volume on gold symbols:

| strategy | TP / emits-on-gold | precision |
|---|---|---:|
| `inline-is-a` | 77 / 512 | 15.0% |
| `kernel-ambient` | 59 / 493 | 12.0% |
| `the-Y-X` | 10 / 255 | 3.9% |
| `let-binding` | 32 / 123 | 26.0% |
| `section-context` | 1 / 36 | 2.8% |
| `denotation` | 5 / 16 | 31.2% |

Mission §5 success criterion is **≥50% precision on a 30-paper sample**.
We are not there. We have not seen scaling alone close that gap.

## 2. What scales with N papers, what doesn't

### Scales linearly (volume only)
- Grounded-symbol marks emitted (~3000/paper × N)
- Constructor-declaration scopes (~50/paper × N)
- Per-paper output files

These add disk + processing cost, not insight.

### Stabilises (variance decreases with √N)
- Per-strategy defeat-rate and corroboration-rate aggregates
- Strategy ranking on the trust signal

By N=200 the meta-learning rates are within ~5pp of their long-run
values. We already have these from the 4-paper demo, refined to a
honest range from the 41-paper First Proof run.

### Compounds super-linearly
- **Cross-paper vocab `common` list** — (symbol, body) pairs that
  recur in ≥2 papers. Saturates slowly. Rough estimates from
  CTAN/preamble analyses elsewhere:
  - N=5K → ~500–2000 common pairs
  - N=50K → ~5K–15K
  - N=500K → ~30K–80K
- **Bootstrap pseudo-gold** — where ≥5 papers independently agree
  on (sym, canon), accept as gold. Empty at N=5K; only meaningful
  at N=50K+.

### Does NOT improve from scale
- **Per-paper grounding precision.** Running the same engine on more
  papers does not raise the per-paper precision number. PM eval will
  report the same 12.8%/26.7% on the 500,001st paper as on the first.
- **Engine recall ceiling.** Pattern-based strategies catch what they
  catch. Scale changes nothing about that.

## 3. What we expect at each scale

### N = 5,000 papers (current Stage 5 default)

**What we'd get:**
- A `learned-newcommand-vocab.json` with ~1K common (sym, body) pairs.
  Useful as a prior for future runs.
- Cross-paper meta-learning table with stable defeat/corroboration
  rates per strategy. Confirms or refutes the rankings PM eval
  produced.
- Full pipeline shakedown (timing, OOM safety, side-file integrity)
  on a non-trivial corpus.
- A baseline number per paper that future runs can compare against.

**What we would NOT get:**
- Any per-paper precision improvement. Each paper still grounds at
  ~13% loose precision.
- Enough cross-paper agreement to bootstrap pseudo-gold.
- A defensible "≥50% precision" claim.

**Cost:** Rob's setup time + ~30–60 GPU-hours (estimate from
existing superpod-job benchmarks).

**Honest case for running:** infrastructure validation and a
baseline. Modest.

### N = 50,000 papers

**What we'd get:**
- Vocab common list of ~5–15K (sym, body) pairs. Becomes a real
  "shared math vocabulary" usable as a paper-wide prior on fresh
  runs.
- Meta-learning stable enough to drive *gating decisions*: e.g.
  turn off strategies whose defeat rate exceeds 70% in production.
- Beginnings of bootstrap pseudo-gold: pairs where ≥5 papers
  independently emit the same (sym, canon) binding. Could be ~500
  pairs at this scale.
- Domain stratification — per-subdomain (math.CT, math.AG, hep-th,
  cs.LG) meta-learning rates.

**What we would NOT get:**
- Per-paper precision improvement still doesn't happen automatically.
  But the bootstrap pseudo-gold lets us *measure* per-domain
  precision honestly for the first time.

**Cost:** Rob's setup + ~300–600 GPU-hours. Possibly a week of his
clock time.

**Honest case for running:** the bootstrap pseudo-gold is a real
asset, but only useful if we then re-tune and re-run. So this is a
two-round-trip commitment.

### N = 500,000 (whole arXiv math + adjacent)

**What we'd get:**
- Comprehensive shared math vocabulary across all of arXiv.
- Bootstrap pseudo-gold large enough (maybe ~30K pairs) to be the
  primary eval corpus going forward.
- Per-subdomain strategy rankings and learned defaults.
- Infrastructure for a future researcher-facing query: "what does
  symbol X mean in subdomain Y?"

**What we would NOT get from this alone:**
- Mission §5 success. The engine's pattern strategies, even with
  scale-data tuning, will plateau well short of 50% per-paper
  precision without architectural changes (richer parsing, LLM-
  augmented strategy, etc.).

**Cost:** Rob's setup + ~3K–6K GPU-hours. Multiple weeks. Real money.

**Honest case for running:** if and only if we treat the engine as
*foundational infrastructure* for a longer-horizon research agenda
(your "25-year framing"), not as a pre-deployment validation. The
output is a corpus tool, not a precision number.

## 4. The hard question: precision plateau

The current engine is **pattern-based**: regex on prose declaration
shapes, plus a NER-kernel lookup. Pattern-based grounding has a
known ceiling — Joe's UKRN-S working paper context, Codex's
infrastructure work, and the AIQA literature all peg this at
~20–35% precision on math prose, depending on domain.

To get past that, the engine needs ONE of:
- **A.** Richer parsing (deep LaTeX AST, type inference) — months of work.
- **B.** LLM-augmented strategy (Claude calls inline) — ongoing cost
  per paper; bias concerns Joe raised earlier in the session.
- **C.** Iterative tuning against bootstrap gold (the N=50K loop) —
  could plausibly reach 35–45% with several rounds.

None of these are "ship the current engine to Rob and watch precision
improve." Scale gives us tuning DATA, not engine improvement.

## 5. Preparation phase — explicit gates

Each gate has a pass/fail criterion AND a fallback decision if it
doesn't pass. Gates run in order; later gates depend on earlier
ones. Rob is not asked until every gate passes.

### Gate P1 — Wikipedia gold extractor (2–3 days)

**What:** Build a second gold extractor that handles MediaWiki XML
dumps. Wikipedia uses `<math>X</math>` for math and `[[concept]]` or
`[[concept|display]]` for wiki-links, both different from PM's
`$X$` + `\PMlinkname`. Source: `~/Downloads/math.tar` (Zenodo
record 15107679, ~1.6 GB tar of multi-lingual MediaWiki math subsets).

Extractor outputs the same JSON shape `build-grounding-gold.py`
already emits, with `source: "wikipedia.en"` (or whichever language).

**Pass criterion:** Wikipedia gold JSON written with ≥1000 (symbol,
canon) pairs across ≥500 articles. Hand-sample 30 of those to
verify the pairs are well-formed (no obvious noise from
non-mathematical contexts).

**Fallback if fail:** investigate the markup actually used in
Wikipedia math articles — fall back to PM-only if Wikipedia's
inline-link density is too low or the patterns don't generalise.
Document why and continue with PM-only at later gates.

**Estimated effort:** 2–3 days.

### Gate P2 — Combined gold ≥ 1500 pairs across ≥ 800 entries

**What:** Combine PM gold (currently 469 pairs across 409 entries)
with Wikipedia gold. Re-run the eval; report per-source breakdown
so PM vs Wikipedia precision can be compared.

**Pass criterion:** total ≥1500 gold pairs covering ≥800 distinct
entries; per-source eval shows the engine doesn't catastrophically
fail on one source vs. the other (e.g. >20pp precision delta).

**Fallback if fail:** if PM and Wikipedia diverge sharply, the
engine is overfitting to one corpus. Investigate which strategies
are corpus-specific and either fix or document the bias before
proceeding.

**Estimated effort:** 1 day.

### Gate P3 — Strategy gating + canon-ancestry comparison (1–2 days)

**What:** Two tightening changes:
- **Strategy gating**: a CLI flag `--gate-strategies` that suppresses
  strategies below a precision threshold on the combined gold. By
  default, suppresses any strategy under 10% precision. Predicted
  effect on PM gold (loose): `the-Y-X` (3.9%) and `section-context`
  (2.8%) get gated, `kernel-ambient` (12.0%) borderline.
- **Canon-ancestry comparison**: use the kernel's hierarchical
  data (PM's `\pmrelated` lists; Wikipedia's category tree; nLab's
  `[[parent]]` links) so that "Group" and "TopologicalGroup" count
  as matching when both refer to the same ancestor concept.

**Pass criterion:** combined-gold precision (loose match) rises
from current 12.8% to ≥25%. If it doesn't, the strategies are not
just "noisy" — they are wrong, and gating won't save them.

**Fallback if fail:** investigate which strategies are
fundamentally mis-aligned with the gold. May need architectural
work (richer parsing, LLM-augmented strategy) before any scale
operation makes sense.

**Estimated effort:** 1–2 days.

### Gate P4 — Reach a defensible precision number (target: ≥30%)

**What:** Final eval after P3 tuning. Report by source, by
strategy, by sub-domain (math.CT / math.AG / cs.LG / etc. for the
arXiv portion).

**Pass criterion:** ≥30% loose precision across the combined gold,
with no individual high-volume strategy below 15%. Recall should
not drop more than 5pp from the pre-P3 baseline (the tightening
should improve precision without crashing recall).

**Fallback if fail:** if we land at 25–30%, send to Rob with
explicit "this is a baseline + vocab harvest, not a production
quality run" framing. If we land below 25%, do not send to Rob;
escalate to architectural work.

**Estimated effort:** 0.5 day (eval + report; the work is in P3).

### Gate P5 — Production shakedown on a 100-paper subsample

**What:** Run the full grounding pipeline (including
`learned-newcommand-vocab` aggregation) on a 100-paper sample of
arXiv batch-008. Verify:
- No OOM, no crashes, no malformed JSON output
- Loss-snapshot lines fire on schedule
- Manifest's `stage5_stats` carries the expected fields
- The `learned-newcommand-vocab.json` shape matches what the
  next-batch-run's `LearnedVocabStrategy` would consume

**Pass criterion:** clean run; output JSON loads in the eval
script without errors; vocab side-file structurally valid.

**Fallback if fail:** debug the breakage; do not send to Rob until
the 100-paper run is clean.

**Estimated effort:** 0.5 day.

### Gate P6 — Pre-send dry run, with Joe's explicit OK

**What:** Hand the precision report + 100-paper output + the
expected scale-economics from §3 to Joe. He decides whether the
case is strong enough to ask Rob to commit GPU-hours. This is the
"Joe-as-operator" consent step, not a technical gate.

**Pass criterion:** Joe says yes.

**Fallback if fail:** by definition, we don't send.

**Estimated effort:** 0 days from claude-7; bounded by Joe's
reading time + any clarifications needed.

### Total preparation time

5–7 days of focused work, assuming P3 lands in the 25–30% range
on the first attempt. Budget could double if P3 plateaus and
needs a second tuning pass.

## 6. The honest "should we ship to Rob now" answer

**No.** Not yet.

The 13% loose precision number is below the threshold where running
at scale buys us anything we couldn't get from a 1-day tuning pass.
We risk burning Rob's time on infrastructure validation that doesn't
need 5000 papers to validate.

The 1–2 days of tuning in §5.1 should be done before any Rob ask.
Then we either have a defensible number (ship), or we know we need
deeper architectural work (don't ship, redirect).

## 7. What this plan is NOT

- Not a claim that the engine is bad. Pattern-based grounding at
  13% loose / 26% recall is a real signal — it just isn't what
  Mission §5 promised.
- Not a recommendation to abandon. The infrastructure (defeasible
  strategy library, meta-learning loop, gold extractor, eval) is
  the durable contribution; scaling is a deployment decision.
- Not a precision claim about Rob's future batch. The engine will
  produce the same 13% per-paper number on Rob's 5K as it does on
  PM, *unless* we tune first.

## 8. Decision asked of Joe

Two decisions, in sequence:

**Decision 1 (now):** Does the preparation phase (§5) match what
you want?
- (a) Yes, run the gates P1–P6 as written. Send to Rob only on P6 ok.
- (b) Some gates are wrong / missing / wrongly sequenced — push
  back specifically.
- (c) A completely different path I haven't surfaced.

**Decision 2 (when P6 fires):** Looking at the precision report
plus the expected-economics from §3, do we commit Rob's time?
- (a) Yes, send the 5K batch with explicit framing about what we
  expect to learn from it.
- (b) Yes but at a different scale (sample first, e.g. 500 papers,
  to confirm the per-paper output looks reasonable before
  committing 5K).
- (c) No — the engine is foundational, but not ready for
  publication-quality output. Redirect effort to architectural
  work.

My recommended path: Decision 1 = (a), commit to the gates.
Decision 2 cannot be made until P6 fires.

## 9. Status update on previous "embarrassing" pipelines

Joe noted (2026-05-23) that we have shipped under-performing
pipelines twice before, leading to embarrassing rework cycles. The
plan above is structured to avoid a third such incident by
front-loading the precision validation. Specifically:

- No "we'll learn it as it runs" framing — we know what we're
  measuring before Rob commits.
- The fallback decisions on each gate are pre-specified, so a
  failed gate doesn't trigger another round of "let's just see
  what happens at scale."
- Joe's explicit OK in P6 is the consent gate (see [[operator
  not Sovereign]] in MEMORY.md: Joe-as-operator means his
  decisions are signals plus a final yes/no, not optional).
