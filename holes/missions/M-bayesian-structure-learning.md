# Mission: Bayesian structure learning — formalising what we mean by it

**Date:** 2026-05-23
**Status:** IDENTIFY → DERIVE — proposal stage; Joe to push back on framing
**Owner:** Joe (frames it) / claude-7 (drafted)
**Predecessor:** [M-symbol-grounding.md](M-symbol-grounding.md),
[M-symbol-grounding-scaling-plan.md](M-symbol-grounding-scaling-plan.md)
**Source pointer:** Friston, Lin, Frith, Pezzulo, Hobson, Ondobaka
(2017). *Active Inference, Curiosity and Insight.* Neural Computation
29(11):2633–2683. (`~/Downloads/Friston_Active Inference Curiosity and
Insight.pdf`)

## 1. Why this mission exists

We have shipped two iterations of "structure learning":
- Round 1: pattern-mining (`detect_learned_patterns`) — heuristic regex
  extraction with coverage-lift gates.
- Round 2: symbol-grounding strategy library — defeasible bindings,
  per-strategy meta-learning rates (emit / defeat / corroborate).

Both are **heuristic**. We accumulate counters; we don't accumulate
posteriors. The P3 plateau at 18% precision exposes the limitation:
we can't say *how much more data we need* to be confident in any
precision claim, and we can't say *which papers to prioritise* for
maximum learning. The system is structure-mining without
model-of-the-miner.

Joe's framing (2026-05-23, verbatim): *"What about improving and
formalising what we mean by 'structure learning' so that we include
a proper Bayesian approach? That way we'd both be learning the
structure of mathematical writing, and learning the model for
learning that structure."* That is the dual problem Friston et al.
treat under active inference: **infer hidden states + learn the
generative model that says how those states give rise to
observations**.

## 2. The Friston framing, applied here

The 2017 paper's abstract names the move:

> This article offers a formal account of curiosity and insight in
> terms of active (Bayesian) inference. It deals with the dual
> problem of inferring states of the world and learning its
> statistical structure. […] We then move from epistemic learning to
> model selection or structure learning to show how abductive
> processes emerge when agents test plausible hypotheses about
> symmetries (i.e., invariances or rules) in their generative
> models. The ensuing Bayesian model reduction evinces mechanisms
> associated with sleep and has all the hallmarks of "aha" moments.

Map to our setting:

| Friston construct | Our analogue |
|---|---|
| Hidden states of the world | True `(symbol → canon)` mapping per paper |
| Sensations (observations) | The text the engine reads; the bindings strategies emit |
| Generative model | Strategy library + per-strategy reliability + the kernel |
| Inference (state estimation) | Per-binding posterior `P(canon | symbol, evidence)` |
| Learning (parameter estimation) | Updating per-strategy reliability priors from observed agreement / defeat / gold matches |
| Structure learning / model reduction | Periodically asking "could two strategies be merged?" or "is there a latent strategy that explains agreement patterns?" |
| Variational free energy | Combined: prediction error + model complexity |
| Epistemic value (curiosity) | Information gain expected from processing the next paper |
| Insight (eureka) | A change in the generative model that suddenly reduces free energy on past data — the "aha" of seeing a pattern we were generating noise from |

## 3. What replaces the current heuristic loop

### 3.1 Per-strategy reliability as a posterior

Currently we keep counters: `emitted`, `defeated`, `corroborated`,
`solo`. These are derived statistics, not posteriors.

Replace with: each strategy s has a reliability distribution. The
simplest workable model is `s_reliability ~ Beta(α_s, β_s)`, where
the mean `α_s/(α_s+β_s)` is the strategy's expected precision and
the variance encodes our uncertainty about it.

Updates (each time we observe a binding by s):
- Binding turns out correct (matches gold OR is corroborated by
  another high-reliability strategy): `α_s += 1`.
- Binding turns out wrong (defeated by a high-reliability strategy
  emitting a different canon at the same scope): `β_s += 1`.
- Binding ambiguous (no signal): no update.

Starting prior: `Beta(1, 1)` (uniform) for new strategies. Strategies
inherit their prior from their family if related (e.g. all
declarative strategies start at `Beta(3, 1)` because the prior
domain-knowledge says these are usually right).

After N observations the posterior tightens. We can report:
> Strategy `let-binding` has reliability posterior `Beta(95, 230)`,
> mean precision 29.2% with 95% credible interval [24.4%, 34.4%].

This is what was missing from §3 of the scaling plan.

### 3.2 Per-binding posterior

Each binding becomes a probability distribution over candidate
canons, not a single point estimate. For symbol X in paper p, after
observing strategies' outputs:

```
P(canon = c | x, p) ∝ P(canon = c) × ∏_s P(strategy_s vote | canon = c, reliability_s)
```

The kernel-known canons get a non-zero prior; novel canons get a
small "open-vocabulary" baseline. The posterior is computed
analytically because reliability priors are Beta and votes are
binary.

The viewer surfaces this as a top-1 canon with a confidence band, not
as a single canon with a strategy attribution.

### 3.3 Expected information gain — the scale decision

This is the part that addresses Joe's "is sending to Rob worth it"
question with a real number.

For any candidate batch B (e.g. Rob's 5000 papers), compute:

> Expected information gain = E[H(reliability_priors) - H(reliability_posteriors_after_B)]

Where H is entropy. This is the **epistemic value** of processing B,
in nats. Roughly:

- Strategies whose Beta posteriors are already tight (lots of
  evidence) won't update much → low expected info gain per binding.
- Strategies that are still uncertain → big expected info gain per
  binding, but only if B contains content those strategies fire on.

The decision rule becomes:
> Send B to Rob iff `expected_info_gain(B) > cost(B) × value_per_nat`.

For Rob's 5K batch: we estimate it would update the `inline-is-a`
posterior (current widest CI) and the `learned-vocab` posterior
(empty without scale data) significantly; it would barely move
`denotation` (already tight). The CI tightening is the deliverable.

### 3.4 Structure learning — Bayesian model reduction

Periodically (e.g. after each batch), ask: **does a simpler
generative model fit the data equally well?** Concretely:

- Can two strategies be merged because their reliability posteriors
  are identical AND their bindings overlap >90%? (Merge.)
- Is there a latent variable that explains why kernel-ambient and
  section-context co-fire? (Replace with one strategy that
  captures the latent.)
- Should a strategy be split because its reliability is bimodal
  (high on one subdomain, low on another)? (Split by subdomain.)

This is the "sleep / consolidation" phase Friston compares to. It
happens offline, between scale runs. It produces an "aha" — a
simpler model that explains the data we have.

## 4. What the implementation would look like

### 4.1 New data structures

```python
@dataclass
class StrategyReliability:
    strategy_name: str
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean_precision(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, level=0.95) -> tuple[float, float]:
        # via scipy.stats.beta.ppf
        ...

    def update(self, correct: bool, weight: float = 1.0) -> None:
        if correct:
            self.alpha += weight
        else:
            self.beta += weight


@dataclass
class CanonPosterior:
    symbol: str
    candidates: dict[str, float]  # canon -> probability mass
    null_mass: float  # "no canon" probability
```

### 4.2 New code paths

- `src/futon6/bayesian_grounding.py` — the posterior layer.
- `scripts/eval-grounding-bayesian.py` — replaces the count-based
  eval; reports per-strategy reliability posteriors and per-binding
  confidence bands.
- `scripts/structure-reduction.py` — the offline model-reduction
  step. Takes the current strategy library + accumulated bindings,
  proposes merges/splits, validates each on held-out data.

### 4.3 What's preserved

The strategy library (`symbol_grounding.py`) doesn't go away. The
pattern strategies remain the OBSERVERS. The new layer treats them
as evidence sources to be combined via Bayes.

The defeasibility cascade (`merge_bindings`) becomes a special case
of the posterior calculation — defeat is a hard `P(canon | newer
evidence) = 0` for the older binding within the narrowed range.

The meta-learning counters become summary statistics derived from the
underlying Beta posteriors, not the source of truth.

## 5. How this addresses the scaling plan

The scaling plan §3 said "per-paper precision doesn't improve from
scale." Under the Bayesian framing:
- Per-paper precision STILL doesn't improve from scale alone, but
- Our CONFIDENCE in our precision estimate improves with √N
- We can predict the value of N additional papers BEFORE running on
  them
- Structure-reduction at the meta-level can change the engine,
  improving per-paper precision in the NEXT round

This converts Joe's "no scale guarantee" worry from a binary into a
quantified expectation:

> After 5K papers, posterior precision is X ± Y. After 50K, X' ± Y'.
> Structure-reduction between rounds adds Z to per-paper precision
> with some probability.

We can then commit Rob's time on the basis that the resulting
posterior tightening is worth more than the GPU-hours cost.

## 6. What this is NOT

- Not a claim that this will reach 50% precision. The architectural
  ceiling of pattern-strategies still exists; Bayesian integration
  doesn't invent recall.
- Not a replacement for the gold extractors / eval harness. They
  remain the supervised signal that updates posteriors.
- Not an LLM strategy. Friston-style active inference is classical
  Bayes; if we want LLM-augmented strategies they can be added as
  additional strategies in the library, with their own reliability
  posteriors.
- Not a one-week project. Realistically 3–6 weeks of careful work:
  the math is well-understood, but the engineering and the
  validation against held-out gold are real time.

## 7. Decision asked of Joe

(a) Is the framing in §2 / §3 the right reformulation, or would you
    sequence it differently?
(b) If yes, this becomes the path forward INSTEAD of P4–P6 in the
    scaling plan. The scaling plan's gates were "tune the heuristic
    engine to 25% then ship to Rob." This mission replaces that with
    "build the Bayesian layer first, then the send-to-Rob decision
    becomes a calculation, not a judgment call."
(c) If yes to both, what's the right OWNERSHIP arrangement? This
    is a 3–6 week project that touches the maths, the eval, and the
    engine. Probably needs paired work — not me sprinting solo.

I recommend (a) "yes, this is the right reformulation," (b)
"redirect P4-P6 to this mission," (c) "pair with someone who has
read more of the active-inference literature than I have." This
isn't a shipping decision; it's a redirection decision.

## 8. What I'd do tomorrow if you green-light this

The first artifact would be a **side-by-side eval**: run the current
heuristic engine AND a minimal Bayesian-posterior implementation
on the existing PM + Wikipedia gold. Report side-by-side numbers,
plus the FIRST POSTERIORS — Beta(α, β) for each strategy. That
gives us a baseline to iterate from, and Joe a number to push back
on before more code lands.

That's 2–3 days. Everything else in §4 is on top of that, and
sequenced based on what the side-by-side reveals.

## 9. Topic-aware priors (2026-05-23 follow-on)

Joe's framing: garbage like "stable → StableMarriageProblem"
survives even after the kernel cleanup because some PM entries are
articles whose primary MSC sits outside the paper's topic. Two
independent prior signals slot into `combine_strategy_votes` as
`context_factors`:

  - **MSC topic prior** — `P(canon | MSC primary)` from PM's
    `:msc-codes` per entry. arxiv categories resolve to MSC
    primaries via `ARXIV_TO_MSC_PRIMARY` (math.CT→18, math.QA→
    16/17/18/81, etc.). Down-weights canons whose mass sits in
    unrelated MSC.
  - **SE corpus-frequency prior** — marginal `P(canon)` over
    math.SE + MathOverflow Q/A bodies. Cuts off canons that
    essentially never appear in real math discourse.

Both factors are NOT renormalised over candidates inside arbitration
— single-candidate symbols (where domain-mismatched garbage fires
alone) need to be able to lose to the null hypothesis when context
support is uniformly low. The composed prior leaves them holding
absolute mass that the constant null_prior can beat.

`MSCTopicPrior.update_from_run(emissions, msc_primaries,
min_confidence)` is the online-EM hook for arxiv-scale runs: each
paper's accepted bindings above `--confidence-threshold` fold back
into the prior, gated to avoid degenerate collapse.

First evidence (30 arxiv math.CT papers, nLab held-out vocab):
  - baseline + clean kernel: 47.9% emissions in nLab vocab
  - + MSC topic prior: 52.7% (+4.8pp)
  - + MSC + SE corpus prior: **61.1%** (+13.2pp over baseline)
  - high-confidence (p≥0.5) at the +13.2pp point: 54.9% (was 42.6%)

Total emissions also dropped 4978 → 3510 (down 30%) — the priors push
spurious annotations into the null mass rather than silently passing
them through. Lower volume + higher quality is the correct direction
for this kind of corrective signal.

SE prior uses log-scaled `(log(n+1)+1)/(log(max+1)+1)` rather than
linear `n/max` because the raw counts span 5 orders of magnitude
(top single-letter "canons" hit 94k vs longtail at 1). Linear scaling
flattened all real concepts to near-zero. Also filtered single-token
phrases shorter than 3 chars at index-build time, so "A", "C", "Pi"
etc. don't enter the count race.

Pattern (C) noise (`morphism → StructureHomomorphism`,
`pushout → CategoricalPullback`) still survives — same shape Joe
flagged in the audit. These canons exist in PM under MSC primaries
that overlap math.CT (16/17/18 ring-algebra-category border) so the
MSC prior alone can't disambiguate. Deferred until canon
fingerprint store + cross-domain signal is rich enough for semantic
mismatch detection.
