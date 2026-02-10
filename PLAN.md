# O-0: Classical Baseline — Chapter 6 at SE Scale

## What We're Building

Reproduce Corneli (2014) Chapter 6 learning-event detection on physics.SE
using the NER kernel. Classical statistics, no GPU, no LLM. The script
should run comfortably on a laptop.

## Data Available (all in `data/physics-se/`)

| File | Size | What we need from it |
|------|------|---------------------|
| PostHistory.xml | 1.8 GB | User text over time (PostHistoryTypeId 2=Initial Body, 5=Edit Body) |
| Posts.xml | 979 MB | Question→Answer mapping (ParentId), AcceptedAnswerId, OwnerUserId |
| Comments.xml | 365 MB | Treatment events: comments on user's posts |
| Users.xml | 82 MB | User metadata, account creation dates |

Also: `data/ner-kernel/terms.tsv` (19,236 terms) — the dictionary.

## The Method (from thesis Chapter 6 + flexiarg pattern)

### Definitions

- **Learning event**: User X uses NER kernel term T for the first time
  in any of their posts/edits. Timestamped by the PostHistory CreationDate.

- **Treatment event**: Something happens TO user X that might cause learning:
  - Someone answers X's question (Posts where ParentId = X's question)
  - Someone comments on X's post (Comments where PostId = X's post)
  - X's answer gets accepted (AcceptedAnswerId = X's answer)
  All timestamped.

- **Vocabulary trajectory**: For user X, the cumulative set of NER kernel
  terms they have used, plotted over time. The derivative of this curve
  is the learning-event rate.

- **Ornstein-Uhlenbeck impulse-damping model**: The learning-event rate
  r(t) follows: dr = -θ(r - μ)dt + σdW + α·δ(t - t_treatment)
  - θ = damping rate (how fast learning rate returns to baseline)
  - μ = baseline learning rate
  - σ = noise
  - α = impulse magnitude (the treatment effect we're measuring)

### Pipeline (5 stages)

**Stage 1: Parse PostHistory → per-user text timeline**
- Stream PostHistory.xml (1.8GB, constant memory)
- For each row with PostHistoryTypeId in {2, 5} and UserId present:
  extract (user_id, post_id, timestamp, text)
- Group by user_id, sort by timestamp
- Output: `user_timelines.json` — dict of user_id → [(timestamp, text), ...]

**Stage 2: Parse Posts → treatment event index**
- Stream Posts.xml
- Build question_owner map: {post_id → owner_user_id} for PostTypeId=1
- Build answer_to_question map: {answer_post_id → question_post_id} for PostTypeId=2
- From Posts: when an answer is created for a question, that's a treatment
  event for the question owner
- From Posts: AcceptedAnswerId → treatment event for the answerer
- Stream Comments.xml: comment on post_id → treatment for post owner
- Output: `treatment_events.json` — dict of user_id → [(timestamp, type), ...]

**Stage 3: NER term spotting per user**
- Load NER kernel (19,236 terms from terms.tsv)
- For each user's text timeline (chronological):
  - Spot terms in each text entry
  - Track cumulative vocabulary: which terms has this user ever used?
  - Mark learning events: term T first used at timestamp t
- Output: `learning_events.json` — dict of user_id → [(timestamp, term), ...]
- Output: `vocab_trajectories.json` — dict of user_id → [(timestamp, cumulative_term_count), ...]

**Stage 4: O-U model fitting**
- For each user with sufficient data (>= 20 learning events, >= 10 treatment events):
  - Construct learning-event rate time series (events per week)
  - Mark treatment event times
  - Fit O-U parameters: θ (damping), μ (baseline), α (impulse), σ (noise)
- Aggregate: population-level estimates with confidence intervals
- Output: `ou_fit.json` — per-user parameters + population summary

**Stage 5: Report**
- Compare with Corneli (2014) findings:
  - Original: 445 PlanetMath users, 9 years
  - Expected: thousands of physics.SE users, 14 years (2010-2024)
- Produce summary statistics and plots data
- Output: `o0_report.json` + stdout summary

## Implementation

Single Python script: `scripts/o0-classical-baseline.py`

Dependencies (all standard/pip):
- xml.etree.ElementTree (iterparse for streaming)
- numpy, scipy (O-U fitting via scipy.optimize)
- collections, json (standard lib)

No GPU. No LLM. No heavy deps. Should run on a laptop in under an hour
(PostHistory is the bottleneck at 1.8GB; streaming parse is single-pass).

### Memory Budget

- NER kernel: ~19K terms × ~50 bytes = ~1MB
- Per-user data: accumulate in dict. If 50K users × 200 terms avg = ~10M entries.
  Keep only (user_id, timestamp, term_first_used) triples = ~400MB worst case.
- Treatment events: similar scale.
- Total: should stay under 2GB RAM for the full dataset.

### Reuse

- NER kernel loading: adapt from spot-terms.bb's logic (tokenize + multi-word match)
- XML streaming: adapt from process-stackexchange.py's ElementTree iterparse pattern
- Term spotting algorithm: port spot-terms.bb's approach to Python

### Checkpointing

Each stage writes its output to `data/o0/`. If Stage 1 completes but Stage 3
crashes, you don't re-parse PostHistory. This matters on a laptop.

## Success Criteria (from M-f6-eval.md)

- [ ] NER kernel produces meaningful vocabulary trajectories on SE data
- [ ] Learning events are detectable (non-zero rate of new term adoption)
- [ ] O-U model fits with plausible parameters (positive impulse, finite damping)
- [ ] Results are comparable to Corneli (2014) PlanetMath findings

## Risk

The main risk is that physics.SE vocabulary is narrower than PlanetMath's
(physics vs pure math), so the 19K-term kernel (built from PlanetMath MSC
repos) may have lower coverage on physics content. The kernel does include
SE tags (filtered for frequency >= 5), which helps. Coverage was 100% on
entity-level spotting — the question is whether per-user trajectories show
enough variety to detect learning events.
