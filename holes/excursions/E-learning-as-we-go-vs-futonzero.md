# Learning-as-we-go (mark7) ↔ FutonZero / AlphaZero — and the agent-in-the-loop

*futon6 note, 2026-06-23. Companion to `mark7-superpod-run-playbook.md`,
`holes/missions/M-metric-harness.md`, and futon2's `docs/futonzero-alphazero.md`.*

## 1. The relation

FutonZero plays the **development** game AlphaZero-style: policy (a cascade) + search (rollout)
+ reward (a peradam), over the futon stack's own structure. Its honest v1 caveat:
single-agent MDP, **not yet AlphaZero** — it lacks the **closed self-play loop** (R2:
search-result trains the prior) and an adversary; and the self-graded value `C` is "Goodhart's
door."

mark7's **"improve as we run"** is a different game on a different board — but it has, in
embryo, exactly the loop FutonZero v1 is missing, one level down (substrate mining, not
development). The board is the **corpus**; the "experience" is the accreted graphs / lexicons /
shapes; and the system **grounds itself against its own harvest** (the corpus's move-vocabulary
re-grounds its own proof-moves; concept-coverage rises with the corpus). That is a genuine
data-level flywheel — AlphaZero's "learn from self-generated experience, no human labels," at
the level of *mining math*, not *playing a game*.

## 2. The mapping (mark7 → AlphaZero)

| AlphaZero | mark7 "improve as we run" |
|---|---|
| self-play move | per-paper LLaMA extraction (S3 IATC, S4 expository, S7 box-typing) |
| accumulated experience | the accreted graphs + harvested lexicons + canonical shapes |
| value / reward signal | the **accretion metrics** — recognition rising with corpus, reground lift, coverage-gap closure, anchor-confidence, macro-entropy |
| the **policy** | the **recognizers + vocab + prompts** — macro-from-methods, the move-cues, the SFC binder grammar, the structure features |
| training step (policy update) | **← the gap.** Today this is *manual and once-per-session.* |

## 3. The crux: this session **was** the agent-in-the-loop — run once, by hand

Everything that makes mark7 better than the old runners came from a **review-and-improve
step on the accreted data**: macro-collapse → derive macro from methods; proof-move grounding
0.14 → harvest the corpus's own move-cues and reground; tight embedding → z-norm + bigrams;
SFC `:hole` gaps → teach the binders. That *is* the AlphaZero training step — *search/experience
improving the policy* — but performed **manually, once, by a human-driven agent** at the end of
a session.

**Joe's idea closes the loop:** put an **agent-in-the-loop (Codex, not LLaMA)** that, every
**100–500 papers**, reads the accretion metrics + the diagnostics and *improves the policy* —
grows a vocab, fixes a recognizer, widens a feature, patches an SFC gap — then the next batch is
processed by the improved policy. That is **R2 for mark7**: the experience trains the prior,
automatically and continuously, instead of once per human session. It is what would move mark7
from "AlphaTensor-shaped" (single pass, fixed policy) toward the **closed self-play loop**.

The infrastructure is already there: the diagnostics we built **are the review signals** — the
coverage-gap flags (`:hole` patterns naming what to teach SFC), the anchor-confidence
distribution (what's low-confidence), the reground lift, the accretion-curve plateaus (where a
tier stops improving = where the policy needs work). The agent acts on those, not on raw text.

## 4. The same two cautions FutonZero names — they bite here too

1. **Goodhart's door (self-graded reward).** The accretion metrics are computed by the system
   over its own output — an agent optimizing them can *game* them. We have direct evidence: the
   mark5 macro-collapse "passed structural checkers with uniform shells," and the standing rule
   is **gate substance, not checker-PASS** (verify against golden + distribution +
   source-faithfulness). So the agent-in-the-loop must **verify substance**, author≠reviewer
   (adversarial check), not "the number went up."
2. **The anchor must be exogenous.** AlphaZero's incorruptible reward is the win; FutonZero's is
   the peradam. mark7's exogenous anchors already exist and must drive the loop, not the
   self-graded metric: **source-faithfulness** (does the IATC anchor actually contain its
   claim?), **explicit-structure-recall** (did we recover the `prooftree`/`\judge` the author
   *wrote*? — ground truth, no self-grading), and ultimately the teleological one from
   `f6/graph-enhanced-evaluation`: **does the mined structure make downstream reasoning
   measurably better?** An agent-in-the-loop chasing the accretion metric *without* these is the
   "ceremony" failure FutonZero already measured and recorded (argument-across-worlds, falsified
   on the realized-`G(π)` floor).

## 5. The falsifiable test (in FutonZero's spirit)

Before believing the agent-in-the-loop is more than ceremony: **does reviewing-and-improving
every N papers beat a fixed-policy run at equal n?** Run mark7 twice over the same corpus
prefix — once policy-frozen, once with the Codex review every 100/500 papers — and compare the
**exogenously-anchored** metrics (explicit-structure-recall, source-faithfulness), not the
self-graded ones. If the loop doesn't beat frozen on the grounded yardstick, it's ceremony —
record it honestly, as FutonZero did with arguing-across-worlds.

## 6. Why "learning as we go" is the crucial part (Joe)

The old runners were throughput ("N papers done"). mark7 is a **flywheel**: data improves the
policy, the policy improves the data. The agent-in-the-loop is the bearing that lets the wheel
spin on its own — and it slots cleanly onto mark7 because the metrics + diagnostics are already
the review surface. The 20-hour-window constraint even helps: the natural review cadence
(every 100–500 papers) **is** a checkpoint, so the loop and the accretion-sweep checkpoints are
the same heartbeat.
