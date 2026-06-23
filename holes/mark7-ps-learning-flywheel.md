# PS to the mark7 runner email — the bigger picture

*A one-minute note on why mark7 is more than a run.*

mark7 isn't just "process math.CT" — it's the **first turn of a learning flywheel**, and the
reason it has more juice than the old throughput runners. The pipeline doesn't just extract;
it **grounds the corpus against its own harvest**: the inference/expository move-vocabulary it
mines gets fed back to recognize the next proofs, the definition-shapes canonicalize, and the
metrics are *accretion curves* (do they rise + converge as the corpus grows?) rather than
counts. That's the AlphaZero-style "learn from your own experience, no human labels" idea, but
applied to **mining math** instead of playing a game.

The piece that would close the loop: an **agent-in-the-loop** — a Codex-style coding agent
(not the LLaMA doing extraction) that, **every 100–500 papers**, reads the run's own diagnostics
(coverage gaps, low-confidence anchors, plateaued curves) and makes a **small, verified edit to
the recognizers/vocab/parsers**, so the *next* batch is processed by a better policy. That's the
"R2" return channel — search trains the prior — that FutonZero v1 is still missing. For math.CT
it's only ~46 passes, mostly light (and increasingly no-ops as the policy converges); the policy
surface is a handful of version-controlled files, so every pass is a reviewable diff. In effect
it automates, batch by batch, the manual "review-and-improve" work that produced mark7's fixes
in the first place.

Two honest guardrails (so it's an engine, not ceremony): each edit is verified against an
**exogenous anchor** — *did we recover the* `prooftree`/`\judge` *the author actually wrote* —
not the self-graded metric; and there's a falsifiable test (loop vs frozen-policy on that
grounded yardstick). Full write-up: `holes/excursions/E-learning-as-we-go-vs-futonzero.md`.

No action needed for the run itself — it already emits every signal such a loop would consume.
Just flagging where this is pointed, in case it's worth a slot of its own down the line.
