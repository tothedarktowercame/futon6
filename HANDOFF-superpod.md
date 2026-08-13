# Superpod handoff — math.CT structural mining

**For:** Rob · **From:** Joe / claude-4 · **Date:** 2026-08-10
**Attached:** `capability-proof-arxiv.pdf` — the local workup, 19pp, including a
hazard ledger of everything that broke while getting here.

---

## The short version

Start the run. It will refuse to start if the host isn't ready, and it will tell
you why. There is **no override flag on either gate** — that is deliberate, and
§3 explains what it costs to have one.

```bash
cd ~/code/futon6
export FUTON6_EPRINTS=/path/to/arxiv-math-ct-eprints    # source .tar.gz per paper
export OPENAI_BASE_URL=http://<serving-host>/v1
export MODEL=<served-model-name>

.venv/bin/python scripts/linode_stepper.py --plan --profile superpod   # read first
.venv/bin/python scripts/linode_stepper.py --run  --profile superpod \
    --run-id <pick-one> --corpus-id math-ct
```

`--run-id` **defaults to `adhoc`** and everything is scoped by it. Pick a real
one; two runs sharing an id share directories, which is how one corpus's outputs
previously landed in another's counts.

`--plan` prints the 13 stages and their commands and runs nothing. Worth reading
once — there are **11 halt points**, and the run stops at each by design. Use
`--no-halt` only when you want it to run through unattended.

---

## 1. The first two minutes: your abort point

Two mandatory gates run before any stage:

| gate | question | on failure |
|---|---|---|
| `preflight.py` | is everything the run needs **present**? | refuses to start, names each missing thing and its remedy |
| `conformance.py` | does this host **behave** as the pipeline assumes? | refuses to start |

**The second one is the one to watch, and it exists because of you** — because
your serving stack is not the one we developed against.

Every LLM stage depends on the endpoint honouring
`response_format: {"type": "json_schema"}`. llama.cpp binds it. A stack that
accepts the field and *ignores* it raises no error anywhere: the model answers
with its own key names, every lookup misses, each stage falls back to a
deterministic template, the gates pass, and the window produces a stub wearing
the model's voice. We shipped exactly that failure locally and it took four
passes to see.

So conformance asks the model what colour a ripe banana is, under a schema whose
enum is `["purple", "octagonal"]`. **"Purple" is the passing answer** — it proves
the grammar overrode the model's own knowledge. "Yellow" means the field was
accepted and ignored, and the LLM stages will silently produce nothing real.

It also measures **decode throughput** and reports what the cascade would cost at
that rate. On our host: 4.9 tok/s → ~5.6 h for the 818-call Tier-1 cascade. If
your number makes the window arithmetic impossible, that is worth knowing in
minutes rather than at hour six.

Run it standalone any time:

```bash
.venv/bin/python scripts/conformance.py \
    --endpoint $OPENAI_BASE_URL --model $MODEL --json conformance.json
```

Three of its six checks are **negative** — they assert that something correctly
*fails*. This project has shipped three checks that could not fail, so "the gate
passed" carries no information unless the gate has been seen to refuse something.

---

## 2. What normal looks like

**A stage that fails exits non-zero.** It did not always; a scheduler could
previously record a successful window over a run that stopped hours earlier.

**One measurement is red on purpose.** S3 runs anchor-faithfulness and writes
`data/runs/$RUN_ID/anchor-faithfulness.txt`. It **exits non-zero by design**
while H38 is open, and it is wrapped so it cannot fail the stage. If you see

```
anchor-faithfulness: N graph(s), ... -- FAIL
  frame-mismatch SUSPECTED in M/N graph(s) (every node matched 0 key terms)
```

that is expected and is not your problem. The checker and the graphs currently
use different line bases; the anchors are probably fine and the *checker* cannot
read them. Locally: 64.3% pass raw, **86.3% excluding the frame-mismatch
graphs**, and the repair is scoped but not done.

**S3 also emits `retry-rate-$RUN_ID.json`** — the fraction of accepted graphs
that needed more than one attempt. This is the honesty bound on first-pass model
quality and we would very much like it back. We cannot derive ours: the attempt
directory was not run-scoped, so the figure the paper once quoted is gone. It is
now measured in-loop, so your run produces it for free.

---

## 2b. These gates were demonstrated on three hosts, not just written

A gate that has only ever run on the authors' machine is an assurance, not
evidence. So both were run on **three hosts of deliberately different readiness**,
all on the same gate code:

| host | what it is | `preflight` | `conformance` |
|---|---|---|---|
| **linode-chicago** | clean checkout, no venv, py3.8, 2 CPU / 3 GB | **3/11 — DO NOT START** | **3/6 — ABORT** |
| **zone** | prepared, 32 CPU / 249 GB / 1.4 TB free | **10/11** | **3/6 — ABORT** |
| **dev box** | prepared, with a live endpoint | **11/11 — GO** | **6/6 — CONFORMS** |

The one thing separating `zone` from `GO` is `model:endpoint` — no LLM is served
there. That is the same check that will be the *first* thing your host answers,
and it is the reason the middle row is worth showing: a fully-provisioned machine
with 1.4 TB free and every substrate file present still does not get a GO without
a model behind it.

Read the rows as a discrimination test. On the unprepared host every failure was
specific and carried a remedy — `latexmlmath` absent, `edn_format` and
`sentence_transformers` missing, 1/6 substrate files,
`math-informal=0, math-informal-CT=0`, no eprint store, 27 GB against a 50 GB
floor. On the prepared ones, those same checks pass and say what they found:
`39` and `6` patterns, `25/25` sampled eprint ids resolving. Zone's
`chain:gate-subprocess` even reports *"R2d reports through the composer"* rather
than the dev box's *"no graphs yet — nothing to exercise"*, because Zone has real
graphs and the check ran against them.

**The exercise found and fixed a real defect, which is the point of doing it.**
Run standalone on the clean host, `conformance` first died with a nine-frame
`ModuleNotFoundError: edn_format` traceback, and then — after a first, inadequate
fix — could not evaluate two of its six checks at all. `conformance` is otherwise
stdlib-only, but it imports the stepper to read S3's declared gate, and the
stepper parsed its DAG contract from EDN *at import time*. So the gate whose job
is inspecting unprepared hosts was blind on exactly those hosts. The stepper is
now importable without `edn_format` (dependency access raises at point of use
instead, naming the remedy), and those two checks now return real verdicts on a
machine with no venv: `gate:refuses-bad-input` and `run:artifact-paths-scoped`
both pass on linode-chicago. That is the 1/6 → 3/6 improvement in the table.

Four passes of local verification had not surfaced this. One foreign-host run
did.

**`conformance` has also passed against a serving stack that is not llama.cpp.**
The 6/6 was measured against an Ollama endpoint serving `qwen2.5-coder`, which
binds `response_format: json_schema` correctly — the banana check returned
"purple". So the schema check discriminates between stacks rather than merely
agreeing with the one it was written on. Its throughput reading there was
5.5 tok/s → ~5.0 h for the 818-call cascade, the same order as our 4.9 tok/s.

**What this does not prove.** None of the three ran the pipeline. Two of them
could not — linode-chicago has 2 CPUs and no model, Zone has no model served.
The gates are demonstrated; the run is not. Section 6 stands.

---

## 3. Why there is no override

Both gates are mandatory with no `--skip` flag, and that is the single most
considered decision here.

We audited LaTeXML, fixed it, recorded READY — and it was then absent from the
run host for a month while a stage that depended on it silently did nothing. The
check existed. It was optional. Optional checks are decoration.

If a gate refuses and you believe it is wrong, that is a bug in the gate and we
want to hear about it, not route around it. `preflight.py --fix` handles the
automatable failures.

---

## 4. If it stops

- **Read the halt message.** 11 stages halt by design; that is not a failure.
- **`--from S<n> --to S<m>`** resumes a range. The phase ledger in
  `data/runs/<run-id>` records what passed.
- **`--reuse`** accepts upstream stages from a previous run — never S2, which
  must match the corpus it is checked against.
- **A blocked stage** (upstream has no ledger entry) exits non-zero and says so
  rather than running on stale inputs.

---

## 5. What we would like back

1. `data/runs/$RUN_ID/` — the phase ledger and emitted artifacts
2. `retry-rate-$RUN_ID.json` — the number we cannot produce ourselves
3. `conformance.json` — particularly the throughput figure, which is the first
   real measurement of this pipeline on your hardware
4. `anchor-faithfulness.txt` — red as described; useful to us anyway

---

## 6. What we know is still weak

Stated plainly so nothing surprises you; the attached PDF has the full ledger.

- **The pipeline has never run end-to-end on a superpod.** This run is the first.
  Every stage has run somewhere; the sequence has not run there.
- **H38, anchor frame mismatch** — a measurement defect, not a run defect.
- **CT pattern hotwords are weakly validated** — the lexical prior over prose
  passages ranks the Tier-1-confirmed pattern top-1 6.4% of the time (chance
  2.6%) and top-3 14.1% (chance 7.7%), converging to chance by top-10; measured
  on 94 passages / 283 verified matches (futon6 `00a7360`). Real signal at small
  K only, and the validation is biased *against* the prior — patterns invisible
  to extraction cannot earn credit. Fine as a Tier-0 retrieval prior, which is
  all they are used as; Tier-1 filters at run time. Not a precision claim.
- **A3′ anchors, A8 export counts** — see the PDF; both restated this week
  against re-measured artifacts rather than left as they were.

The attached capability proof is organised around sub-claims A1–A14 with a
warrant column that says `mechanical`, `weak`, or `designed, not run` for each.
Where something is not established, it says so.
