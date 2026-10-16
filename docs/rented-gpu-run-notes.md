# Running on a rented GPU: what the first non-CT corpus taught

These notes come from the first run of the pipeline on a corpus outside arXiv
math.CT: 501 informal solutions to qualifying-exam problems (the APM corpus).
It ran on one rented A100-80G on 2026-10-15 and 16. Each item below cost GPU
time once. The fix is named where there is one; the rest are still open.

The general point: **a rented box bills by the hour whether or not the GPU is
working.** Anything that can be checked on a CPU should be checked before the
box exists, and anything that goes wrong should be diagnosed after the box is
stopped, not while it waits.

## Before renting

**Vocabulary.** The staged concept vocabulary is math.CT's. For CT it is right
and for nothing else: on APM it is demo content. S2 (WARP) grounds concepts
only from definitions the corpus contains, and a corpus of problem solutions
defines almost nothing. The first attempt grounded 56 concepts. Preflight now
reports this as `vocabulary:fit` (a WARN, not a refusal). Fixes:
- `--build-lexicon` builds the run's lexicon from its own papers with a
  classical scan of the background index. It found 248 concepts on APM.
- `FUTON6_WARP_LEXICON_IDS` names a hand-made concept index that S2 reads and
  no stage mines. For APM this was a 2002 index of the same problems, which
  gave 228 concepts.

Do this before the GPU stages, not after them.

**Citations.** TAPESTRY (S12) weaves the citation graph. A collection whose
papers cite nothing has no graph to weave: APM has 0 of 200 papers citing,
against 197 of 200 for CT. Preflight reports this as `citations:tapestry`;
run with `--no-tapestry`.

**Math delimiters.** S1 used to read only `$...$`. About 9% of CT papers use
`\(...\)`, and S1 now reads both (c51b9d8). Check a sample of a new corpus for
anything else it uses.

**Pilot size.** A 16-paper pilot passed S4 at 91% (65/71). The full 501 then
passed at 67%, under the 75% floor. The pilot was too small to show a failure
in 29% of regions. Pilot 50 or more papers, and take the pilot through S4.

## Bringing the box up

**Driver.** Images that ship NVIDIA driver 535 (CUDA 12.2) cannot run a
current vLLM. `vllm==0.10.1.1` with `transformers==4.55.4` works: set
`VLLM_SPEC` for `linode-4gpu-setup.sh` (808c925). Conformance for mark8-v1
passes on that stack.

**Disk.** A 100 GB root disk holds the 70B AWQ model with room left, but less
than the default free-space check expects. Set `FUTON6_MIN_DISK_GB` (da370a2).

**Checkout.** Clone over HTTPS, or add GitHub to the box's `known_hosts`.
Otherwise the box cannot `git pull` a fix during the run. Before launching,
check that the box's `git rev-parse HEAD` is master's. On APM the box ran two
commits behind and nobody noticed.

## When a stage stops

1. **Copy the run back first.** Use `retrieve_run.py pack`, or rsync `~/runs`.
   On prepaid billing, check the balance at the same time. Then decide whether
   to stop the box. The run directory is portable, so the diagnosis does not
   need the GPU.
2. **A rerun is not a retry.** Every model call runs at temperature 0, so
   re-invoking a stage without a code change sends the same prompts and gets
   the same refusals. Re-invoking does keep every item already accepted and
   retries only the rest, but that helps only once something has changed:
   the code, the schema or the serving.
3. **Read the refusals grouped by reason.** Read
   `accounting/<stage>/<invocation>/<stage>.loop.json`, not the three examples
   in the log. On APM, 1,167 of 1,318 S4 failures had a single cause.

## S4 on non-CT prose

**Fill and held_reason.** A scope must have exactly one of `fill` and
`held_reason`. The rule was stated in the prompt and checked afterwards, but
it was not in the decoding schema. On APM the model wrote both (for example
`fill: "PDE"`, `held_reason: "No"`) in 1,167 of 4,016 regions, and S4 stopped
at 67%. The scope schema is now an `anyOf` of a filled branch and a held
branch, which guided decoding enforces (4ee3818).

General rule: any constraint the checker enforces that the grammar can
express belongs in the schema.

**Truncation.** 142 of 4,016 S4 regions hit `max_tokens=2048`. S3, S4 and
the S7 typing stage all threw a truncated answer's text away, so nothing was
left to say why. They now keep it: S3 and S4 next to the attempt's other
files (`<item>.partial.txt`, named in the attempt record), and S7 under
`<out>/.failed-responses/`. Five of the 142, chosen at random and rerun by
hand under the new fill/held schema, all finished, in 49 to 178 tokens. So
the old schema caused the truncation, but without the text there is no saying
how. The retry's own count of truncations is the measure for all 142.

**LaTeX in JSON.** A JSON string allows only `" \ / b f n r t u` after a
backslash, and guided decoding enforces that, so most LaTeX commands cannot
be written in an answer at all. S3 saw `\Sigma` forced into `\triangle`; S4
on APM got a raw TAB where `\chi`, `\mathbb`, `\cup` or `\mapsto` belonged,
which made 9 of 4,016 answers unparseable. A repair that guessed the lost
letters afterwards was tried and dropped: two of the twelve could not be
recovered, because the command was gone.

`scripts/latex_json.py` fixes it on both sides of the model, in S3 and S4:
- `allow_latex` puts a pattern on every free text field: no control
  characters, no unescaped quote, any character after a backslash. With it
  the model writes LaTeX as in a paper, `$g = \chi_{[-h,h]}$`.
- `loads` doubles every backslash that is not a JSON escape before
  `json.loads`, so it decodes to the backslash the model wrote. `\theta`,
  `\frac` and `\nabla` count as LaTeX: an escape letter followed by another
  letter is a command.

On mark4-70b, the S4 regions that had come back with a TAB for `\chi` and
`\cup` now decode to `$g = \chi_{[-h,h]}$` and `$({0},{1}) \cup ({2},{3})$`.
The price is that a line break written straight before a word (`\nThen`)
comes back as the command `\nThen`.

## S3

In the 16-paper pilot, S3's batch substance gate failed the whole batch
because of one degenerate output: a single warrant repeated 30 times. In the
full run the gate passed 500 of 500. S3 accepted 500 of 501. A
1800-second `FUTON6_LLM_TIMEOUT` was needed for the longest proofs. The
default of 300 timed out 2 of 16 in the pilot.
