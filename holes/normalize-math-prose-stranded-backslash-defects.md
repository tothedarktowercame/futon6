# normalize-math-prose strands the backslash of commands it half-matches

claude-opus-5, 2026-09-23, found while tracing why Agency went unresponsive.

`scripts/normalize-math-prose.py` converts prose notation (`pi_1`, `y_n`,
`Hom(Z/2,Z)`) into TeX. Several of its rules place a `$` immediately next to a
backslash that belongs to a TeX command, producing the byte pair `\$`. Until
today that pair also made the scanner loop forever, which is how 84 KB of input
reached 57 GB resident and throttled every process in the `futon3c-zone.service`
cgroup, the Agency JVM included.

The hang is fixed. What remains is wrong output, at small but non-zero scale.

**Correction to anything you may have heard from me earlier today:** I first
reported that `$$` display math was being fed to the prose rules, and put the
figure at 811 lines across 170 files. That is wrong. `process_file` (L1251-1254)
passes any line containing `$$`, and any line inside a `$$` block, through
untouched. Display math is protected. My measurement had called `process_line`
directly and bypassed that guard. The real numbers are below and they are much
smaller. Measure through `process_file`, never `process_line`.

## Already fixed — do not redo this part

In the working tree, not yet committed:

| Site | Change |
|---|---|
| `split_inline_math_dollar` else-branch, L776 | Scan starts past an escaped `$`, so it always advances |
| `pi_N` rule, L1192 | `(?<!\\)` added — `\pi_1` no longer re-wrapped |
| general subscript rule, L1204 | `(?<!\\)` added — `\mu_k` no longer re-wrapped |
| `tests/test_normalize_math_prose_escaped_dollar.py` | New, 15 tests, passing |

Measured over the 491 files in `apm-lean/problems/*/informal-solution.md`,
running each file through `process_file` in a subprocess capped at 2 GiB:

| | before | after |
|---|---|---|
| files that exhausted a 2 GiB cap | **15** | 0 |
| files whose output carries a generated `\$` | 18 | **5** |
| lines with a generated `\$` | 34 | **5** |
| files whose output changes at all | 430 | 444 |

The 15 files that blew the cap are the ones that would hang in production; with
no cap the first of them reached 57 GB. `\$` occurs in **zero** source files
across the corpus, so every occurrence in output is manufactured by this script.
That makes "output contains `\$`" a clean acceptance signal — it should reach 0,
and no test has to decide what a legitimate `\$` would look like.

The new tests call anything that could stall through a subprocess with a 1 GiB
cap and a 20 s timeout. Keep that shape. An earlier version of the same tests
called in-process and, against the unfixed script, hung the runner instead of
failing it — reproducing the original incident inside CI.

## 1. A rule's match boundary lands on a backslash

One shape, two edges. A rule matches part of a TeX command and wraps it in
`$...$`; the backslash ends up outside the wrap.

**Left edge** — `\b` sits between the backslash and the first letter, so the rule
matches the command *name*:

```
\pi_1   ->  \$\pi_{1}$      (pi_N rule, L1192 — FIXED)
\mu_k   ->  \$mu_{k}$       (general subscript, L1204 — FIXED)
\psi(x) ->  \$\psi(x)$      (PSI_CALL_RE, L86, applied L1061 — OPEN)
\Hom(A,B) -> \$\mathrm{Hom}(A,B)$   (Hom functor, L1180 — OPEN)
```

`PSI_CALL_RE` has a second problem: L1061 applies it with a direct `.sub`, not
through `_sub_outside_inline_dollar`, so it also fires inside inline math. Worth
grepping for other direct `.sub` calls that should be going through the helper.

`\Hom` produced no corpus hits — it is latent, not observed.

**Right edge** — a lazy tail plus a lookahead that stops at `,` ends the match on
the backslash of `\,`, so the *closing* `$` lands after it:

```python
BAR_COMPARE_WRAP_RE = re.compile(          # L218
    r"(?:(?<=\s)|^)((?:\|\|[^|\n]+\|\||\|[^|\n]+\|)\s*(?:<=|>=|<|>|=)\s*[^.;,\n]*?)"
    r"(?=\s+\b(?:for|if|with|where|when|and|or)\b|\s{2,}\(|[.;,:]|$)"
)
```

Minimal reproduction:

```
|f(z)| = \left|x\,dt\right|   ->   $|f(z)| = \left|x\$,dt\right|
```

`[^.;,\n]*?` is lazy and the lookahead fires at the `,` of `\,`, leaving the `\`
as the last character of group 1. `PROB_COMPARE_WRAP_RE` at L214 has the
identical tail and lookahead and is presumably latent in the same way — it was
not observed in the corpus.

The `(?<!\\)` idiom is already used in this file at L126 (`Phi(`), L241
(`diag(`) and L1228, so the convention exists; it has just been applied
unevenly. A lookbehind fixes the left edge. The right edge needs the trailing
class to stop before a backslash, or the wrap to trim a trailing `\` — that is a
judgement call, not a mechanical edit.

### The 5 remaining damaged lines

| File:line | Rule | Output |
|---|---|---|
| `a93J06:20` | `BAR_COMPARE_WRAP_RE` | `\left|\int_0^1 z f'(tz)\$,dt\right|` |
| `a98A06:14` | `BAR_COMPARE_WRAP_RE` | `$|g(z)|=|z|\$,|f(z)|=...` |
| `a99J02:22` | `BAR_COMPARE_WRAP_RE` | `∫ $\chi$_E(y) $\chi$_E(y−t)\$,dy` |
| `b94A02:82` | `PSI_CALL_RE` | `\Phi\big(\$\psi(s+I)$\big)=...` |
| `m02J01:14` | `PSI_CALL_RE` | `the subtle direction is that $\$\psi(x)$=\int_{-` |

Three from the right edge, two from the left. Fixing both edges should take the
count to 0, which is the acceptance bar.

## 2. An unbalanced backtick exposes a line of TeX to the prose rules

This is what put `t00J06` on the pathological path.
`problems/t00J06/informal-solution.md:31` contains one backtick:

```
determines has infinitely many sheets, and by the **lifting criterion** (`f_*(\pi_1\mathbb T) = H \subseteq
```

`split_inline_code` finds no closing backtick on the line and returns the
remainder as `text` rather than `code`, so the prose rules run on TeX source.
Across the corpus, outside fenced blocks and after applying `process_file`'s own
skip rules (`$$` lines, indented blocks), **307 such lines in 59 files actually
reach `process_line`**. None are filtered out.

Mostly these are inline-code spans the author wrapped across two lines, which
GFM does not join. This is arguably a content defect rather than a code defect,
and the choice is worth making explicitly rather than by default:

- treat an unterminated backtick as opening a code span to end of line, which
  protects the TeX but changes how those 307 lines render; or
- repair the 307 source lines and leave the parser alone; or
- leave both, and rely on defect 1 being fixed so the exposure is harmless.

The third is defensible, and is why this is listed second rather than first. It
is also the only one of the three that costs nothing. Do not fix it silently.

## Acceptance

Suggest splitting this into separate handoffs — the left edge is four one-line
regex edits, the right edge is a judgement call about two patterns, the backtick
question is a decision with no code attached. One packet each.

For any of them:

1. `tests/test_normalize_math_prose_escaped_dollar.py` stays green
   (`.venv/bin/python -m pytest tests/test_normalize_math_prose_escaped_dollar.py -q`),
   with new cases added in the same bounded-subprocess shape.
2. Generated-`\$` count over the corpus goes from 5 to **0** and stays there.
   That number is not in the repo yet — land the survey as a test or a script so
   the claim is checkable rather than asserted.
3. Differential diff of old vs new output over all 491 files **through
   `process_file`**, with every changed line classified. Anything that does not
   fall into a named category is a regression until explained. Expect a large
   number of files to change for innocent reasons — 444 of 491 already change
   under normal operation — so a bare file count proves nothing; classify lines.
4. The pipeline still runs end to end:
   `apm-lean/scripts/informal-proofs-to-tex.sh storage/apm/first-proof-sprint-ids.txt`
   (needs `pandoc`). It stages copies into `mktemp -d` and removes them on EXIT,
   so a crash takes the evidence with it — copy the temp dir aside before
   killing anything.

## Reproducing the measurements

In this session's scratch, which will not survive:
`/tmp/claude-1000/-home-joe-code/f6425a38-a686-48a7-9cb3-d7207e3f8a96/scratchpad/`
— `survey2.py` (corpus counts via `process_file`), `survey3.py` (the same with a
per-file memory cap and timeout, which is what produced the before/after table).
Copy them into the repo if they are worth keeping. `survey.py` in that directory
is the flawed `process_line` version that produced the retracted figure; do not
build on it.

Cap the memory of anything that processes this corpus on Zone. These scripts,
and the agents running them, share the `futon3c-zone.service` cgroup with the
Agency JVM (`memory.high` = 64 GB); that is how the original incident took
Agency down rather than just one process.

---

## Status update — defect 1 closed

claude-7, 2026-09-23, while converting the APM Lean informal proofs to LaTeX.

Both edges of defect 1 are fixed and the acceptance bar is met. Defect 2 is
untouched and still needs the decision it asks for.

| Site | Change |
|---|---|
| `PSI_CALL_RE` (L86) | `(?<!\\)` added — no longer matches the name of an existing `\psi` |
| `PSI_CALL_RE` application (L1061) | routed through `_sub_outside_inline_dollar` instead of a bare `.sub`, so it no longer fires inside inline math |
| `Hom` rule (L1180) | `(?<!\\)` added — latent left edge closed |
| `PROB_COMPARE_WRAP_RE`, `BAR_COMPARE_WRAP_RE` (L214, L218) | lookahead punctuation class is now `(?<!\\)[.;,:]` |

The right edge was described as a judgement call between stopping the trailing
class before a backslash and trimming a trailing `\` off the wrap. Neither was
needed. The cleaner fix is to stop the *lookahead* firing on punctuation that
belongs to a TeX command: in `\left|x\,dt\right|` the match ended at the `,` of
`\,`, so making that comma invisible to the lookahead lets the match run to a
real sentence boundary. Trimming the backslash would have removed the `\$` while
leaving the wrap boundary in the wrong place.

Acceptance, all four items:

1. `tests/test_normalize_math_prose_escaped_dollar.py` — 15 passed.
2. Generated `\$` over the corpus: **5 → 0**. Landed as
   `scripts/survey-normalize-generated-dollars.py`, which exits non-zero on any
   hang, memory-cap breach or generated `\$`, so the claim is checkable.
3. Differential diff through `process_file` over all 501 files: exactly **5
   files, 5 lines** differ — `a93J06:20`, `a98A06:14`, `a99J02:22`, `b94A02:82`,
   `m02J01:14`, the five this note listed. Every changed line classifies as
   "stranded backslash removed". No line fell outside that category, and no new
   `\$` appeared anywhere.
4. Pipeline runs end to end: all 501 files convert.

Corpus is now 501 files, not 491 — the 6 stubs were written out and 10 problems
that had no informal solution now have one.

### What this did not fix

Defect 2 (the unbalanced backtick, 307 lines in 59 files) is untouched, by
design: this note asks for that to be decided rather than fixed silently.

Converting all 501 files through the whole pipeline and measuring the *final*
LaTeX shows the remaining damage is downstream of this script, in
`pandoc-mathify.lua` or pandoc itself:

- **499 escaped dollars across 129 files** in the emitted `.tex`. Zero source
  files contain `\$` and `normalize-math-prose` now generates none, so all 499
  are manufactured after it runs.
- **579 TeX commands printed as literal text** (`\textbackslash in`,
  `\textbackslash mathbb`, …) in body prose across 95 files. A further 1,542
  occurrences sit in `\texorpdfstring` bookmark strings, which are cosmetic.
- **11 unbalanced math spans in 5 files** that `repair-math-spans.py` cannot
  settle; it repairs 124 spans across the corpus.

Those want their own packet, against the filter rather than this script.
