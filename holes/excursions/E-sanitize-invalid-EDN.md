# Excursion: E-sanitize-invalid-EDN — making model-emitted EDN parse safely

**Date:** 2026-06-18
**Status:** PARTIAL — the core fix is **shipped + tested** (commit on `master`, see below);
the **thorough scan / hardening** is OPEN (this excursion's forward half).
**Repo:** futon6 — `scripts/iatc_repair.bb` (the mechanical canonicalization step the IATC
driver runs before the gate), `tests/test_iatc_repair_sanitize.py`.
**Spawned from:** the first real-GPU Stage-A smoke run (`linode-test-runner`, 2026-06-18).
9/10 papers passed; the one failure — `0712.0724`, a category-theory paper — turned out to be
a class of bug, not a one-off, so Joe asked to fix it now *and* open this excursion for the
part we can't be sure we've covered.

## HEAD (one line)
**The 70B writes raw LaTeX inside EDN strings, and EDN's reader can't take it.** A backslash
command like `\circ` either crashes the reader (loud) or silently decodes to a control
character (silent). The fix is a text-level, string-aware escape sanitizer in the repair
step; the open question is whether "double every non-EDN escape inside strings" misses any
case a more thorough EDN-aware pass would catch.

## The problem (two hazard classes)

The model embeds LaTeX in description strings — `"u \circ \phi"`, `"A \otimes B"`,
`"\nabla f"`. Inside an EDN string a backslash starts an escape, and:

- **LOUD** — `\circ`, `\otimes`, `\alpha`, `\cong`: the leading letter is **not** a legal EDN
  escape, so the reader throws `Unsupported escape character: \c` and the *entire* graph fails
  to parse. The driver then drops it at the gate. This is what bit `0712.0724` (attempt 0).
- **SILENT** — `\times`→`\t` (TAB), `\nabla`→`\n` (newline), `\beta`→`\b`, `\rho`→`\r`,
  `\frac`→`\f`: the leading letter **is** a legal escape, so the string parses *without error*
  and silently decodes to a control character — corrupting the anchor text. This never trips
  the gate, so it would survive as an **L4 anchor-faithfulness** defect. This is the class Joe
  flagged as "the items we might miss."

`iatc_repair.bb` already back-filled `:source` and coerced node `:kind`s, but its own
`edn/read-string` threw on the loud class and silently no-op'd, so the broken file passed
through untouched.

## What's shipped (the core fix)

`sanitize-edn-escapes` in `iatc_repair.bb`, run at **text level before any parse** (so it
also un-breaks the repair step's own reader), and persisted even if the structural parse still
fails for some other reason:

> Inside a double-quoted string, double the backslash of **anything** that isn't a genuine
> `\"`, `\\`, or `\uXXXX` (4 hex). Backslashes outside strings are left untouched.

Doubling (rather than a LaTeX→unicode lookup table) is deliberate: a lookup table is a
whitelist that is *always incomplete*, whereas "double any non-escape" catches the **whole
class** — loud and silent — and **faithfully preserves** what the model wrote (`\circ` stays
`\circ`, just as a literal backslash). Mechanical + checkable against the text itself, so no
consent gate (per `futon3/library/.../mechanical-vs-semantic-consent`).

**Evidence:**
- On the real failing artifact (`0712.0724.attempt0.edn`): argcheck `FAIL [edn-parse]` →
  `PASS` after repair; `\circ` preserved as `\\circ`.
- Silent class: a synthetic `"A \times B then \nabla f"` reads back with `has-tab? false`,
  `has-newline? false` — no corruption.
- `tests/test_iatc_repair_sanitize.py`: 3 tests (loud-unparseable-before, parseable-after,
  faithfulness-no-silent-corruption) — pass.
- Gates: clj-kondo 0/0; parens balanced (bb reader reads all 4 forms).

## What's still OPEN (the forward half — where we might have missed something)

1. **Char literals outside strings.** The sanitizer toggles in/out of "string" state on bare
   `"`. An EDN **character literal** that is a quote written outside a string (`\"` as a char,
   or `\space`, `\newline`) could in principle confuse the state machine. The current IATC
   graphs contain no char literals (only keywords/strings/numbers/vectors), so this is
   theoretical *here* — but a general-purpose sanitizer needs a real EDN-aware tokenizer, not a
   quote-toggle. **Action:** if char literals ever appear, switch to a tokenizer (or
   bb's `clojure.tools.reader` with custom escape handling).
2. **Corpus scan.** We fixed the step; we have not swept the **already-produced** graphs for
   silent corruption that predates the fix. **Action:** scan `data/iatc-argument-graphs/**`
   (and any persisted run dirs) for control chars inside `:text`/string fields and for the
   tell-tale doubled-then-decoded patterns; re-repair + re-gate any hits. Quantify how many of
   the historical "passes" carried a silent `\t`/`\n`.
3. **Prompt-side prevention.** Sanitizing is a backstop. A prompt instruction ("use unicode
   operators ∘ ⊗ ≅ ∇, not LaTeX backslashes") would reduce the rate at the source. Worth an
   A/B once we re-run on GPU.
4. **LaTeX→unicode nicety.** Optional: after sanitizing, map a small set of very common
   commands to unicode for *readability* of the stored graphs (`\circ`→∘). Strictly cosmetic;
   must stay faithful and reversible. Lower priority than 1–2.
5. **Adjacent, not-this-excursion:** `0712.0724` attempt 1 failed on edges missing
   `:source {:lines [a b]}` (and inverted spans like `[886 885]`). That's a *schema-compliance*
   gap, not an escape one — repair's `:source` back-fill only fires when the referenced nodes
   carry line spans. Tracked here as context; its own fix (more attempts / prompt / span
   normalization) belongs with the IATC gate work, not the escape sanitizer.

## Exit condition
This excursion closes when (1) the corpus scan has run and any historical silent-corruption is
remediated or shown absent, and (2) a decision is recorded on the tokenizer upgrade (do it, or
document that the quote-toggle is sufficient for the IATC graph grammar). Until then the core
fix stands on its own; this file is the standing reminder of what it does *not* yet guarantee.
