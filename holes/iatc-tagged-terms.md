# Tagged terms in IATC nodes — closing the gap the 0919b probe exposed

Joe, 2026-09-20, on the worst case from the probe (a node that should read
"for all X in T there exists a Sigma^{-1}A-precover alpha: Sigma^{-1}A -> X",
stored as control-character noise): *"what we should have is more like tagged
terms for each, showing what the symbols mean."*

The machinery for that already exists upstream. It is thrown away at S3.

## What S1 already knows

Every candidate carries an `enrichment` list, and S1 populates it with symbol
bindings — measured over the probe's 124 candidates:

| enrichment kind | rows |
|---|---|
| `symbol-grounded` | 2,959 |
| `bind/typed` | 450 |
| `constrain/relation` | 324 |
| `definiendum` | 239 |
| `let-binder` | 238 |
| `bind/let` | 96 |

**118 of 124 candidates (95%) carry at least one symbol binding**; 1,182 rows
bind a symbol to a meaning. For the worked example above, the candidate states:

    bind/let    · symbol:\T     | type:a triangulated category
    bind/typed  · symbol:\alpha | type:\Sigma^{-1}A\rightarrow X
    definiendum #0: $\T$

So `\T`, `\alpha`, `\beta` and `X` — the symbols that were destroyed — were
bound, in that candidate, before the model was called.

## Where it is lost

`render_enrichment` puts those rows in the S3 prompt as advisory lines. But
`nodes_schema(lo, hi)` gave each node exactly one free field for the
mathematics: `text`, a string of at most 300 characters. There was no slot for
a symbol, so the model retyped the mathematics by hand into prose.

That is where both failures happened:

- **Serialization.** 361 of 1,280 nodes (28.2%) carry raw control characters
  where a LaTeX command should be — `\t` from `\text`, `\b` from `\beta`, `\r`
  from `\rightarrow`, `\f` from `\forall` — because the writer did not escape
  backslashes and the reader consumed them as escapes. **87.5% of the damaged
  nodes are in candidates that already carried the bindings.**
- **Content.** codex-1's audit found reversed membership (`\notin` where the
  source says `\in`), a dropped hypothesis in `0806.1324__p31` where the
  condition needed for that direction of the proof is simply gone, definitions
  replaced by dashes, and 40 nodes where a proposition is replaced by 20 or more
  consecutive arrows. These survive escape repair.

Every gate passed on all of it: checker 242/242, substance 121/121. The gates
read shape and cross-item variety, not mathematics.

## The change

`nodes_schema` now takes the proof's bound symbols and requires each node to
declare which of them it is about, drawn from that enumeration.

This is the move `steps_schema` already makes and its docstring already argues
for: premises are drawn from the nodes that exist, so citing a node that was
never written is *unrepresentable* rather than rejected afterwards — which is
how three of twenty proofs in the first live run stopped failing. Symbols get
the same treatment. `\triangle` was never bound by that window, so under
constrained decoding it cannot be emitted.

    bound_symbols(candidate) -> ['\\T', '\\alpha', 'X', '\\beta']
    nodes_schema(349, 366, syms)  # each node: "symbols": enum of those four

Declaring an empty list stays legal — a node may be about no bound symbol.
Inventing one does not. Omitting the argument leaves the old contract untouched,
so this is inert until a caller passes symbols.

## What this does and does not fix

**Does:** gives every node a machine-checkable record of which bound objects it
concerns, independent of its prose. A downstream check can then ask whether a
node's `text` mentions the symbols it declared — which is exactly the test the
40 arrow-spam nodes would fail, since they declare objects and then contain none
of them.

**Does not:** repair the escaping bug. `text` is still a free string and can
still be mangled. That is a separate and simpler fix — escape backslashes on
write — and should land regardless. The value here is that a mangled `text` no
longer destroys the *only* record of what the node was about.

**Does not:** stop the model asserting something false about a symbol it
correctly names. Reversed membership and dropped hypotheses are content errors;
constrained decoding cannot reach them.

## The root cause, measured

It is not a writer that forgot to escape. Constrained decoding follows a JSON
string grammar, which permits only `" \ / b f n r t u` after a backslash. A
LaTeX command whose first letter is not one of those **cannot be written at
all**: the decoder forces a legal letter and the model completes a different
command.

Over the probe's raw model output: **8,437 backslash sequences, zero beginning
an illegal escape, 5,891 of them `\t`.** That is why `\T`, `\Sigma` and
`\alpha` all arrived as `\triangle` — `t` was forced and `\triangle` is the
likeliest completion. And why a source `\in` arrived as `\notin`: `i` is
illegal, `n` is legal, `\notin` is the likeliest completion. **A membership
claim was negated by a serialisation grammar.** 708 commands in the probe's
output appear nowhere in their own source window.

Commands starting `b f n r t u` survive in the bytes and are recoverable.
Everything else — `\Sigma`, `\alpha`, `\in`, `\cong`, `\coprod`,
`\mathcal` — is genuinely gone, replaced at generation time.

## The change, in full

Joe, 2026-09-20: *"the output contract is to produce something actually useful
and usable, and I abhor any kind of conservative policy that maintains shitty
work."* So the contract changed rather than being preserved.

A node now carries its mathematics by reference and never retypes it:

| field | what it holds | why it cannot be corrupted |
|---|---|---|
| `quote_lines` | absolute line numbers, an **enum** of the window's non-blank lines | code extracts the text; the model picks numbers |
| `symbols` | which bound objects the node concerns, enum from S1's bindings | never passes through a JSON string |
| `text` | a prose **gloss**, explicitly not the mathematics | nothing downstream reads formulae from it |

`quote_lines` is an enumeration rather than a range: a node drawing on lines 350
and 365 must say so instead of claiming everything between. A blank line is not
offered, so a node cannot anchor to nothing. Median window is 22 non-blank
lines, max 97 — an enum that size is reliable under constrained decoding.

`quoted_source(node, candidate)` returns the verbatim source. The destroyed
example, end to end:

    BEFORE  '$\x0corall X \text{ in } \triangle, ... \triangle^{-1}A ...'
    AFTER   '(i) For all objects $X$ of $\T$ there exists an
             $\Sigma^{-1}\A$-precover $\alpha: \Sigma^{-1}A\rightarrow X$.'

The NODES_TASK prompt now tells the model why, in its own terms: formulae typed
into `text` are not used and can only be wrong.

## What remains unfixed

Quotation makes retyped mathematics incorruptible. It does not stop the model
choosing the *wrong lines*, and it does not reach a false claim about correctly
quoted material — the reversed membership and the dropped hypothesis codex-1
found are choices, not transcription. Those need the semantic audit, which is
still the open piece.

`invented_commands()` flags LaTeX absent from the source window and catches the
`\notin` case. It misses runaway-arrow nodes because `\to` genuinely occurs in
these sources; the symbols rule covers 29 of those 40.
