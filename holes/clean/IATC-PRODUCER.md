# IATC argument-graph → CLean producer (the LLaMA superpod path)

`scripts/iatc_to_clean.py` turns the IATC argument-graphs LLaMA already produces
(`mark3_iatc_loop`, the `:nodes`/`:edges`/`:holes` shape) into CLean — leaving
exactly **one** thing for the model to do: type each box. Everything else is
deterministic. This is the path that runs on the superpod/Linode where there is
no Claude or Codex, only LLaMA.

## The split

| part | who | what |
|---|---|---|
| skeleton | deterministic | inference edge → box (`:premise`→`:consumes`, `:conclusion`→`:produces`); `:missing-warrant` → `:hole`; provided `:warrant` → `:discharges`; claim-flow → `:clean/wires`; edge order → `:clean/seq` |
| typing | **LLaMA** | the `:method` tag per box + `:clean/shape :macro`, constrained to `clean-method-vocab.edn` |

## Two-step run

```bash
# 1. mechanical skeleton (methods = :untyped-step) + the box-typing prompt
iatc_to_clean.py GRAPH.edn --out OUT.clean.edn --emit-prompt OUT.prompt.txt
# 2. LLaMA answers the prompt with JSON {"<box>": "<method>", "_macro": "..."};
#    inject it:
iatc_to_clean.py GRAPH.edn --apply typing.json --out OUT.clean.edn
```

The prompt hands LLaMA the vocabulary (with glosses), the macro choices, and each
step as `FROM: <premises> THEN: <conclusion> (open obligation: …)`. The skeleton
already passes `clean_argcheck.bb` with `:untyped-step` placeholders, so typing is
the only model-dependent step and it can't break well-formedness (it only fills
the method values; `:clean/seq` is rebuilt from them on `--apply`).

## Worked example (verified end-to-end)

`data/iatc-argument-graphs/gh200/math__0204218.edn` — a Brown-representability
proof (strong generator → H is a direct summand of `Hom(-,Q_n)` → projector →
Yoneda → Karoubian splitting → represents H → H representable):

1. produce → 6 boxes, 6 wires, 1 hole (`e-represents`), 1 discharge (`e-yoneda`,
   to the Yoneda identification);
2. type (here by hand, the LLaMA role): construct → reduce-to-known → transport →
   construct → reduce-to-known → reduce-to-known; macro `construct-exploit-discharge`;
3. `clean_argcheck.bb` → **PASS**;
4. `clean_to_lean.py` → DarkTower Lean → **compiles 0-sorry**.

So the same chain that lifts the APM proofs runs on an arXiv IATC graph with only
LLaMA in the loop: **IATC graph → CLean → gate → Lean → Rob's neo4j+pgvector.**

## What the gate catches (a real finding)

The DAG gate (G7) rejects **cyclic-equivalence proofs** where the producer's
claim-flow forms a cycle — e.g. `1512.07573` (conditions 1⇔2⇔3⇔4, proven round a
loop) and `0705.0102`. These are not malformed *proofs*, but they are not acyclic
combs: an equivalence cycle needs a different encoding (a `copar` of the two
implication directions, or splitting the cycle into named lemmas) before it has a
CLean comb. Until then the gate correctly refuses them rather than emitting a
cyclic "comb" — the same gate-before-ingest discipline as the IATC substance gate.
Degenerate graphs (e.g. `0801.2567`: fan-in to one claim, a self-loop step) pass
as DAGs but produce no wires; the self-loop (X⇒X) is dropped and noted.

*Cross-refs:* `scripts/iatc_to_clean.py`, `clean-method-vocab.edn`,
`CLEAN-LEAN-RELATION.md`, `NEO4J-PGVECTOR-MAPPING.md`, `E-clean.md`.
