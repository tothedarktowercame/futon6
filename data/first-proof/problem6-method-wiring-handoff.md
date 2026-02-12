# Handoff to Claude: Problem 6 Method Wiring Library

Date: 2026-02-12
Author: Codex

## What was added

1. `data/first-proof/problem6-method-wiring-library.md`
- Human-readable method library.
- Defines the reduced Problem 6 bridge target (`RP-VLS`):
  - find universal `c0>0` with induced subset `S`, `|S| >= c0*epsilon*n`, `L_{G[S]} <= epsilon L`, no reweighting.
- Defines a wiring-diagram similarity schema (`Q,D,M,C,O,B`) so methods can be compared structurally.
- Instantiates 10 related method diagrams (D1..D10), each mapped to a paper/method and tagged with bridge status to RP-VLS.

2. `data/first-proof/problem6-method-wiring-library.json`
- Machine-readable version of the same content.
- Includes:
  - reduced-problem wiring graph,
  - similarity schema,
  - all 10 method diagrams (nodes/edges + bridge status).

## How I produced the diagrams

I used the house style from existing proof wiring artifacts (`problem6-wiring.json`, `problem6-v2.mmd`) and abstracted a reusable shape:

- `Q`: target spectral objective
- `D`: decomposition into selectable atoms
- `M`: selection mechanism
- `C`: proof certificate (concentration/potential/interlacing/etc.)
- `O`: output guarantee
- `B`: explicit bridge verdict back to RP-VLS

For each paper/method, I extracted the method skeleton (not full theorem restatements) and instantiated this shape with concise node texts and typed edges (`clarify`, `assert`, `reference`, `reform`/`challenge`).

## Papers/methods instantiated

- D1: `arXiv:0803.0929`
- D2: `arXiv:0808.0163`
- D3: `arXiv:0912.1623`
- D4: `arXiv:0911.1114`
- D5: `arXiv:1306.3969`
- D6: `arXiv:1410.4273`
- D7: `arXiv:1810.08345`
- D8: `arXiv:1810.03224`
- D9: `arXiv:1811.10834`
- D10: `arXiv:1906.10530` + `arXiv:2005.02368`

## Why this is useful for Problem 6

- It isolates the precise missing theorem-level bridge (vertex-induced, no reweighting).
- It separates directly reusable primitives from non-transferable ones.
- It suggests a concrete hybrid direction for an unconditional proof attempt:
  - `D5`-style existence machinery (interlacing/polynomial method), plus
  - `D7`-style dependence-robust concentration on vertex indicators.

## Caveats

- This is method mining, not a new theorem proof.
- Bridge statuses are intentionally conservative; all current entries are `partial`.
- No claim is made that edge-sparsification results imply RP-VLS directly.

## Quick validation done

- JSON syntax check passed:
  - `python3 -m json.tool data/first-proof/problem6-method-wiring-library.json`

