# Excursion: E-scopes-math-to-code — port the math-NLP "scope" notion to code

**Date:** 2026-06-01
**Status:** DRAFT stub (Joe-sized as *excursion, not mission*, 2026-06-01). Owner TBD — assign on reswarm (natural fit: claude-6, paired with claude-2's `M-differentiable-math` timeline).
**Relation:** out-of-campaign scout feeding `C-substrate-completion`'s keystone `M-substrate-metric` **O1** by escalation only (not an escrow member). Supplies the candidate code-embedding grain `M-differentiable-code` (E2) needs.

---

## The move (deliberately small)

**Read how `scope` is used for math, then port it to code.** Not a formalization epic — a bounded read-then-port.

- **Read:** futon6's existing scope usage — `detect_scopes()` / `_find_scope_end()` (`scripts/nlab-wiring.py:919-1030`). Today a scope = a **heuristic char-window span** (`hard_span` 900/420), started by a token (`forall`/`exists`/`let`/`define`/`diagram-family`), typed + labelled `["scope","symbolic-binder",…]`. Used, but not crisply defined.
- **Port:** carry that notion to code as the candidate **`:scope`/overlay grain** — a region + binding structure over code, embeddable on its text window (BGE, *not* the collapsed R-GCN arm). This is the grain that is neither `:file` (E2's 115k-line conditioning problem) nor bare `:symbol` (context-free).

## Why an excursion, not a refactor (Joe's framing)

Avoid a heavy code refactor by building an **overlay** instead of restructuring code. Inspiration: spec-as-data (malli for Clojure; an Emacs analogue) — a spec sits *over* code without changing it. Open question carried, not pre-answered: **scopes are likely *different* from malli** (malli = value-shapes; a scope = extent + binders + overlap), so "use them properly" may mean a light spec *of scopes* falls out of the port — but lead with read→port, let any specification need emerge from the porting, don't start by formalizing from scratch.

## Boundary (the one rule)

Prototype the `:scope` grain freely here (out-of-campaign, escrow-clean). **If the port concludes the real metric wants `:scope`, bring it to codex-3 as an O1 schema escalation** — never a unilateral re-author of the ratified `:file/:namespace/:symbol/:boundary` identity set.

## Pointers

- `M-differentiable-math.md` (claude-2) — the math timeline that already has the BGE embedding; pair with it.
- `M-differentiable-code.md` MAP (claude-6) — the `:scope`-grain finding + O1-escalation discipline already recorded.
- futon4 `M-self-representing-stack` / VSATARCS — the typed-hypergraph-overlay analogue.
