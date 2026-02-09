# Claude Code Instructions for Futon6

## What This Project Is

Futon6 is a comprehensive mathematics dictionary where informal and formal
arguments coexist, indexed by patterns at multiple levels of abstraction.
The development map lives at `../futon3/holes/futon6.devmap` — read it first.

Python project. `pyproject.toml` has dependencies. Use `.venv/` for virtualenv.
Tests: `pytest tests/`. Scripts: `scripts/` (mix of Python and Babashka).

## The Futon Ecosystem

Futon6 does not exist in isolation. Sibling repos provide infrastructure:

| Repo | Role | What F6 uses from it |
|------|------|---------------------|
| futon3 | Pattern library + check DSL | `library/*.flexiarg` — reasoning patterns, hotword lists, PSR/PUR discipline |
| futon3b | Coordination rewrite + query layer | core.logic federated search; gate pipeline (G5-G0) |
| futon5 | Category theory + tensor math | Wiring diagrams, hexagram/eigendecomposition, CT DSL |
| futon1a | Durable store (XTDB) | Evidence persistence, proof paths |
| futon3a | Meme store (SQLite + ANN) | Embeddings, vector search |

When starting work on futon6, check what sibling repos offer before building
from scratch. Composition over reinvention.

## The Development Protocol

Futon development follows a pattern-backed methodology. The key discipline:
every significant decision should be traceable to a pattern, and every pattern
application should be recorded.

### Starting a New Prototype

The devmap defines prototypes P0-P9. When beginning work on a prototype:

1. **Read the devmap entry** — note maturity, dependencies, evidence, next-steps
2. **Check dependencies** — a prototype with `:depends-on [f6/P1]` needs P1 done first
3. **Mine existing evidence** — search session history and sibling repos for related work
4. **Write before coding** — define what success looks like before writing code

### The Derivation Xenotype

When deriving new patterns, domain constructs, or architectural decisions,
follow this sequence:

```
IDENTIFY  →  MAP  →  DERIVE  →  ARGUE  →  VERIFY  →  INSTANTIATE
```

- **IDENTIFY**: Name the tension or gap. What's missing? What breaks?
- **MAP**: Find related work — in the pattern library, in sibling repos, in
  session history. Use `grep -r` on `~/code/futon3/library/` for patterns.
- **DERIVE**: Produce a candidate solution. Could be a pattern, a data model,
  an algorithm, a schema.
- **ARGUE**: Write the argument in IF/HOWEVER/THEN/BECAUSE form:
  ```
  IF: <goal or desideratum>
  HOWEVER: <obstacle or tension>
  THEN: <proposed action>
  BECAUSE: <grounding reason>
  ```
- **VERIFY**: Check the argument against evidence. Does it hold? What breaks it?
- **INSTANTIATE**: Produce the artifact — code, config, documentation, test.

You don't need to follow every step for small changes. But for anything that
establishes a precedent or commits to an architecture, the full sequence pays off.

### PSR/PUR Discipline

When applying a pattern from the library:

**PSR (Pattern Selection Record)** — before applying:
```
Pattern chosen: <namespace/pattern-name>
Candidates considered: <what else you looked at>
Rationale: <why this one fits>
```

**PUR (Pattern Use Record)** — after applying:
```
Pattern: <namespace/pattern-name>
Actions taken: <what you did>
Outcome: <success/partial/fail>
Prediction error: <what surprised you>
```

This creates traceability. Future agents (and humans) can see why decisions
were made and whether the reasoning held up.

### Evidence-First Design

The devmap tracks evidence for each prototype. When you do work:

- Add `evidence[...]` lines to the devmap entry
- Be specific: counts, file names, validation results
- Distinguish between what's demonstrated and what's optative (hoped-for)

Example from P0:
```
evidence[496/535 entries tagged; all 25 patterns fire (27-291 hits each)]
```

Not:
```
evidence[tagging works well]
```

## Current State (Read This)

**Active prototypes** (have real evidence, not just plans):

- **P0 — Informal Argument Support**: 25 math-informal patterns created,
  496/535 PlanetMath entries tagged via classical hotword scoring.
  Closed loop validated: LLM extracts -> flexiarg formalises -> keyword recognises.

- **P7 — StackExchange Import**: 114K physics QA pairs processed, NER kernel
  built (19,236 terms from full PlanetMath), classical term spotter at 100%
  SE coverage. Superpod batch job ready for math.SE processing.

**Greenfield prototypes** (defined but no code yet): P1-P6, P8-P9.

**Key data artifacts**:
- `data/pattern-tags.edn` — pattern-to-entry mappings from P0
- `data/se-physics.json` — 114K physics.SE QA pairs (438MB)
- `data/ner-kernel/` — NER term dictionaries
- `data/math-terms.tsv`, `data/pm-terms.tsv` — term dictionaries

**Key scripts**:
- `scripts/tag-patterns.bb` — Babashka: match patterns to entries via hotwords
- `scripts/spot-terms.bb` — Babashka: classical NER term spotter
- `scripts/build-ner-kernel.bb` — Babashka: build NER kernel from PlanetMath
- `scripts/superpod-job.py` — Python: 4-stage GPU pipeline for SE processing
- `scripts/process-stackexchange.py` — Python: streaming SE XML parser
- `scripts/process-all-planetmath.sh` — Shell: batch PlanetMath processing

## The Flexiformal Pathway

A core idea: mathematical content exists on a spectrum from informal to formal.
Futon6 should support content at every level and provide pathways between them.

```
EXPLORE       →  ASSIMILATE      →  CANALIZE
(prose only)     (prose + logic)    (prose + logic + tensor)
```

- **EXPLORE**: Write in natural language. Patterns as IF/HOWEVER/THEN/BECAUSE.
- **ASSIMILATE**: Add machine-checkable structure. Could be a query, a type,
  a constraint. In the futon ecosystem, this often means core.logic relations
  (futon3b) or check DSL entries (futon3).
- **CANALIZE**: Add geometric/algebraic characterization. In the futon
  ecosystem, this means wiring diagrams or tensor representations (futon5).

Not everything needs to reach CANALIZE. But the pathway should exist.

## Cross-Agent Protocol

Futon6 work may involve multiple agents (Claude, Codex, etc.):

- **Read before writing**: Always read the devmap and recent session history
  before starting work. Check `~/.claude/projects/` for recent transcripts.
- **Commit evidence**: When you do something, update the devmap evidence lines.
- **Review across agents**: If another agent wrote code, review it before
  building on top. Codex and Claude have complementary strengths.
- **Coordinate via devmap**: The devmap is the shared state. Update `:next`
  steps when you complete work. Mark maturity transitions.

## Practical Conventions

- **Babashka scripts** (`.bb`): Used for pattern tagging, NER, term spotting.
  These are Clojure scripts run via `bb`. They interoperate with futon3's
  pattern library and EDN data formats.
- **Python modules** (`src/futon6/`): Core processing — LaTeX term extraction,
  PlanetMath parsing, StackExchange processing, similarity computation.
- **Data flow**: Raw sources (PlanetMath .tex, SE XML dumps) -> Python parsers
  -> JSON/TSV intermediates -> Babashka enrichment (tagging, NER) -> EDN output.
- **Large files**: `.gitignore` should exclude `data/se-*.json` and other
  multi-GB dumps. Check before committing.
- **Tests**: `pytest tests/` — keep coverage on core parsing logic. Tests
  should be self-contained (use fixtures, not live data).

## When You're Stuck

1. Search session history: `grep -r "<keyword>" ~/.claude/projects/`
2. Search the pattern library: `grep -r "<keyword>" ~/code/futon3/library/`
3. Read the devmap: `../futon3/holes/futon6.devmap`
4. Check sibling repos for prior art before building new
5. If a design question has no clear answer, write the argument
   (IF/HOWEVER/THEN/BECAUSE) and let the tension sit rather than guessing
