# M-tickle-driven-ct — Tickle-Driven Category Theory Processing

## Status: In Progress

## Motivation

The futon stack has £400/month of agent compute capacity (OpenAI Pro + Claude
Max). The CT validation corpus (313 PlanetMath entries) and incoming arXiv
math.CT corpus (~5000 articles) represent real mathematical research that
produces documented value — wiring diagrams, pattern extraction, theorem
discovery.

Tickle orchestrates overnight batch processing: Codex extracts wiring
diagrams from raw mathematical text, Claude reviews the extractions, evidence
documents every step.

## Architecture: Cross-Surface Collaboration

The raw arXiv math.CT data lives on the **laptop** (Joe's local machine).
This is a deliberate architectural choice — the data's physical location
enforces a collaboration pattern through peripheral constraints:

- **Codex (laptop)**: Has direct filesystem access to raw arXiv .tex files.
  Role: research assistant — searches, reads, extracts wiring diagrams.
- **Claude (Linode)**: Reviews extractions, validates against ground truth,
  runs the mathematical analysis. Role: researcher.
- **Tickle (Linode)**: Orchestrates the pipeline, detects stalls, restarts
  agents. Role: conductor.

This isn't a prompt instruction or a hard rule — it's topology as invariant.
Codex physically has the data. Claude physically reviews on Linode. The WS
bridge and evidence replication connect them.

## Data Sources

### 1. PlanetMath CT Validation Set (on Linode — ready now)
- 313 entries in `futon6/data/ct-validation/entities.json`
- Full .tex in `/home/joe/code/planetmath/18_Category_theory_homological_algebra/`
- 20 golden exemplars with expected wiring output
- Ground truth counts (components, wires, ports) in `topology.json`

### 2. arXiv math.CT Corpus (on laptop — ready for processing)
- ~5000 articles, raw/unprocessed
- Format: TBD (Codex to explore and report)
- This is the real workload — the PlanetMath set is the validation baseline

### 3. nLab Reference (on Linode)
- 20,441 pages in `futon6/data/nlab-ct-reference.json`
- 8 categorical pattern types
- Cross-reference for validation

## Pipeline

```
Raw .tex (laptop) → Codex extracts wiring diagram → Evidence emitted
     → Claude reviews extraction → Evidence emitted
     → Ground truth comparison → Evidence emitted
     → Morning report (IRC + evidence summary)
```

Each entry produces:
- A wiring diagram JSON (components/ports/wires arrays)
- Review verdict (approve/request-changes/unclear)
- Ground truth delta (extracted counts vs classical baseline)

## Extraction Taxonomy (Wiring Metatheory v3.0)

- 30 component types (bind, quant, assume, constrain, formula, proof, conclude)
- 11 port types (demonstrative, structural, persistence)
- 5 wire types (adversative, causal, consequential, clarifying, intuitive)
- 32 wire labels (strategy, explain, correct, approx, epistemic, construct)

Full taxonomy in `futon6/data/ct-validation/golden/*.prompt.txt`.

## Implementation Status

### Done (futon3c side — Tickle orchestration)
- [x] Escalation → restart agents (dev.clj notify-fn)
- [x] CT work queue module (tickle_work_queue.clj)
- [x] REPL helpers (run-ct-entry!, run-ct-batch!, ct-progress!)
- [x] Integration tests (13 tests, 52 assertions)
- [x] Tickle autostart env var (FUTON3C_TICKLE_AUTOSTART)

### Next (futon6 side — Codex handoff)
- [ ] Explore arXiv math.CT data on laptop (format, structure, count)
- [ ] Write arXiv loader in futon6 (parallel to PlanetMath loader)
- [ ] Wire arXiv loader into tickle_work_queue interface
- [ ] Run validation batch on PlanetMath (10 entries, prove pipeline)
- [ ] Run first arXiv batch (50 entries, overnight)

## Economic Case

313 PlanetMath entries × ~3 min = ~16 hours
5000 arXiv entries × ~3 min = ~250 hours = ~10 nights

At full utilization: documented mathematical value every night, evidence
accumulating, patterns discovered, wiring diagrams produced. Way better
for society than mining bitcoins.
