# Mission: CT-Backed Verifier for First Proof Problems

## Context

The First Proof Sprint (problems 1-10) produced wiring diagrams capturing
proof structure as typed graphs: nodes are proof steps, edges are
argumentative moves (assert, challenge, clarify, etc.), and hyperedges
record categorical structure.

We now have two complementary datasets:

1. **CT Reference Dictionary** (`data/nlab-ct-reference.json`) — extracted
   from 20,441 nLab pages. Contains 8 categorical pattern types with
   required/typical links, discourse signatures, and diagram counts.

2. **Thread Wiring Diagrams** (`data/thread-wiring/*.json`) — 200 SE/MO
   threads assembled into three-level nested wiring diagrams with IATC
   edges, port matching, and per-node categorical annotation.

The goal: build a **just-in-time CT verifier** that checks whether each
edge in a proof wiring diagram has port types consistent with the
categorical patterns at each node. This turns the CT reference into an
active checker, not just a passive index.

## Task for Codex

Build a Python script `scripts/ct-verifier.py` that:

1. Reads a proof wiring diagram (same format as `problem4-wiring.json`)
2. Reads the CT reference dictionary
3. For each edge in the wiring diagram, checks:
   - Does the source node's categorical annotation match the target's?
   - Are the port types compatible across the edge?
   - Does the IATC edge type match the discourse signature in the reference?
4. Produces a verification report: which edges pass, which fail, and why

### Concrete deliverable

```
python scripts/ct-verifier.py verify \
    --wiring data/first-proof/problem4-wiring.json \
    --reference data/nlab-ct-reference.json \
    --output data/first-proof/problem4-verification.json
```

### Input formats

**Proof wiring diagram** (`problem4-wiring.json`):
```json
{
  "thread_id": "problem4",
  "nodes": [
    {
      "id": "step-1",
      "node_type": "claim|evidence|method|...",
      "body_text": "...",
      "score": 5
    }
  ],
  "edges": [
    {
      "source": "step-2",
      "target": "step-1",
      "edge_type": "supports|challenges|...",
      "evidence": "...",
      "detection": "regex|llm|manual"
    }
  ],
  "hyperedges": [
    {
      "hx/id": "...",
      "hx/type": "cat/adjunction|wire/consequential|...",
      "hx/ends": [{"role": "...", "label": "..."}],
      "hx/content": {"match": "...", "position": 0},
      "hx/labels": ["categorical", "adjunction"]
    }
  ]
}
```

**Thread wiring diagram** (`data/thread-wiring/*.json`):
```json
{
  "thread_id": 445920,
  "site": "mathoverflow.net",
  "topic": "category-theory",
  "level": "thread",
  "nodes": [
    {
      "id": "q-445920",
      "type": "question",
      "input_ports": [
        {"id": "...", "type": "bind/let", "label": "G : a group object"}
      ],
      "output_ports": [
        {"id": "...", "type": "constrain/such-that", "label": "..."}
      ],
      "categorical": [
        {"hx/type": "cat/equivalence", "hx/content": {"evidence": [...], "score": 8.1}}
      ],
      "discourse": [
        {"hx/id": "...", "hx/type": "bind/let", "hx/ends": [...], "hx/content": {...}}
      ]
    }
  ],
  "edges": [
    {
      "from": "a-445950",
      "to": "q-445920",
      "type": "responds-to",
      "iatc": "assert",
      "port_matches": [["a-445950:scope-002", "q-445920:scope-003", 2]]
    }
  ]
}
```

**CT Reference** (`data/nlab-ct-reference.json`):
```json
{
  "patterns": {
    "cat/adjunction": {
      "instances": ["nlab-193", ...],
      "instance_count": 927,
      "required_links": ["left adjoint", "right adjoint", ...],
      "typical_links": ["adjunction unit", "triangle identity", ...],
      "discourse_signature": {"components": {...}, "wires": {...}},
      "avg_diagrams": 5.0
    }
  },
  "link_weights": {
    "functor": {"definition-ref": 286, "prose-ref": 819},
    "category": {"definition-ref": 416, "prose-ref": 1979}
  }
}
```

### Verification checks to implement

**Check 1: Categorical consistency across edges**

For each edge `A → B`:
- Extract the categorical patterns at node A and node B
- If both nodes claim `cat/adjunction`, check that the required_links for
  adjunction are present in both nodes' text (via NER or substring match)
- If node A claims `cat/adjunction` but node B claims `cat/limit`, check
  whether these patterns co-occur in the reference (adjunctions preserve
  limits — this is a known theorem, so co-occurrence is valid)

Score: fraction of edges where both endpoints' categorical annotations are
mutually consistent.

**Check 2: Port type compatibility**

For each edge with `port_matches`:
- Check that the source's output port type is compatible with the target's
  input port type (e.g., `bind/let` output can feed `bind/let` or
  `constrain/such-that` input, but not `quant/existential`)
- Use the discourse_signature from the reference to validate allowed
  port type pairings

Score: fraction of port matches with compatible types.

**Check 3: IATC-discourse alignment**

For each edge with an `iatc` type:
- An `assert` edge should connect from a node with `wire/consequential`
  or `wire/causal` discourse elements
- A `challenge` edge should connect from a node with interrogative scopes
- A `clarify` edge should connect from a node with `wire/elaborative`
  discourse elements
- An `exemplify` edge should connect from a node with `env/example` or
  example-like content

Score: fraction of edges where IATC type aligns with source node's
discourse records.

**Check 4: Reference completeness**

For each node claiming a categorical pattern:
- Count how many of the pattern's `required_links` appear in the node's
  text or NER terms
- Count how many of the pattern's `typical_links` appear
- Compute a completeness score: `(required_found / required_total) * 0.7 +
  (typical_found / typical_total) * 0.3`

Score: mean completeness across all categorical annotations.

### Output format

```json
{
  "wiring_id": "problem4",
  "timestamp": "2026-02-15T...",
  "summary": {
    "edges_checked": 41,
    "categorical_consistent": 35,
    "port_compatible": 8,
    "iatc_aligned": 29,
    "completeness_mean": 0.72,
    "overall_score": 0.78
  },
  "edge_reports": [
    {
      "edge": {"source": "step-2", "target": "step-1"},
      "checks": {
        "categorical": {"pass": true, "detail": "both claim cat/adjunction"},
        "ports": {"pass": true, "detail": "bind/let → bind/let"},
        "iatc": {"pass": false, "detail": "assert edge but no consequential wire in source"},
        "completeness": {"source": 0.8, "target": 0.6}
      }
    }
  ],
  "node_reports": [
    {
      "node": "step-1",
      "categorical": ["cat/adjunction"],
      "completeness": 0.8,
      "missing_required": ["triangle identity"],
      "missing_typical": ["adjunction counit"]
    }
  ]
}
```

### Tests

Create `tests/test_ct_verifier.py` with:

1. **TestCategoricalConsistency**: synthetic 3-node graph, verify consistent
   and inconsistent edges are correctly classified
2. **TestPortCompatibility**: synthetic port pairs, verify type checking
3. **TestIATCAlignment**: edges with known IATC types, verify discourse match
4. **TestCompleteness**: nodes with partial reference coverage, verify scoring
5. **TestIntegrationProblem4**: run on actual problem4-wiring.json, verify
   it produces a report without errors and the scores are in [0, 1]

### Stretch goal: live verification mode

```
python scripts/ct-verifier.py live \
    --reference data/nlab-ct-reference.json \
    --thread-wiring data/thread-wiring/
```

In live mode, the verifier watches a directory for new wiring diagrams
(from an ongoing proof session) and produces verification reports as they
arrive. This is the "just-in-time verifier" that would work alongside a
future First Proof exercise.

## Available data

| File | Size | Description |
|------|------|-------------|
| `data/nlab-ct-reference.json` | 178 KB | CT reference: 8 patterns, 20K+ pages |
| `data/thread-wiring/mathoverflow.net__category-theory.json` | 1.1 MB | 50 MO-CT thread wirings |
| `data/thread-wiring/math.stackexchange.com__category-theory.json` | 0.9 MB | 50 SE-CT thread wirings |
| `data/thread-wiring/mathoverflow.net__mathematical-physics.json` | 1.2 MB | 50 MO-MP thread wirings |
| `data/thread-wiring/math.stackexchange.com__mathematical-physics.json` | 0.5 MB | 50 SE-MP thread wirings |
| `data/first-proof/problem4-wiring.json` | 83 KB | First Proof problem 4 wiring |
| `data/ct-evaluation-2x2.json` | 11 KB | Evaluator results (accepted-answer signal) |

## Dependencies

- Python 3.12+, no GPU needed
- Uses `scripts/nlab-wiring.py` for `detect_scopes()`, `detect_wires()`,
  `detect_ports()`, `detect_labels()`, `spot_terms()`, `load_ner_kernel()`
- Uses `scripts/assemble-wiring.py` for `detect_categorical_for_se()`,
  `extract_ports()`, `match_ports()`
- NER kernel at `data/ner-kernel/terms.tsv`

## Success criteria

1. All tests pass
2. Running on problem4-wiring.json produces a meaningful report (not all
   pass, not all fail — the interesting part is which edges fail and why)
3. Running on thread-wiring data shows CT threads score higher than MP
   threads on categorical consistency and completeness
4. The verification report is JSON-serializable and human-readable
