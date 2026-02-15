# Mission: Enrich Proof Wiring with Discourse, Ports, and Categorical Annotations

## Context

Codex built `scripts/ct-verifier.py` (commit 266a1ae) — a 4-check verifier
for proof wiring diagrams. Running it on `data/first-proof/problem4-wiring.json`
produces an overall_score of 0.006 because the nodes are bare: they have
`body_text` and `edge_type`, but lack the layers the verifier needs:

- **No `discourse` annotations** → IATC alignment check fails on all 41 edges
- **No `input_ports` / `output_ports`** → port compatibility check has nothing
  to compare
- **No `categorical` annotations** → categorical consistency check can't fire
- **No `port_matches` on edges** → port compatibility gets "no port matches"

The wiring *already has* rich `body_text` on every node. The enrichment task
is to run each node through the same annotation pipeline that
`assemble-wiring.py` applies to SE thread nodes, producing the fields the
verifier expects.

## Task for Codex

Build a Python script `scripts/enrich-proof-wiring.py` that:

1. Reads a bare proof wiring diagram (`problem4-wiring.json` format)
2. Reads the CT reference dictionary + NER kernel
3. For each node, adds:
   - `discourse` — scope/wire/port detection via `nlab-wiring.py` functions
   - `input_ports` / `output_ports` — extracted from body_text
   - `categorical` — CT pattern detection (may be sparse for non-CT proofs)
4. For each edge, adds:
   - `port_matches` — term overlap between source output ports and target
     input ports, boosted by CT reference weights
5. Writes the enriched wiring to an output file
6. Re-runs `ct-verifier.py verify` on the enriched output

### Concrete deliverable

```
python scripts/enrich-proof-wiring.py \
    --wiring data/first-proof/problem4-wiring.json \
    --reference data/nlab-ct-reference.json \
    --ner-kernel data/ner-kernel \
    --output data/first-proof/problem4-wiring-enriched.json
```

Then:
```
python scripts/ct-verifier.py verify \
    --wiring data/first-proof/problem4-wiring-enriched.json \
    --reference data/nlab-ct-reference.json \
    --output data/first-proof/problem4-verification-enriched.json
```

The enriched verification score should be meaningfully higher than 0.006.

## Worked examples: before and after

### Example 1: p4-s4a (challenge node with [ERROR] tag)

**Before** (bare):
```json
{
  "id": "p4-s4a",
  "node_type": "comment",
  "post_id": 4041,
  "body_text": "[ERROR] Direction wrong: concavity + f(0)=0 gives SUBadditivity f(a+b) <= f(a)+f(b), not superadditivity. For superadditivity need CONVEXITY + f(0)=0.",
  "score": -1,
  "creation_date": "2026-02-11",
  "parent_post_id": 404
}
```

**After** (enriched):
```json
{
  "id": "p4-s4a",
  "node_type": "comment",
  "post_id": 4041,
  "body_text": "[ERROR] Direction wrong: concavity + f(0)=0 gives SUBadditivity f(a+b) <= f(a)+f(b), not superadditivity. For superadditivity need CONVEXITY + f(0)=0.",
  "score": -1,
  "creation_date": "2026-02-11",
  "parent_post_id": 404,
  "discourse": [
    {"hx/type": "wire/adversative", "hx/content": {"match": "[ERROR] Direction wrong", "position": 0}},
    {"hx/type": "constrain/such-that", "hx/content": {"match": "For superadditivity need CONVEXITY + f(0)=0", "position": 103}}
  ],
  "input_ports": [
    {"id": "p4-s4a:in-0", "type": "bind/let", "label": "concavity + f(0)=0 → subadditivity"},
    {"id": "p4-s4a:in-1", "type": "bind/let", "label": "superadditivity claim from p4-s4"}
  ],
  "output_ports": [
    {"id": "p4-s4a:out-0", "type": "constrain/such-that", "label": "need CONVEXITY + f(0)=0 for superadditivity"}
  ],
  "categorical": []
}
```

**Why this helps the verifier:**
- The edge `p4-s4a → p4-s4` has `edge_type: "challenge"`. The verifier's
  IATC check requires adversative/interrogative discourse in the source node.
  Now `discourse` contains `wire/adversative` → **IATC check passes**.
- No categorical annotation (correct — this is about convexity, not CT).
- Ports enable matching: `p4-s4a:in-1` references the superadditivity claim,
  which should match `p4-s4:out-*` ports.

### Example 2: p4-n4-conclusion (assert node summarizing n=4 proof)

**Before** (bare):
```json
{
  "id": "p4-n4-conclusion",
  "node_type": "answer",
  "post_id": 420,
  "body_text": "n=4 proof COMPLETE via Path 2: 3-piece Cauchy-Schwarz decomposition 1/Phi_4 = T1+T2+R. T1_surplus >= 0 by Titu's lemma. T2+R surplus = K_red (487 terms) >= 0 by P-convexity: K_red = A(P,Q)+sqrt(PQ)*B(P,Q), B<=0 (bilinear corners), d²K/dP² >= 0 (two non-negative terms), boundary K >= 0 (AM-GM). Bypasses case-by-case critical point analysis entirely.",
  "score": 2,
  "creation_date": "2026-02-12",
  "parent_post_id": 4
}
```

**After** (enriched):
```json
{
  "id": "p4-n4-conclusion",
  "node_type": "answer",
  "post_id": 420,
  "body_text": "n=4 proof COMPLETE via Path 2: 3-piece Cauchy-Schwarz decomposition 1/Phi_4 = T1+T2+R. T1_surplus >= 0 by Titu's lemma. T2+R surplus = K_red (487 terms) >= 0 by P-convexity: K_red = A(P,Q)+sqrt(PQ)*B(P,Q), B<=0 (bilinear corners), d²K/dP² >= 0 (two non-negative terms), boundary K >= 0 (AM-GM). Bypasses case-by-case critical point analysis entirely.",
  "score": 2,
  "creation_date": "2026-02-12",
  "parent_post_id": 4,
  "discourse": [
    {"hx/type": "wire/consequential", "hx/content": {"match": "n=4 proof COMPLETE", "position": 0}},
    {"hx/type": "bind/let", "hx/content": {"match": "1/Phi_4 = T1+T2+R", "position": 67}},
    {"hx/type": "constrain/such-that", "hx/content": {"match": "T1_surplus >= 0 by Titu's lemma", "position": 87}},
    {"hx/type": "constrain/such-that", "hx/content": {"match": "K_red >= 0 by P-convexity", "position": 130}}
  ],
  "input_ports": [
    {"id": "p4-n4-conclusion:in-0", "type": "bind/let", "label": "1/Phi_4 = T1+T2+R"},
    {"id": "p4-n4-conclusion:in-1", "type": "bind/let", "label": "K_red = A(P,Q)+sqrt(PQ)*B(P,Q)"},
    {"id": "p4-n4-conclusion:in-2", "type": "bind/let", "label": "B<=0 (bilinear corners)"},
    {"id": "p4-n4-conclusion:in-3", "type": "bind/let", "label": "d²K/dP² >= 0"}
  ],
  "output_ports": [
    {"id": "p4-n4-conclusion:out-0", "type": "wire/consequential", "label": "n=4 proof COMPLETE via Path 2"}
  ],
  "categorical": []
}
```

**Why this helps the verifier:**
- This node is the source of 7 `assert` and `reference` edges. The verifier's
  IATC check for `assert` requires consequential/causal discourse in the
  source. Now `discourse` contains `wire/consequential` → **IATC passes**.
- The 4 input ports reference specific results from earlier nodes. Port
  matching against those nodes' output ports will produce nonzero overlap.
- This node has 6 `reference` edges to case nodes (p4-n4-boundary through
  p4-n4-case3c). Each case node's output (its proved result) should match
  one of this node's input ports via NER term overlap.

### Enriched edge example

The edge `p4-n4-conclusion → p4-n4-case1` is currently bare:
```json
{
  "source": "p4-n4-conclusion",
  "target": "p4-n4-case1",
  "edge_type": "reference",
  "evidence": "Case 1 algebraically exact",
  "detection": "structural"
}
```

After enrichment, it gets `port_matches`:
```json
{
  "source": "p4-n4-conclusion",
  "target": "p4-n4-case1",
  "edge_type": "reference",
  "evidence": "Case 1 algebraically exact",
  "detection": "structural",
  "port_matches": [
    ["p4-n4-conclusion:in-0", "p4-n4-case1:out-0", 2]
  ]
}
```

The `2` is the match score (term overlap count, potentially boosted by CT
reference weights).

## Implementation approach

### Core function

```python
def enrich_node(node, reference, singles, multi_index):
    """Add discourse, ports, and categorical annotations to a proof node.

    Uses the same functions as assemble-wiring.py:
    - nlab_wiring.detect_scopes(text) → scope records
    - nlab_wiring.detect_wires(text) → wire records
    - nlab_wiring.detect_ports(scopes, wires) → input/output ports
    - assemble_wiring.detect_categorical_for_se(text, reference, singles, multi_index)
    - assemble_wiring.extract_ports(node_dict) (may need thin adapter)

    Returns enriched node dict.
    """
```

### Edge enrichment

```python
def enrich_edges(nodes_by_id, edges, reference):
    """Add port_matches to edges based on enriched node ports.

    Uses assemble_wiring.match_ports(source_node, target_node, reference)
    or equivalent.
    """
```

### Proof-specific adaptations

The body_text in proof wirings is denser than SE posts — compressed proof
notes rather than conversational prose. Two adjustments:

1. **Discourse detection relaxation**: The proof text uses markers like
   "[ERROR]", "[FAILED]", "[PROVED]", "[PENDING]", "QED", "WLOG" that
   don't appear in SE posts. Add these to the wire detection patterns:
   - `[ERROR]`, `[FAILED]` → `wire/adversative`
   - `[PROVED]`, `COMPLETE`, `QED` → `wire/consequential`
   - `[PENDING]` → `wire/tentative` (or skip)
   - `WLOG` → `constrain/such-that`

2. **Port extraction from equations**: The body_text is rich in equations
   like `1/Phi_4 = T1+T2+R`. These should become `bind/let` ports. A
   simple regex for `<symbol> = <expression>` and `<expression> >= 0`
   patterns will capture most of them.

### What NOT to do

- Don't try to force categorical annotations where they don't fit. Problem 4
  is about polynomial inequalities and free convolution — not category theory.
  Most nodes will have `categorical: []` and that's correct.
- Don't modify ct-verifier.py. The verifier is correct; the input was starved.
- Don't modify the original problem4-wiring.json. Write enriched output to a
  new file.

## Dependencies

- `scripts/nlab-wiring.py` — `detect_scopes()`, `detect_wires()`,
  `detect_ports()`, `spot_terms()`, `load_ner_kernel()`
- `scripts/assemble-wiring.py` — `detect_categorical_for_se()`,
  `extract_ports()`, `match_ports()`
- `scripts/ct-verifier.py` — for re-verification after enrichment
- CT reference: `data/nlab-ct-reference.json`
- NER kernel: `data/ner-kernel/`

## Tests

Create `tests/test_enrich_proof_wiring.py` with:

1. **TestNodeEnrichment**: Verify p4-s4a gets `wire/adversative` in discourse
2. **TestConsequentialDetection**: Verify nodes with "PROVED"/"QED"/"COMPLETE"
   get `wire/consequential`
3. **TestPortExtraction**: Verify equations become `bind/let` ports
4. **TestEdgePortMatching**: Verify edges between related nodes get port_matches
5. **TestEndToEnd**: Enrich problem4-wiring.json, re-verify with ct-verifier,
   assert overall_score > 0.2 (meaningful improvement over 0.006)

## Success criteria

1. All existing tests still pass (174 tests)
2. New tests pass
3. Enriched problem4 verification score > 0.2 (vs 0.006 bare)
4. IATC alignment specifically: assert edges from nodes with "PROVED"/"QED"
   text should pass, challenge edges from nodes with "[ERROR]"/"[FAILED]"
   should pass
5. Port matches appear on edges where source references target's results

## Available data

| File | Description |
|------|-------------|
| `data/first-proof/problem4-wiring.json` | 28 nodes, 41 edges, bare |
| `data/first-proof/problem4-verification.json` | Current verification (score 0.006) |
| `data/nlab-ct-reference.json` | 8 CT patterns from 20K nLab pages |
| `data/ner-kernel/` | NER terms from PlanetMath |
| `scripts/ct-verifier.py` | The verifier (don't modify) |
| `scripts/assemble-wiring.py` | Node enrichment functions to reuse |
| `scripts/nlab-wiring.py` | Discourse detection functions to reuse |
