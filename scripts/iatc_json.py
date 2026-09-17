"""S3 output contract: the model returns schema-constrained JSON; code writes EDN.

The previous contract asked the model to write an EDN argument graph directly. It
could not do so reliably, and the pipeline grew an escape repair, a canonicalising
repair script, retries fed with parser errors, and gates for cycles and missing
conclusions to compensate. Here the model only chooses content:

  nodes  numbered 1..N: kind, text, first/last source line
  steps  in proof order: relation, premise node numbers, conclusion node number,
         warrant kind and text, first/last source line

The JSON schema (enforced by the serving stack, checked by conformance) fixes the
shape, the vocabularies, and the line range. What a schema cannot express is
checked here and rejects the item with reasons: node references exist, a step does
not conclude its own premise, and a premise concluded by some step is concluded by
an EARLIER step. That ordering rule makes premise->conclusion cycles impossible in
an accepted graph; an equivalence is one `iff` step.

Code then assigns ids, mirrors missing warrants into :holes, and serialises EDN in
the shape every downstream reader already consumes. Nothing is repaired: an output
that does not satisfy the contract is rejected, and the reason is accounted.
"""
from __future__ import annotations

import re

GENERATOR = "iatc-json/v1"
NODE_KINDS = ("claim", "object", "definition", "ref")
RELATIONS = ("implies", "iff", "by-definition", "by-construction", "by-contradiction",
             "by-cases", "by-induction", "by-computation")
WARRANT_KINDS = {"stated": "claim", "citation": "citation", "missing": "missing-warrant"}
MAX_NODES = 60
MAX_STEPS = 60
MAX_PREMISES = 8


def schema(lo: int, hi: int) -> dict:
    """JSON schema for one proof whose numbered source lines are lo..hi."""
    line = {"type": "integer", "minimum": lo, "maximum": hi}
    node_ref = {"type": "integer", "minimum": 1, "maximum": MAX_NODES}
    node = {"type": "object", "additionalProperties": False,
            "required": ["kind", "text", "citation", "first_line", "last_line"],
            "properties": {"kind": {"type": "string", "enum": list(NODE_KINDS)},
                           "text": {"type": "string", "minLength": 1, "maxLength": 300},
                           "citation": {"type": "string", "maxLength": 160},
                           "first_line": line, "last_line": line}}
    step = {"type": "object", "additionalProperties": False,
            "required": ["relation", "premises", "conclusion", "warrant_kind", "warrant",
                         "first_line", "last_line"],
            "properties": {"relation": {"type": "string", "enum": list(RELATIONS)},
                           "premises": {"type": "array", "maxItems": MAX_PREMISES, "items": node_ref},
                           "conclusion": node_ref,
                           "warrant_kind": {"type": "string", "enum": list(WARRANT_KINDS)},
                           "warrant": {"type": "string", "minLength": 1, "maxLength": 240},
                           "first_line": line, "last_line": line}}
    return {"type": "object", "additionalProperties": False, "required": ["nodes", "steps"],
            "properties": {"nodes": {"type": "array", "minItems": 2, "maxItems": MAX_NODES, "items": node},
                           "steps": {"type": "array", "minItems": 1, "maxItems": MAX_STEPS, "items": step}}}


def problems(doc, lo: int, hi: int) -> list[str]:
    """Contract violations a JSON schema cannot express (empty = acceptable)."""
    found: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("nodes"), list) or not isinstance(doc.get("steps"), list):
        return ["output is not an object with nodes and steps (the endpoint did not enforce the schema)"]
    nodes, steps = doc["nodes"], doc["steps"]
    if len(nodes) < 2 or not steps:
        found.append(f"too small: {len(nodes)} node(s), {len(steps)} step(s)")
    for i, n in enumerate(nodes, 1):
        if not isinstance(n, dict) or n.get("kind") not in NODE_KINDS or not str(n.get("text", "")).strip():
            found.append(f"node {i}: missing kind/text")
            continue
        a, b = n.get("first_line"), n.get("last_line")
        if not (isinstance(a, int) and isinstance(b, int) and lo <= a <= b <= hi):
            found.append(f"node {i}: lines {a}-{b} not an ordered range inside {lo}-{hi}")
    concluded_at: dict[int, int] = {}
    for s_index, s in enumerate(steps):
        if isinstance(s, dict) and isinstance(s.get("conclusion"), int):
            concluded_at.setdefault(s["conclusion"], s_index)
    for s_index, s in enumerate(steps):
        label = f"step {s_index + 1}"
        if not isinstance(s, dict) or s.get("relation") not in RELATIONS or s.get("warrant_kind") not in WARRANT_KINDS:
            found.append(f"{label}: missing relation/warrant kind")
            continue
        premises, conclusion = s.get("premises"), s.get("conclusion")
        refs = (premises if isinstance(premises, list) else [None]) + [conclusion]
        bad = [r for r in refs if not (isinstance(r, int) and 1 <= r <= len(nodes))]
        if bad:
            found.append(f"{label}: refers to node(s) {bad} but there are {len(nodes)} nodes")
            continue
        if conclusion in premises:
            found.append(f"{label}: node {conclusion} is both premise and conclusion")
        if isinstance(nodes[conclusion - 1], dict) and nodes[conclusion - 1].get("kind") not in ("claim", "definition"):
            found.append(f"{label}: conclusion node {conclusion} is a {nodes[conclusion - 1].get('kind')}, not a claim")
        for p in premises:
            if p in concluded_at and concluded_at[p] >= s_index:
                found.append(f"{label}: premise node {p} is only concluded by step {concluded_at[p] + 1}; "
                             "steps must follow the order of the argument (an equivalence is one iff step)")
        a, b = s.get("first_line"), s.get("last_line")
        if not (isinstance(a, int) and isinstance(b, int) and lo <= a <= b <= hi):
            found.append(f"{label}: lines {a}-{b} not an ordered range inside {lo}-{hi}")
        if not str(s.get("warrant", "")).strip() or (s["warrant_kind"] == "missing" and not slug(s["warrant"])):
            found.append(f"{label}: empty warrant")
    return found


def slug(text: str, limit: int = 60) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(text).lower())).strip("-")[:limit].strip("-")


def edn_string(value) -> str:
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + '"'


def to_edn(doc: dict, cand: dict, model: str) -> str:
    """Serialise an acceptable document as an IATC argument graph."""
    lo, hi = cand["window-lines"]
    lines = lambda x: f'{{:lines [{x["first_line"]} {x["last_line"]}]}}'
    nodes, holes = [], []
    for i, n in enumerate(doc["nodes"], 1):
        fields = [f":id :n{i}", f":kind :{n['kind']}", f":text {edn_string(n['text'].strip())}"]
        if n["kind"] == "ref":
            if n.get("citation", "").strip():
                fields.append(f":citation {edn_string(n['citation'].strip())}")
            else:
                holes.append(f"{{:kind :unresolved-ref :node :n{i}}}")
        fields.append(f":source {lines(n)}")
        nodes.append("{" + " ".join(fields) + "}")
    edges = []
    for j, s in enumerate(doc["steps"], 1):
        kind = WARRANT_KINDS[s["warrant_kind"]]
        text = s["warrant"].strip()
        if kind == "missing-warrant":
            wanted = slug(text)
            warrant = f"{{:kind :missing-warrant :wanted :{wanted} :text {edn_string(text)}}}"
            holes.append(f"{{:kind :missing-warrant :edge :e{j} :wanted :{wanted}}}")
        else:
            warrant = f"{{:kind :{kind} :text {edn_string(text)}}}"
        premises = " ".join(f":n{p}" for p in s["premises"])
        edges.append(f"{{:id :e{j} :kind :infer :relation :{s['relation']} :premise [{premises}] "
                     f":conclusion :n{s['conclusion']} :warrant {warrant} :source {lines(s)}}}")
    proved = cand.get("proved")
    provenance = [f":generator {edn_string(GENERATOR)}", f":model {edn_string(model)}",
                  f":proof-lines [{cand['proof-lines'][0]} {cand['proof-lines'][1]}]"]
    if proved:
        provenance.append(f":proved {{:kind :{proved['kind']} :lines [{proved['lines'][0]} {proved['lines'][1]}]}}")
    return ("{:paper/id " + edn_string(cand["paper-id"]) + "\n :passage/id " + edn_string(cand["passage-id"])
            + f"\n :source {{:lines [{lo} {hi}] :kind :proof}}"
            + "\n :provenance {" + " ".join(provenance) + "}"
            + "\n :nodes [" + "\n  ".join(nodes) + "]"
            + "\n :edges [" + "\n  ".join(edges) + "]"
            + "\n :holes [" + "\n  ".join(holes) + "]}\n")
