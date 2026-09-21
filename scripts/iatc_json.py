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


SYMBOL_RE = re.compile(r"symbol:(\S+)")
DEFINIENDUM_RE = re.compile(r"definiendum #\d+: \$(.+?)\$")


def bound_symbols(candidate: dict) -> list[str]:
    """The symbols S1 already bound to a meaning in this proof's window.

    S1 emits these as enrichment rows — `bind/typed · symbol:\\alpha |
    type:\\Sigma^{-1}A\\rightarrow X`, `definiendum #0: $\\T$` — and they are
    rendered into the S3 prompt as advice. They are advice only: the node schema
    has one free `text` string and no slot for a symbol, so the model retypes the
    mathematics by hand. In the 0919b probe that is where the damage happened —
    361 of 1280 nodes carried control characters from LaTeX the JSON writer never
    escaped, and 87.5% of those were in proofs whose bindings were already
    present. Returning them here lets the schema name what the node may refer to.
    """
    found: list[str] = []
    for row in candidate.get("enrichment") or ():
        tip = row.get("tip", "")
        for match in (SYMBOL_RE.search(tip), DEFINIENDUM_RE.search(tip)):
            if match:
                symbol = match.group(1).strip().strip("$")
                if symbol and symbol not in found:
                    found.append(symbol)
    return found


def nodes_schema(lo: int, hi: int, symbols: "list[str] | tuple[str, ...]" = (),
                 spans: "list[str] | tuple[str, ...]" = ()) -> dict:
    """Phase 1: what the proof talks about, in the order the model lists it.

    Two things the model must not retype, because a JSON string grammar cannot
    spell them. It permits only " \\ / b f n r t u after a backslash, so a
    command starting with any other letter is unwritable: the decoder forces a
    legal letter and the model completes a different command. Measured over the
    0919b probe — 8,437 backslash sequences, none illegal, 5,891 of them \\t —
    that is how \\T, \\Sigma and \\alpha all became \\triangle, and how a
    source \\in became \\notin.

    So the mathematics arrives by reference:

      symbols      which bound objects the node is about, from what S1 bound
      quote_spans  which S1-marked clause units carry it, by span id

    `text` survives as a PROSE GLOSS and is no longer the record of what the node
    claims; quoted_source() reads that from the source. This is the move
    steps_schema makes with premises, applied twice more: naming something the
    window never supplied is unrepresentable rather than rejected afterwards.
    """
    line = {"type": "integer", "minimum": lo, "maximum": hi}
    properties = {"kind": {"type": "string", "enum": list(NODE_KINDS)},
                  "text": {"type": "string", "minLength": 1, "maxLength": 300,
                           "description": "Prose gloss. NOT the node's mathematics: "
                                          "quote_spans carries that. Do not retype formulae."},
                  "citation": {"type": "string", "maxLength": 160},
                  "first_line": line, "last_line": line}
    required = ["kind", "text", "citation", "first_line", "last_line"]
    if symbols:
        properties["symbols"] = {"type": "array", "maxItems": len(symbols),
                                 "items": {"type": "string", "enum": list(symbols)}}
        required.append("symbols")
    if spans:
        # Clause-sized units, not lines. A node cannot select a hypothesis and its
        # conclusion together unless S1 marked them as one unit, so the vacuous
        # "X implies X" edge becomes unrepresentable rather than detectable.
        properties["quote_spans"] = {"type": "array", "minItems": 1, "maxItems": len(spans),
                                     "items": {"type": "string", "enum": list(spans)}}
        required.append("quote_spans")
    node = {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}
    return {"type": "object", "additionalProperties": False, "required": ["nodes"],
            "properties": {"nodes": {"type": "array", "minItems": 2, "maxItems": MAX_NODES, "items": node}}}


SPAN_KINDS = ("bind/let", "assume/explicit", "quant/universal", "constrain/relation",
              "bind/typed", "definiendum", "definiens", "let-binder",
              "constrain/such-that", "env/proposition", "env/proof")


def source_spans(candidate: dict) -> list[dict]:
    """The clause-sized units S1 already marked, in source order.

    Line numbers were the wrong granularity and are gone. A LaTeX line routinely
    carries a hypothesis AND the conclusion drawn from it -- 0705.0102 line 622
    holds two hypotheses and a preenvelope claim -- so three different nodes could
    only cite the same line, and the edge between them read as "this text implies
    this same text". 43 of 648 edges in the first clean corpus (6.6%) did exactly
    that. S1 had already segmented that line into bind/let, assume/explicit and
    quant/universal units with character offsets; the contract simply never
    offered them.
    """
    return [dict(sp, id=f"s{i}") for i, sp in enumerate(candidate.get("spans") or (), 1)]


def spans_of(candidate: dict) -> list[str]:
    return [sp["id"] for sp in source_spans(candidate)]


def quoted_source(node: dict, candidate: dict) -> str:
    """The node's mathematics, taken from the source rather than from the model.

    The model chooses WHICH spans; code does the extraction, so the LaTeX never
    passes through a JSON string and a grammar that cannot spell \\Sigma cannot
    corrupt it.
    """
    wanted = set(node.get("quote_spans") or ())
    by_id = {sp["id"]: sp for sp in source_spans(candidate)}
    return "\n".join(by_id[i]["text"] for i in
                      sorted(wanted, key=lambda x: by_id[x]["start"]) if i in by_id)


def steps_schema(lo: int, hi: int, node_count: int) -> dict:
    """Phase 2: for each node the proof derives, how it is derived.

    Two things are structural here rather than checked afterwards. The premises of
    a derivation are drawn from the nodes that now exist, which is why this is a
    second call (3 of 20 proofs in the first live run cited nodes never written).
    And each node's premises are drawn from the OTHER nodes, so a node cannot be
    derived from itself: that self-loop was every remaining rejection in the third
    and fourth live runs. Multiple derivations of one node stay expressible (5.4%
    of concluded nodes in the 98-graph corpus have more than one).
    """
    line = {"type": "integer", "minimum": lo, "maximum": hi}
    derivations = {}
    for node in range(1, max(node_count, 1) + 1):
        others = [n for n in range(1, max(node_count, 1) + 1) if n != node]
        derivation = {"type": "object", "additionalProperties": False,
                      "required": ["relation", "premises", "warrant_kind", "warrant",
                                   "first_line", "last_line"],
                      "properties": {"relation": {"type": "string", "enum": list(RELATIONS)},
                                     "premises": {"type": "array", "minItems": 1, "maxItems": MAX_PREMISES,
                                                  "items": {"type": "integer", "enum": others or [node]}},
                                     "warrant_kind": {"type": "string", "enum": list(WARRANT_KINDS)},
                                     "warrant": {"type": "string", "minLength": 1, "maxLength": 240},
                                     "first_line": line, "last_line": line}}
        derivations[str(node)] = {"type": "array", "minItems": 1, "maxItems": 3, "items": derivation}
    return {"type": "object", "additionalProperties": False, "required": ["derivations"],
            "properties": {"derivations": {"type": "object", "additionalProperties": False,
                                           "required": [], "properties": derivations}}}


def steps_of(doc: dict) -> list[dict]:
    """The derivations as steps, in node order — the form the rest of the code uses."""
    steps = []
    for node, entries in sorted((doc.get("derivations") or {}).items(), key=lambda kv: int(kv[0])):
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict):
                steps.append({**entry, "conclusion": int(node)})
    return steps


def problems(doc, lo: int, hi: int) -> list[str]:
    """Contract violations a JSON schema cannot express (empty = acceptable)."""
    found: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("nodes"), list) or not isinstance(doc.get("steps"), list):
        return ["output is not an object with nodes and derivations (the endpoint did not enforce the schema)"]
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
    derives: dict[int, set] = {}      # premise node -> nodes derived from it
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
        # No rule on the conclusion's kind. A construction step establishes the object
        # it builds, and a proof establishes the paper's own labelled statement, which
        # the model records as a ref carrying that label (`Theorem~\ref{StrongYone}`).
        # Both were rejected by earlier versions of this contract; neither is a defect.
        for p in premises:
            derives.setdefault(p, set()).add(conclusion)
        a, b = s.get("first_line"), s.get("last_line")
        if not (isinstance(a, int) and isinstance(b, int) and lo <= a <= b <= hi):
            found.append(f"{label}: lines {a}-{b} not an ordered range inside {lo}-{hi}")
        if not str(s.get("warrant", "")).strip() or (s["warrant_kind"] == "missing" and not slug(s["warrant"])):
            found.append(f"{label}: empty warrant")
    cycle = find_cycle(derives)
    if cycle:
        found.append("the steps derive " + " -> ".join(f"node {n}" for n in cycle)
                     + ", so the argument assumes what it proves; an equivalence is one iff step")
    return found


def find_cycle(derives: dict[int, set]) -> list:
    """A cycle in premise -> conclusion, or [].

    Acyclicity is the real requirement, not the order the steps are listed in: a
    proof may state its conclusion and justify it afterwards, which an earlier
    version of this contract rejected (0708.1921__p7 in the second live run).
    """
    state: dict[int, int] = {}
    def walk(node, path):
        state[node] = 1
        for nxt in sorted(derives.get(node, ())):
            if state.get(nxt) == 1:
                return path + [node, nxt]
            if state.get(nxt, 0) == 0:
                found_here = walk(nxt, path + [node])
                if found_here:
                    return found_here
        state[node] = 2
        return []
    for start in sorted(derives):
        if state.get(start, 0) == 0:
            cycle = walk(start, [])
            if cycle:
                return cycle
    return []


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
        # :text carries the SOURCE when the node quoted it, because every
        # downstream reader — semcheck, the substance gate, term extraction —
        # reads :text and would otherwise be reading the model's prose. The
        # model's own wording is kept as :gloss, where nothing depends on it.
        quoted = quoted_source(n, cand)
        text = quoted if quoted else n["text"].strip()
        fields = [f":id :n{i}", f":kind :{n['kind']}", f":text {edn_string(text)}"]
        if quoted:
            fields.append(f":gloss {edn_string(n['text'].strip())}")
            fields.append(":quoted true")
            fields.append(f":quote-spans [{' '.join(edn_string(x) for x in n['quote_spans'])}]")
        if n.get("symbols"):
            fields.append(f":symbols [{' '.join(edn_string(x) for x in n['symbols'])}]")
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
