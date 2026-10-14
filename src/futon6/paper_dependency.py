"""Claim-level dependency review layered over the classical paper graph.

The review JSON has the shape ``{"paper", "built_at", "nodes", "edges",
"unresolved_refs", "counts", "verdicts"}``.  Emacs may append verdicts as
``{"edge_id": ..., "verdict": "right"|"wrong", "at": ...}``; this module
never writes verdicts itself.

KnowTeX D1 is read from existing proof derivation edges.  D2 adds dashed
statement-reference edges.  D3 applies an explicit proof-title reference as
an override of the positional proof association; the H-rules belong to later
packets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REF_RE = re.compile(
    r"\\(?P<command>ref|cref|Cref|eqref|autoref)\s*\{(?P<label>[^}]+)\}"
)
_NODE_TYPES = {"claim", "definition", "proof"}


def _stable_edge_id(rule: str, source: str, target: str) -> str:
    """Return a short deterministic id for one rule/source/target triple."""

    raw = f"{rule}\0{source}\0{target}".encode("utf-8")
    return f"{rule.lower()}-{hashlib.sha256(raw).hexdigest()[:12]}"


def _line(text: str, offset: int) -> int:
    """Return the one-based source line containing a character offset."""

    return text.count("\n", 0, max(0, offset)) + 1


def _location(text: str | None, offset: int | None,
              source_files: Sequence[Mapping[str, Any]] = ()) -> int | dict[str, Any] | None:
    """Encode a deterministic offset, with a line when source text is known."""

    if offset is None:
        return None
    for source in source_files:
        start = int(source.get("start", -1))
        end = int(source.get("end", -1))
        if start <= offset < end:
            relative = offset - start
            fragment = text[start:offset] if text is not None else ""
            return {"file": source.get("file", ""), "line": _line(fragment, len(fragment)) + 0}
    if text is None:
        return offset
    return {"char_offset": offset, "line": _line(text, offset)}


def _node_record(node: Mapping[str, Any]) -> dict[str, Any]:
    """Project a classical node into the review-node schema."""

    attrs = node.get("attrs") or {}
    return {
        "id": node.get("id"),
        "type": node.get("type"),
        "env": attrs.get("env", node.get("subtype", node.get("type"))),
        "number": attrs.get("number"),
        "label": attrs.get("label"),
        "section": attrs.get("section"),
        "char_span": attrs.get("char_span"),
    }


def _blocks(blocks_or_text: Any, text: str | None) -> tuple[list[Any], str | None]:
    """Normalize text or a parsed block sequence for the D2 scanner."""

    if isinstance(blocks_or_text, str):
        from .paper_hypergraph import parse_latex_blocks

        parsed, _ = parse_latex_blocks(blocks_or_text)
        return parsed, blocks_or_text
    return list(blocks_or_text or []), text


def _tarjan(nodes: Iterable[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    """Return strongly connected components in deterministic traversal order."""

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(adjacency.get(node, ())):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == node:
                    break
            components.append(sorted(component))

    for node in sorted(set(nodes)):
        if node not in indices:
            visit(node)
    return components


def build_dependency_graph(hg: Mapping[str, Any], blocks_or_text: Any,
                           *, source_files: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Build the D1/D2 review graph over one classical hypergraph.

    ``blocks_or_text`` may be the text used to create ``hg`` or the result of
    ``paper_hypergraph.parse_latex_blocks``.  D2 scans claim/definition blocks
    only, so proof references are not accidentally reclassified as statement
    references.
    """

    source_text = blocks_or_text if isinstance(blocks_or_text, str) else None
    blocks, source_text = _blocks(blocks_or_text, source_text)
    source_nodes = {
        node["id"]: node for node in hg.get("nodes", [])
        if node.get("type") in _NODE_TYPES and node.get("id")
    }
    labels = {
        node.get("attrs", {}).get("label"): node_id
        for node_id, node in source_nodes.items()
        if node.get("attrs", {}).get("label")
    }
    review_edges: dict[str, dict[str, Any]] = {}
    unresolved: Counter[str] = Counter()

    def add_edge(rule: str, source: str, target: str, style: str,
                 location: Any, evidence: Any,
                 attrs: Mapping[str, Any] | None = None) -> None:
        edge_id = _stable_edge_id(rule, source, target)
        record = {
            "id": edge_id,
            "source": source,
            "target": target,
            "rule": rule,
            "style": style,
            "location": location,
            "evidence": evidence,
        }
        if attrs is not None:
            record["attrs"] = dict(attrs)
        review_edges.setdefault(edge_id, record)

    # Index proof blocks by their hypergraph node.  The classical arm stores
    # the span on both objects, while deliberately leaving proof titles off
    # the node, so the title must be read from the parsed block here.
    blocks_by_span = {
        tuple(getattr(block, "span", ())): block
        for block in blocks
        if getattr(block, "env", None) == "proof"
    }
    proof_headers: dict[str, dict[str, Any]] = {}
    for proof_id, proof_node in source_nodes.items():
        if proof_node.get("type") != "proof":
            continue
        span = tuple((proof_node.get("attrs") or {}).get("char_span", ()))
        block = blocks_by_span.get(span)
        title = getattr(block, "title", None) if block is not None else None
        match = _REF_RE.search(title or "")
        if match is None:
            continue
        label = match.group("label").strip()
        referenced = labels.get(label)
        if referenced is None or source_nodes.get(referenced, {}).get("type") != "claim":
            unresolved["proof-header"] += 1
            continue
        proof_headers[proof_id] = {
            "title": title,
            "label": label,
            "target": referenced,
        }

    # D1: existing proof-level dependency ends.
    derivation_proofs: set[str] = set()
    for edge in hg.get("edges", []):
        if edge.get("type") != "derivation":
            continue
        roles = edge.get("roles") or {}
        target = next((node_id for node_id, role in roles.items()
                       if role == "target" and node_id in source_nodes), None)
        if target is None or source_nodes[target].get("type") not in {"claim", "definition"}:
            continue
        proof_id = next((node_id for node_id, role in roles.items()
                         if role == "proof_of_target" and node_id in source_nodes), None)
        if proof_id is not None:
            derivation_proofs.add(proof_id)
        header = proof_headers.get(proof_id)
        edge_target = target
        rule = "D1"
        header_attrs: dict[str, Any] | None = None
        if header is not None:
            edge_target = header["target"]
            if edge_target != target:
                rule = "D3"
                displaced: str | None = target
            else:
                displaced = None
                header_attrs = {"header-confirmed": True}
            header_attrs = {
                "title": header["title"],
                "label": header["label"],
                "displaced": displaced,
                **(header_attrs or {}),
            }
        for source, role in roles.items():
            if role != "depends_on" or source not in source_nodes:
                continue
            if source_nodes[source].get("type") not in {"claim", "definition"}:
                continue
            add_edge(rule, source, edge_target, "solid", edge.get("attrs", {}).get("section"),
                     {"kind": "derivation", "edge_type": edge.get("type")}, header_attrs)

    # A proof can be parsed without a classical derivation edge when it
    # follows another proof (the positional pass has no current claim).  A
    # resolved D3 title still gives that proof an explicit claim association;
    # use its claim/definition references as the proof-level source and leave
    # the displaced positional claim null.
    for proof_id, header in proof_headers.items():
        if proof_id in derivation_proofs:
            continue
        proof_node = source_nodes[proof_id]
        span = tuple((proof_node.get("attrs") or {}).get("char_span", ()))
        block = blocks_by_span.get(span)
        if block is None:
            continue
        for match in _REF_RE.finditer(getattr(block, "body", "") or ""):
            source = labels.get(match.group("label").strip())
            if source is None or source == header["target"]:
                continue
            if source_nodes.get(source, {}).get("type") not in {"claim", "definition"}:
                continue
            add_edge(
                "D3", source, header["target"], "solid",
                _location(source_text, span[0] + match.start(), source_files),
                {"kind": "derivation", "edge_type": "derivation"},
                {"title": header["title"], "label": header["label"], "displaced": None},
            )

    # D2: references in statement blocks, resolved only to claim/definition nodes.
    for block in blocks:
        if getattr(block, "env", None) not in {"theorem", "lemma", "proposition", "corollary", "definition"}:
            continue
        source = None
        for node_id, node in source_nodes.items():
            attrs = node.get("attrs") or {}
            if attrs.get("char_span") == list(getattr(block, "span", ())) \
                    or attrs.get("char_span") == getattr(block, "span", None):
                source = node_id
                break
        if source is None:
            env = getattr(block, "env", "")
            number = getattr(block, "number", 0)
            candidate_type = "definition" if env == "definition" else "claim"
            source = f"{candidate_type}:{'def-' if candidate_type == 'definition' else env + '-'}{number}"
        if source not in source_nodes:
            continue
        body = getattr(block, "body", "") or ""
        for match in _REF_RE.finditer(body):
            label = match.group("label").strip()
            target = labels.get(label)
            if target is None:
                unresolved[label.split(":", 1)[0] if ":" in label else "bare"] += 1
                continue
            add_edge("D2", target, source, "dashed",
                     _location(source_text, getattr(block, "span", (0, 0))[0] + match.start(), source_files),
                     {"kind": "statement-ref", "command": match.group("command"), "label": label})

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in review_edges.values():
        adjacency[edge["source"]].add(edge["target"])
    components = _tarjan(source_nodes, adjacency)
    cyclic_nodes = {node for component in components if len(component) > 1 for node in component}
    for edge in review_edges.values():
        edge["cycle"] = edge["source"] in cyclic_nodes and edge["target"] in cyclic_nodes

    ordered_edges = [review_edges[key] for key in sorted(review_edges)]
    return {
        "nodes": [_node_record(source_nodes[key]) for key in sorted(source_nodes)],
        "edges": ordered_edges,
        "unresolved_refs": dict(sorted(unresolved.items())),
        "counts": dict(sorted(Counter(edge["rule"] for edge in ordered_edges).items())),
        "scc_count": len(components),
        "cycle_scc_count": sum(len(component) > 1 for component in components),
        "verdicts": [],
    }


def write_dependency_review(path: str | Path, paper: str,
                            graph: Mapping[str, Any]) -> Path:
    """Write one review JSON, adding its paper and UTC build stamp."""

    output = dict(graph)
    output["paper"] = paper
    output["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination
