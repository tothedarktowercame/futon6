#!/usr/bin/env python3
"""Enrich proof wiring diagrams with discourse, ports, and categorical signals."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import importlib

assemble_wiring = importlib.import_module("assemble-wiring")
ct_verifier = importlib.import_module("ct-verifier")
nlab_wiring = importlib.import_module("nlab-wiring")


PROOF_WIRE_MARKERS = [
    (re.compile(r"\[(?:ERROR|FAILED)\]", re.IGNORECASE), "wire/adversative"),
    (re.compile(r"\[(?:PROVED)\]", re.IGNORECASE), "wire/consequential"),
    (re.compile(r"\b(?:COMPLETE|QED)\b", re.IGNORECASE), "wire/consequential"),
    (re.compile(r"\[(?:PENDING)\]", re.IGNORECASE), "wire/tentative"),
    (re.compile(r"\bWLOG\b", re.IGNORECASE), "constrain/such-that"),
    (
        re.compile(
            r"\b(?:therefore|thus|hence|implies|shows|gives|proves|verified|verification)\b",
            re.IGNORECASE,
        ),
        "wire/consequential",
    ),
    (
        re.compile(r"\b(?:namely|i\.e\.|that is|more precisely|equivalently)\b", re.IGNORECASE),
        "wire/clarifying",
    ),
]

EQUATION_ASSIGN_RE = re.compile(
    r"(?P<lhs>[A-Za-z0-9_][A-Za-z0-9_/\^\-\+\*\(\)]{0,40})\s*(?<![<>])=(?!=)\s*(?P<rhs>[^.;\n]{2,120})"
)
EQUATION_INEQ_RE = re.compile(
    r"(?P<expr>[A-Za-z0-9_][A-Za-z0-9_/\^\-\+\*\(\)\s]{0,80})\s*(?P<op>>=|<=|≥|≤)\s*(?P<rhs>0|[A-Za-z0-9_][^.;\n]{0,40})"
)


def _build_discourse(node_id: str, text: str) -> list[dict[str, Any]]:
    discourse: list[dict[str, Any]] = []
    discourse.extend(nlab_wiring.detect_scopes(node_id, text))
    discourse.extend(nlab_wiring.detect_wires(node_id, text))
    discourse.extend(nlab_wiring.detect_ports(node_id, text))
    discourse.extend(nlab_wiring.detect_labels(node_id, text))

    marker_idx = 0
    for pattern, wire_type in PROOF_WIRE_MARKERS:
        for m in pattern.finditer(text):
            discourse.append(
                {
                    "hx/id": f"{node_id}:proof-wire-{marker_idx:03d}",
                    "hx/role": "wire",
                    "hx/type": wire_type,
                    "hx/content": {"match": m.group(), "position": m.start()},
                    "hx/labels": ["wire", wire_type.split("/")[-1]],
                }
            )
            marker_idx += 1

    if ":" in text:
        discourse.append(
            {
                "hx/id": f"{node_id}:proof-wire-{marker_idx:03d}",
                "hx/role": "wire",
                "hx/type": "wire/clarifying",
                "hx/content": {"match": ":", "position": text.find(":")},
                "hx/labels": ["wire", "clarifying"],
            }
        )
        marker_idx += 1

    discourse.sort(key=lambda row: row.get("hx/content", {}).get("position", 0))
    return discourse


def _extract_equation_ports(
    node_id: str,
    text: str,
    in_start: int,
    out_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_ports: list[dict[str, Any]] = []
    output_ports: list[dict[str, Any]] = []
    extra_discourse: list[dict[str, Any]] = []

    in_idx = in_start
    out_idx = out_start
    disc_idx = 0

    for m in EQUATION_ASSIGN_RE.finditer(text):
        label = f"{m.group('lhs').strip()} = {m.group('rhs').strip()}"
        in_port = {
            "id": f"{node_id}:eq-in-{in_idx:03d}",
            "type": "bind/let",
            "label": label[:140],
            "text": m.group(0)[:120],
            "position": m.start(),
        }
        out_port = {
            "id": f"{node_id}:eq-out-{out_idx:03d}",
            "type": "bind/let",
            "label": label[:140],
            "text": m.group(0)[:120],
            "position": m.start(),
        }
        input_ports.append(in_port)
        output_ports.append(out_port)
        in_idx += 1
        out_idx += 1
        extra_discourse.append(
            {
                "hx/id": f"{node_id}:eq-scope-{disc_idx:03d}",
                "hx/role": "component",
                "hx/type": "bind/let",
                "hx/content": {"match": m.group(0)[:120], "position": m.start()},
                "hx/labels": ["scope", "equation"],
            }
        )
        disc_idx += 1

    for m in EQUATION_INEQ_RE.finditer(text):
        label = f"{m.group('expr').strip()} {m.group('op')} {m.group('rhs').strip()}"
        in_port = {
            "id": f"{node_id}:eq-in-{in_idx:03d}",
            "type": "constrain/such-that",
            "label": label[:140],
            "text": m.group(0)[:120],
            "position": m.start(),
        }
        out_port = {
            "id": f"{node_id}:eq-out-{out_idx:03d}",
            "type": "constrain/such-that",
            "label": label[:140],
            "text": m.group(0)[:120],
            "position": m.start(),
        }
        input_ports.append(in_port)
        output_ports.append(out_port)
        in_idx += 1
        out_idx += 1
        extra_discourse.append(
            {
                "hx/id": f"{node_id}:eq-constrain-{disc_idx:03d}",
                "hx/role": "component",
                "hx/type": "constrain/such-that",
                "hx/content": {"match": m.group(0)[:120], "position": m.start()},
                "hx/labels": ["scope", "inequality"],
            }
        )
        disc_idx += 1

    # Explicit proof-completion signals become consequential output ports.
    for pattern in (
        re.compile(r"\[(?:PROVED)\]", re.IGNORECASE),
        re.compile(r"\bCOMPLETE\b", re.IGNORECASE),
        re.compile(r"\bQED\b", re.IGNORECASE),
    ):
        for m in pattern.finditer(text):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 70)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            output_ports.append(
                {
                    "id": f"{node_id}:eq-out-{out_idx:03d}",
                    "type": "wire/consequential",
                    "label": snippet[:140],
                    "text": m.group(0),
                    "position": m.start(),
                }
            )
            out_idx += 1

    return input_ports, output_ports, extra_discourse


def _dedupe_ports(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for p in sorted(ports, key=lambda row: row.get("position", 0)):
        key = (p.get("type"), p.get("label", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def enrich_node(
    node: dict[str, Any],
    reference: dict[str, Any],
    singles: dict[str, tuple[str, str]] | None,
    multi_index: dict[str, list[tuple[str, str, str]]] | None,
) -> dict[str, Any]:
    """Add discourse, ports, and categorical annotations to one proof node."""
    enriched = dict(node)
    node_id = str(enriched.get("id") or enriched.get("post_id"))
    text = str(enriched.get("body_text") or enriched.get("body") or "")
    tags = enriched.get("tags", []) if isinstance(enriched.get("tags"), list) else []

    discourse = _build_discourse(node_id, text)
    input_ports, output_ports = assemble_wiring.extract_ports(text, node_id)
    extra_in, extra_out, extra_discourse = _extract_equation_ports(
        node_id=node_id,
        text=text,
        in_start=len(input_ports),
        out_start=len(output_ports),
    )
    input_ports.extend(extra_in)
    output_ports.extend(extra_out)
    discourse.extend(extra_discourse)
    discourse.sort(key=lambda row: row.get("hx/content", {}).get("position", 0))

    categorical = assemble_wiring.detect_categorical_for_se(text, tags, reference)
    ner_terms = (
        nlab_wiring.spot_terms(text, singles, multi_index)
        if singles is not None and multi_index is not None
        else []
    )

    enriched["discourse"] = discourse
    enriched["input_ports"] = _dedupe_ports(input_ports)
    enriched["output_ports"] = _dedupe_ports(output_ports)
    enriched["categorical"] = categorical
    enriched["ner_terms"] = ner_terms
    enriched["text_length"] = len(text)
    return enriched


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    source = str(edge.get("source") or edge.get("from") or "")
    target = str(edge.get("target") or edge.get("to") or "")
    return source, target


def _merge_matches(primary: list[tuple[str, str, float]], secondary: list[tuple[str, str, float]]) -> list[list[Any]]:
    by_pair: dict[tuple[str, str], float] = {}
    for src, dst, score in primary + secondary:
        key = (src, dst)
        by_pair[key] = max(score, by_pair.get(key, 0.0))
    ranked = sorted(by_pair.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [[src, dst, round(score, 2)] for (src, dst), score in ranked[:12]]


def enrich_edges(
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add port_matches to proof edges based on enriched node ports."""
    enriched_edges = []
    for edge in edges:
        source_id, target_id = _edge_endpoints(edge)
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if not source or not target:
            enriched_edges.append(dict(edge))
            continue

        forward = assemble_wiring.match_ports(
            source.get("output_ports", []),
            target.get("input_ports", []),
            reference,
        )
        merged = _merge_matches(forward, [])

        row = dict(edge)
        row["port_matches"] = merged
        edge_type = str(edge.get("edge_type") or edge.get("type") or "").lower()
        if "iatc" not in row and edge_type:
            row["iatc"] = edge_type
        enriched_edges.append(row)
    return enriched_edges


def enrich_wiring(
    wiring: dict[str, Any],
    reference: dict[str, Any],
    singles: dict[str, tuple[str, str]] | None,
    multi_index: dict[str, list[tuple[str, str, str]]] | None,
) -> dict[str, Any]:
    nodes = [enrich_node(node, reference, singles, multi_index) for node in wiring.get("nodes", [])]
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    edges = enrich_edges(nodes_by_id, wiring.get("edges", []), reference)

    out = dict(wiring)
    out["nodes"] = nodes
    out["edges"] = edges
    out.setdefault("stats", {})
    out["stats"]["n_nodes"] = len(nodes)
    out["stats"]["n_edges"] = len(edges)
    out["stats"]["nodes_with_discourse"] = sum(1 for n in nodes if n.get("discourse"))
    out["stats"]["nodes_with_ports"] = sum(
        1 for n in nodes if n.get("input_ports") or n.get("output_ports")
    )
    out["stats"]["edges_with_port_matches"] = sum(1 for e in edges if e.get("port_matches"))
    return out


def _default_verification_output(output_path: Path) -> Path:
    if output_path.name.endswith("-wiring-enriched.json"):
        return output_path.with_name(output_path.name.replace("-wiring-enriched.json", "-verification-enriched.json"))
    return output_path.with_name(output_path.stem + "-verification.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiring", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, default=None)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    wiring = json.loads(args.wiring.read_text(encoding="utf-8"))
    kernel_tsv = args.ner_kernel / "terms.tsv" if args.ner_kernel.is_dir() else args.ner_kernel
    singles, multi_index, _ = nlab_wiring.load_ner_kernel(kernel_tsv)

    enriched = enrich_wiring(wiring, reference, singles, multi_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    report = ct_verifier.verify_wiring_dict(enriched, reference)
    verification_output = args.verification_output or _default_verification_output(args.output)
    verification_output.parent.mkdir(parents=True, exist_ok=True)
    verification_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report.get("summary", {})
    print(
        f"enriched={args.output} "
        f"verification={verification_output} "
        f"overall_score={summary.get('overall_score', 0):.4f} "
        f"edges_checked={summary.get('edges_checked', 0)}"
    )


if __name__ == "__main__":
    main()
