#!/usr/bin/env python3
"""CT-backed verifier for proof/thread wiring diagrams.

Checks:
1) Categorical consistency across edges
2) Port type compatibility across edge port matches
3) IATC-discourse alignment
4) Reference completeness for node categorical annotations
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Script-local imports for hyphenated module names.
sys.path.insert(0, str(Path(__file__).parent))
import importlib

assemble_wiring = importlib.import_module("assemble-wiring")
nlab_wiring = importlib.import_module("nlab-wiring")


EXAMPLE_RE = re.compile(
    r"\b(?:for example|for instance|e\.g\.|consider the case|as an example)\b",
    re.IGNORECASE,
)
CLARIFY_RE = re.compile(
    r"\b(?:to clarify|in other words|more precisely|that is)\b",
    re.IGNORECASE,
)
REFERENCE_RE = re.compile(
    r"\b(?:see |cf\.|as shown in|as proved in|according to|theorem)\b",
    re.IGNORECASE,
)
RETRACT_RE = re.compile(
    r"\b(?:retract|mistake|correction|erratum|I was wrong|I stand corrected)\b",
    re.IGNORECASE,
)


@dataclass
class NodeInfo:
    node_id: str
    node_type: str
    text: str
    tags: list[str]
    categorical: list[str]
    discourse_types: set[str]
    input_ports: list[dict[str, Any]]
    output_ports: list[dict[str, Any]]
    ner_terms: set[str]
    completeness: float = 0.0
    missing_required: list[str] | None = None
    missing_typical: list[str] | None = None


def _norm_iatc(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower().strip()
    if "/" in value:
        value = value.split("/")[-1]
    return value


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_terms_from_node(node: dict[str, Any]) -> set[str]:
    terms = set()
    raw = node.get("ner_terms") or []
    for item in raw:
        if isinstance(item, str):
            terms.add(item.lower())
        elif isinstance(item, dict):
            term = item.get("term") or item.get("text") or item.get("label")
            if term:
                terms.add(str(term).lower())
    return terms


def _detect_node_discourse_types(node_id: str, text: str, existing: set[str]) -> set[str]:
    discourse = set(existing)
    if text:
        for wire in nlab_wiring.detect_wires(node_id, text):
            discourse.add(wire.get("hx/type", ""))
    return {d for d in discourse if d}


def _detect_node_ports(node_id: str, text: str, node: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_ports = node.get("input_ports")
    output_ports = node.get("output_ports")
    if isinstance(input_ports, list) and isinstance(output_ports, list):
        return input_ports, output_ports

    inferred_in, inferred_out = ([], [])
    if text:
        inferred_in, inferred_out = assemble_wiring.extract_ports(text, node_id)

    if not isinstance(input_ports, list):
        input_ports = inferred_in
    if not isinstance(output_ports, list):
        output_ports = inferred_out
    return input_ports, output_ports


def _detect_node_categorical(
    node: dict[str, Any],
    node_text: str,
    tags: list[str],
    reference: dict[str, Any],
) -> list[str]:
    explicit = node.get("categorical")
    cats: list[str] = []
    if isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, str):
                cats.append(item)
            elif isinstance(item, dict):
                cat = item.get("hx/type")
                if cat:
                    cats.append(cat)
    if cats:
        return sorted(set(cats))

    inferred = assemble_wiring.detect_categorical_for_se(node_text, tags, reference)
    return sorted({item.get("hx/type") for item in inferred if item.get("hx/type")})


def _load_nodes(wiring: dict[str, Any], reference: dict[str, Any]) -> dict[str, NodeInfo]:
    topic = wiring.get("topic")
    default_tags = [topic] if isinstance(topic, str) and topic else []
    nodes: dict[str, NodeInfo] = {}

    for raw in wiring.get("nodes", []):
        node_id = str(raw.get("id") or raw.get("post_id") or "")
        if not node_id:
            continue
        text = str(raw.get("body_text") or raw.get("body") or raw.get("text") or "")
        tags = raw.get("tags") if isinstance(raw.get("tags"), list) else list(default_tags)
        node_type = str(raw.get("type") or raw.get("node_type") or "unknown")

        categorical = _detect_node_categorical(raw, text, tags, reference)
        discourse_existing = set()
        for d in raw.get("discourse", []):
            if isinstance(d, dict) and d.get("hx/type"):
                discourse_existing.add(d["hx/type"])
        discourse_types = _detect_node_discourse_types(node_id, text, discourse_existing)
        input_ports, output_ports = _detect_node_ports(node_id, text, raw)

        nodes[node_id] = NodeInfo(
            node_id=node_id,
            node_type=node_type,
            text=text,
            tags=tags,
            categorical=categorical,
            discourse_types=discourse_types,
            input_ports=input_ports,
            output_ports=output_ports,
            ner_terms=_extract_terms_from_node(raw),
        )
    return nodes


def _build_pattern_cooccurrence(reference: dict[str, Any]) -> dict[str, set[str]]:
    patterns = reference.get("patterns", {})
    instance_sets = {
        p: set(data.get("instances", []) or [])
        for p, data in patterns.items()
    }
    co: dict[str, set[str]] = {p: {p} for p in patterns}
    for p1 in patterns:
        for p2 in patterns:
            if p1 == p2:
                continue
            if instance_sets[p1] & instance_sets[p2]:
                co[p1].add(p2)
    return co


def _term_present(term: str, text_lower: str, ner_terms: set[str]) -> bool:
    tl = term.lower()
    if tl in text_lower:
        return True
    for item in ner_terms:
        if tl == item or tl in item or item in tl:
            return True
    return False


def _node_completeness(node: NodeInfo, reference: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    patterns = reference.get("patterns", {})
    if not node.categorical:
        return 0.0, [], []

    text_lower = node.text.lower()
    scores = []
    missing_required: set[str] = set()
    missing_typical: set[str] = set()

    for cat in node.categorical:
        p = patterns.get(cat)
        if not p:
            continue
        req = p.get("required_links", []) or []
        typ = p.get("typical_links", []) or []

        req_found = [t for t in req if _term_present(t, text_lower, node.ner_terms)]
        typ_found = [t for t in typ if _term_present(t, text_lower, node.ner_terms)]

        req_total = len(req) if req else 1
        typ_total = len(typ) if typ else 1
        score = 0.7 * (len(req_found) / req_total) + 0.3 * (len(typ_found) / typ_total)
        scores.append(score)

        for t in req:
            if t not in req_found:
                missing_required.add(t)
        for t in typ:
            if t not in typ_found:
                missing_typical.add(t)

    if not scores:
        return 0.0, sorted(missing_required), sorted(missing_typical)
    return _safe_mean(scores), sorted(missing_required), sorted(missing_typical)


def _build_port_type_index(ports: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for port in ports:
        pid = str(port.get("id") or "")
        ptype = str(port.get("type") or "")
        if not pid:
            continue
        index[pid] = ptype
        if ":" in pid:
            index[pid.split(":")[-1]] = ptype
    return index


def _compatible_by_reference(src_type: str, dst_type: str, reference: dict[str, Any]) -> bool:
    for pdata in reference.get("patterns", {}).values():
        comps = pdata.get("discourse_signature", {}).get("components", {})
        if src_type in comps and dst_type in comps:
            return True
    return False


def is_port_type_compatible(src_type: str, dst_type: str, reference: dict[str, Any]) -> tuple[bool, str]:
    if not src_type or not dst_type:
        return False, "missing port type"
    if src_type == dst_type:
        return True, "exact type match"
    if src_type.startswith("bind/") and dst_type.startswith(("bind/", "assume/", "constrain/")):
        return True, "bind output feeds assumption/constraint"
    if src_type.startswith("assume/") and dst_type.startswith(("assume/", "constrain/")):
        return True, "assumption-compatible flow"
    if src_type.startswith("constrain/") and dst_type.startswith(("constrain/", "quant/")):
        return True, "constraint-compatible flow"
    if src_type.startswith("quant/") and dst_type.startswith(("quant/", "constrain/")):
        return True, "quantifier-compatible flow"
    if _compatible_by_reference(src_type, dst_type, reference):
        return True, "co-occurs in CT discourse signature"
    return False, f"incompatible: {src_type} -> {dst_type}"


def _edge_iatc_alignment(iatc: str, source: NodeInfo) -> tuple[bool, str]:
    text = source.text
    discourse = source.discourse_types
    detected_iatc = {hit[0] for hit in assemble_wiring.detect_iatc(text)}
    interrogative = source.node_type == "question" or "?" in text
    example_like = bool(EXAMPLE_RE.search(text))
    clarifying = bool(CLARIFY_RE.search(text))

    if not iatc:
        return False, "missing iatc/edge type"
    if iatc in detected_iatc:
        return True, f"iatc marker detected in source text: {iatc}"

    if iatc == "assert":
        ok = "wire/consequential" in discourse or "wire/causal" in discourse
        return ok, "assert needs consequential/causal discourse in source"
    if iatc == "challenge":
        ok = interrogative or "wire/adversative" in discourse
        return ok, "challenge needs interrogative/adversative source"
    if iatc == "clarify":
        ok = "wire/clarifying" in discourse or clarifying
        return ok, "clarify needs clarifying discourse marker"
    if iatc == "exemplify":
        ok = "env/example" in discourse or example_like
        return ok, "exemplify needs example-like discourse"
    if iatc == "reference":
        ok = bool(REFERENCE_RE.search(text))
        return ok, "reference needs citation-like marker"
    if iatc == "retract":
        ok = bool(RETRACT_RE.search(text))
        return ok, "retract needs correction-like marker"
    if iatc == "reform":
        ok = "wire/clarifying" in discourse or clarifying
        return ok, "reform needs rephrasing/clarifying marker"
    return True, f"no strict rule for iatc={iatc}"


def _edge_categorical_consistency(
    source: NodeInfo,
    target: NodeInfo,
    cooccurrence: dict[str, set[str]],
) -> tuple[bool, str]:
    src = set(source.categorical)
    dst = set(target.categorical)
    if not src or not dst:
        return False, "missing categorical annotation on one or both endpoints"
    inter = src & dst
    if inter:
        return True, f"shared categorical pattern(s): {sorted(inter)}"
    for s in src:
        for t in dst:
            if t in cooccurrence.get(s, set()):
                return True, f"reference co-occurrence allows {s} with {t}"
    return False, f"no shared/co-occurring patterns: src={sorted(src)} dst={sorted(dst)}"


def _normalize_edges(wiring: dict[str, Any]) -> list[dict[str, Any]]:
    norm = []
    for edge in wiring.get("edges", []):
        source = str(edge.get("source") or edge.get("from") or "")
        target = str(edge.get("target") or edge.get("to") or "")
        if not source or not target:
            continue
        norm.append(
            {
                "source": source,
                "target": target,
                "iatc": _norm_iatc(edge.get("iatc") or edge.get("edge_type") or edge.get("type")),
                "port_matches": edge.get("port_matches") or [],
                "raw": edge,
            }
        )
    return norm


def verify_wiring_dict(
    wiring: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    nodes = _load_nodes(wiring, reference)
    edges = _normalize_edges(wiring)
    cooccurrence = _build_pattern_cooccurrence(reference)

    # Node-level completeness first, then edge checks can embed source/target scores.
    node_reports = []
    for node in nodes.values():
        comp, miss_req, miss_typ = _node_completeness(node, reference)
        node.completeness = comp
        node.missing_required = miss_req
        node.missing_typical = miss_typ
        if node.categorical:
            node_reports.append(
                {
                    "node": node.node_id,
                    "categorical": node.categorical,
                    "completeness": round(comp, 4),
                    "missing_required": miss_req,
                    "missing_typical": miss_typ,
                }
            )

    edge_reports = []
    cat_pass = 0
    port_pass = 0
    iatc_pass = 0

    for edge in edges:
        source = nodes.get(edge["source"])
        target = nodes.get(edge["target"])
        if not source or not target:
            continue

        cat_ok, cat_detail = _edge_categorical_consistency(source, target, cooccurrence)
        cat_pass += 1 if cat_ok else 0

        # Determine port matches; for proof-style wiring this is derived.
        port_matches = list(edge["port_matches"])
        if not port_matches:
            port_matches = assemble_wiring.match_ports(
                source.output_ports,
                target.input_ports,
                reference,
            )

        src_port_types = _build_port_type_index(source.output_ports)
        dst_port_types = _build_port_type_index(target.input_ports)

        if port_matches:
            compat_results = []
            for match in port_matches:
                if not isinstance(match, (list, tuple)) or len(match) < 2:
                    continue
                src_port_id = str(match[0])
                dst_port_id = str(match[1])
                src_type = src_port_types.get(src_port_id, "")
                dst_type = dst_port_types.get(dst_port_id, "")
                ok, reason = is_port_type_compatible(src_type, dst_type, reference)
                compat_results.append(
                    {
                        "source_port": src_port_id,
                        "target_port": dst_port_id,
                        "source_type": src_type,
                        "target_type": dst_type,
                        "pass": ok,
                        "detail": reason,
                    }
                )
            if compat_results:
                port_ok = all(item["pass"] for item in compat_results)
                port_detail = f"{sum(1 for x in compat_results if x['pass'])}/{len(compat_results)} compatible"
            else:
                port_ok = False
                port_detail = "no usable port matches"
        else:
            port_ok = False
            port_detail = "no port matches"
            compat_results = []

        port_pass += 1 if port_ok else 0

        iatc_ok, iatc_detail = _edge_iatc_alignment(edge["iatc"], source)
        iatc_pass += 1 if iatc_ok else 0

        edge_reports.append(
            {
                "edge": {
                    "source": edge["source"],
                    "target": edge["target"],
                    "iatc": edge["iatc"],
                },
                "checks": {
                    "categorical": {"pass": cat_ok, "detail": cat_detail},
                    "ports": {
                        "pass": port_ok,
                        "detail": port_detail,
                        "matches": compat_results,
                    },
                    "iatc": {"pass": iatc_ok, "detail": iatc_detail},
                    "completeness": {
                        "source": round(source.completeness, 4),
                        "target": round(target.completeness, 4),
                    },
                },
            }
        )

    edges_checked = len(edge_reports)
    completeness_mean = _safe_mean([n["completeness"] for n in node_reports])
    cat_rate = _safe_rate(cat_pass, edges_checked)
    port_rate = _safe_rate(port_pass, edges_checked)
    iatc_rate = _safe_rate(iatc_pass, edges_checked)
    overall = _safe_mean([cat_rate, port_rate, iatc_rate, completeness_mean])

    return {
        "wiring_id": wiring.get("thread_id") or wiring.get("id") or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "edges_checked": edges_checked,
            "categorical_consistent": cat_pass,
            "port_compatible": port_pass,
            "iatc_aligned": iatc_pass,
            "completeness_mean": round(completeness_mean, 4),
            "overall_score": round(overall, 4),
        },
        "edge_reports": edge_reports,
        "node_reports": node_reports,
    }


def verify_wiring_file(
    wiring_path: Path,
    reference_path: Path,
    output_path: Path,
) -> None:
    wiring_obj = json.loads(wiring_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    if isinstance(wiring_obj, list):
        reports = [verify_wiring_dict(item, reference) for item in wiring_obj]
        payload: Any = reports
    else:
        payload = verify_wiring_dict(wiring_obj, reference)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_live(
    reference_path: Path,
    watch_dir: Path,
    output_dir: Path,
    interval_s: float = 2.0,
) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    while True:
        for path in sorted(watch_dir.glob("*.json")):
            if str(path) in seen:
                continue
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    for item in obj:
                        report = verify_wiring_dict(item, reference)
                        rid = str(report.get("wiring_id", "unknown"))
                        out = output_dir / f"{path.stem}__{rid}-verification.json"
                        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                else:
                    report = verify_wiring_dict(obj, reference)
                    out = output_dir / f"{path.stem}-verification.json"
                    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                seen.add(str(path))
                print(f"[live] verified {path}", file=sys.stderr)
            except Exception as exc:  # pragma: no cover - defensive live mode
                print(f"[live] failed {path}: {exc}", file=sys.stderr)
                seen.add(str(path))
        time.sleep(interval_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Verify one wiring file")
    p_verify.add_argument("--wiring", type=Path, required=True)
    p_verify.add_argument("--reference", type=Path, required=True)
    p_verify.add_argument("--output", type=Path, required=True)

    p_live = sub.add_parser("live", help="Watch directory and verify new wiring files")
    p_live.add_argument("--reference", type=Path, required=True)
    p_live.add_argument("--thread-wiring", type=Path, required=True)
    p_live.add_argument("--output-dir", type=Path, required=True)
    p_live.add_argument("--interval", type=float, default=2.0)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "verify":
        verify_wiring_file(args.wiring, args.reference, args.output)
    elif args.command == "live":
        run_live(args.reference, args.thread_wiring, args.output_dir, interval_s=args.interval)


if __name__ == "__main__":
    main()
