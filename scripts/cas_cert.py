#!/usr/bin/env python3
"""CAS-CERT conformance certificate aggregator.

This script does not implement new checks. It reads emitted rung outputs
(`iatc_semcheck` and optionally `cas_select`) and partitions their verdicts into
the CAS certificate port ledger.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import edn_format

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b"
SCHEMA = "futon6/cas-cert/v1"
GRAINS = ("symbol", "concept", "technique", "proof")


def keyword_name(value: Any) -> str:
    text = str(value)
    return text[1:] if text.startswith(":") else text


def plain(value: Any) -> Any:
    if isinstance(value, edn_format.Keyword):
        return ":" + keyword_name(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {keyword_name(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or (
        not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
    ):
        return [plain(v) for v in value]
    return value


def load_edn(path: Path) -> dict[str, Any]:
    text = re.sub(r":([A-Za-z0-9_./?=-]+)'", r":\1-prime", path.read_text())
    return plain(edn_format.loads(text))


def run_semcheck(graph_dir: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".edn") as fh:
        result = subprocess.run(
            ["bb", "scripts/iatc_semcheck.bb", "--out", fh.name, str(graph_dir)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr or result.stdout)
        return load_edn(Path(fh.name))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def check_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("check", "")).lstrip(":"): row for row in graph.get("checks", [])}


def port(
    *,
    grain: str,
    item: str,
    state: str,
    rung: str,
    scoped_query: str,
    evidence: Any = None,
    kind: str | None = None,
) -> dict[str, Any]:
    row = {
        "grain": grain,
        "item": item,
        "state": state,
        "rung": rung,
        "scoped_query": scoped_query,
        "evidence": evidence,
    }
    if kind:
        row["kind"] = kind
    return row


def symbol_ports(symbols: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if symbols:
        out = []
        for row in symbols.get("groundings") or []:
            status = str(row.get("status", ""))
            if status == "grounded":
                state = "filled"
                kind = None
            elif status in {"undefined-in-context", "unsupported"}:
                state = "empty"
                kind = status
            else:
                state = "empty"
                kind = "unknown-symbol-status"
            symbol = str(row.get("symbol", ""))
            out.append(
                port(
                    grain="symbol",
                    item=f"symbol:{symbol}",
                    state=state,
                    rung="SFC2b",
                    scoped_query=f"per-paper grounding for symbol '{symbol}'",
                    evidence={
                        "status": status,
                        "binding": row.get("binding", ""),
                        "evidence": row.get("evidence", ""),
                    },
                    kind=kind,
                )
            )
        return out
    return [
        port(
            grain="symbol",
            item="symbol-grounding",
            state="na",
            rung="SFC2b",
            scoped_query="per-paper symbol/domain grounding",
            evidence="SFC2b not wired into CAS-CERT yet",
        )
    ]


def concept_ports(graph: dict[str, Any]) -> list[dict[str, Any]]:
    check = check_by_id(graph).get("concept-coverage", {})
    if str(check.get("status")) == ":na":
        return [
            port(
                grain="concept",
                item="concept-coverage",
                state="na",
                rung="R2d",
                scoped_query="concept coverage unavailable at this resolution",
                evidence=check.get("reasons"),
            )
        ]
    out = []
    for row in check.get("per-item") or []:
        bucket = row.get("bucket")
        state = "empty" if bucket == "undefined" else "filled"
        kind = "undefined" if state == "empty" else None
        out.append(
            port(
                grain="concept",
                item=str(row.get("concept")),
                state=state,
                rung="R2d",
                scoped_query=f"definition/known/import status for '{row.get('concept')}'",
                evidence={"bucket": bucket, "reason": row.get("reason"), "sources": row.get("sources", [])},
                kind=kind,
            )
        )
    return out


def anchor_ports(graph: dict[str, Any]) -> list[dict[str, Any]]:
    check = check_by_id(graph).get("anchor-faithfulness", {})
    out = []
    for row in check.get("per-item") or []:
        status = str(row.get("status", "")).lstrip(":")
        if status == "pass":
            state = "filled"
        elif status == "fail":
            state = "miswired"
        else:
            state = "na"
        out.append(
            port(
                grain="proof",
                item=f"anchor:{row.get('id')}",
                state=state,
                rung="R2a",
                scoped_query="node source lines contain the node's key terms",
                evidence={"lines": (row.get("source") or {}).get("lines"), "missing": row.get("missing", [])},
                kind="anchor" if state == "miswired" else None,
            )
        )
    return out


def closure_ports(graph: dict[str, Any]) -> list[dict[str, Any]]:
    check = check_by_id(graph).get("closure", {})
    items = check.get("per-item") or []
    if not items:
        return [
            port(
                grain="proof",
                item="closure",
                state="na",
                rung="R2b",
                scoped_query="argument graph closure",
                evidence=check.get("reasons"),
            )
        ]
    out = []
    item = items[0]
    for node in item.get("orphan-nodes") or []:
        out.append(
            port(
                grain="proof",
                item=f"orphan:{node}",
                state="empty",
                rung="R2b",
                scoped_query="node participates in proof wiring",
                evidence={"file": item.get("file")},
                kind="orphan",
            )
        )
    if item.get("cycle"):
        out.append(
            port(
                grain="proof",
                item="cycle",
                state="miswired",
                rung="R2b",
                scoped_query="argument graph is acyclic",
                evidence=item.get("cycle"),
                kind="cycle",
            )
        )
    if not out:
        out.append(
            port(
                grain="proof",
                item="closure",
                state="filled",
                rung="R2b",
                scoped_query="argument graph has reachable terminal(s) and no orphan/cycle",
                evidence={"rate": check.get("rate")},
            )
        )
    return out


def warrant_ports(graph: dict[str, Any]) -> list[dict[str, Any]]:
    profile = graph.get("profile") or {}
    reasoning = profile.get("reasoning") or []
    if not reasoning:
        return [
            port(
                grain="proof",
                item="warrant-resolution",
                state="na",
                rung="R2c",
                scoped_query="edge warrants present",
                evidence="no reasoning edges at this resolution",
            )
        ]
    out = []
    for edge in reasoning:
        warrant = edge.get("warrant") or {}
        status = str(warrant.get("status", "")).lstrip(":")
        state = "filled" if status == "resolved" else "empty"
        out.append(
            port(
                grain="proof",
                item=f"warrant:{edge.get('id')}",
                state=state,
                rung="R2c",
                scoped_query="edge has a resolved warrant",
                evidence=warrant,
                kind="missing-warrant" if state == "empty" else None,
            )
        )
    return out


def technique_ports(paper_id: str, cas_select: dict[str, Any] | None) -> list[dict[str, Any]]:
    out = [
        port(
            grain="technique",
            item="rung-3-technique-grounding",
            state="na",
            rung="rung-3",
            scoped_query="semantic technique grounding verdict",
            evidence="rung-3 technique checker not built yet",
        )
    ]
    if not cas_select:
        return out
    result = (cas_select.get("results") or {}).get(paper_id)
    if not result:
        return out
    for row in result.get("sorry") or []:
        if row.get("kind") == "thin":
            out.append(
                port(
                    grain="technique",
                    item=f"thin:{row.get('step')}",
                    state="empty",
                    rung="CAS-SEL",
                    scoped_query="step matched a known technique pattern",
                    evidence=row,
                    kind="thin",
                )
            )
        else:
            out.append(
                port(
                    grain="technique",
                    item=f"declared:{row.get('step')}:{row.get('pattern')}",
                    state="filled",
                    rung="CAS-SEL",
                    scoped_query="matched pattern declares residual obligation",
                    evidence=row,
                )
            )
    return out


def grain_summary(ports: list[dict[str, Any]], grain: str, rung: str) -> dict[str, Any]:
    rows = [p for p in ports if p["grain"] == grain]
    counts = Counter(p["state"] for p in rows)
    only_na = bool(rows) and all(p["state"] == "na" for p in rows)
    denom = counts["filled"] + counts["empty"] + counts["miswired"]
    return {
        "filled": counts["filled"],
        "empty": counts["empty"],
        "miswired": counts["miswired"],
        "na": only_na,
        "rate": (counts["filled"] / denom) if denom else None,
        "rung": rung,
    }


def grain_solidity(summary: dict[str, Any]) -> float | None:
    denom = summary["filled"] + summary["empty"] + summary["miswired"]
    if denom == 0:
        return None
    return (summary["filled"] + summary["miswired"]) / denom


def confidence(by_grain: dict[str, dict[str, Any]]) -> dict[str, Any]:
    limiting_factors = []
    low_solidity = False
    for grain in ("symbol", "concept", "technique"):
        summary = by_grain[grain]
        if summary["na"]:
            if grain == "symbol":
                limiting_factors.append("symbol grain N/A — SFC2b not built")
            elif grain == "technique":
                limiting_factors.append("technique grain N/A — rung-3 not built")
            else:
                limiting_factors.append(f"{grain} grain N/A — {summary['rung']} not built")
            continue
        solidity = grain_solidity(summary)
        if solidity is not None and solidity < 0.5:
            low_solidity = True
            limiting_factors.append(f"{grain} grain low solidity {solidity:.3f}")
    if low_solidity:
        level = "low"
    elif any(by_grain[grain]["na"] for grain in ("symbol", "concept", "technique")):
        level = "medium"
    else:
        level = "high"
    return {"level": level, "limiting_factors": limiting_factors}


def residual_sorries(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in ports:
        if p["state"] != "empty":
            continue
        kind = p.get("kind") or {
            "concept": "undefined",
            "symbol": "ungrounded",
            "technique": "thin",
            "proof": "missing-warrant",
        }.get(p["grain"], "empty")
        out.append(
            {
                "grain": p["grain"],
                "kind": kind,
                "item": p["item"],
                "scoped_query": p["scoped_query"],
                "arse_seed": arse_seed(kind, p),
            }
        )
    return out


def arse_seed(kind: str, p: dict[str, Any]) -> str:
    if kind == "undefined":
        return f"What defines {p['item']} in this paper's concept substrate?"
    if kind == "orphan":
        return f"How should {p['item']} be wired into the proof graph?"
    if kind == "missing-warrant":
        return f"What warrant licenses {p['item']}?"
    if kind == "thin":
        return f"What technique pattern grounds {p['item']}?"
    return f"What fills the {p['grain']} port {p['item']}?"


def certificate_for_graph(
    graph: dict[str, Any],
    cas_select: dict[str, Any] | None = None,
    symbols_by_paper: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paper_id = graph.get("paper-id") or (graph.get("profile") or {}).get("paper-id")
    symbol_doc = (symbols_by_paper or {}).get(str(paper_id))
    ports = []
    ports.extend(symbol_ports(symbol_doc))
    ports.extend(concept_ports(graph))
    ports.extend(technique_ports(str(paper_id), cas_select))
    ports.extend(anchor_ports(graph))
    ports.extend(closure_ports(graph))
    ports.extend(warrant_ports(graph))

    by_grain = {
        "symbol": grain_summary(ports, "symbol", "SFC2b"),
        "concept": grain_summary(ports, "concept", "R2d"),
        "technique": grain_summary(ports, "technique", "CAS-SEL/rung-3"),
        "proof": grain_summary(ports, "proof", "R2a/R2b/R2c"),
    }
    miswires = [p for p in ports if p["state"] == "miswired"]
    concept_rate = by_grain["concept"]["rate"]
    return {
        "paper_id": paper_id,
        "schema": SCHEMA,
        "conformance": {
            "by_grain": by_grain,
            "headline": "vector-by-grain; symbol and full rung-3 technique grains are N/A until wired",
        },
        "confidence": confidence(by_grain),
        "ports": ports,
        "residual_sorries": residual_sorries(ports),
        "value_signals": {
            "centrality": None,
            "novelty": None,
            "connections": [],
            "conjectures": [],
            "pct_grounded": concept_rate,
            "status": "partial; centrality/novelty/connections/conjectures not wired",
        },
        "verdict": {
            "well_wired": not miswires,
            "miswires": [p["item"] for p in miswires],
            "gate": "FAIL" if miswires else "PASS",
        },
    }


def symbols_by_paper(payload: Any) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    if isinstance(payload, list):
        docs = payload
    elif isinstance(payload, dict) and "papers" in payload:
        docs = payload.get("papers") or []
    else:
        docs = [payload]
    out = {}
    for doc in docs:
        paper_id = doc.get("paper_id") or doc.get("paper-id")
        if paper_id:
            out[str(paper_id)] = doc
    return out


def build_certificates(
    semcheck: dict[str, Any],
    cas_select: dict[str, Any] | None = None,
    symbols: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    certs = [
        certificate_for_graph(graph, cas_select, symbols)
        for graph in semcheck.get("graphs", [])
    ]
    return {
        "schema": SCHEMA,
        "paper_count": len(certs),
        "gate": "FAIL" if any(c["verdict"]["gate"] == "FAIL" for c in certs) else "PASS",
        "certificates": certs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semcheck", type=Path, help="EDN emitted by iatc_semcheck.bb --out")
    ap.add_argument("--cas-select", type=Path, help="JSON emitted by cas_select.py")
    ap.add_argument("--symbols", type=Path, help="JSON emitted by sfc_ground_paper.py")
    ap.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args(argv)

    semcheck = load_edn(args.semcheck) if args.semcheck else run_semcheck(args.graph_dir)
    cas_select = load_json(args.cas_select) if args.cas_select else None
    symbols = symbols_by_paper(load_json(args.symbols)) if args.symbols else None
    payload = build_certificates(semcheck, cas_select, symbols)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 1 if args.gate and payload["gate"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
