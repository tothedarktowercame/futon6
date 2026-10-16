#!/usr/bin/env python3
"""Convert a mark7 margin page's payload to shared scope-view data."""
from __future__ import annotations

import argparse
import html
import json
from html.parser import HTMLParser
from pathlib import Path


class _Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.parts = {}

    def handle_starttag(self, tag, attrs):
        ident = dict(attrs).get("id")
        if tag == "script" and ident in ("m7-data", "m7-source"):
            self.current = ident
            self.parts[ident] = []

    def handle_data(self, data):
        if self.current:
            self.parts[self.current].append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


def page_blocks(path: Path) -> tuple[dict, str]:
    parser = _Blocks()
    parser.feed(path.read_text(encoding="utf-8"))
    try:
        payload = json.loads(html.unescape("".join(parser.parts["m7-data"])))
        source = json.loads(html.unescape("".join(parser.parts["m7-source"])))
    except KeyError as exc:
        raise ValueError(f"page lacks {exc.args[0]} script block") from exc
    if not isinstance(payload, dict) or not isinstance(source, str):
        raise ValueError("margin page blocks have the wrong JSON types")
    return payload, source


def _range(start, end, length, owner):
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start or end > length):
        raise ValueError(f"{owner} range [{start},{end}) outside source length {length}")
    return start, end


def _scope_range(scope, starts, length):
    if scope.get("span") is not None:
        start, end = scope["span"]
        return (*_range(start, end, length, "S4 scope"), 1.0)
    lo, hi = scope["lines"]
    if type(lo) is not int or type(hi) is not int or lo < 1 or hi < lo:
        raise ValueError(f"invalid S4 lines {scope['lines']!r}")
    start = starts[lo - 1] if lo <= len(starts) else length
    end = starts[hi] if hi < len(starts) else length
    start, end = _range(min(start, length), min(end, length), length, "S4 lines")
    return start, end, 0.6


def scopes_data(payload: dict, source: str) -> dict:
    """Return shared scope-view data for one mark7 margin payload."""
    length = len(source)
    summary = payload["summary"]
    kinds = payload["kinds"]
    glossary = payload["glossary"]
    starts = payload["starts"]
    spans, edges, legend = [], [], {}

    for index, mark in enumerate(payload["marks"]):
        start, end, kind_index, grounded, tip = mark
        _range(start, end, length, f"S1 mark {index}")
        try:
            kind = kinds[kind_index][0]
            meaning = glossary["S1 mark kinds"][kind]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"S1 mark {index} has no defined kind") from exc
        spans.append({"id": f"s1-{index}", "seg": "mark7-0708.2185",
                      "start": start, "end": end, "layer": "s1", "kind": kind,
                      "weight": 1.0 if grounded else 0.5, "note": tip})
        legend[kind] = meaning

    node_spans = {}
    proofs = [note for note in payload["notes"] if note.get("type") == "proof"]
    for proof in proofs:
        for node in proof["nodes"]:
            at = node.get("at") or []
            if not at:
                raise ValueError(f"S3 node {proof['id']}-{node['id']} has no source range")
            ids = []
            for index, pair in enumerate(at):
                start, end = _range(*pair, length, f"S3 node {proof['id']}-{node['id']}")
                base = f"{proof['id']}-{node['id']}"
                ident = base if len(at) == 1 else f"{base}-{index}"
                kind = node["kind"]
                spans.append({"id": ident, "seg": "mark7-0708.2185",
                              "start": start, "end": end, "layer": "s3", "kind": kind,
                              "weight": 1.0,
                              "note": f"{node.get('gloss', '')} · {node.get('verdict', '')}"})
                legend[kind] = glossary["S3 node kinds"][kind]
                ids.append(ident)
            node_spans[(proof["id"], node["id"])] = ids
        for edge in proof["edges"]:
            conclusion = node_spans[(proof["id"], edge["conclusion"])][0]
            relation = edge["relation"]
            warrant = edge["warrant"]
            legend[relation] = relation
            legend[warrant] = glossary["S3 warrants"][warrant]
            for premise in edge["premises"]:
                edges.append({"from": node_spans[(proof["id"], premise)][0],
                              "to": conclusion, "relation": relation, "label": warrant})

    scope_count = 0
    for region in (note for note in payload["notes"] if note.get("type") == "region"):
        for scope in region["scopes"]:
            start, end, weight = _scope_range(scope, starts, length)
            ident = f"{region['id']}-{scope['scope']}"
            kind = scope["kind"]
            flags = ", ".join(scope.get("flags") or []) or "no flags"
            mock = scope.get("mock") or {}
            note = f"{scope.get('fill') or 'no fill'} · {flags} · {mock.get('verdict', 'no mock verdict')}"
            spans.append({"id": ident, "seg": "mark7-0708.2185",
                          "start": start, "end": end, "layer": "s4", "kind": kind,
                          "weight": weight, "note": note})
            legend[kind] = glossary["S4 scope kinds"][kind]
            scope_count += 1

    premise_pairs = len(edges)
    return {
        "turn": "mark7-0708.2185", "record": summary["run"],
        "offset_unit": "unicode-codepoints-zero-based-end-exclusive",
        "segments": [{"id": "mark7-0708.2185", "role": "paper",
                      "label": f"0708.2185 · {summary['run']}", "text": source}],
        "layers": [{"id": "s1", "name": "S1 source marks", "ink": "c"},
                   {"id": "s3", "name": "S3 proof graph", "ink": "m"},
                   {"id": "s4", "name": "S4 expository scopes", "ink": "y"}],
        "spans": spans, "edges": edges, "legend": legend,
        "spot_check": {
            "drawn": {"marks": len(payload["marks"]),
                      "nodes": sum(len(node["at"]) for proof in proofs for node in proof["nodes"]),
                      "edges": premise_pairs, "scopes": scope_count},
            "summary": {"marks": len(payload["marks"]), "nodes": summary["nodes"],
                        "edges": summary["edges"], "scopes": summary["scopes"]},
            "omitted_region_effects": "whole-proof 0.3 and refused-proof 0.12 shading",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload, source = page_blocks(args.page)
    data = scopes_data(payload, source)
    text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"{args.out}: {data['spot_check']}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
