"""Build a flat retrieval index from an audited Mark7 scope fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from holes.inkup.grammar.inventory_mark7 import DEFAULT_FIXTURE, census


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    REPO_ROOT / "holes/inkup/grammar/mark7-0708.2185-inventory.json"
)

FUNCTION_KINDS = {
    "connection": "connection",
    "universal-property/characterizes": "characterization",
    "auxiliary-construction": "auxiliary-construction",
    "difficulty-assessment": "difficulty-assessment",
    "generalisation": "generalisation",
    "open-problem/status": "open-problem-status",
}
PLACEMENT_KINDS = {
    "carved/section-lead": "section-lead",
    "carved/inflight": "inflight",
    "carved/in-proof": "in-proof",
    "carved/leaf-section": "leaf-section",
}
ARGUMENT_RELATIONS = {"implies", "iff", "by-cases"}


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def _contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    return outer["start"] <= inner["start"] and inner["end"] <= outer["end"]


def _axis_labels(span: dict[str, Any] | None) -> dict[str, str | None] | None:
    if span is None:
        return None
    return {
        "communicative_function": FUNCTION_KINDS.get(span["kind"]),
        "placement": PLACEMENT_KINDS.get(span["kind"]),
    }


def _innermost_region(
    span: dict[str, Any], regions: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    containers = [region for region in regions if _contains(region, span)]
    if not containers:
        return None
    return min(
        containers,
        key=lambda region: (
            region["end"] - region["start"],
            -region["start"],
            region["id"],
        ),
    )


def build_index(
    document: dict[str, Any],
    fixture_sha256: str,
    *,
    expected_s4_count: int | None = None,
) -> list[dict[str, Any]]:
    """Return one retrieval record for each S4 span in *document*."""
    segments = {segment["id"]: segment for segment in document["segments"]}
    spans_by_id = {span["id"]: span for span in document["spans"]}
    s4 = sorted(
        (span for span in document["spans"] if span["layer"] == "s4"),
        key=lambda span: (span["start"], span["end"], span["id"]),
    )
    if expected_s4_count is not None and len(s4) != expected_s4_count:
        raise ValueError(
            f"S4 count {len(s4)} does not match audited inventory "
            f"{expected_s4_count}"
        )

    regions = [
        span for span in document["spans"] if span["layer"] == "s1-region"
    ]
    proof_regions = [region for region in regions if region["kind"] == "env/proof"]
    paper_id = document["turn"].removeprefix("mark7-")
    records: list[dict[str, Any]] = []

    for index, span in enumerate(s4):
        segment = segments[span["seg"]]
        paper_text = segment.get("text")
        text = (
            paper_text[span["start"] : span["end"]]
            if isinstance(paper_text, str)
            else None
        )
        region = _innermost_region(span, regions)
        within_region = None
        if region is not None and region["end"] > region["start"]:
            within_region = (span["start"] - region["start"]) / (
                region["end"] - region["start"]
            )
        within_paper = None
        if isinstance(paper_text, str) and paper_text:
            within_paper = span["start"] / len(paper_text)

        argument_relations = []
        for edge in document.get("edges", []):
            if edge.get("relation") not in ARGUMENT_RELATIONS:
                continue
            source = spans_by_id.get(edge.get("from"))
            target = spans_by_id.get(edge.get("to"))
            if source and target and _overlaps(span, source) and _overlaps(span, target):
                argument_relations.append(
                    {
                        "relation": edge["relation"],
                        "from": edge["from"],
                        "to": edge["to"],
                    }
                )

        records.append(
            {
                "span_id": span["id"],
                "paper_id": paper_id,
                "fixture_sha256": fixture_sha256,
                "communicative_function": FUNCTION_KINDS.get(span["kind"]),
                "placement": PLACEMENT_KINDS.get(span["kind"]),
                "enclosing_region": (
                    {"id": region["id"], "kind": region["kind"]}
                    if region is not None
                    else None
                ),
                "inside_proof": any(_contains(proof, span) for proof in proof_regions),
                "position": {
                    "within_enclosing_region": within_region,
                    "within_paper": within_paper,
                },
                "argument_relations": argument_relations,
                "neighbours": {
                    "previous": _axis_labels(s4[index - 1] if index else None),
                    "next": _axis_labels(
                        s4[index + 1] if index + 1 < len(s4) else None
                    ),
                },
                "text": text,
                "word_count": len(text.split()) if text is not None else None,
            }
        )
    return records


def load_index(
    fixture: Path = DEFAULT_FIXTURE,
    inventory: Path = DEFAULT_INVENTORY,
) -> list[dict[str, Any]]:
    """Load a fixture and enforce its audited S4 count and SHA-256."""
    audited = json.loads(inventory.read_text())
    measured = census(fixture)
    if measured["fixture_sha256"] != audited["fixture_sha256"]:
        raise ValueError("fixture SHA-256 does not match audited inventory")
    document = json.loads(fixture.read_text())
    return build_index(
        document,
        measured["fixture_sha256"],
        expected_s4_count=audited["layer_counts"]["s4"],
    )


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
