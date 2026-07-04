#!/usr/bin/env python3
"""mark3 expository candidate extractor.

This is the scripted half of Phase 5.4: carve expository regions CPU-side and
emit self-contained candidate windows for the model loop. It does not classify
or fill scopes; `mark3_expository_loop.py` owns that step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MARKS_DIR = REPO / "data" / "showcases" / "ct-anatomy" / "golden"
VOCAB = REPO / "holes" / "excursions" / "expository-superpod-vocab.edn"
SCHEMA = "expo-candidate/v1"

sys.path.insert(0, str(REPO / "scripts"))
import expository_region_extract as expo  # noqa: E402
import mark3_extract_candidates as iatc_candidates  # noqa: E402


def line_starts(text: str) -> list[int]:
    return iatc_candidates.line_starts(text)


def window_text(text: str, starts: list[int], lo_line: int, hi_line: int) -> str:
    start_char = starts[lo_line - 1]
    end_char = starts[hi_line] if hi_line < len(starts) else len(text)
    return text[start_char:end_char]


def load_marks(paper_id: str) -> tuple[str, list[dict[str, Any]], Path]:
    marks_path = MARKS_DIR / f"fable-{paper_id}-dp-emacs.json"
    if not marks_path.exists():
        raise FileNotFoundError(f"missing golden marks JSON: {marks_path}")
    data = json.loads(marks_path.read_text(encoding="utf-8"))
    return str(data["text"]), [m for m in data.get("marks", []) if "start" in m and "end" in m], marks_path


def extract(paper_id: str) -> list[dict[str, Any]]:
    text, marks, marks_path = load_marks(paper_id)
    entity_id, raw_text = expo.load_text(paper_id)
    if raw_text != text:
        raise ValueError(f"text mismatch between extractor and golden marks for {paper_id}")
    carved = expo.extract_regions(entity_id, text)
    starts = line_starts(text)
    out: list[dict[str, Any]] = []
    for region in carved.get("regions", []):
        lo = int(region["line_start"])
        hi = int(region["line_end"])
        region_id = str(region["region_id"])
        out.append(
            {
                "schema": SCHEMA,
                "paper-id": paper_id,
                "passage-id": f"{paper_id}:{region_id}:L{lo}-{hi}",
                "region-id": region_id,
                "region-type": region["type"],
                "window-lines": [lo, hi],
                "source-window": window_text(text, starts, lo, hi),
                "enrichment": iatc_candidates.window_enrichment(marks, starts, lo, hi),
                "vocab-path": str(VOCAB.relative_to(REPO)),
                "marks-path": str(marks_path.relative_to(REPO)),
            }
        )
    return out


def default_papers() -> list[str]:
    return ["0710.2254", "0711.1761", "0801.2567", "0807.1872", "0905.0595"]


def safe_name(candidate: dict[str, Any]) -> str:
    region = str(candidate["region-id"]).replace("/", "_").replace(":", "_")
    return f"{candidate['paper-id']}.{region}.candidate.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "data" / "expository-candidates"))
    parser.add_argument("--papers", nargs="*", help="paper ids; default = dp-demo papers")
    parser.add_argument("--list", help="file of paper ids, one per line (same as emit_marks --list)")
    args = parser.parse_args()

    papers = args.papers or (args.list and [l.strip() for l in open(args.list) if l.strip()]) or default_papers()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = 0
    for paper_id in papers:
        try:
            candidates = extract(paper_id)
        except Exception as exc:
            print(f"  skip {paper_id}: {exc}")
            continue
        for candidate in candidates:
            path = outdir / safe_name(candidate)
            path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
            total += 1
            manifest.append(
                {
                    "paper-id": candidate["paper-id"],
                    "passage-id": candidate["passage-id"],
                    "region-id": candidate["region-id"],
                    "window-lines": candidate["window-lines"],
                    "enrichment": len(candidate["enrichment"]),
                }
            )
        print(f"  {paper_id}: {len(candidates)} expository candidates")
    (outdir / "manifest.json").write_text(json.dumps({"candidates": manifest}, indent=2), encoding="utf-8")
    print(f"\n{total} candidates -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
