#!/usr/bin/env python3
"""mark3 expository candidate extractor.

This is the scripted half of Phase 5.4: carve expository regions CPU-side and
emit self-contained candidate windows for the model loop. It does not classify
or fill scopes; `mark3_expository_loop.py` owns that step.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config
import run_manifest
import stage_accounting as accounting


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MARKS_DIR = config.marks()
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
    carved = expo.extract_regions(entity_id, text, marks)
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
                "marks-path": iatc_candidates._display(marks_path),
            }
        )
    return out


def default_papers() -> list[str]:
    return ["0710.2254", "0711.1761", "0801.2567", "0807.1872", "0905.0595"]


def safe_name(candidate: dict[str, Any]) -> str:
    region = str(candidate["region-id"]).replace("/", "_").replace(":", "_")
    return f"{candidate['paper-id']}.{region}.candidate.json"


def select_even(candidates: list[dict[str, Any]], cap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-paper selection named in the run manifest (even spacing in source order).

    Regions are ordered by source position, not filename, and the kept indices are
    floor(i * n / cap), so a cap samples the whole paper rather than its opening.
    Returns (selected, deferred); cap 0 selects everything.
    """
    ordered = sorted(candidates, key=lambda c: (c["window-lines"][0], c["window-lines"][1], c["region-id"]))
    if cap <= 0 or len(ordered) <= cap:
        return ordered, []
    keep = {i * len(ordered) // cap for i in range(cap)}
    return ([c for i, c in enumerate(ordered) if i in keep],
            [c for i, c in enumerate(ordered) if i not in keep])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "data" / "expository-candidates"))
    parser.add_argument("--papers", nargs="*", help="paper ids; default = dp-demo papers")
    parser.add_argument("--list", help="file of paper ids, one per line (same as emit_marks --list)")
    parser.add_argument("--cap-per-paper", type=int,
                        default=int(os.environ.get("FUTON6_EXPOSITORY_CAP_PER_PAPER", "0") or 0),
                        help="select at most N regions per paper; the rest are accounted as deferred")
    args = parser.parse_args()
    if args.cap_per_paper < 0:
        parser.error("--cap-per-paper must be nonnegative")

    papers = args.papers or (args.list and [l.strip() for l in open(args.list) if l.strip()]) or default_papers()
    outdir = Path(args.out)
    # Every carved region is kept under regions/; the loop reads only the selected
    # candidates at the top level, so deferred regions stay inspectable but unused.
    regions = outdir / "regions"
    regions.mkdir(parents=True, exist_ok=True)
    extract_ledger = accounting.Accounting("S4", "extract", papers)
    carved: list[dict[str, Any]] = []
    for paper_id in papers:
        try:
            candidates = extract(paper_id)
        except Exception as exc:
            extract_ledger.record(paper_id, "errored", f"{type(exc).__name__}: {exc}", paper=paper_id)
            print(f"  ERROR {paper_id}: {exc}")
            continue
        paths = []
        for candidate in candidates:
            path = regions / safe_name(candidate)
            path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
            paths.append(path)
        carved.extend(candidates)
        extract_ledger.record(paper_id, "accepted", "" if candidates else "no expository region carved",
                              paper=paper_id, artifacts=[accounting.relative(p) for p in paths],
                              outputs=[c["passage-id"] for c in candidates])
        print(f"  {paper_id}: {len(candidates)} expository candidates")

    select_ledger = accounting.Accounting("S4", "select", accounting.accepted_outputs(extract_ledger.document()))
    selected_names = set()
    manifest = []
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for candidate in carved:
        by_paper.setdefault(candidate["paper-id"], []).append(candidate)
    for paper_id, candidates in by_paper.items():
        selected, deferred = select_even(candidates, args.cap_per_paper)
        for candidate in selected:
            path = outdir / safe_name(candidate)
            path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
            selected_names.add(path.name)
            select_ledger.record(candidate["passage-id"], "accepted", paper=paper_id,
                                 artifacts=[accounting.relative(path)], outputs=[candidate["passage-id"]])
            manifest.append({"paper-id": paper_id, "passage-id": candidate["passage-id"],
                             "region-id": candidate["region-id"], "window-lines": candidate["window-lines"],
                             "enrichment": len(candidate["enrichment"])})
        for candidate in deferred:
            select_ledger.record(candidate["passage-id"], "deferred",
                                 f"cap {args.cap_per_paper} per paper; not selected by "
                                 f"{run_manifest.EXPOSITORY_SELECTION}", paper=paper_id)
        if deferred:
            print(f"  {paper_id}: selected {len(selected)}, deferred {len(deferred)} (cap {args.cap_per_paper})")
    stale = sorted(p.name for p in outdir.glob("*.candidate.json") if p.name not in selected_names)
    if stale:
        # A selected candidate left by another selection would be modelled as if chosen now.
        print(f"FATAL: {len(stale)} candidate file(s) outside this selection, e.g. {stale[:3]}")
        return 2
    (outdir / "manifest.json").write_text(json.dumps({"candidates": manifest,
                                                      "cap-per-paper": args.cap_per_paper,
                                                      "selection": run_manifest.EXPOSITORY_SELECTION
                                                      if args.cap_per_paper else "all-regions"}, indent=2),
                                          encoding="utf-8")
    print(f"\n{len(manifest)} selected of {len(carved)} candidates -> {outdir}")
    return 1 if extract_ledger.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
