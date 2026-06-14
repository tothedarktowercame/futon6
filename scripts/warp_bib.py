#!/usr/bin/env python3
"""Extract corpus bibliography and citation-key data for WARP W1."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
DEFAULT_OUT = ROOT / "data" / "warp"

sys.path.insert(0, str(ROOT / "scripts"))
from anatomy_v0_sweep import (  # noqa: E402
    parse_balanced_brace,
    read_eprint_files,
    skip_space_and_options,
    skip_ws,
    strip_archive_suffix,
    strip_comments,
)


TEXT_COMMANDS = {
    "emph",
    "textit",
    "textbf",
    "textsc",
    "textrm",
    "textsf",
    "texttt",
    "mathrm",
    "mathbf",
    "mathit",
    "mathsf",
    "mathcal",
    "mathbb",
    "href",
    "url",
}


def clean_latex(text: str) -> str:
    """Best-effort text normalization for paper and bibliography metadata."""
    text = text.replace("~", " ")
    text = re.sub(r"\\['`^\"~=.]\\?\{?([A-Za-z])\}?", r"\1", text)
    text = re.sub(r"\\[cvuHkbtcd]\{([A-Za-z])\}", r"\1", text)
    for cmd in TEXT_COMMANDS:
        pattern = re.compile(r"\\" + re.escape(cmd) + r"\*?\s*\{([^{}]*)\}")
        while True:
            updated = pattern.sub(r"\1", text)
            if updated == text:
                break
            text = updated
    text = re.sub(r"\\(?:url|href)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\s*\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\.", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,\n\t")


def command_args(text: str, command: str) -> list[str]:
    out: list[str] = []
    cmd_re = re.compile(r"\\" + re.escape(command) + r"\*?\b")
    for m in cmd_re.finditer(text):
        pos = skip_space_and_options(text, m.end())
        if pos < len(text) and text[pos] == "{":
            value, _ = parse_balanced_brace(text, pos)
            if value is not None:
                cleaned = clean_latex(value)
                if cleaned:
                    out.append(cleaned)
    return out


def split_authors(values: list[str]) -> list[str]:
    authors: list[str] = []
    seen: set[str] = set()
    for value in values:
        parts = re.split(r"\s+(?:and|\\and)\s+|;", value)
        for part in parts:
            part = clean_latex(part)
            part = re.sub(r"\s+", " ", part).strip(" ,")
            if part and part.lower() not in seen:
                authors.append(part)
                seen.add(part.lower())
    return authors


def extract_identity(files: list[dict[str, str]]) -> tuple[str | None, list[str]]:
    titles: list[str] = []
    authors: list[str] = []
    for f in files:
        text = strip_comments(f["text"])
        titles.extend(command_args(text, "title"))
        authors.extend(command_args(text, "author"))
    title = titles[0] if titles else None
    return title, split_authors(authors)


def iter_cite_keys(text: str) -> list[str]:
    cite_re = re.compile(r"\\(?:no)?cite[a-zA-Z@]*\*?")
    keys: list[str] = []
    seen: set[str] = set()
    for m in cite_re.finditer(text):
        pos = skip_space_and_options(text, m.end())
        if pos >= len(text) or text[pos] != "{":
            continue
        value, _ = parse_balanced_brace(text, pos)
        if value is None:
            continue
        for key in value.split(","):
            key = key.strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def extract_cites(files: list[dict[str, str]]) -> list[str]:
    cites: list[str] = []
    seen: set[str] = set()
    for f in files:
        for key in iter_cite_keys(strip_comments(f["text"])):
            if key not in seen:
                cites.append(key)
                seen.add(key)
    return cites


def parse_bibitem_at(text: str, start: int) -> tuple[str | None, int] | None:
    m = re.match(r"\\bibitem\b\*?", text[start:])
    if not m:
        return None
    pos = skip_space_and_options(text, start + m.end())
    if pos >= len(text) or text[pos] != "{":
        return None
    key, end = parse_balanced_brace(text, pos)
    if key is None:
        return None
    return key.strip(), end


def bibitem_spans(text: str) -> list[tuple[str, int, int]]:
    starts: list[tuple[str, int]] = []
    for m in re.finditer(r"\\bibitem\b\*?", text):
        parsed = parse_bibitem_at(text, m.start())
        if parsed is not None:
            key, body_start = parsed
            starts.append((key, body_start))
    spans: list[tuple[str, int, int]] = []
    for i, (key, body_start) in enumerate(starts):
        next_start = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        if i + 1 < len(starts):
            next_match = text.rfind("\\bibitem", body_start, starts[i + 1][1])
            body_end = next_match if next_match >= body_start else starts[i + 1][1]
        else:
            end_bib = text.find(r"\end{thebibliography}", body_start)
            body_end = end_bib if end_bib >= 0 else next_start
        spans.append((key, body_start, body_end))
    return spans


def parse_bib_metadata(raw: str) -> dict[str, str]:
    raw_no_comments = strip_comments(raw)
    title = None
    emphasized = re.search(
        r"(?:\\(?:emph|textit|textbf)\s*\{|\{\\(?:em|it|sl|bf)\s+)([^{}]+)\}",
        raw_no_comments,
        flags=re.S,
    )
    if emphasized:
        title = clean_latex(emphasized.group(1))

    cleaned = clean_latex(raw)
    out: dict[str, str] = {}

    arxiv = re.search(
        r"(?i)\barxiv\s*:?\s*([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        cleaned,
    )
    if arxiv:
        out["arxiv_id"] = arxiv.group(1)

    year = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", cleaned)
    if year:
        out["year"] = year.group(1)

    quoted = re.search(r"[\"“`](.*?)[\"”']", cleaned)
    if quoted and len(quoted.group(1).split()) >= 2:
        title = quoted.group(1)
    if title:
        out["title"] = title.strip(" .,")

    author = cleaned
    if title and title in author:
        author = author.split(title, 1)[0]
    elif year:
        author = author.split(year.group(1), 1)[0]
    author = re.sub(r"\bet\s+al\b\.?", "", author, flags=re.I).strip(" .,")
    if author:
        out["author"] = author

    return out


def extract_bibitems(files: list[dict[str, str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in files:
        text = strip_comments(f["text"])
        for key, start, end in bibitem_spans(text):
            raw = text[start:end].strip()
            if not key or key in seen:
                continue
            item: dict[str, Any] = {"key": key, "raw": clean_latex(raw) or raw}
            item.update(parse_bib_metadata(raw))
            items.append(item)
            seen.add(key)
    return items


def process_eprint(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    paper_id = strip_archive_suffix(path)
    files, meta = read_eprint_files(path)
    if not files:
        return None, {"paper_id": paper_id, "path": str(path), "status": "skipped", "meta": meta}
    title, authors = extract_identity(files)
    paper = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "bibitems": extract_bibitems(files),
        "cites": extract_cites(files),
    }
    return paper, {"paper_id": paper_id, "path": str(path), "status": "covered", "meta": meta}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def eprint_paths(eprints_dir: Path, paper_ids: list[str] | None) -> list[Path]:
    if paper_ids:
        paths = []
        for paper_id in paper_ids:
            matches = sorted(eprints_dir.glob(paper_id + "*"))
            if not matches:
                raise SystemExit(f"missing eprint for paper id {paper_id}")
            paths.append(matches[0])
        return paths
    return sorted(eprints_dir.glob("*.tar.gz"))


def build_index(papers: list[dict[str, Any]], skipped: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    totals = Counter()
    for paper in papers:
        totals["bibitems"] += len(paper["bibitems"])
        totals["cites"] += len(paper["cites"])
        if paper.get("title"):
            totals["titles"] += 1
        if paper.get("authors"):
            totals["authors"] += 1
    return {
        "papers": papers,
        "stats": {
            "covered": len(papers),
            "skipped": len(skipped),
            "bibitems": totals["bibitems"],
            "cites": totals["cites"],
            "papers_with_title": totals["titles"],
            "papers_with_authors": totals["authors"],
            "elapsed_sec": round(elapsed, 3),
        },
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eprints-dir", type=Path, default=DEFAULT_EPRINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper", action="append", dest="paper_ids", help="process one paper id; repeatable")
    parser.add_argument("--limit", type=int, help="explicit verification cap; omitted means full corpus")
    args = parser.parse_args()

    paths = eprint_paths(args.eprints_dir, args.paper_ids)
    if args.limit is not None:
        paths = paths[: args.limit]
        print(f"explicit limit: processing first {len(paths)} eprints", file=sys.stderr)

    bib_dir = args.out_dir / "bib"
    papers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    started = time.time()
    for i, path in enumerate(paths, 1):
        try:
            paper, status = process_eprint(path)
        except Exception as exc:  # Keep corpus pass bounded per archive.
            paper = None
            status = {
                "paper_id": strip_archive_suffix(path),
                "path": str(path),
                "status": "skipped",
                "error": repr(exc),
            }
        if paper is None:
            skipped.append(status)
        else:
            papers.append(paper)
            write_json(bib_dir / f"{paper['paper_id']}.json", paper)
        if i % 500 == 0:
            print(f"processed={i} covered={len(papers)} skipped={len(skipped)}", file=sys.stderr)

    elapsed = time.time() - started
    index = build_index(papers, skipped, elapsed)
    write_json(args.out_dir / "bib-index.json", index)
    stats = index["stats"]
    print(
        "covered={covered} skipped={skipped} bibitems={bibitems} cites={cites} "
        "papers_with_title={papers_with_title} papers_with_authors={papers_with_authors} "
        "elapsed_sec={elapsed_sec}".format(**stats),
        file=sys.stderr,
    )
    return 0 if not skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
