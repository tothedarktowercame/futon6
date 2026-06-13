#!/usr/bin/env python3
"""Build the WARP citation graph over the math.CT eprint corpus.

The graph depends on W1's bibliography JSON when present, but the corpus paper
identity index is built directly from the eprint sources so this script can be
developed and spot-checked before W1 lands.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from anatomy_v0_sweep import (  # noqa: E402
    DEFAULT_EPRINTS,
    parse_balanced_brace,
    read_eprint_files,
    strip_archive_suffix,
    strip_comments,
)

DEFAULT_WARP = ROOT / "data" / "warp"
DEFAULT_BIB_INDEX = DEFAULT_WARP / "bib-index.json"
DEFAULT_BIB_DIR = DEFAULT_WARP / "bib"
DEFAULT_OUT = DEFAULT_WARP / "citations.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "via",
    "with",
}

VENUE_WORDS = {
    "advances",
    "annals",
    "bulletin",
    "cambridge",
    "commun",
    "communications",
    "compositio",
    "contemporary",
    "dissertation",
    "doctor",
    "doctoral",
    "geometry",
    "homology",
    "journal",
    "lecture",
    "letters",
    "math",
    "mathematical",
    "mathematics",
    "memoirs",
    "notes",
    "phd",
    "preprint",
    "proceedings",
    "proc",
    "publ",
    "publication",
    "series",
    "springer",
    "thesis",
    "topology",
    "transactions",
    "trans",
}


@dataclass(frozen=True)
class PaperIdentity:
    paper_id: str
    title: str
    title_norm: str
    title_tokens: frozenset[str]
    author_names: tuple[str, ...]
    author_tokens: frozenset[str]
    source: str


def iter_eprints(eprint_dir: Path) -> list[Path]:
    suffixes = (".tar.gz", ".gz", ".tar", ".tex", ".bin")
    return sorted(
        [p for p in eprint_dir.iterdir() if p.is_file() and p.name.endswith(suffixes)],
        key=lambda p: p.name,
    )


def arxiv_aliases(paper_id: str) -> set[str]:
    aliases = {paper_id}
    if "__" in paper_id:
        aliases.add(paper_id.replace("__", "/"))
    if "/" in paper_id:
        aliases.add(paper_id.replace("/", "__"))
    legacy = re.fullmatch(r"([A-Za-z.-]+)__([0-9]{7})", paper_id)
    if legacy:
        archive, number = legacy.groups()
        aliases.add(f"{archive.split('.')[0]}__{number}")
        aliases.add(f"{archive.split('.')[0]}/{number}")
    return aliases


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    text = re.sub(r"(?i)^arxiv\s*:?\s*", "", text)
    text = re.sub(r"(?i)^(?:https?://)?(?:www\.)?arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"(?i)^https?://front\.math\.ucdavis\.edu/", "", text)
    text = text.strip().strip("!.,;()[]{}<> ")
    text = re.sub(r"\.pdf$", "", text, flags=re.I)
    text = re.sub(r"v\d+$", "", text, flags=re.I)
    legacy = re.fullmatch(r"([A-Za-z]+)(?:\.[A-Za-z]{2})?/(\d{7})", text)
    if legacy:
        return f"{legacy.group(1).lower()}__{legacy.group(2)}"
    text = text.replace("/", "__")
    if re.fullmatch(r"\d{4}\.\d{4,5}", text) or re.fullmatch(r"[A-Za-z.-]+__\d{7}", text):
        return text
    return None


def find_arxiv_ids(text: str) -> list[str]:
    if not text:
        return []
    patterns = [
        r"(?i)arxiv\s*:?\s*([A-Za-z]+(?:\.[A-Za-z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"(?i)arxiv\.org/(?:abs|pdf)/([A-Za-z]+(?:\.[A-Za-z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"(?i)front\.math\.ucdavis\.edu/([A-Za-z]+(?:\.[A-Za-z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"(?i)\b([A-Za-z]+(?:\.[A-Za-z]{2})?/\d{7})\b",
        r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d.])",
    ]
    out: list[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            got = normalize_arxiv_id(match.group(1))
            if got and got not in seen:
                seen.add(got)
                out.append(got)
    return out


def tex_to_words(text: str) -> str:
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\\(?:emph|textit|textbf|textsc|mathrm|mathbf|mathcal|mathbb)\s*\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\.", " ", text)
    text = re.sub(r"[{}$^_]", " ", text)
    text = re.sub(r"[-‐‑–—]", " ", text)
    return text


def normalize_title(text: str | None) -> str:
    if not text:
        return ""
    text = tex_to_words(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def title_tokens(title_norm: str) -> frozenset[str]:
    return frozenset(t for t in title_norm.split() if len(t) > 2 and t not in STOPWORDS)


def is_probable_venue_title(text: str | None) -> bool:
    norm = normalize_title(text)
    if not norm:
        return True
    tokens = title_tokens(norm)
    if len(tokens) <= 2 and tokens & VENUE_WORDS:
        return True
    if tokens and len(tokens & VENUE_WORDS) / len(tokens) >= 0.6:
        return True
    return False


def clean_author_text(text: str | None) -> str:
    if not text:
        return ""
    text = tex_to_words(text)
    text = re.sub(r"(?i)\b(and|with)\b", " and ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def author_last_tokens(text: str | None) -> frozenset[str]:
    text = clean_author_text(text)
    if not text:
        return frozenset()
    parts = re.split(r"\band\b|;|\n", text)
    tokens: set[str] = set()
    for part in parts:
        part = part.strip(" ,")
        if not part:
            continue
        if "," in part:
            last = part.split(",", 1)[0]
        else:
            words = re.findall(r"[A-Za-z][A-Za-z'.-]*", part)
            words = [w for w in words if len(w) > 1 and not re.fullmatch(r"[A-Z]\.?", w)]
            last = words[-1] if words else ""
        last = re.sub(r"[^A-Za-z]", "", last).lower()
        if len(last) > 1:
            tokens.add(last)
    return frozenset(tokens)


def extract_command_arg(text: str, command: str) -> str | None:
    pattern = re.compile(r"\\" + re.escape(command) + r"\*?(?:\s*\[[^\]]*\])?\s*\{", re.S)
    match = pattern.search(text)
    if not match:
        return None
    return parse_balanced_brace(text, match.end() - 1)[0]


def choose_main_tex(files: list[dict[str, str]]) -> list[dict[str, str]]:
    tex = [f for f in files if f["file"].lower().endswith((".tex", ".ltx"))]
    return tex or files


def identity_from_eprint(path: Path) -> PaperIdentity | None:
    paper_id = strip_archive_suffix(path)
    files, meta = read_eprint_files(path)
    if not files:
        return None
    title = None
    authors = None
    for f in choose_main_tex(files):
        text = strip_comments(f["text"])
        title = title or extract_command_arg(text, "title")
        authors = authors or extract_command_arg(text, "author")
        if title and authors:
            break
    title_norm = normalize_title(title)
    if not title_norm:
        return None
    author_tokens = author_last_tokens(authors)
    author_names = tuple(sorted(author_tokens))
    return PaperIdentity(
        paper_id=paper_id,
        title=(title or "").strip(),
        title_norm=title_norm,
        title_tokens=title_tokens(title_norm),
        author_names=author_names,
        author_tokens=author_tokens,
        source=meta.get("status", "unknown"),
    )


def identity_from_bib_row(row: dict[str, Any]) -> PaperIdentity | None:
    paper_id = paper_id_of(row)
    title = row.get("title")
    authors = row.get("authors")
    if isinstance(authors, list):
        author_text = " and ".join(str(a) for a in authors)
    elif isinstance(authors, str):
        author_text = authors
    else:
        author_text = ""
    if not isinstance(title, str) or not title.strip():
        return None
    title_norm = normalize_title(title)
    if not paper_id or not title_norm:
        return None
    author_tokens = author_last_tokens(author_text)
    return PaperIdentity(
        paper_id=paper_id,
        title=title.strip(),
        title_norm=title_norm,
        title_tokens=title_tokens(title_norm),
        author_names=tuple(sorted(author_tokens)),
        author_tokens=author_tokens,
        source="bib-index",
    )


def add_identity(
    ident: PaperIdentity,
    identities: dict[str, PaperIdentity],
    arxiv_to_id: dict[str, str],
    exact_title: dict[str, list[str]],
    token_index: dict[str, set[str]],
) -> None:
    identities[ident.paper_id] = ident
    for alias in arxiv_aliases(ident.paper_id):
        arxiv_to_id[alias] = ident.paper_id
    exact_title[ident.title_norm].append(ident.paper_id)
    for token in ident.title_tokens:
        token_index[token].add(ident.paper_id)


def build_identity_index_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, PaperIdentity], dict[str, str], dict[str, list[str]], dict[str, set[str]], dict[str, Any]]:
    identities: dict[str, PaperIdentity] = {}
    arxiv_to_id: dict[str, str] = {}
    exact_title: dict[str, list[str]] = defaultdict(list)
    token_index: dict[str, set[str]] = defaultdict(set)
    stats = Counter({"candidate_rows": len(rows)})
    for row in rows:
        ident = identity_from_bib_row(row)
        if ident is None:
            stats["identity_missing"] += 1
            continue
        add_identity(ident, identities, arxiv_to_id, exact_title, token_index)
        stats["identity_indexed"] += 1
    return identities, arxiv_to_id, exact_title, token_index, dict(stats)


def build_identity_index(eprints: Path, limit: int | None, paper_ids: list[str]) -> tuple[dict[str, PaperIdentity], dict[str, str], dict[str, list[str]], dict[str, set[str]], dict[str, Any]]:
    paths = iter_eprints(eprints)
    if paper_ids:
        wanted = set(paper_ids)
        paths = [p for p in paths if strip_archive_suffix(p) in wanted]
    if limit is not None:
        paths = paths[:limit]

    identities: dict[str, PaperIdentity] = {}
    arxiv_to_id: dict[str, str] = {}
    exact_title: dict[str, list[str]] = defaultdict(list)
    token_index: dict[str, set[str]] = defaultdict(set)
    stats = Counter({"candidate_eprints": len(paths)})
    for path in paths:
        try:
            ident = identity_from_eprint(path)
        except Exception:
            stats["identity_errors"] += 1
            continue
        if ident is None:
            stats["identity_missing"] += 1
            continue
        add_identity(ident, identities, arxiv_to_id, exact_title, token_index)
        stats["identity_indexed"] += 1
    return identities, arxiv_to_id, exact_title, token_index, dict(stats)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_paper_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("papers", "rows", "items", "bibliography", "entries"):
            val = payload.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        if "bibitems" in payload or "paper_id" in payload:
            return [payload]
        return [dict(v, paper_id=k) for k, v in payload.items() if isinstance(v, dict)]
    return []


def load_bibliography_rows(bib_index: Path, bib_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if bib_index.exists():
        rows = coerce_paper_rows(load_json(bib_index))
        return rows, {"source": str(bib_index), "papers": len(rows)}
    if bib_dir.exists():
        rows: list[dict[str, Any]] = []
        for path in sorted(bib_dir.glob("*.json")):
            try:
                rows.extend(coerce_paper_rows(load_json(path)))
            except Exception:
                continue
        return rows, {"source": str(bib_dir), "papers": len(rows)}
    return [], {"source": None, "papers": 0, "missing": True}


def paper_id_of(row: dict[str, Any]) -> str | None:
    for key in ("paper_id", "paper", "id", "entity"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def bibitem_arxiv_id(item: dict[str, Any]) -> str | None:
    for key in ("arxiv_id", "arxiv", "eprint", "archive_id"):
        val = item.get(key)
        if isinstance(val, str):
            got = normalize_arxiv_id(val)
            if got:
                return got
    text = " ".join(str(item.get(key) or "") for key in ("raw", "author", "authors", "title"))
    found = find_arxiv_ids(text)
    if found:
        return found[0]
    return None


def bibitem_arxiv_ids(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen = set()
    for key in ("arxiv_id", "arxiv", "eprint", "archive_id"):
        val = item.get(key)
        if isinstance(val, str):
            got = normalize_arxiv_id(val)
            if got and got not in seen:
                seen.add(got)
                out.append(got)
    text = " ".join(str(item.get(key) or "") for key in ("raw", "author", "authors", "title"))
    for got in find_arxiv_ids(text):
        if got not in seen:
            seen.add(got)
            out.append(got)
    return out


def bibitem_titles(item: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        value = clean_reference_fragment(value)
        norm = normalize_title(value)
        if len(norm) < 12 or norm in seen:
            return
        seen.add(norm)
        titles.append(value)

    for key in ("title", "paper_title", "work_title"):
        val = item.get(key)
        if isinstance(val, str) and val.strip() and not is_probable_venue_title(val):
            add(val)
    for key in ("author", "raw"):
        val = item.get(key)
        if isinstance(val, str):
            for candidate in titles_from_reference(val):
                add(candidate)
    return titles


def bibitem_title(item: dict[str, Any]) -> str:
    titles = bibitem_titles(item)
    return titles[0] if titles else ""


def clean_reference_fragment(text: str) -> str:
    text = re.sub(r"(?i)https?://\S+", " ", text)
    text = re.sub(r"(?i)\bdoi\s*:?\s*\S+", " ", text)
    text = re.sub(r"(?i)\barxiv\s*:?\s*(?:[A-Za-z]+(?:\.[A-Za-z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", " ", text)
    text = re.sub(r"(?i)\b[A-Za-z]+(?:\.[A-Za-z]{2})?/\d{7}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!;:")


def title_from_raw(raw: str) -> str:
    if not raw:
        return ""
    quoted = re.search(r"[\"“](.{12,240}?)[\"”]", raw, re.S)
    if quoted:
        return quoted.group(1)
    emph = re.search(r"\\(?:emph|textit)\s*\{(.{12,240}?)\}", raw, re.S)
    if emph:
        return emph.group(1)
    return ""


def titles_from_reference(raw: str) -> list[str]:
    if not raw:
        return []
    raw = clean_reference_fragment(raw)
    out: list[str] = []
    quoted = re.findall(r"[\"“](.{12,240}?)[\"”]", raw, re.S)
    out.extend(quoted)
    out.extend(re.findall(r"\\(?:emph|textit)\s*\{(.{12,240}?)\}", raw, re.S))

    # Most W1 raw strings are "Authors, Title, Venue ..."; W1's author field
    # often also includes that title when no better title was extracted.
    parts = [p.strip() for p in re.split(r"\s*,\s*", raw) if p.strip()]
    for idx in range(1, min(len(parts), 4)):
        candidate = parts[idx]
        if is_probable_venue_title(candidate):
            continue
        out.append(candidate)

    # Also keep a larger post-author fragment for titles containing commas.
    if len(parts) >= 3:
        joined = ", ".join(parts[1:3])
        if not is_probable_venue_title(joined):
            out.append(joined)

    cleaned: list[str] = []
    seen = set()
    for candidate in out:
        candidate = re.split(
            r"(?i)\b(?:to appear|preprint|available|submitted|journal|j\.|proc\.|proceedings|trans\.|adv\.|ann\.|lect(?:ure)? notes|vol\.|no\.|pp\.)\b",
            candidate,
            maxsplit=1,
        )[0]
        candidate = clean_reference_fragment(candidate)
        norm = normalize_title(candidate)
        if len(norm) < 12 or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(candidate)
    return cleaned


def bibitem_authors(item: dict[str, Any]) -> str:
    for key in ("author", "authors"):
        val = item.get(key)
        if isinstance(val, list):
            return " and ".join(str(v) for v in val)
        if isinstance(val, str):
            return val
    raw = item.get("raw")
    if isinstance(raw, str):
        return raw[:220]
    return ""


def bibitem_cache_key(item: dict[str, Any]) -> str:
    return "\n".join(str(item.get(key) or "") for key in ("raw", "author", "authors", "title", "arxiv_id", "arxiv", "eprint", "archive_id"))


def candidate_ids_for_title(tokens: frozenset[str], token_index: dict[str, set[str]], max_candidates: int) -> list[str]:
    counts: Counter[str] = Counter()
    ranked_tokens = sorted(tokens, key=lambda token: len(token_index.get(token, ())))[:8]
    for token in ranked_tokens:
        for paper_id in token_index.get(token, ()):
            counts[paper_id] += 1
    return [paper_id for paper_id, _count in counts.most_common(max_candidates)]


def prefiltered_title_candidates(
    tokens: frozenset[str],
    identities: dict[str, PaperIdentity],
    token_index: dict[str, set[str]],
    max_candidates: int,
) -> list[str]:
    counts: Counter[str] = Counter()
    ranked_tokens = sorted(tokens, key=lambda token: len(token_index.get(token, ())))[:8]
    for token in ranked_tokens:
        for paper_id in token_index.get(token, ()):
            counts[paper_id] += 1
    if not counts:
        return []
    ranked: list[tuple[str, float, int]] = []
    for paper_id, shared in counts.most_common(max_candidates * 4):
        ident = identities.get(paper_id)
        if ident is None or not ident.title_tokens:
            continue
        containment = shared / max(1, min(len(tokens), len(ident.title_tokens)))
        jaccard = shared / max(1, len(tokens | ident.title_tokens))
        if containment >= 0.55 or (shared >= 4 and jaccard >= 0.35):
            ranked.append((paper_id, containment + jaccard, shared))
    ranked.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return [paper_id for paper_id, _score, _shared in ranked[:max_candidates]]


def score_candidate(title_norm: str, authors: frozenset[str], ident: PaperIdentity) -> tuple[float, float, float]:
    title_ratio = SequenceMatcher(None, title_norm, ident.title_norm).ratio()
    if authors and ident.author_tokens:
        author_overlap = len(authors & ident.author_tokens) / max(1, min(len(authors), len(ident.author_tokens)))
    else:
        author_overlap = 0.0
    q_tokens = title_tokens(title_norm)
    token_overlap = 0.0
    if q_tokens and ident.title_tokens:
        token_overlap = len(q_tokens & ident.title_tokens) / len(q_tokens | ident.title_tokens)
        containment = len(q_tokens & ident.title_tokens) / min(len(q_tokens), len(ident.title_tokens))
    else:
        containment = 0.0
    score = (0.50 * title_ratio) + (0.25 * token_overlap) + (0.15 * containment) + (0.10 * author_overlap)
    return score, title_ratio, author_overlap


def fuzzy_link(
    item: dict[str, Any],
    identities: dict[str, PaperIdentity],
    exact_title: dict[str, list[str]],
    token_index: dict[str, set[str]],
    max_candidates: int,
    threshold: float,
) -> tuple[str | None, dict[str, Any] | None]:
    titles = bibitem_titles(item)
    if not titles:
        return None, None
    authors = author_last_tokens(bibitem_authors(item))

    best: tuple[str, float, float, float, str] | None = None
    for title in titles:
        title_norm = normalize_title(title)
        q_tokens = title_tokens(title_norm)
        candidates = exact_title.get(title_norm, [])
        if not candidates:
            candidates = prefiltered_title_candidates(q_tokens, identities, token_index, max_candidates)
        for paper_id in candidates:
            ident = identities.get(paper_id)
            if ident is None:
                continue
            score, title_ratio, author_overlap = score_candidate(title_norm, authors, ident)
            if best is None or score > best[1]:
                best = (paper_id, score, title_ratio, author_overlap, title)
    if best is None or best[1] < threshold:
        return None, None
    paper_id, score, title_ratio, author_overlap, title = best
    if title_ratio < 0.88 and author_overlap <= 0:
        return None, None
    return paper_id, {
        "method": "fuzzy-author-title",
        "score": round(score, 4),
        "title_ratio": round(title_ratio, 4),
        "author_overlap": round(author_overlap, 4),
        "title": title,
    }


def link_citations(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    rows, bib_stats = load_bibliography_rows(args.bib_index, args.bib_dir)
    if rows and not args.force_eprint_identity and not args.identity_limit and not args.identity_paper_id:
        identities, arxiv_to_id, exact_title, token_index, identity_stats = build_identity_index_from_rows(rows)
        identity_stats["source"] = "bib-index"
    else:
        identities, arxiv_to_id, exact_title, token_index, identity_stats = build_identity_index(
            args.eprints,
            args.identity_limit,
            args.identity_paper_id,
        )
        identity_stats["source"] = "eprints"
    if args.paper_id:
        wanted = set(args.paper_id)
        rows = [r for r in rows if paper_id_of(r) in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    stats = Counter()
    stats.update({f"identity_{k}": v for k, v in identity_stats.items() if isinstance(v, int)})
    stats["bib_papers"] = len(rows)
    stats["edges_arxiv"] = 0
    stats["edges_fuzzy"] = 0
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    link_cache: dict[str, tuple[str | None, dict[str, Any] | None]] = {}

    for row in rows:
        src = paper_id_of(row)
        if not src:
            stats["rows_without_paper_id"] += 1
            continue
        bibitems = row.get("bibitems")
        if not isinstance(bibitems, list):
            stats["rows_without_bibitems"] += 1
            continue
        stats["bibitems"] += len(bibitems)
        for item in bibitems:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "")
            target = None
            via: dict[str, Any] = {"key": key}
            cache_key = bibitem_cache_key(item)
            cached = link_cache.get(cache_key)
            if cached is not None:
                target, cached_via = cached
                if cached_via:
                    via.update(cached_via)
                    stats["cache_hits"] += 1
                    if cached_via.get("method") == "arxiv-id":
                        stats["edges_arxiv"] += 1
                    elif cached_via.get("method") == "fuzzy-author-title":
                        stats["edges_fuzzy"] += 1
            else:
                arxiv_ids = bibitem_arxiv_ids(item)
                for arxiv_id in arxiv_ids:
                    target = arxiv_to_id.get(arxiv_id)
                    if target:
                        via.update({"method": "arxiv-id", "arxiv_id": arxiv_id})
                        stats["edges_arxiv"] += 1
                        break
                    else:
                        stats["arxiv_id_not_in_corpus"] += 1
                if arxiv_ids:
                    stats["bibitems_with_arxiv_id"] += 1
                if target is None:
                    target, fuzzy_via = fuzzy_link(
                        item,
                        identities,
                        exact_title,
                        token_index,
                        args.max_candidates,
                        args.threshold,
                    )
                    if target and fuzzy_via:
                        via.update(fuzzy_via)
                        stats["edges_fuzzy"] += 1
                link_cache[cache_key] = (target, {k: v for k, v in via.items() if k != "key"} if target else None)
            if target is None:
                stats["unlinked_bibitems"] += 1
                continue
            edge_key = (src, target, key)
            if edge_key in seen_edges:
                stats["duplicate_edges"] += 1
                continue
            seen_edges.add(edge_key)
            edges.append({"from": src, "to": target, "via": via})

    cited_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        cited_by[edge["to"]].append({"from": edge["from"], "via": edge["via"]})

    stats["edges"] = len(edges)
    stats["link_cache_entries"] = len(link_cache)
    linked = stats["edges"]
    total = stats["bibitems"]
    stats["linkage_rate"] = round(linked / total, 6) if total else 0.0
    stats["elapsed_sec"] = round(time.time() - start, 3)
    return {
        "edges": edges,
        "cited_by": {k: v for k, v in sorted(cited_by.items())},
        "stats": dict(stats),
        "inputs": {
            "eprints": str(args.eprints),
            "bib": bib_stats,
            "identity": identity_stats,
            "threshold": args.threshold,
            "max_candidates": args.max_candidates,
        },
    }


def spot_check(payload: dict[str, Any], identities: dict[str, PaperIdentity] | None = None) -> dict[str, Any]:
    if identities is None:
        identities = {}
    checked = []
    missing = 0
    for edge in payload.get("edges", [])[:20]:
        to_id = edge.get("to")
        ok = not identities or to_id in identities
        if not ok:
            missing += 1
        checked.append({"from": edge.get("from"), "to": to_id, "ok": ok, "via": edge.get("via")})
    return {"checked": len(checked), "missing_targets": missing, "sample": checked[:5]}


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eprints", type=Path, default=DEFAULT_EPRINTS)
    ap.add_argument("--bib-index", type=Path, default=DEFAULT_BIB_INDEX)
    ap.add_argument("--bib-dir", type=Path, default=DEFAULT_BIB_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="Limit W1 bibliography paper rows")
    ap.add_argument("--paper-id", action="append", default=[], help="Restrict W1 bibliography rows by source paper id")
    ap.add_argument("--identity-limit", type=int, default=None, help="Limit identity-index eprints for sampling")
    ap.add_argument("--identity-paper-id", action="append", default=[], help="Restrict identity-index eprints by paper id")
    ap.add_argument("--force-eprint-identity", action="store_true", help="Build identities from eprint sources even when W1 paper identities are available")
    ap.add_argument("--threshold", type=float, default=0.82)
    ap.add_argument("--max-candidates", type=int, default=150)
    ap.add_argument("--no-write", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = link_citations(args)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(args.out)
    print(json.dumps(payload["stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
