#!/usr/bin/env python3
"""Seed an OED-shape dictionary from the local PlanetMath corpus."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, Optional

import edn_format


DEFAULT_PLANETMATH_ROOT = Path.home() / "code" / "planetmath"
DEFAULT_KERNEL_TSV = Path.home() / "code" / "storage" / "futon6" / "data" / "ner-kernel" / "terms.tsv"
DEFAULT_OUT_DIR = Path.home() / "code" / "futon6" / "data" / "dictionary"
DEFAULT_SCHEMA_PATH = Path.home() / "code" / "futon6" / "holes" / "excursions" / "dictionary-schema.edn"
PROGRESS_EVERY = 500

DEFINITION_ENV_NAMES = (
    "definition",
    "definition*",
    "defn",
    "defn*",
    "defi",
    "defi*",
)

THEOREM_ENV_NAMES = (
    "theorem",
    "theorem*",
    "thm",
    "thm*",
    "lemma",
    "lemma*",
    "prop",
    "prop*",
    "proposition",
    "proposition*",
    "cor",
    "cor*",
    "corollary",
    "corollary*",
    "conjecture",
    "conjecture*",
    "result",
    "result*",
)

THEOREM_LIKE_PM_TYPES = {
    "Theorem",
    "Result",
    "Corollary",
    "Conjecture",
}

NON_DICTIONARY_PM_TYPES = {
    "Proof",
    "Example",
}

STOPWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "generic-emphasis": (
        "unique", "asymmetric", "complete", "important", "simple", "complex",
        "special", "basic", "key", "main", "new", "novel", "fundamental",
        "classical", "modern", "recent", "obvious", "trivial", "natural",
        "standard", "common", "general", "particular", "specific", "certain",
    ),
    "bibliography-journal-name": (
        "acta universitatis apulensis", "adv in math", "advances in mathematics",
        "j math anal", "j math phys", "comm math phys", "ann math",
        "proc amer math soc", "bull amer math soc", "math z",
    ),
    "reference-marker": (
        "ibid", "op cit", "loc cit", "et al", "cf",
    ),
    "proper-noun-not-concept": (
        "let", "suppose", "define", "assume", "consider",
    ),
    "section-marker-fragment": (
        "section", "chapter", "appendix", "theorem", "lemma",
    ),
}

LATEX_TEXT_MACROS = (
    "emph",
    "textbf",
    "textit",
    "textrm",
    "texttt",
    "textsc",
    "underline",
    "operatorname",
    "mathrm",
    "mathit",
    "mathbf",
    "mathcal",
    "mathbb",
    "mathsf",
    "mathfrak",
)

DEFINITION_FALLBACK_NEGATIVE_CUES = (
    "stub entry",
    "guide to",
    "holding bay",
    "in reverse chronological order",
    "list of",
    "table of",
    "is now empty",
    "website/server",
    "appendix",
    "bibliography",
)

THEOREM_STATEMENT_NEGATIVE_CUES = (
    "proof sketch is omitted",
    "the proof is",
    "we prove",
    "we will prove",
    "the next result shows",
    "comments",
    "remark",
)


class PMArticle(NamedTuple):
    """One PlanetMath article ready for dictionary conversion."""

    canon_id: str
    headword: str
    msc_code: str
    subject_area: str
    tex_path: Path
    body_text: str
    raw_tex: str


@dataclass(frozen=True)
class KeywordValue:
    name: str


@dataclass(frozen=True)
class InstValue:
    iso_utc: str


def kw(name: str) -> KeywordValue:
    return KeywordValue(name)


def inst_value(iso_utc: str) -> InstValue:
    return InstValue(iso_utc)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planetmath-root", type=Path, default=DEFAULT_PLANETMATH_ROOT)
    parser.add_argument("--kernel-tsv", type=Path, default=DEFAULT_KERNEL_TSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--timestamp",
        help="Stable UTC timestamp for deterministic outputs, e.g. 2026-05-19T00:00:00Z",
    )
    return parser.parse_args(argv)


def iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def extract_pm_field(raw_tex: str, field: str) -> Optional[str]:
    match = re.search(rf"\\{field}\{{([^}}]+)\}}", raw_tex)
    return match.group(1).strip() if match else None


def extract_document_body(raw_tex: str) -> Optional[str]:
    match = re.search(r"\\begin\{document\}(.*?)(?:\\end\{document\}|$)", raw_tex, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def remove_tex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def brace_balance_ok(text: str) -> bool:
    depth = 0
    escaped = False
    for ch in remove_tex_comments(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def is_numeric_id_headword(headword: str) -> bool:
    normalized = re.sub(r"[^0-9]+", "", headword)
    raw = re.sub(r"[\s\-_.]+", "", headword)
    return bool(normalized) and normalized == raw


def camel_to_kebab(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", value)
    value = value.replace("_", "-")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-").lower()


def fallback_headword_from_filename(tex_path: Path) -> str:
    stem = tex_path.stem
    _msc, _, tail = stem.partition("-")
    if not tail:
        return stem
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tail)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return spaced.replace("_", " ").strip()


def normalize_lookup_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def normalize_display_headword(headword: str) -> str:
    return re.sub(r"\s+", " ", headword.strip())


def source_id_from_canonical(canonical_name: str) -> str:
    return f"planetmath:{canonical_name}"


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def replace_tex_text_macros(text: str) -> str:
    result = text
    result = re.sub(r"\\PMlinkname\{([^}]*)\}\{[^}]*\}", r"\1", result)
    result = re.sub(r"\\PMlinkescapeword\{([^}]*)\}", r"\1", result)
    result = re.sub(r"\{\\(?:bf|it|rm|sc|tt|sl)\s+([^{}]+)\}", r"\1", result)
    for macro in LATEX_TEXT_MACROS:
        result = re.sub(rf"\\{macro}\{{([^{{}}]*)\}}", r"\1", result)
    return result


def latex_to_text(text: str) -> str:
    result = remove_tex_comments(text)
    result = replace_tex_text_macros(result)
    result = re.sub(r"\\item\b", " ", result)
    result = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", result)
    result = re.sub(r"\\(?:label|cite|ref|eqref|url|footnote)\{[^}]*\}", " ", result)
    result = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{[^}]*\}", " ", result)
    result = re.sub(r"\\[A-Za-z@]+(?:\[[^\]]*\])?", " ", result)
    result = result.replace("~", " ")
    result = result.replace("\\", " ")
    return collapse_whitespace(result)


def strip_leading_tex_setup(body_text: str) -> str:
    lines = body_text.splitlines()
    kept: list[str] = []
    started = False
    skip_re = re.compile(r"\\(?:newcommand|renewcommand|newtheorem|theoremstyle|DeclareMathOperator)\b")
    for line in lines:
        stripped = line.strip()
        if not started and (not stripped or skip_re.match(stripped)):
            continue
        started = True
        kept.append(line)
    return "\n".join(kept).strip()


def iter_paragraphs(body_text: str) -> Iterator[str]:
    body_text = strip_leading_tex_setup(body_text)
    current: list[str] = []
    stop_re = re.compile(r"\\(?:begin\{thebibliography\}|bibliography)\b")
    for raw_line in body_text.splitlines():
        stripped = raw_line.strip()
        if stop_re.match(stripped):
            break
        if not stripped:
            if current:
                yield "\n".join(current)
                current = []
            continue
        if re.match(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{[^}]*\}", stripped):
            if current:
                yield "\n".join(current)
                current = []
            continue
        current.append(stripped)
    if current:
        yield "\n".join(current)


def extract_definition_block(body_text: str) -> Optional[str]:
    for env_name in DEFINITION_ENV_NAMES:
        pattern = rf"\\begin\{{{re.escape(env_name)}\}}(.*?)\\end\{{{re.escape(env_name)}\}}"
        match = re.search(pattern, body_text, re.DOTALL)
        if match:
            return latex_to_text(match.group(1))
    return None


def extract_theorem_block(body_text: str) -> Optional[str]:
    for env_name in THEOREM_ENV_NAMES:
        pattern = rf"\\begin\{{{re.escape(env_name)}\}}(?:\[[^\]]*\])?(.*?)\\end\{{{re.escape(env_name)}\}}"
        match = re.search(pattern, body_text, re.DOTALL)
        if match:
            return latex_to_text(match.group(1))
    return None


def first_nonempty_paragraph(body_text: str) -> Optional[str]:
    for paragraph in iter_paragraphs(body_text):
        return paragraph
    return None


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$\\])", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def paragraph_looks_definitional(paragraph_text: str, headword: str) -> bool:
    lower = paragraph_text.lower()
    headword_lower = normalize_lookup_term(headword)
    historical_cues = (
        "historical artefact",
        "history of mathematics",
        "ancient egypt",
        "papyrus",
        "museum",
        "discussed mainly as",
    )
    if any(cue in lower for cue in historical_cues):
        return False
    if headword_lower and lower.startswith(f"{headword_lower} is"):
        return True
    if headword_lower and lower.startswith(f"{headword_lower} are"):
        return True
    if headword_lower and lower.startswith(f"the {headword_lower} is"):
        return True
    if headword_lower and lower.startswith(f"the {headword_lower} are"):
        return True
    if headword_lower and headword_lower in lower[: max(140, len(headword_lower) + 20)]:
        if re.search(r"\b(is|are|refers to|denotes|means|consists of|describes|called)\b", lower):
            return True
    return False


def definition_article_fallback_ok(article: PMArticle, paragraph_text: str) -> bool:
    if article.subject_area.startswith("01_"):
        return False
    if len(paragraph_text) < 35:
        return False
    lower = paragraph_text.lower()
    if any(cue in lower for cue in DEFINITION_FALLBACK_NEGATIVE_CUES):
        return False
    return any(re.search(pattern, lower) for pattern in (
        r"\b(is|are|refers to|denotes|means|consists of|describes|is said to|is called|iff|if and only if)\b",
        r"^for\b",
        r"^to\b",
        r"^in mathematics\b",
        r"^let\b",
        r"^suppose\b",
        r"^when\b",
    ))


def clean_theorem_statement_text(text: str) -> str:
    text = re.sub(r"^\s*(?:\{\\(?:bf|it|rm|sc|tt|sl)\s*)?(Theorem|Lemma|Proposition|Corollary|Result|Conjecture)\s*[:.\-]*\}?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(Theorem|Lemma|Proposition|Corollary|Result|Conjecture)\s*[:.\-]\s*", "", text, flags=re.IGNORECASE)
    return collapse_whitespace(text)


def paragraph_looks_theorem_like(paragraph_text: str) -> bool:
    lower = paragraph_text.lower()
    if len(paragraph_text) < 25:
        return False
    if any(cue in lower for cue in THEOREM_STATEMENT_NEGATIVE_CUES):
        return False
    if re.match(r"^(theorem|lemma|proposition|corollary|result|conjecture)\b", lower):
        return True
    if re.match(r"^(if|let|suppose|for)\b", lower) and " then " in lower:
        return True
    if re.match(r"^(if|let|suppose|for)\b", lower) and lower.endswith("."):
        return True
    return False


def extract_theorem_statement(article: PMArticle) -> Optional[str]:
    theorem_block = extract_theorem_block(article.body_text)
    if theorem_block:
        return theorem_block
    for paragraph_raw in iter_paragraphs(article.body_text):
        if re.search(r"\\begin\{proof\}", paragraph_raw):
            break
        paragraph_text = clean_theorem_statement_text(latex_to_text(paragraph_raw))
        if paragraph_looks_theorem_like(paragraph_text):
            return first_two_sentences(paragraph_text)
    return None


def first_two_sentences(text: str) -> Optional[str]:
    sentences = split_sentences(text)
    if not sentences:
        return None
    return " ".join(sentences[:2]).strip()


def record_skip(skip_recorder: Optional[Callable[[str, Path], None]], reason: str, path: Path) -> None:
    if skip_recorder is not None:
        skip_recorder(reason, path)


def find_pm_articles(
    planetmath_root: Path,
    skip_recorder: Optional[Callable[[str, Path], None]] = None,
) -> Iterator[PMArticle]:
    """Walk the PM corpus; yield one PMArticle per article."""

    subject_dirs = sorted(
        path for path in planetmath_root.iterdir()
        if path.is_dir() and re.match(r"\d+_.*", path.name)
    )
    for subject_dir in subject_dirs:
        for tex_path in sorted(subject_dir.glob("*.tex")):
            try:
                raw_tex = safe_read_text(tex_path)
            except OSError:
                record_skip(skip_recorder, "unreadable-tex", tex_path)
                continue
            body_text = extract_document_body(raw_tex)
            if not body_text or not brace_balance_ok(body_text):
                record_skip(skip_recorder, "malformed-tex", tex_path)
                continue
            title = extract_pm_field(raw_tex, "pmtitle") or fallback_headword_from_filename(tex_path)
            canonical_name = extract_pm_field(raw_tex, "pmcanonicalname") or tex_path.stem.partition("-")[2] or tex_path.stem
            yield PMArticle(
                canon_id=camel_to_kebab(canonical_name),
                headword=normalize_display_headword(title),
                msc_code=tex_path.stem.partition("-")[0],
                subject_area=subject_dir.name,
                tex_path=tex_path.resolve(),
                body_text=body_text,
                raw_tex=raw_tex,
            )


def extract_definition_from_pm(article: PMArticle) -> Optional[str]:
    """Extract the first definitional sentence or paragraph from a PM article body."""

    if is_numeric_id_headword(article.headword):
        return None
    article_type = extract_pm_field(article.raw_tex, "pmtype") or ""
    if article_type in THEOREM_LIKE_PM_TYPES:
        theorem_statement = extract_theorem_statement(article)
        if theorem_statement:
            return theorem_statement
    block = extract_definition_block(article.body_text)
    if block:
        return block
    paragraph = first_nonempty_paragraph(article.body_text)
    if not paragraph:
        return None
    paragraph_text = latex_to_text(paragraph)
    if not paragraph_looks_definitional(paragraph_text, article.headword):
        if article_type != "Definition" or not definition_article_fallback_ok(article, paragraph_text):
            return None
    return first_two_sentences(paragraph_text)


def load_kernel_lookup(kernel_tsv_path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with kernel_tsv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("source") != "pm-title":
                continue
            lookup[normalize_lookup_term(row["term_lower"])] = row
    return lookup


def extract_cross_refs(raw_tex: str) -> list[dict]:
    refs = sorted(set(re.findall(r"\\ref\{([^}]+)\}", raw_tex)))
    return [{"rel": kw("references"), "target": ref} for ref in refs]


def usage_example_context(article: PMArticle) -> str:
    paragraph = first_nonempty_paragraph(article.body_text)
    return latex_to_text(paragraph or article.body_text)[:1000]


def pm_article_to_entry(
    article: PMArticle,
    kernel_row: Optional[dict[str, str]] = None,
    *,
    extracted_at_iso: Optional[str] = None,
) -> Optional[dict]:
    """Convert one PM article to a dictionary entry per the OED-shape schema."""

    if is_numeric_id_headword(article.headword):
        return None
    article_type = extract_pm_field(article.raw_tex, "pmtype") or ""
    if article_type in NON_DICTIONARY_PM_TYPES:
        return None
    extracted_at_iso = extracted_at_iso or iso_utc_now()
    canonical_name = (
        (kernel_row or {}).get("canon_or_count")
        or extract_pm_field(article.raw_tex, "pmcanonicalname")
        or article.tex_path.stem.partition("-")[2]
        or article.headword
    )
    term_id = camel_to_kebab(canonical_name)
    headword = normalize_display_headword(article.headword)
    source_id = source_id_from_canonical(canonical_name)
    definition_text = extract_definition_from_pm(article)
    body_context = latex_to_text(article.body_text)
    entry = {
        "term/id": term_id,
        "term/headword": headword,
        "term/lower": normalize_lookup_term(headword),
        "term/part": kw("noun"),
        "term/aliases": [],
        "term/etymology": {
            "first-source": source_id,
            "first-source-date": None,
            "first-extractor": kw("pm-seed-loader/v1"),
            "note": "PlanetMath canonical entry; seeded from local PM corpus.",
        },
        "term/usage-examples": [],
        "term/canon-source": kw("planetmath-seed"),
        "term/first-seen": None,
        "term/last-seen": None,
        "term/occurrence-count": 1,
        "term/cross-refs": extract_cross_refs(article.raw_tex),
        "term/review-notes": [f"Seeded from PM {extracted_at_iso[:10]}."],
        "term/graduated-at": inst_value(extracted_at_iso),
        "term/source-metadata": {
            "msc-code": article.msc_code,
            "pm-type": article_type or None,
            "subject-area": article.subject_area,
            "tex-path": str(article.tex_path),
        },
    }
    if definition_text:
        entry["term/definitions"] = [{
            "def/id": f"{term_id}-d1",
            "def/text": definition_text,
            "def/extracted-from": source_id,
            "def/source-context": body_context,
            "def/extraction-method": kw("pm-seed"),
            "def/extracted-at": inst_value(extracted_at_iso),
            "def/confidence": 1.0,
            "def/status": kw("canonical"),
        }]
        entry["term/usage-examples"] = [{
            "example/paper": source_id,
            "example/role": kw("canonical-source"),
            "example/context": usage_example_context(article),
            "example/seen-at": None,
        }]
        entry["term/status"] = kw("canonical")
    else:
        entry["term/definitions"] = []
        entry["term/status"] = kw("canonical-no-definition")
        entry["term/review-notes"].append(
            f"PM article had no extractable definition. Body length: {len(body_context)} chars."
        )
    return entry


def hand_seeded_stopwords(*, timestamp_iso: Optional[str] = None) -> list[dict]:
    """Return the hand-seeded stopword list."""

    timestamp_iso = timestamp_iso or iso_utc_now()
    stopwords: list[dict] = []
    for reason, values in STOPWORD_GROUPS.items():
        for value in values:
            stopwords.append({
                "stopword/id": camel_to_kebab(value),
                "stopword/lower": value,
                "stopword/reason": kw(reason),
                "stopword/first-flagged-at": inst_value(timestamp_iso),
                "stopword/example-context": "(none - hand-seeded)",
                "stopword/source-paper": "(none - hand-seeded)",
                "stopword/flag-method": kw("hand-seed"),
            })
    return stopwords


def audit_sample(entries: list[dict], n: int = 100, seed: int = 13) -> list[dict]:
    """Deterministic random sample for operator review."""

    if len(entries) <= n:
        return list(entries)
    picker = random.Random(seed)
    return sorted(
        picker.sample(entries, n),
        key=lambda entry: entry["term/id"],
    )


def json_ready(value):
    if isinstance(value, KeywordValue):
        return f":{value.name}"
    if isinstance(value, InstValue):
        return value.iso_utc
    if isinstance(value, dict):
        return {key: json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def edn_ready(value):
    if isinstance(value, KeywordValue):
        return edn_format.Keyword(value.name)
    if isinstance(value, InstValue):
        return f"__EDN_INST__{value.iso_utc}"
    if isinstance(value, dict):
        return {
            edn_format.Keyword(key): edn_ready(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [edn_ready(item) for item in value]
    return value


def replace_inst_sentinels(edn_text: str) -> str:
    return re.sub(
        r'"__EDN_INST__([^"]+)"',
        lambda match: f'#inst "{match.group(1)}"',
        edn_text,
    )


def write_edn_file(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = edn_format.dumps(edn_ready(value))
    path.write_text(replace_inst_sentinels(rendered) + "\n", encoding="utf-8")


def validate_edn_round_trip(path: Path) -> None:
    edn_format.loads(path.read_text(encoding="utf-8"))


def build_entries_document(entries: list[dict], *, timestamp_iso: str, planetmath_root: Path) -> dict:
    return {
        "dictionary/version": "0.1-pm-seed",
        "dictionary/created": inst_value(timestamp_iso),
        "dictionary/created-by": kw("pm-seed-loader/v1"),
        "dictionary/source-root": str(planetmath_root),
        "dictionary/entry-count": len(entries),
        "dictionary/entries": entries,
    }


def build_stopwords_document(stopwords: list[dict], *, timestamp_iso: str) -> dict:
    return {
        "dictionary-stopwords/version": "0.1-pm-seed",
        "dictionary-stopwords/created": inst_value(timestamp_iso),
        "dictionary-stopwords/created-by": kw("pm-seed-loader/v1"),
        "dictionary-stopwords/count": len(stopwords),
        "dictionary-stopwords/entries": stopwords,
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    started = time.time()
    timestamp_iso = args.timestamp or iso_utc_now()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kernel_lookup = load_kernel_lookup(args.kernel_tsv)
    entries: list[dict] = []
    skip_counts: Counter[str] = Counter()
    skipped_examples: dict[str, list[str]] = {}

    def note_skip(reason: str, tex_path: Path) -> None:
        skip_counts[reason] += 1
        skipped_examples.setdefault(reason, [])
        if len(skipped_examples[reason]) < 10:
            skipped_examples[reason].append(str(tex_path))

    processed = 0
    for article in find_pm_articles(args.planetmath_root, skip_recorder=note_skip):
        processed += 1
        if processed % PROGRESS_EVERY == 0:
            print(f"Processed {processed} PlanetMath articles...", flush=True)
        kernel_row = kernel_lookup.get(normalize_lookup_term(article.headword))
        entry = pm_article_to_entry(article, kernel_row, extracted_at_iso=timestamp_iso)
        if entry is None:
            article_type = extract_pm_field(article.raw_tex, "pmtype") or ""
            if is_numeric_id_headword(article.headword):
                note_skip("numeric-id-skip", article.tex_path)
            elif article_type in NON_DICTIONARY_PM_TYPES:
                note_skip("non-dictionary-pmtype-skip", article.tex_path)
            else:
                note_skip("entry-filtered-out", article.tex_path)
            continue
        entries.append(entry)

    entries.sort(key=lambda entry: entry["term/id"])
    stopwords = hand_seeded_stopwords(timestamp_iso=timestamp_iso)
    sample = audit_sample(entries)

    entries_doc = build_entries_document(entries, timestamp_iso=timestamp_iso, planetmath_root=args.planetmath_root)
    stopwords_doc = build_stopwords_document(stopwords, timestamp_iso=timestamp_iso)

    entries_path = args.out_dir / "entries-pm-seed.edn"
    stopwords_path = args.out_dir / "stopwords.edn"
    audit_path = args.out_dir / "audit-sample.json"
    stats_path = args.out_dir / "run-stats.json"

    write_edn_file(entries_path, entries_doc)
    validate_edn_round_trip(entries_path)
    write_edn_file(stopwords_path, stopwords_doc)
    validate_edn_round_trip(stopwords_path)
    audit_path.write_text(json.dumps(json_ready(sample), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elapsed_seconds = round(time.time() - started, 3)
    stats = {
        "planetmath_root": str(args.planetmath_root),
        "kernel_tsv": str(args.kernel_tsv),
        "schema_path": str(args.schema_path),
        "timestamp": timestamp_iso,
        "processed_articles": processed,
        "succeeded_entries": len(entries),
        "audit_sample_size": len(sample),
        "stopword_count": len(stopwords),
        "skipped": {
            "total": sum(skip_counts.values()),
            "by_reason": dict(sorted(skip_counts.items())),
            "examples": skipped_examples,
        },
        "elapsed_seconds": elapsed_seconds,
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(entries)} entries, {len(stopwords)} stopwords, "
        f"{sum(skip_counts.values())} skips in {elapsed_seconds:.3f}s.",
        flush=True,
    )
    return {
        "entries_path": entries_path,
        "stopwords_path": stopwords_path,
        "audit_path": audit_path,
        "stats_path": stats_path,
        "stats": stats,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.planetmath_root.exists():
        raise SystemExit(f"PlanetMath root not found: {args.planetmath_root}")
    if not args.kernel_tsv.exists():
        raise SystemExit(f"Kernel TSV not found: {args.kernel_tsv}")
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
