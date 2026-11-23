"""Mechanical measures for proof candidates in an extracted Mark7 run.

``words`` deliberately uses ``len(text.split())``, the exact rule used by
InkUp's telling-sidecar validator.  It therefore counts raw TeX whitespace
tokens: inline or displayed mathematics is *not* collapsed to one token.

Sentence boundaries are approximate: a boundary is a full stop, question
mark, or exclamation mark followed by whitespace and an uppercase character,
or a blank line.  TeX display environments do not themselves end a sentence.

``opens_with`` is purely lexical.  After TeX proof wrappers are removed, the
patterns are tried in this order: let/suppose/assume; by a citation, reference,
or named result; we show/prove/claim or "it suffices"; clearly/obviously/
immediately/"it follows from"; other.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

CITE_RE = re.compile(r"\\(?:[A-Za-z]*cite[A-Za-z]*)(?:\[[^]]*\])*\{[^}]*\}")
REF_RE = re.compile(r"\\(?:eqref|[A-Za-z]*ref[A-Za-z]*)(?:\[[^]]*\])*\{[^}]*\}")
DISPLAY_RE = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|"
    r"\\begin\{(?:displaymath|equation\*?|align\*?|alignat\*?|gather\*?|"
    r"multline\*?|flalign\*?|tikzcd|diagram)\}", re.S)
DIAGRAM_RE = re.compile(r"\\(?:xymatrix|begin\{(?:tikzcd|diagram)\})")
LIST_RE = re.compile(r"\\begin\{(?:enumerate|itemize)\}")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n+")
PROOF_WRAPPER_RE = re.compile(
    r"^\s*(?:\\begin\{(?:proof|pf)\}(?:\[[^]]*\])?|\\(?:proof|bpf)\b|\\noindent\s*\{?\\bf\s+Proof\.?\}?|Proof\.)\s*",
    re.I,
)


PROOF_END_RE = re.compile(r"\s*\\(?:end\{(?:proof|pf)\}|epf\b|qed\b)\s*$")


def strip_wrapper(text: str) -> str:
    """Remove the proof environment's opening and closing markers.

    The tellings carry no such markers, so counting them would add two words
    to every paper proof.
    """
    return PROOF_END_RE.sub("", PROOF_WRAPPER_RE.sub("", text, count=1))


def words(text: str) -> int:
    """Return the telling-sidecar validator's whitespace-token count."""
    return len(text.split())


def sentences(text: str) -> list[str]:
    """Split text using the documented conservative sentence approximation."""
    stripped = text.strip()
    return [part.strip() for part in SENTENCE_BOUNDARY_RE.split(stripped) if part.strip()]


def opens_with(text: str) -> str:
    first = PROOF_WRAPPER_RE.sub("", text, count=1).lstrip()
    first = re.sub(r"^(?:\\(?:noindent|smallskip|medskip|par)\s*)+", "", first)
    # A leading part label such as "(i):" or "\\begin{enumerate} \\item" is skipped.
    first = re.sub(r"^(?:\\begin\{(?:enumerate|itemize)\}(?:\[[^]]*\])?\s*\\item(?:\[[^]]*\])?\s*|\(?[ivx0-9a-z]{1,4}\)\s*:?\s*)", "", first, flags=re.I)
    if re.match(r"^(?:Let|Suppose|Assume|Consider|Fix|Choose|Take|Given|Soit|Soient|Supposons)\b", first, re.I):
        return "let/suppose/assume"
    if re.match(
        r"^By\s+(?:\\(?:[A-Za-z]*cite[A-Za-z]*|(?:eq|[A-Za-z]*)ref[A-Za-z]*)\b|"
        r"(?:the\s+)?[A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*)*\s+(?:lemma|theorem|proposition))",
        first,
        re.I,
    ):
        return "by-cite/ref/named-result"
    if re.match(r"^By\b", first, re.I):
        return "by-other"
    if re.match(
        r"^(?:We\s+(?:first|will|shall|begin|start|need|only|have\s+to|must|proceed|argue|show|prove|claim)|"
        r"First,?\s+we|To\s+(?:prove|show|begin|see)|It\s+(?:suffices|is\s+enough|remains)|"
        r"Without\s+loss)\b",
        first,
        re.I,
    ):
        return "we-show/prove/claim/it-suffices"
    if re.match(
        r"^(?:Clearly|Obviously|Immediate(?:ly)?|(?:It|This)\s+follows|Follows\s+from|"
        r"This\s+is\s+(?:a|an|the|just|clear|immediate|obvious|standard|well)|"
        r"(?:The\s+)?(?:proof|argument)\s+is|As\s+in|Same\s+as|See)\b",
        first,
        re.I,
    ):
        return "clearly/obvious/immediate/follows-from"
    return "other"


def slice_candidate(candidate: dict) -> tuple[str, bool]:
    """Return the verbatim proof-line slice and whether it was unsliceable."""
    try:
        window_start, window_end = candidate["window-lines"]
        proof_start, proof_end = candidate["proof-lines"]
        lines = candidate["source-window"].splitlines(keepends=True)
        if not (window_start <= proof_start <= proof_end <= window_end):
            raise ValueError
        start = proof_start - window_start
        end = proof_end - window_start + 1
        if start < 0 or end > len(lines):
            raise ValueError
        return "".join(lines[start:end]), False
    except (KeyError, TypeError, ValueError):
        return "", True


def parse_faithfulness(path: Path) -> dict[str, dict[str, bool]]:
    results: dict[str, dict[str, bool]] = {}
    row = re.compile(r"^(PASS|FAIL)\s+\S*/([^/]+)\.edn\s+rate=(\S+)")
    for line in path.read_text().splitlines():
        match = row.match(line)
        if match:
            status, proof_id, rate = match.groups()
            results[proof_id] = {
                "faithfulness_fail": status == "FAIL",
                "input_error": rate == "unscored",
            }
    return results


def measure_text(text: str) -> dict:
    text = strip_wrapper(text)
    parts = sentences(text)
    first = parts[0] if parts else ""
    cite_count = len(CITE_RE.findall(text))
    ref_count = len(REF_RE.findall(text))
    return {
        "words": words(text),
        "words_whitespace": len(text.split()),
        "sentences": len(parts),
        "cite_count": cite_count,
        "ref_count": ref_count,
        "first_sentence_has_cite_or_ref": bool(CITE_RE.search(first) or REF_RE.search(first)),
        "display_count": len(DISPLAY_RE.findall(text)),
        "has_diagram": bool(DIAGRAM_RE.search(text)),
        "has_list": bool(LIST_RE.search(text)),
        "opens_with": opens_with(text),
    }


def proof_records(run: Path) -> list[dict]:
    candidates = run / "artifacts/candidates"
    faithfulness = parse_faithfulness(run / "anchor-faithfulness.txt")
    clean_dir = run / "artifacts/clean"
    graph_dir = run / "artifacts/graphs"
    records = []
    for path in sorted(candidates.glob("*.candidate.json")):
        candidate = json.loads(path.read_text())
        proof_id = candidate.get("proof-id", path.name.removesuffix(".candidate.json"))
        text, unsliceable = slice_candidate(candidate)
        faith = faithfulness.get(proof_id, {})
        no_clean = not (clean_dir / f"{proof_id}.clean.edn").exists() or not any(
            (graph_dir / f"{proof_id}{suffix}").exists()
            for suffix in (".edn", ".accepted.edn")
        )
        flags = {
            "unsliceable": unsliceable,
            "faithfulness_fail": faith.get("faithfulness_fail", False),
            "input_error": faith.get("input_error", False),
            "no_clean_proof": no_clean,
        }
        record = {
            "paper_id": candidate.get("paper-id"),
            "proof_id": proof_id,
            "passage_id": candidate.get("passage-id"),
            "statement_kind": (candidate.get("proved") or {}).get("kind"),
            "proof_lines": candidate.get("proof-lines"),
            "text": text,
            **measure_text(text),
            "flags": flags,
            "usable": not unsliceable,
        }
        records.append(record)
    return records


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    if not ordered:
        return {str(q): None for q in (10, 25, 50, 75, 90)}
    answer = {}
    for q in (10, 25, 50, 75, 90):
        position = (len(ordered) - 1) * q / 100
        lower, upper = math.floor(position), math.ceil(position)
        value = ordered[lower] if lower == upper else (
            ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        )
        answer[str(q)] = round(value, 3)
    return answer


def distribution(records: list[dict]) -> dict:
    records = [row for row in records if row["usable"] and row["words"] > 0]
    lengths = [row["words"] for row in records]
    densities = [100 * (row["cite_count"] + row["ref_count"]) / row["words"] for row in records]
    return {
        "proofs": len(records),
        "word_quantiles": quantiles(lengths),
        "share_words_le_120": round(sum(n <= 120 for n in lengths) / len(lengths), 6) if lengths else None,
        "share_words_le_168": round(sum(n <= 168 for n in lengths) / len(lengths), 6) if lengths else None,
        "cite_ref_per_100_words_quantiles": quantiles(densities),
        "share_first_sentence_cite_or_ref": round(
            sum(row["first_sentence_has_cite_or_ref"] for row in records) / len(records), 6
        ) if records else None,
        "opens_with": dict(sorted(Counter(row["opens_with"] for row in records).items())),
    }


def summarize(records: list[dict], paper_ids: Iterable[str] = ()) -> tuple[list[dict], dict]:
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for paper_id in paper_ids:
        by_paper[paper_id] = []
    for row in records:
        by_paper[row["paper_id"]].append(row)
    papers = []
    for paper_id, rows in sorted(by_paper.items()):
        usable = [row for row in rows if row["usable"]]
        lengths = [row["words"] for row in usable]
        papers.append({
            "paper_id": paper_id,
            "proofs": len(rows),
            "usable_proofs": len(usable),
            "median_words": median(lengths) if lengths else None,
            "total_words": sum(lengths),
        })
    flag_counts = {name: sum(row["flags"][name] for row in records)
                   for name in ("unsliceable", "faithfulness_fail", "input_error", "no_clean_proof")}
    usable = [row for row in records if row["usable"]]
    midrange = [row for row in usable if 40 <= row["words"] <= 400]
    medians = [row["median_words"] for row in papers if row["median_words"] is not None]
    summary = {
        "candidate_files": len(records),
        "flags": flag_counts,
        "all_usable": distribution(usable),
        "words_40_to_400": distribution(midrange),
        "per_paper": {
            "papers": len(papers),
            "papers_with_usable_proofs": len(medians),
            "median_word_quantiles": quantiles(medians),
        },
        "word_rule": "len(text.split()); identical to validate_sidecar.py",
    }
    return papers, summary


def write_run(run: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records = proof_records(run)
    manifest = json.loads((run / "run-manifest.json").read_text())
    papers, summary = summarize(records, manifest.get("papers", []))
    candidate_files = [path.name for path in (run / "artifacts/candidates").iterdir() if path.is_file()]
    summary["candidate_directory_files"] = len(candidate_files)
    summary["non_candidate_files"] = sorted(
        name for name in candidate_files if not name.endswith(".candidate.json")
    )
    with (output / "proofs.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "papers.jsonl").open("w") as stream:
        for row in papers:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

