#!/usr/bin/env python3
"""R2d concept coverage for IATC proof graphs.

Scopes the structure-first concept substrate to one proof graph.  The checker
extracts concepts from IATC node text, classifies them against the existing SFC
definition evidence, and emits a check-graph-shaped result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import edn_format

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sfc_concept_coverage as sfc  # noqa: E402

DEFAULT_CONCEPT_INDEX = ROOT / "data" / "warp" / "concept-index.json"
DEFAULT_SNIPPETS = ROOT / "data" / "warp" / "def-snippets.json"
DEFAULT_DEFINED = ROOT / "data" / "warp" / "defined-index.json"
DEFAULT_ENCYCLOPEDIA = ROOT / "data" / "concept-encyclopedia-ct.json"
DEFAULT_GRAPH_DIR = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b"
DEFAULT_SPEC = ROOT / "holes" / "excursions" / "r2d-spec.md"
DEFAULT_REPORT = ROOT / "holes" / "excursions" / "r2d-concept-coverage.md"

MAX_NGRAM = 5
KNOWN_DF_THRESHOLD = 25

DROP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "chosen",
    "consider",
    "every",
    "for",
    "if",
    "in",
    "include",
    "is",
    "of",
    "ought",
    "the",
    "to",
    "with",
}

GENERIC_EXACT = {
    "case",
    "condition",
    "existence",
    "problem",
    "situation",
    "version",
    "versions",
}


def kw(name: str) -> edn_format.Keyword:
    return edn_format.Keyword(name)


def keyword_name(value: Any) -> str:
    text = str(value)
    return text[1:] if text.startswith(":") else text


def edn_to_plain(value: Any) -> Any:
    if isinstance(value, edn_format.Keyword):
        return ":" + keyword_name(value)
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {keyword_name(k): edn_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or (
        not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
    ):
        return [edn_to_plain(v) for v in value]
    return value


def plain_to_edn(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            kw(k) if isinstance(k, str) and not k.startswith("_") else k: plain_to_edn(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [plain_to_edn(v) for v in value]
    if isinstance(value, str) and value.startswith(":") and re.fullmatch(r":[\w./?=-]+", value):
        return kw(value[1:])
    return value


def load_edn(path: Path) -> dict[str, Any]:
    text = re.sub(r":([A-Za-z0-9_./?=-]+)'", r":\1-prime", path.read_text())
    return edn_to_plain(edn_format.loads(text))


def clean_text(text: str) -> str:
    text = re.sub(r"\\[A-Za-z]+(?:\{([^{}]*)\})?", r" \1 ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    return " ".join(text.lower().split())


def normalize_phrase(text: str) -> str:
    phrase = sfc.normalize_concept(clean_text(text))
    words = phrase.split()
    if words and words[-1] == "bicategories":
        phrase = " ".join([*words[:-1], "bicategory"])
    return phrase


def tokenized(text: str) -> list[str]:
    return [t for t in clean_text(text).split() if t and t not in DROP_TOKENS]


def worth_unresolved_phrase(phrase: str) -> bool:
    words = phrase.split()
    if not words or len(words) > 5:
        return False
    if phrase in GENERIC_EXACT:
        return False
    if words[0] in {"consider", "last"} or words[-1] in {"used"}:
        return False
    if len(words) == 1 and (len(words[0]) < 4 or words[0] in GENERIC_EXACT):
        return False
    if any(len(w) == 1 for w in words):
        return False
    return True


def ngram_candidates(text: str, known_concepts: set[str]) -> set[str]:
    words = tokenized(text)
    out: set[str] = set()
    for n in range(1, min(MAX_NGRAM, len(words)) + 1):
        for i in range(0, len(words) - n + 1):
            phrase = sfc.normalize_concept(" ".join(words[i : i + n]))
            phrase = normalize_phrase(phrase)
            if phrase in known_concepts and phrase not in GENERIC_EXACT:
                out.add(phrase)
    return out


def pattern_candidates(text: str) -> set[str]:
    cleaned = clean_text(text)
    out: set[str] = set()

    if match := re.search(r"\bevery (.+?) is ([a-z]+)\b", cleaned):
        subject = normalize_phrase(match.group(1))
        adjective = normalize_phrase(match.group(2))
        if worth_unresolved_phrase(subject):
            out.add(subject)
        subject_words = subject.split()
        if subject_words and adjective:
            out.add(sfc.normalize_concept(f"{adjective} {subject_words[-1]}"))

    if match := re.search(r"\b([a-z]+) versions? of the rules governing (.+)$", cleaned):
        out.add(sfc.normalize_concept(f"{match.group(1)} rules"))
        governed = normalize_phrase(match.group(2))
        if worth_unresolved_phrase(governed):
            out.add(governed)

    if match := re.search(r"\b(standard) rules governing (.+)$", cleaned):
        out.add(sfc.normalize_concept(f"{match.group(1)} rules"))
        governed = normalize_phrase(match.group(2))
        if worth_unresolved_phrase(governed):
            out.add(governed)

    if match := re.search(r"\b(.+?) if and only if (.+)$", cleaned):
        left = normalize_phrase(match.group(1))
        right = normalize_phrase(match.group(2))
        if worth_unresolved_phrase(left):
            out.add(left)
        if worth_unresolved_phrase(right):
            out.add(right)

    return out


def extract_node_concepts(graph: dict[str, Any], concept_index: dict[str, Any]) -> dict[str, set[str]]:
    known_concepts = set(concept_index)
    by_concept: dict[str, set[str]] = defaultdict(set)
    for node in graph.get("nodes") or []:
        if node.get("kind") == ":ref":
            continue
        text = str(node.get("text") or "").strip()
        if not text:
            continue
        candidates = set()
        candidates.update(pattern_candidates(text))
        candidates.update(ngram_candidates(text, known_concepts))

        exact = normalize_phrase(text)
        if exact in known_concepts or (not candidates and worth_unresolved_phrase(exact)):
            candidates.add(exact)
        elif worth_unresolved_phrase(exact) and any("-like" in text.lower() for _ in [0]):
            candidates.add(exact)

        for concept in candidates:
            if concept and not sfc.boilerplate_phrase(concept):
                by_concept[concept].add(text)
    return by_concept


def encyclopedia_known(encyclopedia: dict[str, Any]) -> dict[str, str]:
    known: dict[str, str] = {}
    for entry in encyclopedia.get("entries") or []:
        if not isinstance(entry, dict) or not entry.get("concept"):
            continue
        concept = sfc.normalize_concept(str(entry["concept"]))
        target = str((entry.get("provenance") or {}).get("target") or "")
        if target.startswith("nlab-") or target.startswith("nnexus:"):
            known[concept] = target
    return known


@dataclass(frozen=True)
class Substrate:
    concept_index: dict[str, Any]
    definition_sources: dict[str, set[str]]
    known_provenance: dict[str, str]
    known_df_threshold: int


def load_substrate(args: argparse.Namespace) -> Substrate:
    snippets = sfc.load_json(args.def_snippets)
    defined = sfc.load_json(args.defined_index)
    encyclopedia = sfc.load_json(args.concept_encyclopedia)
    return Substrate(
        concept_index=sfc.load_json(args.concept_index),
        definition_sources=sfc.definition_sets(snippets, defined, encyclopedia),
        known_provenance=encyclopedia_known(encyclopedia),
        known_df_threshold=args.known_df_threshold,
    )


def restrict_substrate(substrate: Substrate, paper_ids) -> Substrate:
    """Scope the substrate to a RUN-CORPUS (a set of papers): a concept is only present if
    it occurs in a scope paper, and its df is recomputed over the scope. This makes
    comprehension corpus-relative to the RUN — so it RISES as the run grows (the accretion
    sweep), instead of grounding against the full archive (the mark6 floor, finding #1)."""
    scope = set(paper_ids)
    ci = {}
    for concept, row in substrate.concept_index.items():
        inscope = [p for p in (row.get("papers") or []) if p in scope]
        if inscope:
            ci[concept] = {**row, "df": len(inscope), "papers": inscope}
    # a concept is "defined/known" in the run-corpus only if it also OCCURS in it: filter
    # the (global) definition evidence + provenance to concepts present in the scope.
    defs = {c: s for c, s in substrate.definition_sources.items() if c in ci}
    prov = {c: p for c, p in substrate.known_provenance.items() if c in ci}
    return Substrate(
        concept_index=ci,
        definition_sources=defs,
        known_provenance=prov,
        known_df_threshold=substrate.known_df_threshold,
    )


def classify_concept(concept: str, substrate: Substrate) -> dict[str, Any]:
    index_row = substrate.concept_index.get(concept) or {}
    sources = sorted(substrate.definition_sources.get(concept, set()) | set(index_row.get("sources") or []))
    df = int(index_row.get("df") or 0)
    genuine = bool(index_row.get("genuine", False))
    provenance = substrate.known_provenance.get(concept)

    if sources or bool(index_row.get("defined", False)):
        bucket = "defined"
        reason = "definition evidence in " + ", ".join(sources or ["concept-index"])
    elif provenance:
        bucket = "known"
        reason = f"canonical provenance pointer {provenance}"
    elif genuine and df >= substrate.known_df_threshold:
        bucket = "known"
        reason = f"recurring core concept: df={df} >= {substrate.known_df_threshold}"
    else:
        bucket = "undefined"
        reason = "no definition evidence, provenance pointer, or recurring-core support"

    return {
        "concept": concept,
        "bucket": bucket,
        "sources": sources,
        "df": df,
        "genuine": genuine,
        "known_provenance": provenance,
        "reason": reason,
    }


def graph_paper_id(graph: dict[str, Any], path: Path) -> str:
    return str(graph.get("paper/id") or str(graph.get("passage/id") or "").split(":")[0] or path.stem)


def check_graph(graph: dict[str, Any], path: Path, substrate: Substrate) -> dict[str, Any]:
    extracted = extract_node_concepts(graph, substrate.concept_index)
    rows = []
    for concept in sorted(extracted):
        row = classify_concept(concept, substrate)
        row["source_texts"] = sorted(extracted[concept])
        rows.append(row)

    total = len(rows)
    counts = Counter(row["bucket"] for row in rows)
    covered = counts["defined"] + counts["known"] + counts["imported"]
    undefined = [row["concept"] for row in rows if row["bucket"] == "undefined"]

    if total == 0:
        return {
            "check": ":concept-coverage",
            "status": ":na",
            "pass": True,
            "rate": None,
            "reasons": ["N/A: no extractable IATC node-text concepts"],
            "per-item": [],
            "paper-id": graph_paper_id(graph, path),
            "buckets": {"defined": 0, "known": 0, "imported": 0, "undefined": 0},
            "undefined": [],
            "imported": [],
            "concept-source": "iatc-node-text",
        }

    reasons = [
        f"{covered}/{total} concepts covered; undefined concepts are report-only gaps",
        "imported bucket wired but N/A until R2d-3 descent artifact lands",
    ]
    if undefined:
        reasons.append("undefined: " + ", ".join(undefined[:10]))

    return {
        "check": ":concept-coverage",
        "status": ":pass",
        "pass": True,
        "rate": covered / total,
        "reasons": reasons,
        "per-item": rows,
        "paper-id": graph_paper_id(graph, path),
        "buckets": {
            "defined": counts["defined"],
            "known": counts["known"],
            "imported": counts["imported"],
            "undefined": counts["undefined"],
        },
        "undefined": undefined,
        "imported": [],
        "concept-source": "iatc-node-text",
    }


def graph_files(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(p for p in path.glob("*.edn") if p.is_file()))
        elif path.suffix == ".edn":
            out.append(path)
    return out


def spread(rates: list[float]) -> dict[str, float | None]:
    if not rates:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(rates), "max": max(rates), "mean": sum(rates) / len(rates)}


def run_paths(args: argparse.Namespace) -> list[dict[str, Any]]:
    substrate = load_substrate(args)
    results = []
    for path in graph_files(args.graphs):
        graph = load_edn(path)
        result = check_graph(graph, path, substrate)
        result["file"] = str(path)
        results.append(result)
    return results


def render_spec(results: list[dict[str, Any]]) -> str:
    by_id = {r["paper-id"]: r for r in results}
    worked = [pid for pid in ["0706.1286", "0709.0248", "0708.2067"] if pid in by_id]
    lines = [
        "# R2d Spec Spike",
        "",
        "Generated by `scripts/r2d_concept_coverage.py --write-spec`.",
        "",
        "## Concept Source",
        "",
        "Use IATC graph node `:text` as the deterministic R2d-1/R2d-2 source.",
        "The proof-region `fable-<id>-dp-emacs.json` marks carry useful canon/nLab",
        "pointers, but in the current artifacts they are whole-paper character-offset",
        "marks rather than line-scoped proof-region concepts. Joining them now would",
        "pollute a proof check with concepts outside the proof span. They should be",
        "reconsidered once a line-aligned proof-region mark artifact exists.",
        "",
        "## Known Threshold",
        "",
        f"- `defined`: any SFC definition source reused from `sfc_concept_coverage.definition_sets`.",
        "- `known`: nLab/NNexus provenance in the encyclopedia, or a genuine",
        f"  concept-index recurring core with `df >= {KNOWN_DF_THRESHOLD}`.",
        "- `imported`: slot wired but always empty/N/A until R2d-3 gets the",
        "  WARP-ORCH-3 descent/phylogeny artifact.",
        "- `undefined`: no definition source, no canonical provenance pointer,",
        "  and not above the recurring-core threshold.",
        "",
        "## Worked Proofs",
        "",
    ]
    for pid in worked:
        row = by_id[pid]
        lines.extend(
            [
                f"### `{pid}`",
                "",
                f"- coverage: `{row['rate']:.3f}`",
                f"- buckets: `{row['buckets']}`",
                "- undefined: "
                + (", ".join(f"`{x}`" for x in row["undefined"]) if row["undefined"] else "`none`"),
                "",
                "| concept | bucket | reason |",
                "| --- | --- | --- |",
            ]
        )
        for item in row["per-item"]:
            lines.append(f"| `{item['concept']}` | `{item['bucket']}` | {item['reason']} |")
        lines.append("")
    return "\n".join(lines)


def render_report(results: list[dict[str, Any]]) -> str:
    rates = [float(r["rate"]) for r in results if r["rate"] is not None]
    s = spread(rates)
    lines = [
        "# R2d Concept Coverage — loop-run-70b",
        "",
        "Generated by `scripts/r2d_concept_coverage.py --report`.",
        "",
        f"- proofs: `{len(results)}`",
        f"- coverage spread: min `{s['min']:.3f}`, max `{s['max']:.3f}`, mean `{s['mean']:.3f}`"
        if s["min"] is not None
        else "- coverage spread: `n/a`",
        "- gate semantics: undefined concepts are report-only flagged gaps, not hard failures",
        "- imported bucket: wired empty/N/A pending R2d-3 descent artifact",
        "",
        "| paper | coverage | defined | known | imported | undefined | undefined concepts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        buckets = row["buckets"]
        undefined = ", ".join(f"`{x}`" for x in row["undefined"]) or "-"
        lines.append(
            f"| `{row['paper-id']}` | {row['rate']:.3f} | {buckets['defined']} | "
            f"{buckets['known']} | {buckets['imported']} | {buckets['undefined']} | {undefined} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", type=Path, nargs="*", default=[DEFAULT_GRAPH_DIR])
    parser.add_argument("--concept-index", type=Path, default=DEFAULT_CONCEPT_INDEX)
    parser.add_argument("--def-snippets", type=Path, default=DEFAULT_SNIPPETS)
    parser.add_argument("--defined-index", type=Path, default=DEFAULT_DEFINED)
    parser.add_argument("--concept-encyclopedia", type=Path, default=DEFAULT_ENCYCLOPEDIA)
    parser.add_argument("--known-df-threshold", type=int, default=KNOWN_DF_THRESHOLD)
    parser.add_argument("--edn", action="store_true", help="print single/batch results as EDN")
    parser.add_argument("--json", action="store_true", help="print single/batch results as JSON")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write-spec", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_paths(args)
    if args.write_spec:
        args.write_spec.parent.mkdir(parents=True, exist_ok=True)
        args.write_spec.write_text(render_spec(results))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(results))
    if args.edn:
        payload: Any = results[0] if len(results) == 1 else results
        print(edn_format.dumps(plain_to_edn(payload)))
    elif args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.report and not args.write_spec:
        print(render_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
