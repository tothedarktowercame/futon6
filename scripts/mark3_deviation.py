#!/usr/bin/env python3
"""mark3 H9: concept deviation detector.

Flags per-(paper, concept) records where local usage diverges from the per-MSC
concept encyclopedia and embedding norm. The detector is deliberately
conservative: structural contradictions get explicit cue evidence; embedding
anomalies are reported as scores rather than treated as proof.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import edn_format
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MARK3_EMBED_SPEC = importlib.util.spec_from_file_location("mark3_embed", SCRIPTS / "mark3_embed.py")
mark3_embed = importlib.util.module_from_spec(MARK3_EMBED_SPEC)
assert MARK3_EMBED_SPEC.loader is not None
sys.modules[MARK3_EMBED_SPEC.name] = mark3_embed
MARK3_EMBED_SPEC.loader.exec_module(mark3_embed)

TOKEN_RE = re.compile(r"[a-z0-9]+")
WORDISH = re.compile(r"[A-Za-z][A-Za-z0-9 -]{2,}")
CONTRADICTION_CUES = [
    "not a",
    "not an",
    "need not",
    "without",
    "no longer",
    "rather than",
    "instead of",
    "unlike",
    "different from",
    "we redefine",
    "we call",
    "against the usual",
]
ROLE_CUES = [
    "left adjoint",
    "right adjoint",
    "endofunctor",
    "natural transformation",
    "unit",
    "multiplication",
    "tensor product",
    "object",
    "morphism",
]


@dataclass(frozen=True)
class ConceptEntry:
    concept_id: str
    name: str
    kind: str
    depends_on: tuple[str, ...]
    axiom_text: str
    raw: dict


@dataclass(frozen=True)
class Usage:
    concept: str
    paper: str
    start: int
    end: int
    context: str
    anchor: dict


def keyword_name(x) -> str:
    if isinstance(x, edn_format.Keyword):
        return str(x)[1:]
    if isinstance(x, edn_format.Symbol):
        return str(x)
    return str(x)


def normalize_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def slug_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def read_edn(path: Path) -> dict:
    return edn_format.loads(path.read_text())


def load_encyclopedia(paths: Sequence[Path]) -> dict[str, ConceptEntry]:
    entries: dict[str, ConceptEntry] = {}
    for root in paths:
        for path in sorted(root.glob("*.edn")):
            raw = read_edn(path)
            name = raw.get(edn_format.Keyword("name")) or raw.get("name") or path.stem.replace("-", " ")
            concept_id = raw.get(edn_format.Keyword("concept/id")) or raw.get("concept/id") or slug_label(str(name))
            kind = raw.get(edn_format.Keyword("kind")) or raw.get("kind") or "concept"
            deps_raw = raw.get(edn_format.Keyword("depends-on")) or raw.get("depends-on") or []
            axioms = raw.get(edn_format.Keyword("axioms")) or raw.get("axioms") or []
            axiom_text = " ".join(str(a.get(edn_format.Keyword("statement")) or a.get("statement") or a) for a in axioms)
            entry = ConceptEntry(
                concept_id=keyword_name(concept_id),
                name=str(name),
                kind=keyword_name(kind),
                depends_on=tuple(keyword_name(x) for x in deps_raw),
                axiom_text=axiom_text,
                raw=raw,
            )
            entries[normalize_label(entry.name)] = entry
            entries[normalize_label(entry.concept_id.replace("-", " "))] = entry
    return entries


def load_paper(path: Path) -> dict:
    data = json.loads(path.read_text())
    data.setdefault("paper", path.stem.replace("fable-", "").replace("-dp-emacs", ""))
    return data


def line_col(text: str, offset: int) -> dict:
    line = text.count("\n", 0, offset) + 1
    last_nl = text.rfind("\n", 0, offset)
    col = offset + 1 if last_nl < 0 else offset - last_nl
    return {"line": line, "col": col}


def mark_concept_from_tip(tip: str) -> str | None:
    match = re.search(r"concept:\s*([^\[]+)", tip)
    if match:
        return normalize_label(match.group(1))
    return None


def context_window(text: str, start: int, end: int, width: int) -> str:
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def find_usages(paper: dict, entries: dict[str, ConceptEntry], context_chars: int) -> list[Usage]:
    text = paper.get("text", "")
    paper_id = str(paper.get("paper"))
    usages: list[Usage] = []
    seen: set[tuple[str, int, int]] = set()
    for mark in paper.get("marks", []):
        if mark.get("kind") not in {"concept", "concept-typed", "definiens", "let-binder"}:
            continue
        labels = []
        tip_label = mark_concept_from_tip(mark.get("tip", ""))
        if tip_label:
            labels.append(tip_label)
        span_text = normalize_label(text[mark.get("start", 0) : mark.get("end", 0)])
        if span_text:
            labels.append(span_text)
        for label in labels:
            entry = entries.get(label)
            if not entry:
                continue
            start = int(mark.get("start", 0))
            end = int(mark.get("end", start))
            key = (entry.concept_id, start, end)
            if key in seen:
                continue
            seen.add(key)
            usages.append(
                Usage(
                    concept=entry.name,
                    paper=paper_id,
                    start=start,
                    end=end,
                    context=context_window(text, start, end, context_chars),
                    anchor={"start": start, "end": end, **line_col(text, start), "mark-kind": mark.get("kind")},
                )
            )
    # Text fallback catches concepts that layer-(a) did not mark as concept spans.
    for label, entry in entries.items():
        if len(label) < 5 or " " not in label:
            continue
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(label).replace(r"\ ", r"\s+") + r"(?![a-z0-9])", re.I)
        for match in pattern.finditer(text):
            key = (entry.concept_id, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            usages.append(
                Usage(
                    concept=entry.name,
                    paper=paper_id,
                    start=match.start(),
                    end=match.end(),
                    context=context_window(text, match.start(), match.end(), context_chars),
                    anchor={"start": match.start(), "end": match.end(), **line_col(text, match.start()), "mark-kind": "text"},
                )
            )
    return usages


def dependency_support(entry: ConceptEntry, paper_text: str, context: str) -> tuple[float, list[str]]:
    if not entry.depends_on:
        return 1.0, []
    context_norm = normalize_label(context)
    paper_norm = normalize_label(paper_text)
    missing = []
    for dep in entry.depends_on:
        dep_text = normalize_label(dep.replace("-", " "))
        if dep_text not in context_norm and dep_text not in paper_norm:
            missing.append(dep)
    support = 1.0 - (len(missing) / len(entry.depends_on))
    return support, missing


def structural_score(entry: ConceptEntry, paper_text: str, context: str, synthetic: bool = False) -> tuple[float, list[str]]:
    ctx = context.lower()
    cue_hits = [cue for cue in CONTRADICTION_CUES if cue in ctx]
    support, missing = dependency_support(entry, paper_text, context)
    entry_vocab = tokens(entry.axiom_text + " " + " ".join(entry.depends_on) + " " + entry.kind)
    role_hits = [cue for cue in ROLE_CUES if cue in ctx and not (tokens(cue) <= entry_vocab)]
    score = 0.0
    if cue_hits:
        score += 0.45
    if role_hits:
        score += min(0.25, 0.08 * len(role_hits))
    if missing and support < 0.35:
        score += 0.20
    if synthetic:
        score += 0.25
    evidence = []
    evidence.extend(f"cue:{c}" for c in cue_hits)
    evidence.extend(f"unsupported-role:{c}" for c in role_hits)
    if missing:
        evidence.append("missing-depends-on:" + ",".join(missing[:8]))
    return min(1.0, score), evidence


def load_embedding_norms(embed_dir: Path, space: str) -> tuple[list[dict], np.ndarray, dict]:
    items = json.loads((embed_dir / f"{space}-items.json").read_text())
    embeddings = np.load(embed_dir / f"{space}-embeddings.npy")
    report_path = embed_dir / "infer-report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return items, embeddings, report


def concept_embedding_index(items: Sequence[dict]) -> dict[str, int]:
    out = {}
    for idx, item in enumerate(items):
        if item.get("item_type") not in {"concept", "term"}:
            continue
        out[normalize_label(item.get("label", ""))] = idx
        out[normalize_label(item.get("item_id", "").replace("concept:", "").replace("term:", "").replace("-", " "))] = idx
    return out


def encode_contexts(contexts: Sequence[str], embed_dir: Path, space: str, backend: str) -> np.ndarray:
    items, embeddings, report = load_embedding_norms(embed_dir, space)
    dim = int(embeddings.shape[1])
    selected = backend
    if selected == "auto":
        selected = "hash" if report.get("backend") == "hash" else "bge"
    if selected == "hash":
        return mark3_embed.stable_hash_embeddings(contexts, dim)
    model = report.get("model") or "BAAI/bge-small-en-v1.5"
    args = argparse.Namespace(
        backend="bge",
        model=model,
        model_out=Path("__missing_model_dir__"),
        device=None,
        embed_workers=1,
        num_gpus=1,
        batch_size=32,
        hash_dim=dim,
    )
    pseudo_items = [mark3_embed.EmbedItem(f"context:{idx}", "context", "ct", f"context {idx}", text) for idx, text in enumerate(contexts)]
    return mark3_embed.encode_items(pseudo_items, args)


def embedding_scores(usages: Sequence[Usage], embed_dir: Path, space: str, backend: str) -> dict[tuple[str, int], tuple[float, float]]:
    if not usages:
        return {}
    items, embeddings, _report = load_embedding_norms(embed_dir, space)
    idx = concept_embedding_index(items)
    contexts = [u.context for u in usages]
    ctx_emb = encode_contexts(contexts, embed_dir, space, backend)
    emb_norm = mark3_embed.normalize_rows(embeddings.astype(np.float32))
    ctx_norm = mark3_embed.normalize_rows(ctx_emb.astype(np.float32))
    scores = {}
    for pos, usage in enumerate(usages):
        cidx = idx.get(normalize_label(usage.concept))
        if cidx is None:
            continue
        sim = float(ctx_norm[pos] @ emb_norm[cidx])
        anomaly = max(0.0, min(1.0, (0.72 - sim) / 0.35))
        scores[(usage.concept, usage.start)] = (anomaly, sim)
    return scores


def detect(
    paper_path: Path,
    encyclopedia_dirs: Sequence[Path],
    embed_dir: Path,
    output: Path,
    space: str,
    context_backend: str,
    threshold: float,
    context_chars: int,
    synthetic: bool = False,
) -> list[dict]:
    entries = load_encyclopedia(encyclopedia_dirs)
    paper = load_paper(paper_path)
    usages = find_usages(paper, entries, context_chars)
    by_name = {}
    for entry in entries.values():
        by_name.setdefault(entry.name, entry)
    emb = embedding_scores(usages, embed_dir, space, context_backend)
    records = []
    for usage in usages:
        entry = by_name.get(usage.concept)
        if not entry:
            continue
        s_score, s_evidence = structural_score(entry, paper.get("text", ""), usage.context, synthetic=synthetic)
        e_score, sim = emb.get((usage.concept, usage.start), (0.0, None))
        if s_score >= threshold:
            records.append(
                {
                    "concept": usage.concept,
                    "paper": usage.paper,
                    "deviation-kind": "structural",
                    "score": round(s_score, 6),
                    "evidence-anchor": usage.anchor,
                    "evidence": s_evidence,
                    "context": usage.context[:500],
                }
            )
        if e_score >= threshold:
            records.append(
                {
                    "concept": usage.concept,
                    "paper": usage.paper,
                    "deviation-kind": "embedding/notation",
                    "score": round(e_score, 6),
                    "similarity-to-norm": None if sim is None else round(sim, 6),
                    "evidence-anchor": usage.anchor,
                    "context": usage.context[:500],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def write_synthetic_paper(path: Path, clean: bool) -> None:
    if clean:
        text = (
            "Let C be a category. A monad on C is an endofunctor T : C to C "
            "with multiplication mu and unit eta satisfying associativity and unit laws."
        )
    else:
        text = (
            "In this paper, a monad is not an endofunctor and has no natural transformations. "
            "We redefine monad as a left adjoint object without unit or multiplication."
        )
    start = text.lower().index("monad")
    path.write_text(
        json.dumps(
            {
                "paper": "synthetic-clean" if clean else "synthetic-redefinition",
                "text": text,
                "marks": [
                    {
                        "start": start,
                        "end": start + len("monad"),
                        "layer": "dp",
                        "kind": "concept",
                        "tip": "concept: monad [synthetic]",
                    }
                ],
            }
        )
    )


def validate(args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="mark3-h9-") as tmp_s:
        tmp = Path(tmp_s)
        clean_path = tmp / "clean.json"
        bad_path = tmp / "bad.json"
        write_synthetic_paper(clean_path, clean=True)
        write_synthetic_paper(bad_path, clean=False)
        clean_out = tmp / "clean.jsonl"
        bad_out = tmp / "bad.jsonl"
        clean_records = detect(
            clean_path,
            args.encyclopedia_dir,
            args.embed_dir,
            clean_out,
            args.space,
            args.context_backend,
            args.threshold,
            args.context_chars,
            synthetic=False,
        )
        bad_records = detect(
            bad_path,
            args.encyclopedia_dir,
            args.embed_dir,
            bad_out,
            args.space,
            args.context_backend,
            args.threshold,
            args.context_chars,
            synthetic=True,
        )
    sample_records = detect(
        args.paper,
        args.encyclopedia_dir,
        args.embed_dir,
        args.output,
        args.space,
        args.context_backend,
        args.threshold,
        args.context_chars,
    )
    clean_max = max((r["score"] for r in clean_records), default=0.0)
    bad_max = max((r["score"] for r in bad_records), default=0.0)
    synthetic_flagged = bad_max >= args.threshold and bad_max > clean_max
    predicted_positive = bool(clean_records) + bool(bad_records)
    true_positive = 1 if bad_records else 0
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    scores = [r["score"] for r in sample_records]
    report = {
        "synthetic_positive_flagged": synthetic_flagged,
        "synthetic_bad_max_score": bad_max,
        "synthetic_clean_max_score": clean_max,
        "synthetic_precision": precision,
        "sample_records": len(sample_records),
        "sample_score_min": min(scores) if scores else None,
        "sample_score_mean": float(np.mean(scores)) if scores else None,
        "sample_score_max": max(scores) if scores else None,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_synthetic and not synthetic_flagged:
        raise SystemExit("synthetic positive was not flagged above clean usage")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mark3 H9 concept deviation detector")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["detect", "validate"]:
        p = sub.add_parser(name)
        p.add_argument("--paper", type=Path, default=ROOT / "data/showcases/ct-anatomy/golden/fable-0708.2757-dp-emacs.json")
        p.add_argument(
            "--encyclopedia-dir",
            type=Path,
            action="append",
            default=[ROOT / "data/concept-encyclopedia/ct", ROOT / "data/concept-encyclopedia/ct-golden"],
        )
        p.add_argument("--embed-dir", type=Path, default=ROOT / "tmp/mark3-embed/ct-sample")
        p.add_argument("--space", default="per-msc")
        p.add_argument("--context-backend", choices=["auto", "hash", "bge"], default="auto")
        p.add_argument("--threshold", type=float, default=0.55)
        p.add_argument("--context-chars", type=int, default=480)
        p.add_argument("--output", type=Path, default=ROOT / "tmp/mark3-deviation/sample-deviations.jsonl")
        p.add_argument("--report", type=Path, default=ROOT / "tmp/mark3-deviation/validation-report.json")
        p.add_argument("--require-synthetic", action="store_true", default=name == "validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "detect":
        records = detect(
            args.paper,
            args.encyclopedia_dir,
            args.embed_dir,
            args.output,
            args.space,
            args.context_backend,
            args.threshold,
            args.context_chars,
        )
        print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))
    elif args.cmd == "validate":
        validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
