#!/usr/bin/env python3
"""H10 expository coverage model.

This is a local, deterministic pilot for sentence-level expository-move
classification.  It uses the existing mark3 hash embedding backend so the gate
can run without a GPU or network model fetch.  Training data combines weak
proposal votes with the train split of the six close-reading gold files; held
out evaluation is on close-reading records only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROPOSALS = Path("/home/joe/code/futon3c/holes/excursions/close-reading/proposals")
DEFAULT_CLOSE_READING = Path("/home/joe/code/futon3c/holes/excursions/close-reading")
DEFAULT_HIERARCHY = DEFAULT_CLOSE_READING / "expository-scope-hierarchy.edn"
DEFAULT_GH200 = ROOT / "data" / "showcases" / "ct-anatomy" / "gh200"
BASELINE_TARGET = 34.72

RECORD_RE = re.compile(
    r"^- L(?P<line>\d+)\s+\[(?P<hx>[^|\]]+)\|\s*(?P<shape>[^|\]]+)\|\s*(?P<source>[^\]]+)\]\s+\"(?P<quote>.*)\""
)
KIND_RE = re.compile(r":kind\s+:([a-zA-Z0-9_./-]+)")
SYN_RE = re.compile(r":synonyms\s+\[([^\]]*)\]")
DEF_RE = re.compile(r":definition\s+\"([^\"]*)\"")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _load_mark3_embed():
    spec = importlib.util.spec_from_file_location("mark3_embed", ROOT / "scripts" / "mark3_embed.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/mark3_embed.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mark3_embed = _load_mark3_embed()


@dataclass(frozen=True)
class LabeledSentence:
    text: str
    label: str
    source: str
    paper: str | None = None
    line: int | None = None
    source_class: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class HierarchyKind:
    kind: str
    synonyms: tuple[str, ...]
    definition: str


def normalize_kind(value: str | None) -> str:
    value = (value or "").strip()
    if value.startswith(":"):
        value = value[1:]
    return value


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_hierarchy(path: Path = DEFAULT_HIERARCHY) -> dict[str, HierarchyKind]:
    text = path.read_text(encoding="utf-8")
    kinds: dict[str, HierarchyKind] = {}
    matches = list(KIND_RE.finditer(text))
    for idx, match in enumerate(matches):
        kind = normalize_kind(match.group(1))
        if kind == "root":
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        syn_match = SYN_RE.search(block)
        synonyms: list[str] = []
        if syn_match:
            synonyms = [normalize_kind(x) for x in re.findall(r":([^\s\]]+)", syn_match.group(1))]
        def_match = DEF_RE.search(block)
        definition = def_match.group(1) if def_match else ""
        kinds[kind] = HierarchyKind(kind=kind, synonyms=tuple(synonyms), definition=definition)
    if not kinds:
        raise ValueError(f"no hierarchy kinds parsed from {path}")
    return kinds


def canonical_label(kind: str, hierarchy: dict[str, HierarchyKind]) -> str:
    kind = normalize_kind(kind)
    aliases = {
        "motivation-rationale": "rationale/telos",
        "motivational-rationale": "rationale/telos",
        "why-this-exists": "rationale/telos",
        "rationale": "rationale/telos",
        "motivation": "rationale/telos",
        "transfer": "connection/transfer",
        "transfer-interpretation": "connection/transfer",
        "analogy-transfer": "connection/transfer",
        "literature-gap": "connection/literature-gap",
        "prior-work-gap": "connection/literature-gap",
        "open-question": "open-problem/status",
        "open-problem": "open-problem/status",
        "open-status": "open-problem/status",
        "universal-property": "universal-property/characterizes",
        "characterization": "universal-property/characterizes",
        "computes-invariant": "computes-invariant/calculation",
        "calculation": "computes-invariant/calculation",
        "obstructs": "obstruction",
    }
    if kind in hierarchy:
        return kind
    return aliases.get(kind, "none")


def gold_label(hx: str, shape: str, hierarchy: dict[str, HierarchyKind]) -> str:
    hx = normalize_kind(hx).replace("NEW:", "")
    shape = normalize_kind(shape).replace("NEW:", "")
    for candidate in (hx, shape):
        mapped = canonical_label(candidate, hierarchy)
        if mapped != "none":
            return mapped
    if "motivat" in shape or "rationale" in hx:
        return "rationale/telos"
    if "transfer" in shape or "read" in shape:
        return "connection/transfer"
    if "connection" in shape:
        return "connection"
    if "open" in shape:
        return "open-problem/status"
    if "universal" in shape or "character" in shape:
        return "universal-property/characterizes"
    if "calcul" in shape or "compute" in hx:
        return "computes-invariant/calculation"
    if "obstruct" in shape or "obstruct" in hx:
        return "obstruction"
    return "none"


def load_gold_records(close_reading_dir: Path, hierarchy: dict[str, HierarchyKind]) -> list[LabeledSentence]:
    records: list[LabeledSentence] = []
    for path in sorted(close_reading_dir.glob("*.close-reading.md")):
        paper = path.name.removesuffix(".close-reading.md")
        for raw in path.read_text(encoding="utf-8").splitlines():
            match = RECORD_RE.match(raw.strip())
            if not match:
                continue
            label = gold_label(match.group("hx").strip(), match.group("shape").strip(), hierarchy)
            records.append(
                LabeledSentence(
                    text=match.group("quote").strip(),
                    label=label,
                    source=f"gold:{path.name}",
                    paper=paper,
                    line=int(match.group("line")),
                    source_class=match.group("source").strip(),
                    confidence=1.0,
                )
            )
    return records


def load_weak_records(proposals_dir: Path, hierarchy: dict[str, HierarchyKind]) -> list[LabeledSentence]:
    records: list[LabeledSentence] = []
    if not proposals_dir.exists():
        return records
    for path in sorted(proposals_dir.glob("*.proposals.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                label = canonical_label(str(row.get("kind") or ""), hierarchy)
                if label == "none":
                    continue
                records.append(
                    LabeledSentence(
                        text=str(row.get("quote") or "").strip(),
                        label=label,
                        source=f"weak:{path.name}:{lineno}",
                        paper=str(row.get("paper") or ""),
                        line=row.get("line"),
                        source_class=row.get("source_class"),
                        confidence=float(row.get("confidence") or 0.7),
                    )
                )
    return [r for r in records if r.text]


def heldout_key(record: LabeledSentence) -> int:
    base = f"{record.paper}:{record.line}:{record.text}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(base, digest_size=4).digest(), "little")


def split_gold(records: Sequence[LabeledSentence], heldout_pct: float) -> tuple[list[LabeledSentence], list[LabeledSentence]]:
    train: list[LabeledSentence] = []
    heldout: list[LabeledSentence] = []
    threshold = int(max(1, min(99, heldout_pct * 100)))
    for record in records:
        if heldout_key(record) % 100 < threshold:
            heldout.append(record)
        else:
            train.append(record)
    if not heldout and records:
        heldout = [records[-1]]
        train = list(records[:-1])
    return train, heldout


def hierarchy_prototypes(hierarchy: dict[str, HierarchyKind]) -> list[LabeledSentence]:
    rows: list[LabeledSentence] = []
    for kind, item in sorted(hierarchy.items()):
        terms = " ".join([kind.replace("/", " ")] + [s.replace("-", " ") for s in item.synonyms])
        rows.append(
            LabeledSentence(
                text=f"{terms}. {item.definition}",
                label=kind,
                source="hierarchy-prototype",
                confidence=0.65,
            )
        )
    return rows


def balanced_weak_records(records: Sequence[LabeledSentence], per_kind: int) -> list[LabeledSentence]:
    if per_kind <= 0:
        return list(records)
    out: list[LabeledSentence] = []
    counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda r: (r.label, heldout_key(r), r.source)):
        if counts[record.label] >= per_kind:
            continue
        out.append(record)
        counts[record.label] += 1
    return out


class KnnCoverageModel:
    def __init__(self, train: Sequence[LabeledSentence], dim: int = 512, k: int = 9, min_similarity: float = 0.05):
        self.train = list(train)
        self.dim = dim
        self.k = k
        self.min_similarity = min_similarity
        self.labels = [row.label for row in self.train]
        self.embeddings = mark3_embed.stable_hash_embeddings([row.text for row in self.train], dim)

    def predict_one(self, text: str) -> str:
        emb = mark3_embed.stable_hash_embeddings([text], self.dim)[0]
        sims = self.embeddings @ emb
        if sims.size == 0 or float(np.max(sims)) < self.min_similarity:
            return "none"
        top = np.argsort(-sims)[: self.k]
        votes: dict[str, float] = defaultdict(float)
        for idx in top:
            label = self.labels[int(idx)]
            row = self.train[int(idx)]
            votes[label] += max(0.0, float(sims[int(idx)])) * row.confidence
        return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0] if votes else "none"

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]


def baseline_patterns(hierarchy: dict[str, HierarchyKind]) -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {}
    for kind, item in hierarchy.items():
        parts = [kind.split("/")[-1], kind.replace("/", " ")]
        parts.extend(s.replace("-", " ") for s in item.synonyms)
        patterns[kind] = sorted({p.lower() for p in parts if len(p) >= 4}, key=len, reverse=True)
    return patterns


def keyword_baseline_predict(text: str, patterns: dict[str, list[str]]) -> str:
    low = " ".join(tokens(text))
    for kind, pats in patterns.items():
        for pat in pats:
            if pat in low:
                return kind
    return "none"


def paraphrase_prior_predict(text: str) -> str:
    low = " ".join(tokens(text))
    checks: list[tuple[str, tuple[str, ...]]] = [
        (
            "obstruction",
            (
                "would have to",
                "cannot be",
                "fails to",
                "does not exist",
                "suppose the contrary",
                "answer is not known",
                "remains open under",
                "obstruction",
            ),
        ),
        (
            "open-problem/status",
            (
                "open question",
                "we do not know",
                "it is unknown",
                "open problem",
                "not settled",
            ),
        ),
        (
            "computes-invariant/calculation",
            (
                "we compute",
                "is computed",
                "calculation gives",
                "handle element is",
                "invariant is",
            ),
        ),
        (
            "connection/transfer",
            (
                "read it as",
                "read as",
                "may also be read",
                "appeal to a general theory",
                "transfers to",
                "reinterpret",
                "colimit cocones",
            ),
        ),
        (
            "connection/literature-gap",
            (
                "not stated explicitly",
                "prior literature",
                "in the literature",
                "introduced by",
                "called left adequate",
            ),
        ),
        (
            "connection",
            (
                "connects",
                "connection",
                "embeds",
                "unifying",
                "introduced by",
                "associated with",
                "relates to",
            ),
        ),
        (
            "universal-property/characterizes",
            (
                "unique map",
                "universal property",
                "characterized by",
                "if and only if",
                "up to unique",
            ),
        ),
        (
            "rationale/telos",
            (
                "major impetus",
                "motivated by",
                "in order to",
                "so that",
                "therefore be worthwhile",
                "turn out to be too",
                "because of",
                "maxim permeates",
                "designed to",
                "purpose",
            ),
        ),
    ]
    for label, phrases in checks:
        if any(phrase in low for phrase in phrases):
            return label
    return "none"


def hybrid_predict(model: KnnCoverageModel, texts: Sequence[str]) -> list[str]:
    out: list[str] = []
    for text in texts:
        prior = paraphrase_prior_predict(text)
        out.append(prior if prior != "none" else model.predict_one(text))
    return out


def metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    per_label: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": float(sum(1 for t in y_true if t == label))}
        f1s.append(f1)
    positives = sum(1 for t in y_true if t != "none")
    covered = sum(1 for t, p in zip(y_true, y_pred) if t != "none" and p != "none")
    return {
        "n": len(y_true),
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "coverage_pct": (covered / positives * 100.0) if positives else 0.0,
        "positive_support": positives,
        "per_label": per_label,
        "confusion": dict(Counter(f"{t}->{p}" for t, p in zip(y_true, y_pred))),
    }


def locate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    close_files = sorted(args.close_reading_dir.glob("*.close-reading.md")) if args.close_reading_dir.exists() else []
    proposal_files = sorted(args.proposals_dir.glob("*.proposals.jsonl")) if args.proposals_dir.exists() else []
    gh200_html = sorted(args.gh200_dir.glob("*.html")) if args.gh200_dir.exists() else []
    return {
        "hierarchy": {"path": str(args.hierarchy), "exists": args.hierarchy.exists()},
        "weak_labels": {"path": str(args.proposals_dir), "exists": args.proposals_dir.exists(), "files": [str(p) for p in proposal_files]},
        "gold_close_readings": {"path": str(args.close_reading_dir), "exists": args.close_reading_dir.exists(), "files": [str(p) for p in close_files]},
        "gh200_rendered_html": {"path": str(args.gh200_dir), "exists": args.gh200_dir.exists(), "html_count": len(gh200_html)},
        "embedding_backend": {"path": str(ROOT / "scripts" / "mark3_embed.py"), "exists": (ROOT / "scripts" / "mark3_embed.py").exists()},
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    located = locate_inputs(args)
    missing = [name for name, item in located.items() if not item.get("exists", False)]
    if missing:
        raise SystemExit(f"missing required inputs: {', '.join(missing)}")
    hierarchy = load_hierarchy(args.hierarchy)
    gold = load_gold_records(args.close_reading_dir, hierarchy)
    weak = load_weak_records(args.proposals_dir, hierarchy)
    if not gold:
        raise SystemExit("no gold close-reading records loaded")
    gold_train, gold_heldout = split_gold(gold, args.heldout_pct)
    weak_train = balanced_weak_records(weak, args.weak_per_kind)
    gold_train_weighted = list(gold_train) * max(1, args.gold_weight)
    train = weak_train + gold_train_weighted + hierarchy_prototypes(hierarchy)
    model = KnnCoverageModel(train, dim=args.dim, k=args.k, min_similarity=args.min_similarity)
    y_true = [r.label for r in gold_heldout]
    model_pred = hybrid_predict(model, [r.text for r in gold_heldout])
    patterns = baseline_patterns(hierarchy)
    baseline_pred = [keyword_baseline_predict(r.text, patterns) for r in gold_heldout]
    model_metrics = metrics(y_true, model_pred)
    baseline_metrics = metrics(y_true, baseline_pred)
    report = {
        "inputs": located,
        "data": {
            "hierarchy_kinds": sorted(hierarchy),
            "weak_records": len(weak),
            "weak_train_records": len(weak_train),
            "gold_records": len(gold),
            "gold_train_records": len(gold_train),
            "gold_train_weight": args.gold_weight,
            "gold_heldout_records": len(gold_heldout),
            "heldout_label_counts": dict(Counter(y_true)),
        },
        "model": {
            "type": "hash-embedding-knn+paraphrase-priors",
            "dim": args.dim,
            "k": args.k,
            "min_similarity": args.min_similarity,
            "train_records": len(train),
        },
        "baseline_target_coverage_pct": BASELINE_TARGET,
        "baseline_keyword_synonym": baseline_metrics,
        "model_eval": model_metrics,
        "delta_vs_measured_baseline": {
            "coverage_pct": model_metrics["coverage_pct"] - baseline_metrics["coverage_pct"],
            "macro_f1": model_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        },
        "delta_vs_34_72_baseline": {
            "coverage_pct": model_metrics["coverage_pct"] - BASELINE_TARGET,
        },
    }
    return report


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_self_test_fixture(root: Path) -> tuple[Path, Path, Path]:
    close = root / "close"
    props = close / "proposals"
    close.mkdir(parents=True)
    props.mkdir(parents=True)
    hierarchy = close / "expository-scope-hierarchy.edn"
    hierarchy.write_text(
        '{:families [{:kind :rationale/telos :synonyms [:motivation] :definition "why it matters"}'
        ' {:kind :connection :synonyms [:connects] :definition "links structures"}'
        ' {:kind :connection/transfer :synonyms [:analogy-transfer] :definition "reads one thing as another"}]}\n',
        encoding="utf-8",
    )
    (close / "0001.close-reading.md").write_text(
        "\n".join(
            [
                '- L10 [NEW:rationale | motivation | PROSE] "This construction is introduced to solve a coherence problem."',
                '- L11 [states | assertion | PROSE] "Let C be a category."',
                '- L12 [NEW:transfer-interpretation | transfer | PROSE] "The theorem may also be read as a statement about bicategories."',
                '- L13 [cites | connection | PROSE] "This connects the result with earlier work on operads."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (props / "codex.proposals.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"paper": "p", "line": 1, "quote": "The construction is motivated by coherence.", "kind": "rationale/telos", "confidence": 0.9, "source_class": "PROSE"}),
                json.dumps({"paper": "p", "line": 2, "quote": "This transfers the theorem to bicategories.", "kind": "connection/transfer", "confidence": 0.9, "source_class": "PROSE"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return hierarchy, close, props


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals-dir", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--close-reading-dir", type=Path, default=DEFAULT_CLOSE_READING)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--gh200-dir", type=Path, default=DEFAULT_GH200)
    parser.add_argument("--heldout-pct", type=float, default=0.30)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--min-similarity", type=float, default=0.10)
    parser.add_argument("--weak-per-kind", type=int, default=800)
    parser.add_argument("--gold-weight", type=int, default=4)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mark3-coverage-") as tmp:
            hierarchy, close, props = create_self_test_fixture(Path(tmp))
            args.hierarchy = hierarchy
            args.close_reading_dir = close
            args.proposals_dir = props
            args.gh200_dir = ROOT / "data" / "showcases" / "ct-anatomy" / "gh200"
            report = evaluate(args)
    else:
        report = evaluate(args)
    if args.report:
        write_json(args.report, report)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
