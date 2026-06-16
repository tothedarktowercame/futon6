#!/usr/bin/env python3
"""Build a non-CT arXiv pool scored against APM informal-proof terms.

Inputs:
  * APM informal proofs in futon3c/data/apm-informal-proofs/*.md
  * mark2 batch tarballs containing batch-XYZ.jsonl and eprints/

Outputs:
  * data/apm-crossdisc-pool/pool.jsonl
  * data/apm-crossdisc-pool/eprints/<selected-id>.tar.gz
  * data/apm-crossdisc-pool/keyword-profile.json
  * data/apm-crossdisc-pool/summary.json

The n-gram extraction mirrors scripts/build_term_prior.py when available and
falls back to the same stopword-bounded extractor when that file is absent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APM = Path("/home/joe/code/futon3c/data/apm-informal-proofs")
DEFAULT_BATCHES = [
    Path("/home/joe/code/storage/mark2/inbox/batch-007.tar.gz"),
    Path("/home/joe/code/storage/mark2/inbox/batch-008.tar.gz"),
]
DEFAULT_OUT = ROOT / "data" / "apm-crossdisc-pool"
DEFAULT_TARGET_SIZE = 150

WORD_RE = re.compile(r"[a-z][a-z-]*")
STOP = {
    "the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "with",
    "is", "are", "be", "been", "being", "that", "this", "we", "it", "its",
    "by", "as", "from", "at", "if", "then", "which", "such", "any", "all",
    "each", "every", "some", "no", "not", "there", "where", "these", "those",
    "one", "two", "both", "also", "only", "so", "thus", "hence", "let",
    "given", "when",
}
TEXISH_TOKENS = {
    "align", "begin", "bibliography", "cal", "cdot", "cite", "documentclass",
    "end", "eqref", "frac", "hline", "label", "left", "mathbb", "mathcal",
    "mathrm", "newcommand", "newtheorem", "right", "text", "theoremstyle",
    "usepackage",
}
PROSE_DRIFT_TOKENS = {
    "actually", "also", "center", "clearly", "could", "do", "does", "done",
    "following", "however", "indeed", "may", "needed", "observe", "rather",
    "red", "see", "should", "since", "therefore", "via", "was", "were",
    "would",
    # APM proof-file template/provenance headers, not mathematical content.
    "agent", "apm", "complete", "connects", "cross-references", "date",
    "generated", "harvest", "hints", "import", "insight", "key", "lean",
    "mathlib", "measuretheory", "nl-proof-harvest", "noncomputable", "proof",
    "section", "source", "stage", "structures", "tactic", "theorem", "types",
    "what",
}
DOMAIN_HINTS = {
    "algebra", "analytic", "basis", "borel", "bounded", "compact", "continuous",
    "converges", "convergence", "differentiable", "dimension", "finite", "field",
    "function", "group", "ideal", "integrable", "integral", "linear", "matrix",
    "measurable", "measure", "metric", "norm", "open", "operator", "polynomial",
    "probability", "ring", "sequence", "series", "space", "subgroup", "subset",
    "topological", "vector",
}


@dataclass(frozen=True)
class BatchRecord:
    record: dict
    batch_tar: Path
    eprint_member: str
    eprint_name: str
    score: float = 0.0
    matched_terms: tuple[str, ...] = ()


def fallback_ngrams(words: list[str]) -> Iterable[str]:
    max_n = 4
    for i, word in enumerate(words):
        if word in STOP:
            continue
        for n in range(1, max_n + 1):
            if i + n > len(words):
                break
            seg = words[i : i + n]
            if seg[-1] in STOP:
                continue
            yield " ".join(seg)


def load_term_extractor():
    path = ROOT / "scripts" / "build_term_prior.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("build_term_prior", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module._WORD, module.ngrams, "scripts/build_term_prior.py"
    return WORD_RE, fallback_ngrams, "embedded build_term_prior-compatible fallback"


def strip_tex(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[$_^{}\\\\]", " ", text)
    return text


def is_math_candidate(term: str) -> bool:
    parts = term.split()
    if not parts:
        return False
    if any(p in TEXISH_TOKENS or p in PROSE_DRIFT_TOKENS for p in parts):
        return False
    if any(len(p) <= 2 for p in parts):
        return False
    if len(parts) == 1 and parts[0] not in DOMAIN_HINTS:
        return False
    return True


def extract_terms(text: str, word_re, ngrams) -> set[str]:
    return {
        t for t in ngrams(word_re.findall(strip_tex(text).lower()))
        if is_math_candidate(t)
    }


def build_keyword_profile(apm_dir: Path, top_k: int = 500) -> dict:
    word_re, ngrams, extractor = load_term_extractor()
    paths = sorted(apm_dir.glob("*.md"))
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        terms = list(extract_terms(text, word_re, ngrams))
        tf.update(terms)
        df.update(set(terms))
    ranked = []
    for term, count in tf.items():
        doc_count = df[term]
        score = count * math.log1p(doc_count)
        ranked.append({"term": term, "tf": count, "df": doc_count, "score": score})
    ranked.sort(key=lambda r: (-r["score"], -r["df"], r["term"]))
    return {
        "apm_dir": str(apm_dir),
        "proof_count": len(paths),
        "extractor": extractor,
        "terms": ranked[:top_k],
    }


def norm_eprint_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "__")


def eprint_candidates(record: dict) -> list[str]:
    ids = [record.get("id"), record.get("base_id")]
    out = []
    for raw in ids:
        if not raw:
            continue
        norm = norm_eprint_name(str(raw))
        for suffix in (".tar.gz", ".bin", ".tar", ".tex.gz", ".gz", ".tex"):
            out.append(norm + suffix)
    return out


def batch_jsonl_member(tf: tarfile.TarFile) -> str:
    matches = [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")]
    if not matches:
        raise FileNotFoundError("batch tarball has no .jsonl member")
    return sorted(matches)[0]


def eprint_member_index(tf: tarfile.TarFile) -> dict[str, str]:
    idx = {}
    for m in tf.getmembers():
        if m.isfile() and "/eprints/" in m.name:
            idx[Path(m.name).name] = m.name
    return idx


def iter_batch_records(batch_tars: list[Path]) -> Iterable[BatchRecord]:
    for batch_tar in batch_tars:
        with tarfile.open(batch_tar, "r:gz") as tf:
            eprints = eprint_member_index(tf)
            jsonl = batch_jsonl_member(tf)
            handle = tf.extractfile(jsonl)
            if handle is None:
                continue
            for raw in handle:
                record = json.loads(raw)
                member = None
                eprint_name = None
                for candidate in eprint_candidates(record):
                    if candidate in eprints:
                        member = eprints[candidate]
                        eprint_name = candidate
                        break
                if member and eprint_name:
                    yield BatchRecord(record=record, batch_tar=batch_tar,
                                      eprint_member=member, eprint_name=eprint_name)


def categories(record: dict) -> list[str]:
    cats = record.get("categories") or []
    if isinstance(cats, str):
        cats = cats.split()
    return [str(c) for c in cats]


def is_non_ct_math(record: dict) -> bool:
    cats = categories(record)
    return bool(cats) and "math.CT" not in cats and any(c.startswith("math.") for c in cats)


def score_record(record: dict, weights: dict[str, float], word_re, ngrams) -> tuple[float, tuple[str, ...]]:
    text = f"{record.get('title', '')} {record.get('abstract', '')}"
    present = extract_terms(text, word_re, ngrams)
    matched = sorted(present & set(weights))
    score = sum(weights[t] for t in matched)
    # Slightly prefer papers with multiple relevant terms over one repeated broad hit.
    score += 0.05 * len(matched)
    return score, tuple(matched)


def select_pool(records: list[BatchRecord], profile: dict, target_size: int) -> list[BatchRecord]:
    word_re, ngrams, _ = load_term_extractor()
    weights = {r["term"]: float(r["score"]) for r in profile["terms"]}
    scored = []
    for rec in records:
        if not is_non_ct_math(rec.record):
            continue
        score, matched = score_record(rec.record, weights, word_re, ngrams)
        if score <= 0:
            continue
        scored.append(BatchRecord(rec.record, rec.batch_tar, rec.eprint_member,
                                  rec.eprint_name, score, matched))
    scored.sort(key=lambda r: (-r.score, r.record.get("id", "")))
    return scored[:target_size]


def write_atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_path.replace(path)


def extract_selected_eprints(selected: list[BatchRecord], out_dir: Path) -> None:
    eprint_dir = out_dir / "eprints"
    if eprint_dir.exists():
        shutil.rmtree(eprint_dir)
    eprint_dir.mkdir(parents=True, exist_ok=True)
    by_batch: dict[Path, list[BatchRecord]] = {}
    for rec in selected:
        by_batch.setdefault(rec.batch_tar, []).append(rec)
    for batch_tar, rows in by_batch.items():
        with tarfile.open(batch_tar, "r:gz") as tf:
            for rec in rows:
                member = tf.extractfile(rec.eprint_member)
                if member is None:
                    raise FileNotFoundError(rec.eprint_member)
                target = eprint_dir / rec.eprint_name
                with target.open("wb") as out:
                    shutil.copyfileobj(member, out)


def category_histogram(selected: list[BatchRecord]) -> Counter[str]:
    hist: Counter[str] = Counter()
    for rec in selected:
        hist.update(categories(rec.record))
    return hist


def build_pool(apm_dir: Path, batch_tars: list[Path], out_dir: Path, target_size: int) -> dict:
    profile = build_keyword_profile(apm_dir)
    records = list(iter_batch_records(batch_tars))
    selected = select_pool(records, profile, target_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_jsonl(out_dir / "pool.jsonl", [r.record for r in selected])
    extract_selected_eprints(selected, out_dir)
    hist = category_histogram(selected)
    missing = [r.eprint_name for r in selected if not (out_dir / "eprints" / r.eprint_name).exists()]
    summary = {
        "schema": "futon6.apm-crossdisc-pool.v1",
        "apm_proofs": profile["proof_count"],
        "batch_tars": [str(p) for p in batch_tars],
        "candidate_records_with_eprints": len(records),
        "target_size": target_size,
        "pool_size": len(selected),
        "math_ct_count": hist.get("math.CT", 0),
        "eprints_complete": not missing,
        "missing_eprints": missing,
        "category_histogram": dict(sorted(hist.items())),
        "keyword_top20": profile["terms"][:20],
        "selected_ids": [str(r.record.get("id")) for r in selected],
        "selected_sample": [
            {
                "id": r.record.get("id"),
                "title": r.record.get("title"),
                "categories": categories(r.record),
                "score": r.score,
                "matched_terms": list(r.matched_terms[:20]),
                "eprint": r.eprint_name,
            }
            for r in selected[:20]
        ],
    }
    (out_dir / "keyword-profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apm-dir", type=Path, default=DEFAULT_APM)
    parser.add_argument("--batch-tar", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batch_tars = args.batch_tar or DEFAULT_BATCHES
    summary = build_pool(args.apm_dir, batch_tars, args.out_dir, args.target_size)
    print(f"APM proofs: {summary['apm_proofs']}")
    print("Top-20 keywords:")
    for row in summary["keyword_top20"]:
        print(f"  {row['term']}: score={row['score']:.2f} tf={row['tf']} df={row['df']}")
    print(f"Pool size: {summary['pool_size']}")
    print(f"math.CT selected: {summary['math_ct_count']}")
    print(f"Eprints complete: {summary['eprints_complete']}")
    print("Category histogram:")
    for cat, n in sorted(summary["category_histogram"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cat}: {n}")
    if summary["math_ct_count"] or not summary["eprints_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
