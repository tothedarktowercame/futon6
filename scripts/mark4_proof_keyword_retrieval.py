#!/usr/bin/env python3
"""CPU-only keyword retrieval for mark4 APM structure-learning evals.

The script extracts domain-agnostic technical terms from frozen APM informal
proofs, then ranks batch-007/008 arXiv papers by exact keyword hits in
title+abstract. Full-text eprint search is available behind --full-text but is
off by default. No external corpus or network resource is used.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = Path("/home/joe/code/storage/apm/mark4-frozen-candidates.txt")
DEFAULT_PROOF_DIR = Path("/home/joe/code/futon3c/data/apm-informal-proofs")
DEFAULT_BATCHES = [
    Path("/home/joe/code/storage/mark2/inbox/batch-007.tar.gz"),
    Path("/home/joe/code/storage/mark2/inbox/batch-008.tar.gz"),
]
DEFAULT_KEYWORDS_OUT = ROOT / "data" / "mark4-proof-keywords.json"
DEFAULT_HITS_OUT = ROOT / "data" / "mark4-batch-keyword-hits.json"
DEFAULT_TOP_TSV = ROOT / "data" / "mark4-retrieval-top200.tsv"

WORD_RE = re.compile(r"[a-z][a-z]+(?:-[a-z]+)?")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\$+")
CONTROL_RE = re.compile(r"\\([a-zA-Z]+)")
URL_RE = re.compile(r"https?://\S+")

LATEX_WORDS = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "varepsilon": "epsilon",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "pi": "pi",
    "rho": "rho",
    "sigma": "sigma",
    "tau": "tau",
    "phi": "phi",
    "varphi": "phi",
    "omega": "omega",
    "infty": "infinity",
    "infinite": "infinite",
    "sup": "supremum",
    "limsup": "limsup",
    "liminf": "liminf",
    "inf": "infimum",
    "min": "minimum",
    "max": "maximum",
    "ker": "kernel",
    "dim": "dimension",
    "rank": "rank",
    "det": "determinant",
    "hom": "hom",
    "ext": "ext",
    "tor": "tor",
}

STOPWORDS = {
    "a", "all", "also", "among", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "being", "between", "both", "but", "by", "can",
    "could", "does", "each", "every", "for", "from", "has", "have", "having",
    "hence", "if", "in", "into", "is", "it", "its", "let", "may", "more",
    "most", "must", "no", "not", "now", "of", "on", "one", "only", "or",
    "our", "over", "rather", "since", "so", "some", "such", "than", "that",
    "the", "their", "then", "there", "these", "this", "those", "through",
    "to", "two", "under", "using", "was", "we", "what", "when", "where",
    "which", "while", "with", "within", "would", "you",
}

BOILERPLATE_TERMS = {
    "above", "abs", "absmax", "aeval", "apm", "apply", "arbitrary", "asks",
    "assume", "assumption", "below", "bound", "case", "cases", "claim",
    "class", "classes", "cleaner", "combine", "combining", "complete",
    "complete proof", "conclude", "condition", "conditions", "consider",
    "core", "definition", "defined", "definitionally", "different", "direct",
    "direction", "elpnorm", "equal", "equality", "exactly", "exists",
    "explicit", "fact", "filter", "finding", "first", "fixed", "following",
    "form", "forms", "general", "get", "given", "gives", "hold", "holds",
    "implicit", "key", "key insight", "largest", "lean", "lemma", "geq",
    "ispreconnected", "isn", "leq", "left", "lower", "many", "mathlib",
    "means", "method", "methods", "mul", "nat", "natural", "need", "nonneg",
    "order", "orders", "previous", "problem", "problems", "proof", "prove",
    "pure", "relation", "relations", "representation", "requires", "respect",
    "result", "right", "rpow", "second", "self", "show", "shown", "shows",
    "side", "sides", "simply", "smallest", "statement", "step", "suppose",
    "support", "take", "taking", "technique", "tendsto", "theorem",
    "therefore", "thus", "time", "type", "univ", "upper", "way", "well",
    "why", "why hard", "without",
}

BOILERPLATE_PARTS = {
    "abs", "absmax", "aeval", "all", "also", "apm", "apply", "asks", "bound",
    "case", "cases", "class", "classes", "cleaner", "condition", "conditions",
    "core", "definitionally", "direct", "elpnorm", "equal", "equality",
    "explicit", "filter", "finding", "first", "form", "forms", "general",
    "get", "implicit", "largest", "lower", "many", "means", "method",
    "methods",
    "proof", "theorem", "lemma", "definition", "defined", "suppose", "given",
    "claim", "geq", "ispreconnected", "isn", "leq", "lean", "mathlib", "mul",
    "nat", "natural", "nonneg", "order", "orders", "previous", "problem",
    "problems", "pure", "relation", "relations", "representation", "respect",
    "rpow", "second", "self", "show", "shows", "simply", "smallest",
    "complete", "insight", "hard", "support", "tendsto", "then", "thus",
    "hence", "therefore", "time", "type", "univ", "upper", "using", "what",
    "when", "you",
}


def normalize_text(text: str, *, drop_fences: bool = True) -> str:
    if drop_fences:
        text = FENCE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("L^p", "lp").replace("L^\\infty", "l infinity")
    text = text.replace("L^∞", "l infinity")
    text = text.replace("∞", " infinity ")
    text = text.replace("≤", " leq ").replace("≥", " geq ")
    text = text.replace("∈", " in ").replace("∫", " integral ")
    text = text.replace("∑", " sum ").replace("∏", " product ")
    text = text.replace("∂", " boundary ").replace("∇", " gradient ")
    text = text.replace("→", " to ").replace("⇒", " implies ")
    text = INLINE_MATH_RE.sub(" ", text)

    def control_to_word(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        return " " + LATEX_WORDS.get(name, " ") + " "

    text = CONTROL_RE.sub(control_to_word, text)
    text = re.sub(r"[_^{}()[\],.;:!?*=<>|/#`~\"'’“”]", " ", text)
    return text.lower()


def tokens(text: str, *, drop_fences: bool = True) -> list[str]:
    return WORD_RE.findall(normalize_text(text, drop_fences=drop_fences))


def term_counts(words: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for i in range(len(words)):
        for width in (1, 2, 3):
            if i + width > len(words):
                break
            phrase = words[i:i + width]
            if usable_phrase(phrase):
                counts[" ".join(phrase)] += 1
    return counts


def usable_unigram(word: str) -> bool:
    return (
        len(word) >= 3
        and word not in STOPWORDS
        and word not in BOILERPLATE_TERMS
        and "mathlib" not in word
        and not word.isdigit()
    )


def usable_phrase(words: list[str]) -> bool:
    if len(words) == 1:
        return usable_unigram(words[0])
    if len(set(words)) != len(words):
        return False
    if any(word in STOPWORDS for word in words):
        return False
    if any(word in BOILERPLATE_PARTS for word in words):
        return False
    term = " ".join(words)
    return term not in BOILERPLATE_TERMS and all(usable_unigram(word) for word in words)


def load_frozen_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def extract_keywords(
    proof_ids: list[str],
    proof_dir: Path,
    df_threshold: float,
    per_proof_k: int,
) -> dict[str, list[str]]:
    proof_counts: dict[str, Counter[str]] = {}
    proof_df: Counter[str] = Counter()
    max_df = max(1, int(len(proof_ids) * df_threshold))

    for proof_id in proof_ids:
        text = proof_text(proof_dir, proof_id)
        counts = term_counts(tokens(text, drop_fences=True))
        proof_counts[proof_id] = counts
        proof_df.update(counts.keys())

    out: dict[str, list[str]] = {}
    for proof_id in proof_ids:
        scored = []
        for term, count in proof_counts[proof_id].items():
            if proof_df[term] > max_df:
                continue
            parts = term.split()
            salience = count * len(parts)
            scored.append((salience, len(parts), count, len(term), term))
        scored.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], row[4]))
        out[proof_id] = [term for _, _, _, _, term in scored[:per_proof_k]]
    return out


def proof_text(proof_dir: Path, proof_id: str) -> str:
    proof_path = proof_dir / f"apm-{proof_id}.md"
    text = proof_path.read_text(errors="replace")
    return re.split(r"^## Lean 4 Theorem Statements\b", text, flags=re.MULTILINE)[0]


def batch_name(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".tar.gz") else path.stem


def iter_batch_records(batch_tar: Path):
    bname = batch_name(batch_tar)
    member_name = f"{bname}/{bname}.jsonl"
    with tarfile.open(batch_tar, "r:gz") as tf:
        member = tf.getmember(member_name)
        stream = tf.extractfile(member)
        if stream is None:
            return
        for raw in stream:
            if raw.strip():
                yield json.loads(raw)


def eprint_member_name(record: dict, bname: str) -> str:
    arxiv_id = str(record["id"]).replace("/", "__")
    return f"{bname}/eprints/{arxiv_id}.tar.gz"


def read_eprint_text(tf: tarfile.TarFile, record: dict, bname: str) -> str:
    name = eprint_member_name(record, bname)
    try:
        member = tf.getmember(name)
    except KeyError:
        return ""
    outer_file = tf.extractfile(member)
    if outer_file is None:
        return ""
    try:
        blob = outer_file.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as inner:
            chunks = []
            for inner_member in sorted(inner.getmembers(), key=lambda m: m.name):
                lname = inner_member.name.lower()
                if not inner_member.isfile() or not lname.endswith((".tex", ".ltx", ".bbl")):
                    continue
                f = inner.extractfile(inner_member)
                if f is not None:
                    chunks.append(f.read().decode("utf-8", errors="replace"))
            return "\n".join(chunks)
    except tarfile.TarError:
        return ""


def count_keyword_hits(text: str, keywords: set[str]) -> tuple[int, list[str]]:
    counts = term_counts(tokens(text, drop_fences=False))
    matched = []
    n_hits = 0
    for keyword in sorted(keywords):
        c = counts.get(keyword, 0)
        if c:
            matched.append(keyword)
            n_hits += c
    return n_hits, matched


def search_batches(
    batch_tars: list[Path],
    keywords_by_proof: dict[str, list[str]],
    *,
    full_text: bool,
) -> list[dict]:
    keyword_sources: dict[str, list[str]] = defaultdict(list)
    for proof_id, terms in sorted(keywords_by_proof.items()):
        for term in terms:
            keyword_sources[term].append(proof_id)
    keywords = set(keyword_sources)

    hits = []
    for batch_tar in batch_tars:
        bname = batch_name(batch_tar)
        if full_text:
            with tarfile.open(batch_tar, "r:gz") as tf:
                for record in iter_batch_records(batch_tar):
                    text = paper_text(record)
                    text += "\n" + read_eprint_text(tf, record, bname)
                    add_hit(record, text, keywords, keyword_sources, hits)
        else:
            for record in iter_batch_records(batch_tar):
                add_hit(record, paper_text(record), keywords, keyword_sources, hits)

    hits.sort(key=lambda row: (-row["n_hits"], -row["n_distinct"], row["id"]))
    return hits


def paper_text(record: dict) -> str:
    return f"{record.get('title', '')}\n{record.get('abstract', '')}"


def add_hit(
    record: dict,
    text: str,
    keywords: set[str],
    keyword_sources: dict[str, list[str]],
    hits: list[dict],
) -> None:
    n_hits, matched = count_keyword_hits(text, keywords)
    if not matched:
        return
    source_proofs = sorted({pid for kw in matched for pid in keyword_sources[kw]})
    hits.append(
        {
            "id": record.get("id"),
            "title": record.get("title", ""),
            "primary_category": record.get("primary_category", ""),
            "n_distinct": len(matched),
            "n_hits": n_hits,
            "matched_keywords": matched,
            "source_proofs": source_proofs,
        }
    )


def write_top_tsv(path: Path, hits: list[dict], top_n: int) -> None:
    lines = ["rank\tid\tprimary_category\tn_distinct\tn_hits\ttitle\tmatched_keywords"]
    for rank, row in enumerate(hits[:top_n], start=1):
        title = str(row["title"]).replace("\t", " ").replace("\n", " ")
        matched = ", ".join(row["matched_keywords"])
        lines.append(
            f"{rank}\t{row['id']}\t{row['primary_category']}\t{row['n_distinct']}"
            f"\t{row['n_hits']}\t{title}\t{matched}"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-candidates", type=Path, default=DEFAULT_FROZEN)
    ap.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    ap.add_argument("--batch", type=Path, action="append", dest="batches")
    ap.add_argument("--keywords-out", type=Path, default=DEFAULT_KEYWORDS_OUT)
    ap.add_argument("--hits-out", type=Path, default=DEFAULT_HITS_OUT)
    ap.add_argument("--top-tsv", type=Path, default=DEFAULT_TOP_TSV)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument(
        "--df-threshold",
        type=float,
        default=0.1,
        help="drop terms appearing in more than this fraction of frozen proofs",
    )
    ap.add_argument("--per-proof-k", type=int, default=20)
    ap.add_argument("--full-text", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batches = args.batches or DEFAULT_BATCHES
    proof_ids = load_frozen_ids(args.frozen_candidates)

    keywords_by_proof = extract_keywords(
        proof_ids,
        args.proof_dir,
        args.df_threshold,
        args.per_proof_k,
    )
    args.keywords_out.parent.mkdir(parents=True, exist_ok=True)
    args.keywords_out.write_text(json.dumps(keywords_by_proof, indent=2, sort_keys=True) + "\n")

    hits = search_batches(batches, keywords_by_proof, full_text=args.full_text)
    args.hits_out.parent.mkdir(parents=True, exist_ok=True)
    args.hits_out.write_text(json.dumps(hits, indent=2, sort_keys=True) + "\n")
    write_top_tsv(args.top_tsv, hits, args.top)

    print(f"proofs: {len(proof_ids)}")
    print(f"unique keywords: {len({kw for kws in keywords_by_proof.values() for kw in kws})}")
    print(f"matched papers: {len(hits)}")
    print(f"wrote: {args.keywords_out}")
    print(f"wrote: {args.hits_out}")
    print(f"wrote: {args.top_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
