#!/usr/bin/env python3
"""CPU-only keyword retrieval for mark4 APM structure-learning evals.

The script extracts distinctive uni- and bi-grams from frozen APM informal
proofs, then ranks batch-007/008 arXiv papers by exact keyword hits in
title+abstract. Full-text eprint search is available behind --full-text but is
off by default.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = Path("/home/joe/code/storage/apm/mark4-frozen-candidates.txt")
DEFAULT_PROOF_DIR = Path("/home/joe/code/futon3c/data/apm-informal-proofs")
DEFAULT_PRIOR = ROOT / "data" / "ct-term-prior.json"
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
    "because", "been", "being", "between", "both", "but", "by", "can", "could", "does", "each",
    "every", "for", "from", "has", "have", "having", "hence", "if", "in",
    "into", "is", "it", "its", "let", "may", "more", "must", "no", "not",
    "now", "of", "on", "one", "only", "or", "our", "over", "so", "some",
    "rather", "since", "such", "than", "that", "the", "their", "then", "there",
    "these", "this", "those", "through", "to", "two", "under", "using", "was",
    "we", "when", "where", "which", "while", "with", "within", "would", "you",
}

BOILERPLATE_TERMS = {
    "above", "abs", "absmax", "aeval", "apm", "arbitrary", "asks", "assume", "assumption", "below", "claim",
    "cleaner", "combine", "combining", "complete", "complete proof", "conclude",
    "consider", "core", "definition", "defined", "definitionally", "different",
    "direction", "elpnorm", "exactly", "exists", "fact", "filter", "fixed", "following",
    "given", "gives", "hold", "holds", "key", "key insight", "lean", "lemma",
    "geq", "ispreconnected", "isn", "leq", "left", "mathlib", "mul", "nat",
    "need", "nonneg", "proof", "prove", "requires", "result", "right", "rpow",
    "show", "shown", "shows", "side", "sides", "statement", "step", "suppose",
    "take", "taking", "technique", "tendsto", "theorem", "therefore", "thus",
    "univ", "way", "well", "why", "why hard", "without",
}

BOILERPLATE_PARTS = {
    "abs", "absmax", "aeval", "all", "also", "apm", "asks", "cleaner", "core", "definitionally", "elpnorm", "filter",
    "proof", "theorem", "lemma", "definition", "defined", "suppose", "given",
    "claim", "geq", "ispreconnected", "isn", "leq", "lean", "mathlib", "mul",
    "nat", "nonneg", "rpow", "show", "shows", "complete", "insight", "hard",
    "tendsto", "then", "thus", "hence", "therefore", "univ", "using", "when",
    "you",
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
    for word in words:
        if usable_unigram(word):
            counts[word] += 1
    for left, right in zip(words, words[1:]):
        if usable_bigram(left, right):
            counts[f"{left} {right}"] += 1
    return counts


def usable_unigram(word: str) -> bool:
    return (
        len(word) >= 3
        and word not in STOPWORDS
        and word not in BOILERPLATE_TERMS
        and "mathlib" not in word
        and not word.isdigit()
    )


def usable_bigram(left: str, right: str) -> bool:
    if left == right:
        return False
    if left in STOPWORDS or right in STOPWORDS:
        return False
    if left in BOILERPLATE_PARTS or right in BOILERPLATE_PARTS:
        return False
    term = f"{left} {right}"
    return term not in BOILERPLATE_TERMS and usable_unigram(left) and usable_unigram(right)


def read_prior(path: Path) -> tuple[int, dict[str, int], dict[str, int]]:
    with path.open() as f:
        prior = json.load(f)
    return (
        int(prior["n_docs"]),
        {str(k): int(v) for k, v in prior.get("unigram_df", {}).items()},
        {str(k): int(v) for k, v in prior.get("bigram_df", {}).items()},
    )


def load_frozen_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def distinctiveness_score(
    term: str,
    count: int,
    n_docs: int,
    unigram_df: dict[str, int],
    bigram_df: dict[str, int],
    df_threshold: float,
) -> float | None:
    df_table = bigram_df if " " in term else unigram_df
    df = df_table.get(term, 0)
    if df and df / n_docs > df_threshold:
        return None
    if df == 0 and not all(part.isalpha() and len(part) >= 3 for part in term.split()):
        return None
    rarity = math.log((n_docs + 1.0) / (df + 1.0))
    phrase_bonus = 1.35 if " " in term else 1.0
    freq_bonus = 1.0 + math.log1p(count)
    return rarity * phrase_bonus * freq_bonus


def extract_keywords(
    proof_ids: list[str],
    proof_dir: Path,
    n_docs: int,
    unigram_df: dict[str, int],
    bigram_df: dict[str, int],
    df_threshold: float,
    per_proof_k: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for proof_id in proof_ids:
        proof_path = proof_dir / f"apm-{proof_id}.md"
        text = proof_path.read_text(errors="replace")
        text = re.split(r"^## Lean 4 Theorem Statements\b", text, flags=re.MULTILINE)[0]
        counts = term_counts(tokens(text, drop_fences=True))
        scored = []
        for term, count in counts.items():
            score = distinctiveness_score(
                term, count, n_docs, unigram_df, bigram_df, df_threshold
            )
            if score is not None:
                scored.append((score, len(term.split()), count, term))
        scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        out[proof_id] = [term for _, _, _, term in scored[:per_proof_k]]
    return out


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

    hits.sort(key=lambda row: (-row["n_distinct"], -row["n_hits"], row["id"]))
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
    ap.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    ap.add_argument("--batch", type=Path, action="append", dest="batches")
    ap.add_argument("--keywords-out", type=Path, default=DEFAULT_KEYWORDS_OUT)
    ap.add_argument("--hits-out", type=Path, default=DEFAULT_HITS_OUT)
    ap.add_argument("--top-tsv", type=Path, default=DEFAULT_TOP_TSV)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--df-threshold", type=float, default=0.4)
    ap.add_argument("--per-proof-k", type=int, default=20)
    ap.add_argument("--full-text", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batches = args.batches or DEFAULT_BATCHES
    proof_ids = load_frozen_ids(args.frozen_candidates)
    n_docs, unigram_df, bigram_df = read_prior(args.prior)

    keywords_by_proof = extract_keywords(
        proof_ids,
        args.proof_dir,
        n_docs,
        unigram_df,
        bigram_df,
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
