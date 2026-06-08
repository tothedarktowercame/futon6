#!/usr/bin/env python3
"""Build the mission-domain NER kernel and the two mission lexicons.

Pass 2 for M-web-arxana-missions Layer 3.  The prior is empirical
document-frequency over mission docs; this script adds the domain arbiter:
typed registry seeds reclaim English-word FUTON concepts, while generic
software tokens and ubiquitous boilerplate are suppressed.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/joe/code")
FUTON6 = ROOT / "futon6"
PRIOR = FUTON6 / "data" / "mission-term-prior.json"
KERNEL_OUT = FUTON6 / "data" / "mission-ner-kernel.json"
SELF_REP_OUT = FUTON6 / "data" / "mission-self-representing-lexicon.json"
PROJECTION_OUT = FUTON6 / "data" / "mission-projection-into-english-lexicon.json"
WORDS = Path("/usr/share/dict/words")

PUDDING_REGISTRY = ROOT / "futon7" / "holes" / "pudding-prover-registry.edn"
PUDDING_DOC = ROOT / "futon7" / "holes" / "C-pudding-prover.md"
STAR_MAP = ROOT / "futon0" / "holes" / "missions" / "M-capability-star-map.graph.edn"
FLEXIARG_ROOT = ROOT / "futon3" / "library"

HIGH_DF_MIN = 20
BOILERPLATE_P = 0.60
TOP_N = 40

GENERIC_TECH = {
    "api", "bb", "cli", "cljs", "clj", "config", "css", "csv", "db",
    "dev", "docker", "edn", "emacs", "git", "github", "html", "http",
    "https", "ide", "java", "js", "json", "jvm", "linux", "localhost",
    "markdown", "md", "npm", "oauth", "pdf", "py", "python", "repl",
    "repo", "repos", "rest", "script", "sql", "ssh", "svg", "tcp",
    "test", "tests", "tsv", "txt", "ui", "url", "urls", "uuid", "venv",
    "web", "xml", "yaml",
}

CONTRACTION_CRUMBS = {
    "aren", "can", "couldn", "didn", "doesn", "don", "hadn", "hasn",
    "haven", "isn", "shouldn", "wasn", "weren", "won", "wouldn",
}

EIGHTFOLD_PHASES = [
    "identify", "map", "derive", "argue", "verify", "instantiate",
    "observe", "dissolve",
]

MISSION_LIFECYCLE = [
    "mission", "missions", "lifecycle", "phase", "phases", "xenotype",
    "exotype", "genotype", "phenotype", "invariant", "invariants",
    "evidence", "scope", "pattern", "patterns", "agent", "agents",
    "futon", "arxana", "hypergraph", "hyperedges", "capability",
    "capabilities", "sorry", "sorries", "proof", "state", "proof-state",
    "handoff", "kernel", "prior", "lexicon", "lexicons",
]

GLOSS_STUBS = {
    "aif": "",
    "arxana": "",
    "exotype": "",
    "flexiarg": "",
    "futon": "",
    "hyperedge": "",
    "hyperedges": "",
    "hypergraph": "",
    "invariant": "",
    "invariants": "",
    "lifecycle": "",
    "mission": "",
    "pattern": "",
    "scope": "",
    "xenotype": "",
}

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that keeps contractions out of the crumb tail."""
    out: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if "'" in token:
            left, right = token.split("'", 1)
            if right in {"s", "t", "re", "ve", "ll", "d", "m"}:
                token = left
        if len(token) > 2 and token not in CONTRACTION_CRUMBS:
            out.append(token)
    return out


def load_common_words() -> set[str]:
    if not WORDS.exists():
        return set()
    return {
        w.strip().lower()
        for w in WORDS.read_text(encoding="utf-8", errors="ignore").splitlines()
        if w.strip().isalpha()
    }


def add_seed(seed: dict[str, set[str]], term: str, source: str) -> None:
    term = term.lower().strip().strip(":`.,;()[]{}")
    if not term:
        return
    term = term.replace("_", "-")
    if not re.search(r"[a-z]", term):
        return
    seed[term].add(source)


def extract_star_map_seeds(seed: dict[str, set[str]]) -> set[str]:
    text = STAR_MAP.read_text(encoding="utf-8", errors="ignore")
    cap_block = text.split(":capabilities", 1)[1].split(":missions", 1)[0]
    capabilities = set(re.findall(r"(?m)^\s*:([a-z][a-z0-9-]+)\s*\{", cap_block))
    for cap in capabilities:
        add_seed(seed, cap, "star-map-capability")
    for edge_term in re.findall(r":type\s+:([a-z][a-z0-9-]+)", text):
        add_seed(seed, edge_term, "star-map-edge-type")
    return capabilities


def extract_pudding_registry_seeds(seed: dict[str, set[str]]) -> None:
    text = PUDDING_REGISTRY.read_text(encoding="utf-8", errors="ignore")
    for ident in re.findall(r":id\s+:([A-Za-z0-9_.-]+)", text):
        add_seed(seed, ident, "pudding-sorry-id")
    for kind in re.findall(r":kind\s+:([a-z][a-z0-9-]+)", text):
        add_seed(seed, kind, "pudding-kind")
    for status in re.findall(r":status\s+:([a-z][a-z0-9-]+)", text):
        add_seed(seed, status, "pudding-status")


def extract_pudding_doc_seeds(seed: dict[str, set[str]]) -> None:
    text = PUDDING_DOC.read_text(encoding="utf-8", errors="ignore")
    section = text.split("### 8.2.1", 1)[-1].split("### 9.1", 1)[0]
    for term in re.findall(r"`:?([A-Za-z][A-Za-z0-9_-]+)`", section):
        add_seed(seed, term, "pudding-doc-8.2")


def extract_flexiarg_seeds(seed: dict[str, set[str]]) -> None:
    for path in FLEXIARG_ROOT.glob("**/*.flexiarg"):
        add_seed(seed, path.stem, "flexiarg-basename")


def build_seed_vocab() -> tuple[dict[str, set[str]], set[str]]:
    seed: dict[str, set[str]] = defaultdict(set)
    star_map_caps = extract_star_map_seeds(seed)
    extract_pudding_registry_seeds(seed)
    extract_pudding_doc_seeds(seed)
    extract_flexiarg_seeds(seed)
    for term in EIGHTFOLD_PHASES:
        add_seed(seed, term, "eightfold-phase")
    for term in MISSION_LIFECYCLE:
        add_seed(seed, term, "mission-lifecycle")
    return seed, star_map_caps


def raw_self_representing(prior: dict, common: set[str], n: int = TOP_N) -> list[dict]:
    n_docs = prior["n_docs"]
    ranked = sorted(prior["unigram_df"].items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"term": t, "df": c, "p": c / n_docs}
        for t, c in ranked
        if t not in common
    ][:n]


def is_hard_drop(term: str) -> bool:
    return term in GENERIC_TECH or term in CONTRACTION_CRUMBS


def sip_score(df: int, n_docs: int, seed_vouched: bool, dict_word: bool = False) -> float:
    p = df / n_docs
    boilerplate_discount = 0.35 if p >= BOILERPLATE_P and seed_vouched and not dict_word else 1.0
    reclaim_bonus = 40 if seed_vouched and dict_word else 0
    return df * max(0.0, 1.0 - p) * boilerplate_discount + reclaim_bonus


def build_kernel(prior: dict, common: set[str], seed: dict[str, set[str]]) -> dict:
    n_docs = prior["n_docs"]
    uni = prior["unigram_df"]
    terms: dict[str, dict] = {}
    dropped: dict[str, list[str]] = defaultdict(list)

    candidates = set(seed)
    for term, df in uni.items():
        if df >= HIGH_DF_MIN and term not in common:
            candidates.add(term)

    for term in sorted(candidates):
        df = int(uni.get(term, 0))
        p = df / n_docs if n_docs else 0.0
        seed_sources = sorted(seed.get(term, set()))
        seed_vouched = bool(seed_sources)
        dict_word = term in common

        if is_hard_drop(term):
            dropped["generic-tech-or-tokenizer-crumb"].append(term)
            continue
        if dict_word and not seed_vouched:
            dropped["dictionary-word-not-seeded"].append(term)
            continue
        if p >= BOILERPLATE_P and not seed_vouched:
            dropped["boilerplate-not-seeded"].append(term)
            continue
        if df and df < HIGH_DF_MIN and not seed_vouched:
            dropped["below-high-df-and-unseeded"].append(term)
            continue

        source_kind = "seed" if seed_vouched else "prior-high-df"
        terms[term] = {
            "term": term,
            "df": df,
            "p": p,
            "score": sip_score(df, n_docs, seed_vouched, dict_word) if df else 0.0,
            "dict_word": dict_word,
            "seed_sources": seed_sources,
            "source_kind": source_kind,
        }

    return {
        "metadata": {
            "mission": "M-web-arxana-missions",
            "pass": "Layer-3 Pass 2 mission-domain NER kernel",
            "prior": os.fspath(PRIOR),
            "n_docs": n_docs,
            "high_df_min": HIGH_DF_MIN,
            "boilerplate_p": BOILERPLATE_P,
            "generic_tech_terms": sorted(GENERIC_TECH),
            "contraction_crumbs": sorted(CONTRACTION_CRUMBS),
        },
        "terms": sorted(terms.values(), key=lambda r: (-r["score"], r["term"])),
        "dropped": {k: sorted(v) for k, v in sorted(dropped.items())},
    }


def self_representing_lexicon(kernel: dict, limit: int | None = None) -> list[dict]:
    rows = [
        row for row in kernel["terms"]
        if row["df"] > 0 and not is_hard_drop(row["term"])
    ]
    rows = sorted(rows, key=lambda r: (-r["score"], -r["df"], r["term"]))
    if limit is not None:
        rows = rows[:limit]
    return rows


def projection_lexicon(
    prior: dict, common: set[str], kernel_terms: set[str], top_self_rep: Iterable[dict]
) -> dict:
    n_docs = prior["n_docs"]
    generic_drop = [
        {"term": t, "df": c, "p": c / n_docs}
        for t, c in sorted(prior["unigram_df"].items(), key=lambda kv: (-kv[1], kv[0]))
        if t in common and t not in kernel_terms
    ]
    gloss_terms = [row["term"] for row in list(top_self_rep)[:TOP_N]]
    glosses = {term: GLOSS_STUBS.get(term, "") for term in gloss_terms}
    return {
        "metadata": {
            "kind": "projection-into-English",
            "source": os.fspath(PRIOR),
            "note": "Dictionary words not in the mission kernel, plus hand-fillable jargon gloss stubs.",
        },
        "generic_drop": generic_drop,
        "jargon_to_english_gloss_stub": glosses,
    }


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_ranked(title: str, rows: list[dict]) -> None:
    print(f"\n--- {title} ---")
    for row in rows[:TOP_N]:
        print(f"  P={row['p']:4.2f}  df={row['df']:4d}  {row['term']}")


def main() -> None:
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    common = load_common_words()
    seed, star_map_caps = build_seed_vocab()

    before = raw_self_representing(prior, common)
    kernel = build_kernel(prior, common, seed)
    self_rep = self_representing_lexicon(kernel)
    kernel_terms = {row["term"] for row in kernel["terms"]}
    projection = projection_lexicon(prior, common, kernel_terms, self_rep)

    write_json(KERNEL_OUT, kernel)
    write_json(
        SELF_REP_OUT,
        {
            "metadata": {
                "kind": "self-representing",
                "source": os.fspath(KERNEL_OUT),
                "ranking": "SIP-style df * (1 - df/n_docs), with a small seeded-dictionary reclaim bonus and seeded boilerplate discounted not erased.",
            },
            "terms": self_rep,
        },
    )
    write_json(PROJECTION_OUT, projection)

    print(f"kernel_terms={len(kernel_terms)} -> {KERNEL_OUT}")
    print(f"self_representing_terms={len(self_rep)} -> {SELF_REP_OUT}")
    print(f"projection_generic_drop={len(projection['generic_drop'])} -> {PROJECTION_OUT}")
    print_ranked("BEFORE Pass-1 raw self-representing top-40", before)
    print_ranked("AFTER kernel-cleaned self-representing top-40", self_rep)

    aligned = sorted(cap for cap in star_map_caps if cap in kernel_terms)
    print(
        "\n--- star-map gold alignment ---\n"
        f"  capability names in kernel: {len(aligned)}/{len(star_map_caps)}\n"
        f"  {', '.join(aligned)}"
    )


if __name__ == "__main__":
    main()
