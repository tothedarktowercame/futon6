from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_mission_kernel.py"
SPEC = importlib.util.spec_from_file_location("build_mission_kernel", SCRIPT)
assert SPEC and SPEC.loader
mission_kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mission_kernel)


def test_tokenizer_drops_contraction_crumbs() -> None:
    toks = mission_kernel.tokenize("doesn't isn't didn't can't won't evidence")
    assert "doesn" not in toks
    assert "isn" not in toks
    assert "didn" not in toks
    assert "evidence" in toks


def test_kernel_reclaims_seeded_dictionary_terms_and_drops_generic_tech() -> None:
    prior = {
        "n_docs": 100,
        "unigram_df": {
            "futon": 80,
            "pattern": 70,
            "evidence": 65,
            "scope": 64,
            "agent": 50,
            "mission": 95,
            "clj": 60,
            "edn": 50,
            "doesn": 40,
            "arxana": 30,
            "ordinary": 35,
        },
        "bigram_df": {},
    }
    common = {"futon", "pattern", "evidence", "scope", "agent", "mission", "ordinary"}
    seed = defaultdict(set)
    for term in ["futon", "pattern", "evidence", "scope", "agent", "mission"]:
        seed[term].add("test-seed")

    kernel = mission_kernel.build_kernel(prior, common, seed)
    terms = {row["term"] for row in kernel["terms"]}

    assert {"futon", "pattern", "evidence", "scope", "agent", "mission", "arxana"} <= terms
    assert "clj" not in terms
    assert "edn" not in terms
    assert "doesn" not in terms
    assert "ordinary" not in terms
