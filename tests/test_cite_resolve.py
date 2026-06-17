import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cite_resolve  # noqa: E402
import cite_resolve_check  # noqa: E402


def test_cite_mark_split_and_bib_markers():
    mark = {"fields": [["cite", "foo, bar"], ["via", "\\cite"]]}
    assert cite_resolve.cite_keys(mark) == ["foo", "bar"]
    by_key, markers = cite_resolve.bib_maps(
        {"bibitems": [{"key": "foo", "raw": "Foo"}, {"key": "bar", "raw": "Bar"}]}
    )
    assert set(by_key) == {"foo", "bar"}
    assert markers == {"foo": "[1]", "bar": "[2]"}


def test_checker_accepts_resolved_and_hole_records(tmp_path):
    corpus = tmp_path / "ids.jsonl"
    corpus.write_text(json.dumps({"id": "2306.09745", "safe_id": "2306.09745"}) + "\n")
    out = tmp_path / "paper.cite-resolution.json"
    out.write_text(json.dumps({
        "schema": cite_resolve.SCHEMA,
        "paper-id": "sample",
        "records": [
            {
                "cite/marker": "[1]",
                "cite/key": "x",
                "char-anchor": [10, 18],
                "confidence": 1.0,
                "method": "arxiv-id",
                "resolved-arxiv-id": "2306.09745",
                "resolved-corpus-id": "2306.09745",
                "hole": None,
            },
            {
                "cite/marker": "[?]",
                "cite/key": "y",
                "char-anchor": [20, 28],
                "confidence": 0.0,
                "method": "hole",
                "resolved-arxiv-id": None,
                "resolved-corpus-id": None,
                "hole": {"kind": "unresolved-citation", "reason": "no-corpus-match"},
            },
        ],
        "stats": {"total": 2, "resolved": 1, "holes": 1, "resolution-rate": 0.5},
    }))
    canonical, safe = cite_resolve_check.load_corpus_ids(corpus)
    assert cite_resolve_check.check_file(out, canonical, safe) == []


def test_checker_rejects_resolved_id_outside_corpus(tmp_path):
    corpus = tmp_path / "ids.jsonl"
    corpus.write_text(json.dumps({"id": "2306.09745", "safe_id": "2306.09745"}) + "\n")
    out = tmp_path / "bad.cite-resolution.json"
    out.write_text(json.dumps({
        "schema": cite_resolve.SCHEMA,
        "paper-id": "sample",
        "records": [
            {
                "cite/marker": "[1]",
                "cite/key": "x",
                "char-anchor": [10, 18],
                "confidence": 1.0,
                "method": "arxiv-id",
                "resolved-arxiv-id": "9999.99999",
                "resolved-corpus-id": "9999.99999",
                "hole": None,
            }
        ],
        "stats": {"total": 1, "resolved": 1, "holes": 0, "resolution-rate": 1.0},
    }))
    canonical, safe = cite_resolve_check.load_corpus_ids(corpus)
    errors = cite_resolve_check.check_file(out, canonical, safe)
    assert any("resolved-arxiv-id not in corpus" in error for error in errors)
