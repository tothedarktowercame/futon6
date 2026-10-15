"""preflight warns when the concept vocabulary does not belong to the corpus.

A warning (ok=None) is printed and does not refuse the run; a lexicon paper's
defined terms count as vocabulary.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import preflight as pf  # noqa: E402

DOC = "\\documentclass{article}\\begin{document}%s\\end{document}\n"


def corpus(tmp_path, body):
    ep = tmp_path / "ep"
    ep.mkdir()
    for pid in ("p1", "p2"):
        (ep / f"{pid}.tex").write_text(DOC % body)
    ids = tmp_path / "ids.txt"
    ids.write_text("p1\np2\n")
    enc = tmp_path / "enc.json"
    enc.write_text(json.dumps({"entries": [{"concept": c} for c in
                                           ("monoidal category", "natural transformation",
                                            "adjoint functor", "full subcategory")]}))
    return ep, ids, enc


def run_check(ep, ids, enc, env=()):
    pf.R.clear()
    with patch.dict(os.environ, {"FUTON6_EPRINTS": str(ep), **dict(env)}), \
            patch.object(pf.config, "concept_encyclopedia", return_value=enc):
        pf.check_vocabulary(str(ids))
    return pf.R[-1]


def test_mismatched_vocabulary_warns_without_failing(tmp_path):
    ep, ids, enc = corpus(tmp_path, "Let f be a compact operator on a Banach space.")
    _name, ok, _detail, remedy = run_check(ep, ids, enc)
    assert ok is None
    assert "does not match" in remedy


def test_matching_vocabulary_passes(tmp_path):
    ep, ids, enc = corpus(tmp_path, "A monoidal category with an adjoint functor "
                                    "and a natural transformation.")
    assert run_check(ep, ids, enc)[1] is True


def test_lexicon_definitions_count_as_vocabulary(tmp_path):
    ep, ids, enc = corpus(tmp_path, "A compact operator on a Banach space has a spectral radius.")
    (ep / "lex.tex").write_text(DOC % (
        "\\begin{definition}\\emph{compact operator}: x\\end{definition}"
        "\\begin{definition}\\emph{banach space}: y\\end{definition}"
        "\\begin{definition}\\emph{spectral radius}: z\\end{definition}"))
    lex = tmp_path / "lex.ids"
    lex.write_text("lex\n")
    _name, ok, detail, _remedy = run_check(ep, ids, enc, {"FUTON6_WARP_LEXICON_IDS": str(lex)})
    assert ok is True
    assert "lexicon lex (3)" in detail


def test_tapestry_warns_when_the_corpus_cites_nothing(tmp_path):
    ep, ids, _enc = corpus(tmp_path, "A solution with no references at all.")
    pf.R.clear()
    with patch.dict(os.environ, {"FUTON6_EPRINTS": str(ep), "FUTON6_ENABLE_TAPESTRY": "1"}):
        pf.check_citations(str(ids))
    _name, ok, detail, remedy = pf.R[-1]
    assert ok is None and detail.startswith("0/2") and "--no-tapestry" in remedy


def test_tapestry_check_passes_with_citations_and_is_silent_when_off(tmp_path):
    ep, ids, _enc = corpus(tmp_path, "As shown in \\cite{Mac71}.")
    pf.R.clear()
    with patch.dict(os.environ, {"FUTON6_EPRINTS": str(ep), "FUTON6_ENABLE_TAPESTRY": "1"}):
        pf.check_citations(str(ids))
    assert pf.R[-1][1] is True
    pf.R.clear()
    with patch.dict(os.environ, {"FUTON6_EPRINTS": str(ep), "FUTON6_ENABLE_TAPESTRY": "0"}):
        pf.check_citations(str(ids))
    assert pf.R == []
