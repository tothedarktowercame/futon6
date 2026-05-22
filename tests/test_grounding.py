"""Tests for the shared `futon6.grounding` orchestration module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6 import grounding


def test_load_learned_vocab_returns_common_slot(tmp_path):
    path = tmp_path / "vocab.json"
    payload = {
        "by_symbol": {"\\RR": []},
        "common": [
            {"symbol": "\\RR", "body": "{\\mathbb R}", "canon": "R", "support": 2},
            {"symbol": "\\ZZ", "body": "{\\mathbb Z}", "canon": "Z", "support": 3},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    common = grounding.load_learned_vocab(path)
    assert len(common) == 2
    assert {entry["symbol"] for entry in common} == {"\\RR", "\\ZZ"}


def test_load_learned_vocab_missing_file_returns_empty(tmp_path):
    assert grounding.load_learned_vocab(tmp_path / "nope.json") == []


def test_load_learned_vocab_malformed_returns_empty(tmp_path):
    path = tmp_path / "garbage.json"
    path.write_text("not valid json", encoding="utf-8")
    assert grounding.load_learned_vocab(path) == []


def test_make_kernel_phrase_lookup_resolves_known_phrase():
    singles = {"category": ("category", "Category")}
    multi_index = {"abelian": [("abelian group", "abelian group", "AbelianGroup")]}
    lookup = grounding.make_kernel_phrase_lookup(singles, multi_index)
    assert lookup("category") == "Category"
    assert lookup("abelian group") == "AbelianGroup"
    assert lookup("frobnicator") is None


def test_walk_math_atoms_yields_letters_inside_dollar():
    atoms = list(grounding.walk_math_atoms("$AB$"))
    texts = [a[0] for a in atoms]
    assert texts == ["A", "B"]


def test_walk_math_atoms_yields_macro_tokens():
    atoms = list(grounding.walk_math_atoms(r"$\Hom(A, B)$"))
    macros = [a[0] for a in atoms if a[0].startswith("\\")]
    assert r"\Hom" in macros
