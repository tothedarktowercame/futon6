"""Lean retrieval: index proved declarations, answer informal claims by meaning."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lean_retrieval as R  # noqa: E402


def tree(tmp_path):
    (tmp_path / "problems" / "t01A04" / "lean").mkdir(parents=True)
    (tmp_path / "problems" / "t01A04" / "lean" / "Main.lean").write_text(
        "/-- The torus map is a covering map. -/\n"
        "lemma apm_t01a04_torusCoverMap_isCoveringMap : True := trivial\n"
        "lemma apm_t01a04_unfinished : True := by sorry\n")
    (tmp_path / "ConstructionTargets").mkdir()
    (tmp_path / "ConstructionTargets" / "Surfaces.lean").write_text(
        "/-- Euler characteristic of a connected sum. -/\n"
        "theorem eulerCharacteristic_connectedSum (a b : ℕ) : True := trivial\n"
        "def helper : ℕ := 0\n")
    return tmp_path


def test_index_keeps_proved_theorems_with_their_owner(tmp_path):
    idx = R.build_index(tree(tmp_path), ["problems/*/lean/*.lean", "ConstructionTargets/*.lean"])
    names = {d["name"]: d for d in idx["declarations"]}
    assert set(names) == {"apm_t01a04_torusCoverMap_isCoveringMap", "eulerCharacteristic_connectedSum"}
    assert names["apm_t01a04_torusCoverMap_isCoveringMap"]["paper"] == "t01A04"
    assert names["eulerCharacteristic_connectedSum"]["paper"] == "LIB:Surfaces"
    assert names["eulerCharacteristic_connectedSum"]["file"] == "ConstructionTargets/Surfaces.lean"


def test_informal_queries_find_the_right_declaration(tmp_path):
    r = R.Retriever(R.build_index(tree(tmp_path), ["problems/*/lean/*.lean", "ConstructionTargets/*.lean"]))
    assert r.query("χ of a connected sum of surfaces")[0]["decl"] == "eulerCharacteristic_connectedSum"
    assert r.query("the map on the torus is a covering")[0]["decl"] == "apm_t01a04_torusCoverMap_isCoveringMap"
    assert all(h["owner"] != "t01A04" for h in r.query("a covering of the torus", exclude="t01A04"))


def test_command_line_round_trip(tmp_path):
    root = tree(tmp_path)
    out = tmp_path / "idx.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/lean_retrieval.py"), "index", "--root", str(root),
                    "--glob", "ConstructionTargets/*.lean", "--out", str(out)], check=True, capture_output=True)
    res = subprocess.run([sys.executable, str(ROOT / "scripts/lean_retrieval.py"), "query", "--index", str(out),
                          "--json", "Euler characteristic connected sum"], check=True, capture_output=True, text=True)
    assert json.loads(res.stdout)[0]["hits"][0]["decl"] == "eulerCharacteristic_connectedSum"


def test_greek_variables_and_primes_are_not_words(tmp_path):
    import align_informal_lean as A
    assert A.tokens("Let π:Σ→Σ′ be a regular covering") == ["regular", "cover"]
    assert "chi" in A.tokens(r"$\chi(\Sigma) = 5\chi(\Sigma')$")       # χ is the Euler characteristic
    assert "product" in A.tokens(r"$T^2 = S^1\times S^1$")


def test_an_informal_solution_is_read_sentence_by_sentence(tmp_path):
    md = ("<!-- source: agent -->\n# Title\n\nEuler characteristic multiplies under a finite cover. "
          "So the torus covers itself. Then $x = y. z$ holds here.\n\n```lean\ntheorem foo : True := trivial\n```\n")
    got = R.sentences(md)
    assert got == ["Euler characteristic multiplies under a finite cover.", "So the torus covers itself."]
    # a full stop inside a formula does not end the sentence
    assert R.sentences("The covering map $p. Q$ is regular on every torus component.") == \
        ["The covering map $p. Q$ is regular on every torus component."]
