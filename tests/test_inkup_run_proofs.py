import json
from pathlib import Path

from futon6.inkup_run_proofs import measure_text, opens_with, proof_records, slice_candidate, words


def candidate(**changes):
    result = {
        "paper-id": "paper",
        "proof-id": "paper__p0",
        "passage-id": "paper:proof0:L10-14",
        "window-lines": [10, 14],
        "proof-lines": [12, 14],
        "proved": {"kind": "theorem"},
        "source-window": "statement one\nstatement two\nProof starts.\nMiddle.\nEnd.\n",
    }
    result.update(changes)
    return result


def write_run(tmp_path: Path, rows: list[dict], *, clean=True):
    candidates = tmp_path / "artifacts/candidates"
    candidates.mkdir(parents=True)
    (tmp_path / "artifacts/clean").mkdir()
    (tmp_path / "artifacts/graphs").mkdir()
    for row in rows:
        proof_id = row["proof-id"]
        (candidates / f"{proof_id}.candidate.json").write_text(json.dumps(row))
        if clean:
            (tmp_path / f"artifacts/clean/{proof_id}.clean.edn").write_text("{}")
            (tmp_path / f"artifacts/graphs/{proof_id}.edn").write_text("{}")
    (tmp_path / "anchor-faithfulness.txt").write_text("")
    (tmp_path / "run-manifest.json").write_text(json.dumps({"papers": ["paper", "empty"]}))


def test_slice_uses_proof_lines():
    text, bad = slice_candidate(candidate())
    assert not bad
    assert text == "Proof starts.\nMiddle.\nEnd.\n"


def test_unsliceable_is_retained_and_flagged(tmp_path):
    row = candidate(**{"proof-lines": [9, 14]})
    write_run(tmp_path, [row])
    result = proof_records(tmp_path)
    assert len(result) == 1
    assert result[0]["flags"]["unsliceable"] is True
    assert result[0]["usable"] is False


def test_word_rule_inline_display_and_list():
    text = "Take $x + y$. Then\\n\\[x=y.\\]\\n\\begin{enumerate}\\item One item.\\end{enumerate}"
    assert words(text) == len(text.split())
    measured = measure_text(text)
    assert measured["display_count"] == 1
    assert measured["has_list"] is True


def test_opens_with_classes():
    assert opens_with("Let x be fixed.") == "let/suppose/assume"
    assert opens_with(r"By \cite{X}, the result follows.") == "by-cite/ref/named-result"
    assert opens_with("We prove the claim.") == "we-show/prove/claim/it-suffices"
    assert opens_with("Clearly this is zero.") == "clearly/obvious/immediate/follows-from"
    assert opens_with("The map is injective.") == "other"


def test_missing_clean_or_graph_is_flagged(tmp_path):
    write_run(tmp_path, [candidate()], clean=False)
    result = proof_records(tmp_path)[0]
    assert result["flags"]["no_clean_proof"] is True
