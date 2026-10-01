from pathlib import Path

from scripts import cite_resolve


def test_plain_ids_preserve_safe_and_canonical_forms(tmp_path: Path) -> None:
    path = tmp_path / "corpus.ids.txt"
    path.write_text("2401.12345\nmath__9901001\n")

    ids = cite_resolve.load_plain_ids(path)

    assert ids.canonical == {"2401.12345", "math/9901001"}
    assert ids.safe == {"2401.12345", "math__9901001"}
    assert ids.canonical_to_safe["math/9901001"] == "math__9901001"


def test_ids_manifest_drives_the_complete_paper_list(tmp_path: Path) -> None:
    path = tmp_path / "corpus.ids.txt"
    path.write_text("# frozen\np1\np2\n")
    args = type("Args", (), {"paper": None, "ids": path, "gh200": None, "sample_size": 1})()

    assert cite_resolve.iter_papers(args) == ["p1", "p2"]
