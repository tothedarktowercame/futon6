import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import background_corpus_index as bg


def test_normalize_variants_and_resolution_strength():
    idx = {}
    bg.add_resolution(idx, "Kirillov model", "ct-prior", "ct-term-prior:kirillov model")
    bg.add_resolution(idx, "Kirillov model", "page", "nlab-1")
    bg.add_resolution(idx, "Kirillov model", "definition-site", "nlab-2")
    hit = bg.resolve({"terms": idx}, "kirillov models")
    assert hit["resolution-kind"] == "definition-site"
    assert hit["target"] == "nlab-2"


def test_build_index_with_synthetic_sources(tmp_path):
    pages = tmp_path / "pages" / "1" / "2" / "3" / "4" / "99"
    pages.mkdir(parents=True)
    (pages / "name").write_text("Kirillov model", encoding="utf-8")
    wiring = tmp_path / "pages.json"
    wiring.write_text('[{"page_id":"nlab-99","page_name":"Kirillov model","stats":{"env_types":{"env/definition":1}}}]')
    prior = tmp_path / "prior.json"
    prior.write_text('{"unigram_df":{"orphanology":3},"bigram_df":{"external concept":2}}')
    out = tmp_path / "index.json"
    doc = bg.build_index(["external concept"], out, pages.parent.parent.parent.parent.parent, wiring, prior)
    assert doc["nlab-name-count"] == 1
    assert doc["ct-prior-count"] == 1
    assert bg.resolve(doc, "Kirillov models")["resolution-kind"] == "definition-site"
    assert bg.resolve(doc, "external concept")["resolution-kind"] == "ct-prior"
    assert bg.resolve(doc, "orphanology") is None
