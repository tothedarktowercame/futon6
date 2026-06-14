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
    nnexus = tmp_path / "nnexus.sql"
    nnexus.write_text(
        "INSERT INTO \"concepts\" VALUES(1,'external','concept','11Axx','msc','Wikipedia','wikipedia.org/external_concept',7);\n"
        "INSERT INTO \"concepts\" VALUES(2,'external','concept','11Bxx','msc','Mathworld','mathworld.wolfram.com/ExternalConcept.html',8);\n"
        "INSERT INTO \"concepts\" VALUES(3,'gel''fand','triple','81P10','msc','Planetmath','planetmath.org/gelfandtriple',9);\n",
        encoding="utf-8",
    )
    out = tmp_path / "index.json"
    doc = bg.build_index(["external concept"], out, pages.parent.parent.parent.parent.parent, wiring, prior, nnexus)
    assert doc["nlab-name-count"] == 1
    assert doc["nnexus-row-count"] == 3
    assert doc["nnexus-domain-counts"] == {"Mathworld": 1, "Planetmath": 1, "Wikipedia": 1}
    assert doc["ct-prior-count"] == 1
    assert bg.resolve(doc, "Kirillov models")["resolution-kind"] == "definition-site"
    external = bg.resolve(doc, "external concept")
    assert external["resolution-kind"] == "nnexus"
    assert external["domains"] == ["Mathworld", "Wikipedia"]
    assert bg.resolve(doc, "gel'fand triple")["resolution-kind"] == "nnexus"
    assert bg.resolve(doc, "orphanology") is None


def test_parse_sql_values_handles_commas_and_doubled_quotes():
    row = bg.parse_sql_values("2878,'gel''fand','triple, weak','81P10','msc','Planetmath','planetmath.org/gelfandtriple',649")
    assert row == [2878, "gel'fand", "triple, weak", "81P10", "msc", "Planetmath", "planetmath.org/gelfandtriple", 649]
