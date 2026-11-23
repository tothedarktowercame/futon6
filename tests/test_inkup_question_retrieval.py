"""Tests for futon6.inkup_question_retrieval on a small synthetic store.

The embedder is injected (tests/fixtures/inkup-qa-stores/build_fixture.py); the BGE
model is never loaded.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from futon6 import inkup_question_retrieval as retrieval

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "inkup-qa-stores"
CLI = REPO / "scripts" / "inkup-question-retrieval.py"
STORES = {"math": FIXTURES / "math", "mo": FIXTURES / "mathoverflow"}

FEATURED = {"math": ("1001", "1002", "1003", "1004", "1005", "1006", "1007"),
            "mo": ("2001", "2002")}
PROVENANCE_FIELDS = ("entity_id", "site", "thread_id", "title", "passage", "tags", "created",
                     "method", "rank", "score", "retrieval_reason")


def _fake_embedder():
    spec = importlib.util.spec_from_file_location("inkup_fixture_embedder",
                                                  FIXTURES / "build_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_embedder()


def _entities(site: str) -> list[dict]:
    return list(retrieval.iter_json_array(STORES[site] / "entities.json"))


@pytest.fixture
def embedder():
    return _fake_embedder()


@pytest.fixture
def query(tmp_path):
    path = tmp_path / "query.json"
    path.write_text(json.dumps([
        {"term": "Schwarz lemma", "source": "Blueprint a96J06", "category": "named-method"},
        {"term": "polynomial", "source": "vocabulary note", "category": "object"},
        {"term": "quasiconformal", "source": "vocabulary note"},
    ]), encoding="utf-8")
    return path


@pytest.fixture
def run(tmp_path, query, embedder):
    outdir = retrieval.run_retrieval(query, list(STORES.items()), tmp_path / "out", embedder)
    candidates = [json.loads(line) for line in
                  (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    return outdir, candidates, json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))


def _ids(candidates, method, site):
    return [c["thread_id"] for c in candidates if c["method"] == method and c["site"] == site]


def _bodies(site):
    return {e["entity/id"]: e for e in _entities(site)}


# 1. ranking order per method -------------------------------------------------

def test_lexical_ranking_order_is_pinned(run):
    _, candidates, _ = run
    assert _ids(candidates, "lexical", "math") == ["1003", "1001", "1002"]
    assert _ids(candidates, "lexical", "mo") == ["2001"]
    by_row = {(c["site"], c["method"], c["thread_id"]): c for c in candidates}
    assert by_row["math", "lexical", "1003"]["retrieval_reason"] == {"matched_terms": ["polynomial"]}
    assert by_row["math", "lexical", "1001"]["retrieval_reason"] == {"matched_terms": ["schwarz lemma"]}


def test_lexical_scores_are_sorted_and_ranks_contiguous(run):
    _, candidates, _ = run
    for site in STORES:
        for method in ("lexical", "embedding"):
            rows = [c for c in candidates if c["site"] == site and c["method"] == method]
            scores = [c["score"] for c in rows]
            assert scores == sorted(scores, reverse=True)
            assert [c["rank"] for c in rows] == list(range(1, len(rows) + 1))


def test_embedding_ranking_order_is_pinned(run):
    _, candidates, _ = run
    assert _ids(candidates, "embedding", "math")[:3] == ["1001", "1003", "1006"]
    assert _ids(candidates, "embedding", "mo")[:2] == ["2001", "2002"]


# 2. provenance fields ---------------------------------------------------------

def test_every_candidate_carries_all_provenance_fields(run):
    _, candidates, _ = run
    assert candidates
    for candidate in candidates:
        for field in PROVENANCE_FIELDS:
            assert field in candidate, (candidate.get("entity_id"), field)
        assert candidate["site"] in STORES
        assert candidate["thread_id"].isdigit()
        assert candidate["entity_id"].endswith(candidate["thread_id"])
        assert isinstance(candidate["tags"], list) and candidate["tags"]
        assert isinstance(candidate["created"], int)
        assert isinstance(candidate["score"], float)
        assert candidate["method"] in ("lexical", "embedding")


def test_entity_matches_the_store_row(run):
    for site, store in STORES.items():
        bodies = _bodies(site)
        site_rows = [c for c in run[1] if c["site"] == site]
        assert site_rows
        for candidate in site_rows:
            entity = bodies[candidate["entity_id"]]
            assert candidate["title"] == entity["title"]
            assert candidate["tags"] == entity["tags"]
            assert candidate["created"] == entity["created"]


def test_every_embedding_candidate_says_it_matched_the_question_answer_vector(run):
    _, candidates, _ = run
    dense = [c for c in candidates if c["method"] == "embedding"]
    assert dense
    for candidate in dense:
        assert candidate["answer_text_may_drive_match"] is True
        assert "answer" in candidate["matched_vector_scope"]
        assert candidate["retrieval_reason"]["query_text"]
        assert candidate["retrieval_reason"]["match_scope"] == candidate["matched_vector_scope"]
    for candidate in (c for c in candidates if c["method"] == "lexical"):
        assert "answer_text_may_drive_match" not in candidate
        assert candidate["retrieval_reason"]["matched_terms"]


def test_no_relevance_or_question_function_labels(run):
    _, candidates, manifest = run
    forbidden = {"relevant", "relevance", "label", "labels", "question_function", "function"}
    for candidate in candidates:
        assert not forbidden & set(candidate)
        assert not forbidden & set(candidate["retrieval_reason"])
    assert "none" in manifest["labels"]


# 3. passage is a verbatim substring of the question body ----------------------

def test_passage_is_a_verbatim_substring_of_the_question_body(run):
    _, candidates, _ = run
    for site in STORES:
        bodies = _bodies(site)
        for candidate in (c for c in candidates if c["site"] == site):
            question = bodies[candidate["entity_id"]]["question-body"]
            answer = bodies[candidate["entity_id"]]["answer-body"]
            assert candidate["passage"]
            assert candidate["passage"] in question
            assert candidate["passage"] not in answer


def test_lexical_passage_contains_a_matched_term(run):
    _, candidates, _ = run
    for candidate in (c for c in candidates if c["method"] == "lexical"):
        assert any(term.lower() in candidate["passage"].lower()
                   for term in candidate["retrieval_reason"]["matched_terms"])


def test_embedding_passage_opens_the_question(run):
    _, candidates, _ = run
    for site in STORES:
        bodies = _bodies(site)
        for candidate in (c for c in candidates if c["method"] == "embedding" and c["site"] == site):
            question = bodies[candidate["entity_id"]]["question-body"]
            assert candidate["passage"] == question[:retrieval.WINDOW]


# 4. an answer-only term is not a lexical hit ---------------------------------

def test_a_term_that_appears_only_in_an_answer_gives_no_lexical_hit(embedder):
    with_answer_only = "quasiconformal"
    entities = _entities("math")
    assert any(with_answer_only in e["answer-body"] for e in entities)
    assert not any(with_answer_only in e["question-body"] or with_answer_only in e["title"]
                   for e in entities)
    result = retrieval.lexical_retrieve(STORES["math"] / "entities.json", [with_answer_only], 100)
    assert result["hits"] == []


# 5. refusal when no store is supplied -----------------------------------------

def test_no_store_is_refused_without_arguments_or_environment(monkeypatch):
    monkeypatch.delenv(retrieval.STORES_ENV, raising=False)
    with pytest.raises(ValueError, match="no store supplied"):
        retrieval.parse_stores(None)


def test_no_store_is_refused_for_a_malformed_argument():
    with pytest.raises(ValueError, match="SITE=PATH"):
        retrieval.parse_stores(["just-a-path"])


def test_no_store_is_refused_for_an_incomplete_store(tmp_path):
    (tmp_path / "entities.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="embeddings.npy"):
        retrieval.parse_stores([f"math={tmp_path}"])


def test_cli_exits_non_zero_with_a_clear_message_when_no_store_is_supplied(tmp_path, monkeypatch):
    monkeypatch.delenv(retrieval.STORES_ENV, raising=False)
    query = tmp_path / "query.json"
    query.write_text(json.dumps([{"term": "schwarz lemma", "source": "note"}]), encoding="utf-8")
    completed = subprocess.run([sys.executable, str(CLI), "--query", str(query),
                                "--output-dir", str(tmp_path / "out")],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    assert retrieval.STORES_ENV in completed.stderr
    assert "--store" in completed.stderr
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_environment_variable_supplies_stores(monkeypatch):
    setting = os.pathsep.join([f"math={STORES['math']}", f"mo={STORES['mo']}"])
    monkeypatch.setenv(retrieval.STORES_ENV, setting)
    assert retrieval.parse_stores(None) == [("math", STORES["math"]), ("mo", STORES["mo"])]


# 6. row-alignment check --------------------------------------------------------

def test_row_alignment_is_one_on_the_aligned_fixture(embedder):
    for store in STORES.values():
        report = retrieval.check_row_alignment(store / "entities.json",
                                               store / "embeddings.npy", embedder)
        assert report["own_row_nearest_fraction"] == 1.0
        assert report["trusted"] is True


@pytest.mark.parametrize("shift", [1, 7])
def test_row_alignment_is_below_one_when_rows_are_shuffled(tmp_path, embedder, shift):
    store = tmp_path / "shuffled"
    store.mkdir()
    (store / "entities.json").write_bytes((STORES["math"] / "entities.json").read_bytes())
    vectors = np.load(STORES["math"] / "embeddings.npy")
    np.save(store / "embeddings.npy", np.roll(vectors, shift, axis=0))
    report = retrieval.check_row_alignment(store / "entities.json",
                                           store / "embeddings.npy", embedder)
    assert report["own_row_nearest_fraction"] < 1.0
    assert report["trusted"] is False


def test_sample_count_is_never_below_two_hundred(embedder):
    report = retrieval.check_row_alignment(STORES["math"] / "entities.json",
                                           STORES["math"] / "embeddings.npy", embedder, 5)
    assert report["sample_count_requested"] >= 200
    assert report["sample_capped_by_store_size"] is True


def test_embedding_method_is_withheld_when_rows_do_not_align(tmp_path, query, embedder):
    store = tmp_path / "shuffled"
    store.mkdir()
    (store / "entities.json").write_bytes((STORES["math"] / "entities.json").read_bytes())
    vectors = np.load(STORES["math"] / "embeddings.npy")
    np.save(store / "embeddings.npy", np.roll(vectors, 5, axis=0))
    outdir = retrieval.run_retrieval(query, [("math", store)], tmp_path / "out", embedder)
    candidates = [json.loads(line) for line in
                  (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {c["method"] for c in candidates} == {"lexical"}
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["by_method"]["embedding"] == 0
    assert "embedding_skipped" in manifest["stores"][0]


def test_embedding_method_is_refused_when_entity_and_vector_counts_differ(tmp_path, query,
                                                                          embedder):
    """Vectors are tied to entities by row position; a dropped row would silently shift every
    embedding hit onto the wrong question, so the method must be withheld and say why."""
    store = tmp_path / "short"
    store.mkdir()
    entities = _entities("math")[:-1]
    (store / "entities.json").write_text(json.dumps(entities), encoding="utf-8")
    (store / "embeddings.npy").write_bytes((STORES["math"] / "embeddings.npy").read_bytes())
    outdir = retrieval.run_retrieval(query, [("math", store)], tmp_path / "out", embedder)
    candidates = [json.loads(line) for line in
                  (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {c["method"] for c in candidates} == {"lexical"}
    report = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))["stores"][0]
    assert "row count mismatch" in report["embedding_skipped"]
    assert report["row_count"] == {"entities": len(entities),
                                   "stored_vectors": len(entities) + 1,
                                   "entities_match_vectors": False}


def test_rare_term_outranks_common_term_with_equal_occurrences(embedder, tmp_path):
    """BM25 document frequency: the same number of occurrences of a rarer phrase scores higher."""
    rows = [
        {"entity/id": "se-t-1", "title": "common word", "question-body": "common word here"},
        {"entity/id": "se-t-2", "title": "common word again", "question-body": "and again"},
        {"entity/id": "se-t-3", "title": "common word twice", "question-body": "and again"},
        {"entity/id": "se-t-4", "title": "rare word", "question-body": "rare word here"},
    ]
    path = tmp_path / "entities.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    hits = retrieval.lexical_retrieve(path, ["common word", "rare word"], 10)["hits"]
    assert [hit[2]["entity/id"] for hit in hits][:2] == ["se-t-4", "se-t-1"]
    assert hits[0][3] == ["rare word"] and hits[1][3] == ["common word"]


# manifest ----------------------------------------------------------------------

def test_manifest_records_the_run(run):
    outdir, candidates, manifest = run
    assert outdir.name == "out"
    assert manifest["schema"] == "inkup.question-retrieval-run/v0"
    assert manifest["run_id"] and manifest["model"] == retrieval.MODEL_NAME
    assert len(manifest["query_sha256"]) == 64
    assert manifest["terms"] == ["schwarz lemma", "polynomial", "quasiconformal"]
    assert manifest["top_per_method_per_site"] == 100
    assert manifest["counts"]["total"] == len(candidates)
    assert manifest["counts"]["by_site"] == {site: sum(c["site"] == site for c in candidates)
                                             for site in STORES}
    assert any("answer" in limit for limit in manifest["corpus_limits"])
    assert any("comment" in limit for limit in manifest["corpus_limits"])
    assert "One answer per question." in manifest["corpus_limits"]
    for entry in manifest["stores"]:
        for name in ("entities.json", "embeddings.npy"):
            assert entry["files"][name]["size"] == (STORES[entry["site"]] / name).stat().st_size
            assert entry["files"][name]["mtime_ns"] > 0
        assert entry["row_alignment"]["own_row_nearest_fraction"] == 1.0
        assert entry["row_count"]["entities_match_vectors"] is True


def test_git_sha_is_recorded():
    assert retrieval.git_sha(REPO)


def test_limit_is_applied_per_method_per_site(tmp_path, query, embedder):
    outdir = retrieval.run_retrieval(query, [("math", STORES["math"])], tmp_path / "out",
                                     embedder, limit=2)
    candidates = [json.loads(line) for line in
                  (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(candidates) == 4
    assert {c["rank"] for c in candidates} == {1, 2}


def test_run_refuses_to_overwrite_an_earlier_run(tmp_path, query, embedder):
    outdir = tmp_path / "out"
    retrieval.run_retrieval(query, [("math", STORES["math"])], outdir, embedder, run_id="a")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        retrieval.run_retrieval(query, [("math", STORES["math"])], outdir, embedder, run_id="b")
    assert json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "a"


# query shapes --------------------------------------------------------------------

def test_query_accepts_a_bare_list_a_terms_key_and_categories(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"term": "disk", "source": "note"}]), encoding="utf-8")
    assert retrieval.query_terms(retrieval.load_query(bare)) == ["disk"]

    keyed = tmp_path / "keyed.json"
    keyed.write_text(json.dumps({"problem": "a96J06", "terms": [
        {"term": "Disk", "source": {"note": "Blueprint"}, "category": "object"},
        {"term": "disk", "source": "duplicate"}]}), encoding="utf-8")
    query = retrieval.load_query(keyed)
    assert retrieval.query_terms(query) == ["disk"]
    assert retrieval.query_embedding_text(query) == "object: Disk. term: disk"


@pytest.mark.parametrize("payload", [
    [], {"vocabulary": []}, {"vocabulary": [{"term": "disk"}]},
    {"vocabulary": [{"source": "note"}]}, {"vocabulary": "disk"}, "disk",
])
def test_query_without_terms_or_sources_is_refused(tmp_path, payload):
    path = tmp_path / "query.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        retrieval.load_query(path)


# streaming decoder and fixture integrity ----------------------------------------

@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 17, 4096])
def test_streaming_decoder_survives_every_chunk_boundary(tmp_path, chunk_size):
    awkward = "] [ } { , \" \\ ∀ξ"  # every character the decoder must not treat as structure
    rows = [
        {"entity/id": "se-x-1", "title": "brackets ] [ inside", "tags": ["a", "b"]},
        {"entity/id": "se-x-2", "title": awkward,
         "nested": {"inner": [1, 2, {"deep": awkward, "more": [awkward, [awkward]]}]}, "n": 1.25e3},
        {"entity/id": "se-x-3", "title": ""},
        {"entity/id": "se-x-4", "body": "x" * 5000},
    ]
    path = tmp_path / "entities.json"
    path.write_text("\n[\n" + ",\n\t".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n]\n",
                    encoding="utf-8")
    assert list(retrieval.iter_json_array(path, chunk_size)) == rows


@pytest.mark.parametrize("text", ["", "[", "[1, 2]", "[{\"a\": 1}", "{\"a\": 1}",
                                  "[{\"a\": 1}, "])
def test_streaming_decoder_refuses_files_that_are_not_whole_arrays(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(retrieval.iter_json_array(path))


def test_fixture_vectors_regenerate_exactly_from_the_entities(embedder):
    """Guards the fixture's row alignment: anyone editing entities.json must rebuild."""
    for store in STORES.values():
        entities = list(retrieval.iter_json_array(store / "entities.json"))
        expected = np.asarray(embedder.encode([retrieval.embedding_text(e) for e in entities]),
                              dtype=np.float32)
        assert np.array_equal(np.load(store / "embeddings.npy"), expected)


def test_fake_similarity_reflects_shared_tokens_for_the_pinned_rows(embedder):
    """The pinned embedding order is only meaningful if the fake vectors are honest for the rows
    it ranks: a featured row's similarity to the test query must be positive exactly when the
    two share a token, and exactly zero otherwise (a hash collision would break this)."""
    query = "named-method: Schwarz lemma. object: polynomial. term: quasiconformal"
    query_vector = embedder.encode_query(query)
    query_tokens = set(retrieval.tokenize(query))
    rows = [(thread, entity) for site, threads in FEATURED.items()
            for thread in threads for entity in _entities(site)
            if entity["entity/id"].endswith(thread)]
    assert len(rows) == sum(len(threads) for threads in FEATURED.values())
    for thread, entity in rows:
        shared = query_tokens.intersection(retrieval.tokenize(retrieval.embedding_text(entity)))
        text = retrieval.embedding_text(entity)
        similarity = float(embedder.encode([text])[0] @ query_vector)
        if shared:
            assert similarity > 0.0, (thread, shared, similarity)
        else:
            assert similarity == 0.0, (thread, similarity)


def test_fixture_pinned_row_counts():
    assert len(_entities("math")) == 30
    assert len(_entities("mo")) == 12


# stated embedding queries -----------------------------------------------------

def test_stated_embedding_queries_are_used_and_each_hit_names_its_best_query(tmp_path, embedder):
    texts = ["Schwarz lemma for a holomorphic self-map of the unit disk",
             "roots of a polynomial"]
    path = tmp_path / "query.json"
    path.write_text(json.dumps({
        "terms": [{"term": "polynomial", "source": "statement", "kind": "object"}],
        "embedding_queries": [{"text": t, "source": "statement"} for t in texts],
    }), encoding="utf-8")
    outdir = retrieval.run_retrieval(path, list(STORES.items()), tmp_path / "out", embedder)
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query_texts"] == texts
    hits = [json.loads(line) for line in
            (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    dense = [c for c in hits if c["method"] == "embedding"]
    assert dense and {c["retrieval_reason"]["query_text"] for c in dense} == set(texts)
    vectors = retrieval.normalize_rows(np.stack(
        [retrieval.encode_query_text(embedder, t) for t in texts]).astype(np.float32))
    for site, store in STORES.items():
        matrix = retrieval.normalize_rows(np.load(store / "embeddings.npy").astype(np.float32))
        rows = {e["entity/id"]: i for i, e in enumerate(_entities(site))}
        for c in (c for c in dense if c["site"] == site):
            cosines = matrix[rows[c["entity_id"]]] @ vectors.T
            assert c["score"] == pytest.approx(float(cosines.max()), abs=1e-5)
            assert c["retrieval_reason"]["query_index"] == int(cosines.argmax())
            assert c["retrieval_reason"]["query_text"] == texts[int(cosines.argmax())]


def test_empty_or_malformed_embedding_queries_are_refused(tmp_path, embedder):
    path = tmp_path / "query.json"
    path.write_text(json.dumps({
        "terms": [{"term": "polynomial", "source": "statement"}],
        "embedding_queries": [{"source": "statement"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        retrieval.run_retrieval(path, list(STORES.items()), tmp_path / "out", embedder)


def test_the_a96J06_query_file_loads():
    path = Path(__file__).resolve().parents[1] / "holes/inkup/pilots/a96J06/retrieval-query.json"
    query = retrieval.load_query(path)
    assert len(retrieval.query_terms(query)) == 30
    assert len(retrieval.query_embedding_texts(query)) == 5


# row alignment on a store larger than the sample ------------------------------
# The fixture stores are smaller than the 200-row sample, so there the sampled rows
# are rows 0..n-1 and a check that compared against sample POSITIONS would pass.
# On the real MathOverflow store that comparison reported 1/200 own-row hits with
# mean own cosine 1.0.

def _hash_embedder(texts):
    import hashlib
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rows.append(np.random.default_rng(seed).standard_normal(32))
    return np.asarray(rows, dtype=np.float32)


def _big_store(path, count=700):
    path.mkdir()
    entities = [{"entity/id": f"se-big-{i}", "title": f"title {i}",
                 "question-body": f"question body number {i}",
                 "answer-body": f"answer body number {i}", "tags": [], "created": None}
                for i in range(count)]
    (path / "entities.json").write_text(json.dumps(entities), encoding="utf-8")
    vectors = retrieval.normalize_rows(
        _hash_embedder([retrieval.embedding_text(e) for e in entities]))
    np.save(path / "embeddings.npy", vectors)
    return vectors


def test_row_alignment_is_one_when_the_store_is_larger_than_the_sample(tmp_path):
    _big_store(tmp_path / "big")
    report = retrieval.check_row_alignment(tmp_path / "big" / "entities.json",
                                           tmp_path / "big" / "embeddings.npy", _hash_embedder)
    assert report["sample_count"] == 200 and report["sample_capped_by_store_size"] is False
    assert report["own_row_nearest_fraction"] == 1.0
    assert report["trusted"] is True


def test_row_alignment_is_zero_when_a_large_store_is_shifted(tmp_path):
    vectors = _big_store(tmp_path / "big")
    np.save(tmp_path / "big" / "embeddings.npy", np.roll(vectors, 1, axis=0))
    report = retrieval.check_row_alignment(tmp_path / "big" / "entities.json",
                                           tmp_path / "big" / "embeddings.npy", _hash_embedder)
    assert report["own_row_nearest_fraction"] == 0.0
    assert report["trusted"] is False


# anchor terms: a lexical candidate must match one when the query marks any -------------
# Fixture rows used below: 1001 and 1002 say "Schwarz lemma"; 1003 says "polynomial" five
# times and never "Schwarz" or "lemma"; without the rule 1003 is the top lexical hit.

ANCHORED = [
    {"term": "Schwarz lemma polynomial", "source": "test", "anchor": True},
    {"term": "polynomial", "source": "test"},
    {"term": "disk", "source": "test"},
]


def _anchored_run(tmp_path, embedder, terms):
    path = tmp_path / "anchored-query.json"
    path.write_text(json.dumps(terms), encoding="utf-8")
    outdir = retrieval.run_retrieval(path, list(STORES.items()), tmp_path / "out", embedder)
    candidates = [json.loads(line) for line in
                  (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    return candidates, json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))


def test_without_anchors_the_manifest_and_reasons_carry_no_anchor_rule(run):
    _, candidates, manifest = run
    assert manifest["anchors"] == [] and manifest["lexical_anchor_rule"] is None
    for entry in manifest["stores"]:
        assert "anchor_filter" not in entry
    assert all("admitted_by" not in c["retrieval_reason"]
               for c in candidates if c["method"] == "lexical")


def test_a_query_whose_anchor_flag_is_not_true_is_treated_as_unanchored(tmp_path, query, embedder, run):
    plain = [dict(item) for item in json.loads(query.read_text())]
    plain[0]["anchor"] = "true"  # only the boolean true marks an anchor
    flagged = tmp_path / "flagged.json"
    flagged.write_text(json.dumps(plain))
    outdir = retrieval.run_retrieval(flagged, list(STORES.items()), tmp_path / "flagged-out", embedder)
    candidates = [json.loads(line) for line in
                  (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(c["site"], c["method"], c["thread_id"], round(c["score"], 9), c["retrieval_reason"])
            for c in candidates] == [(c["site"], c["method"], c["thread_id"], round(c["score"], 9), c["retrieval_reason"])
                                     for c in run[1]]


def test_with_an_anchor_the_strongest_non_anchor_match_is_absent(tmp_path, embedder):
    candidates, manifest = _anchored_run(tmp_path, embedder, ANCHORED)
    lexical = _ids(candidates, "lexical", "math")
    assert "1003" not in lexical            # five "polynomial", one anchor token of three
    assert set(lexical) == {"1001", "1002"}
    for entry in manifest["stores"]:
        if entry["site"] == "math":
            assert entry["anchor_filter"]["documents_passing"] == 2
            assert entry["anchor_filter"]["documents"] == 30
    assert manifest["anchors"] == ["schwarz lemma polynomial"]
    assert "60%" in manifest["lexical_anchor_rule"]


def test_an_admitted_candidate_names_the_anchor_that_admitted_it(tmp_path, embedder):
    candidates, _ = _anchored_run(tmp_path, embedder, ANCHORED)
    by_id = {c["thread_id"]: c for c in candidates if c["method"] == "lexical" and c["site"] == "math"}
    assert by_id["1001"]["retrieval_reason"]["admitted_by"] == ["schwarz lemma polynomial"]
    assert by_id["1002"]["retrieval_reason"]["admitted_by"] == ["schwarz lemma polynomial"]
    # the anchor is matched by its tokens, not as a phrase: neither post contains the phrase
    assert "schwarz lemma polynomial" not in by_id["1001"]["retrieval_reason"]["matched_terms"]


def test_one_token_of_a_three_token_anchor_admits_nothing():
    """The bad case: a post with "norm" but not "essential" or "supremum" is not admitted,
    however well it scores on the other terms."""
    entities = STORES["math"] / "entities.json"
    tokens, required = retrieval.anchor_required_tokens("essential supremum norm")
    assert (tokens, required) == ({"essential", "supremum", "norm"}, 2)
    assert retrieval.anchors_admitting({"norm", "sup", "polynomial"}, {"essential supremum norm": (tokens, required)}) == []
    result = retrieval.lexical_retrieve(entities, ["essential supremum norm", "polynomial", "sup norm"], 10,
                                        anchors=["essential supremum norm"])
    assert result["hits"] == []
    assert result["anchor_filter"]["documents_passing"] == 0
    unanchored = retrieval.lexical_retrieve(entities, ["essential supremum norm", "polynomial", "sup norm"], 10)
    assert [hit[2]["entity/id"] for hit in unanchored["hits"]] == ["se-math-1003"]


def test_short_anchors_need_the_tokens_they_have():
    assert retrieval.anchor_required_tokens("sup norm")[1] == 2
    assert retrieval.anchor_required_tokens("quasiconformal")[1] == 1
    # LaTeX leaves single letters behind; a command name is a content token
    assert retrieval.anchor_required_tokens("$\\alpha < \\|f\\|_\\infty$") == ({"alpha", "infty"}, 2)


def test_an_anchor_without_content_tokens_or_outside_the_terms_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no content token"):
        retrieval.anchor_required_tokens("$x$ a")
    entities = STORES["math"] / "entities.json"
    with pytest.raises(ValueError, match="among the query terms"):
        retrieval.lexical_retrieve(entities, ["polynomial"], 5, anchors=["schwarz lemma"])


def test_the_anchor_rule_does_not_touch_the_embedding_method(tmp_path, embedder):
    anchored, _ = _anchored_run(tmp_path, embedder, ANCHORED)
    path = tmp_path / "plain.json"
    path.write_text(json.dumps([{k: v for k, v in item.items() if k != "anchor"} for item in ANCHORED]))
    outdir = retrieval.run_retrieval(path, list(STORES.items()), tmp_path / "plain-out", embedder)
    plain = [json.loads(line) for line in
             (outdir / "question-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    key = lambda cs: [(c["site"], c["thread_id"], round(c["score"], 6)) for c in cs if c["method"] == "embedding"]
    assert key(anchored) == key(plain) and key(anchored)


def test_function_words_are_not_anchor_content_tokens():
    """Before this, "as", "to" and "norm" alone satisfied the first a00J01 anchor."""
    tokens, required = retrieval.anchor_required_tokens("L^p norm as p tends to infinity")
    assert (tokens, required) == ({"norm", "tend", "infinity"}, 2)
    spec = {"anchor": (tokens, required)}
    assert retrieval.anchors_admitting({"as", "to", "norm"}, spec) == []
    assert retrieval.anchors_admitting({"norm", "infinity"}, spec) == ["anchor"]


def test_anchor_tokens_only_in_the_answer_body_do_not_admit(tmp_path):
    store = tmp_path / "entities.json"
    store.write_text(json.dumps([
        {"entity/id": "se-math-1", "title": "A polynomial question", "question-body": "Why is this polynomial bounded?",
         "answer-body": "Use the Schwarz lemma."},
        {"entity/id": "se-math-2", "title": "Schwarz lemma", "question-body": "A polynomial and the Schwarz lemma.",
         "answer-body": ""},
    ]), encoding="utf-8")
    result = retrieval.lexical_retrieve(store, ["schwarz lemma", "polynomial"], 10, anchors=["schwarz lemma"])
    assert [hit[2]["entity/id"] for hit in result["hits"]] == ["se-math-2"]
    assert result["anchor_filter"]["documents_passing"] == 1
