from pathlib import Path

import edn_format
import pytest

from futon6 import peradam_cert as pc


ROOT = Path(__file__).resolve().parents[1]
REAL_FOLD_TURNS = ROOT / "data" / "fold-turns"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def kwdump(value):
    return edn_format.dumps(value)


def fixture_files(tmp_path):
    fold_turns = tmp_path / "fold-turns"
    deposit = write(
        fold_turns / "ft-good-001.edn",
        '{:fold-turn/id "ft-good-001" :arming {:operator "joe" :word "go" :at "t" :scope :one-fold}}\n',
    )
    deposit_sha = pc.sha256_file(deposit)
    seal = write(tmp_path / "sealed" / "a-next-key.edn", "{:sealed true :gold [:a :b]}\n")
    seal_sha = pc.sha256_file(seal)
    score = write(
        tmp_path / "scores" / "score.edn",
        kwdump(
            {
                edn_format.Keyword("score/id"): "score-1",
                edn_format.Keyword("deposit-sha"): deposit_sha,
                edn_format.Keyword("seal-sha"): seal_sha,
                edn_format.Keyword("scorer-id"): "reviewer",
                edn_format.Keyword("author-id"): "author",
                edn_format.Keyword("verdict"): "pass",
                edn_format.Keyword("timestamp"): "2026-07-05T00:00:00Z",
            }
        ),
    )
    score_sha = pc.sha256_file(score)
    cert = {
        edn_format.Keyword("peradam/id"): "peradam-good-1",
        edn_format.Keyword("deposit-ref"): "ft-good-001",
        edn_format.Keyword("deposit-sha"): deposit_sha,
        edn_format.Keyword("seal-ref"): {
            edn_format.Keyword("path"): str(seal),
            edn_format.Keyword("sha256"): seal_sha,
            edn_format.Keyword("sealed?"): True,
        },
        edn_format.Keyword("blind-score"): {
            edn_format.Keyword("path"): str(score),
            edn_format.Keyword("sha256"): score_sha,
        },
        edn_format.Keyword("scorer-identity"): {edn_format.Keyword("id"): "reviewer"},
        edn_format.Keyword("author-identity"): {edn_format.Keyword("id"): "author"},
        edn_format.Keyword("scorer-not-author?"): True,
        edn_format.Keyword("arming-ref"): {edn_format.Keyword("fold-turn/id"): "ft-good-001"},
    }
    return fold_turns, cert


def refusal_cause(excinfo):
    return excinfo.value.refusal.cause


def test_empty_store_loads_as_empty(tmp_path):
    out = pc.load_certificates(tmp_path / "peradams")
    assert out == {"certified": [], "refused": []}


def test_valid_structured_certificate_loads(tmp_path):
    fold_turns, cert = fixture_files(tmp_path)
    path = write(tmp_path / "peradams" / "good.edn", kwdump(cert))

    loaded = pc.load_certificate(path, fold_turns_dir=fold_turns)

    assert loaded["status"] == "certified"
    assert loaded["scorer-id"] == "reviewer"
    assert loaded["author-id"] == "author"
    assert loaded["scorer-not-author?"] is True


def test_refuses_scorer_equals_author(tmp_path):
    fold_turns, cert = fixture_files(tmp_path)
    cert[edn_format.Keyword("author-identity")] = {edn_format.Keyword("id"): "reviewer"}
    path = write(tmp_path / "peradams" / "self.edn", kwdump(cert))

    with pytest.raises(pc.PeradamRefusal) as excinfo:
        pc.load_certificate(path, fold_turns_dir=fold_turns)

    assert refusal_cause(excinfo) == "scorer-is-author"
    assert excinfo.value.refusal.witness == "scorer-not-author"


def test_refuses_unsealed_key(tmp_path):
    fold_turns, cert = fixture_files(tmp_path)
    cert[edn_format.Keyword("seal-ref")][edn_format.Keyword("sealed?")] = False
    path = write(tmp_path / "peradams" / "unsealed.edn", kwdump(cert))

    with pytest.raises(pc.PeradamRefusal) as excinfo:
        pc.load_certificate(path, fold_turns_dir=fold_turns)

    assert refusal_cause(excinfo) == "unsealed-key"
    assert excinfo.value.refusal.witness == "seal-ref"


def test_refuses_tampered_score_sha(tmp_path):
    fold_turns, cert = fixture_files(tmp_path)
    cert[edn_format.Keyword("blind-score")][edn_format.Keyword("sha256")] = "0" * 64
    path = write(tmp_path / "peradams" / "tampered.edn", kwdump(cert))

    with pytest.raises(pc.PeradamRefusal) as excinfo:
        pc.load_certificate(path, fold_turns_dir=fold_turns)

    assert refusal_cause(excinfo) == "tampered-score"
    assert excinfo.value.refusal.witness == "blind-score"


def test_refuses_missing_arming_or_mana(tmp_path):
    fold_turns, cert = fixture_files(tmp_path)
    cert.pop(edn_format.Keyword("arming-ref"))
    path = write(tmp_path / "peradams" / "no-consent.edn", kwdump(cert))

    with pytest.raises(pc.PeradamRefusal) as excinfo:
        pc.load_certificate(path, fold_turns_dir=fold_turns)

    assert refusal_cause(excinfo) == "missing-arming-or-mana"
    assert excinfo.value.refusal.witness == "arming-or-mana"


def test_store_keeps_refused_record_boundaried(tmp_path):
    fold_turns, cert = fixture_files(tmp_path)
    cert[edn_format.Keyword("blind-score")][edn_format.Keyword("sha256")] = "0" * 64
    write(tmp_path / "peradams" / "tampered.edn", kwdump(cert))

    out = pc.load_certificates(tmp_path / "peradams", fold_turns_dir=fold_turns)

    assert out["certified"] == []
    assert out["refused"][0]["cause"] == "tampered-score"
    assert "peradam-good-1" in out["refused"][0]["raw"]


def test_real_corpus_refusals_name_current_causes():
    autoclock = pc.refusal_for_fold_turn(REAL_FOLD_TURNS / "ft-autoclock-in-001.edn")
    assert autoclock["cause"] == "unstructured-witnesses"
    assert "prose" in autoclock["message"]

    for name in [
        "ft-bayesian-structure-learning-003.edn",
        "ft-aif-head-004.edn",
        "ft-action-vocabulary-005.edn",
    ]:
        refusal = pc.refusal_for_fold_turn(REAL_FOLD_TURNS / name)
        assert refusal["cause"] == "missing-seal"
        assert refusal["witness"] == "seal-ref"
