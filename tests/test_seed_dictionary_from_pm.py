"""Tests for the PlanetMath dictionary seed loader."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import edn_format


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "seed-dictionary-from-pm.py"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "planetmath-mini"


def load_script_module():
    spec = importlib.util.spec_from_file_location("seed_dictionary_from_pm", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mini_kernel_tsv(tmp_path: Path) -> Path:
    path = tmp_path / "terms.tsv"
    path.write_text(
        "\n".join([
            "term_lower\tterm\tsource\tcanon_or_count",
            "ringed space\tRinged space\tpm-title\tRingedSpace",
            "moscow mathematical papyrus\tMoscow Mathematical Papyrus\tpm-title\tMoscowMathematicalPapyrus",
            "simple definitional paragraph\tSimple definitional paragraph\tpm-title\tSimpleDefinitionalParagraph",
            "0\t0\tpm-title\t0",
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def edn_to_plain(value):
    if isinstance(value, edn_format.ImmutableDict):
        out = {}
        for key, inner in value.items():
            key_name = key.name if hasattr(key, "name") else str(key)
            out[key_name] = edn_to_plain(inner)
        return out
    if isinstance(value, (edn_format.ImmutableList, list, tuple)):
        return [edn_to_plain(item) for item in value]
    if hasattr(value, "name"):
        return value.name
    return value


def test_extract_definition_from_definition_block():
    module = load_script_module()
    tex_path = FIXTURE_ROOT / "14_Algebraic_geometry" / "14A15-RingedSpace.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="ringed-space",
        headword="Ringed space",
        msc_code="14A15",
        subject_area="14_Algebraic_geometry",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    definition = module.extract_definition_from_pm(article)
    assert definition is not None
    assert "A ringed space is a pair" in definition


def test_find_pm_articles_skips_malformed_tex():
    module = load_script_module()
    skipped = []
    articles = list(module.find_pm_articles(FIXTURE_ROOT, skip_recorder=lambda reason, path: skipped.append((reason, path.name))))
    names = {article.tex_path.name for article in articles}
    assert "54E40-MalformedTexExample.tex" not in names
    assert ("malformed-tex", "54E40-MalformedTexExample.tex") in skipped


def test_no_definition_article_is_marked_canonical_no_definition(tmp_path: Path):
    module = load_script_module()
    kernel_path = mini_kernel_tsv(tmp_path)
    lookup = module.load_kernel_lookup(kernel_path)
    tex_path = FIXTURE_ROOT / "00_General" / "00X00-MoscowMathematicalPapyrus.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="moscow-mathematical-papyrus",
        headword="Moscow Mathematical Papyrus",
        msc_code="00X00",
        subject_area="00_General",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    entry = module.pm_article_to_entry(
        article,
        lookup[module.normalize_lookup_term(article.headword)],
        extracted_at_iso="2026-05-19T00:00:00Z",
    )
    assert entry["term/status"].name == "canonical-no-definition"
    assert entry["term/definitions"] == []


def test_definition_article_without_headword_in_subject_still_extracts(tmp_path: Path):
    module = load_script_module()
    tex_path = FIXTURE_ROOT / "00_General" / "00A05-TimeInvariant.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="time-invariant",
        headword="time invariant",
        msc_code="00A05",
        subject_area="00_General",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    definition = module.extract_definition_from_pm(article)
    assert definition is not None
    assert "time-invariant" in definition


def test_theorem_article_uses_statement_as_definition(tmp_path: Path):
    module = load_script_module()
    tex_path = FIXTURE_ROOT / "03_Mathematical_logic_and_foundations" / "03-00-FiniteInjectionIsBijective.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="finite-injection-is-bijective",
        headword="finite injection is bijective",
        msc_code="03-00",
        subject_area="03_Mathematical_logic_and_foundations",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    definition = module.extract_definition_from_pm(article)
    assert definition is not None
    assert "If $f : A B$ is injective, then $f$ is bijective." in definition


def test_example_article_is_skipped_from_dictionary(tmp_path: Path):
    module = load_script_module()
    kernel_path = mini_kernel_tsv(tmp_path)
    lookup = module.load_kernel_lookup(kernel_path)
    tex_path = FIXTURE_ROOT / "00_General" / "00A05-WorkedInductionExample.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="worked-induction-example",
        headword="worked induction example",
        msc_code="00A05",
        subject_area="00_General",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    assert module.pm_article_to_entry(
        article,
        lookup.get(module.normalize_lookup_term(article.headword)),
        extracted_at_iso="2026-05-19T00:00:00Z",
    ) is None


def test_theorem_like_plain_paragraph_before_proof_is_extracted():
    module = load_script_module()
    tex_path = FIXTURE_ROOT / "54_General_topology" / "54E50-ClosedSubsetComplete.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="closed-subset-complete",
        headword="a closed subset of a complete metric space is complete",
        msc_code="54E50",
        subject_area="54_General_topology",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    definition = module.extract_definition_from_pm(article)
    assert definition is not None
    assert "Then $Y$ is complete." in definition


def test_numeric_id_article_is_skipped(tmp_path: Path):
    module = load_script_module()
    kernel_path = mini_kernel_tsv(tmp_path)
    lookup = module.load_kernel_lookup(kernel_path)
    tex_path = FIXTURE_ROOT / "00_General" / "00-0.tex"
    raw = tex_path.read_text(encoding="utf-8")
    article = module.PMArticle(
        canon_id="0",
        headword="0",
        msc_code="00",
        subject_area="00_General",
        tex_path=tex_path,
        body_text=module.extract_document_body(raw),
        raw_tex=raw,
    )
    assert module.pm_article_to_entry(
        article,
        lookup[module.normalize_lookup_term(article.headword)],
        extracted_at_iso="2026-05-19T00:00:00Z",
    ) is None


def test_end_to_end_run_is_idempotent_with_fixed_timestamp(tmp_path: Path):
    module = load_script_module()
    kernel_path = mini_kernel_tsv(tmp_path)
    out_dir = tmp_path / "out"
    argv = [
        "--planetmath-root", str(FIXTURE_ROOT),
        "--kernel-tsv", str(kernel_path),
        "--out-dir", str(out_dir),
        "--schema-path", str(REPO_ROOT / "holes" / "excursions" / "dictionary-schema.edn"),
        "--timestamp", "2026-05-19T00:00:00Z",
    ]

    module.main(argv)
    first_entries = (out_dir / "entries-pm-seed.edn").read_text(encoding="utf-8")
    module.main(argv)
    second_entries = (out_dir / "entries-pm-seed.edn").read_text(encoding="utf-8")

    assert first_entries == second_entries

    parsed_entries = edn_to_plain(edn_format.loads(first_entries))
    assert parsed_entries["dictionary/entry-count"] == 6
    assert len(parsed_entries["dictionary/entries"]) == 6

    audit_sample = json.loads((out_dir / "audit-sample.json").read_text(encoding="utf-8"))
    assert len(audit_sample) == 6

    stats = json.loads((out_dir / "run-stats.json").read_text(encoding="utf-8"))
    assert stats["succeeded_entries"] == 6
    assert stats["skipped"]["by_reason"]["malformed-tex"] == 1
    assert stats["skipped"]["by_reason"]["non-dictionary-pmtype-skip"] == 1
    assert stats["skipped"]["by_reason"]["numeric-id-skip"] == 1
