"""Smoke tests for scripts/superpod-job.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path


def _run_superpod(root: Path, outdir: Path, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    posts = root / "tests/fixtures/se-mini/Posts.xml"
    comments = root / "tests/fixtures/se-mini/Comments.xml"
    cmd = [
        sys.executable,
        "scripts/superpod-job.py",
        str(posts),
        "--comments-xml",
        str(comments),
        "--site",
        "math.stackexchange",
        "--output-dir",
        str(outdir),
        "--min-score",
        "0",
        "--skip-embeddings",
        "--skip-llm",
        "--skip-clustering",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_arxiv_superpod(
    root: Path,
    outdir: Path,
    input_dir: Path,
    extra_args: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "scripts/superpod-job.py",
        "--input-dir",
        str(input_dir),
        "--arxiv-jsonl",
        "batch-001.jsonl",
        "--site",
        "arxiv.math",
        "--output-dir",
        str(outdir),
        "--skip-embeddings",
        "--skip-llm",
        "--skip-clustering",
        "--skip-graph-embed",
        "--skip-faiss",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_superpod_job_module(root: Path):
    spec = importlib.util.spec_from_file_location(
        "superpod_job_for_test",
        root / "scripts" / "superpod-job.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_arxiv_fixture(input_dir: Path, *, count: int = 1) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    rows = []
    for idx in range(count):
        numeric = 102067 + idx
        arxiv_id = f"math/{numeric:07d}v1"
        rows.append(
            json.dumps(
                {
                    "id": arxiv_id,
                    "title": f"A toy theorem {idx + 1}",
                    "abstract": f"We prove that $x_{idx + 1}=x_{idx + 1}$ by a short argument.",
                    "categories": ["math.CT"],
                    "date": "2001-02-07",
                }
            )
        )
        (eprints / f"math__{numeric:07d}v1.tex").write_text(
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            f"\\begin{{theorem}}For every object $X_{{{idx + 1}}}$, "
            f"$X_{{{idx + 1}}}=X_{{{idx + 1}}}$.\\end{{theorem}}\n"
            f"\\begin{{proof}}Use the identity morphism $1_{{X_{{{idx + 1}}}}}:"
            f"X_{{{idx + 1}}}\\to X_{{{idx + 1}}}$.\\end{{proof}}\n"
            "\\end{document}\n",
            encoding="utf-8",
        )
    (input_dir / "batch-001.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return eprints


def _write_legacy_alias_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "A toy theorem",
            "abstract": "We prove that $x=x$ by a short argument.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\newtheorem{thm}{Theorem}\n"
        "\\begin{document}\n"
        "\\begin{thm}For every object $X$, $X=X$.\\end{thm}\n"
        "\\begin{proof}Use the identity morphism $1_X:X\\to X$.\\end{proof}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def _write_minimal_ner_kernel(tmp_path: Path) -> Path:
    path = tmp_path / "terms.tsv"
    path.write_text(
        "term_lower\tterm_orig\tunused\tcanon\n"
        "object\tObject\t_\tObject\n"
        "identity morphism\tidentity morphism\t_\tIdentityMorphism\n"
        "localization\tlocalization\t_\tLocalization\n",
        encoding="utf-8",
    )
    return path


def _write_structure_learning_ner_kernel(tmp_path: Path) -> Path:
    path = tmp_path / "structure-terms.tsv"
    path.write_text(
        "term_lower\tterm_orig\tunused\tcanon\n"
        "toy localization\ttoy localization\t_\tToyLocalization\n"
        "known concept\tknown concept\t_\tKnownConcept\n"
        "setting\tsetting\t_\tSetting\n"
        "bilinear form\tbilinear form\t_\tBilinearForm\n"
        "inner product\tinner product\t_\tInnerProduct\n"
        "norm\tnorm\t_\tNorm\n",
        encoding="utf-8",
    )
    return path


def _write_minimal_seed_edn(path: Path, terms: list[str]) -> Path:
    entries = "\n".join(
        f'    {{:term/lower "{term}"}}' for term in terms
    )
    path.write_text(
        "{:dictionary/entries [\n"
        f"{entries}\n"
        "]}\n",
        encoding="utf-8",
    )
    return path


def _write_minimal_nnexus_stopwords(path: Path) -> Path:
    path.write_text(
        "package NNexus::StopWordList;\n"
        "sub getStopWords {\n"
        "  return [qw/free internal smallest/];\n"
        "}\n"
        "1;\n",
        encoding="utf-8",
    )
    return path


def _write_minimal_nnexus_snapshot(path: Path) -> Path:
    path.write_text(
        "INSERT INTO \"concepts\" VALUES(1,'etale','morphism',0,0);\n",
        encoding="utf-8",
    )
    return path


def _write_preregister_baseline_manifest(path: Path, *, entity_count: int = 1,
                                         scope_cov: float = 1.0, avg_nodes: float = 15.0,
                                         with_claims: int = 1, papers: int = 1) -> Path:
    payload = {
        "entity_count": entity_count,
        "paper_eprint_dir": "/tmp/eprints",
        "readiness": {"status": "pass", "issues": 0, "preflight": False},
        "health_issues": [],
        "stage5_stats": {
            "scope_coverage": scope_cov,
            "text_source_counts": {"eprint": entity_count, "abstract": 0},
        },
        "stage9a_stats": {
            "paper_text_source": "eprints",
            "avg_nodes": avg_nodes,
            "avg_edges": max(0.0, avg_nodes - 1.0),
            "geometry_stats": {
                "papers": papers,
                "with_claims": with_claims,
            },
        },
        "stage_status": {
            "ner_scopes": {
                "status": "completed",
                "entities_processed": entity_count,
                "text_source_counts": {"eprint": entity_count, "abstract": 0},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_scope_free_abstract_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "A theorem with a scope-free abstract",
            "abstract": (
                "We study localization phenomena in model categories and prove "
                "several new existence results."
            ),
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\begin{theorem}For every object $X$, $X=X$.\\end{theorem}\n"
        "\\begin{proof}Use the identity morphism $1_X:X\\to X$.\\end{proof}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def _write_discovery_learning_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "Learning terms from eprints",
            "abstract": "We introduce a new categorical construction.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\begin{definition}\n"
        "A \\emph{known concept} is defined as a previously named construction.\n"
        "\\end{definition}\n"
        "\\begin{definition}\n"
        "A \\emph{toy localization} is defined as a localization with a witness.\n"
        "\\end{definition}\n"
        "\\begin{theorem}\n"
        "Every toy localization preserves the known concept.\n"
        "\\end{theorem}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def _write_discourse_rich_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "Discourse rich eprint",
            "abstract": "We study a functorial construction.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\begin{definition}\n"
        "Fix $X$ to be a cofibrant object. We write $F$ for the identity functor on $X$.\n"
        "\\end{definition}\n"
        "\\begin{proof}\n"
        "Since $F : X \\to X$, therefore this functor is explicit. "
        "In other words, there exists $f$ such that $f : X \\to X$.\n"
        "\\end{proof}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def _write_structure_learning_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "Structure learning starter",
            "abstract": "We study toy localization and known concept.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "We study toy localization and known concept in a simple setting.\n"
        "We study toy localization and known concept in another setting.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def _write_structure_learning_arxiv_fixture_generalization(input_dir: Path) -> Path:
    """Different concrete terms, different cue padding, same backbone as fixture-1.

    Run-1 fixture-1 learns `we study <term> and <term>`. This fixture-2 residual
    skeletonizes to `we study <term> and <term> when there exists <term>`, a strict
    superset. A correct subsequence matcher fires the prior on this residual.
    """
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102099v1",
            "title": "Generalization probe",
            "abstract": "We study bilinear forms and inner products.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102099v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "We study bilinear forms and inner products when there exists a unique norm.\n"
        "We study bilinear forms and inner products when there exists another norm.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def test_structure_seed_signature_tokens_round_trip():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    assert mod._signature_tokens("we prove that <term> be <term>") == (
        "we", "prove", "that", "<term>", "be", "<term>",
    )
    assert mod._signature_tokens("") == ()


def test_structure_seed_subsequence_match_basic():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    prior = "we prove that <term> be <term>"
    new = "we prove that the <term> be <term> for every prime"
    priors = [(prior, mod._signature_tokens(prior))]
    matched = mod._match_structure_seed_signature(new, priors)
    assert matched == prior


def test_structure_seed_match_rejects_too_short_prior():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    short_prior = "we <term>"
    new = "we study <term> and <term>"
    priors = [(short_prior, mod._signature_tokens(short_prior))]
    assert mod._match_structure_seed_signature(new, priors) is None


def test_structure_seed_match_returns_longest_prior():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    shorter = "we prove <term>"
    longer = "we prove that <term> be <term>"
    new = "we prove that the <term> be <term>"
    priors = [
        (shorter, mod._signature_tokens(shorter)),
        (longer, mod._signature_tokens(longer)),
    ]
    assert mod._match_structure_seed_signature(new, priors) == longer


def test_audit_classify_terms_reports_depth_distribution():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    # Text with whitespace tokenization so spot_terms_entity can find "monad".
    # Outer scope at [0, len(text)] (env/proof), inner at [50, 100] (bind/typed).
    # "monad" placed at offset ~60 sits inside both → depth=2.
    prefix = "a " * 30  # 60 chars
    middle = "monad "
    suffix = "b " * 100
    text = prefix + middle + suffix
    base_records = [
        {"hx/type": "env/proof", "hx/content": {"position": 0, "end": len(text), "match": text}},
        {"hx/type": "bind/typed", "hx/content": {"position": 50, "end": 100, "match": text[50:100]}},
    ]
    singles = {"monad": ("monad", "Monad")}
    multi_index: dict = {}
    stats = mod._audit_classify_terms(text, base_records, singles, multi_index)
    assert stats["total"] == 1
    assert stats["inhabited"] == 1
    assert stats["outer"] == 0
    assert stats["straddled"] == 0
    assert stats["depth_distribution"] == {2: 1}


def test_audit_classify_terms_root_for_term_outside_all_scopes():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    # Scope at [0, 50]; term "functor" placed at ~70 sits outside it.
    text = "a " * 30 + "functor " + "b " * 50  # 60 chars before "functor"
    base_records = [
        {"hx/type": "env/proof", "hx/content": {"position": 0, "end": 50, "match": text[:50]}},
    ]
    singles = {"functor": ("functor", "Functor")}
    multi_index: dict = {}
    stats = mod._audit_classify_terms(text, base_records, singles, multi_index)
    assert stats["total"] == 1
    assert stats["outer"] == 1
    assert stats["inhabited"] == 0
    assert stats["depth_distribution"] == {}


def test_build_scope_tree_inline_matches_viewer_semantics():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    spans = [
        {"start": 0, "end": 100, "label": "env/proof"},
        {"start": 20, "end": 60, "label": "bind/typed"},
        {"start": 80, "end": 90, "label": "constrain/relation"},
    ]
    tree = mod._build_scope_tree(spans, [(30, 35), (85, 88)])
    # Both children of env/proof; inner term in bind/typed, second in constrain/relation.
    assert len(tree["children"]) == 1
    outer = tree["children"][0]
    assert outer["label"] == "env/proof"
    assert outer["depth"] == 1
    assert {c["label"] for c in outer["children"]} == {"bind/typed", "constrain/relation"}
    # Term (30,35) placed in bind/typed at depth 2.
    bind_typed = next(c for c in outer["children"] if c["label"] == "bind/typed")
    constrain = next(c for c in outer["children"] if c["label"] == "constrain/relation")
    assert (30, 35) in bind_typed["terms"]
    assert (85, 88) in constrain["terms"]


def test_structure_seed_match_rejects_when_prior_not_subsequence():
    root = Path(__file__).parent.parent
    mod = _load_superpod_job_module(root)
    prior = "we prove that <term> be <term>"  # has "that"
    new = "we prove <term> be <term>"  # missing "that"
    priors = [(prior, mod._signature_tokens(prior))]
    assert mod._match_structure_seed_signature(new, priors) is None


def test_superpod_job_ct_pipeline_smoke(tmp_path: Path):
    root = Path(__file__).parent.parent
    outdir = tmp_path / "superpod-out"
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    run = _run_superpod(root, outdir, ["--thread-limit", "4", "--ner-kernel", str(ner_kernel)])
    assert run.returncode == 0, (
        "superpod-job failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    # CPU-only mode runs parse, ner, wiring, expressions, hypergraphs (+ graph_embed/faiss if available)
    completed = manifest["stages_completed"]
    assert "parse" in completed
    assert "ner_scopes" in completed
    assert "thread_wiring" in completed
    assert "classical_edge_rate" in manifest["stage7_stats"]
    assert manifest["stage7_stats"]["threads_processed"] == 4
    assert "stage_status" in manifest
    assert manifest["stage_status"]["parse"]["status"] == "completed"
    assert manifest["stage_status"]["embeddings"]["status"] == "skipped"
    assert "skip_reason" in manifest["stage_status"]["embeddings"]
    assert manifest["stage_status"]["reverse_morphogenesis"]["status"] == "skipped"
    assert manifest["readiness"]["status"] in {"pass", "warn"}
    assert isinstance(manifest.get("health_gate_thresholds"), dict)

    wiring = json.loads((outdir / "thread-wiring.json").read_text(encoding="utf-8"))
    assert isinstance(wiring, list)
    assert len(wiring) == 4


def test_superpod_job_limit_defaults_thread_limit(tmp_path: Path):
    root = Path(__file__).parent.parent
    outdir = tmp_path / "superpod-out-limit"

    run = _run_superpod(root, outdir, ["--limit", "2"])
    assert run.returncode == 0, (
        "superpod-job failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "Pilot mode: --thread-limit defaulted to --limit (2)" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entity_count"] == 2
    assert manifest["stage7_stats"]["threads_processed"] == 2
    assert manifest["stage_status"]["thread_wiring"]["status"] == "completed"


def test_arxiv_paper_eprint_dir_feeds_all_paper_stages(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints", "--ner-kernel", str(ner_kernel)],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv eprint run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "Paper eprint preflight: 1/1 arXiv entities" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_hg_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_eprint"]["source"] == "paper-eprint-dir"
    assert manifest["paper_eprint"]["preflight"]["candidate_matches"] == 1
    assert manifest["stage5_stats"]["eprint_text_used"] == 1
    assert manifest["stage_status"]["ner_scopes"]["text_source_counts"]["eprint"] == 1
    assert manifest["llm_batch_sizes"]["stage5d"] == 4
    assert manifest["stage_status"]["technique_ner"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage_status"]["paper_hypergraph"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage9a_stats"]["paper_text_source"] == "eprints"
    assert manifest["stage9a_stats"]["eprint_text_used"] == 1
    geometry = json.loads((outdir / "geometry.json").read_text(encoding="utf-8"))
    assert isinstance(geometry, list)
    assert geometry[0]["paper_id"].startswith("arxiv-")
    assert geometry[0]["n_claims"] == 1
    assert geometry[0]["T_total"] >= 0
    assert "laplacian_summary" in geometry[0]
    assert manifest["stage9a_stats"]["geometry_stats"]["papers"] == 1
    assert manifest["stage9a_stats"]["geometry_stats"]["with_claims"] == 1
    assert manifest["stage9a_stats"]["geometry_source"] == "paper-hypergraphs"
    assert manifest["stage_status"]["hypergraphs"]["geometry_with_claims"] == 1


def test_arxiv_batch_local_eprints_auto_default(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, ["--ner-kernel", str(ner_kernel)])
    assert run.returncode == 0, (
        "superpod-job arxiv auto-default eprint run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "defaulting --discover-terms-eprint-dir to batch-local eprints" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["discover_terms_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_eprint"]["source"] == "discover-terms-eprint-dir-default"
    assert manifest["stage5_stats"]["eprint_text_used"] == 1
    assert manifest["stage_status"]["ner_scopes"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage9a_stats"]["eprint_text_used"] == 1
    assert manifest["stage9a_stats"]["geometry_source"] == "paper-hypergraphs"
    assert manifest["stage9a_stats"]["geometry_stats"]["with_claims"] == 1
    assert (outdir / "geometry.json").exists()


def test_arxiv_stage5_uses_eprints_not_scope_free_abstracts(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_scope_free_abstract_arxiv_fixture(input_dir)
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints", "--ner-kernel", str(ner_kernel)],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv scope-source run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage5_stats"]["eprint_text_used"] == 1
    assert manifest["stage5_stats"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage5_stats"]["scope_coverage"] == 1.0
    assert manifest["stage_status"]["ner_scopes"]["text_source_counts"]["eprint"] == 1

    scopes = json.loads((outdir / "scopes.json").read_text(encoding="utf-8"))
    assert len(scopes) == 1
    assert scopes[0]["count"] > 0


def test_arxiv_run_emits_preregister_qc_sidecar(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_scope_free_abstract_arxiv_fixture(input_dir)
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    baseline_dir = tmp_path / "preregister-baselines"
    baseline_dir.mkdir()
    _write_preregister_baseline_manifest(baseline_dir / "001.json", avg_nodes=10.0)
    _write_preregister_baseline_manifest(baseline_dir / "002.json", avg_nodes=20.0)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--ner-kernel",
            str(ner_kernel),
            "--preregister-qc-baseline-dir",
            str(baseline_dir),
        ],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv preregister QC run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "Preregistered QC:" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    prereg = manifest["preregister_qc"]
    assert prereg["enabled"] is True
    assert prereg["profile"] == "broad-arxiv"
    assert prereg["status"] in {"pass", "warn", "fail"}
    assert Path(prereg["output_path"]).name == "qc-preregister.json"

    report = json.loads((outdir / "qc-preregister.json").read_text(encoding="utf-8"))
    gate_names = {gate["name"]: gate for gate in report["evaluation"]["gates"]}
    assert gate_names["paper_text_provenance"]["status"] == "pass"
    assert report["profile"] == "broad-arxiv"


def test_arxiv_discover_terms_learns_seed_aware_dictionary_entries(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_discovery_learning_arxiv_fixture(input_dir)
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    pm_seed = _write_minimal_seed_edn(tmp_path / "pm-seed.edn", ["known concept"])
    nlab_seed = _write_minimal_seed_edn(tmp_path / "nlab-seed.edn", [])
    nnexus_stopwords = _write_minimal_nnexus_stopwords(tmp_path / "StopWordList.pm")
    nnexus_snapshot = _write_minimal_nnexus_snapshot(tmp_path / "snapshot.sql")
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--discover-terms",
            "--discover-terms-min-freq",
            "1",
            "--discover-terms-eprint-dir",
            "eprints",
            "--discover-terms-pm-seed",
            str(pm_seed),
            "--discover-terms-nlab-seed",
            str(nlab_seed),
            "--discover-terms-nnexus-stopwords",
            str(nnexus_stopwords),
            "--discover-terms-nnexus-snapshot",
            str(nnexus_snapshot),
            "--ner-kernel",
            str(ner_kernel),
        ],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv discovery-learning run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "learned dictionary:" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    open_ner = manifest["stage5_stats"]["open_ner"]
    assert open_ner["enabled"] is True
    assert open_ner["new_terms_learned"] == 1
    assert open_ner["seed_known_terms_missing_from_kernel"] == 1
    assert open_ner["rhs_supported_terms"] == 2
    assert open_ner["pm_seed_terms"] == 1
    assert open_ner["nnexus_snapshot_terms"] == 1
    assert open_ner["learned_dictionary_written"] == 2
    assert Path(open_ner["output_dictionary_jsonl"]).name == "learned-term-dictionary.jsonl"

    candidate_rows = [
        json.loads(line)
        for line in (outdir / "candidate-new-terms.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    learned_rows = [
        json.loads(line)
        for line in (outdir / "learned-term-dictionary.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_term = {row["term_lower"]: row for row in candidate_rows}
    dict_by_term = {row["term_lower"]: row for row in learned_rows}

    assert by_term["known concept"]["novel_vs_seed"] == "known"
    assert by_term["known concept"]["learned_status"] == "seed-known-missing-from-kernel"
    assert by_term["toy localization"]["novel_vs_seed"] == "novel"
    assert by_term["toy localization"]["learned_status"] == "new-term"
    assert by_term["toy localization"]["rhs_support_counts"]["definition-env"] >= 1
    assert by_term["toy localization"]["rhs_support_counts"]["theorem-statement"] >= 1

    assert dict_by_term["known concept"]["term_status"] == "seed-known"
    assert dict_by_term["toy localization"]["term_status"] == "provisional"
    assert dict_by_term["toy localization"]["definitions"]
    assert dict_by_term["toy localization"]["usage_examples"]

    paper_hypergraphs = json.loads((outdir / "paper-hypergraphs.json").read_text(encoding="utf-8"))
    concept_terms = {
        node["attrs"].get("term")
        for node in paper_hypergraphs[0]["nodes"]
        if node.get("type") == "concept"
    }
    assert "toy localization" in concept_terms


def test_arxiv_stage5_emits_discourse_wiring_and_hypergraph_nodes(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_discourse_rich_arxiv_fixture(input_dir)
    ner_kernel = _write_minimal_ner_kernel(tmp_path)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, ["--ner-kernel", str(ner_kernel)])
    assert run.returncode == 0, (
        "superpod-job arxiv discourse run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage5_stats"]["total_wires"] >= 1
    assert manifest["stage5_stats"]["total_ports"] >= 1
    assert manifest["stage5_stats"]["total_labels"] >= 1
    assert Path(manifest["stage5_stats"]["output_discourse_json"]).name == "discourse-wiring.json"

    discourse_rows = json.loads((outdir / "discourse-wiring.json").read_text(encoding="utf-8"))
    assert discourse_rows[0]["counts"]["wires"] >= 1
    assert discourse_rows[0]["counts"]["ports"] >= 1
    assert discourse_rows[0]["counts"]["labels"] >= 1

    hypergraphs = json.loads((outdir / "hypergraphs.json").read_text(encoding="utf-8"))
    node_types = {node["type"] for node in hypergraphs[0]["nodes"]}
    assert "wire" in node_types
    assert "port" in node_types
    assert "label" in node_types


def test_arxiv_stage5_learns_and_reseeds_structure_signatures(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir1 = tmp_path / "arxiv-input-1"
    _write_structure_learning_arxiv_fixture(input_dir1)
    ner_kernel = _write_structure_learning_ner_kernel(tmp_path)

    outdir1 = tmp_path / "arxiv-out-1"
    run1 = _run_arxiv_superpod(
        root,
        outdir1,
        input_dir1,
        [
            "--paper-eprint-dir",
            "eprints",
            "--ner-kernel",
            str(ner_kernel),
            "--discover-structures",
            "--discover-structures-min-signature-freq",
            "1",
        ],
    )
    assert run1.returncode == 0, (
        "superpod-job arxiv structure-learning run failed\n"
        f"stdout:\n{run1.stdout}\n"
        f"stderr:\n{run1.stderr}"
    )

    manifest1 = json.loads((outdir1 / "manifest.json").read_text(encoding="utf-8"))
    structure1 = manifest1["stage5_stats"]["structure_learning"]
    assert structure1["enabled"] is True
    assert structure1["candidates_written"] >= 1
    assert structure1["loss"]["uncovered_sentences_with_known_terms"] >= 1
    candidates = json.loads((outdir1 / "learned-structure-candidates.json").read_text(encoding="utf-8"))
    assert any(row["signature"] == "we study <term> and <term>" for row in candidates), (
        f"expected 'we study <term> and <term>' in signatures, got: "
        f"{[row['signature'] for row in candidates]}"
    )

    # Run 2: DIFFERENT paper, DIFFERENT concrete terms, longer cue chain.
    # Residual signature is `we study <term> and <term> when there exists <term>`,
    # which strictly contains the run-1 signature as a subsequence.
    input_dir2 = tmp_path / "arxiv-input-2"
    _write_structure_learning_arxiv_fixture_generalization(input_dir2)
    outdir2 = tmp_path / "arxiv-out-2"
    run2 = _run_arxiv_superpod(
        root,
        outdir2,
        input_dir2,
        [
            "--paper-eprint-dir",
            "eprints",
            "--ner-kernel",
            str(ner_kernel),
            "--discover-structures",
            "--discover-structures-min-signature-freq",
            "1",
            "--discover-structures-seed-json",
            str(outdir1 / "learned-structure-summary.json"),
        ],
    )
    assert run2.returncode == 0, (
        "superpod-job arxiv structure-reseed run failed\n"
        f"stdout:\n{run2.stdout}\n"
        f"stderr:\n{run2.stderr}"
    )

    manifest2 = json.loads((outdir2 / "manifest.json").read_text(encoding="utf-8"))
    structure2 = manifest2["stage5_stats"]["structure_learning"]
    assert structure2["seed_signatures_loaded"] >= 1
    assert structure2["seed_matches_applied"] >= 1, (
        "run-2 used a different paper and different terms than run-1; the subsequence "
        "matcher should still fire the prior signature here. seed_matches_applied=0 means "
        "the replay loop is still doing exact-match, not generalization."
    )
    assert manifest2["stage5_stats"]["learned_structure_matches"] >= 1
    discourse_rows = json.loads((outdir2 / "discourse-wiring.json").read_text(encoding="utf-8"))
    assert discourse_rows[0]["counts"]["learned_structure"] >= 1
    # The fired records should carry both the run-2 signature and the matched run-1 prior.
    wiring = json.loads((outdir2 / "discourse-wiring.json").read_text(encoding="utf-8"))
    seed_records = []
    for row in wiring:
        for rec in row.get("records", []):
            if rec.get("hx/type") == "learned/structure-seed":
                seed_records.append(rec)
    assert seed_records, "no learned/structure-seed records emitted in run 2"
    matched = seed_records[0]["hx/content"]["matched_prior_signature"]
    assert matched == "we study <term> and <term>", (
        f"expected matched_prior_signature to be the run-1 backbone, got {matched!r}"
    )


def test_arxiv_legacy_theorem_aliases_are_normalized(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_legacy_alias_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, [])
    assert run.returncode == 0, (
        "superpod-job arxiv legacy alias run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["paper_hypergraph"]["normalized_papers"] == 1
    assert manifest["stage_status"]["paper_hypergraph"]["normalization_rewrites"] >= 2

    hypergraphs = json.loads((outdir / "paper-hypergraphs.json").read_text(encoding="utf-8"))
    claim_nodes = [n for n in hypergraphs[0]["nodes"] if n["type"] == "claim"]
    assert len(claim_nodes) == 1
    assert claim_nodes[0]["attrs"]["block_origin"] == "alias_expanded"
    assert "newtheorem alias thm->theorem" in claim_nodes[0]["attrs"]["source_cue"]
    geometry = json.loads((outdir / "geometry.json").read_text(encoding="utf-8"))
    assert geometry[0]["n_claims"] == 1
    assert manifest["stage9a_stats"]["geometry_source"] == "paper-hypergraphs"


def test_arxiv_paper_hg_eprint_dir_is_legacy_alias(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-hg-eprint-dir",
            "eprints",
            "--llm-stage5d-batch-size",
            "2",
            "--dry-run",
        ],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv dry-run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert f"--paper-eprint-dir {eprints.resolve()}" in run.stdout
    assert "--llm-stage5d-batch-size 2" in run.stdout
    assert "--paper-hg-eprint-dir" not in run.stdout


def test_arxiv_paper_eprint_dir_fails_when_no_sources_match(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    (eprints / "math__0102067v1.tex").unlink()
    (eprints / "unrelated.tex").write_text("$y=y$", encoding="utf-8")
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints"],
    )
    assert run.returncode != 0
    assert "no eprint filenames matched the arXiv entities" in run.stderr


def test_arxiv_moist_run_uses_paper_shape_prompt(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, ["--moist-run"])
    assert run.returncode == 0, (
        "superpod-job arxiv moist-run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    prompt_path = outdir / "moist-prompts" / "stage3-pattern-tagging.jsonl"
    first = json.loads(prompt_path.read_text(encoding="utf-8").splitlines()[0])
    prompt = first["prompt"]
    assert "mathematics paper-shape classifier" in prompt
    assert "math-strategy/" in prompt
    assert "math.stackexchange" not in prompt


def test_stage3_existing_arxiv_chunks_reparse_when_parser_version_changes(tmp_path: Path):
    root = Path(__file__).parent.parent
    module = _load_superpod_job_module(root)
    outdir = tmp_path / "arxiv-out"
    chunk_dir = outdir / "stage3-pattern-tags-chunks"
    chunk_dir.mkdir(parents=True)
    old_meta = {
        "version": 1,
        "total_pairs": 1,
        "chunks_per_shard": 1,
        "effective_chunks": 1,
        "max_new_tokens": 192,
    }
    (chunk_dir / "meta.json").write_text(
        json.dumps(old_meta, indent=2) + "\n",
        encoding="utf-8",
    )
    raw = (
        '{'
        '"family": "math-strategy/characterization-result",'
        '"leaf": "math-informal/structural-characterization",'
        '"family_confidence": 0.9,'
        '"leaf_confidence": 0.8,'
        '"rationale": "Classifies $(\\mathbb{T},\\mathsf{V})$-categories.",'
        '"collapsed": null'
        '}'
    )
    (chunk_dir / "chunk-000.json").write_text(
        json.dumps(
            [
                {
                    "entry_id": "arxiv-math/0102067v1",
                    "status": "failed",
                    "reason": "stage3-parse-error",
                    "error": "json-decode: Invalid \\\\escape",
                    "patterns": [],
                    "raw": raw,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FailIfCalledPool:
        def run_stage3(self, items, batch_size):
            raise AssertionError("Stage 3 LLM should not rerun existing chunks")

    pairs = [
        SimpleNamespace(
            question=SimpleNamespace(title="A toy paper", body_text="abstract"),
            answer=SimpleNamespace(body_text="abstract"),
        )
    ]
    results = module.run_stage3_pattern_tagging_chunked(
        pairs,
        ["arxiv-math/0102067v1"],
        outdir,
        pipe=None,
        tokenizer=None,
        batch_size=1,
        chunks_per_shard=1,
        llm_pool=FailIfCalledPool(),
        max_new_tokens=192,
    )

    assert results[0]["status"] == "ok"
    assert "\\mathbb" in results[0]["rationale"]
    meta = json.loads((chunk_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["parse_version"] == "arxiv-paper-shapes-v2"
    merged = json.loads((outdir / "pattern-tags.json").read_text(encoding="utf-8"))
    assert merged[0]["status"] == "ok"


def test_arxiv_paper_stage_resume_survives_missing_manifest_via_stage_sidecars(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    first = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert first.returncode == 0, (
        "initial superpod-job arxiv eprint run failed\n"
        f"stdout:\n{first.stdout}\n"
        f"stderr:\n{first.stderr}"
    )

    manifest_path = outdir / "manifest.json"
    manifest_path.unlink()
    assert not manifest_path.exists()
    assert (outdir / "stage-status" / "technique_ner.json").exists()
    assert (outdir / "stage-status" / "paper_hypergraph.json").exists()

    second = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert second.returncode == 0, (
        "resumed superpod-job arxiv eprint run failed\n"
        f"stdout:\n{second.stdout}\n"
        f"stderr:\n{second.stderr}"
    )
    assert "Reusing existing techniques.json" in second.stdout
    assert "Reusing existing paper-hypergraphs.json" in second.stdout

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage_status"]["technique_ner"]["resumed"] is True
    assert manifest["stage_status"]["paper_hypergraph"]["resumed"] is True
    assert manifest["stage_status"]["technique_ner"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage_status"]["paper_hypergraph"]["text_source_counts"]["eprint"] == 1

    manifest_path.unlink()

    third = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints"],
    )
    assert third.returncode == 0, (
        "repeat resumed superpod-job arxiv eprint run failed\n"
        f"stdout:\n{third.stdout}\n"
        f"stderr:\n{third.stderr}"
    )
    assert "Reusing existing techniques.json" in third.stdout
    assert "Reusing existing paper-hypergraphs.json" in third.stdout


def test_arxiv_paper_stage_resume_requires_stage_sidecar_provenance(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    first = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert first.returncode == 0, (
        "initial superpod-job arxiv eprint run failed\n"
        f"stdout:\n{first.stdout}\n"
        f"stderr:\n{first.stderr}"
    )

    (outdir / "manifest.json").unlink()
    (outdir / "stage-status" / "technique_ner.json").unlink()

    second = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert second.returncode != 0
    assert "Stage 5c cannot safely resume existing techniques.json" in second.stderr
    assert "Move" in second.stderr
    assert "fresh provenance" in second.stderr


def test_arxiv_paper_stage5d_chunk_resume_reuses_completed_chunks(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_arxiv_fixture(input_dir, count=33)
    outdir = tmp_path / "arxiv-out"

    first = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert first.returncode == 0, (
        "initial superpod-job arxiv stage5d run failed\n"
        f"stdout:\n{first.stdout}\n"
        f"stderr:\n{first.stderr}"
    )

    chunk_dir = outdir / "stage5d-paper-hypergraph-chunks"
    assert (chunk_dir / "chunk-000.json").exists()
    assert (chunk_dir / "chunk-001.json").exists()

    (outdir / "manifest.json").unlink()
    (outdir / "stage-status" / "paper_hypergraph.json").unlink()
    (outdir / "paper-hypergraphs.json").unlink()
    (chunk_dir / "chunk-001.json").unlink()

    second = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert second.returncode == 0, (
        "resumed superpod-job arxiv stage5d run failed\n"
        f"stdout:\n{second.stdout}\n"
        f"stderr:\n{second.stderr}"
    )
    assert "Stage 5d chunking: 2 chunks" in second.stdout
    assert "chunk 1/2 exists (32 papers), skipping" in second.stdout
    assert "chunk 2/2:" in second.stdout
    assert "chunk 2/2 written: chunk-001.json" in second.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["paper_hypergraph"]["resumed_from_chunks"] is True
    assert manifest["stage_status"]["paper_hypergraph"]["resumed_chunks"] == 1
    assert manifest["stage_status"]["paper_hypergraph"]["resumed_papers"] == 32


def test_arxiv_paper_stage5d_resume_keeps_existing_chunk_geometry_when_default_changes(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_arxiv_fixture(input_dir, count=33)
    outdir = tmp_path / "arxiv-out"

    first = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-eprint-dir",
            "eprints",
            "--llm-stage5d-checkpoint-chunk-size",
            "32",
        ],
    )
    assert first.returncode == 0, (
        "initial superpod-job arxiv stage5d geometry run failed\n"
        f"stdout:\n{first.stdout}\n"
        f"stderr:\n{first.stderr}"
    )

    chunk_dir = outdir / "stage5d-paper-hypergraph-chunks"
    (outdir / "manifest.json").unlink()
    (outdir / "stage-status" / "paper_hypergraph.json").unlink()
    (outdir / "paper-hypergraphs.json").unlink()
    (chunk_dir / "chunk-001.json").unlink()

    second = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints"],
    )
    assert second.returncode == 0, (
        "resumed superpod-job arxiv stage5d geometry-preserving run failed\n"
        f"stdout:\n{second.stdout}\n"
        f"stderr:\n{second.stderr}"
    )
    assert "Reusing existing Stage 5d checkpoint geometry (32 papers/chunk)" in second.stdout
    assert "Stage 5d chunking: 2 chunks" in second.stdout
    assert "chunk 1/2 exists (32 papers), skipping" in second.stdout
