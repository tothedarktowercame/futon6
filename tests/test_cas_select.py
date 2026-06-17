import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cas_select", ROOT / "scripts" / "cas_select.py")
cas = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cas
SPEC.loader.exec_module(cas)

FIXTURES = ROOT / "tests" / "fixtures" / "cas-select"


def load_fixture(paper_id):
    steps = json.loads((FIXTURES / f"{paper_id}.steps.json").read_text())
    oracle_rows = json.loads((FIXTURES / f"{paper_id}.oracle.json").read_text())["matches"]
    oracle = {row["step"]: row for row in oracle_rows}
    return steps, oracle_rows, oracle


def select_with_patterns(paper_id, patterns):
    steps, _, oracle = load_fixture(paper_id)
    return cas.select_proof(steps, patterns, backend="stub", oracle=oracle, k=4)


def test_happy_path_stub_reproduces_ground_truth_topology_and_sorry():
    patterns = cas.load_patterns()
    for paper_id in ["a93J05", "a96J01", "b97J01", "a96J04"]:
        _, oracle_rows, _ = load_fixture(paper_id)
        result = select_with_patterns(paper_id, patterns)

        assert result["topology"] == [row["pattern"] for row in oracle_rows]
        assert result["induce_queue"] == []
        assert [
            (row["step"], row["pattern"])
            for row in result["sorry"]
            if row["kind"] == "declared"
        ] == [
            (row["step"], row["pattern"])
            for row in oracle_rows
            if row.get("declares_sorry", True)
        ]
        assert all(row["obligation"] for row in result["sorry"])


def test_trigger_path_premint_pool_enqueues_exactly_three_minted_steps():
    excluded = {
        "separate-into-independent-pieces",
        "count-over-a-decomposition",
        "epsilon-of-room",
    }
    allowed = {p.stem for p in cas.DEFAULT_LIBRARY.glob("*.flexiarg")} - excluded
    patterns = cas.load_patterns(allowed=allowed)

    queued = []
    for paper_id in ["a93J05", "a96J01", "b97J01", "a96J04"]:
        result = select_with_patterns(paper_id, patterns)
        queued.extend((paper_id, row["step"]) for row in result["induce_queue"])

    assert queued == [
        ("a96J01", "s3"),
        ("b97J01", "s4"),
        ("a96J04", "s5"),
    ]


def test_tier0_retrieval_recall_is_honest():
    """Tier-0 hotword retrieval has a real recall ceiling (prose<->hotword vocabulary
    mismatch). Documented, not hidden: recall@4 = 16/22; even at full-pool k three steps
    have ZERO lexical overlap with their oracle pattern and are unreachable by hotword
    alone. Those steps need a semantic/embedding retriever (or LLM-side retrieval) — see
    CAS-SEL-3b follow-on. This test pins the honest number so a regression can't quietly
    inflate it again."""
    patterns = cas.load_patterns()
    hit = miss = 0
    full_pool_misses = set()
    for paper_id in ["a93J05", "a96J01", "b97J01", "a96J04"]:
        steps, _, oracle = load_fixture(paper_id)
        for step in steps["steps"]:
            want = oracle[step["id"]]["pattern"]
            top4 = {row["pattern"] for row in cas.retrieve(step["text"], patterns, k=4)}
            if want in top4:
                hit += 1
            else:
                miss += 1
            full = {row["pattern"] for row in cas.retrieve(step["text"], patterns, k=len(patterns))}
            if want not in full:
                full_pool_misses.add(f"{paper_id}/{step['id']}")
    assert (hit, miss) == (16, 6), f"recall@4 drifted: {hit} hit / {miss} miss"
    # the irreducible (zero-lexical-overlap) ceiling — needs a non-hotword retriever:
    assert full_pool_misses == {"a93J05/s3", "a96J01/s2", "b97J01/s6"}


def test_assemble_reads_conclusions_and_however_from_flexiarg():
    patterns = cas.load_patterns()
    matches = [
        {
            "step": "s1",
            "pattern": "reduce-to-known-result",
            "slot": "EVT",
            "score": 1.0,
            "tier1": "verified",
            "declares_sorry": True,
        }
    ]

    result = cas.assemble("toy", matches, [], patterns)

    assert "Identify the candidate theorem" in result["wiring"][0]["conclusion"]
    assert "wrong theorem" in result["sorry"][0]["obligation"]
    assert result["checks"] == [
        {"step": "s1", "pattern": "reduce-to-known-result", "fires": ["R2c-warrant"]}
    ]


def test_tier0_retrieve_is_model_free(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("Tier 0 should not call OpenAI")

    monkeypatch.setattr(cas, "call_openai", boom)
    patterns = cas.load_patterns()
    candidates = cas.retrieve("Invoke a known theorem and apply the standard lemma.", patterns, k=4)

    assert candidates
    assert any(row["pattern"] == "reduce-to-known-result" for row in candidates)
