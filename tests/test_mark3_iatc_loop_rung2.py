import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mark3_iatc_loop", ROOT / "scripts" / "mark3_iatc_loop.py"
)
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


GRAPH = """{:paper/id "9999.0001"
 :passage/id "9999.0001:p"
 :nodes [{:id :p :kind :claim :text "premise" :source {:lines [1 1]}}
         {:id :c :kind :claim :text "conclusion" :source {:lines [2 2]}}]
 :edges [{:id :e :kind :infer :premise [:p] :conclusion :c
          :warrant {:kind :missing-warrant :wanted :lemma}
          :source {:lines [1 2]}}]
 :holes [{:kind :missing-warrant :edge :e :wanted :lemma}]}"""


def write_candidate(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": getattr(loop, "CANDIDATE_SCHEMA", "iatc-candidate/v2-enriched"),
                "paper-id": "9999.0001",
                "window-lines": [1, 2],
                "source-window": "1 premise\n2 conclusion",
                "binder-context": [],
                "enrichment": [],
            }
        )
    )


def args(candidates: Path, out: Path, *, rung2_gate: bool = False):
    return argparse.Namespace(
        candidates=str(candidates),
        out=str(out),
        backend="stub",
        model="stub",
        shots=0,
        rung2_gate=rung2_gate,
    )


def test_rung2_soft_default_emits_graph_and_profile(monkeypatch, tmp_path):
    cands = tmp_path / "candidates"
    cands.mkdir()
    write_candidate(cands / "9999.0001.candidate.json")
    out = tmp_path / "out"

    monkeypatch.setattr(loop, "load_seeds", lambda _n: "")
    monkeypatch.setattr(loop, "call_stub", lambda _prompt, _cand, _attempt: GRAPH)
    monkeypatch.setattr(loop, "candidate_check", lambda _edn, _cand: (True, "ok"))
    monkeypatch.setattr(loop, "gate_one", lambda _path: (True, "ok"))

    def fail_rung2(_graph_path, report_path, *, gate):
        report_path.write_text("{:schema :futon6.iatc-semcheck.v1 :pass false :profile {}}")
        return False, "rung2-soft-fail"

    monkeypatch.setattr(loop, "run_rung2", fail_rung2)

    assert loop.run(args(cands, out)) == 0
    assert (out / "9999.0001.edn").exists()
    assert (out / "9999.0001.rung2.edn").exists()


def test_rung2_gate_retries_until_semcheck_passes(monkeypatch, tmp_path):
    cands = tmp_path / "candidates"
    cands.mkdir()
    write_candidate(cands / "9999.0001.candidate.json")
    out = tmp_path / "out"
    calls = {"n": 0}

    monkeypatch.setattr(loop, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(loop, "load_seeds", lambda _n: "")
    monkeypatch.setattr(loop, "call_stub", lambda _prompt, _cand, _attempt: GRAPH)
    monkeypatch.setattr(loop, "candidate_check", lambda _edn, _cand: (True, "ok"))
    monkeypatch.setattr(loop, "gate_one", lambda _path: (True, "ok"))

    def gated_rung2(_graph_path, report_path, *, gate):
        calls["n"] += 1
        if gate and calls["n"] == 1:
            report_path.write_text("{:schema :futon6.iatc-semcheck.v1 :pass false}")
            return False, "rung2-hard-fail"
        report_path.write_text("{:schema :futon6.iatc-semcheck.v1 :pass true}")
        return True, "rung2-pass"

    monkeypatch.setattr(loop, "run_rung2", gated_rung2)

    assert loop.run(args(cands, out, rung2_gate=True)) == 0
    assert calls["n"] == 3
    assert (out / "9999.0001.edn").exists()
    assert (out / "9999.0001.rung2.edn").exists()
