from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mark4_iatc_concurrent", ROOT / "scripts" / "mark4_iatc_concurrent.py"
)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


def graph_for(pid: str) -> str:
    return f'''{{:paper/id "{pid}"
 :passage/id "{pid}:p"
 :nodes [{{:id :p :kind :claim :text "premise" :source {{:lines [1 1]}}}}
         {{:id :c :kind :claim :text "conclusion" :source {{:lines [2 2]}}}}]
 :edges [{{:id :e :kind :infer :premise [:p] :conclusion :c
          :warrant {{:kind :missing-warrant :wanted :lemma}}
          :source {{:lines [1 2]}}}}]
 :holes [{{:kind :missing-warrant :edge :e :wanted :lemma}}]}}'''


def write_candidate(path: Path, pid: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": getattr(driver.loop, "CANDIDATE_SCHEMA", "iatc-candidate/v2-enriched"),
                "paper-id": pid,
                "window-lines": [1, 2],
                "source-window": "1 premise\n2 conclusion",
                "binder-context": [],
                "enrichment": [],
            }
        )
    )


def args(candidates: Path, out: Path, *, concurrency: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        candidates=str(candidates),
        out=str(out),
        backend="stub",
        model="stub",
        shots=0,
        rung2_gate=False,
        concurrency=concurrency,
    )


def install_stubs(monkeypatch, *, delay: float = 0.0) -> None:
    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(driver.loop, "load_seeds", lambda _n: "")

    def call_stub(_prompt, cand, _attempt):
        if delay:
            time.sleep(delay)
        return graph_for(cand["paper-id"])

    monkeypatch.setattr(driver.loop, "call_stub", call_stub)
    monkeypatch.setattr(driver.loop, "candidate_check", lambda _edn, _cand: (True, "ok"))
    monkeypatch.setattr(driver.loop, "gate_one", lambda _path: (True, "ok"))

    def rung2(_graph_path, report_path, *, gate):
        report_path.write_text("{:schema :futon6.iatc-semcheck.v1 :pass true}")
        return True, "rung2-pass"

    monkeypatch.setattr(driver.loop, "run_rung2", rung2)
    monkeypatch.setattr(driver, "repair_attempt", lambda _path: None)
    monkeypatch.setattr(driver, "batch_substance_gate", lambda _paths, _outdir: (True, "ok"))
    monkeypatch.setattr(driver.loop.subprocess, "run", lambda *_, **__: Completed())


def read_outputs(out: Path) -> dict[str, str]:
    return {
        path.stem: path.read_text()
        for path in sorted(out.glob("*.edn"))
        if not path.name.endswith(".rung2.edn")
    }


def run_sequential_per_paper(candidates: Path, out: Path) -> dict[str, str]:
    for cf in sorted(candidates.glob("*.candidate.json")):
        one = out / f"cand-{cf.stem}"
        one.mkdir(parents=True)
        target = one / cf.name
        target.write_text(cf.read_text())
        assert driver.loop.run(args(one, out, concurrency=1)) == 0
    return read_outputs(out)


def test_concurrent_stub_outputs_match_sequential_and_bound_in_flight(monkeypatch, tmp_path):
    install_stubs(monkeypatch, delay=0.02)
    cands = tmp_path / "candidates"
    cands.mkdir()
    for i in range(6):
        write_candidate(cands / f"9999.000{i}.candidate.json", f"9999.000{i}")

    sequential = run_sequential_per_paper(cands, tmp_path / "sequential")
    meter = driver.InFlightMeter()
    concurrent_out = tmp_path / "concurrent"

    assert driver.run(args(cands, concurrent_out, concurrency=2), meter=meter) == 0

    assert read_outputs(concurrent_out) == sequential
    assert meter.max_seen <= 2


def test_concurrent_stub_is_order_independent(monkeypatch, tmp_path):
    install_stubs(monkeypatch, delay=0.01)
    forward = tmp_path / "forward"
    reverse = tmp_path / "reverse"
    forward.mkdir()
    reverse.mkdir()
    pids = [f"9999.10{i}" for i in range(5)]
    for pid in pids:
        write_candidate(forward / f"{pid}.candidate.json", pid)
    for pid in reversed(pids):
        write_candidate(reverse / f"{pid}.candidate.json", pid)

    assert driver.run(args(forward, tmp_path / "out-forward", concurrency=3)) == 0
    assert driver.run(args(reverse, tmp_path / "out-reverse", concurrency=3)) == 0

    assert read_outputs(tmp_path / "out-forward") == read_outputs(tmp_path / "out-reverse")


def test_concurrent_driver_resumes_completed_paper(monkeypatch, tmp_path):
    install_stubs(monkeypatch)
    cands = tmp_path / "candidates"
    cands.mkdir()
    write_candidate(cands / "9999.2000.candidate.json", "9999.2000")
    out = tmp_path / "out"
    out.mkdir()
    (out / "9999.2000.edn").write_text("existing graph")
    (out / "9999.2000.rung2.edn").write_text("existing rung2")

    assert driver.run(args(cands, out, concurrency=2)) == 0
    assert (out / "9999.2000.edn").read_text() == "existing graph"
