"""S4 and S7 send the model several items at once, as S3 does, and give the serial result.

S3 keeps FUTON6_CONCURRENCY proofs in flight. S4 (the expository loop) and S7 (CLean typing)
sent one request at a time, so eight GPUs hardly sped them up: in run mark7rhgrh100-20260928
S4's loop took about 4 s a region, 3 h for 2,750 regions. Both now keep FUTON6_CONCURRENCY
items in flight. S4 runs each region in a pool thread and updates what they share (the counts,
the ledger, the log lines) under one lock; S7 sends only its model calls through the pool and
gates and records each answer in the main thread, in input order.

Each test runs a stage serially and at concurrency 4 on the same input, with stand-ins for the
model and for what runs Babashka (the gates, S4's vocabulary reader), and requires the same
outputs, attempt files, provenance, ledger, metric records and summary, and model calls that
really overlapped.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_box_typing as s7  # noqa: E402
import mark3_expository_loop as loop  # noqa: E402
import mark3_extract_candidates as iatc_candidates  # noqa: E402
import region_units  # noqa: E402
import stage_accounting as accounting  # noqa: E402

MODEL_SECONDS = 0.2   # long enough that requests sent together are in flight together


class InFlight:
    """Counts the stand-in model calls under way at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    @contextlib.contextmanager
    def call(self, seconds: float = MODEL_SECONDS):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(seconds)
            yield
        finally:
            with self.lock:
                self.now -= 1


def files(directory: Path) -> dict[str, bytes]:
    """Every file under a directory, by its relative path."""
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def ledger(directory: Path) -> dict:
    """The accounting document a stage wrote, with its items keyed by id. Items are written in
    the order they finish, which is all that concurrency may change; "updated" is a time."""
    (path,) = directory.glob("*.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("updated")
    document["items"] = {item["id"]: item for item in document["items"]}
    return document


def split(printed: str, summary: str) -> tuple[list[str], str]:
    """(the per-item log lines, sorted, since items are logged as they finish; the summary)."""
    head, _, tail = printed.partition(summary)
    return sorted(line for line in head.splitlines() if line.startswith("  ")), summary + tail


# ---------------------------------------------------------------- S4: mark3_expository_loop
SOURCE = "\n".join([
    "Let $G$ be a finite group acting on a set $X$.",
    "The orbit of a point is the set of its images under $G$.",
    "Every orbit is finite, since $G$ is finite.",
    "The stabiliser of a point is a subgroup of $G$.",
    "Its index equals the size of the orbit.",
    "Hence the size of each orbit divides the order of $G$.",
    "We use this count in the next section.",
    "{\\bf Proof}. Notice that",
])
GATE_REJECTS = "p:r2:L3-3"     # the stand-in argcheck rejects this region
MODEL_ERROR = "q:r2:L6-6"      # the stand-in endpoint fails on this one
CARRIED = "p:r4:L2-3"          # accepted by an earlier invocation of the run
# A region without units is refused at selection (mark3_extract_expository_candidates), so the
# loop never sees one.
REGIONS = ["p:r1:L1-2", GATE_REJECTS, "q:r1:L4-5", MODEL_ERROR, "q:r3:L7-7", CARRIED]
# Three kinds of holes/excursions/expository-superpod-vocab.edn, which expository_json.vocabulary()
# reads with bb. Which kinds they are does not bear on the comparison.
VOCABULARY = {"auxiliary-construction": "auxiliary", "computes-invariant/calculation": "computed-value",
              "connection": "other-structure"}


def region(passage: str) -> dict:
    """An S4 candidate as S1 writes it (mark3_extract_expository_candidates.extract)."""
    paper, region_id, span = passage.split(":")
    lo, hi = (int(n) for n in span[1:].split("-"))
    starts = iatc_candidates.line_starts(SOURCE)
    return {"schema": loop.CANDIDATE_SCHEMA, "paper-id": paper, "passage-id": passage,
            "region-id": region_id, "region-type": "paragraph", "window-lines": [lo, hi],
            "source-window": "\n".join(SOURCE.split("\n")[lo - 1:hi]),
            "units": region_units.units_for(SOURCE, starts, lo, hi),
            "enrichment": [], "vocab-path": loop.VOCAB.relative_to(loop.REPO).as_posix()}


class ExpositoryRegionsAtOnce(unittest.TestCase):
    def run_stage(self, base: Path, concurrency: int) -> tuple[int, str, int]:
        candidates, out, books = base / "candidates", base / "out", base / "accounting"
        candidates.mkdir(parents=True)
        books.mkdir()
        for passage in REGIONS:
            candidate = region(passage)
            name = loop.safe_output_name(candidate)[:-len(".edn")]
            (candidates / f"{name}.candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
        model, stub = InFlight(), loop.call_stub

        def call(prompt, candidate, kinds):
            with model.call():
                if candidate["passage-id"] == MODEL_ERROR:
                    raise loop.ModelCallError("HTTP 500: stand-in endpoint failure")
                return stub(prompt, candidate, kinds)

        def gate(path):
            if Path(path).name == loop.safe_output_name({"passage-id": GATE_REJECTS}):
                return False, "G3: stand-in argcheck rejection"
            return True, "ok"

        args = argparse.Namespace(candidates=str(candidates), out=str(out), backend="stub",
                                  model="stub", concurrency=concurrency, run_dir=None,
                                  run_id="t", corpus_id="t")
        printed = io.StringIO()
        with mock.patch.dict(os.environ, {accounting.DIR_ENV: str(books), "FUTON6_RUN_DIR": str(base)}), \
                mock.patch.object(loop.expository_json, "vocabulary", lambda: VOCABULARY), \
                mock.patch.object(loop, "call_stub", call), mock.patch.object(loop, "gate_one", gate), \
                contextlib.redirect_stdout(printed):
            for name in ("RUN_ID", accounting.INVOCATION_ENV):
                os.environ.pop(name, None)
            carried = out / loop.safe_output_name({"passage-id": CARRIED})
            accounting.publish_accepted(out, CARRIED, carried, b"{:carried true}\n", {"path": "earlier"})
            code = loop.run(args)
        return code, printed.getvalue().replace(str(base), "<base>"), model.peak

    def test_regions_at_once_give_the_serial_result(self):
        with tempfile.TemporaryDirectory() as d:
            serial, together = Path(d) / "serial", Path(d) / "together"
            code, printed, peak = self.run_stage(serial, 1)
            code4, printed4, peak4 = self.run_stage(together, 4)

            self.assertEqual((code, code4), (0, 0))
            self.assertEqual(peak, 1)
            self.assertGreater(peak4, 1, "no two regions were at the model at once")
            self.assertNotIn("at concurrency", printed)
            self.assertIn("== 5 expository region(s) at concurrency 4 ==", printed4)
            self.assertEqual(files(serial / "out"), files(together / "out"))
            self.assertEqual(ledger(serial / "accounting"), ledger(together / "accounting"))
            lines, summary = split(printed, "\nexpository-loop:")
            self.assertEqual((lines, summary), split(printed4, "\nexpository-loop:"))
            self.assertIn("accepted 4 (carried 1) · rejected 1 · errored 1 of 6", summary)
            items = ledger(serial / "accounting")["items"]
            self.assertEqual({item: entry["status"] for item, entry in items.items()},
                             {"p:r1:L1-2": "accepted", GATE_REJECTS: "rejected", "q:r1:L4-5": "accepted",
                              MODEL_ERROR: "errored", "q:r3:L7-7": "accepted", CARRIED: "accepted"})


# ---------------------------------------------------------------- S7: clean_box_typing
GRAPH = """{:paper/id "PID" :passage/id "PID:proof1" :source {:lines [1 3] :kind :proof}
 :nodes [{:id :hyp :kind :object :text "Let G be the group of PID." :source {:lines [1 1]}}
         {:id :mid :kind :claim :text "G acts on itself by translation." :source {:lines [2 2]}}
         {:id :goal :kind :claim :text "G embeds in a symmetric group." :source {:lines [3 3]}}]
 :edges [{:id :e1 :kind :infer :relation :implies :premise [:hyp] :conclusion :mid
          :warrant {:kind :construction :text "translation is a bijection"} :source {:lines [1 2]}}
         {:id :e2 :kind :infer :relation :implies :premise [:mid] :conclusion :goal
          :warrant {:kind :known-result :text "Cayley"} :source {:lines [2 3]}}]
 :holes []}
"""
STALE = "g0"          # a final without acceptance provenance: refused before any typing
GRAPHS = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]
# Each pair ends in the other order when typed at once, so the summary's lists are only in
# graph order if the stage puts them back.
ARGCHECK_REJECTS = {"g2": 0.6, "g4": 0.0}  # graph -> seconds the stand-in argcheck takes
TYPING_FAILS = {"g3": 0.6, "g5": 0.0}      # graph -> seconds the stand-in model takes
MALFORMED = "g6"      # unreadable EDN: rejected when loaded


class GraphTypingAtOnce(unittest.TestCase):
    def run_stage(self, base: Path, concurrency: int) -> tuple[int, str, int]:
        graphs, out, books, run = base / "graphs", base / "out", base / "accounting", base / "run"
        graphs.mkdir(parents=True)
        books.mkdir()
        (graphs / f"{STALE}.edn").write_text(GRAPH.replace("PID", STALE), encoding="utf-8")
        model = InFlight()
        real_run, real_system = subprocess.run, os.system

        def query(endpoint, model_name, prompt, sk, methods):
            pid = prompt.split("(arXiv ", 1)[1].split(")", 1)[0]
            with model.call(TYPING_FAILS.get(pid, MODEL_SECONDS)):
                if pid in TYPING_FAILS:
                    raise s7.TypingCallError("stand-in endpoint returned non-JSON")
                return s7.stub_typing(sk)

        def argcheck(argv, *args, **kwargs):
            if argv[:2] == ["bb", "scripts/clean_vocab_gate.bb"]:
                return subprocess.CompletedProcess(argv, 0, f"PASS {Path(argv[2]).name}\n", "")
            if argv[:2] != ["bb", "scripts/clean_argcheck.bb"]:
                return real_run(argv, *args, **kwargs)
            name = Path(argv[2]).name
            pid = name[:-len(".clean.edn")]
            if pid not in ARGCHECK_REJECTS:
                return subprocess.CompletedProcess(argv, 0, f"PASS {name}\n", "")
            time.sleep(ARGCHECK_REJECTS[pid])
            return subprocess.CompletedProcess(argv, 1, f"FAIL {name}\nG7: stand-in cycle in {pid}\n", "")

        def system(command):
            return 0 if "clean_vocab_gate.bb" in command else real_system(command)

        argv = ["clean_box_typing.py", "--graphs", str(graphs), "--out", str(out),
                "--concurrency", str(concurrency), "--run-dir", str(run), "--run-id", "t", "--corpus-id", "t"]
        printed = io.StringIO()
        with mock.patch.dict(os.environ, {accounting.DIR_ENV: str(books), "FUTON6_RUN_DIR": str(base)}), \
                mock.patch.object(sys, "argv", argv), mock.patch.object(s7, "wait_for_server", lambda _: None), \
                mock.patch.object(s7, "query_model", query), \
                mock.patch.object(s7.subprocess, "run", argcheck), mock.patch.object(s7.os, "system", system), \
                contextlib.redirect_stdout(printed):
            os.environ.pop(accounting.INVOCATION_ENV, None)
            for pid in GRAPHS:
                text = "{:nodes [" if pid == MALFORMED else GRAPH.replace("PID", pid)
                accounting.publish_accepted(graphs, pid, graphs / f"{pid}.edn", text.encode("utf-8"),
                                            {"path": "S3"})
            with self.assertRaises(SystemExit) as exited:
                s7.main()
        return exited.exception.code, printed.getvalue().replace(str(base), "<base>"), model.peak

    def test_graphs_typed_at_once_give_the_serial_result(self):
        with tempfile.TemporaryDirectory() as d:
            serial, together = Path(d) / "serial", Path(d) / "together"
            code, printed, peak = self.run_stage(serial, 1)
            code4, printed4, peak4 = self.run_stage(together, 4)

            # Typing failures and the refused stale final are logged, and S7 goes on, in both.
            self.assertEqual((code, code4), (0, 0))
            self.assertEqual(peak, 1)
            self.assertGreater(peak4, 1, "no two graphs were at the model at once")
            self.assertNotIn("at concurrency", printed)
            self.assertIn("== 7 CLean typing item(s) at concurrency 4 ==", printed4)
            self.assertEqual(files(serial / "out"), files(together / "out"))
            self.assertEqual(ledger(serial / "accounting"), ledger(together / "accounting"))
            metrics = [sorted((base / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
                       for base in (serial, together)]
            self.assertEqual(metrics[0], metrics[1])
            self.assertEqual(len(metrics[0]), 3)
            lines, summary = split(printed, "\ntyped ")
            self.assertEqual((lines, summary), split(printed4, "\ntyped "))
            self.assertIn("typed 3 / rejected 3 (gate) / failed 3 (typing)", summary)
            self.assertIn("typing-failed: ['g0.edn', 'g3', 'g5']", summary)
            self.assertLess(summary.index("stand-in cycle in g2"), summary.index("stand-in cycle in g4"))


if __name__ == "__main__":
    unittest.main()
