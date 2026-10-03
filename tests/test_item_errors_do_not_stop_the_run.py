"""One item's error never stops a run: the item is recorded, and the stage goes on.

A Mark7 run is judged item by item. A stage records each paper, proof or graph it could not
handle as rejected or errored, and the runner weighs those refusals against the run's floor
(stage_accounting.blocking). Several stages exited 1 over any single refusal instead, and so
stopped the whole run over one item: S6 over 0810.0425, whose acknowledgements are typeset as a
proof (run mark7rhgrh100-20260928); S7 over any proof that failed to type or an off-vocabulary
CLean; S1, S1b and the S3 and S4 extractions over one paper without a source or marks.

The later stages (S5 and S7-S12) and the S3 and S4 measurements keep no ledger. They report an
item they cannot read and leave it out, and fail only when they could handle no item at all.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_box_typing as s7  # noqa: E402
import emit_marks  # noqa: E402
import rung3_technique as r3  # noqa: E402
import stage_accounting as accounting  # noqa: E402

MISSING = "0000.00000"   # a paper nothing is held for


def run_script(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT,
                          capture_output=True, text=True, env={**os.environ, **env})


class PapersWithoutInputs(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.marks = self.base / "marks"
        self.marks.mkdir()
        self.ids = self.base / "ids.txt"

    def test_s1_records_a_paper_without_an_eprint_as_its_own_failure(self):
        empty = self.base / "eprints"
        empty.mkdir()
        with mock.patch.object(emit_marks.dpv, "EPRINTS", empty):
            result = emit_marks.mark_paper(MISSING, out=str(self.marks), coverage=False)
        self.assertTrue(result["error"].startswith("EprintNotFound: no eprint for"), result)

    def test_s1b_lists_a_paper_without_marks_and_goes_on(self):
        (self.marks / "fable-1111.0001-dp-emacs.json").write_text(
            json.dumps({"paper": "1111.0001", "text": "Let $x$ be a real number.", "marks": []}))
        strategies = ["--marks", str(self.marks), "--out", str(self.base / "strategies")]
        self.ids.write_text(f"1111.0001\n{MISSING}\n")
        done = run_script("markup_strategies.py", "--list", str(self.ids), *strategies, env={})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(json.loads(done.stdout.strip().splitlines()[-1])["missing"], [MISSING])
        # When no paper resolves, it is the stage that failed.
        self.ids.write_text(f"{MISSING}\n")
        done = run_script("markup_strategies.py", "--list", str(self.ids), *strategies, env={})
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)

    def extraction(self, script: str, stage: str, *args: str) -> None:
        self.ids.write_text(f"{MISSING}\n")
        books = self.base / f"accounting-{stage}"
        done = run_script(script, "--list", str(self.ids), "--out", str(self.base / f"out-{stage}"), *args,
                          env={"FUTON6_MARKS": str(self.marks), accounting.DIR_ENV: str(books)})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        items = accounting.load(books, stage, "extract")["items"]
        self.assertEqual([(item["id"], item["status"]) for item in items], [(MISSING, "errored")])

    def test_s3_extraction_records_a_paper_without_marks_and_goes_on(self):
        self.extraction("mark3_extract_candidates.py", "S3", "--all-proofs")

    def test_s4_extraction_records_a_paper_without_marks_and_goes_on(self):
        self.extraction("mark3_extract_expository_candidates.py", "S4")


GRAPH = """{:paper/id "PID" :passage/id "PID:proof1" :source {:lines [1 2] :kind :proof}
 :nodes [{:id :hyp :kind :object :text "Let G be the group of PID." :source {:lines [1 1]}}
         {:id :goal :kind :claim :text "G acts on itself by translation." :source {:lines [2 2]}}]
 :edges [{:id :e1 :kind :infer :relation :implies :premise [:hyp] :conclusion :goal
          :warrant {:kind :construction :text "translation is a bijection"} :source {:lines [1 2]}}]
 :holes []}
"""


class ProofsThatFailInS7(unittest.TestCase):
    """S7 in process, with stand-ins for the model and for the Babashka gates."""

    def run_s7(self, base: Path, *, typing_fails=()):
        graphs, out, books = base / "graphs", base / "clean", base / "accounting"
        graphs.mkdir()
        out.mkdir()
        real_run, real_system = subprocess.run, os.system

        def query(endpoint, model, prompt, sk, methods):
            if any(f"(arXiv {pid})" in prompt for pid in typing_fails):
                raise s7.TypingCallError("stand-in endpoint returned non-JSON")
            return s7.stub_typing(sk)

        def gates(argv, *args, **kwargs):
            if argv[:1] != ["bb"]:
                return real_run(argv, *args, **kwargs)
            name = Path(argv[2]).name
            return subprocess.CompletedProcess(argv, 0, f"PASS {name}\n", "")

        def system(command):
            return 0 if "clean_vocab_gate.bb" in command else real_system(command)

        argv = ["clean_box_typing.py", "--graphs", str(graphs), "--out", str(out)]
        printed = io.StringIO()
        with mock.patch.dict(os.environ, {accounting.DIR_ENV: str(books)}), mock.patch.object(sys, "argv", argv), \
                mock.patch.object(s7, "wait_for_server", lambda _: None), mock.patch.object(s7, "query_model", query), \
                mock.patch.object(s7.subprocess, "run", gates), mock.patch.object(s7.os, "system", system), \
                contextlib.redirect_stdout(printed):
            os.environ.pop(accounting.INVOCATION_ENV, None)
            for pid in ("p1", "p2", "p3"):
                accounting.publish_accepted(graphs, pid, graphs / f"{pid}.edn",
                                            GRAPH.replace("PID", pid).encode("utf-8"), {"path": "S3"})
            with self.assertRaises(SystemExit) as exited:
                s7.main()
        items = {item["id"]: item for item in accounting.load(books, "S7", "typing")["items"]}
        return exited.exception.code, items, printed.getvalue(), sorted(p.name for p in out.iterdir())

    def test_a_proof_that_fails_to_type_is_an_errored_item(self):
        with tempfile.TemporaryDirectory() as d:
            code, items, printed, _ = self.run_s7(Path(d), typing_fails={"p2"})
        self.assertEqual(code, 0, printed)
        self.assertEqual({item: entry["status"] for item, entry in items.items()},
                         {"p1": "accepted", "p2": "errored", "p3": "accepted"})


def graph_file(directory: Path, pid: str) -> Path:
    path = directory / f"{pid}__p0.edn"
    path.write_text(GRAPH.replace("PID", pid), encoding="utf-8")
    return path


def clean_text(pid: str) -> str:
    """A typed CLean of GRAPH, made the way S7 makes one (with the stub typing)."""
    from clean_macro_fix import derive_macro
    with tempfile.TemporaryDirectory() as d:
        nodes, edges = s7.itc.load_graph(graph_file(Path(d), pid))
    skeleton = s7.itc.build_skeleton(nodes, edges)
    typing = s7.stub_typing(skeleton)
    typing["_macro"] = derive_macro([typing[b["id"]] for b in skeleton["boxes"]])
    return s7.itc.render_edn(pid, s7.itc.build_skeleton(nodes, edges, typing), 0) + "\n"


class LaterStagesLeaveOutWhatTheyCannotRead(unittest.TestCase):
    """S5 and S7-S12 keep no ledger: an item they cannot read is reported and left out, and only a
    stage that could handle no item at all fails."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)

    def folder(self, name: str) -> Path:
        path = self.base / name
        path.mkdir()
        return path

    def cleans(self, *pids: str, unreadable=()) -> Path:
        directory = self.folder("clean")
        for pid in pids:
            (directory / f"{pid}.clean.edn").write_text(clean_text(pid), encoding="utf-8")
        for pid in unreadable:
            (directory / f"{pid}.clean.edn").write_text("{:proof", encoding="utf-8")
        return directory

    def test_a_missing_eprint_is_an_exception_and_the_command_line_still_refuses_in_one_line(self):
        empty = self.folder("eprints")
        with mock.patch.object(emit_marks.dpv, "EPRINTS", empty):
            with self.assertRaises(FileNotFoundError):     # which every `except Exception` catches
                emit_marks.dpv.build(MISSING)
        done = run_script("dp_paper_view.py", MISSING, env={"FUTON6_EPRINTS": str(empty)})
        self.assertEqual(done.returncode, 1)
        self.assertIn(f"no eprint for {MISSING}", done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_s5_segmentation_leaves_out_a_graph_it_cannot_read(self):
        graphs, steps = self.folder("graphs"), self.folder("steps")
        good, bad = graph_file(graphs, "g1"), graphs / "g2__p0.edn"
        bad.write_text("{:nodes [", encoding="utf-8")
        done = run_script("cas_segment.py", str(good), str(bad), "--out-dir", str(steps), env={})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual([p.name for p in steps.iterdir()], ["g1__p0.steps.json"])
        self.assertIn("SKIP g2__p0.edn: EDNDecodeError", done.stderr)
        # Nothing it could segment: that is the stage failing.
        self.assertEqual(run_script("cas_segment.py", str(bad), "--out-dir", str(steps), env={}).returncode, 1)

    def test_s5_gap_maps_leave_out_a_steps_file_they_cannot_read(self):
        graphs, steps, rung3 = self.folder("graphs"), self.folder("steps"), self.folder("rung3")
        run_script("cas_segment.py", str(graph_file(graphs, "g1")), "--out-dir", str(steps), env={})
        (steps / "g2__p0.steps.json").write_text("{", encoding="utf-8")
        printed = io.StringIO()
        with mock.patch.object(r3.cas_select, "load_patterns", lambda **kw: {}), \
                contextlib.redirect_stderr(printed), contextlib.redirect_stdout(io.StringIO()):
            code = r3.main(["--steps-dir", str(steps), "--out-dir", str(rung3)])
        self.assertEqual(code, 0)
        self.assertEqual([p.name for p in rung3.iterdir()], ["g1.technique.json"])
        self.assertIn("SKIP g2__p0.steps.json: JSONDecodeError", printed.getvalue())

    def test_s5_comprehension_leaves_out_a_graph_whose_steps_it_cannot_read(self):
        import clean_comprehension as comprehension
        graphs, steps, empty = self.folder("graphs"), self.folder("steps"), self.folder("empty")
        for pid in ("g1", "g2"):
            graph_file(graphs, pid)
        run_script("cas_segment.py", str(graphs / "g1__p0.edn"), "--out-dir", str(steps), env={})
        (steps / "g2__p0.steps.json").write_text("{", encoding="utf-8")
        out = self.base / "comprehension.json"
        argv = ["clean_comprehension.py", "--graphs", str(graphs), "--steps", str(steps),
                "--rung3", str(empty), "--candidates", str(empty), "--out", str(out)]
        r2d, printed = comprehension.r2d, io.StringIO()
        noun = {"rate": 0.5, "buckets": {}, "undefined": []}      # stands in for the concept substrate
        with mock.patch.object(sys, "argv", argv), mock.patch.object(r2d, "parse_args", lambda argv: None), \
                mock.patch.object(r2d, "load_substrate", lambda args: None), \
                mock.patch.object(r2d, "load_edn", lambda path: {}), \
                mock.patch.object(r2d, "check_graph", lambda graph, path, substrate: noun), \
                mock.patch.object(comprehension.cas_select, "load_patterns", lambda **kw: {}), \
                contextlib.redirect_stdout(printed):
            comprehension.main()
        self.assertEqual([row["pid"] for row in json.loads(out.read_text())["proofs"]], ["g1__p0"])
        self.assertIn("SKIP g2__p0: JSONDecodeError", printed.getvalue())

    def test_s7_embedding_leaves_out_a_clean_it_cannot_read_and_takes_a_lone_proof(self):
        import clean_structure_embed as embed
        clean, out = self.cleans("p1", "p2", unreadable=("p3",)), self.base / "demo"
        argv = ["clean_structure_embed.py", "--clean-dir", str(clean), "--out", str(out)]
        stand_in = lambda texts: (embed.np.zeros((len(texts), 4)), "stand-in")    # no model is fetched
        for ids in (["p1", "p2"], ["p1"]):
            printed = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(embed, "text_embeddings", stand_in), \
                    contextlib.redirect_stdout(printed):
                embed.main()
            self.assertEqual(json.loads((out / "clean-embed.json").read_text())["ids"], ids)
            self.assertIn("SKIP p3.clean.edn", printed.getvalue())
            (clean / "p2.clean.edn").unlink(missing_ok=True)     # then a proof with no neighbour

    def test_s8_export_leaves_out_a_clean_it_cannot_read(self):
        import clean_graph_export as export
        clean, out = self.cleans("p1", "p2", unreadable=("p3",)), self.base / "ingest"
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            export.build(str(clean), str(out), str(self.base / "no-embed.json"))
        graph = json.loads((out / "clean-graph.json").read_text())
        self.assertEqual(sorted(n["id"] for n in graph["nodes"] if n["label"] == "Proof"), ["p1", "p2"])
        self.assertIn("SKIP p3.clean.edn", printed.getvalue())

    def test_s9_reports_a_graph_it_leaves_out(self):
        graphs = self.folder("graphs")
        graph_file(graphs, "g1")
        (graphs / "g2__p0.edn").write_text("{:edges [", encoding="utf-8")
        done = run_script("warrant_normalize.py", "--graphs", str(graphs), "--out", str(self.base / "holes.json"),
                          env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("SKIP g2__p0: EDNDecodeError", done.stdout)

    def test_s10_and_s12_measure_without_what_they_cannot_read(self):
        graphs, candidates = self.folder("graphs"), self.folder("candidates")
        graph_file(graphs, "g1")
        (graphs / "g2__p0.edn").write_bytes(b"\xff\xfe not UTF-8")
        (candidates / "g1__p0.candidate.json").write_text(json.dumps({"source-window": "It suffices to check this."}))
        (candidates / "g2__p0.candidate.json").write_text("{", encoding="utf-8")
        # No e-print either: the harvest's text lookup fails for g1 and is caught.
        env = {"FUTON6_EPRINTS": str(self.base / "no-eprints"), "TMPDIR": str(self.base), "PYTHONUTF8": "1"}
        for script, extra in (("iatc_move_reground.py", ()), ("accretion_curves.py", ("--run-dir", str(self.base / "run")))):
            done = run_script(script, "--graphs", str(graphs), "--candidates", str(candidates), *extra, env=env)
            self.assertEqual(done.returncode, 0, script + done.stdout + done.stderr)
            self.assertIn("SKIP g2__p0.candidate.json: JSONDecodeError", done.stderr)
            self.assertIn("SKIP g2__p0.edn: UnicodeDecodeError", done.stderr)

    def test_s10_expository_lift_goes_on_without_a_paper_that_has_no_eprint(self):
        ids = self.base / "ids.txt"
        ids.write_text(f"{MISSING}\n")
        done = run_script("expository_reground.py", "--scopes", str(self.folder("scopes")), "--measure-ids", str(ids),
                          env={"FUTON6_EPRINTS": str(self.base / "no-eprints")})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"(skip {MISSING}: no eprint for {MISSING}", done.stdout)

    def test_s11_signatures_take_fewer_than_two_papers(self):
        import clean_paper_signature as signature
        shape = {"macro": "construct-exploit-discharge", "n_boxes": 2, "n_wires": 1, "n_holes": 0,
                 "n_discharges_known": 0, "max_fanout": 1, "depth": 2, "n_sources": 1, "n_sinks": 1}
        embed_json, run = self.base / "clean-embed.json", self.folder("run")
        # one paper with a signature (two proofs), then none (one proof each)
        for ids, signed in ((["p1__p0", "p1__p1"], ["p1"]), (["p1__p0", "p2__p0"], [])):
            embed_json.write_text(json.dumps({"ids": ids, "breakdowns": [shape, shape]}))
            argv = ["clean_paper_signature.py", "--embed", str(embed_json), "--run-dir", str(run)]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                signature.main()
            canon = json.loads((run / "structural-canon.json").read_text())
            self.assertEqual([s["paper"] for s in canon["signatures"]], signed)
            self.assertTrue(all(s["nearest"] is None for s in canon["signatures"]))
            self.assertIsNone(canon["paper_twin_sim"])


@unittest.skipUnless(shutil.which("bb"), "bb reads the graphs, as in a run")
class MeasurementsLeaveOutWhatTheyCannotRead(unittest.TestCase):
    """The S3 and S4 measurements read every graph through one bb process."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)

    def test_the_reader_returns_none_for_a_file_it_cannot_read(self):
        import expository_scope_audit as audit
        good, bad, empty = self.base / "good.edn", self.base / "bad.edn", self.base / "empty.edn"
        good.write_text("{:a 1}")
        bad.write_text("{:a")
        empty.write_text("")
        printed = io.StringIO()
        with contextlib.redirect_stderr(printed):
            docs = audit.read_edn([good, bad, empty, good])
        self.assertEqual(docs, [{"a": 1}, None, None, {"a": 1}])
        self.assertIn("SKIP bad.edn", printed.getvalue())
        self.assertIn("SKIP empty.edn: empty", printed.getvalue())

    def test_s3_quote_check_leaves_out_a_graph_it_cannot_read(self):
        graphs, candidates = self.base / "graphs", self.base / "candidates"
        graphs.mkdir()
        candidates.mkdir()
        graph_file(graphs, "g1")
        (graphs / "g2__p0.edn").write_text("{:nodes [", encoding="utf-8")
        done = run_script("iatc_quote_check.py", "--graphs", str(graphs), "--candidates", str(candidates), env={})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(json.loads(done.stdout.strip().splitlines()[-1])["graphs"], 1)
        self.assertIn("SKIP g2__p0.edn", done.stderr)

    def test_s4_audit_leaves_out_a_passage_it_cannot_read(self):
        import run_manifest
        run = self.base / "run"
        expo = run / "artifacts" / "expo"
        expo.mkdir(parents=True)
        (run / "artifacts" / "expo-candidates").mkdir()
        ids = run / "corpus.ids.txt"
        ids.write_text("p\n")
        (run / run_manifest.NAME).write_text(json.dumps({
            "schema-version": 1, "artifacts": run_manifest.ARTIFACTS, "ids": "corpus.ids.txt", "papers": ["p"],
            "run-id": "t", "corpus-id": "t", "corpus-sha256": run_manifest.digest(ids)}))
        (expo / "p.r1.edn").write_text('{:paper/id "p" :passage/id "p:r1:L1-2" :scopes [{:id :s1 '
                                       ':kind :connection :source {:lines [1 2]} :slot-fill {:link "the orbit"}}]}')
        (expo / "p.r2.edn").write_text('{:paper/id "p" :passage/id "no-region" :scopes []}')
        (expo / "p.r3.edn").write_text("{:paper/id")
        done = run_script("expository_scope_audit.py", "--run-dir", str(run), env={})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        summary = json.loads((run / "expository-scope-audit.json").read_text())["summary"]
        self.assertEqual((summary["passages"], summary["scopes"]), (1, 1))
        self.assertIn("SKIP p.r2.edn: ValueError", done.stderr)
        self.assertIn("SKIP p.r3.edn", done.stderr)
        # With no scope at all the shares are None, and they print as "-".
        for path in expo.iterdir():
            path.unlink()
        done = run_script("expository_scope_audit.py", "--run-dir", str(run), env={})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("top kind - -; bare parent kinds -", done.stdout)


if __name__ == "__main__":
    unittest.main()
