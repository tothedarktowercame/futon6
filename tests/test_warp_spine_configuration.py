"""The WARP spine's locations are configuration, and the default IS the CT run.

Two claims are tested together because either alone is a trap. The subject's
concept substrate must move as ONE set when it is re-pointed — a half-moved
vocabulary is exactly the silent staleness warp_substrate_check.py exists to
catch — and with nothing set, every path must still be the one this checkout
has always used.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import futon6_config as config


def eprint_archive(directory: Path, paper_id: str, body: str) -> Path:
    """One arXiv-shaped source archive: a .tar.gz holding the paper's TeX."""
    archive = directory / f"{paper_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = body.encode()
        info = tarfile.TarInfo(f"{paper_id}.tex")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return archive


class SpineDefaultsTests(unittest.TestCase):
    """With nothing configured, the spine writes where it always wrote."""

    def setUp(self):
        self.cleared = patch.dict(os.environ, {}, clear=False)
        self.cleared.start()
        for name in ("FUTON6_WARP_DIR", "FUTON6_SUBJECT", "FUTON6_SUBJECT_DATA",
                     "FUTON6_PROSE_SOURCE", "FUTON6_MARKS"):
            os.environ.pop(name, None)
        self.addCleanup(self.cleared.stop)

    def test_warp_and_vocabulary_defaults_are_the_ct_paths(self):
        self.assertEqual(config.warp(), (config.ROOT / "data/warp").resolve())
        self.assertEqual(config.subject(), "ct")
        self.assertEqual(config.subject_data(), (config.ROOT / "data").resolve())
        self.assertEqual(config.term_prior(),
                         (config.ROOT / "data/term-prior-ct.json").resolve())
        self.assertEqual(config.concept_encyclopedia(),
                         (config.ROOT / "data/concept-encyclopedia-ct.json").resolve())
        self.assertEqual(config.prose_source(), "marks")

    def test_the_stage_table_and_its_scripts_name_the_default_paths(self):
        import warp_run

        stages = {stage.stage_id: stage for stage in warp_run.SPINE_STAGES}
        self.assertEqual(stages["S2"].outputs,
                         ((config.ROOT / "data/warp/defined-index.json").resolve(),))
        self.assertEqual(stages["S6t"].outputs,
                         ((config.ROOT / "data/term-prior-ct.json").resolve(),))
        self.assertEqual(stages["S6b"].outputs,
                         ((config.ROOT / "data/concept-encyclopedia-ct.json").resolve(),
                          (config.ROOT / "data/concept-encyclopedia/ct").resolve()))
        # S6t reads the DP marks by default, so its freshness tracks them, and the
        # concordance still leads the spine without depending on the defined-index.
        self.assertEqual(stages["S6t"].inputs, (config.marks(),))
        self.assertEqual(warp_run.SPINE_STAGES[0].stage_id, "S1a")
        self.assertNotIn((config.ROOT / "data/warp/defined-index.json").resolve(),
                         stages["S1a"].inputs)

    def test_the_substrate_gate_checks_the_default_substrate(self):
        import warp_substrate_check as gate

        self.assertIn((config.ROOT / "data/warp/concept-index.json").resolve(), gate.SUBSTRATE)
        self.assertIn((config.ROOT / "data/concept-encyclopedia-ct.json").resolve(),
                      gate.SUBSTRATE)


class SubjectOverrideTests(unittest.TestCase):
    """A second subject's vocabulary must not touch the first one's."""

    def test_one_override_moves_the_whole_vocabulary(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            with patch.dict(os.environ, {
                "FUTON6_WARP_DIR": str(base / "warp"),
                "FUTON6_SUBJECT": "rh-grh",
                "FUTON6_SUBJECT_DATA": str(base / "vocabulary"),
            }):
                self.assertEqual(config.warp(), (base / "warp").resolve())
                self.assertEqual(config.term_prior(),
                                 (base / "vocabulary/term-prior-rh-grh.json").resolve())
                self.assertEqual(config.concept_encyclopedia(),
                                 (base / "vocabulary/concept-encyclopedia-rh-grh.json").resolve())

    def test_every_spine_script_agrees_on_the_configured_locations(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            env = {k: v for k, v in os.environ.items() if not k.startswith("FUTON")}
            env.update(FUTON6_WARP_DIR=str(base / "warp"), FUTON6_SUBJECT="rh-grh",
                       FUTON6_SUBJECT_DATA=str(base / "vocabulary"),
                       FUTON6_EPRINTS=str(base / "sources"),
                       FUTON6_MARKS=str(base / "marks"))
            probe = '''import json, sys
sys.path.insert(0, "scripts")
import futon6_config as config
import warp_concordance, warp_defined_pass, warp_hitlist, warp_def_snippets
import warp_concept_usage, warp_concept_graph, warp_run, warp_substrate_check
import sfc_concept_coverage, sfc_concept_index, build_concept_encyclopedia
warp = config.warp()
assert warp_concordance.DEFAULT_OUT == warp / "concordance.json"
assert warp_defined_pass.OUT == warp / "defined-index.json"
assert warp_hitlist.W == warp_concept_usage.W == warp_concept_graph.W == warp
assert warp_def_snippets.OUT == warp / "def-snippets.json"
assert warp_run.WARP == warp and warp_run.MANIFEST == warp / "warp-manifest.json"
assert warp_run.TERM_PRIOR == config.term_prior()
assert warp_run.ENCYCLOPEDIA == config.concept_encyclopedia()
assert sfc_concept_index.DEFAULT_INDEX == warp / "concept-index.json"
assert sfc_concept_coverage.DEFAULT_ENCYCLOPEDIA == config.concept_encyclopedia()
assert build_concept_encyclopedia.WARP == warp
assert config.concept_encyclopedia() in warp_substrate_check.SUBSTRATE
print(json.dumps({"warp": str(warp), "prior": str(config.term_prior())}))
'''
            result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            seen = json.loads(result.stdout)
            self.assertEqual(Path(seen["warp"]), (base / "warp").resolve())
            self.assertEqual(Path(seen["prior"]),
                             (base / "vocabulary/term-prior-rh-grh.json").resolve())

    def test_a_child_is_told_the_same_locations(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            with patch.dict(os.environ, {"FUTON6_WARP_DIR": str(base / "warp"),
                                         "FUTON6_SUBJECT": "rh-grh",
                                         "FUTON6_SUBJECT_DATA": str(base / "vocabulary")}), \
                    patch.object(config, "scale", return_value={"shards": 1,
                                                                "concurrency-per-shard": 1}):
                env = config.child_environment()
            self.assertEqual(Path(env["FUTON6_WARP_DIR"]), (base / "warp").resolve())
            self.assertEqual(env["FUTON6_SUBJECT"], "rh-grh")
            self.assertEqual(Path(env["FUTON6_SUBJECT_DATA"]), (base / "vocabulary").resolve())
            self.assertEqual(env["FUTON6_PROSE_SOURCE"], "marks")


class ConceptFilterTests(unittest.TestCase):
    """A caller-curated filter of non-concept phrasing, honoured where concepts are first
    recorded -- and inert unless configured."""

    FILTER = ("## proof scaffolding\n"
              "re: proof of (theorem|lemma)\n"
              "## discourse\n"
              "well known\n")
    TEXT = (r"It is \emph{well known} that a \emph{Dirichlet character} is periodic. "
            r"\emph{proof of lemma} and \emph{main result}.")

    def _filter_file(self, base: Path) -> Path:
        path = base / "phrasing.txt"
        path.write_text(self.FILTER, encoding="utf-8")
        return path

    def test_entries_match_exactly_or_by_family_and_keep_their_category(self):
        import concept_filter

        with tempfile.TemporaryDirectory() as d:
            active = concept_filter.load(self._filter_file(Path(d)))
        self.assertEqual(active.match("well known").category, "discourse")
        self.assertEqual(active.match("proof of lemma").category, "proof scaffolding")
        self.assertIsNone(active.match("proof of concept"))
        self.assertIsNone(active.match("dirichlet character"))

    def test_the_defined_pass_drops_filtered_phrasing_only_when_configured(self):
        import warp_defined_pass as defined_pass

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FUTON6_CONCEPT_FILTER", None)
            unfiltered = defined_pass.mined_concepts(self.TEXT, exclude_references=True)
        self.assertIn("well known", unfiltered)
        self.assertEqual(unfiltered, defined_pass.defined_concepts(self.TEXT))
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"FUTON6_CONCEPT_FILTER": str(self._filter_file(Path(d)))}):
                filtered = defined_pass.mined_concepts(self.TEXT, exclude_references=True)
        self.assertNotIn("well known", filtered)
        self.assertNotIn("proof of lemma", filtered)
        self.assertIn("dirichlet character", filtered)
        self.assertIn("main result", filtered)     # not in this filter, so kept

    def test_a_bad_family_is_refused_with_its_line(self):
        import concept_filter

        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.txt"
            bad.write_text("re: proof of (theorem\n", encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                concept_filter.load(bad)
        self.assertIn(":1:", str(raised.exception))


class ConcordanceWriterTests(unittest.TestCase):
    """The concordance is streamed to disk to keep peak memory bounded at corpus scale,
    and must still be byte-for-byte the file `json.dumps(..., indent=2, sort_keys=True)`
    wrote, since other stages and the shipped substrate read it."""

    def _old_bytes(self, result: dict) -> str:
        materialized = dict(result)
        materialized["terms"] = {
            term: [{"paper": paper, "count": count, "role": role} for paper, count, role in rows]
            for term, rows in sorted(result["terms"].items())}
        return json.dumps(materialized, indent=2, sort_keys=True) + "\n"

    def _new_bytes(self, result: dict) -> str:
        import io as _io
        import warp_concordance as concordance

        buffer = _io.StringIO()
        concordance.write_concordance(result, buffer)
        return buffer.getvalue()

    def test_identical_to_the_whole_file_serializer(self):
        result = {
            "schema": "warp-concordance-v1",
            "generated_at": "2026-09-25T00:00:00Z",
            "stats": {"papers": 2, "sources": {"raw-sweep+prose": 2}, "eprint-roots": ["a", "b"]},
            "failures": [{"paper": "x", "error": "ValueError('bad \"quote\"')"}],
            "terms": {
                "\\zeta": [("1111.0001", 3, "used"), ("2222.0002", 1, "used")],
                "riemann hypothesis": [("1111.0001", 2, "defined"), ("1111.0001", 5, "used")],
                "théorème": [("math__0206203", 1, "used")],
                "a \"quoted\" term": [("2222.0002", 4, "used")],
            },
        }
        self.assertEqual(self._new_bytes(result), self._old_bytes(result))

    def test_identical_when_there_are_no_terms(self):
        result = {"schema": "warp-concordance-v1", "generated_at": "t", "stats": {},
                  "failures": [], "terms": {}}
        self.assertEqual(self._new_bytes(result), self._old_bytes(result))

    def test_streamed_terms_equal_json_load_at_every_chunk_boundary(self):
        """The hit-list reads the concordance one term at a time; what it reads must be
        exactly what json.load reads, wherever a chunk happens to split a token."""
        import warp_concordance as concordance

        result = {
            "schema": "warp-concordance-v1", "generated_at": "2026-09-25T00:00:00Z",
            "stats": {"papers": 12345, "unique-terms": 4, "sources": {"raw-sweep+prose": 2}},
            "failures": [{"paper": "x", "error": "ValueError('a \"terms\": {} trap')"}],
            "terms": {
                "\\zeta": [("1111.0001", 1234567, "used"), ("2222.0002", 1, "used")],
                "riemann hypothesis": [("1111.0001", 2, "defined"), ("1111.0001", 5, "used")],
                "théorème": [("math__0206203", 1, "used")],
                "terms": [("2222.0002", 4, "used")],
            },
        }
        with tempfile.TemporaryDirectory() as d:
            written = Path(d) / "concordance.json"
            written.write_text(self._new_bytes(result), encoding="utf-8")
            expected = list(json.loads(written.read_text(encoding="utf-8"))["terms"].items())
            for chunk in (1, 2, 3, 7, 64, 1 << 20):
                with self.subTest(chunk=chunk):
                    streamed = list(concordance.iter_concordance_terms(written, chunk=chunk))
                    self.assertEqual(streamed, expected)

    def test_a_compact_file_streams_the_same_as_an_indented_one(self):
        """The reader parses JSON, not this writer's layout."""
        import warp_concordance as concordance

        document = {"failures": [], "schema": "s", "stats": {}, "generated_at": "t",
                    "terms": {"a b": [{"count": 1, "paper": "p", "role": "used"}], "c": []}}
        with tempfile.TemporaryDirectory() as d:
            compact = Path(d) / "compact.json"
            compact.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
            self.assertEqual(list(concordance.iter_concordance_terms(compact, chunk=5)),
                             list(document["terms"].items()))


class CorpusRootsTests(unittest.TestCase):
    """A corpus is many directories plus an id list, each named in a file.

    Copying an acquired corpus into one directory to satisfy a reader cost 9.5 GB and 90
    minutes on exFAT, which has no links -- so the directories are read in place.
    """

    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for name in ("FUTON6_EPRINTS", "FUTON6_CORPUS_IDS"):
            os.environ.pop(name, None)
        self.addCleanup(self.env.stop)

    def _corpus(self, base: Path) -> tuple[Path, Path]:
        first, second = base / "batch-1/eprints", base / "batch-2/eprints"
        for directory in (first, second):
            directory.mkdir(parents=True)
        eprint_archive(first, "1111.0001v1", r"An \emph{old version} appears here.")
        eprint_archive(second, "1111.0001v2", r"A \emph{revised version} appears here.")
        eprint_archive(second, "2222.0002v1", r"A \emph{second paper} appears here.")
        eprint_archive(second, "3333.0003v1", r"A \emph{paper of another subject} appears.")
        roots_file = base / "corpus.roots.txt"
        roots_file.write_text(f"# the corpus\n{first.as_posix()}\n\n{second.as_posix()}\n")
        ids_file = base / "corpus.ids.txt"
        ids_file.write_text("1111.0001\n2222.0002\n")
        return roots_file, ids_file

    def test_a_directory_is_still_just_that_directory(self):
        with tempfile.TemporaryDirectory() as d:
            sources = Path(d) / "sources"
            sources.mkdir()
            with patch.dict(os.environ, {"FUTON6_EPRINTS": str(sources)}):
                self.assertEqual(config.eprint_roots(), (config.eprints(),))
                self.assertIsNone(config.corpus_ids())

    def test_a_roots_file_names_every_directory(self):
        with tempfile.TemporaryDirectory() as d:
            roots_file, _ids = self._corpus(Path(d))
            with patch.dict(os.environ, {"FUTON6_EPRINTS": str(roots_file)}):
                roots = config.eprint_roots()
            self.assertEqual([root.name for root in roots], ["eprints", "eprints"])
            self.assertEqual([root.parent.name for root in roots], ["batch-1", "batch-2"])

    def test_the_ids_file_says_which_papers_are_the_corpus(self):
        import eprint_corpus

        with tempfile.TemporaryDirectory() as d:
            roots_file, ids_file = self._corpus(Path(d))
            with patch.dict(os.environ, {"FUTON6_EPRINTS": str(roots_file),
                                         "FUTON6_CORPUS_IDS": str(ids_file)}):
                self.assertEqual(config.corpus_ids(), frozenset({"1111.0001", "2222.0002"}))
                ids = eprint_corpus.corpus_paper_ids()
                archives = eprint_corpus.iter_corpus_eprints()
            # the third paper shares the directories but is not this corpus
            self.assertEqual(ids, ["1111.0001", "2222.0002"])
            # one paper held at two versions is read once, at its latest
            self.assertEqual([a.name for a in archives],
                             ["1111.0001v2.tar.gz", "2222.0002v1.tar.gz"])

    def test_a_paper_resolves_and_reads_from_whichever_root_holds_it(self):
        import eprint_corpus
        import warp_defined_pass as defined_pass

        with tempfile.TemporaryDirectory() as d:
            roots_file, ids_file = self._corpus(Path(d))
            with patch.dict(os.environ, {"FUTON6_EPRINTS": str(roots_file),
                                         "FUTON6_CORPUS_IDS": str(ids_file)}):
                roots = config.eprint_roots()
                found = eprint_corpus.find_corpus_eprint("1111.0001", roots)
                self.assertIsNotNone(found)
                self.assertEqual(found.name, "1111.0001v2.tar.gz")
                with patch.object(defined_pass, "EPRINT_ROOTS", roots), \
                        patch.object(defined_pass, "EPRINTS", roots[0]):
                    self.assertIn("revised version", defined_pass.read_text("1111.0001"))
                    self.assertEqual(defined_pass.corpus_paper_ids(),
                                     ["1111.0001", "2222.0002"])

    def test_a_list_of_archives_is_read_without_listing_any_directory(self):
        """The corpus's owner can say exactly which archive each paper is; then nothing is
        walked or searched, and each paper is opened where it lies."""
        import eprint_corpus

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self._corpus(base)
            listed = base / "corpus.archives.txt"
            listed.write_text(f"{(base / 'batch-2/eprints/1111.0001v2.tar.gz').as_posix()}\n"
                              f"{(base / 'batch-2/eprints/2222.0002v1.tar.gz').as_posix()}\n")
            with patch.dict(os.environ, {"FUTON6_EPRINTS": str(listed)}), \
                    patch.object(eprint_corpus, "iter_eprints",
                                 side_effect=AssertionError("a directory was listed")):
                self.assertEqual(eprint_corpus.corpus_paper_ids(), ["1111.0001", "2222.0002"])
                found = eprint_corpus.find_corpus_eprint("1111.0001")
            self.assertEqual(found.name, "1111.0001v2.tar.gz")

    def test_an_empty_roots_file_is_refused_rather_than_read_as_no_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "corpus.roots.txt"
            empty.write_text("# nothing but a comment\n")
            with patch.dict(os.environ, {"FUTON6_EPRINTS": str(empty)}):
                with self.assertRaises(ValueError):
                    config.eprint_roots()


class ProseSourceTests(unittest.TestCase):
    """A subject without DP markup is not a subject without text: the e-prints
    carry the full TeX, and the spine can read its prose from there."""

    def test_an_unknown_source_is_refused_rather_than_guessed(self):
        with patch.dict(os.environ, {"FUTON6_PROSE_SOURCE": "arxiv"}):
            with self.assertRaises(ValueError):
                config.prose_source()

    def test_the_stages_declare_the_inputs_the_source_actually_reads(self):
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if not k.startswith("FUTON")}
            env.update(FUTON6_PROSE_SOURCE="eprints",
                       FUTON6_EPRINTS=str(Path(d) / "sources"))
            probe = '''import sys
sys.path.insert(0, "scripts")
import warp_run
stages = {s.stage_id: s for s in warp_run.SPINE_STAGES}
assert stages["S6t"].inputs == (warp_run.EPRINTS,), stages["S6t"].inputs
# the prose layer reads S2's defined-index, so S2 is declared and runs first
assert warp_run.w("defined-index.json") in stages["S1a"].inputs, stages["S1a"].inputs
assert [s.stage_id for s in warp_run.SPINE_STAGES][:2] == ["S2", "S1a"]
assert len(warp_run.SPINE_STAGES) == len(warp_run._SPINE_STAGES)
print("ok")
'''
            result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_prose_layer_recovers_definienda_and_bounded_usage(self):
        """Without it an unmarked paper contributes only control sequences, and the
        hit-list — which needs a term used AND defined — collapses."""
        import warp_concordance as concordance

        text = (r"\begin{definition} A \emph{Dirichlet character} is a homomorphism."
                r"\end{definition} Every Dirichlet character is periodic, and the "
                r"\emph{Riemann hypothesis} concerns the critical line."
                "\n\\begin{thebibliography}{9}\n"
                r"\bibitem{S} J. Smith, \emph{Invent. Math.} 12 (1970) 1--20." "\n"
                r"\end{thebibliography}")
        vocabulary = {"dirichlet character", "riemann hypothesis", "critical line"}
        counts, used, defined = concordance.prose_counts(text, vocabulary)
        self.assertIn(("dirichlet character", concordance.ROLE_DEFINED), counts)
        self.assertIn(("riemann hypothesis", concordance.ROLE_DEFINED), counts)
        self.assertGreater(defined, 0)
        # usage is the SET of vocabulary phrases the paper uses -- counted per concept by
        # `build`, never written as rows
        self.assertEqual(used, {"dirichlet character", "riemann hypothesis", "critical line"})
        self.assertFalse(any(role == concordance.ROLE_USED for _term, role in counts))
        # nothing outside the corpus-defined vocabulary is considered used
        self.assertNotIn("homomorphism", used)
        # the italicised journal in the reference list is a citation, not a definiendum
        self.assertNotIn(("invent math", concordance.ROLE_DEFINED), counts)

    def test_no_vocabulary_means_no_invented_usage(self):
        import warp_concordance as concordance

        counts, used, _ = concordance.prose_counts(
            r"A \emph{Dirichlet character} is periodic.", None)
        self.assertEqual(used, set())
        self.assertEqual([role for _term, role in counts],
                         [concordance.ROLE_DEFINED])

    def test_mining_a_marked_up_corpus_keeps_every_definiendum(self):
        """The CT defined-index must not change: references are excluded only when
        the corpus is mined from raw e-prints."""
        import warp_defined_pass as defined_pass

        text = (r"A \emph{monoidal category} has a tensor product."
                "\n\\begin{thebibliography}{9}\n"
                r"\bibitem{M} S. Mac Lane, \emph{Categories for the Working}." "\n"
                r"\end{thebibliography}")
        kept = defined_pass.mined_concepts(text, exclude_references=False)
        mined = defined_pass.mined_concepts(text, exclude_references=True)
        self.assertEqual(kept, defined_pass.defined_concepts(text))
        self.assertIn("categories for the working", kept)
        self.assertNotIn("categories for the working", mined)
        self.assertIn("monoidal category", mined)

    def test_the_eprint_source_mines_the_paper_text_the_archives_carry(self):
        import build_term_prior

        with tempfile.TemporaryDirectory() as d:
            sources = Path(d) / "sources"
            sources.mkdir()
            out = Path(d) / "term-prior-rh-grh.json"
            for n in range(3):
                eprint_archive(sources, f"24{n:02d}.0000{n}",
                               r"The \emph{Riemann zeta function} is entire except "
                               r"at one. A Dirichlet L function has an Euler product.")
            self.assertEqual(build_term_prior.main([
                "--source", "eprints", "--eprints-dir", str(sources),
                "--min-papers", "3", "--msc", "rh-grh", "--out", str(out)]), 0)
            written = json.loads(out.read_text())
            self.assertEqual(written["_meta"]["source"], "eprints")
            self.assertEqual(written["_meta"]["papers"], 3)
            self.assertIn("riemann zeta function", written["df"])
            self.assertIn("euler product", written["df"])

    def test_the_marks_source_still_reads_the_marks_text_field(self):
        import build_term_prior

        with tempfile.TemporaryDirectory() as d:
            marks = Path(d) / "marks"
            marks.mkdir()
            out = Path(d) / "term-prior-ct.json"
            for n in range(3):
                (marks / f"fable-{n}-dp-emacs.json").write_text(json.dumps(
                    {"text": "A monoidal category has a tensor product."}))
            self.assertEqual(build_term_prior.main([
                "--source", "marks", "--marks-dir", str(marks),
                "--min-papers", "3", "--out", str(out)]), 0)
            written = json.loads(out.read_text())
            self.assertEqual(written["_meta"]["source"], "marks")
            self.assertIn("monoidal category", written["df"])


if __name__ == "__main__":
    unittest.main()
