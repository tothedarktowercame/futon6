"""--build-lexicon: a classical scan of the run's own papers becomes WARP's lexicon.

Covers the scanner's term filters, the pinned feature, S2's wiring (a run-local
eprint view holding the corpus plus the built lexicon paper), and preflight
judging the vocabulary the scan will build instead of warning.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_lexicon  # noqa: E402
import mark7_corpus_layers as layers  # noqa: E402
import preflight as pf  # noqa: E402
import run_manifest as manifest  # noqa: E402

DOC = "\\documentclass{article}\\begin{document}%s\\end{document}\n"
INDEX = {"terms": {
    "banach space": [{"term": "Banach space", "resolution-kind": "nnexus",
                      "domains": ["Planetmath", "Mathworld"], "msc": ["46B99"]}],
    "compact operator": [{"term": "compact operator", "resolution-kind": "nnexus",
                          "domains": ["Planetmath"], "msc": ["47B07"]}],
    "spectral radius": [{"term": "spectral radius", "resolution-kind": "nnexus",
                         "domains": ["Wikipedia"], "msc": ["47A10"]}],
    "complete proof": [{"term": "complete proof", "resolution-kind": "ct-prior"}],
    "such that": [{"term": "such that", "resolution-kind": "nnexus", "domains": ["Mathworld"]}],
    "right": [{"term": "right", "resolution-kind": "nnexus", "domains": ["Planetmath", "Wikipedia"]}],
    "operator": [{"term": "operator", "resolution-kind": "nnexus", "domains": ["Planetmath"]}],
}}
BODY = ("Let T be a compact operator on a Banach space such that the spectral radius "
        "is right here. Complete proof follows.")


class BuildLexiconTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.ep = self.base / "ep"
        self.ep.mkdir()
        for pid in ("p1", "p2", "p3"):
            (self.ep / f"{pid}.tex").write_text(DOC % BODY)
        # A fourth paper without the terms: a term in every paper is dropped as
        # near-universal (max-share 0.9), so a corpus must not be uniform.
        (self.ep / "p4.tex").write_text(DOC % "An unrelated note about nothing much.")
        self.ids = self.base / "ids.txt"
        self.ids.write_text("p1\np2\np3\np4\n")
        self.index = self.base / "index.json"
        self.index.write_text(json.dumps(INDEX))
        self.addCleanup(patch.stopall)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUTON6_")}
        patch.dict(os.environ, {**clean, "FUTON6_EPRINTS": str(self.ep)}, clear=True).start()
        patch.object(build_lexicon.config, "authority", return_value=self.index).start()

    def test_filters_keep_terms_and_drop_noise(self):
        vocab = build_lexicon.vocabulary(self.index)
        terms = {v["term"] for v in vocab.values()}
        self.assertEqual(terms, {"banach space", "compact operator", "spectral radius"})
        # ct-prior n-grams, connectives, common words and single-source single words go

    def test_build_writes_a_definitions_paper_and_a_report(self):
        out = self.base / "lex"
        report = build_lexicon.build(["p1", "p2", "p3", "p4"], out, index=self.index)
        self.assertEqual({t["term"] for t in report["terms"]},
                         {"banach space", "compact operator", "spectral radius"})
        tex = (out / f"{build_lexicon.LEXICON_ID}.tex").read_text()
        self.assertIn("\\begin{definition}\\emph{banach space}", tex)
        self.assertEqual(dict(report["msc-profile"]), {"46": 3, "47": 6})

    def test_feature_is_pinned_only_when_on(self):
        self.assertNotIn("build-lexicon", manifest.declared_features())
        self.assertTrue(manifest.declared_features(build_lexicon=True)["build-lexicon"])

    def test_s2_gives_warp_a_view_with_the_built_lexicon(self):
        run_dir = self.base / "run"
        doc = {"papers": ["p1", "p2", "p3"], "ids": "corpus.ids.txt",
               "features": {"warp": True, "build-lexicon": True}}
        run_dir.mkdir()
        (run_dir / "corpus.ids.txt").write_text("p1\np2\np3\n")
        seen = []

        def fake(argv, env):
            if argv[0] == "scripts/build_lexicon.py":
                build_lexicon.build(["p1", "p2", "p3"], Path(argv[argv.index("--out") + 1]),
                                    index=self.index)
            seen.append((argv[0], env.get("FUTON6_EPRINTS"), env.get("FUTON6_CORPUS_IDS")))

        with patch.object(layers, "run_command", fake), \
                patch.object(layers.config, "warp", return_value=run_dir / "warp"):
            layers.warp_phase(run_dir, doc)
        warp_call = next(s for s in seen if s[0] == "scripts/warp_run.py")
        view = Path(warp_call[1])
        self.assertEqual(sorted(p.name for p in view.iterdir()),
                         ["lexicon-built.tex", "p1.tex", "p2.tex", "p3.tex"])
        self.assertEqual(Path(warp_call[2]).read_text().split(),
                         ["p1", "p2", "p3", "lexicon-built"])
        self.assertEqual((run_dir / "corpus.ids.txt").read_text().split(), ["p1", "p2", "p3"])

    def test_preflight_judges_the_lexicon_it_will_build(self):
        enc = self.base / "enc.json"
        enc.write_text(json.dumps({"entries": [{"concept": "monoidal category"}]}))
        with patch.object(pf.config, "concept_encyclopedia", return_value=enc), \
                patch.object(pf.config, "authority", return_value=self.index):
            pf.R.clear()
            pf.check_vocabulary(str(self.ids))
            self.assertIsNone(pf.R[-1][1])
            self.assertIn("--build-lexicon", pf.R[-1][3])
            os.environ["FUTON6_BUILD_LEXICON"] = "1"
            pf.R.clear()
            pf.check_vocabulary(str(self.ids))
            self.assertIs(pf.R[-1][1], True)
            self.assertIn("lexicon to be built at S2", pf.R[-1][2])
