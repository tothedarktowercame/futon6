"""Lexicon-only papers join WARP's corpus and no other stage's.

FUTON6_WARP_LEXICON_IDS names concept sources (the APM corpus's Xi) that S2
should read but S3/S4 must not mine. They are pinned in the run manifest apart
from the corpus, and only the WARP phase adds them to its id list.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_manifest as manifest  # noqa: E402
import mark7_corpus_layers as layers  # noqa: E402


class LexiconTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.run_dir = self.base / "run"
        self.ids = self.base / "corpus.ids"
        self.ids.write_text("a00J01\na00J02\n")
        self.lexicon = self.base / "lexicon.ids"
        self.addCleanup(patch.stopall)
        patch.object(manifest, "source_identity", return_value={"git-head": "f", "source-sha256": "c"}).start()
        patch.object(manifest, "substrate_identity", return_value={"fixture": "s"}).start()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUTON6_")}
        patch.dict(os.environ, clean, clear=True).start()

    def prepare(self):
        with manifest.lock(self.run_dir):
            return manifest.prepare(self.run_dir, "r", "c", self.ids)

    def test_no_lexicon_leaves_the_manifest_as_before(self):
        self.assertNotIn("lexicon", self.prepare())

    def test_lexicon_is_pinned_apart_from_the_corpus(self):
        self.lexicon.write_text("# concept index\nxi\n")
        os.environ[manifest.LEXICON_ENV] = str(self.lexicon)
        doc = self.prepare()
        self.assertEqual(doc["lexicon"]["ids"], ["xi"])
        self.assertEqual(doc["papers"], ["a00J01", "a00J02"])
        self.assertEqual((self.run_dir / doc["ids"]).read_text(), "a00J01\na00J02\n")

    def test_lexicon_overlapping_the_corpus_is_refused(self):
        self.lexicon.write_text("a00J01\n")
        os.environ[manifest.LEXICON_ENV] = str(self.lexicon)
        with self.assertRaisesRegex(ValueError, "also in the mining corpus"):
            self.prepare()

    def test_only_warp_sees_the_lexicon(self):
        self.lexicon.write_text("xi\n")
        os.environ[manifest.LEXICON_ENV] = str(self.lexicon)
        doc = self.prepare()
        warp_dir = self.base / "warp"
        seen = []
        with patch.object(layers, "run_command", lambda argv, env: seen.append(env["FUTON6_CORPUS_IDS"])), \
                patch.object(layers.config, "warp", return_value=warp_dir), \
                patch.object(layers, "enabled", return_value=True):
            layers.warp_phase(self.run_dir, doc)
        self.assertTrue(seen)
        self.assertEqual(Path(seen[0]).read_text().split(), ["a00J01", "a00J02", "xi"])
        self.assertEqual((self.run_dir / doc["ids"]).read_text().split(), ["a00J01", "a00J02"])
