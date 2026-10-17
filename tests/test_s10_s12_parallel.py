"""S10 and S12 compile each recognizer pattern once, use FUTON6_WORKERS processes, and give the serial result.

In run mark7rhgrh100-20260928 S10 took 87 min and S12 115 min in one process: the recognizer
recompiled almost every one of its 3,365 harvested cues for every sentence (`re` caches 512
patterns), S10 built each paper's text four times, and S12 copied and re-harvested its graphs from
scratch at each of its 12 checkpoints. Re-run with these changes on that run's data, on every CPU
the job could use (now FUTON6_WORKERS=all), S10 and S12 wrote byte-identical products and logs in
9.5 and 2.3 min.

Each test compares the new code with the code it replaced, kept here as the reference.
"""
from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import accretion_curves  # noqa: E402
import expository_reground  # noqa: E402
import iatc_lexicon_harvest as harvest_module  # noqa: E402
import iatc_move_reground  # noqa: E402
import strategy_recognizer as sr  # noqa: E402


def reference_matches(text, patterns):
    for p in patterns:
        try:
            if re.search(p, text):
                return p
        except re.error:
            if p in text:
                return p
    return None


def reference_score(vocab, windows):
    with mock.patch.object(sr, "matches", reference_matches):
        bs = Counter()
        for w in windows:
            bs.update(sr.recognize_text(w, vocab)[0])
    g, t, u = bs["grounded"], bs["thin"], bs["ungrounded"]
    tot = g + t + u
    return {"grounded": g, "thin": t, "ungrounded": u,
            "proof-move-grounding": round((g + 0.5 * t) / tot, 3) if tot else 0.0}


def reference_harvest(graph_dir):
    """iatc_lexicon_harvest.harvest before 2026-10-02: one process, one pass."""
    h = harvest_module
    lex = defaultdict(lambda: {"count": 0, "conf": [], "kinds": Counter(), "layer": Counter(), "exemplars": []})
    grammar = Counter()
    flattened = []
    cache = {}
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.edn"))):
        if "rung2" in f:
            continue
        pid = h._pid_of(f)
        lines = h._text_lines(pid, cache)
        try:
            t = open(f).read()
        except (OSError, UnicodeDecodeError):
            continue
        expository = ":scopes" in t or ":slot-fill" in t

        def add(key, conf, kind, layer, a=None, b=None):
            e = lex[h._norm(key)]; e["count"] += 1; e["conf"].append(conf)
            e["kinds"][kind] += 1; e["layer"][layer] += 1
            if a and len(e["exemplars"]) < 3:
                e["exemplars"].append({"pid": pid, "lines": [a, b], "conf": conf})

        if expository:
            for block in re.split(r'(?=:kind :)', t):
                km = re.search(r':kind (:[a-z/-]+)', block)
                if not km:
                    continue
                kind = km.group(1)
                grammar[kind] += 1
                lm = re.search(r':lines \[(\d+) (\d+)\]', block)
                fills = re.findall(r'"([^"]{4,})"', block.split(":slot-fill", 1)[1]) if ":slot-fill" in block else []
                fill = " ".join(fills)[:90]
                if fill and lm:
                    a, b = int(lm.group(1)), int(lm.group(2))
                    conf, _, _ = h._confidence(fill, h._span(lines, a, b))
                    add(fill, conf, kind, "expository", a, b)
        else:
            for m in re.finditer(r':kind :([a-z-]+),?\s*:text "([^"]{4,80})"[^}]*?:source \{:lines \[(\d+) (\d+)\]', t):
                kind, text, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                conf, faith, flat = h._confidence(text, h._span(lines, a, b))
                add(text, conf, kind, "iatc", a, b)
                if flat:
                    flattened.append({"pid": pid, "lines": [a, b], "text": text})
            for w in re.findall(r':warrant \{[^}]*?:text "([^"]{4,80})"', t):
                add(w, 0.5, "warrant", "iatc")
            for r in re.findall(r":relation :([a-z-]+)", t):
                grammar[r] += 1
    return lex, grammar, flattened


def plain(lex):
    return [(k, {**e, "kinds": list(e["kinds"].items()), "layer": list(e["layer"].items())})
            for k, e in lex.items()]


def iatc_graph(x, y):
    return ('{:nodes [{:id "n1" :kind :claim, :text "' + x + ' reduces to the case of ' + y + '" '
            ':source {:lines [3 4]}} {:id "n2" :kind :step, :text "by functoriality of ' + y + '" '
            ':source {:lines [5 6]}}] :edges [{:relation :implies :warrant {:text "naturality of ' + x + '"}} '
            '{:relation :iff}]}')


def expository_graph(x, y):
    return ('{:scopes [{:kind :difficulty-assessment :lines [2 3] :slot-fill ["the hard part is ' + x + '"]} '
            '{:kind :generalisation :lines [4 5] :slot-fill ["this extends to ' + y + ' spaces"]}]}')


def write_corpus(root: Path) -> Path:
    graphs = root / "graphs"
    graphs.mkdir()
    words = [("sheaves", "cohomology"), ("modules", "extensions"), ("spectra", "localization"),
             ("sheaves", "descent")]
    for paper, (a, b) in zip(["9901.00001", "9901.00002", "math__9902003"], words):
        for n, (x, y) in enumerate(words):
            graph = iatc_graph if n % 2 == 0 else expository_graph
            (graphs / f"{paper}__p{n + 1}.edn").write_text(graph(f"{x} of {a}", f"{y} {b}"))
    (graphs / "9901.00001__p9.rung2.edn").write_text(iatc_graph("left", "out"))   # rung-2: left out
    (graphs / "9901.00002__p8.edn").write_bytes(b"\xff\xfe not utf-8")   # unreadable: skipped
    return graphs


class S10S12ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        # No paper text: dp_paper_view finds no e-print here, in this process and in workers. The
        # corpus holds an unreadable graph, which these runs leave out as the reference does
        # (FUTON6_CONTINUE_PAST_ITEM_ERRORS; unset, it fails the harvest).
        eprints = self.tmp / "eprints"
        eprints.mkdir()
        patcher = mock.patch.dict(os.environ, {"FUTON6_EPRINTS": str(eprints),
                                               "FUTON6_CONTINUE_PAST_ITEM_ERRORS": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_matches_gives_what_re_search_gave_and_compiles_each_pattern_once(self):
        patterns = ["suffices", r"\bby (induction|cases)\b", "f(x", "", "hölder", "a.b"]
        texts = ["it suffices to show", "by induction on n", "apply f(x here", "hölder's inequality",
                 "a-b and axb", "nothing"]
        for text in texts:
            for k in range(len(patterns)):
                self.assertEqual(sr.matches(text, patterns[k:]), reference_matches(text, patterns[k:]))
        many = [f"cue{n:04d}x" for n in range(600)] + [rf"\bcue{n:04d}y\b" for n in range(600)]
        sr._MATCHERS.clear()
        with mock.patch.object(sr.re, "compile", wraps=re.compile) as compile_:
            for _ in range(3):
                for text in texts:
                    sr.matches(text, many)
        self.assertEqual(compile_.call_count, 600)   # the regex patterns, each once; literals need none

    def test_bucket_counts_in_workers_equal_the_serial_sum(self):
        vocab = sr.load_vocab(str(ROOT / "holes/clean/tactic-gesture-vocab.edn"))
        vocab = {**vocab, "heuristic": {**vocab["heuristic"], "corpus-move": ["functoriality", "suffices"]}}
        windows = [f"By induction on n we get the claim number {n}. It suffices to note that "
                   f"functoriality gives {n}! Perhaps this is clear. We conjecture {n} more." * (n % 4)
                   for n in range(41)]
        self.assertEqual(iatc_move_reground.score(vocab, windows), reference_score(vocab, windows))
        with mock.patch.object(sr, "matches", reference_matches):
            serial = Counter()
            for w in windows:
                serial.update(sr.recognize_text(w, vocab)[0])
        self.assertEqual(sr.bucket_counts(vocab, windows, workers=3), serial)

    def test_harvest_in_workers_gives_one_pass_over_the_files(self):
        graphs = write_corpus(self.tmp)
        lex, grammar, flattened = harvest_module.harvest(str(graphs), workers=3)
        ref_lex, ref_grammar, ref_flattened = reference_harvest(str(graphs))
        self.assertTrue(ref_lex)
        self.assertEqual(plain(lex), plain(ref_lex))
        self.assertEqual(list(grammar.items()), list(ref_grammar.items()))
        self.assertEqual(flattened, ref_flattened)
        self.assertEqual(iatc_move_reground.cluster_cues(lex), iatc_move_reground.cluster_cues(ref_lex))

    def test_accretion_checkpoints_give_what_harvesting_each_prefix_gave(self):
        graphs = write_corpus(self.tmp)
        candidates = self.tmp / "candidates"
        candidates.mkdir()
        for n in range(6):
            (candidates / f"c{n}.candidate.json").write_text(
                '{"source-window": "We reduce to the case of sheaves. By functoriality of descent '
                f'extensions the claim {n} follows. It suffices to check localization."}}')
        ns = [0, 1, 2, 3, 5, 7, 10, 30]
        points, n_windows = accretion_curves.move_curve(str(graphs), str(candidates), ns=ns)
        golden = sorted(g for g in glob.glob(str(graphs / "*.edn")) if "rung2" not in g)
        windows = iatc_move_reground.candidate_windows(str(candidates))
        vocab = sr.load_vocab(str(ROOT / "holes/clean/tactic-gesture-vocab.edn"))
        expected = []
        for k in ns:
            k = min(k, len(golden))
            if k == 0:
                expected.append((0, 0, reference_score(vocab, windows)["proof-move-grounding"]))
                continue
            prefix = self.tmp / f"prefix-{k}"
            prefix.mkdir(exist_ok=True)
            for g in golden[:k]:
                shutil.copy(g, prefix)
            cues = iatc_move_reground.cluster_cues(reference_harvest(str(prefix))[0])
            aug = {**vocab, "heuristic": {**vocab["heuristic"], "corpus-move": cues}}
            expected.append((k, len(cues), reference_score(aug, windows)["proof-move-grounding"]))
        self.assertEqual(n_windows, len(windows))
        self.assertEqual(points, expected)

    def test_expository_reground_builds_each_paper_once(self):
        built = []

        def build(pid):
            built.append(pid)
            if pid == "missing":
                raise FileNotFoundError("no e-print")
            return {"text": f"Note that {pid} is standard.\nIndeed it is clear."}

        with mock.patch.object(expository_reground.dpv, "build", build):
            texts = expository_reground.paper_texts(["a", "b", "a", "missing", "b"], workers=1)
        self.assertEqual(built, ["a", "b", "missing"])
        self.assertEqual(texts["missing"], (False, "no e-print"))
        self.assertEqual(expository_reground._paper_lines(texts, "a"), ["Note that a is standard.", "Indeed it is clear."])
        self.assertEqual(expository_reground._paper_lines(texts, "missing"), [])


if __name__ == "__main__":
    unittest.main()
