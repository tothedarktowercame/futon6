#!/usr/bin/env python3
"""Tier-2 reground, EXPOSITORY layer (mirror of iatc_move_reground.py for proof-moves).

The expository vocab seeds discourse cues for 4 "empirical mint" kinds (difficulty-
assessment, heuristic-plausibility, auxiliary-construction, generalisation). The corpus
uses many more discourse signals; we harvest them from the LLM-classified scope spans
(each scope = a kind + a :source span of expository prose) and feed them back as cues,
then measure expository-move recognition over expository prose — base vs augmented.

Reuses strategy_recognizer.recognize_text by building an expository vocab in its format.

  futon6/.venv/bin/python scripts/expository_reground.py
"""
import glob
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import strategy_recognizer as sr  # noqa: E402
import dp_paper_view as dpv  # noqa: E402

VOCAB = os.path.join(ROOT, "holes/excursions/expository-superpod-vocab.edn")
# Default is the mark4-era 70B output; the stepper passes the RUN's scopes via
# --scopes. Left hardcoded, this measured a stale corpus silently (H19).
SCOPES = os.path.join(ROOT, "data/expository-scope-graphs/loop-run-70b")
# discourse stems that signal an expository (not proof) move — the prior; the corpus
# supplies the actual phrases around them.
DISCOURSE = re.compile(r"\b(note|recall|observ|clear|eas[iy]|obvious|intuit|general|"
                       r"instance|particular|key idea|the key|strateg|goal|motivat|suffic|"
                       r"expect|presum|straightforward|well.known|standard|natural to|"
                       r"roughly|morally|in fact|indeed|of course|for example)\w*", re.I)


def base_vocab():
    """Expository :cues -> recognize_text vocab format (cues as heuristic gestures)."""
    txt = open(VOCAB).read()
    heur = {}
    for block in re.split(r"(?=:kind :)", txt):     # one scope per :kind block (handles nested :hole {})
        km = re.search(r":kind (:[a-z/-]+)", block)
        cm = re.search(r":cues \[([^\]]*)\]", block)
        if not (km and cm):
            continue
        pats = []
        for c in re.findall(r'"([^"]+)"', cm.group(1)):
            pats += [p.strip().lower() for p in re.split(r"/", c) if len(p.strip()) > 2]
        if pats:
            heur[km.group(1)] = pats
    return {"tactics": {}, "heuristic": heur, "residue": [], "conjecture": []}


def harvest_cues():
    """The corpus's own expository signals: discourse-marker phrases in the classified
    scope spans (3-word windows around a discourse stem)."""
    cues = Counter()
    # dpv.build() re-parses the whole paper; without a cache this ran once per
    # SCOPE FILE rather than once per PAPER. On 12 papers / 280 scopes that is a
    # 23x waste; at 4,616 papers x ~30 regions it is ~138k rebuilds and the stage
    # never finishes (E-superpod-hardening H20, 2026-08-06). Same cache shape
    # iatc_lexicon_harvest._text_lines already uses.
    _text_cache = {}

    def _lines_for(pid):
        if pid not in _text_cache:
            try:
                _text_cache[pid] = dpv.build(pid)["text"].split("\n")
            except Exception:
                _text_cache[pid] = []
        return _text_cache[pid]

    for f in glob.glob(os.path.join(SCOPES, "*.edn")):
        # split("_")[0] collapsed old-style ids (math__0310337 -> "math") and
        # aborted the stage on the missing eprint (H19b); one shared parser now.
        try:
            import sys as _sys
            _h = os.path.dirname(os.path.abspath(__file__))
            if _h not in _sys.path:
                _sys.path.insert(0, _h)
            from paper_ids import paper_id_from_name
            pid = paper_id_from_name(f)
        except Exception:
            pid = os.path.basename(f).split("_")[0]
        lines = _lines_for(pid) if pid else []
        t = open(f).read()
        lm = re.search(r":lines \[(\d+) (\d+)\]", t)
        if not lm or not lines:
            continue
        a, b = int(lm.group(1)), int(lm.group(2))
        span = sr.strip_latex(" ".join(lines[a - 1:b])).lower()
        for m in DISCOURSE.finditer(span):
            s, e = m.start(), m.end()
            phrase = " ".join(span[max(0, s - 12):e + 12].split())
            cues[m.group(0).lower()] += 1
    return [c for c, n in cues.most_common()]


def score(vocab, windows):
    bs = Counter()
    for w in windows:
        b, _ = sr.recognize_text(w, vocab)
        bs.update(b)
    g, t, u = bs["grounded"], bs["thin"], bs["ungrounded"]
    tot = g + t + u
    return {"thin": t, "ungrounded": u, "expository-move-recognition": round(t / tot, 3) if tot else 0.0}


def main():
    global SCOPES
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--scopes", default=SCOPES)
    _ap.add_argument("--measure-ids", default=None,
                     help="file of paper ids to measure lift over; default is the "
                          "5 hardcoded goldens, which measure the WRONG corpus on "
                          "any run but the original (E-superpod-hardening H19c)")
    _a = _ap.parse_args()
    SCOPES = _a.scopes if os.path.isabs(_a.scopes) else os.path.join(ROOT, _a.scopes)
    base = base_vocab()
    cues = harvest_cues()
    print(f"base expository cues: {sum(len(v) for v in base['heuristic'].values())} "
          f"across {len(base['heuristic'])} kinds")
    print(f"harvested corpus discourse-cues ({len(cues)}): {', '.join(cues)}\n")
    aug = {**base, "heuristic": {**base["heuristic"], "corpus-discourse": cues}}
    # measure over the goldens' expository prose (full dpv text; proof sentences stay
    # ungrounded in both, so the delta is the pure expository-cue lift)
    pids = ["0705.0452", "0706.1286", "0708.2185", "0712.0724", "0801.0199"]
    if _a.measure_ids:
        pids = [ln.strip() for ln in open(_a.measure_ids) if ln.strip()]
    windows = []
    for p in pids:            # a missing eprint must not abort the stage
        try:
            windows.append(dpv.build(p)["text"])
        except Exception as e:
            print(f"  (skip {p}: {e})")
    if not windows:
        print("no measurable papers — skipping expository reground lift")
        return
    b, a = score(base, windows), score(aug, windows)
    print(f"over {len(windows)} papers' expository prose:")
    print(f"  BASE      : {b}")
    print(f"  AUGMENTED : {a}")
    d = a["expository-move-recognition"] - b["expository-move-recognition"]
    print(f"\n  expository-move recognition: {b['expository-move-recognition']} -> "
          f"{a['expository-move-recognition']}  ({'+' if d >= 0 else ''}{round(d, 3)})")


if __name__ == "__main__":
    main()
