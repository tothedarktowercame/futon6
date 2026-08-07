"""Anchor precision, measured INSIDE the window the model was shown.

The model gets the passage window as raw text plus its bounds; the correct
anchor for a node is therefore necessarily inside that window. Searching the
whole paper (as a first attempt did) finds other legitimate occurrences of a
restated phrase and reports them as drift, which says more about the locator
than the model. Restricting to the window removes that confound.
"""
import sys, glob, re, statistics
sys.path.insert(0, "scripts")
import dp_paper_view as dpv
from paper_ids import proof_pid_from_graph_name

def toks(s):
    return set(w.lower() for w in re.findall(r"[A-Za-z]{4,}", s))

NODE = re.compile(r':text "([^"]{15,120})"[^}]*?:source \{:lines \[(\d+) (\d+)\]')
PASSAGE = re.compile(r':source \{:lines \[(\d+) (\d+)\], :kind :proof\}')

cache = {}
drift = []
located = total = 0
for f in sorted(glob.glob("data/iatc-argument-graphs/run/*.edn")):
    if "rung2" in f:
        continue
    t = open(f).read()
    pm = PASSAGE.search(t)
    if not pm:
        continue
    wlo, whi = int(pm.group(1)), int(pm.group(2))
    pid = proof_pid_from_graph_name(f)
    if pid not in cache:
        try:
            cache[pid] = dpv.build(pid)["text"].split("\n")
        except Exception:
            cache[pid] = []
    lines = cache[pid]
    if not lines:
        continue
    for m in NODE.finditer(t):
        txt, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        tt = toks(txt)
        if len(tt) < 3:
            continue
        total += 1
        best, bestscore = None, 0.0
        for i in range(wlo, min(whi, len(lines)) + 1):     # window only
            sc = len(tt & toks(lines[i - 1])) / len(tt)
            if sc > bestscore:
                bestscore, best = sc, i
        if bestscore >= 0.6:
            located += 1
            drift.append(0 if a <= best <= b else min(abs(best - a), abs(best - b)))

print("node texts located inside their own window: %d/%d" % (located, total))
if drift:
    exact = sum(1 for d in drift if d == 0)
    off = [d for d in drift if d]
    print("  anchor covers the true line  : %d/%d (%.0f%%)" % (exact, len(drift), 100.0*exact/len(drift)))
    if off:
        print("  median drift when off        : %.0f lines" % statistics.median(off))
        print("  max drift                    : %d lines" % max(off))
    print("  within 3 lines               : %.0f%%" % (100.0*sum(1 for d in drift if d <= 3)/len(drift)))
