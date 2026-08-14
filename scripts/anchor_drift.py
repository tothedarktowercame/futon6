import sys, glob, re, statistics
sys.path.insert(0, "scripts")
import dp_paper_view as dpv
from paper_ids import proof_pid_from_graph_name

def toks(s):
    return set(w.lower() for w in re.findall(r"[A-Za-z]{4,}", s))

NODE = re.compile(r':text "([^"]{15,120})"[^}]*?:source \{:lines \[(\d+) (\d+)\]')

cache = {}
drift = []
located = total = 0
for f in sorted(glob.glob("data/iatc-argument-graphs/run/*.edn")):
    if "rung2" in f:
        continue
    pid = proof_pid_from_graph_name(f)
    if pid not in cache:
        try:
            cache[pid] = dpv.build(pid)["text"].split("\n")
        except Exception:
            cache[pid] = []
    lines = cache[pid]
    if not lines:
        continue
    t = open(f).read()
    for m in NODE.finditer(t):
        txt, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        tt = toks(txt)
        if len(tt) < 3:
            continue
        total += 1
        best, bestscore = None, 0.0
        for i, ln in enumerate(lines, start=1):
            sc = len(tt & toks(ln)) / len(tt)
            if sc > bestscore:
                bestscore, best = sc, i
        if bestscore >= 0.6:
            located += 1
            drift.append(0 if a <= best <= b else min(abs(best - a), abs(best - b)))

print("node texts confidently located in source: %d/%d" % (located, total))
if drift:
    exact = sum(1 for d in drift if d == 0)
    off = [d for d in drift if d]
    print("  anchor lands ON claimed span : %d/%d (%.0f%%)" % (exact, len(drift), 100.0*exact/len(drift)))
    print("  median drift when off        : %s lines" % (statistics.median(off) if off else 0))
    print("  within 5 lines               : %.0f%%" % (100.0*sum(1 for d in drift if d <= 5)/len(drift)))
    print("  within 10 lines              : %.0f%%" % (100.0*sum(1 for d in drift if d <= 10)/len(drift)))
