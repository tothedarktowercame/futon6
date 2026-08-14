"""What 16 papers gave back — the Figure-1 witnesses, weighted by count."""
import glob, json, re, os, sys
from collections import Counter
sys.path.insert(0, "scripts")

def rd(p): return open(p, errors="replace").read()

G = [g for g in glob.glob("data/iatc-argument-graphs/run/*.edn") if "rung2" not in g]
E = glob.glob("data/expository-scope-graphs/run/*.edn")
C = glob.glob("holes/clean-run/*.clean.edn")
M = glob.glob("data/showcases/ct-anatomy/golden/*.json")

print("== S1 anatomy (deterministic marks) ==")
kinds = Counter(); papers = 0; marks = 0
for f in M:
    try: d = json.loads(rd(f))
    except Exception: continue
    papers += 1
    for m in d.get("marks", []):
        kinds[m.get("kind")] += 1; marks += 1
print(f"  papers with marks: {papers}   total marks: {marks}")
for k, n in kinds.most_common(8): print(f"    {k:22s} {n}")

print("\n== S3 IATC argument graphs ==")
nk = Counter(); rel = Counter(); wk = Counter(); holes = 0; nodes = edges = 0
for f in G:
    t = rd(f)
    for m in re.finditer(r":kind :(object|claim|ref)", t): nk[m.group(1)] += 1; nodes += 1
    for m in re.finditer(r":relation :([a-z-]+)", t): rel[m.group(1)] += 1; edges += 1
    for m in re.finditer(r":warrant \{:kind :([a-z-]+)", t): wk[m.group(1)] += 1
    holes += len(re.findall(r":wanted :", t))
print(f"  graphs {len(G)}   nodes {nodes}   inference edges {edges}   declared holes {holes}")
print(f"    node kinds: {dict(nk)}")
print(f"    top relations: {dict(rel.most_common(6))}   distinct: {len(rel)}")
print(f"    warrants: {dict(wk)}")

print("\n== S4 expository scopes ==")
sk = Counter(); slots = Counter()
for f in E:
    t = rd(f)
    for m in re.finditer(r":kind :([a-z/-]+)", t): sk[m.group(1)] += 1
    for m in re.finditer(r":slot-fill \{:([a-z-]+)", t): slots[m.group(1)] += 1
print(f"  scope graphs {len(E)}   distinct scope kinds {len(sk)}")
for k, n in sk.most_common(8): print(f"    {k:34s} {n}")
print(f"    slot types filled: {dict(slots.most_common(6))}")

print("\n== S7 CLean typing ==")
meth = Counter(); sorries = 0
for f in C:
    t = rd(f)
    for m in re.finditer(r":method :([a-z-]+)", t): meth[m.group(1)] += 1
    sorries += len(re.findall(r":kind :sorry", t))
print(f"  typed proofs {len(C)}   typed boxes {sum(meth.values())}   sorry-holes carried {sorries}")
for k, n in meth.most_common(8): print(f"    {k:30s} {n}")

print("\n== S5 comprehension / S10 lexicon / S12 curve ==")
for p, label in [("data/runs/mark7z/inference-lexicon.json", "lexicon"),
                 ("data/runs/mark7z/accretion-curve.json", "curve"),
                 ("data/runs/mark7z/structural-canon.json", "canon")]:
    if os.path.exists(p):
        d = json.load(open(p))
        if label == "lexicon":
            print(f"  lexicon: {d['distinct_entries']} entries, {len(d['grammar'])} relation types, "
                  f"conf mean {d['confidence']['mean']}")
        elif label == "curve":
            print(f"  curve: {len(d['points'])} checkpoints, rise {d['rise']}, rising={d['rising']}")
        else:
            print(f"  canon: {len(d['shapes'])} shapes, {len(d['signatures'])} paper signatures, "
                  f"twin-sim mean {d['paper_twin_sim']['mean']}")
