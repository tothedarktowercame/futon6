#!/usr/bin/env python3
# Mission-anatomy profiler (ensemble / ehipassiko: compute-and-see).
# For M-web-arxana-missions L3: does a mission's TOPOLOGY shift across eras?
# Hypothesis (Joe): early futons lean on CONCEPTS; later futons lean on
# built MATERIAL (surveys, code paths, agent handoffs). Compare two families:
#   Agency family vs War Machine family.
# Anatomy = the lifecycle-typed sections (mission-lifecycle.md / mission_shapes.clj).
# Concept signal = hits of the distinctive self-representing lexicon (Pass 2).
# Material signal = file paths, commit shas, agent-handoff + survey references.
import json, re
from pathlib import Path

ROOT = Path("/home/joe/code")
LEX = ROOT / "futon6" / "data" / "mission-self-representing-lexicon.json"

FAMILIES = {
    "Agency": [
        "futon3/holes/missions/M-agency-forum.md",
        "futon3/holes/missions/M-agency-rebuild.md",
        "futon3c/holes/missions/M-agency-refactor.md",
    ],
    "War Machine": [
        "futon3c/holes/missions/M-war-machine.md",
        "futon3c/holes/missions/M-war-machine-pilot.md",
        "futon3c/holes/missions/M-war-machine-tuning.md",
    ],
}

PHASES = ["head", "identify", "map", "derive", "argue", "verify",
          "instantiate", "document"]
PHASE_RE = re.compile(r"^#{1,4}\s*(?:\d+\.\s*)?([A-Za-z][A-Za-z-]*)", re.M)

# distinctive-concept vocabulary (top of the SIP-ranked self-rep lexicon)
def load_concepts(n=250):
    d = json.loads(LEX.read_text())
    terms = [r["term"] for r in d["terms"]][:n]
    return set(terms)

CONCEPTS = load_concepts()
WORD = re.compile(r"[a-z][a-z-]+")

# MATERIAL / built-infrastructure signals
RE_PATH = re.compile(r"\b[\w./-]+\.(?:clj|cljs|cljc|py|edn|bb|el|json|md)\b")
RE_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
RE_HANDOFF = re.compile(r"\b(codex|bell|whistle|handoff|hand-off|agency|dispatch|survey|inventory|endpoint|api|http|drawbridge|commit|registry|store)\b", re.I)
RE_MISSION_REF = re.compile(r"\bM-[a-z][a-z0-9-]+")
RE_DATE = re.compile(r"\*\*?Date:?\*\*?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")

def split_phases(text):
    out = {}
    cur = "preamble"
    buf = []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s*(?:\d+\.\s*)?([A-Za-z][A-Za-z-]*)", line)
        if m and m.group(1).lower() in PHASES:
            out[cur] = out.get(cur, "") + "\n".join(buf) + "\n"
            buf = []
            cur = m.group(1).lower()
        else:
            buf.append(line)
    out[cur] = out.get(cur, "") + "\n".join(buf) + "\n"
    return out

def profile(path):
    text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
    words = WORD.findall(text.lower())
    nw = max(1, len(words))
    concept_hits = sum(1 for w in words if w in CONCEPTS)
    material_hits = (len(RE_PATH.findall(text)) + len(RE_SHA.findall(text))
                     + len(RE_HANDOFF.findall(text)))
    phases = split_phases(text)
    phase_w = {p: len(WORD.findall(t.lower())) for p, t in phases.items()}
    date = RE_DATE.search(text)
    return {
        "path": path,
        "date": date.group(1) if date else "?",
        "words": nw,
        "concept_density": 1000 * concept_hits / nw,
        "material_density": 1000 * material_hits / nw,
        "mission_refs": len(set(RE_MISSION_REF.findall(text))),
        "phase_w": phase_w,
        "id_vs_map": (phase_w.get("identify", 0) + phase_w.get("argue", 0),
                      phase_w.get("map", 0) + phase_w.get("instantiate", 0)),
    }

def main():
    print(f"concept vocab: top {len(CONCEPTS)} distinctive terms\n")
    fam_summ = {}
    for fam, paths in FAMILIES.items():
        print(f"================  {fam} family  ================")
        rows = []
        for p in paths:
            try:
                r = profile(p)
            except FileNotFoundError:
                print(f"  MISSING: {p}"); continue
            rows.append(r)
            idw, mapw = r["id_vs_map"]
            lean = "CONCEPT" if r["concept_density"] >= r["material_density"] else "MATERIAL"
            print(f"  {Path(p).name:26} {r['date']}  {r['words']:5d}w  "
                  f"concept={r['concept_density']:5.1f}/1k  material={r['material_density']:5.1f}/1k  "
                  f"->{lean}  refs={r['mission_refs']:2d}  IDENTIFY+ARGUE={idw} MAP+INST={mapw}")
        if rows:
            fam_summ[fam] = {
                "concept": sum(r["concept_density"] for r in rows) / len(rows),
                "material": sum(r["material_density"] for r in rows) / len(rows),
                "refs": sum(r["mission_refs"] for r in rows) / len(rows),
                "words": sum(r["words"] for r in rows) / len(rows),
            }
    print("\n================  FAMILY COMPARISON (mean per mission)  ================")
    print(f"  {'family':14} {'concept/1k':>11} {'material/1k':>12} {'mission-refs':>13} {'words':>7}")
    for fam, s in fam_summ.items():
        print(f"  {fam:14} {s['concept']:11.1f} {s['material']:12.1f} {s['refs']:13.1f} {s['words']:7.0f}")
    if len(fam_summ) == 2:
        a, b = fam_summ.values()
        print(f"\n  material-lean ratio (War Machine / Agency): "
              f"{b['material']/max(0.1,a['material']):.2f}x material, "
              f"{b['concept']/max(0.1,a['concept']):.2f}x concept, "
              f"{b['refs']/max(0.1,a['refs']):.2f}x mission-refs")

if __name__ == "__main__":
    main()
