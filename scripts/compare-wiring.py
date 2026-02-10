#!/usr/bin/env python3
"""Compare wiring diagram structure of the same concept across two sources.

Takes two texts (PlanetMath and nLab) for the same concept and produces
a side-by-side classical wiring analysis, showing where the diagrams
agree and where they diverge.

Usage:
    python scripts/compare-wiring.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# --- Reuse from validate-ct.py ---

SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)"),
    ("define", r"\bDefine\s+\$([^$]+)\$\s*(:=|=|\\equiv)\s*([^.,$]+)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$([^$]+)\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?([^$.]{1,60})"),
    ("for-any", r"\b(?:for\s+)?(any|every|each|all)\s+\$([^$]+)\$"),
    ("where-binding", r"\bwhere\s+\$([^$]+)\$\s+(is|denotes|represents)\s+([^.,$]+)"),
    ("set-notation", r"\$([^$]*\\in\s+[^$]+)\$"),
]

CLASSICAL_TO_METATHEORY = {
    "let-binding": "bind/let",
    "define": "bind/define",
    "assume": "assume/explicit",
    "consider": "assume/consider",
    "for-any": "quant/universal",
    "where-binding": "constrain/where",
    "set-notation": "constrain/such-that",
}

WIRE_REGEXES = [
    ("wire/adversative", r"\b(?:but|however|on the other hand|nevertheless|yet)\b", re.IGNORECASE),
    ("wire/causal", r"\b(?:because|since|the reason is|given that)\b", re.IGNORECASE),
    ("wire/consequential", r"\b(?:therefore|thus|hence|it follows|so that|note that|in fact)\b", re.IGNORECASE),
    ("wire/clarifying", r"\b(?:that is|in other words|namely|more precisely|i\.e\.)\b", re.IGNORECASE),
    ("wire/intuitive", r"\b(?:intuitively|roughly speaking|heuristically)\b", re.IGNORECASE),
]

PORT_REGEXES = [
    ("port/that-noun", r"\bthat\s+(?:root|function|map|functor|morphism|object|category|space|set|group|composition|diagram|square)\b", re.IGNORECASE),
    ("port/this-noun", r"\bthis\s+(?:equation|operator|means|functor|morphism|diagram|category|map|approach|example|construction|definition|property)\b", re.IGNORECASE),
    ("port/the-above", r"\b(?:the above|the preceding|the previous|from above|defined above|as above)\b", re.IGNORECASE),
    ("port/the-same", r"\bthe same\s+\w+", re.IGNORECASE),
    ("port/such", r"\bsuch (?:a|an)\s+\w+", re.IGNORECASE),
    ("port/similarly", r"\b(?:similarly|analogously|dually)\b", re.IGNORECASE),
    ("port/likewise", r"\b(?:likewise|correspondingly)\b", re.IGNORECASE),
]

LABEL_REGEXES = [
    ("explain/meaning", r"\bthis means\b", re.IGNORECASE),
    ("explain/think-of", r"\b(?:think of|can be thought of|to be thought of)\b", re.IGNORECASE),
    ("explain/the-idea", r"\bthe (?:idea|trick|key|point) is\b", re.IGNORECASE),
    ("correct/actually", r"\bactually\b", re.IGNORECASE),
    ("correct/subtlety", r"\b(?:subtlety|subtle)\b", re.IGNORECASE),
    ("epistemic/can-show", r"\bone can (?:show|verify|check)\b", re.IGNORECASE),
    ("epistemic/known", r"\b(?:well known|well-known|it is known)\b", re.IGNORECASE),
    ("construct/exists", r"\bthere (?:is|exists|exist)\b", re.IGNORECASE),
    ("construct/explicit", r"\bexplicitly\b", re.IGNORECASE),
    ("strategy/generalize", r"\b(?:generalize|generalise|more generally)\b", re.IGNORECASE),
    ("strategy/example", r"\b(?:for example|for instance|e\.g\.)\b", re.IGNORECASE),
    ("strategy/analogy", r"\b(?:an advantage|analogous)\b", re.IGNORECASE),
]


def analyze(text, label):
    """Run full classical wiring analysis on text. Return structured results."""
    # Components
    components = []
    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, text):
            components.append({
                "type": CLASSICAL_TO_METATHEORY.get(stype, f"scope/{stype}"),
                "match": m.group()[:80],
                "pos": m.start(),
            })
    components.sort(key=lambda x: x["pos"])

    # Wires
    wires = []
    for wtype, pattern, flags in WIRE_REGEXES:
        for m in re.finditer(pattern, text, flags):
            wires.append({
                "type": wtype,
                "match": m.group(),
                "pos": m.start(),
            })
    wires.sort(key=lambda x: x["pos"])

    # Ports
    ports = []
    for ptype, pattern, flags in PORT_REGEXES:
        for m in re.finditer(pattern, text, flags):
            ports.append({
                "type": ptype,
                "match": m.group()[:60],
                "pos": m.start(),
            })
    ports.sort(key=lambda x: x["pos"])

    # Labels
    labels = Counter()
    for ltype, pattern, flags in LABEL_REGEXES:
        count = len(re.findall(pattern, text, flags))
        if count > 0:
            labels[ltype] = count

    return {
        "label": label,
        "text_length": len(text),
        "components": components,
        "wires": wires,
        "ports": ports,
        "labels": labels,
    }


def component_signature(analysis):
    """Extract the type signature of components (what scope types appear)."""
    return Counter(c["type"] for c in analysis["components"])


def wire_signature(analysis):
    """Extract the type signature of wires (what connective types appear)."""
    return Counter(w["type"] for w in analysis["wires"])


def port_signature(analysis):
    """Extract the type signature of ports."""
    return Counter(p["type"] for p in analysis["ports"])


def print_analysis(a):
    """Print a human-readable analysis."""
    print(f"\n{'='*60}")
    print(f"  {a['label']}")
    print(f"  {a['text_length']} chars")
    print(f"{'='*60}")

    print(f"\n  COMPONENTS ({len(a['components'])})")
    csig = component_signature(a)
    for ctype, count in csig.most_common():
        print(f"    {ctype:25s} {count}")
    for c in a["components"][:8]:
        print(f"    [{c['type']:25s}] {c['match'][:65]}")
    if len(a["components"]) > 8:
        print(f"    ... and {len(a['components'])-8} more")

    print(f"\n  WIRES ({len(a['wires'])})")
    wsig = wire_signature(a)
    for wtype, count in wsig.most_common():
        print(f"    {wtype:25s} {count}")

    print(f"\n  PORTS ({len(a['ports'])})")
    psig = port_signature(a)
    for ptype, count in psig.most_common():
        print(f"    {ptype:25s} {count}")
    for p in a["ports"][:5]:
        print(f"    [{p['type']:25s}] {p['match']}")

    print(f"\n  LABELS ({sum(a['labels'].values())})")
    for ltype, count in a["labels"].most_common():
        print(f"    {ltype:25s} {count}")


def compare(a1, a2):
    """Compare two analyses and identify structural differences."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {a1['label']} vs {a2['label']}")
    print(f"{'='*60}")

    # Component signatures
    csig1 = component_signature(a1)
    csig2 = component_signature(a2)
    all_ctypes = sorted(set(list(csig1.keys()) + list(csig2.keys())))

    print(f"\n  COMPONENT SIGNATURE")
    print(f"    {'type':25s} {'PM':>5s} {'nLab':>5s} {'delta':>6s}")
    print(f"    {'-'*25} {'-'*5} {'-'*5} {'-'*6}")
    for ct in all_ctypes:
        v1, v2 = csig1.get(ct, 0), csig2.get(ct, 0)
        delta = v2 - v1
        marker = " " if delta == 0 else ("+" if delta > 0 else "")
        print(f"    {ct:25s} {v1:5d} {v2:5d} {marker}{delta:+5d}" if delta != 0 else
              f"    {ct:25s} {v1:5d} {v2:5d}      =")

    # Wire signatures
    wsig1 = wire_signature(a1)
    wsig2 = wire_signature(a2)
    all_wtypes = sorted(set(list(wsig1.keys()) + list(wsig2.keys())))

    print(f"\n  WIRE SIGNATURE")
    print(f"    {'type':25s} {'PM':>5s} {'nLab':>5s} {'delta':>6s}")
    print(f"    {'-'*25} {'-'*5} {'-'*5} {'-'*6}")
    for wt in all_wtypes:
        v1, v2 = wsig1.get(wt, 0), wsig2.get(wt, 0)
        delta = v2 - v1
        marker = " " if delta == 0 else ("+" if delta > 0 else "")
        print(f"    {wt:25s} {v1:5d} {v2:5d} {marker}{delta:+5d}" if delta != 0 else
              f"    {wt:25s} {v1:5d} {v2:5d}      =")

    # Port signatures
    psig1 = port_signature(a1)
    psig2 = port_signature(a2)
    all_ptypes = sorted(set(list(psig1.keys()) + list(psig2.keys())))

    print(f"\n  PORT SIGNATURE")
    print(f"    {'type':25s} {'PM':>5s} {'nLab':>5s} {'delta':>6s}")
    print(f"    {'-'*25} {'-'*5} {'-'*5} {'-'*6}")
    for pt in all_ptypes:
        v1, v2 = psig1.get(pt, 0), psig2.get(pt, 0)
        delta = v2 - v1
        marker = " " if delta == 0 else ("+" if delta > 0 else "")
        print(f"    {pt:25s} {v1:5d} {v2:5d} {marker}{delta:+5d}" if delta != 0 else
              f"    {pt:25s} {v1:5d} {v2:5d}      =")

    # Label signatures
    lsig1 = a1["labels"]
    lsig2 = a2["labels"]
    all_ltypes = sorted(set(list(lsig1.keys()) + list(lsig2.keys())))

    if all_ltypes:
        print(f"\n  WIRE LABEL SIGNATURE")
        print(f"    {'type':25s} {'PM':>5s} {'nLab':>5s} {'delta':>6s}")
        print(f"    {'-'*25} {'-'*5} {'-'*5} {'-'*6}")
        for lt in all_ltypes:
            v1, v2 = lsig1.get(lt, 0), lsig2.get(lt, 0)
            delta = v2 - v1
            marker = " " if delta == 0 else ("+" if delta > 0 else "")
            print(f"    {lt:25s} {v1:5d} {v2:5d} {marker}{delta:+5d}" if delta != 0 else
                  f"    {lt:25s} {v1:5d} {v2:5d}      =")

    # Ratios (normalize by text length)
    len1, len2 = a1["text_length"], a2["text_length"]
    print(f"\n  DENSITY (per 1000 chars)")
    for name, count1, count2 in [
        ("components", len(a1["components"]), len(a2["components"])),
        ("wires", len(a1["wires"]), len(a2["wires"])),
        ("ports", len(a1["ports"]), len(a2["ports"])),
    ]:
        d1 = count1 / len1 * 1000 if len1 else 0
        d2 = count2 / len2 * 1000 if len2 else 0
        print(f"    {name:15s} PM={d1:.2f}  nLab={d2:.2f}  ratio={d2/d1:.2f}" if d1 > 0 else
              f"    {name:15s} PM={d1:.2f}  nLab={d2:.2f}")

    # Qualitative summary
    print(f"\n  REWIRING SUMMARY")
    # What PM has that nLab doesn't (extra wires/ports)
    pm_extra_wires = {wt: wsig1[wt] - wsig2.get(wt, 0) for wt in wsig1 if wsig1[wt] > wsig2.get(wt, 0)}
    nlab_extra_wires = {wt: wsig2[wt] - wsig1.get(wt, 0) for wt in wsig2 if wsig2[wt] > wsig1.get(wt, 0)}

    if pm_extra_wires:
        print(f"    PM uses MORE:  {dict(pm_extra_wires)}")
    if nlab_extra_wires:
        print(f"    nLab uses MORE: {dict(nlab_extra_wires)}")

    pm_extra_ports = {pt: psig1[pt] - psig2.get(pt, 0) for pt in psig1 if psig1[pt] > psig2.get(pt, 0)}
    nlab_extra_ports = {pt: psig2[pt] - psig1.get(pt, 0) for pt in psig2 if psig2[pt] > psig1.get(pt, 0)}

    if pm_extra_ports:
        print(f"    PM anaphora:   {dict(pm_extra_ports)}")
    if nlab_extra_ports:
        print(f"    nLab anaphora:  {dict(nlab_extra_ports)}")


def extract_pm_body(tex_path):
    """Extract body from PlanetMath .tex file."""
    with open(tex_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_nlab_body(md_path):
    """Extract prose from nLab markdown page (strip markup)."""
    with open(md_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Remove nLab-specific markup
    # Strip {: .toc} blocks, [[!include]], [[!redirects]]
    content = re.sub(r"\+--.*?=--", "", content, flags=re.DOTALL)
    content = re.sub(r"\[\[!include[^\]]*\]\]", "", content)
    content = re.sub(r"\[\[!redirects[^\]]*\]\]", "", content)
    # Strip wiki-link markup but keep text: [[foo|bar]] → bar, [[foo]] → foo
    content = re.sub(r"\[\[([^|\]]*)\|([^\]]*)\]\]", r"\2", content)
    content = re.sub(r"\[\[([^\]]*)\]\]", r"\1", content)
    # Strip markdown headers
    content = re.sub(r"^#+\s+.*$", "", content, flags=re.MULTILINE)
    # Strip {#anchors}
    content = re.sub(r"\{#\w+\}", "", content)
    # Strip \begin{example}, \begin{definition} etc. markers but keep content
    content = re.sub(r"\\begin\{(?:example|definition|remark|theorem|proof)\}(?:\[.*?\])?", "", content)
    content = re.sub(r"\\end\{(?:example|definition|remark|theorem|proof)\}", "", content)
    content = re.sub(r"\\label\{[^}]*\}", "", content)
    content = re.sub(r"\\linebreak", "", content)
    # Strip reference section
    content = re.sub(r"## References.*", "", content, flags=re.DOTALL)
    content = re.sub(r"## Related entries.*", "", content, flags=re.DOTALL)

    return content.strip()


def main():
    pm_dir = Path("/home/joe/code/planetmath/18_Category_theory_homological_algebra")
    nlab_dir = Path("/home/joe/code/nlab-content/pages")

    # Natural Transformation comparison
    print("\n" + "#"*60)
    print("# WIRING DIAGRAM COMPARISON: Natural Transformation")
    print("#"*60)

    pm_text = extract_pm_body(pm_dir / "18-00-CompositionsOfNaturalTransformations.tex")
    nlab_text = extract_nlab_body(nlab_dir / "2/2/1/0/122/content.md")

    pm_analysis = analyze(pm_text, "PlanetMath: compositions of natural transformations")
    nlab_analysis = analyze(nlab_text, "nLab: natural transformation")

    print_analysis(pm_analysis)
    print_analysis(nlab_analysis)
    compare(pm_analysis, nlab_analysis)

    # Functor Category comparison
    print("\n\n" + "#"*60)
    print("# WIRING DIAGRAM COMPARISON: Functor Category")
    print("#"*60)

    pm_text2 = extract_pm_body(pm_dir / "18-00-FunctorCategory.tex")
    nlab_path2 = nlab_dir / "5/2/2/0/225/content.md"
    if nlab_path2.exists():
        nlab_text2 = extract_nlab_body(nlab_path2)
        pm_analysis2 = analyze(pm_text2, "PlanetMath: functor category")
        nlab_analysis2 = analyze(nlab_text2, "nLab: functor category")
        print_analysis(pm_analysis2)
        print_analysis(nlab_analysis2)
        compare(pm_analysis2, nlab_analysis2)

    # Write results
    outpath = Path("/home/joe/code/futon6/data/ct-validation/comparison.json")
    results = {
        "natural_transformation": {
            "pm": {
                "components": len(pm_analysis["components"]),
                "wires": len(pm_analysis["wires"]),
                "ports": len(pm_analysis["ports"]),
                "component_sig": dict(component_signature(pm_analysis)),
                "wire_sig": dict(wire_signature(pm_analysis)),
                "port_sig": dict(port_signature(pm_analysis)),
            },
            "nlab": {
                "components": len(nlab_analysis["components"]),
                "wires": len(nlab_analysis["wires"]),
                "ports": len(nlab_analysis["ports"]),
                "component_sig": dict(component_signature(nlab_analysis)),
                "wire_sig": dict(wire_signature(nlab_analysis)),
                "port_sig": dict(port_signature(nlab_analysis)),
            },
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nWritten comparison to {outpath}")


if __name__ == "__main__":
    main()
