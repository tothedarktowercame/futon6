#!/usr/bin/env python3
"""Validate wiring diagram metatheory on PlanetMath category theory entries.

Reads 313 CT .tex files, runs classical NER + scope detection, cross-references
with nLab hyperreal dictionary, and produces a validation report.

This establishes the CLASSICAL BASELINE for CT entries. The wiring diagram
metatheory (components + ports + wires) will then be validated by LLM
extraction on a golden subset.

Usage:
    python scripts/validate-ct.py [--golden N]

Outputs:
    data/ct-validation/
        manifest.json     — summary statistics
        entities.json     — parsed CT entries with body text
        ner-terms.json    — NER hits per entry
        scopes.json       — classical scope detections
        bridge.json       — PM↔nLab bridge concepts
        golden/           — golden subset for LLM validation
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter

# Reuse superpod-job infrastructure
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# --- Baked-in scope regexes (from superpod-job.py) ---

SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)"),
    ("define", r"\bDefine\s+\$([^$]+)\$\s*(:=|=|\\equiv)\s*([^.,$]+)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$([^$]+)\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?([^$.]{1,60})"),
    ("for-any", r"\b(?:for\s+)?(any|every|each|all)\s+\$([^$]+)\$"),
    ("where-binding", r"\bwhere\s+\$([^$]+)\$\s+(is|denotes|represents)\s+([^.,$]+)"),
    ("set-notation", r"\$([^$]*\\in\s+[^$]+)\$"),
]

# --- Metatheory scope types mapped from classical patterns ---
# classical pattern → metatheory component type

CLASSICAL_TO_METATHEORY = {
    "let-binding": "bind/let",
    "define": "bind/define",
    "assume": "assume/explicit",
    "consider": "assume/consider",
    "for-any": "quant/universal",
    "where-binding": "constrain/where",
    "set-notation": "constrain/such-that",  # approximate
}

# --- Connective patterns (wire detection, classical) ---

WIRE_REGEXES = [
    ("wire/adversative", r"\b(?:but|however|on the other hand|nevertheless|yet)\b", re.IGNORECASE),
    ("wire/causal", r"\b(?:because|since|the reason is|given that)\b", re.IGNORECASE),
    ("wire/consequential", r"\b(?:therefore|thus|hence|it follows|so that|note that|in fact)\b", re.IGNORECASE),
    ("wire/clarifying", r"\b(?:that is|in other words|namely|more precisely|i\.e\.)\b", re.IGNORECASE),
    ("wire/intuitive", r"\b(?:intuitively|roughly speaking|heuristically)\b", re.IGNORECASE),
]

# --- Port patterns (anaphora detection, classical) ---

PORT_REGEXES = [
    ("port/that-noun", r"\bthat\s+(?:root|function|map|functor|morphism|object|category|space|set|group)\b", re.IGNORECASE),
    ("port/this-noun", r"\bthis\s+(?:equation|operator|means|functor|morphism|diagram|category|map)\b", re.IGNORECASE),
    ("port/the-above", r"\b(?:the above|the preceding|the previous)\s+\w+", re.IGNORECASE),
    ("port/the-same", r"\bthe same\s+\w+", re.IGNORECASE),
    ("port/such", r"\bsuch (?:a|an)\s+\w+", re.IGNORECASE),
    ("port/similarly", r"\b(?:similarly|analogously)\b", re.IGNORECASE),
    ("port/likewise", r"\b(?:likewise|correspondingly)\b", re.IGNORECASE),
]

# --- Wire label patterns (reasoning moves on wires) ---

LABEL_REGEXES = [
    ("explain/meaning", r"\bthis means\b", re.IGNORECASE),
    ("explain/think-of", r"\b(?:think of|can be thought of)\b", re.IGNORECASE),
    ("explain/the-idea", r"\bthe (?:idea|trick|key|point) is\b", re.IGNORECASE),
    ("correct/actually", r"\bactually\b", re.IGNORECASE),
    ("correct/subtlety", r"\b(?:subtlety|subtle)\b", re.IGNORECASE),
    ("epistemic/can-show", r"\bone can (?:show|verify|check)\b", re.IGNORECASE),
    ("epistemic/known", r"\b(?:well known|well-known|it is known)\b", re.IGNORECASE),
    ("construct/exists", r"\bthere (?:is|exists|exist)\b", re.IGNORECASE),
    ("construct/explicit", r"\bexplicitly\b", re.IGNORECASE),
    ("strategy/generalize", r"\b(?:generalize|generalise|more generally)\b", re.IGNORECASE),
    ("strategy/example", r"\b(?:for example|for instance|e\.g\.)\b", re.IGNORECASE),
]


def extract_body(tex_path):
    """Extract body text from a PlanetMath .tex file.

    Returns (body_text, metadata_dict) where body is between
    \\begin{document} and \\end{document}.
    """
    with open(tex_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Extract metadata from preamble
    meta = {}
    for tag in ["pmcanonicalname", "pmtitle", "pmtype", "pmauthor"]:
        m = re.search(rf"\\{tag}\{{([^}}]*)\}}", content)
        if m:
            meta[tag.replace("pm", "")] = m.group(1)

    # Extract defines
    defines = re.findall(r"\\pmdefines\{([^}]*)\}", content)
    if defines:
        meta["defines"] = defines

    # Extract MSC codes
    mscs = re.findall(r"\\pmclassification\{msc\}\{([^}]*)\}", content)
    if mscs:
        meta["msc_codes"] = mscs

    # Extract body
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", content, re.DOTALL)
    if m:
        body = m.group(1).strip()
    else:
        body = ""

    return body, meta


def load_ner_kernel(path):
    """Load NER kernel from TSV. Returns (singles, multi_index, count)."""
    singles = {}
    multi_index = {}
    multi_count = 0
    skip_prefixes = ("$", "(", '"', "-")

    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            term_lower = parts[0].strip()
            term_orig = parts[1].strip()
            canon = parts[3].strip() if len(parts) > 3 else term_lower

            if not term_lower or any(term_lower.startswith(p) for p in skip_prefixes):
                continue
            if len(term_lower) < 3:
                continue

            words = term_lower.split()
            if len(words) == 1:
                singles[term_lower] = (term_orig, canon)
            else:
                first_key = None
                for w in words:
                    if len(w) >= 3:
                        first_key = w
                        break
                if first_key is None:
                    first_key = words[0]
                if first_key not in multi_index:
                    multi_index[first_key] = []
                multi_index[first_key].append((term_lower, term_orig, canon))
                multi_count += 1

    return singles, multi_index, multi_count


def spot_terms(text, singles, multi_index):
    """Spot NER terms in text. Returns list of {term, term_lower, canon}."""
    text_lower = text.lower()
    words = text_lower.split()
    hits = {}

    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in singles:
            hits[clean] = singles[clean]

    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in multi_index:
            for term_lower, term_orig, canon in multi_index[clean]:
                if term_lower not in hits and term_lower in text_lower:
                    hits[term_lower] = (term_orig, canon)

    return [{"term": orig, "term_lower": tl, "canon": canon}
            for tl, (orig, canon) in sorted(hits.items())]


def detect_scopes(entity_id, text):
    """Detect scope bindings (components) in text."""
    scopes = []
    scope_idx = 0

    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, text):
            scope_id = f"{entity_id}:c-{scope_idx:03d}"
            scope_idx += 1

            metatype = CLASSICAL_TO_METATHEORY.get(stype, f"scope/{stype}")

            ends = [{"role": "entity", "ident": entity_id}]
            if stype == "let-binding":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "type", "text": m.group(3).strip()[:80]})
            elif stype == "define":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "value", "text": m.group(3).strip()[:80]})
            elif stype == "assume":
                ends.append({"role": "condition", "latex": m.group(3).strip()})
            elif stype == "consider":
                obj = (m.group(2) or "").strip()
                if obj:
                    ends.append({"role": "object", "text": obj[:80]})
            elif stype == "for-any":
                ends.append({"role": "quantifier", "text": m.group(1)})
                ends.append({"role": "symbol", "latex": m.group(2).strip()})
            elif stype == "where-binding":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "description", "text": m.group(3).strip()[:80]})
            elif stype == "set-notation":
                ends.append({"role": "membership", "latex": m.group(1).strip()})

            scopes.append({
                "hx/id": scope_id,
                "hx/role": "component",
                "hx/type": metatype,
                "hx/ends": ends,
                "hx/content": {"match": m.group()[:120], "position": m.start()},
                "hx/labels": ["component", stype],
            })

    return scopes


def detect_wires(entity_id, text):
    """Detect connective wires in text (classical approximation)."""
    wires = []
    wire_idx = 0
    for wtype, pattern, flags in WIRE_REGEXES:
        for m in re.finditer(pattern, text, flags):
            wire_id = f"{entity_id}:w-{wire_idx:03d}"
            wire_idx += 1
            wires.append({
                "hx/id": wire_id,
                "hx/role": "wire",
                "hx/type": wtype,
                "hx/content": {"match": m.group()[:60], "position": m.start()},
            })
    return wires


def detect_ports(entity_id, text):
    """Detect anaphoric ports in text (classical approximation)."""
    ports = []
    port_idx = 0
    for ptype, pattern, flags in PORT_REGEXES:
        for m in re.finditer(pattern, text, flags):
            port_id = f"{entity_id}:p-{port_idx:03d}"
            port_idx += 1
            ports.append({
                "hx/id": port_id,
                "hx/role": "port",
                "hx/type": ptype,
                "hx/content": {"match": m.group()[:60], "position": m.start()},
            })
    return ports


def detect_labels(entity_id, text):
    """Detect wire labels (reasoning moves) in text."""
    labels = []
    for ltype, pattern, flags in LABEL_REGEXES:
        count = len(re.findall(pattern, text, flags))
        if count > 0:
            labels.append({"type": ltype, "count": count})
    return labels


def load_nlab_objects(hyperreal_path):
    """Load nLab object names from hyperreal.json (streaming, memory-safe).

    Returns dict of lowercased name → {id, name, term_count}.
    """
    objects = {}
    # hyperreal.json is ~61MB, but the objects section is a dict
    # We need to be careful with memory
    with open(hyperreal_path) as f:
        data = json.load(f)

    for name, obj in data.get("objects", {}).items():
        objects[name.lower().strip()] = {
            "id": obj.get("id", ""),
            "name": name,
            "term_count": obj.get("term_count", 0),
        }

    return objects


def find_bridge_concepts(pm_entries, nlab_objects):
    """Find concepts that exist in both PlanetMath CT and nLab.

    Returns list of {pm_id, pm_title, nlab_name, nlab_id}.
    """
    bridges = []
    for entry in pm_entries:
        title = entry.get("title", "").lower().strip()
        canon = entry.get("canonicalname", "").lower().strip()

        # Try exact title match
        if title in nlab_objects:
            bridges.append({
                "pm_id": entry["entity_id"],
                "pm_title": entry["title"],
                "nlab_name": nlab_objects[title]["name"],
                "nlab_id": nlab_objects[title]["id"],
                "match_type": "title",
            })
        # Try defines
        elif "defines" in entry:
            for d in entry["defines"]:
                dl = d.lower().strip()
                if dl in nlab_objects:
                    bridges.append({
                        "pm_id": entry["entity_id"],
                        "pm_title": entry["title"],
                        "pm_defines": d,
                        "nlab_name": nlab_objects[dl]["name"],
                        "nlab_id": nlab_objects[dl]["id"],
                        "match_type": "defines",
                    })
                    break  # one bridge per PM entry

    return bridges


def select_golden(entries, bridge_concepts, n=20):
    """Select golden subset for LLM validation.

    Prioritises entries that:
    1. Have nLab counterparts (bridge concepts)
    2. Have rich body text (more scope bindings likely)
    3. Cover diverse entry types (definitions, theorems, examples)
    """
    bridge_ids = {b["pm_id"] for b in bridge_concepts}

    # Score each entry
    scored = []
    for e in entries:
        score = 0
        if e["entity_id"] in bridge_ids:
            score += 10  # bridge bonus
        score += min(len(e.get("body", "")), 2000) / 200  # body length (max 10 pts)
        score += e.get("scope_count", 0) * 2  # scope richness
        score += e.get("wire_count", 0)        # wire richness
        score += e.get("port_count", 0) * 1.5  # port richness
        scored.append((score, e))

    scored.sort(key=lambda x: -x[0])

    # Take top N, ensuring type diversity
    selected = []
    types_seen = Counter()
    for score, e in scored:
        etype = e.get("type", "Definition")
        if types_seen[etype] >= n // 3 + 1:
            continue  # don't over-represent one type
        selected.append(e)
        types_seen[etype] += 1
        if len(selected) >= n:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Validate wiring diagram metatheory on PlanetMath CT")
    parser.add_argument("--golden", type=int, default=20,
                        help="Number of golden entries for LLM validation (default: 20)")
    parser.add_argument("--pm-dir", type=str,
                        default="/home/joe/code/planetmath/18_Category_theory_homological_algebra",
                        help="PlanetMath CT .tex directory")
    parser.add_argument("--ner-kernel", type=str,
                        default="/home/joe/code/futon6/data/ner-kernel/terms.tsv",
                        help="NER kernel TSV path")
    parser.add_argument("--hyperreal", type=str,
                        default="/home/joe/code/futon6/data/hyperreal.json",
                        help="nLab hyperreal dictionary path")
    parser.add_argument("--output-dir", type=str,
                        default="/home/joe/code/futon6/data/ct-validation",
                        help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "golden").mkdir(exist_ok=True)

    t0 = time.time()

    # --- Load NER kernel ---
    print(f"[1/6] Loading NER kernel from {args.ner_kernel}")
    singles, multi_index, multi_count = load_ner_kernel(args.ner_kernel)
    print(f"       {len(singles)} single-word + {multi_count} multi-word terms")

    # --- Parse CT .tex files ---
    print(f"\n[2/6] Parsing CT entries from {args.pm_dir}")
    tex_files = sorted(Path(args.pm_dir).glob("*.tex"))
    print(f"       Found {len(tex_files)} .tex files")

    entries = []
    empty_count = 0
    for tf in tex_files:
        body, meta = extract_body(tf)
        if not body:
            empty_count += 1
            continue

        entity_id = f"pm-ct-{meta.get('canonicalname', tf.stem)}"
        entry = {
            "entity_id": entity_id,
            "source_file": tf.name,
            "title": meta.get("title", tf.stem),
            "type": meta.get("type", "Unknown"),
            "canonicalname": meta.get("canonicalname", ""),
            "body": body,
            "body_length": len(body),
        }
        if "defines" in meta:
            entry["defines"] = meta["defines"]
        if "msc_codes" in meta:
            entry["msc_codes"] = meta["msc_codes"]
        entries.append(entry)

    print(f"       Parsed {len(entries)} entries ({empty_count} empty bodies skipped)")

    # --- Run NER + scope detection + wire/port detection ---
    print(f"\n[3/6] Running classical analysis (NER + components + wires + ports)")
    all_ner = []
    all_scopes = []
    all_wires = []
    all_ports = []
    all_labels = []

    component_type_freq = Counter()
    wire_type_freq = Counter()
    port_type_freq = Counter()
    label_type_freq = Counter()

    for entry in entries:
        eid = entry["entity_id"]
        body = entry["body"]

        # NER
        terms = spot_terms(body, singles, multi_index)
        all_ner.append({"entity_id": eid, "count": len(terms), "terms": terms})
        entry["ner_count"] = len(terms)

        # Components (scopes)
        scopes = detect_scopes(eid, body)
        all_scopes.extend(scopes)
        entry["scope_count"] = len(scopes)
        for s in scopes:
            component_type_freq[s["hx/type"]] += 1

        # Wires (connectives)
        wires = detect_wires(eid, body)
        all_wires.extend(wires)
        entry["wire_count"] = len(wires)
        for w in wires:
            wire_type_freq[w["hx/type"]] += 1

        # Ports (anaphora)
        ports = detect_ports(eid, body)
        all_ports.extend(ports)
        entry["port_count"] = len(ports)
        for p in ports:
            port_type_freq[p["hx/type"]] += 1

        # Wire labels
        labels = detect_labels(eid, body)
        all_labels.append({"entity_id": eid, "labels": labels})
        entry["label_count"] = sum(l["count"] for l in labels)
        for l in labels:
            label_type_freq[l["type"]] += l["count"]

    total_ner = sum(e["ner_count"] for e in entries)
    total_scopes = len(all_scopes)
    total_wires = len(all_wires)
    total_ports = len(all_ports)
    total_labels = sum(e["label_count"] for e in entries)

    print(f"       NER:        {total_ner} hits ({total_ner/len(entries):.1f}/entry)")
    print(f"       Components: {total_scopes} ({total_scopes/len(entries):.2f}/entry)")
    print(f"       Wires:      {total_wires} ({total_wires/len(entries):.2f}/entry)")
    print(f"       Ports:      {total_ports} ({total_ports/len(entries):.2f}/entry)")
    print(f"       Labels:     {total_labels} ({total_labels/len(entries):.2f}/entry)")

    # --- nLab bridge ---
    print(f"\n[4/6] Finding PM↔nLab bridge concepts")
    if os.path.exists(args.hyperreal):
        nlab_objects = load_nlab_objects(args.hyperreal)
        print(f"       Loaded {len(nlab_objects)} nLab objects")
        bridges = find_bridge_concepts(entries, nlab_objects)
        print(f"       Found {len(bridges)} bridge concepts ({len(bridges)/len(entries)*100:.0f}% of PM-CT)")
        del nlab_objects  # free memory
    else:
        print(f"       WARNING: hyperreal.json not found at {args.hyperreal}")
        bridges = []

    # --- Select golden subset ---
    print(f"\n[5/6] Selecting golden-{args.golden} for LLM validation")
    golden = select_golden(entries, bridges, n=args.golden)
    print(f"       Selected {len(golden)} entries:")
    for g in golden:
        bridge_mark = "*" if any(b["pm_id"] == g["entity_id"] for b in bridges) else " "
        print(f"         {bridge_mark} {g['title'][:50]:50s} [{g['type']:12s}] "
              f"C={g['scope_count']:2d} W={g['wire_count']:2d} P={g['port_count']:2d}")

    # --- Write outputs ---
    print(f"\n[6/6] Writing outputs to {outdir}")

    # Entities (without body text for size)
    entities_out = []
    for e in entries:
        out = {k: v for k, v in e.items() if k != "body"}
        out["body_preview"] = e["body"][:200]
        entities_out.append(out)

    with open(outdir / "entities.json", "w") as f:
        json.dump(entities_out, f, ensure_ascii=False, indent=1)
    print(f"       entities.json: {len(entries)} entries")

    with open(outdir / "ner-terms.json", "w") as f:
        json.dump(all_ner, f, ensure_ascii=False)
    print(f"       ner-terms.json: {total_ner} hits")

    with open(outdir / "scopes.json", "w") as f:
        json.dump(all_scopes, f, ensure_ascii=False)
    print(f"       scopes.json: {total_scopes} components")

    with open(outdir / "wires.json", "w") as f:
        json.dump(all_wires, f, ensure_ascii=False)
    print(f"       wires.json: {total_wires} wires")

    with open(outdir / "ports.json", "w") as f:
        json.dump(all_ports, f, ensure_ascii=False)
    print(f"       ports.json: {total_ports} ports")

    with open(outdir / "bridge.json", "w") as f:
        json.dump(bridges, f, ensure_ascii=False, indent=1)
    print(f"       bridge.json: {len(bridges)} bridge concepts")

    # Golden subset — write full body + prepared prompt
    prompt_template = (Path(__file__).parent.parent / "data" / "wiring-prompt-template.txt").read_text()

    for i, g in enumerate(golden):
        # Find full body
        full_entry = next(e for e in entries if e["entity_id"] == g["entity_id"])
        golden_out = {
            "entity_id": g["entity_id"],
            "title": g["title"],
            "type": g["type"],
            "body": full_entry["body"],
            "classical": {
                "components": g["scope_count"],
                "wires": g["wire_count"],
                "ports": g["port_count"],
                "ner_terms": g["ner_count"],
            },
        }
        # Check if it's a bridge concept
        for b in bridges:
            if b["pm_id"] == g["entity_id"]:
                golden_out["nlab_bridge"] = b
                break

        with open(outdir / "golden" / f"golden-{i:02d}-{g['entity_id']}.json", "w") as f:
            json.dump(golden_out, f, ensure_ascii=False, indent=2)

        # Also write the prepared prompt
        prompt = prompt_template.replace("{text}", full_entry["body"])
        with open(outdir / "golden" / f"golden-{i:02d}-{g['entity_id']}.prompt.txt", "w") as f:
            f.write(prompt)

    print(f"       golden/: {len(golden)} entries with prompts")

    # Manifest
    elapsed = time.time() - t0
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "planetmath/18_Category_theory_homological_algebra",
        "metatheory_version": "3.0",
        "entries": len(entries),
        "empty_skipped": empty_count,
        "elapsed_seconds": round(elapsed, 1),
        "classical_stats": {
            "total_ner_hits": total_ner,
            "ner_per_entry": round(total_ner / len(entries), 2),
            "total_components": total_scopes,
            "components_per_entry": round(total_scopes / len(entries), 2),
            "total_wires": total_wires,
            "wires_per_entry": round(total_wires / len(entries), 2),
            "total_ports": total_ports,
            "ports_per_entry": round(total_ports / len(entries), 2),
            "total_labels": total_labels,
            "labels_per_entry": round(total_labels / len(entries), 2),
            "component_type_freq": dict(component_type_freq.most_common()),
            "wire_type_freq": dict(wire_type_freq.most_common()),
            "port_type_freq": dict(port_type_freq.most_common()),
            "label_type_freq": dict(label_type_freq.most_common()),
        },
        "bridge_stats": {
            "total_bridges": len(bridges),
            "bridge_pct": round(len(bridges) / len(entries) * 100, 1),
            "match_types": dict(Counter(b["match_type"] for b in bridges)),
        },
        "golden_count": len(golden),
        "entry_type_dist": dict(Counter(e["type"] for e in entries).most_common()),
    }

    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"       manifest.json")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY — PlanetMath Category Theory")
    print(f"{'='*60}")
    print(f"Entries:      {len(entries)} ({manifest['entry_type_dist']})")
    print(f"NER:          {total_ner} hits ({total_ner/len(entries):.1f}/entry)")
    print(f"Components:   {total_scopes} ({total_scopes/len(entries):.2f}/entry, classical)")
    print(f"  types:      {dict(component_type_freq.most_common(5))}")
    print(f"Wires:        {total_wires} ({total_wires/len(entries):.2f}/entry, classical)")
    print(f"  types:      {dict(wire_type_freq.most_common())}")
    print(f"Ports:        {total_ports} ({total_ports/len(entries):.2f}/entry, classical)")
    print(f"  types:      {dict(port_type_freq.most_common(5))}")
    print(f"Labels:       {total_labels} ({total_labels/len(entries):.2f}/entry)")
    print(f"Bridge:       {len(bridges)} PM↔nLab concepts ({len(bridges)/len(entries)*100:.0f}%)")
    print(f"Golden:       {len(golden)} entries prepared for LLM wiring extraction")
    print(f"Time:         {elapsed:.1f}s")
    print(f"Output:       {outdir}/")


if __name__ == "__main__":
    main()
