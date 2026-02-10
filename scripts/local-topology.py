#!/usr/bin/env python3
r"""Local topology analysis of wiring diagrams within individual definitions.

For each PlanetMath CT entry, computes the local wiring DAG and checks
structural properties:

1. CAR extraction: what is being defined, and what are its parameters (arity)?
2. Sink property: does the defined concept appear as the conclude node?
3. Reachability: do all bind/assume components flow toward the conclude?
4. Orphan detection: are there disconnected components?
5. Domain object centrality: which NER terms are most referenced locally?
6. Local-global coherence: do locally central terms appear in \pmrelated{}?

The key insight: a definition is a lambda -- the CAR (defined term) has an
arity (number of top-level bindings), and the wiring diagram is the body.

Usage:
    python scripts/local-topology.py [--entry NAME] [--all] [--golden]

Outputs:
    data/ct-validation/topology.json        -- per-entry topology metrics
    data/ct-validation/topology-report.txt  -- human-readable summary
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

# --- Regexes (shared with validate-ct.py) ---

SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)"),
    ("define", r"\bDefine\s+\$([^$]+)\$\s*(:=|=|\\equiv)\s*([^.,$]+)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$([^$]+)\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?([^$.]{1,60})"),
    ("for-any", r"\b(?:for\s+)?(any|every|each|all)\s+\$([^$]+)\$"),
    ("where-binding", r"\bwhere\s+\$([^$]+)\$\s+(is|denotes|represents)\s+([^.,$]+)"),
    ("set-notation", r"\$([^$]*\\in\s+[^$]+)\$"),
]

WIRE_REGEXES = [
    ("wire/adversative", r"\b(?:but|however|on the other hand|nevertheless|yet)\b", re.IGNORECASE),
    ("wire/causal", r"\b(?:because|since|the reason is|given that)\b", re.IGNORECASE),
    ("wire/consequential", r"\b(?:therefore|thus|hence|it follows|so that|note that|in fact)\b", re.IGNORECASE),
    ("wire/clarifying", r"\b(?:that is|in other words|namely|more precisely|i\.e\.)\b", re.IGNORECASE),
    ("wire/intuitive", r"\b(?:intuitively|roughly speaking|heuristically)\b", re.IGNORECASE),
]

PORT_REGEXES = [
    ("port/that-noun", r"\bthat\s+(?:root|function|map|functor|morphism|object|category|space|set|group)\b", re.IGNORECASE),
    ("port/this-noun", r"\bthis\s+(?:equation|operator|means|functor|morphism|diagram|category|map)\b", re.IGNORECASE),
    ("port/the-above", r"\b(?:the above|the preceding|the previous)\s+\w+", re.IGNORECASE),
    ("port/the-same", r"\bthe same\s+\w+", re.IGNORECASE),
    ("port/such", r"\bsuch (?:a|an)\s+\w+", re.IGNORECASE),
    ("port/similarly", r"\b(?:similarly|analogously)\b", re.IGNORECASE),
    ("port/likewise", r"\b(?:likewise|correspondingly)\b", re.IGNORECASE),
]

# --- Conclude patterns (the thing being defined/proved) ---

CONCLUDE_REGEXES = [
    ("conclude/call", r"(?:is|are)\s+called\s+(?:a\s+|an\s+|the\s+)?\\emph\{([^}]+)\}", re.IGNORECASE),
    ("conclude/call-plain", r"(?:is|are)\s+called\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z][a-zA-Z\s]{2,30}?)(?:\.|,|$)", re.IGNORECASE),
    ("conclude/denote", r"(?:is\s+)?denoted\s+(?:by\s+)?\$([^$]+)\$", re.IGNORECASE),
    ("conclude/define-as", r"(?:we\s+)?define\s+(?:a\s+|an\s+|the\s+)?\\emph\{([^}]+)\}", re.IGNORECASE),
    ("conclude/said-to-be", r"is said to be\s+(?:a\s+|an\s+)?\\emph\{([^}]+)\}", re.IGNORECASE),
    ("conclude/qed", r"\\qed|\\end\{proof\}", 0),
]

# --- CAR extraction: what does this entry define? ---

def extract_car(body, meta):
    r"""Extract the CAR (defined concept) and its arity.

    The CAR is extracted from:
    1. \pmdefines{} metadata (most reliable)
    2. \pmtitle{} metadata
    3. The first \emph{} in a "called" or "define" context
    4. The canonical name

    Arity = number of top-level let-bindings (the parameters).
    """
    car = {}

    # Primary: pmdefines
    defines = meta.get("defines", [])
    if defines:
        car["term"] = defines[0]
        car["source"] = "pmdefines"
        car["all_defines"] = defines
    # Fallback: title
    elif "title" in meta:
        car["term"] = meta["title"]
        car["source"] = "pmtitle"
    # Fallback: canonical name
    elif "canonicalname" in meta:
        car["term"] = meta["canonicalname"]
        car["source"] = "pmcanonicalname"

    # Find where the CAR appears in the body (conclude pattern)
    car["conclude_positions"] = []
    for ctype, pattern, flags in CONCLUDE_REGEXES:
        for m in re.finditer(pattern, body, flags):
            car["conclude_positions"].append({
                "type": ctype,
                "match": m.group()[:100],
                "position": m.start(),
            })

    # Arity = number of top-level let-bindings
    let_bindings = []
    for m in re.finditer(r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)", body):
        let_bindings.append({
            "symbols": m.group(1).strip(),
            "type": m.group(3).strip()[:80],
            "position": m.start(),
        })
    car["arity"] = len(let_bindings)
    car["parameters"] = let_bindings

    return car


def extract_body(tex_path):
    """Extract body text and metadata from a PlanetMath .tex file."""
    with open(tex_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    meta = {}
    for tag in ["pmcanonicalname", "pmtitle", "pmtype", "pmauthor"]:
        m = re.search(rf"\\{tag}\{{([^}}]*)\}}", content)
        if m:
            meta[tag.replace("pm", "")] = m.group(1)

    defines = re.findall(r"\\pmdefines\{([^}]*)\}", content)
    if defines:
        meta["defines"] = defines

    # Extract \pmrelated
    related = re.findall(r"\\pmrelated\{([^}]*)\}", content)
    if related:
        meta["related"] = related

    # Extract body
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", content, re.DOTALL)
    body = m.group(1).strip() if m else ""

    return body, meta


def detect_elements(entity_id, body):
    """Detect all wiring diagram elements with positions."""
    components = []
    comp_idx = 0

    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, body):
            components.append({
                "id": f"c{comp_idx}",
                "type": stype,
                "match": m.group()[:120],
                "position": m.start(),
                "end_position": m.end(),
            })
            comp_idx += 1

    wires = []
    wire_idx = 0
    for wtype, pattern, flags in WIRE_REGEXES:
        for m in re.finditer(pattern, body, flags):
            wires.append({
                "id": f"w{wire_idx}",
                "type": wtype,
                "match": m.group()[:60],
                "position": m.start(),
            })
            wire_idx += 1

    ports = []
    port_idx = 0
    for ptype, pattern, flags in PORT_REGEXES:
        for m in re.finditer(pattern, body, flags):
            ports.append({
                "id": f"p{port_idx}",
                "type": ptype,
                "match": m.group()[:60],
                "position": m.start(),
            })
            port_idx += 1

    # Sort all by position
    components.sort(key=lambda x: x["position"])
    wires.sort(key=lambda x: x["position"])
    ports.sort(key=lambda x: x["position"])

    return components, wires, ports


def build_flow_graph(components, wires):
    """Build the local flow DAG from components and wires.

    Wires connect the nearest preceding component to the nearest
    following component (by text position). This is the classical
    approximation — LLM extraction would give us explicit endpoints.
    """
    if not components:
        return {"edges": [], "adj": {}}

    edges = []
    adj = defaultdict(list)

    for w in wires:
        # Find the component just before this wire
        source = None
        for c in reversed(components):
            if c["position"] < w["position"]:
                source = c["id"]
                break

        # Find the component just after this wire
        target = None
        for c in components:
            if c["position"] > w["position"]:
                target = c["id"]
                break

        if source and target and source != target:
            edges.append({
                "source": source,
                "target": target,
                "wire_type": w["type"],
                "wire_match": w["match"],
            })
            adj[source].append(target)

    return {"edges": edges, "adj": dict(adj)}


def compute_topology(car, components, wires, ports, flow_graph):
    """Compute local topology metrics."""
    if not components:
        return {
            "well_formed": False,
            "problems": ["no components detected"],
            "n_components": 0,
            "n_wires": len(wires),
            "n_ports": len(ports),
            "n_edges": 0,
            "component_sig": {},
            "wire_sig": Counter(w["type"] for w in wires),
            "ratio_comp_wire": 0,
            "roots": [],
            "sinks": [],
            "orphans": [],
            "orphan_rate": 0,
            "last_is_sink": False,
            "root_reaches_sink": {},
        }

    n_comp = len(components)
    n_wires = len(wires)
    n_ports = len(ports)
    n_edges = len(flow_graph["edges"])
    adj = flow_graph["adj"]

    # --- Component type signature ---
    comp_sig = Counter(c["type"] for c in components)
    wire_sig = Counter(w["type"] for w in wires)

    # --- Identify structural roles ---
    # First components (by position) are usually premises/parameters
    # Last components are usually conclusions
    first_comp = components[0]
    last_comp = components[-1]

    # Roots: components that are never targets of any edge
    all_targets = set()
    for e in flow_graph["edges"]:
        all_targets.add(e["target"])
    roots = [c["id"] for c in components if c["id"] not in all_targets]

    # Sinks: components that are never sources of any edge
    all_sources = set()
    for e in flow_graph["edges"]:
        all_sources.add(e["source"])
    sinks = [c["id"] for c in components if c["id"] not in all_sources]

    # --- Reachability from roots to sinks ---
    def reachable_from(start):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, []):
                stack.append(neighbor)
        return visited

    # Check: can every root reach at least one sink?
    root_reaches_sink = {}
    sink_ids = set(sinks)
    for r in roots:
        reached = reachable_from(r)
        root_reaches_sink[r] = bool(reached & sink_ids)

    # --- Orphan detection ---
    # Components not connected to any edge (neither source nor target)
    connected = all_sources | all_targets
    orphans = [c["id"] for c in components if c["id"] not in connected]

    # --- Sink = conclude? ---
    # Check if the last component (positionally) is a sink
    last_is_sink = last_comp["id"] in sinks

    # --- Component:Wire ratio (local) ---
    ratio = n_comp / max(n_wires, 1)

    # --- Well-formedness ---
    problems = []
    if not roots:
        problems.append("no root components (everything is a target)")
    if not sinks:
        problems.append("no sink components (everything is a source)")
    if orphans and len(orphans) > n_comp // 2:
        problems.append(f"{len(orphans)}/{n_comp} components are orphans")
    unreachable = [r for r, ok in root_reaches_sink.items() if not ok]
    if unreachable and len(unreachable) > len(roots) // 2:
        problems.append(f"{len(unreachable)}/{len(roots)} roots cannot reach any sink")

    return {
        "well_formed": len(problems) == 0,
        "problems": problems,
        "n_components": n_comp,
        "n_wires": n_wires,
        "n_ports": n_ports,
        "n_edges": n_edges,
        "component_sig": dict(comp_sig),
        "wire_sig": dict(wire_sig),
        "ratio_comp_wire": round(ratio, 2),
        "roots": roots,
        "sinks": sinks,
        "orphans": orphans,
        "orphan_rate": round(len(orphans) / n_comp, 2) if n_comp else 0,
        "last_is_sink": last_is_sink,
        "root_reaches_sink": root_reaches_sink,
    }


def analyze_entry(tex_path):
    """Full local topology analysis of a single entry."""
    body, meta = extract_body(tex_path)
    if not body:
        return None

    entity_id = meta.get("canonicalname", Path(tex_path).stem)
    entry_type = meta.get("type", "Unknown")

    # Extract CAR (what is being defined?)
    car = extract_car(body, meta)

    # Detect wiring diagram elements
    components, wires, ports = detect_elements(entity_id, body)

    # Build flow graph
    flow_graph = build_flow_graph(components, wires)

    # Compute topology
    topology = compute_topology(car, components, wires, ports, flow_graph)

    # Domain object frequency (which $math$ terms appear most?)
    math_spans = re.findall(r"\$([^$]+)\$", body)
    # Extract meaningful domain objects from math spans
    domain_terms = Counter()
    # Noise words that appear in math mode but aren't domain objects
    noise = {"is", "to", "in", "a", "an", "the", "and", "or", "of", "for",
             "with", "on", "at", "by", ",", ".", "is a", "-small, so is",
             "not", "also", "are", "be"}
    for span in math_spans:
        # Clean up LaTeX commands, keep meaningful content
        clean = re.sub(r"\\(?:mathcal|mathrm|mathbb|operatorname)\{([^}]+)\}", r"\1", span)
        clean = re.sub(r"\\(?:to|rightarrow|Rightarrow|mapsto|colon)", " -> ", clean)
        clean = re.sub(r"\\(?:in|subset|subseteq)", " in ", clean)
        clean = re.sub(r"\\(?:circ|bullet|times|otimes|oplus)", " . ", clean)
        clean = re.sub(r"\\[a-zA-Z]+", "", clean)
        clean = re.sub(r"[{}\\]", "", clean)
        clean = clean.strip()
        if clean and len(clean) <= 40 and clean.lower() not in noise and len(clean) >= 2:
            # Skip pure punctuation/operators
            if re.search(r"[a-zA-Z]", clean):
                domain_terms[clean] += 1

    # Top domain objects
    top_domain = domain_terms.most_common(10)

    # Local-global coherence: do locally central terms appear in pmrelated?
    related = set(r.lower() for r in meta.get("related", []))
    coherence = {}
    if related and top_domain:
        for term, count in top_domain[:5]:
            term_lower = term.lower().strip()
            # Check if any related entry name contains this term
            matches = [r for r in related if term_lower in r.lower()]
            coherence[term] = {
                "local_count": count,
                "in_related": len(matches) > 0,
                "related_matches": matches[:3],
            }

    return {
        "entity_id": entity_id,
        "title": meta.get("title", ""),
        "type": entry_type,
        "body_length": len(body),
        "car": car,
        "topology": topology,
        "top_domain_objects": [{"term": t, "count": c} for t, c in top_domain],
        "local_global_coherence": coherence,
        "flow_edges": flow_graph["edges"][:20],  # cap for readability
    }


def format_report(results):
    """Generate human-readable topology report."""
    lines = []
    lines.append("=" * 72)
    lines.append("LOCAL TOPOLOGY REPORT — PlanetMath CT Wiring Diagrams")
    lines.append("=" * 72)
    lines.append("")

    # Summary statistics
    total = len(results)
    well_formed = sum(1 for r in results if r["topology"]["well_formed"])
    has_car = sum(1 for r in results if r["car"].get("term"))
    has_conclude = sum(1 for r in results if r["car"].get("conclude_positions"))

    avg_arity = sum(r["car"]["arity"] for r in results) / max(total, 1)
    avg_ratio = sum(r["topology"]["ratio_comp_wire"] for r in results) / max(total, 1)
    avg_orphan = sum(r["topology"]["orphan_rate"] for r in results) / max(total, 1)

    lines.append(f"Entries analyzed:     {total}")
    lines.append(f"Well-formed DAGs:     {well_formed}/{total} ({100*well_formed/max(total,1):.0f}%)")
    lines.append(f"CAR identified:       {has_car}/{total}")
    lines.append(f"Conclude detected:    {has_conclude}/{total}")
    lines.append(f"Average arity:        {avg_arity:.1f} parameters")
    lines.append(f"Average comp:wire:    {avg_ratio:.2f}")
    lines.append(f"Average orphan rate:  {avg_orphan:.1%}")
    lines.append("")

    # Arity distribution
    arity_dist = Counter(r["car"]["arity"] for r in results)
    lines.append("--- Arity Distribution (# of let-bindings = parameters) ---")
    for arity in sorted(arity_dist.keys()):
        bar = "#" * min(arity_dist[arity], 50)
        lines.append(f"  arity {arity:2d}: {arity_dist[arity]:4d}  {bar}")
    lines.append("")

    # Type breakdown
    type_dist = Counter(r["type"] for r in results)
    lines.append("--- Entry Type Breakdown ---")
    for t, c in type_dist.most_common():
        lines.append(f"  {t:30s}  {c}")
    lines.append("")

    # Ratio distribution
    lines.append("--- Component:Wire Ratio Distribution ---")
    ratio_buckets = Counter()
    for r in results:
        ratio = r["topology"]["ratio_comp_wire"]
        if ratio == 0:
            bucket = "0 (no wires)"
        elif ratio < 0.5:
            bucket = "<0.5 (wire-dense)"
        elif ratio < 1.0:
            bucket = "0.5-1.0"
        elif ratio < 2.0:
            bucket = "1.0-2.0"
        elif ratio < 5.0:
            bucket = "2.0-5.0 (comp-dense)"
        else:
            bucket = "5.0+ (very comp-dense)"
        ratio_buckets[bucket] += 1
    for bucket in ["0 (no wires)", "<0.5 (wire-dense)", "0.5-1.0",
                    "1.0-2.0", "2.0-5.0 (comp-dense)", "5.0+ (very comp-dense)"]:
        if bucket in ratio_buckets:
            lines.append(f"  {bucket:30s}  {ratio_buckets[bucket]}")
    lines.append("")

    # Well-formedness problems
    problem_counts = Counter()
    for r in results:
        for p in r["topology"]["problems"]:
            # Generalize the problem string
            p_general = re.sub(r"\d+", "N", p)
            problem_counts[p_general] += 1
    if problem_counts:
        lines.append("--- Common Structural Problems ---")
        for p, c in problem_counts.most_common(10):
            lines.append(f"  {c:4d}  {p}")
        lines.append("")

    # Showcase: best and worst entries
    scored = [(r["topology"]["n_edges"] - len(r["topology"]["orphans"]) * 2, r)
              for r in results if r["topology"]["n_components"] >= 3]
    scored.sort(key=lambda x: -x[0])

    if scored:
        lines.append("--- Best-Wired Entries (most connected flow) ---")
        for score, r in scored[:10]:
            car_term = r["car"].get("term", "?")[:30]
            lines.append(f"  {r['entity_id']:45s}  "
                         f"comp={r['topology']['n_components']:2d}  "
                         f"edges={r['topology']['n_edges']:2d}  "
                         f"orphans={len(r['topology']['orphans'])}  "
                         f"arity={r['car']['arity']}  "
                         f"CAR={car_term}")
        lines.append("")

        lines.append("--- Worst-Wired Entries (most orphans / disconnected) ---")
        scored.sort(key=lambda x: x[0])
        for score, r in scored[:10]:
            car_term = r["car"].get("term", "?")[:30]
            problems = "; ".join(r["topology"]["problems"][:2]) or "low connectivity"
            lines.append(f"  {r['entity_id']:45s}  "
                         f"comp={r['topology']['n_components']:2d}  "
                         f"edges={r['topology']['n_edges']:2d}  "
                         f"orphans={len(r['topology']['orphans'])}  "
                         f"{problems}")
        lines.append("")

    # Showcase: entries with interesting CARs
    lines.append("--- CAR Examples (defined term as lambda) ---")
    interesting = [r for r in results
                   if r["car"]["arity"] >= 2
                   and r["car"].get("conclude_positions")]
    interesting.sort(key=lambda x: -x["car"]["arity"])
    for r in interesting[:10]:
        car = r["car"]
        params = [f'{p["symbols"]} : {p["type"][:30]}' for p in car["parameters"]]
        conclude = car["conclude_positions"][0]["match"][:60] if car["conclude_positions"] else "?"
        lines.append(f"  ({car.get('term', '?')[:25]:25s} {' '.join(params[:4])})")
        lines.append(f"    → {conclude}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Local topology analysis")
    parser.add_argument("--entry", type=str, help="Analyze single entry by canonical name")
    parser.add_argument("--all", action="store_true", help="Analyze all 313 CT entries")
    parser.add_argument("--golden", action="store_true", help="Analyze golden-20 entries")
    args = parser.parse_args()

    ct_dir = Path(os.path.expanduser(
        "~/code/planetmath/18_Category_theory_homological_algebra"
    ))
    out_dir = Path(os.path.expanduser("~/code/futon6/data/ct-validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.entry:
        # Find the specific entry
        found = None
        for f in ct_dir.glob("*.tex"):
            with open(f, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            if f"\\pmcanonicalname{{{args.entry}}}" in content:
                found = f
                break
        if not found:
            print(f"Entry '{args.entry}' not found in {ct_dir}")
            sys.exit(1)
        result = analyze_entry(str(found))
        if result:
            print(json.dumps(result, indent=2))
        return

    # Collect .tex files
    if args.golden:
        # Use golden entries
        golden_dir = out_dir / "golden"
        tex_files = []
        for gf in sorted(golden_dir.glob("*.json")):
            with open(gf) as f:
                data = json.load(f)
            canon = data.get("entity_id", "").replace("pm-ct-", "")
            for tf in ct_dir.glob("*.tex"):
                with open(tf, encoding="utf-8", errors="replace") as fh:
                    if f"\\pmcanonicalname{{{canon}}}" in fh.read():
                        tex_files.append(tf)
                        break
    else:
        tex_files = sorted(ct_dir.glob("*.tex"))

    print(f"Analyzing {len(tex_files)} entries...")

    results = []
    for i, tf in enumerate(tex_files):
        result = analyze_entry(str(tf))
        if result:
            results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tex_files)}...")

    print(f"Analyzed {len(results)} entries successfully.")

    # Generate report
    report = format_report(results)
    print()
    print(report)

    # Save outputs
    report_path = out_dir / "topology-report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save JSON (slim version without flow edges)
    slim = []
    for r in results:
        slim.append({
            "entity_id": r["entity_id"],
            "title": r["title"],
            "type": r["type"],
            "car": {
                "term": r["car"].get("term"),
                "source": r["car"].get("source"),
                "arity": r["car"]["arity"],
                "parameters": r["car"]["parameters"],
                "n_conclude": len(r["car"].get("conclude_positions", [])),
            },
            "topology": r["topology"],
            "top_domain_objects": r["top_domain_objects"][:5],
        })

    topo_path = out_dir / "topology.json"
    with open(topo_path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"Topology data saved to {topo_path}")


if __name__ == "__main__":
    main()
