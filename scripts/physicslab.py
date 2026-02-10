#!/usr/bin/env python3
"""PhysicsLab — physics.SE as a free category.

Objects are physics/math concepts (NER terms from StackExchange QA pairs).
Morphisms are co-occurrence: term A → term B if they appear in the same QA pair.
Composition is path-following through the co-occurrence graph.
Enrichment is SE tags shared along each morphism.

Dual to the hyperreal dictionary (nLab → wiki-links → free category):
    hyperreal:   nLab pages → wiki-links → categorical paths
    physicslab:  NER terms  → co-occurrence → categorical paths

Together they ground both sides of the F6 system:
    hyperreal   → grounds category theory / wiring diagrams / futon5
    physicslab  → grounds AIF+ / pattern engine / futon3

Usage:
    python scripts/physicslab.py build <pipeline-output-dir>
    python scripts/physicslab.py query "angular momentum"
    python scripts/physicslab.py path "entropy" "temperature"
    python scripts/physicslab.py neighborhood "Lagrangian" [--depth 2]
    python scripts/physicslab.py bridge "functor"  # cross-reference to hyperreal

Memory-safe: streams NER terms file, never loads full entities.json.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

PHYSICSLAB_DB = Path("data/physicslab.json")
HYPERREAL_DB = Path("data/hyperreal.json")

# Minimum co-occurrence count for a morphism to exist
MIN_COOCCURRENCE = 2

# Maximum terms per entity to consider (prevents hub explosion)
MAX_TERMS_PER_ENTITY = 100

# Generic words that leak through the NER kernel — not physics concepts.
# These create noisy hub nodes in the co-occurrence graph.
STOPWORDS = frozenset({
    "way", "even", "right", "left", "time", "point", "place", "simple",
    "mean", "number", "kind", "case", "part", "form", "term", "type",
    "set", "work", "problem", "question", "answer", "example", "order",
    "general", "certain", "given", "fact", "sense", "note", "thing",
    "result", "value", "side", "end", "hand", "line", "level", "local",
    "real", "physical", "true", "possible", "similar", "different",
    "small", "large", "long", "high", "low", "new", "first", "second",
    "total", "single", "common", "particular", "standard", "correct",
    "actually", "basically", "essentially", "indeed", "well",
})


# --- Build phase ---

def stream_ner_entities(ner_path, entities_path):
    """Stream (entity_id, terms, tags) from pipeline output.

    Loads ner-terms.json fully (small per entity) but streams entity
    metadata only for fields we need (tags).
    """
    # Build entity_id -> tags lookup from entities.json
    # We only need the tags field, not full body text
    tags_by_id = {}
    with open(entities_path) as f:
        for obj in json.load(f):
            tags_by_id[obj["entity/id"]] = obj.get("tags", [])

    with open(ner_path) as f:
        ner_data = json.load(f)

    for entry in ner_data:
        eid = entry["entity_id"]
        terms = [t["term_lower"] for t in entry["terms"]]
        tags = tags_by_id.get(eid, [])
        yield eid, terms, tags


def build_category(pipeline_dir, min_cooccurrence=MIN_COOCCURRENCE):
    """Build the physics.SE free category from pipeline output.

    Objects: NER terms that appear in at least 2 entities.
    Morphisms: (term_A, term_B) co-occurring in at least min_cooccurrence entities.
    Enrichment: SE tags shared across co-occurring entities.
    """
    pipeline_dir = Path(pipeline_dir)
    ner_path = pipeline_dir / "ner-terms.json"
    entities_path = pipeline_dir / "entities.json"

    if not ner_path.exists():
        print(f"Error: {ner_path} not found. Run pipeline with NER first.")
        sys.exit(1)
    if not entities_path.exists():
        print(f"Error: {entities_path} not found.")
        sys.exit(1)

    print("Building physicsLab category...")

    # Pass 1: collect term frequencies and co-occurrence counts
    term_freq = Counter()         # term -> entity count
    cooccur = Counter()           # (term_a, term_b) -> entity count, a < b
    term_tags = defaultdict(Counter)  # term -> {tag: count}
    edge_tags = defaultdict(Counter)  # (a, b) -> {tag: count}
    n_entities = 0

    print("  Pass 1: counting co-occurrences...")
    for eid, terms, tags in stream_ner_entities(ner_path, entities_path):
        n_entities += 1
        # Filter stopwords and cap terms
        terms = sorted(t for t in set(terms)
                       if t not in STOPWORDS)[:MAX_TERMS_PER_ENTITY]

        for t in terms:
            term_freq[t] += 1
            for tag in tags:
                term_tags[t][tag] += 1

        # Co-occurrence: all pairs (ordered to deduplicate)
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                key = (a, b)
                cooccur[key] += 1
                for tag in tags:
                    edge_tags[key][tag] += 1

        if n_entities % 10000 == 0:
            print(f"    [{n_entities}] entities, {len(cooccur)} co-occurrence pairs")

    print(f"  {n_entities} entities, {len(term_freq)} unique terms, "
          f"{len(cooccur)} raw co-occurrence pairs")

    # Filter: objects must appear in >= 2 entities
    objects = {}
    for term, freq in term_freq.items():
        if freq >= 2:
            top_tags = [t for t, _ in term_tags[term].most_common(5)]
            objects[term] = {
                "name": term,
                "frequency": freq,
                "top_tags": top_tags,
            }

    known = set(objects.keys())
    print(f"  {len(objects)} objects (terms appearing in >= 2 entities)")

    # Filter morphisms: both endpoints known, count >= threshold
    morphisms = {}
    adjacency = defaultdict(list)
    reverse_adj = defaultdict(list)
    link_count = 0

    for (a, b), count in cooccur.items():
        if count >= min_cooccurrence and a in known and b in known:
            shared_tags = [t for t, _ in edge_tags[(a, b)].most_common(10)]
            # Bidirectional: both a→b and b→a
            for src, dst in [(a, b), (b, a)]:
                key = f"{src}→{dst}"
                if key not in morphisms:
                    morphisms[key] = {
                        "cooccurrence": count,
                        "shared_tags": shared_tags,
                    }
                    adjacency[src].append(dst)
                    reverse_adj[dst].append(src)
                    link_count += 1

    print(f"  {link_count} morphisms (co-occurrence >= {min_cooccurrence})")

    category = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "physics.stackexchange",
            "type": "physicslab",
            "entities_processed": n_entities,
            "object_count": len(objects),
            "morphism_count": link_count,
            "min_cooccurrence": min_cooccurrence,
            "objects_with_outgoing": sum(1 for v in adjacency.values() if v),
        },
        "objects": objects,
        "morphisms": morphisms,
        "adjacency": {k: sorted(set(v)) for k, v in adjacency.items()},
    }

    return category


# --- Query phase ---

def load_category(path=PHYSICSLAB_DB):
    with open(path) as f:
        return json.load(f)


def query_term(cat, term):
    """Look up a term's categorical position in the physics co-occurrence graph."""
    term_lower = term.lower()
    obj = cat["objects"].get(term_lower)
    if not obj:
        matches = [k for k in cat["objects"] if term_lower in k]
        if not matches:
            print(f"'{term}' not found in physicsLab.")
            return
        print(f"'{term}' not found exactly. Close matches:")
        for m in sorted(matches)[:15]:
            print(f"  {cat['objects'][m]['name']} (freq: {cat['objects'][m]['frequency']})")
        return

    print(f"\n  {obj['name']}")
    print(f"  Frequency: {obj['frequency']} entities")
    print(f"  Top tags: {', '.join(obj['top_tags'])}")

    outgoing = cat["adjacency"].get(term_lower, [])
    print(f"\n  Connected concepts ({len(outgoing)}):")

    # Sort by co-occurrence strength
    enriched = []
    for dst in outgoing:
        key = f"{term_lower}→{dst}"
        morph = cat["morphisms"].get(key, {})
        enriched.append((morph.get("cooccurrence", 0), dst, morph))
    enriched.sort(reverse=True)

    for co, dst, morph in enriched[:20]:
        dst_name = cat["objects"].get(dst, {}).get("name", dst)
        tags = morph.get("shared_tags", [])[:3]
        print(f"    ↔ {dst_name}  [{co} co-occur, tags: {', '.join(tags)}]")

    if len(outgoing) > 20:
        print(f"    ... and {len(outgoing) - 20} more")


def find_path(cat, source, target, max_depth=6):
    """Find shortest path between physics concepts."""
    src = source.lower()
    tgt = target.lower()

    if src not in cat["objects"]:
        print(f"'{source}' not in physicsLab.")
        return
    if tgt not in cat["objects"]:
        print(f"'{target}' not in physicsLab.")
        return

    adj = cat["adjacency"]
    visited = {src}
    queue = deque([(src, [src])])

    while queue:
        node, path = queue.popleft()
        if len(path) > max_depth + 1:
            break

        for neighbor in adj.get(node, []):
            if neighbor == tgt:
                full_path = path + [tgt]
                _print_path(cat, full_path)
                return full_path

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    print(f"No path from '{source}' to '{target}' within depth {max_depth}.")
    return None


def _print_path(cat, path):
    """Print a categorical path with enrichment."""
    print(f"\n  {path[0]}  →  {path[-1]}")
    print(f"  Path length: {len(path) - 1} morphisms\n")

    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        key = f"{src}→{dst}"
        morph = cat["morphisms"].get(key, {})
        co = morph.get("cooccurrence", 0)
        tags = morph.get("shared_tags", [])[:5]
        print(f"  {i+1}. {src}  →  {dst}  [{co} co-occur]")
        if tags:
            print(f"     tags: {', '.join(tags)}")


def neighborhood(cat, term, depth=2):
    """Show the neighborhood of a physics concept."""
    term_lower = term.lower()
    if term_lower not in cat["objects"]:
        print(f"'{term}' not in physicsLab.")
        return

    adj = cat["adjacency"]
    visited = {term_lower}
    frontier = {term_lower}
    layers = []

    for d in range(depth):
        next_frontier = set()
        for node in frontier:
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        layers.append(next_frontier)
        frontier = next_frontier

    print(f"\n  Neighborhood of '{term}' (depth {depth}):")
    print(f"  Total reachable: {len(visited) - 1} concepts\n")

    for i, layer in enumerate(layers):
        names = sorted(layer)
        print(f"  Distance {i+1} ({len(names)} concepts):")
        for name in names[:20]:
            freq = cat["objects"].get(name, {}).get("frequency", 0)
            print(f"    {name} (freq: {freq})")
        if len(names) > 20:
            print(f"    ... and {len(names) - 20} more")
        print()


def bridge_to_hyperreal(cat, term):
    """Find concepts that exist in both physicsLab and hyperreal dictionary.

    This bridges the two free categories: physics.SE ↔ nLab.
    """
    term_lower = term.lower()

    if not HYPERREAL_DB.exists():
        print(f"Hyperreal dictionary not found at {HYPERREAL_DB}")
        print("Build it first: python scripts/hyperreal.py build")
        return

    hr = load_category(HYPERREAL_DB)

    # Find the term in physicsLab
    pl_neighbors = set(cat["adjacency"].get(term_lower, []))
    pl_neighbors.add(term_lower)

    # Find overlapping concepts
    hr_objects = set(hr["objects"].keys())
    bridge = pl_neighbors & hr_objects

    print(f"\n  Bridge: physicsLab ↔ hyperreal for '{term}'")
    print(f"  PhysicsLab neighbors: {len(pl_neighbors)}")
    print(f"  Hyperreal objects: {len(hr_objects)}")
    print(f"  Bridging concepts: {len(bridge)}\n")

    for concept in sorted(bridge):
        pl_obj = cat["objects"].get(concept, {})
        hr_obj = hr["objects"].get(concept, {})
        pl_freq = pl_obj.get("frequency", 0)
        hr_out = len(hr["adjacency"].get(concept, []))
        print(f"    {concept}")
        print(f"      physicsLab: freq {pl_freq}, "
              f"tags: {', '.join(pl_obj.get('top_tags', [])[:3])}")
        print(f"      hyperreal:  {hr_obj.get('name', '?')}, "
              f"{hr_out} outgoing morphisms")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="PhysicsLab Dictionary")
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Build category from pipeline output")
    build_p.add_argument("pipeline_dir",
                         help="Pipeline output directory (with ner-terms.json)")
    build_p.add_argument("--min-cooccurrence", type=int,
                         default=MIN_COOCCURRENCE)

    query_p = sub.add_parser("query", help="Look up a term")
    query_p.add_argument("term")

    path_p = sub.add_parser("path", help="Find path between concepts")
    path_p.add_argument("source")
    path_p.add_argument("target")
    path_p.add_argument("--max-depth", type=int, default=6)

    hood_p = sub.add_parser("neighborhood", help="Show neighborhood")
    hood_p.add_argument("term")
    hood_p.add_argument("--depth", type=int, default=2)

    bridge_p = sub.add_parser("bridge",
                              help="Bridge to hyperreal dictionary")
    bridge_p.add_argument("term")

    args = parser.parse_args()

    if args.command == "build":
        cat = build_category(args.pipeline_dir, args.min_cooccurrence)
        PHYSICSLAB_DB.parent.mkdir(parents=True, exist_ok=True)
        with open(PHYSICSLAB_DB, "w") as f:
            json.dump(cat, f, ensure_ascii=False)
        size = os.path.getsize(PHYSICSLAB_DB) / 1e6
        print(f"\nWritten {PHYSICSLAB_DB} ({size:.1f} MB)")
        print(f"  {cat['meta']['object_count']} objects, "
              f"{cat['meta']['morphism_count']} morphisms")

    elif args.command in ("query", "path", "neighborhood", "bridge"):
        if not PHYSICSLAB_DB.exists():
            print(f"PhysicsLab not built yet. Run: "
                  f"python {sys.argv[0]} build <pipeline-dir>")
            sys.exit(1)
        cat = load_category()
        if args.command == "query":
            query_term(cat, args.term)
        elif args.command == "path":
            find_path(cat, args.source, args.target, args.max_depth)
        elif args.command == "neighborhood":
            neighborhood(cat, args.term, args.depth)
        elif args.command == "bridge":
            bridge_to_hyperreal(cat, args.term)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
