#!/usr/bin/env python3
"""Hyperreal Dictionary — nLab as a free category.

Objects are nLab pages (mathematical concepts).
Morphisms are wiki-links (relationships between concepts).
Composition is path-following through the link graph.
Enrichment is NER terms shared along each morphism.

The result: you query a term, you get its categorical position —
what it connects to, via what typed paths, carrying what shared vocabulary.
The code works. You don't chase diagrams.

Usage:
    python scripts/hyperreal.py build [--limit N]
    python scripts/hyperreal.py query "functor"
    python scripts/hyperreal.py path "group" "cohomology"
    python scripts/hyperreal.py neighborhood "adjunction" [--depth 2]
    python scripts/hyperreal.py export-futon5

Memory-safe throughout.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

NLAB_PAGES = Path(os.environ.get("NLAB_PAGES",
    str(Path(__file__).parent.parent.parent / "nlab-content" / "pages")))
NER_KERNEL = Path("data/ner-kernel/terms.tsv")
HYPERREAL_DB = Path("data/hyperreal.json")


# --- Build phase: construct the free category from nLab ---

def iter_nlab_pages(pages_dir, limit=None):
    """Yield (page_id, name, content_md) from nLab sharded directory."""
    import re
    count = 0
    for name_file in sorted(Path(pages_dir).rglob("name")):
        content_file = name_file.parent / "content.md"
        if not content_file.exists():
            continue
        page_id = name_file.parent.name
        name = name_file.read_text().strip()
        content = content_file.read_text()
        yield page_id, name, content
        count += 1
        if limit and count >= limit:
            break


def extract_wiki_links(content):
    """Extract [[wiki links]] from nLab markdown."""
    import re
    # [[target]] or [[target|display text]]
    return [m.group(1).strip() for m in
            re.finditer(r'\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]', content)]


def load_ner_singles(path):
    """Load just single-word NER terms for fast enrichment."""
    singles = set()
    skip = ("$", "(", "\"", "-")
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            term = parts[0].strip()
            if not term or any(term.startswith(p) for p in skip):
                continue
            if len(term) < 3 or " " in term:
                continue
            singles.add(term)
    return singles


def extract_page_terms(content, singles):
    """Fast NER: extract single-word terms that appear in page content."""
    words = set(content.lower().split())
    return sorted(words & singles)


def build_category(pages_dir, ner_path, limit=None):
    """Build the nLab free category.

    Returns a dict with:
      objects: {name: {id, terms, section_count, ...}}
      morphisms: {(src_name, dst_name): {type, shared_terms}}
      name_to_id: {name: page_id}
      adjacency: {name: [targets]}
    """
    print(f"Loading NER kernel...")
    singles = load_ner_singles(ner_path)
    print(f"  {len(singles)} single-word terms")

    print(f"\nBuilding category from nLab pages...")
    objects = {}       # name -> {id, terms, ...}
    name_to_id = {}    # name -> page_id
    page_terms = {}    # name -> set of terms
    raw_links = []     # (src_name, dst_name)

    n = 0
    for page_id, name, content in iter_nlab_pages(pages_dir, limit):
        name_lower = name.lower()
        terms = extract_page_terms(content, singles)

        objects[name_lower] = {
            "id": f"nlab-{page_id}",
            "name": name,
            "term_count": len(terms),
        }
        name_to_id[name_lower] = page_id
        page_terms[name_lower] = set(terms)

        # Extract outgoing wiki links
        links = extract_wiki_links(content)
        for target in links:
            target_lower = target.lower()
            if target_lower != name_lower:  # no self-loops
                raw_links.append((name_lower, target_lower))

        n += 1
        if n % 2000 == 0:
            print(f"  [{n}] pages scanned, {len(raw_links)} raw links")

    print(f"  {n} pages, {len(raw_links)} raw links")

    # Filter morphisms: only keep links where both endpoints are known pages
    known = set(objects.keys())
    morphisms = {}
    adjacency = defaultdict(list)
    reverse_adj = defaultdict(list)
    link_count = 0

    for src, dst in raw_links:
        if src in known and dst in known:
            key = (src, dst)
            if key not in morphisms:
                # Enrichment: shared NER terms between source and target
                shared = sorted(page_terms.get(src, set()) &
                                page_terms.get(dst, set()))
                morphisms[key] = {
                    "shared_terms": shared[:20],  # cap for storage
                    "shared_count": len(shared),
                }
                adjacency[src].append(dst)
                reverse_adj[dst].append(src)
                link_count += 1

    print(f"  {link_count} morphisms (both endpoints known)")

    # Compute connectivity stats
    has_outgoing = sum(1 for v in adjacency.values() if v)
    has_incoming = sum(1 for v in reverse_adj.values() if v)

    category = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "nlab",
            "pages_processed": n,
            "object_count": len(objects),
            "morphism_count": link_count,
            "objects_with_outgoing": has_outgoing,
            "objects_with_incoming": has_incoming,
            "ner_terms_used": len(singles),
        },
        "objects": objects,
        "morphisms": {f"{s}→{t}": v for (s, t), v in morphisms.items()},
        "adjacency": {k: sorted(set(v)) for k, v in adjacency.items()},
    }

    return category


# --- Query phase ---

def load_category():
    with open(HYPERREAL_DB) as f:
        return json.load(f)


def query_term(cat, term):
    """Look up a term: what is its categorical position?"""
    term_lower = term.lower()
    obj = cat["objects"].get(term_lower)
    if not obj:
        # Fuzzy search
        matches = [k for k in cat["objects"] if term_lower in k]
        if not matches:
            print(f"'{term}' not found in category.")
            return
        print(f"'{term}' not found exactly. Close matches:")
        for m in sorted(matches)[:15]:
            print(f"  {cat['objects'][m]['name']}")
        return

    print(f"\n  {obj['name']}")
    print(f"  ID: {obj['id']}")
    print(f"  NER terms: {obj['term_count']}")

    outgoing = cat["adjacency"].get(term_lower, [])
    incoming = [k.split("→")[0] for k, v in cat["morphisms"].items()
                if k.endswith(f"→{term_lower}")]

    print(f"\n  Outgoing morphisms ({len(outgoing)}):")
    # Show top by shared term count
    out_enriched = []
    for dst in outgoing:
        key = f"{term_lower}→{dst}"
        morph = cat["morphisms"].get(key, {})
        out_enriched.append((morph.get("shared_count", 0), dst, morph))
    out_enriched.sort(reverse=True)
    for sc, dst, morph in out_enriched[:15]:
        dst_name = cat["objects"].get(dst, {}).get("name", dst)
        shared = morph.get("shared_terms", [])[:5]
        print(f"    → {dst_name}  [{sc} shared: {', '.join(shared)}]")

    if len(outgoing) > 15:
        print(f"    ... and {len(outgoing) - 15} more")

    print(f"\n  Incoming morphisms ({len(incoming)}):")
    in_enriched = []
    for src in incoming:
        key = f"{src}→{term_lower}"
        morph = cat["morphisms"].get(key, {})
        in_enriched.append((morph.get("shared_count", 0), src, morph))
    in_enriched.sort(reverse=True)
    for sc, src, morph in in_enriched[:15]:
        src_name = cat["objects"].get(src, {}).get("name", src)
        shared = morph.get("shared_terms", [])[:5]
        print(f"    ← {src_name}  [{sc} shared: {', '.join(shared)}]")

    if len(incoming) > 15:
        print(f"    ... and {len(incoming) - 15} more")


def find_path(cat, source, target, max_depth=6):
    """Find shortest categorical composition from source to target.

    Returns the path as a sequence of morphisms with enrichment data.
    This IS composition in the free category.
    """
    src = source.lower()
    tgt = target.lower()

    if src not in cat["objects"]:
        print(f"Source '{source}' not in category.")
        return
    if tgt not in cat["objects"]:
        print(f"Target '{target}' not in category.")
        return

    adj = cat["adjacency"]

    # BFS for shortest path
    visited = {src}
    queue = deque([(src, [src])])

    while queue:
        node, path = queue.popleft()
        if len(path) > max_depth + 1:
            break

        for neighbor in adj.get(node, []):
            if neighbor == tgt:
                full_path = path + [tgt]
                _print_composition(cat, full_path)
                return full_path

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    print(f"No path from '{source}' to '{target}' within depth {max_depth}.")
    return None


def _print_composition(cat, path):
    """Print a categorical composition path with enrichment."""
    src_name = cat["objects"][path[0]]["name"]
    dst_name = cat["objects"][path[-1]]["name"]
    print(f"\n  {src_name}  ──{'──'.join([''] * (len(path) - 1))}──▸  {dst_name}")
    print(f"  Composition length: {len(path) - 1} morphisms\n")

    # Accumulate shared terms along the path
    all_shared = set()
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        key = f"{src}→{dst}"
        morph = cat["morphisms"].get(key, {})
        shared = morph.get("shared_terms", [])
        all_shared.update(shared)

        src_name = cat["objects"][src]["name"]
        dst_name = cat["objects"][dst]["name"]
        print(f"  {i+1}. {src_name}  →  {dst_name}")
        if shared:
            print(f"     carrying: {', '.join(shared[:8])}")

    print(f"\n  Vocabulary along path: {len(all_shared)} terms")
    if all_shared:
        # Show top terms (those appearing most frequently along the path)
        print(f"  Key terms: {', '.join(sorted(all_shared)[:20])}")


def neighborhood(cat, term, depth=2):
    """Show the categorical neighborhood of a term."""
    term_lower = term.lower()
    if term_lower not in cat["objects"]:
        print(f"'{term}' not in category.")
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

    center_name = cat["objects"][term_lower]["name"]
    print(f"\n  Neighborhood of '{center_name}' (depth {depth}):")
    print(f"  Total reachable: {len(visited) - 1} objects\n")

    for i, layer in enumerate(layers):
        names = sorted(cat["objects"].get(n, {}).get("name", n) for n in layer)
        print(f"  Distance {i+1} ({len(names)} objects):")
        for name in names[:20]:
            print(f"    {name}")
        if len(names) > 20:
            print(f"    ... and {len(names) - 20} more")
        print()


def export_futon5(cat):
    """Export the nLab category as futon5-compatible EDN."""
    objects = set(cat["objects"].keys())
    morphisms = {}
    for key in cat["morphisms"]:
        parts = key.split("→", 1)
        if len(parts) == 2:
            src, dst = parts
            morph_name = f"{src}→{dst}"
            morphisms[morph_name] = {"source": src, "target": dst}

    # Output as EDN-like structure (JSON for now, trivially convertible)
    futon5_cat = {
        "name": "nlab/hyperreal",
        "objects": sorted(objects)[:500],  # cap for readability
        "morphisms": dict(list(morphisms.items())[:2000]),
    }

    out_path = Path("data/hyperreal-futon5.json")
    with open(out_path, "w") as f:
        json.dump(futon5_cat, f, indent=2, ensure_ascii=False)
    print(f"Exported to {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  {len(futon5_cat['objects'])} objects, "
          f"{len(futon5_cat['morphisms'])} morphisms")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Hyperreal Dictionary")
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Build category from nLab")
    build_p.add_argument("--limit", type=int, default=None,
                         help="Max pages to process")
    build_p.add_argument("--pages-dir", default=str(NLAB_PAGES))
    build_p.add_argument("--ner-kernel", default=str(NER_KERNEL))

    query_p = sub.add_parser("query", help="Look up a term")
    query_p.add_argument("term")

    path_p = sub.add_parser("path", help="Find categorical path")
    path_p.add_argument("source")
    path_p.add_argument("target")
    path_p.add_argument("--max-depth", type=int, default=6)

    hood_p = sub.add_parser("neighborhood", help="Show neighborhood")
    hood_p.add_argument("term")
    hood_p.add_argument("--depth", type=int, default=2)

    export_p = sub.add_parser("export-futon5",
                              help="Export as futon5 category")

    args = parser.parse_args()

    if args.command == "build":
        cat = build_category(args.pages_dir, args.ner_kernel, args.limit)
        HYPERREAL_DB.parent.mkdir(parents=True, exist_ok=True)
        with open(HYPERREAL_DB, "w") as f:
            json.dump(cat, f, ensure_ascii=False)
        size = os.path.getsize(HYPERREAL_DB) / 1e6
        print(f"\nWritten {HYPERREAL_DB} ({size:.1f} MB)")
        print(f"  {cat['meta']['object_count']} objects, "
              f"{cat['meta']['morphism_count']} morphisms")

    elif args.command in ("query", "path", "neighborhood", "export-futon5"):
        if not HYPERREAL_DB.exists():
            print(f"Category not built yet. Run: python {sys.argv[0]} build")
            sys.exit(1)
        cat = load_category()
        if args.command == "query":
            query_term(cat, args.term)
        elif args.command == "path":
            find_path(cat, args.source, args.target, args.max_depth)
        elif args.command == "neighborhood":
            neighborhood(cat, args.term, args.depth)
        elif args.command == "export-futon5":
            export_futon5(cat)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
