#!/usr/bin/env python3
"""Pre-validation run: exercise the F6 pipeline on nLab content.

Parses nLab wiki pages into the same entity/relation format as superpod-job.py,
then runs NER term spotting + scope detection (Stage 5). Produces a small,
inspectable output that ships with the superpod job as worked examples.

The nLab corpus is ~20K pages of pure mathematics — ideal for pre-validation
because it's definition-dense, scope-heavy, and exercises the NER kernel hard.

Usage:
    python scripts/nlab-prevalidate.py /path/to/nlab-content/pages \
        --output-dir data/nlab-preview \
        --limit 500

Memory-safe: streams pages from disk, writes results incrementally.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter

# Reuse Stage 5 functions from superpod-job
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module


# --- nLab page parser ---

def iter_nlab_pages(pages_dir, limit=None):
    """Iterate nLab pages from the sharded directory structure.

    Yields (page_id, name, content_md) tuples.
    """
    count = 0
    pages_path = Path(pages_dir)

    for name_file in sorted(pages_path.rglob("name")):
        page_dir = name_file.parent
        content_file = page_dir / "content.md"
        if not content_file.exists():
            continue

        page_id = page_dir.name  # numeric ID from directory name
        name = name_file.read_text().strip()
        content = content_file.read_text()

        yield page_id, name, content
        count += 1

        if limit and count >= limit:
            break


def parse_nlab_page(page_id, name, content):
    """Parse an nLab page into an entity dict + extracted features.

    Returns (entity_dict, plain_text) where plain_text has markup stripped
    for NER/scope processing.
    """
    # Extract wiki links → relations
    wiki_links = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]', content)

    # Extract LaTeX fragments
    display_latex = re.findall(r'\\\[(.*?)\\\]', content, re.DOTALL)
    inline_latex = re.findall(r'\$([^$]+?)\$', content)
    all_latex = display_latex + inline_latex

    # Extract sections
    sections = re.findall(r'^##+ (.+)$', content, re.MULTILINE)

    # Strip markup for plain text (NER + scope processing)
    plain = content
    # Remove nLab navigation boilerplate
    plain = re.sub(r'\+-- \{:.*?\}.*?=--', '', plain, flags=re.DOTALL)
    plain = re.sub(r'\[\[!include.*?\]\]', '', plain)
    # Remove wiki link markup but keep text
    plain = re.sub(r'\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]',
                   lambda m: m.group(2) or m.group(1), plain)
    # Remove Markdown formatting
    plain = re.sub(r'^#+\s+', '', plain, flags=re.MULTILINE)
    plain = re.sub(r'\{:.*?\}', '', plain)
    plain = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', plain)

    entity = {
        "entity/id": f"nlab-{page_id}",
        "entity/type": "NLabPage",
        "entity/source": "nlab",
        "entity/name": name,
        "title": name,
        "body": plain[:500],  # truncate for entity record
        "sections": sections,
        "latex_count": len(all_latex),
        "wiki_link_count": len(wiki_links),
    }

    relations = [{"relation/type": "wiki-link",
                  "relation/src": f"nlab-{page_id}",
                  "relation/dst": f"nlab:{link}"}
                 for link in set(wiki_links)]

    return entity, relations, plain, all_latex


# --- NER + scope from superpod-job (self-contained copies for safety) ---

SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)"),
    ("define", r"\bDefine\s+\$([^$]+)\$\s*(:=|=|\\equiv)\s*([^.,$]+)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$([^$]+)\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?([^$.]{1,60})"),
    ("for-any", r"\b(?:for\s+)?(any|every|each|all)\s+\$([^$]+)\$"),
    ("where-binding", r"\bwhere\s+\$([^$]+)\$\s+(is|denotes|represents)\s+([^.,$]+)"),
    ("set-notation", r"\$([^$]*\\in\s+[^$]+)\$"),
]


def load_ner_kernel(path):
    singles = {}
    multi_index = {}
    multi_count = 0
    skip_prefixes = ("$", "(", "\"", "-")
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            term_lower = parts[0].strip()
            term_orig = parts[1].strip()
            canon = parts[3].strip()
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
    scopes = []
    scope_idx = 0
    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, text):
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1
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
                "hx/type": f"scope/{stype}",
                "hx/ends": ends,
                "hx/content": {"match": m.group()[:120], "position": m.start()},
                "hx/labels": ["scope", stype],
            })
    return scopes


def main():
    parser = argparse.ArgumentParser(
        description="Pre-validate F6 pipeline on nLab content")
    parser.add_argument("pages_dir", help="Path to nlab-content/pages/")
    parser.add_argument("--output-dir", "-o", default="data/nlab-preview")
    parser.add_argument("--ner-kernel", default="data/ner-kernel/terms.tsv")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max pages to process (default 500)")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load NER kernel
    print(f"Loading NER kernel from {args.ner_kernel}...")
    singles, multi_index, multi_count = load_ner_kernel(args.ner_kernel)
    print(f"  {len(singles)} single + {multi_count} multi-word terms")

    # Stream through pages, write results incrementally
    print(f"\nProcessing nLab pages (limit={args.limit})...")

    total_ner = 0
    total_scopes = 0
    pages_with_ner = 0
    pages_with_scopes = 0
    total_latex = 0
    total_links = 0
    stype_freq = Counter()
    n = 0

    ent_path = outdir / "entities.json"
    rel_path = outdir / "relations.json"
    ner_path = outdir / "ner-terms.json"
    scope_path = outdir / "scopes.json"

    with open(ent_path, "w") as ent_f, \
         open(rel_path, "w") as rel_f, \
         open(ner_path, "w") as ner_f, \
         open(scope_path, "w") as scope_f:

        ent_f.write("[\n")
        rel_f.write("[\n")
        ner_f.write("[\n")
        scope_f.write("[\n")

        first_rel = True

        for page_id, name, content in iter_nlab_pages(args.pages_dir, args.limit):
            entity, relations, plain_text, latex_frags = parse_nlab_page(
                page_id, name, content)

            # NER
            terms = spot_terms(plain_text, singles, multi_index)
            if terms:
                pages_with_ner += 1
                total_ner += len(terms)

            # Scopes
            eid = entity["entity/id"]
            scopes = detect_scopes(eid, plain_text)
            if scopes:
                pages_with_scopes += 1
                total_scopes += len(scopes)
                for s in scopes:
                    stype_freq[s["hx/type"]] += 1

            total_latex += len(latex_frags)
            total_links += len(relations)

            # Write incrementally
            sep = ",\n" if n > 0 else ""
            ent_f.write(sep + json.dumps(entity, ensure_ascii=False))
            ner_f.write(sep + json.dumps(
                {"entity_id": eid, "terms": terms, "count": len(terms)},
                ensure_ascii=False))
            scope_f.write(sep + json.dumps(
                {"entity_id": eid, "scopes": scopes, "count": len(scopes)},
                ensure_ascii=False))

            for rel in relations:
                rsep = ",\n" if not first_rel else ""
                rel_f.write(rsep + json.dumps(rel, ensure_ascii=False))
                first_rel = False

            n += 1
            if n % 100 == 0:
                print(f"  [{n}] NER: {total_ner} hits, scopes: {total_scopes}")

        ent_f.write("\n]")
        rel_f.write("\n]")
        ner_f.write("\n]")
        scope_f.write("\n]")

    elapsed = time.time() - t0

    # Write manifest
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "nlab",
        "pages_dir": args.pages_dir,
        "pages_processed": n,
        "limit": args.limit,
        "elapsed_seconds": round(elapsed),
        "stages_completed": ["parse", "ner_scopes"],
        "stage5_stats": {
            "ner_kernel_terms": len(singles) + multi_count,
            "entities_processed": n,
            "total_ner_hits": total_ner,
            "entities_with_ner": pages_with_ner,
            "ner_coverage": pages_with_ner / n if n else 0,
            "total_scopes": total_scopes,
            "entities_with_scopes": pages_with_scopes,
            "scope_coverage": pages_with_scopes / n if n else 0,
            "scope_type_freq": dict(stype_freq.most_common()),
        },
        "corpus_stats": {
            "total_latex_fragments": total_latex,
            "total_wiki_links": total_links,
        },
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"nLab pre-validation complete in {elapsed:.0f}s")
    print(f"  Pages: {n}")
    print(f"  NER:   {total_ner} hits, {pages_with_ner}/{n} pages ({pages_with_ner/n:.0%})")
    print(f"  Scope: {total_scopes} records, {pages_with_scopes}/{n} pages ({pages_with_scopes/n:.0%})")
    print(f"  LaTeX: {total_latex} fragments")
    print(f"  Links: {total_links} wiki-links")
    if stype_freq:
        print(f"\n  Scope types:")
        for stype, count in stype_freq.most_common():
            print(f"    {stype}: {count}")

    print(f"\nOutput: {outdir}/")
    for f in sorted(outdir.iterdir()):
        if f.is_file():
            print(f"  {f.name:30s} {os.path.getsize(f)/1e6:8.3f} MB")


if __name__ == "__main__":
    main()
