"""PlanetMath loader.

Reads PlanetMath EDN exports and .tex source files, produces normalized
entry dicts suitable for graph construction.
"""

import re
import edn_format
from pathlib import Path


def load_edn(path: str) -> list[dict]:
    """Load a PlanetMath EDN file and return a list of entry dicts."""
    raw = Path(path).read_text()
    parsed = edn_format.loads(raw)
    return [_normalize_entry(e) for e in parsed]


def _normalize_entry(edn_map) -> dict:
    """Convert EDN map with namespaced keys to a plain dict."""
    result = {}
    for k, v in edn_map.items():
        # edn_format returns Keyword objects with namespaced names like "entry/id"
        name = k.name if hasattr(k, "name") else str(k)
        # Strip namespace prefix (e.g., "entry/id" -> "id")
        if "/" in name:
            name = name.split("/", 1)[1]
        result[name] = _convert_value(v)
    return result


def _convert_value(v):
    """Recursively convert EDN values to Python natives."""
    if isinstance(v, edn_format.ImmutableDict):
        return {
            _strip_ns(k.name if hasattr(k, "name") else str(k)): _convert_value(val)
            for k, val in v.items()
        }
    if isinstance(v, (edn_format.ImmutableList, list, tuple)):
        return [_convert_value(item) for item in v]
    if isinstance(v, edn_format.Keyword):
        return v.name
    return v


def entries_to_entities(entries: list[dict]) -> list[dict]:
    """Convert PlanetMath entries to futon1a entity shape.

    Each entry becomes an entity with:
    - entity/id: the PlanetMath ID
    - entity/type: the entry type (Definition, Theorem, etc.)
    - Plus all original fields as metadata.

    Deduplicates by entity/id (last entry wins if source has duplicates).
    """
    seen = {}
    for e in entries:
        entity = {
            "entity/id": e["id"],
            "entity/type": e.get("type", "Entry"),
            "entity/source": e.get("source", "planetmath"),
            "title": e.get("title", ""),
            "author": e.get("author", ""),
            "body": e.get("body", ""),
            "keywords": e.get("keywords", []),
            "defines": e.get("defines", []),
            "msc-codes": e.get("msc-codes", []),
            "created": e.get("created", ""),
            "modified": e.get("modified", ""),
            "source-file": e.get("source-file", ""),
        }
        seen[entity["entity/id"]] = entity
    return list(seen.values())


def entries_to_relations(entries: list[dict]) -> list[dict]:
    """Extract explicit relations from PlanetMath entries.

    Sources:
    - :related field -> "related-to" relations
    - :defines field -> "defines" relations (entry defines a term)
    - :msc-codes field -> "classified-by" relations
    """
    relations = []
    seen = set()

    for e in entries:
        eid = e["id"]

        # related-to
        for target in e.get("related", []):
            key = ("related-to", eid, target)
            if key not in seen:
                seen.add(key)
                relations.append({
                    "relation/id": f"rel:{eid}->related-to->{target}",
                    "relation/from": eid,
                    "relation/to": target,
                    "relation/type": "related-to",
                })

        # defines
        for term in e.get("defines", []):
            term_id = f"term:{_slugify(term)}"
            key = ("defines", eid, term_id)
            if key not in seen:
                seen.add(key)
                relations.append({
                    "relation/id": f"rel:{eid}->defines->{term_id}",
                    "relation/from": eid,
                    "relation/to": term_id,
                    "relation/type": "defines",
                })

        # classified-by
        for msc in e.get("msc-codes", []):
            code = msc.get("code", "") if isinstance(msc, dict) else str(msc)
            msc_id = f"msc:{code}"
            key = ("classified-by", eid, msc_id)
            if key not in seen:
                seen.add(key)
                relations.append({
                    "relation/id": f"rel:{eid}->classified-by->{msc_id}",
                    "relation/from": eid,
                    "relation/to": msc_id,
                    "relation/type": "classified-by",
                })

    return relations


def _strip_ns(name: str) -> str:
    """Strip namespace prefix from a key name."""
    return name.split("/", 1)[1] if "/" in name else name


def _slugify(s: str) -> str:
    """Simple slug for term IDs."""
    return s.lower().replace(" ", "-").replace("/", "-")[:80]


def extract_term_entities(entries: list[dict]) -> list[dict]:
    """Create entities for defined terms."""
    terms = {}
    for e in entries:
        for term in e.get("defines", []):
            tid = f"term:{_slugify(term)}"
            if tid not in terms:
                terms[tid] = {
                    "entity/id": tid,
                    "entity/type": "DefinedTerm",
                    "entity/source": "planetmath",
                    "title": term,
                }
    return list(terms.values())


def extract_msc_entities(entries: list[dict]) -> list[dict]:
    """Create entities for MSC classification codes."""
    codes = {}
    for e in entries:
        for msc in e.get("msc-codes", []):
            code = msc.get("code", "") if isinstance(msc, dict) else str(msc)
            mid = f"msc:{code}"
            if mid not in codes:
                codes[mid] = {
                    "entity/id": mid,
                    "entity/type": "MSCCode",
                    "entity/source": "msc",
                    "code": code,
                    "parent": code[:2] if len(code) > 2 else None,
                }
    return list(codes.values())


def load_tex_dir(tex_dir: str) -> dict[str, dict]:
    """Load .tex files from a PlanetMath MSC directory.

    Returns a dict keyed by canonical name (from \\pmcanonicalname),
    with values containing the full body text and metadata extracted
    from the preamble.
    """
    tex_dir = Path(tex_dir)
    entries = {}

    for tex_file in sorted(tex_dir.glob("*.tex")):
        raw = tex_file.read_text(errors="replace")
        canonical = _extract_pm_field(raw, "pmcanonicalname")
        if not canonical:
            continue

        # Extract document body (after \begin{document})
        body_match = re.search(
            r"\\begin\{document\}(.*?)(?:\\end\{document\}|$)",
            raw, re.DOTALL,
        )
        body = body_match.group(1).strip() if body_match else ""

        # Extract preamble metadata
        related = re.findall(r"\\pmrelated\{([^}]+)\}", raw)
        synonyms = [m.group(1) for m in
                     re.finditer(r"\\pmsynonym\{([^}]*)\}\{[^}]*\}", raw)]

        entries[canonical] = {
            "canonical_name": canonical,
            "body_full": body,
            "related_tex": related,
            "synonyms": synonyms,
            "source_tex": tex_file.name,
        }

    return entries


def _extract_pm_field(raw: str, field: str) -> str | None:
    """Extract a \\pm<field>{value} from LaTeX preamble."""
    m = re.search(rf"\\{field}\{{([^}}]+)\}}", raw)
    return m.group(1) if m else None


def merge_tex_bodies(entries: list[dict], tex_data: dict[str, dict]) -> list[dict]:
    """Merge full .tex body text and metadata into EDN entries.

    Matches by entry ID -> canonical name.
    """
    merged = []
    for e in entries:
        eid = e["id"]
        tex = tex_data.get(eid, {})
        updated = {**e}
        if tex:
            updated["body"] = tex.get("body_full", e.get("body", ""))
            # Add synonyms from .tex that aren't in defines
            existing_defines = set(e.get("defines", []))
            new_synonyms = [s for s in tex.get("synonyms", [])
                           if s not in existing_defines]
            if new_synonyms:
                updated["defines"] = e.get("defines", []) + new_synonyms
            # Add related from .tex that aren't in related
            existing_related = set(e.get("related", []))
            new_related = [r for r in tex.get("related_tex", [])
                          if r not in existing_related]
            if new_related:
                updated["related"] = e.get("related", []) + new_related
        merged.append(updated)
    return merged


def build_graph(edn_path: str, tex_dir: str | None = None) -> dict:
    """Load PlanetMath EDN and build the full entity/relation graph.

    If tex_dir is provided, merges full body text and additional metadata
    from .tex source files.

    Returns:
        {
            "entities": [...],
            "relations": [...],
            "stats": {"entries": N, "terms": N, "msc_codes": N, "relations": N}
        }
    """
    entries = load_edn(edn_path)
    if tex_dir:
        tex_data = load_tex_dir(tex_dir)
        entries = merge_tex_bodies(entries, tex_data)
    entities = entries_to_entities(entries)
    term_entities = extract_term_entities(entries)
    msc_entities = extract_msc_entities(entries)
    relations = entries_to_relations(entries)

    all_entities = entities + term_entities + msc_entities

    return {
        "entities": all_entities,
        "relations": relations,
        "stats": {
            "entries": len(entities),
            "terms": len(term_entities),
            "msc_codes": len(msc_entities),
            "relations": len(relations),
        },
    }
