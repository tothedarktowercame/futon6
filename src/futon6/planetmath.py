"""PlanetMath EDN loader.

Reads PlanetMath EDN exports and produces normalized entry dicts
suitable for graph construction.
"""

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


def build_graph(edn_path: str) -> dict:
    """Load PlanetMath EDN and build the full entity/relation graph.

    Returns:
        {
            "entities": [...],
            "relations": [...],
            "stats": {"entries": N, "terms": N, "msc_codes": N, "relations": N}
        }
    """
    entries = load_edn(edn_path)
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
