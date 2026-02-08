"""LaTeX term extraction — low-effort NLP for mathematical text.

Extracts mathematical terms, references, and structural elements from
LaTeX body text without requiring a full parser. This is intentionally
simple — regex-based extraction that catches the common patterns in
PlanetMath content.

For futon6 enrichment: these extracted terms become candidate relations
that the explicit :related and :defines fields don't capture.
"""

import re
from collections import Counter


# LaTeX commands that introduce defined terms
_DEFINE_PATTERNS = [
    # \textbf{term} — bold terms are often definitions
    r"\\textbf\{([^}]+)\}",
    # \emph{term} — emphasized terms
    r"\\emph\{([^}]+)\}",
    # \textit{term} — italic terms
    r"\\textit\{([^}]+)\}",
    # \PMlinkname{display}{target} — PlanetMath cross-references
    r"\\PMlinkname\{[^}]*\}\{([^}]+)\}",
    # \PMlinkid{display}{id} — PlanetMath ID links
    r"\\PMlinkid\{[^}]*\}\{([^}]+)\}",
    # \PMlinkexternal{display}{url} — skip these (external)
]

# LaTeX environments that contain mathematical content
_ENV_PATTERN = re.compile(
    r"\\begin\{(\w+)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

# Cross-reference commands
_XREF_PATTERNS = [
    r"\\PMlinkname\{([^}]*)\}\{([^}]+)\}",  # display, target
    r"\\PMlinkid\{([^}]*)\}\{([^}]+)\}",     # display, id
]


def extract_terms(body: str) -> list[str]:
    """Extract emphasized/bold terms from LaTeX body text.

    Returns a deduplicated list of terms found via \\textbf, \\emph, etc.
    """
    terms = []
    for pattern in _DEFINE_PATTERNS:
        for match in re.finditer(pattern, body):
            term = match.group(1).strip()
            if term and len(term) > 1 and not term.startswith("\\"):
                terms.append(term)
    return list(dict.fromkeys(terms))  # dedupe preserving order


def extract_xrefs(body: str) -> list[dict]:
    """Extract PlanetMath cross-references from LaTeX body text.

    Returns list of {display, target} dicts.
    """
    xrefs = []
    for pattern in _XREF_PATTERNS:
        for match in re.finditer(pattern, body):
            xrefs.append({
                "display": match.group(1).strip(),
                "target": match.group(2).strip(),
            })
    return xrefs


def extract_environments(body: str) -> list[dict]:
    """Extract named LaTeX environments (theorem, proof, lemma, etc.)."""
    envs = []
    for match in _ENV_PATTERN.finditer(body):
        env_name = match.group(1)
        if env_name in (
            "theorem", "lemma", "proposition", "corollary",
            "definition", "example", "remark", "proof",
        ):
            envs.append({
                "environment": env_name,
                "content": match.group(2).strip()[:200],  # truncate
            })
    return envs


def enrich_entry(entry: dict) -> dict:
    """Add NLP-extracted fields to a PlanetMath entry.

    Adds:
    - extracted_terms: terms found in body text
    - extracted_xrefs: cross-references found in body text
    - extracted_envs: mathematical environments found
    - implicit_relations: candidate relations not in explicit :related
    """
    body = entry.get("body", "")
    explicit_related = set(entry.get("related", []))
    explicit_defines = set(entry.get("defines", []))

    terms = extract_terms(body)
    xrefs = extract_xrefs(body)
    envs = extract_environments(body)

    # Find implicit relations: xref targets not already in :related
    implicit = []
    for xref in xrefs:
        target = xref["target"]
        if target not in explicit_related:
            implicit.append({
                "target": target,
                "display": xref["display"],
                "source": "latex-xref",
            })

    # Find implicit terms: extracted terms not already in :defines
    new_terms = [t for t in terms if t.lower() not in
                 {d.lower() for d in explicit_defines}]

    return {
        **entry,
        "extracted_terms": terms,
        "extracted_xrefs": xrefs,
        "extracted_envs": envs,
        "implicit_relations": implicit,
        "new_terms": new_terms,
    }


def enrich_all(entries: list[dict]) -> list[dict]:
    """Enrich all entries and return summary stats."""
    enriched = [enrich_entry(e) for e in entries]
    return enriched


def enrichment_stats(enriched: list[dict]) -> dict:
    """Summarize what enrichment found."""
    total_implicit_rels = sum(len(e.get("implicit_relations", []))
                             for e in enriched)
    total_new_terms = sum(len(e.get("new_terms", [])) for e in enriched)
    total_xrefs = sum(len(e.get("extracted_xrefs", [])) for e in enriched)
    total_envs = sum(len(e.get("extracted_envs", [])) for e in enriched)
    entries_with_implicit = sum(1 for e in enriched
                                if e.get("implicit_relations"))

    return {
        "entries_enriched": len(enriched),
        "total_extracted_xrefs": total_xrefs,
        "total_implicit_relations": total_implicit_rels,
        "entries_with_implicit_relations": entries_with_implicit,
        "total_new_terms": total_new_terms,
        "total_environments": total_envs,
    }
