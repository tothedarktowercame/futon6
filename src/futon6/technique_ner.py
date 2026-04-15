"""Stage 5c: Technique-level NER for papers.

Extracts scope-aware, multi-word technique terms (e.g., "Borel completion
adjunction", "Pettis integral as monad algebra map") rather than
concept-level atoms (e.g., "functor", "adjunction").

Two extraction arms are kept distinct so batch-level analysis can attribute
signal to each:

  extract_techniques_classical(text, concepts=..., ...)
  extract_techniques_llm(text, pipe=..., tokenizer=..., ...)

Output schema matches M-paper-reverse-morphogenesis.md §Stage 5c.

The classical arm is deterministic and cheap (regex + NP-pattern matching
anchored on concept-level terms from stage 5). The LLM arm is a single
few-shot prompt. Provenance is recorded per term: "classical", "llm", or
"both". Union of both arms is what downstream stages consume; intersection
is a high-confidence subset useful for evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# --- Classical arm ---------------------------------------------------------

# Head nouns that typically name a technique, not just a concept. A term is a
# technique candidate if one of these heads appears with qualifying modifiers.
TECHNIQUE_HEADS = frozenset({
    "adjunction", "approach", "argument", "bound", "calculation",
    "characterization", "classification", "completion", "compactification",
    "computation", "construction", "criterion", "decomposition", "embedding",
    "estimate", "extension", "factorization", "formalism", "formula",
    "framework", "identity", "inequality", "lemma", "localization",
    "machinery", "method", "principle", "procedure", "proof", "reduction",
    "representation", "resolution", "sequence", "technique", "theorem",
    "transform", "trick",
})

_HEADS_ALT = "|".join(sorted(TECHNIQUE_HEADS))

# "(The|A|An) <Proper> [<modifier>]{0,3} <TECH_HEAD>" — case-sensitive on the
# first modifier to force proper-noun anchoring. "the" lowercase is fine;
# what matters is that the phrase after it starts with a capital.
_THE_X_HEAD = re.compile(
    r"\b(?:[Tt]he|[Aa]n?)\s+"
    r"(?P<modifiers>[A-Z][A-Za-z0-9'\u2019-]+"
    r"(?:[\s-][A-Za-z][A-Za-z0-9'\u2019-]*){0,3})"
    r"\s+(?P<head>" + _HEADS_ALT + r")\b",
)

# "<Proper>'s <TECH_HEAD>" — e.g. "Wantzel's theorem"
_POSSESSIVE_HEAD = re.compile(
    r"\b(?P<owner>[A-Z][A-Za-z0-9'\u2019-]+)"
    r"(?:'s|\u2019s)\s+"
    r"(?P<head>" + _HEADS_ALT + r")\b",
)

# "by (applying|using|invoking) (the)? <NP>" — e.g. "by applying the spectral
# sequence". Proper-noun anchored modifier.
_BY_APPLYING = re.compile(
    r"\bby\s+(?:applying|using|invoking|means\s+of)\s+(?:the\s+)?"
    r"(?P<term>[A-Z][A-Za-z0-9'\u2019-]+(?:[\s-][A-Za-z0-9'\u2019-]+){0,3})"
    r"(?:\s+(?P<head>" + _HEADS_ALT + r"))?\b",
)

# Named constructions like "X-Y construction" / "X-Y adjunction"
_HYPHENATED_HEAD = re.compile(
    r"\b(?P<term>[A-Z][A-Za-z0-9'\u2019]+"
    r"(?:-[A-Z][A-Za-z0-9'\u2019]+){1,3})"
    r"\s+(?P<head>" + _HEADS_ALT + r")\b",
)

_PATTERNS = [
    ("the_x_head", _THE_X_HEAD),
    ("possessive_head", _POSSESSIVE_HEAD),
    ("by_applying", _BY_APPLYING),
    ("hyphenated_head", _HYPHENATED_HEAD),
]


def _canonicalize(term: str) -> str:
    """Lower, collapse whitespace, trim punctuation. Used as the term key."""
    t = term.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(".,;:!?()[]{}\"'\u2019")
    return t


def _looks_like_technique(term: str, head: str | None) -> bool:
    """Heuristic: require at least one capitalized token OR a head noun.

    Prunes trivial hits like "the theorem" (no proper modifier, no real
    technique signature)."""
    if head is None:
        # Must have a capitalized proper modifier
        return any(t[:1].isupper() for t in term.split())
    # Head present; require at least one non-function-word modifier
    stopwords = {"a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "by"}
    tokens = [t for t in term.split() if t.lower() not in stopwords]
    return len(tokens) >= 1


def _paragraph_index(text: str, char_offset: int) -> int:
    """Paragraph number (0-indexed), computed by counting double-newlines."""
    return text.count("\n\n", 0, char_offset)


def _section_for_offset(section_spans: Sequence[tuple[int, int, str]],
                        offset: int) -> str:
    """Return section id (or '0') for a char offset given section span list."""
    for start, end, sid in section_spans:
        if start <= offset < end:
            return sid
    return "0"


def _parse_section_spans(text: str) -> list[tuple[int, int, str]]:
    """Parse LaTeX `\\section{...}` headers to produce (start, end, id) spans.

    Falls back to a single span covering the whole text if no section
    headers are present."""
    spans: list[tuple[int, int, str]] = []
    pat = re.compile(r"\\(?:section|subsection)\*?\{([^}]*)\}")
    matches = list(pat.finditer(text))
    if not matches:
        return [(0, len(text), "0")]
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sid = str(i + 1)
        spans.append((start, end, sid))
    # Prepend preamble (anything before the first section) as section 0.
    if matches[0].start() > 0:
        spans.insert(0, (0, matches[0].start(), "0"))
    return spans


@dataclass
class TechniqueHit:
    canonical: str
    term: str
    loci: list[dict] = field(default_factory=list)
    first_defined_at: dict | None = None
    extraction_source: str = "classical"  # classical | llm | both


def extract_techniques_classical(
    text: str,
    concepts: Iterable[str] | None = None,
    max_terms: int = 256,
) -> list[TechniqueHit]:
    """Classical technique-NER: pattern-based, no LLM.

    Args:
        text: paper body (prose; LaTeX markers like \\section are honored
            for locus attribution).
        concepts: optional concept vocabulary (stage 5 output terms). When
            supplied, candidates containing at least one concept get a
            small boost in locus-count tiebreaking — not filtered, just
            prioritized when we truncate to max_terms.
        max_terms: cap the output list size.

    Returns:
        list of TechniqueHit records.
    """
    section_spans = _parse_section_spans(text)
    concept_set = {c.lower() for c in concepts} if concepts else set()

    hits: dict[str, TechniqueHit] = {}

    def _record(raw_term: str, head: str | None, start: int, end: int,
                pattern_name: str):
        canon = _canonicalize(raw_term if head is None else f"{raw_term} {head}")
        if not canon or len(canon) < 3:
            return
        if not _looks_like_technique(canon, head):
            return
        locus = {
            "section": _section_for_offset(section_spans, start),
            "paragraph": _paragraph_index(text, start),
            "char_span": [start, end],
            "pattern": pattern_name,
        }
        hit = hits.get(canon)
        if hit is None:
            display = (raw_term.strip() if head is None
                       else f"{raw_term.strip()} {head}")
            hits[canon] = TechniqueHit(
                canonical=canon,
                term=display,
                loci=[locus],
                first_defined_at={"section": locus["section"],
                                  "paragraph": locus["paragraph"]},
                extraction_source="classical",
            )
        else:
            hit.loci.append(locus)

    for pattern_name, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            head = m.groupdict().get("head")
            if pattern_name == "the_x_head":
                raw = m.group("modifiers").strip()
            elif pattern_name == "possessive_head":
                raw = m.group("owner").strip()
            elif pattern_name == "by_applying":
                raw = m.group("term").strip()
            elif pattern_name == "hyphenated_head":
                raw = m.group("term").strip()
            else:
                continue
            _record(raw, head, m.start(), m.end(), pattern_name)

    def _rank_key(hit: TechniqueHit) -> tuple[int, int]:
        locus_count = len(hit.loci)
        concept_overlap = sum(1 for c in concept_set if c in hit.canonical)
        return (-locus_count, -concept_overlap)

    ordered = sorted(hits.values(), key=_rank_key)
    return ordered[:max_terms]


# --- LLM arm ---------------------------------------------------------------

LLM_FEWSHOT_SEED = [
    {
        "paper_snippet": (
            "We use a Borel completion adjunction to lift the measurable "
            "structure from the base space to the completion. This "
            "adjunction, introduced in Section 3, is the key technical "
            "ingredient."
        ),
        "techniques": [
            {"term": "Borel completion adjunction",
             "role": "primary",
             "rationale": "named construction, described as key ingredient"},
        ],
    },
    {
        "paper_snippet": (
            "By Wantzel's theorem, any number constructible by ruler and "
            "compass lies in the quadratic closure of Q. We invoke this "
            "to rule out trisection of arbitrary angles."
        ),
        "techniques": [
            {"term": "Wantzel's theorem",
             "role": "primary",
             "rationale": "invoked as the key reason"},
            {"term": "ruler and compass construction",
             "role": "primary",
             "rationale": "names the technique-space of the argument"},
        ],
    },
    {
        "paper_snippet": (
            "The argument proceeds by a standard spectral sequence "
            "computation followed by a degeneration argument."
        ),
        "techniques": [
            {"term": "spectral sequence computation",
             "role": "primary",
             "rationale": "named method driving the argument"},
            {"term": "degeneration argument",
             "role": "supporting",
             "rationale": "second-step technique"},
        ],
    },
]


def _build_llm_prompt(paper_text: str, max_chars: int = 6000) -> str:
    """Construct a few-shot prompt for LLM technique extraction."""
    snippet = paper_text if len(paper_text) <= max_chars else (
        paper_text[: max_chars // 2]
        + "\n\n[...TRUNCATED...]\n\n"
        + paper_text[-max_chars // 2:]
    )

    examples = []
    for ex in LLM_FEWSHOT_SEED:
        examples.append(
            f"Paper snippet:\n{ex['paper_snippet']}\n\n"
            f"Techniques:\n{json.dumps(ex['techniques'], indent=2)}"
        )
    few_shot = "\n\n---\n\n".join(examples)

    return (
        "You are extracting technique-level terms from a mathematics "
        "paper. A technique-level term is a named method, construction, "
        "theorem, or argument that carries problem-solving force — not a "
        "bare concept like \"functor\" or \"group\", but a composite phrase "
        "like \"Borel completion adjunction\" or \"ruler and compass "
        "construction\" that names a specific mathematical move.\n\n"
        "For each technique, return its term, its role (primary = drives "
        "the main argument; supporting = key subargument; auxiliary = "
        "used but incidental), and a short rationale.\n\n"
        "Return ONLY a JSON array of objects with keys: term, role, "
        "rationale. No prose outside the JSON.\n\n"
        f"EXAMPLES:\n\n{few_shot}\n\n---\n\n"
        f"NOW EXTRACT FROM:\n{snippet}\n\n"
        "Techniques:"
    )


def _parse_llm_response(text: str) -> list[dict]:
    """Parse the LLM's JSON array response. Tolerant of surrounding prose."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict) and x.get("term")]


def extract_techniques_llm(
    text: str,
    pipe,
    tokenizer,
    max_new_tokens: int = 512,
    max_input_chars: int = 6000,
) -> list[TechniqueHit]:
    """LLM-backed technique-NER using a transformers text-generation pipeline.

    pipe / tokenizer are the shared LLM pipeline (created by
    `_create_llm_pipeline` in superpod-job.py, typically reused across
    stages 3/6/5c)."""
    prompt = _build_llm_prompt(text, max_chars=max_input_chars)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        [formatted],
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
    raw = outputs[0][0]["generated_text"] if outputs else ""
    entries = _parse_llm_response(raw)

    section_spans = _parse_section_spans(text)
    hits: dict[str, TechniqueHit] = {}
    lower_text = text.lower()

    for entry in entries:
        term = str(entry.get("term", "")).strip()
        if not term:
            continue
        canon = _canonicalize(term)
        if not canon:
            continue
        loci: list[dict] = []
        start = 0
        needle = canon
        while True:
            idx = lower_text.find(needle, start)
            if idx < 0:
                break
            loci.append({
                "section": _section_for_offset(section_spans, idx),
                "paragraph": _paragraph_index(text, idx),
                "char_span": [idx, idx + len(needle)],
                "pattern": "llm",
            })
            start = idx + len(needle)
        if not loci:
            loci.append({
                "section": "0",
                "paragraph": 0,
                "char_span": [0, 0],
                "pattern": "llm-unlocalized",
            })
        hits[canon] = TechniqueHit(
            canonical=canon,
            term=term,
            loci=loci,
            first_defined_at={"section": loci[0]["section"],
                              "paragraph": loci[0]["paragraph"]},
            extraction_source="llm",
        )

    return list(hits.values())


# --- Merge --------------------------------------------------------------


def merge_technique_arms(
    classical: list[TechniqueHit],
    llm: list[TechniqueHit],
) -> list[TechniqueHit]:
    """Union classical + LLM outputs, marking intersection as 'both'.

    Canonicalization collisions are intersections. Loci are unioned and
    de-duplicated by char_span."""
    merged: dict[str, TechniqueHit] = {}

    for hit in classical:
        merged[hit.canonical] = TechniqueHit(
            canonical=hit.canonical,
            term=hit.term,
            loci=list(hit.loci),
            first_defined_at=hit.first_defined_at,
            extraction_source="classical",
        )

    for hit in llm:
        existing = merged.get(hit.canonical)
        if existing is None:
            merged[hit.canonical] = TechniqueHit(
                canonical=hit.canonical,
                term=hit.term,
                loci=list(hit.loci),
                first_defined_at=hit.first_defined_at,
                extraction_source="llm",
            )
        else:
            seen_spans = {tuple(l.get("char_span", [0, 0])) for l in existing.loci}
            for l in hit.loci:
                span = tuple(l.get("char_span", [0, 0]))
                if span not in seen_spans:
                    existing.loci.append(l)
                    seen_spans.add(span)
            existing.extraction_source = "both"

    return list(merged.values())


def techniques_to_records(
    paper_id: str,
    techniques: list[TechniqueHit],
) -> dict:
    """Convert merged technique hits to the spec schema JSON record."""
    return {
        "paper_id": paper_id,
        "techniques": [
            {
                "term": hit.term,
                "canonical": hit.canonical,
                "loci": hit.loci,
                "first_defined_at": hit.first_defined_at,
                "extraction_source": hit.extraction_source,
            }
            for hit in techniques
        ],
    }
