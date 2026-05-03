"""Arxiv-aware Stage 3 pattern-tag prompt for superpod-mark3.

Replaces the math.SE Q&A prompt in `superpod-job.py` `build_pattern_prompt()`
when input is an arxiv paper. Uses the hierarchical paper-shape taxonomy
authored at `~/code/futon3/library/math-strategy/` + `math-informal/` —
five family parents and a curated set of leaf patterns, with member-pattern
lists declared in the family flexiargs and `@family <parent>` fields in
the new leaves.

Mission: M-superpod-mark3 (R-1).
Choice space: futon3/library/math-strategy/PAPER-SHAPES-INDEX.md.
Pilot validation: futon3/holes/excursions/E-math-prototype-pilot.md (24/25
papers fit without forcing).

This module is the local-only authoring deliverable; it does not yet
depend on a running pipeline. `superpod-mark3` will import
`build_arxiv_pattern_prompt` and `parse_arxiv_pattern_response` once Rob
deploys the Stage 3 fork.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_FLEXIARG_RE = re.compile(r"^@flexiarg\s+(\S+)\s*$")
_TITLE_RE = re.compile(r"^@title\s+(.*)\s*$")
_FAMILY_RE = re.compile(r"^@family\s+(\S+)\s*$")
_REFERENCES_RE = re.compile(r"^@references\s+\[(.*)\]\s*$")
_MEMBER_RE = re.compile(r"^\s*member\[(\S+?)\]")
_EXEMPLAR_RE = re.compile(r"^\s*exemplar\[(\S+?)\]")
_CONCLUSION_RE = re.compile(r"^!\s*conclusion:\s*$")
_CONTEXT_RE = re.compile(r"^\s*\+\s*context:\s*(.*)$")


DEFAULT_FUTON3_LIBRARY = Path.home() / "code" / "futon3" / "library"
FAMILY_PARENTS = [
    "math-strategy/existence-result",
    "math-strategy/characterization-result",
    "math-strategy/structural-relation-result",
    "math-strategy/property-of-object-result",
    "math-strategy/clarification-meta",
]


def _has_paper_shape_families(library_root: Path) -> bool:
    return all((library_root / f"{parent_id}.flexiarg").is_file() for parent_id in FAMILY_PARENTS)


def _default_futon3_library() -> Path:
    candidates: list[Path] = []
    if env_library := os.environ.get("FUTON3_LIBRARY"):
        candidates.append(Path(env_library).expanduser())
    if env_root := os.environ.get("FUTON3_ROOT"):
        candidates.append(Path(env_root).expanduser() / "library")
    candidates.append(DEFAULT_FUTON3_LIBRARY)

    # The superpod installation keeps futons as sibling checkouts under darktower.
    darktower_library = Path(__file__).resolve().parents[2] / "futon3" / "library"
    candidates.append(darktower_library)

    for candidate in candidates:
        if _has_paper_shape_families(candidate):
            return candidate
    return candidates[0]


@dataclass
class Pattern:
    pattern_id: str  # e.g. "math-informal/structural-equivalence"
    title: str
    family: Optional[str] = None  # e.g. "math-strategy/structural-relation-result"
    members: list[str] = field(default_factory=list)  # only on family parents
    exemplars: list[str] = field(default_factory=list)  # only on family parents
    one_liner: str = ""  # the conclusion / context line, used in the prompt


@dataclass
class PaperShapeTaxonomy:
    families: dict[str, Pattern]  # family_id -> Pattern
    leaves: dict[str, Pattern]    # leaf_id -> Pattern (with .family set)

    def all_leaves_for(self, family_id: str) -> list[Pattern]:
        return [p for p in self.leaves.values() if p.family == family_id]


def _parse_one_flexiarg(path: Path) -> Optional[Pattern]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    pattern_id: Optional[str] = None
    title: Optional[str] = None
    family: Optional[str] = None
    members: list[str] = []
    exemplars: list[str] = []
    in_conclusion = False
    one_liner_parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if m := _FLEXIARG_RE.match(line):
            pattern_id = m.group(1)
            continue
        if m := _TITLE_RE.match(line):
            title = m.group(1).strip()
            continue
        if m := _FAMILY_RE.match(line):
            family = m.group(1)
            continue
        if m := _MEMBER_RE.match(line):
            members.append(m.group(1))
            continue
        if m := _EXEMPLAR_RE.match(line):
            exemplars.append(m.group(1))
            continue
        if _CONCLUSION_RE.match(line):
            in_conclusion = True
            continue
        if in_conclusion:
            stripped = line.strip()
            if stripped and not stripped.startswith("+") and not stripped.startswith("@"):
                # Collect the conclusion's first non-clause sentence as the one-liner.
                one_liner_parts.append(stripped)
                if len(" ".join(one_liner_parts)) > 200:
                    in_conclusion = False
            elif stripped.startswith("+") and one_liner_parts:
                in_conclusion = False
    if not pattern_id:
        return None
    return Pattern(
        pattern_id=pattern_id,
        title=title or pattern_id,
        family=family,
        members=members,
        exemplars=exemplars,
        one_liner=" ".join(one_liner_parts).strip(),
    )


def load_paper_shape_taxonomy(library_root: Optional[Path] = None) -> PaperShapeTaxonomy:
    """Load the 5-family + leaves paper-shape taxonomy from futon3 flexiargs.

    Family parents declare member-pattern lists; leaves declare @family.
    Reverse-link members from the parents into the leaves table when a
    leaf's @family is missing (e.g. existing patterns that pre-date the
    taxonomy and haven't been retrofitted with @family).
    """
    if library_root is None:
        library_root = _default_futon3_library()
    families: dict[str, Pattern] = {}
    for parent_id in FAMILY_PARENTS:
        rel = parent_id + ".flexiarg"
        path = library_root / rel
        pat = _parse_one_flexiarg(path)
        if pat is None:
            raise FileNotFoundError(f"Family parent missing: {path}")
        families[parent_id] = pat

    leaves: dict[str, Pattern] = {}
    # Walk math-informal/ and math-strategy/ for any pattern with @family or
    # any pattern named in a parent's MEMBER-PATTERNS list.
    declared_member_to_family: dict[str, str] = {}
    for fid, fam in families.items():
        for member in fam.members:
            declared_member_to_family[member] = fid

    for subdir in ("math-informal", "math-strategy"):
        for path in (library_root / subdir).glob("*.flexiarg"):
            pat = _parse_one_flexiarg(path)
            if pat is None or pat.pattern_id in families:
                continue
            if not pat.family and pat.pattern_id in declared_member_to_family:
                pat.family = declared_member_to_family[pat.pattern_id]
            if pat.family:
                leaves[pat.pattern_id] = pat
    return PaperShapeTaxonomy(families=families, leaves=leaves)


def build_arxiv_pattern_prompt(
    paper_id: str,
    title: str,
    abstract: str,
    theorem_excerpts: Optional[list[str]] = None,
    proof_excerpts: Optional[list[str]] = None,
    taxonomy: Optional[PaperShapeTaxonomy] = None,
    char_budget_per_excerpt: int = 600,
    char_budget_abstract: int = 1200,
) -> str:
    """Build the arxiv-aware Stage 3 prompt for one paper.

    The LLM is asked to classify the paper's contribution shape against
    the hierarchical paper-shape taxonomy and emit a structured JSON
    response. Coverage discipline is enforced by requiring the LLM to
    emit `family: clarification-meta` with a `:reason` if the paper's
    `(context, tension, move)` triple is collapsed into one description.
    """
    if taxonomy is None:
        taxonomy = load_paper_shape_taxonomy()

    family_block_lines: list[str] = []
    for fid, fam in taxonomy.families.items():
        family_block_lines.append(f"  - {fid}: {fam.one_liner or fam.title}")

    leaf_block_lines: list[str] = []
    for fid in taxonomy.families:
        children = taxonomy.all_leaves_for(fid)
        if not children:
            continue
        leaf_block_lines.append(f"  Under {fid}:")
        for leaf in children:
            descr = leaf.one_liner or leaf.title
            descr = descr[:120]
            leaf_block_lines.append(f"    - {leaf.pattern_id}: {descr}")
    family_block = "\n".join(family_block_lines)
    leaf_block = "\n".join(leaf_block_lines)

    abstract_clip = abstract[:char_budget_abstract] if abstract else ""
    th_lines: list[str] = []
    if theorem_excerpts:
        th_lines.append("\nTheorem excerpts:")
        for i, exc in enumerate(theorem_excerpts[:3], start=1):
            th_lines.append(f"  ({i}) {exc[:char_budget_per_excerpt]}")
    pr_lines: list[str] = []
    if proof_excerpts:
        pr_lines.append("\nProof excerpts:")
        for i, exc in enumerate(proof_excerpts[:3], start=1):
            pr_lines.append(f"  ({i}) {exc[:char_budget_per_excerpt]}")

    return f"""You are a mathematics paper-shape classifier.

Given a paper's title, abstract, and (optionally) theorem and proof excerpts,
identify the *kind of contribution* the paper makes — what SHAPE of result it
produces. The choice space is hierarchical: pick exactly one FAMILY (mandatory)
and ideally one LEAF (mandatory unless your leaf-level confidence is below
0.6, in which case set leaf to "uncertain" and the family-only signal is
still useful).

FAMILIES:
{family_block}

LEAVES:
{leaf_block}

OUTPUT:
Reply with ONE JSON object, no surrounding prose, with these keys:
  - "family": string, one of the family ids above
  - "leaf": string, one of the leaf ids OR "uncertain"
  - "family_confidence": float in [0, 1]
  - "leaf_confidence": float in [0, 1]
  - "rationale": 1-2 sentences explaining the family + leaf choice
  - "collapsed": optional; required only if family is "math-strategy/clarification-meta"
      with structure {{"reason": "single-axis" | "extraction-failure" | "other",
                       "explanation": string}}

Coverage rule: if the paper's contribution cannot be cleanly assigned to a
strategic family — typically because the (context, tension, move) of the
paper paraphrases as one sentence — emit family "math-strategy/clarification-meta"
with the reason explicitly stated. Do NOT default to "uncertain"; that is for
leaf-level uncertainty within an otherwise-confident family.

Slot-distinctness rule: the substrate's `(situation_S, xiang_salience,
arrow_constraint)` triple should be substantively distinct (not paraphrases
of each other). If you cannot tell because the abstract is too short, prefer
to emit clarification-meta with reason "extraction-failure".

PAPER:
paper_id: {paper_id}
title: {title}
abstract: {abstract_clip}{"".join(th_lines)}{"".join(pr_lines)}

JSON output:"""


_VALID_FAMILY_IDS = set(FAMILY_PARENTS)
_INVALID_JSON_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


def _json_loads_tolerating_tex_escapes(blob: str) -> tuple[dict | None, str | None]:
    """Parse LLM JSON, retrying after escaping raw TeX backslashes.

    Llama often writes rationale strings containing TeX such as ``\\mathbb``.
    That is semantically valid text but invalid JSON because ``\\m`` is not an
    allowed JSON escape. Doubling only invalid backslashes preserves ordinary
    JSON escapes while recovering these otherwise-useful Stage 3 records.
    """
    try:
        obj = json.loads(blob)
        return (obj if isinstance(obj, dict) else None), None
    except json.JSONDecodeError as first_exc:
        repaired = _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", blob)
        if repaired == blob:
            return None, f"json-decode: {first_exc}"
        try:
            obj = json.loads(repaired)
            return (obj if isinstance(obj, dict) else None), None
        except json.JSONDecodeError as second_exc:
            return None, f"json-decode: {second_exc}"


def _coerce_confidence(value, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def parse_arxiv_pattern_response(
    raw: str,
    taxonomy: Optional[PaperShapeTaxonomy] = None,
) -> dict:
    """Validate and normalise an LLM JSON response to the arxiv-pattern prompt.

    Returns a dict with keys:
      ok, family, leaf, family_confidence, leaf_confidence, rationale,
      collapsed, error.
    `ok` is False if parsing failed; `error` describes the failure mode in
    that case. A failure here should be surfaced as coverage-discipline
    record `:status :failed :reason :stage3-parse-error`.
    """
    if taxonomy is None:
        taxonomy = load_paper_shape_taxonomy()
    text = raw.strip()
    # Tolerate leading/trailing prose by extracting the first {...} block.
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start < 0 or brace_end < brace_start:
        return {"ok": False, "error": "no-json-object", "raw_excerpt": text[:200]}
    blob = text[brace_start:brace_end + 1]
    obj, json_error = _json_loads_tolerating_tex_escapes(blob)
    if obj is None:
        return {
            "ok": False,
            "error": json_error or "json-not-object",
            "raw_excerpt": blob[:200],
        }

    family = obj.get("family", "")
    leaf = obj.get("leaf", "")
    if family not in _VALID_FAMILY_IDS:
        return {
            "ok": False,
            "error": f"invalid-family: {family!r}",
            "raw_excerpt": blob[:200],
        }
    warnings: list[str] = []
    if leaf != "uncertain" and leaf not in taxonomy.leaves and leaf:
        if family == "math-strategy/clarification-meta":
            leaf = ""
        else:
            warnings.append(f"invalid-leaf-normalized: {leaf!r}")
            leaf = "uncertain"
            obj["leaf_confidence"] = min(
                _coerce_confidence(obj.get("leaf_confidence")),
                0.5,
            )

    if family == "math-strategy/clarification-meta":
        collapsed = obj.get("collapsed")
        if not isinstance(collapsed, dict) or "reason" not in collapsed:
            warnings.append("clarification-meta-collapsed-synthesized")
            obj["collapsed"] = {
                "reason": "other",
                "explanation": str(
                    obj.get("rationale", "") or "LLM omitted collapsed metadata."
                ),
            }

    return {
        "ok": True,
        "family": family,
        "leaf": leaf or None,
        "family_confidence": _coerce_confidence(obj.get("family_confidence")),
        "leaf_confidence": _coerce_confidence(obj.get("leaf_confidence")),
        "rationale": obj.get("rationale", ""),
        "collapsed": obj.get("collapsed"),
        "warnings": warnings,
    }
