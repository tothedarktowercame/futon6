#!/usr/bin/env python3
"""Structure-first per-concept reducer.

This is the reduce stage after ``concept-index.json``: group a concept's
definition/use instances into a compact family with a common genus and explicit
variant axes.  The v0 reducer is intentionally classical and deterministic; it
does not attempt symbolic proof of equivalence between variants.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_term_prior  # noqa: E402

DEFAULT_CONCEPT_INDEX = ROOT / "data" / "warp" / "concept-index.json"
DEFAULT_SNIPPETS = ROOT / "data" / "warp" / "def-snippets.json"
DEFAULT_ENCYCLOPEDIA = ROOT / "data" / "concept-encyclopedia-ct.json"
DEFAULT_NLAB = ROOT / "data" / "nlab-wiring" / "pages.json"
DEFAULT_FIXTURE = ROOT / "data" / "warp" / "sfc-adjunction-fixture.json"
DEFAULT_REPORT = ROOT / "holes" / "excursions" / "sfc-concept-aggregate.md"
PLANETMATH_DIR = Path("/home/joe/code/planetmath/18_Category_theory_homological_algebra")

NOISE_PREFIXES = {
    "all",
    "any",
    "both",
    "each",
    "every",
    "more",
    "no",
    "one",
    "some",
    "there",
    "these",
    "those",
    "two",
}
NOISE_PHRASES = {
    "all functors",
    "any two",
    "each other",
    "more generally",
    "there exists",
}
IDIOM_CORES = {
    "any two": "pair",
    "each other": "relation",
}
BAD_CORE_WORDS = set(build_term_prior._STOP) | {"other", "there"}

ADJUNCTION_KEYS = (
    "adjoint functor",
    "adjoint functors",
    "left adjoint",
    "right adjoint",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def normalize_space(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def strip_tex(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = text.replace("$", " ")
    text = re.sub(r"[{}]", " ", text)
    return normalize_space(text)


def singularize(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def df_prior_from_index(concept_index: dict[str, Any]) -> Counter[str]:
    df: Counter[str] = Counter()
    for concept, row in concept_index.items():
        if isinstance(row, dict):
            df[str(concept)] = int(row.get("df") or 0)
    return df


def head_core(surface: str, df: Counter[str], *, min_papers: int = 3) -> tuple[str | None, dict[str, Any]]:
    """Return a retained core for a quantifier/determiner-noise surface.

    ``build_term_prior.resolve_phrase`` is used first to expose existing
    corpus cores.  If the phrase itself is recurring but starts with a generic
    determiner, v0 falls back to a right-head heuristic and records the choice.
    """
    phrase = normalize_space(surface.lower())
    resolved = build_term_prior.resolve_phrase(phrase, df, min_papers=min_papers)
    if phrase in IDIOM_CORES:
        return IDIOM_CORES[phrase], {"method": "idiom", "term_prior": resolved}

    words = build_term_prior._WORD.findall(phrase)
    if not words:
        return None, {"method": "empty", "term_prior": resolved}
    trimmed = list(words)
    while trimmed and trimmed[0] in NOISE_PREFIXES:
        trimmed.pop(0)
    while trimmed and trimmed[-1] in BAD_CORE_WORDS:
        trimmed.pop()
    if not trimmed:
        return None, {"method": "no-content-head", "term_prior": resolved}

    head = singularize(trimmed[-1])
    if len(trimmed) >= 2 and trimmed[-2] not in BAD_CORE_WORDS:
        candidate = " ".join(trimmed[-2:-1] + [head])
    else:
        candidate = head

    candidate_resolution = build_term_prior.resolve_phrase(candidate, df, min_papers=min_papers)
    if candidate_resolution.get("resolution"):
        candidate = str(candidate_resolution["resolution"])
    return candidate, {
        "method": "right-head",
        "term_prior": resolved,
        "candidate_term_prior": candidate_resolution,
    }


def surface_to_core_map(
    concept_index: dict[str, Any],
    *,
    examples: tuple[str, ...] = ("all functors", "any two", "each other"),
    min_papers: int = 3,
) -> dict[str, Any]:
    df = df_prior_from_index(concept_index)
    out: dict[str, Any] = {}
    for surface, row in sorted(concept_index.items()):
        if not isinstance(row, dict):
            continue
        phrase = normalize_space(surface.lower())
        words = phrase.split()
        noisy = (
            phrase in NOISE_PHRASES
            or bool(words and words[0] in NOISE_PREFIXES)
            or row.get("genuine") is False and phrase in examples
        )
        if not noisy and phrase not in examples:
            continue
        core, evidence = head_core(phrase, df, min_papers=min_papers)
        action = "fold" if core else "retain-unfolded"
        out[phrase] = {
            "core": core,
            "action": action,
            "df": int(row.get("df") or 0),
            "retained_papers": len(row.get("papers") or []),
            "genuine": bool(row.get("genuine")),
            "evidence": evidence,
        }
    return out


def tex_excerpt(path: Path, marker: str, *, chars: int = 1600) -> str:
    text = path.read_text(errors="replace")
    i = text.lower().find(marker.lower())
    if i < 0:
        i = text.lower().find("\\begin{document}")
    if i < 0:
        i = 0
    return normalize_space(text[i : i + chars])


def add_instance(
    instances: list[dict[str, Any]],
    *,
    source: str,
    source_id: str,
    concept_surface: str,
    text: str,
    paper: str | None = None,
) -> None:
    instances.append(
        {
            "source": source,
            "source_id": source_id,
            "paper": paper,
            "concept_surface": concept_surface,
            "text": normalize_space(text),
        }
    )


def iter_nlab_pages(path: Path):
    with path.open(errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line in {"[", "]"}:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def nlab_adjunction_instances(path: Path, *, limit: int = 8) -> list[dict[str, Any]]:
    targets = ("adjunction", "adjoint functor", "left adjoint", "right adjoint")
    out: list[dict[str, Any]] = []
    for page in iter_nlab_pages(path):
        name = str(page.get("page_name") or "")
        envs = page.get("environments") or []
        texts = [nlab_env_text(env) for env in envs if isinstance(env, dict)]
        joined = " ".join([name] + texts).lower()
        if not any(t in joined for t in targets):
            continue
        for env in envs:
            if not isinstance(env, dict):
                continue
            text = nlab_env_text(env)
            lower = text.lower()
            if not text or not any(t in lower for t in targets):
                continue
            env_type = str(env.get("hx/type") or env.get("type") or "").lower()
            if env_type not in {"env/definition", "env/proposition", "env/remark", "definition", "proposition", "remark"}:
                continue
            add_instance(
                out,
                source="nLab",
                source_id=f"{page.get('page_id')}:{name}",
                concept_surface="adjunction",
                text=text[:1600],
            )
            break
        if len(out) >= limit:
            break
    return out


def nlab_env_text(env: dict[str, Any]) -> str:
    content = env.get("hx/content") if isinstance(env.get("hx/content"), dict) else {}
    return normalize_space(
        env.get("text")
        or content.get("text")
        or content.get("text_preview")
        or content.get("match")
        or ""
    )


def arxiv_adjunction_instances(snippets: dict[str, Any], *, limit_per_key: int = 4) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows_by_concept = snippets.get("snippets") or {}
    for concept in ADJUNCTION_KEYS:
        for row in rows_by_concept.get(concept, [])[:limit_per_key]:
            if not isinstance(row, dict):
                continue
            add_instance(
                out,
                source="arxiv-def-snippets",
                source_id=f"{row.get('paper')}:{concept}",
                paper=str(row.get("paper") or ""),
                concept_surface=concept,
                text=str(row.get("snippet") or ""),
            )
    return out


def assemble_adjunction_fixture(
    *,
    snippets_path: Path = DEFAULT_SNIPPETS,
    nlab_path: Path = DEFAULT_NLAB,
    planetmath_dir: Path = PLANETMATH_DIR,
) -> dict[str, Any]:
    snippets = load_json(snippets_path)
    instances: list[dict[str, Any]] = []

    add_instance(
        instances,
        source="PlanetMath",
        source_id="18A40-AdjointFunctor.tex",
        concept_surface="adjoint functor",
        text=tex_excerpt(planetmath_dir / "18A40-AdjointFunctor.tex", "left adjoint functor"),
    )
    add_instance(
        instances,
        source="PlanetMath",
        source_id="18A40-UnitOfAdjunction.tex",
        concept_surface="unit of adjunction",
        text=tex_excerpt(planetmath_dir / "18A40-UnitOfAdjunction.tex", "unit"),
    )
    instances.extend(nlab_adjunction_instances(nlab_path))
    instances.extend(arxiv_adjunction_instances(snippets))

    instances = sorted(
        instances,
        key=lambda r: (
            str(r.get("source")),
            str(r.get("source_id")),
            str(r.get("concept_surface")),
            str(r.get("text"))[:80],
        ),
    )
    return {
        "fixture": "adjunction",
        "canonical_concept": "adjunction",
        "aliases": ["adjoint functor", "adjoint functors", "left adjoint", "right adjoint"],
        "source_files": {
            "PlanetMath": [
                str(planetmath_dir / "18A40-AdjointFunctor.tex"),
                str(planetmath_dir / "18A40-UnitOfAdjunction.tex"),
            ],
            "nLab": str(nlab_path),
            "arxiv-def-snippets": str(snippets_path),
        },
        "instances": instances,
    }


def encyclopedia_entries(encyclopedia: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        normalize_space(str(row.get("concept") or "").lower()): row
        for row in encyclopedia.get("entries") or []
        if isinstance(row, dict) and row.get("concept")
    }


def seeded_genus(concept: str, aliases: list[str], encyclopedia: dict[str, Any]) -> tuple[str | None, str]:
    entries = encyclopedia_entries(encyclopedia)
    for key in [concept, *aliases]:
        row = entries.get(normalize_space(key.lower()))
        if not row:
            continue
        components = row.get("components") or {}
        genus = components.get("genus") if isinstance(components, dict) else None
        if isinstance(genus, str) and genus and genus.lower() not in {"sense", "object", "category"}:
            return genus, f"concept-encyclopedia:{key}"
    return None, "inferred-v0"


def classify_framing(text: str) -> str:
    lower = text.lower()
    texless = strip_tex(lower)
    if (
        "universal arrow" in texless
        or "universal property" in texless
        or "existence and uniqueness" in texless
    ):
        return "universal-arrow"
    if (
        "unit" in texless
        or "counit" in texless
        or "eta" in texless
        or "epsilon" in texless
        or "\\eta" in lower
        or "\\epsilon" in lower
    ):
        return "unit-counit-triangle"
    if (
        "hom" in texless
        and ("natural isomorphism" in texless or "natural equivalence" in texless or "bijection" in texless)
    ):
        return "hom-set-natural-bijection"
    return "contextual-use"


FRAMING_FIELDS = {
    "hom-set-natural-bijection": ["left_functor", "right_functor", "hom_equivalence", "naturality"],
    "unit-counit-triangle": ["unit", "counit", "triangle_identities"],
    "universal-arrow": ["universal_arrow_from", "universal_arrow_to", "unique_factorization"],
    "contextual-use": ["local_usage_context"],
}


def reduce_concept(fixture: dict[str, Any], encyclopedia: dict[str, Any]) -> dict[str, Any]:
    concept = str(fixture.get("canonical_concept") or fixture.get("fixture") or "concept")
    aliases = [str(x) for x in fixture.get("aliases") or []]
    genus, genus_source = seeded_genus(concept, aliases, encyclopedia)
    if concept == "adjunction" and not genus:
        genus = "adjunction F⊣G"

    grouped: dict[str, list[dict[str, Any]]] = {}
    for instance in fixture.get("instances") or []:
        label = classify_framing(str(instance.get("text") or ""))
        grouped.setdefault(label, []).append(instance)

    variants = []
    for label in sorted(grouped):
        rows = sorted(grouped[label], key=lambda r: (str(r.get("source")), str(r.get("source_id"))))
        variants.append(
            {
                "label": label,
                "fields": FRAMING_FIELDS.get(label, []),
                "instance_count": len(rows),
                "sources": sorted({str(r.get("source")) for r in rows}),
                "example_source_ids": [str(r.get("source_id")) for r in rows[:5]],
            }
        )

    bridge_labels = [v["label"] for v in variants if v["label"] != "contextual-use"]
    bridges = [
        {
            "from": a,
            "to": b,
            "kind": "iff-lemma",
            "status": "recorded-hole",
        }
        for i, a in enumerate(bridge_labels)
        for b in bridge_labels[i + 1 :]
    ]

    return {
        "concept": concept,
        "aliases": aliases,
        "genus": genus,
        "genus_source": genus_source,
        "schema": {
            "name": "lean-family-v0",
            "lean_analogy": (
                "bundle common fields as a structure-like genus; store each observed "
                "paper/source use as an instance; connect equivalent definition "
                "framings with explicit Iff bridge holes rather than collapsing them"
            ),
        },
        "variant_axes": [
            {
                "axis": "definition-framing",
                "representation": "labelled-family",
                "variants": variants,
                "bridges": bridges,
            }
        ],
        "instances": sorted(
            fixture.get("instances") or [],
            key=lambda r: (str(r.get("source")), str(r.get("source_id")), str(r.get("concept_surface"))),
        ),
    }


def report_markdown(
    *,
    fixture: dict[str, Any],
    reduced: dict[str, Any],
    gc_map: dict[str, Any],
    out_json: Path,
) -> str:
    source_counts = Counter(str(row.get("source")) for row in fixture.get("instances") or [])
    variants = reduced["variant_axes"][0]["variants"]
    lines = [
        "# SFC Concept Aggregate — Adjunction Fixture",
        "",
        "## Run",
        "",
        f"- Fixture JSON: `{out_json}`",
        f"- Instances: {len(fixture.get('instances') or [])}",
        "- Sources: "
        + ", ".join(f"{source}={source_counts[source]}" for source in sorted(source_counts)),
        f"- Genus: `{reduced['genus']}` ({reduced['genus_source']})",
        "",
        "## GC Surface -> Core Retention",
        "",
    ]
    for surface in ("all functors", "any two", "each other"):
        row = gc_map.get(surface, {})
        lines.append(
            f"- `{surface}` -> `{row.get('core')}`; action={row.get('action')}; "
            f"df={row.get('df')}; retained_papers={row.get('retained_papers')}"
        )
    lines.extend(
        [
            "",
            "## Variant-Axes Schema",
            "",
            "Schema `lean-family-v0`: keep a structure-like `genus`, retain every grounded "
            "`instance`, and represent divergent but equivalent definitions as a labelled "
            "family under `variant_axes[].variants`. Equivalence is recorded as explicit "
            "`iff-lemma` bridge holes, matching the Lean pattern of structures/classes with "
            "instances plus equivalence lemmas/defeq where available.",
            "",
            "Recovered definition-framing variants:",
        ]
    )
    for variant in variants:
        lines.append(
            f"- `{variant['label']}`: {variant['instance_count']} instances; "
            f"sources={', '.join(variant['sources'])}"
        )
    lines.extend(
        [
            "",
            "## Remaining holes",
            "",
            "- The reducer records equivalence bridges but does not prove them.",
            "- Framing classification is keyword/classical-prose based; formula grounding is delegated to H-SFC2b.",
            "- Genus inference falls back to the hand-recognised adjunction core when encyclopedia-v0 has only noisy genus data.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-index", type=Path, default=DEFAULT_CONCEPT_INDEX)
    ap.add_argument("--snippets", type=Path, default=DEFAULT_SNIPPETS)
    ap.add_argument("--encyclopedia", type=Path, default=DEFAULT_ENCYCLOPEDIA)
    ap.add_argument("--nlab", type=Path, default=DEFAULT_NLAB)
    ap.add_argument("--fixture-out", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--min-papers", type=int, default=3)
    args = ap.parse_args(argv)

    concept_index = load_json(args.concept_index)
    encyclopedia = load_json(args.encyclopedia)
    gc_map = surface_to_core_map(concept_index, min_papers=args.min_papers)
    fixture = assemble_adjunction_fixture(snippets_path=args.snippets, nlab_path=args.nlab)
    reduced = reduce_concept(fixture, encyclopedia)
    payload = {
        "surface_to_core": gc_map,
        "fixture": fixture,
        "reduced": reduced,
    }
    write_json(args.fixture_out, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_markdown(fixture=fixture, reduced=reduced, gc_map=gc_map, out_json=args.fixture_out))

    variants = reduced["variant_axes"][0]["variants"]
    print(
        f"wrote {args.fixture_out}; instances={len(fixture.get('instances') or [])}; "
        f"variants={','.join(v['label'] for v in variants)}"
    )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
