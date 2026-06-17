#!/usr/bin/env python3
"""Build GOLDEN-30 full-paper anatomy proofread pages.

The output is deliberately static HTML under data/showcases/ct-anatomy/golden/.
It uses the operator-approved mark families from golden-paragraph.html:

  - green: in-paper definition occurrence
  - amber dashed: honest hole, needs canon link
  - blue: appositive symbol/type bind

The detector is conservative by design. Precision matters more than recall for
the proofread queue these pages feed.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_SAMPLE = ROOT / "data" / "golden-30-sample.json"
DEFAULT_EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
DEFAULT_SCOPES = Path("/home/joe/code/storage/mark2/ct-fresh-scopes")
DEFAULT_OUT = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
DEFAULT_SUPERPOD = Path("/tmp/futon6-sbs/scripts/superpod-job.py")

from futon6.tex_env_scopes import detect_tex_env_scopes


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TITLE_CARD = load_module("paper_title_card_golden", ROOT / "scripts" / "paper_title_card.py")


@dataclass(frozen=True)
class Repair:
    damaged: str
    replacement: str
    count: int
    attestations: int


@dataclass(frozen=True)
class Definition:
    term: str
    position: int
    source: str


@dataclass(frozen=True)
class Mark:
    start: int
    end: int
    kind: str
    title: str
    label: str


COMMON_HOLE_STARTS = {
    "A", "An", "The", "This", "That", "These", "Those", "Such", "Some", "Any",
    "Every", "Let", "For", "If", "Then", "By", "In", "On", "We", "Our", "Their",
    "a", "an", "the", "this", "that", "these", "those", "such", "some", "any",
    "every", "let", "for", "if", "then", "by", "in", "on", "we", "our", "their",
}
HOLE_LEADING_TRIM = {
    "a", "an", "the", "and", "or", "has", "have", "without",
    "is", "are", "be", "being", "called",
}
CONCEPT_ENDINGS = (
    "category", "categories", "manifold", "manifolds", "functor", "functors",
    "algebra", "algebras", "space", "spaces", "group", "groups", "symmetry",
    "field", "fields", "geometry", "topology", "theory", "structure", "structures",
    "morphism", "morphisms", "module", "modules", "complex", "complexes",
    # CT vocabulary (DC-1, 2026-06-15): the dp-demo QC surfaced these endings
    # going unrecognised — subcategory/colimit/system etc. in 0905.0595.
    "subcategory", "subcategories", "colimit", "colimits", "limit", "limits",
    "system", "systems", "monad", "monads", "adjunction", "adjunctions",
    "transformation", "transformations", "equivalence", "equivalences",
    "representation", "representations", "form", "forms", "cohomology",
)
APPOSITIVE_ENDINGS = (
    "category", "manifold", "functor", "algebra", "space", "group", "module",
)


def paper_id_to_entity_id(paper_id: str) -> str:
    return "arxiv-" + paper_id.replace("__", "/")


def entity_id_candidates(paper_id: str) -> list[str]:
    primary = paper_id_to_entity_id(paper_id)
    candidates = [primary]
    if re.fullmatch(r"\d{7}", paper_id):
        candidates.append(f"arxiv-math/{paper_id}")
    return candidates


def safe_paper_id(entity_id: str) -> str:
    return entity_id.removeprefix("arxiv-").replace("/", "__")


def load_superpod_loader(path: Path | None = None):
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([DEFAULT_SUPERPOD, ROOT / "scripts" / "superpod-job.py"])
    for candidate in candidates:
        if candidate.exists():
            return load_module("superpod_job_golden", candidate)
    raise FileNotFoundError("could not find superpod-job.py loader")


def load_eprint_text(loader, eprint_dir: Path, paper_id: str, *, max_chars: int) -> tuple[str, dict]:
    last_meta = None
    for entity_id in entity_id_candidates(paper_id):
        text, meta = loader._load_eprint_text_for_entity(
            eprint_dir,
            entity_id,
            max_chars=max_chars,
            max_members=24,
        )
        if text:
            meta = {**meta, "entity_id": entity_id}
            return text, meta
        last_meta = meta
    raise RuntimeError(f"no eprint text for {paper_id}: {last_meta}")


def _subscript_bases(text: str) -> dict[str, Counter]:
    bases: dict[str, Counter] = defaultdict(Counter)
    base_re = r"(?:\\[A-Za-z]+\{[^{}\n]{1,40}\}|[A-Za-z](?:\\?cal)?|\\[A-Za-z]+)"
    intact = re.compile(rf"(?P<base>{base_re})_\{{(?P<sub>[^{{}}\n]{{1,50}})\}}")
    for match in intact.finditer(text):
        base = match.group("base")
        replacement = f"{base}_{{{match.group('sub')}}}"
        bases[base][replacement] += 1
    return bases


def repair_truncated_subscripts(text: str, *, min_attestations: int = 2) -> tuple[str, list[Repair]]:
    """Repair base_$ / base_ damage when one intact base_{...} dominates."""
    attestations = _subscript_bases(text)
    base_re = r"(?:\\[A-Za-z]+\{[^{}\n]{1,40}\}|[A-Za-z]|\\[A-Za-z]+)"
    damaged_re = re.compile(rf"(?P<base>{base_re})_(?=\$|\s)")
    repairs: dict[tuple[str, str], list[int]] = defaultdict(list)

    def repl(match: re.Match) -> str:
        base = match.group("base")
        counts = attestations.get(base)
        if not counts:
            return match.group(0)
        ranked = counts.most_common(2)
        replacement, count = ranked[0]
        if count < min_attestations:
            return match.group(0)
        if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
            return match.group(0)
        repairs[(match.group(0), replacement)].append(count)
        return replacement

    repaired = damaged_re.sub(repl, text)
    log = [
        Repair(damaged=damaged, replacement=replacement, count=len(counts), attestations=counts[0])
        for (damaged, replacement), counts in sorted(repairs.items())
    ]
    return repaired, log


def strip_tex_commands(fragment: str) -> str:
    out = re.sub(r"\\(?:emph|textbf|textit|textsc)\{([^{}]+)\}", r"\1", fragment)
    out = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", out)
    out = re.sub(r"[{}]", "", out)
    return re.sub(r"\s+", " ", out).strip()


def _clean_term(term: str) -> str:
    term = re.sub(r"\s+", " ", term).strip(" .,;:()[]")
    term = term.replace("~", " ")
    return term.strip()


def _definition_body(text: str, env: dict) -> tuple[int, int, str]:
    content = env.get("hx/content", {})
    start = int(content.get("position", 0))
    end = int(content.get("end", start))
    begin = re.search(r"\\begin\{[^}]+\}(?:\[[^\]]*\])?", text[start:end])
    body_start = start + (begin.end() if begin else 0)
    body_end = end
    end_match = re.search(r"\\end\{[^}]+\}\s*$", text[start:end], re.S)
    if end_match:
        body_end = start + end_match.start()
    return body_start, body_end, text[body_start:body_end]


def _add_definition(out: dict[str, Definition], term: str, position: int, source: str):
    term = _clean_term(term)
    if len(term) < 3:
        return
    if term.lower() in {"definition", "example", "remark", "theorem"}:
        return
    key = normalized_term_key(term)
    if key and key not in out:
        out[key] = Definition(term=term, position=position, source=source)


def mine_definitions(text: str, tex_envs: list[dict] | None = None) -> list[Definition]:
    """Extract in-paper definienda from real definition envs plus intro patterns."""
    definitions: dict[str, Definition] = {}
    tex_envs = tex_envs if tex_envs is not None else detect_tex_env_scopes("paper", text)
    for env in tex_envs:
        if env.get("hx/type") != "env-tex/definition":
            continue
        body_start, _body_end, body = _definition_body(text, env)
        for pat in (
            r"\\(?:emph|textbf|textit)\{([^{}]{3,120})\}",
            r"(?:An?|The)\s+([A-Za-z][A-Za-z0-9\\${}_^,\-\s]{2,100}?)\s+(?:is|are|consists|will be called|is called)\b",
        ):
            for match in re.finditer(pat, body):
                _add_definition(definitions, match.group(1), body_start + match.start(1), "definition-env")

    # The golden paragraph standard treats local introduction/notion sentences as
    # paper-local definitions for A-infinity style concepts, even when they live
    # in the introduction rather than a formal definition environment.
    intro_patterns = (
        r"(?P<term>\$[^$\n]{1,120}\$-[A-Za-z][A-Za-z-]*(?:ies|s)?)\s+were introduced\b",
        r"notion of\s+(?P<term>\$[^$\n]{1,120}\$-[A-Za-z][A-Za-z-]*(?:ies|s)?)",
        r"called\s+(?:the\s+)?(?P<term>[A-Za-z][A-Za-z0-9\\${}_^,\-\s]{3,100}?)(?:\.|,|\\cite|\s+by\b)",
    )
    for pat in intro_patterns:
        for match in re.finditer(pat, text):
            _add_definition(definitions, match.group("term"), match.start("term"), "intro-pattern")
    return sorted(definitions.values(), key=lambda d: (d.position, d.term.lower()))


def normalized_term_key(term: str) -> str:
    term = strip_tex_commands(term).lower()
    term = re.sub(r"\$([^$]+)\$", r"\1", term)
    term = term.replace("\\infty", "infty")
    term = re.sub(r"[^a-z0-9]+", " ", term).strip()
    words = term.split()
    if not words:
        return ""
    if words[-1].endswith("ies"):
        words[-1] = words[-1][:-3] + "y"
    elif words[-1].endswith("s") and len(words[-1]) > 4:
        words[-1] = words[-1][:-1]
    return " ".join(words)


def term_variants(term: str) -> list[str]:
    variants = {term}
    if term.endswith("y"):
        variants.add(term[:-1] + "ies")
    elif term.endswith("ies"):
        variants.add(term[:-3] + "y")
    elif term.endswith("s"):
        variants.add(term[:-1])
    else:
        variants.add(term + "s")
    for suffix in ("category", "algebra"):
        if term.endswith(suffix):
            variants.add(term + "s")
        if term.endswith(suffix + "s"):
            variants.add(term[: -1])
    return sorted(variants, key=len, reverse=True)


def _literal_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    spans = []
    pattern = re.compile(re.escape(needle), re.I)
    for match in pattern.finditer(text):
        before = text[match.start() - 1] if match.start() else " "
        after = text[match.end()] if match.end() < len(text) else " "
        if before.isalnum() or after.isalnum():
            continue
        spans.append(match.span())
    return spans


def definition_marks(text: str, definitions: list[Definition]) -> list[Mark]:
    marks: list[Mark] = []
    seen = set()
    for definition in definitions:
        for variant in term_variants(definition.term):
            for start, end in _literal_occurrences(text, variant):
                key = (start, end, normalized_term_key(definition.term))
                if key in seen:
                    continue
                seen.add(key)
                marks.append(Mark(
                    start=start,
                    end=end,
                    kind="defined",
                    title=f"defined in this paper @ {definition.position} ({definition.source})",
                    label=definition.term,
                ))
    return marks


def appositive_bind_marks(text: str) -> list[Mark]:
    endings = "|".join(APPOSITIVE_ENDINGS)
    pat = re.compile(
        rf"\b(?P<article>the|a|an)\s+(?P<type>(?:[A-Za-z][A-Za-z-]*\s+){{0,5}}(?:{endings}))\s+(?P<sym>\$[^$\n]{{1,120}}\$)",
        re.I,
    )
    marks = []
    for match in pat.finditer(text):
        type_phrase = re.sub(r"\s+", " ", match.group("type")).strip()
        if type_phrase.lower() in APPOSITIVE_ENDINGS:
            continue
        symbol = match.group("sym")
        marks.append(Mark(
            start=match.start(),
            end=match.end(),
            kind="bind",
            title=f"appositive bind: {symbol} : {type_phrase}",
            label=type_phrase,
        ))
    return marks


def hole_marks(text: str, definitions: list[Definition]) -> list[Mark]:
    defined_keys = {normalized_term_key(d.term) for d in definitions}
    endings = "|".join(CONCEPT_ENDINGS)
    word = r"[A-Za-z][A-Za-z-]*"
    pat = re.compile(rf"\b(?P<phrase>{word}(?:\s+{word}){{1,5}}\s+(?:{endings}))\b")
    marks = []
    seen = set()
    for match in pat.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group("phrase")).strip()
        start = match.start("phrase")
        words = phrase.split()
        while len(words) > 1 and words[0].lower() in HOLE_LEADING_TRIM:
            start += len(words[0]) + 1
            words = words[1:]
        phrase = " ".join(words)
        first = words[0] if words else ""
        if first in COMMON_HOLE_STARTS:
            continue
        if phrase.lower().startswith(("where ", "there ", "following ", "same ")):
            continue
        key = normalized_term_key(phrase)
        if key in defined_keys or key in seen:
            continue
        # Avoid marking fragments of obvious prose boilerplate.
        if len(phrase) < 12 or phrase.count(" ") < 1:
            continue
        seen.add(key)
        marks.append(Mark(
            start=start,
            end=match.end("phrase"),
            kind="hole",
            title="no in-paper definition — needs canon link (hole)",
            label=phrase,
        ))
    return marks


def select_non_overlapping(marks: list[Mark]) -> list[Mark]:
    priority = {"bind": 0, "defined": 1, "hole": 2}
    marks = sorted(marks, key=lambda m: (priority[m.kind], -(m.end - m.start), m.start))
    accepted: list[Mark] = []
    occupied: list[tuple[int, int]] = []
    for mark in marks:
        if mark.end <= mark.start:
            continue
        if any(not (mark.end <= s or mark.start >= e) for s, e in occupied):
            continue
        accepted.append(mark)
        occupied.append((mark.start, mark.end))
    return sorted(accepted, key=lambda m: m.start)


def audit_concept_term_coverage(text: str, expected_terms: list[str]) -> dict:
    """C-TERM-COVERAGE audit over an explicit row sample.

    `expected_terms` is the independent numerator/denominator: terms a reviewer
    says are named mathematical prose concepts. We then ask whether this script's
    concept marks notice them, and whether produced concept marks correspond to
    one of the expected terms. This is intentionally independent of the rendered
    HTML and can be sampled at sentence/row granularity in tests or audits.
    """
    definitions = mine_definitions(text)
    marks = select_non_overlapping(
        appositive_bind_marks(text)
        + definition_marks(text, definitions)
        + hole_marks(text, definitions)
    )
    concept_marks = [m for m in marks if m.kind in {"defined", "hole", "bind"}]
    expected = {normalized_term_key(t): t for t in expected_terms if normalized_term_key(t)}
    noticed = set()
    false_positive_marks = []
    for mark in concept_marks:
        label_key = normalized_term_key(mark.label)
        span_key = normalized_term_key(text[mark.start:mark.end])
        if label_key in expected:
            noticed.add(label_key)
        elif span_key in expected:
            noticed.add(span_key)
        else:
            false_positive_marks.append(mark)
    missed = sorted(v for k, v in expected.items() if k not in noticed)
    precision_den = len(concept_marks)
    recall_den = len(expected)
    return {
        "sample_rows": recall_den,
        "expected_terms": recall_den,
        "concept_marks": precision_den,
        "true_positive_marks": precision_den - len(false_positive_marks),
        "false_positive_marks": len(false_positive_marks),
        "missed_terms": missed,
        "precision": (precision_den - len(false_positive_marks)) / precision_den
        if precision_den else 1.0,
        "recall": len(noticed) / recall_den if recall_den else 1.0,
    }


def render_marked_text(text: str, marks: list[Mark]) -> str:
    classes = {
        "defined": "mark-defined",
        "hole": "mark-hole",
        "bind": "mark-bind",
    }
    out = []
    cursor = 0
    for mark in marks:
        out.append(html.escape(text[cursor:mark.start]))
        out.append(
            f'<span class="{classes[mark.kind]}" title="{html.escape(mark.title)}">'
            f"{html.escape(text[mark.start:mark.end])}</span>"
        )
        cursor = mark.end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def load_fresh_scopes(scope_dir: Path, paper_id: str) -> dict:
    for sid in [paper_id, *[safe_paper_id(eid) for eid in entity_id_candidates(paper_id)]]:
        path = scope_dir / f"{sid}.json"
        if path.exists():
            return json.loads(path.read_text())
    return {"count": 0, "type_counts": {}, "scopes": []}


def title_from_tex(text: str) -> str:
    match = re.search(r"\\title(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if not match:
        return ""
    return strip_tex_commands(match.group(1))


def render_title_card(entity_id: str, text: str, fresh_scopes: list[dict], tex_envs: list[dict]) -> str:
    try:
        signature = TITLE_CARD.concept_signature(entity_id, text, top=10)
    except Exception:
        signature = []
    try:
        fingerprints = TITLE_CARD.theorem_fingerprints(entity_id, text, fresh_scopes, tex_envs)[:8]
    except Exception:
        fingerprints = []
    try:
        plot = TITLE_CARD.plot_summary(tex_envs)
    except Exception:
        plot = ""
    theorem_counts = Counter(
        env.get("hx/type", "").removeprefix("env-tex/")
        for env in tex_envs
        if env.get("hx/type", "").removeprefix("env-tex/") in TITLE_CARD.THEOREM_KINDS
    )
    signature_html = "".join(
        f"<li>{html.escape(str(canon))} <b>{count}</b></li>" for canon, count in signature
    ) or "<li>no title-card terms found</li>"
    theorem_html = "".join(
        f"<li>{html.escape(kind)} <b>{count}</b></li>" for kind, count in sorted(theorem_counts.items())
    ) or "<li>no theorem-like envs found</li>"
    fp_html = "".join(
        "<li>"
        f"<b>{html.escape(fp['kind'])}</b> @{fp['position']}: "
        f"{html.escape(fp.get('statement_head', '')[:140])}"
        "</li>"
        for fp in fingerprints
    ) or "<li>no theorem fingerprints</li>"
    return f"""
    <section class="title-card">
      <div><h2>Concept Signature</h2><ol>{signature_html}</ol></div>
      <div><h2>Theorem Census</h2><ol>{theorem_html}</ol></div>
      <div><h2>Fingerprints</h2><ol>{fp_html}</ol></div>
      <div><h2>Environment Flow</h2><p>{html.escape(plot[:1200])}</p></div>
    </section>
    """


CSS = """
body{font:16px/1.65 Georgia,serif;margin:0;color:#1d1a16;background:#fffdf8}
main{max-width:1180px;margin:0 auto;padding:30px 28px 70px}
a{color:#174ea6}
.top{border-bottom:1px solid #e8dfcf;background:#fff8e8;padding:24px 28px;margin:0 -28px 24px}
.meta{color:#5f5548;font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px}
.counts,.repair-log,.title-card{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0}
.card,.counts div,.repair-log div,.title-card div{border:1px solid #e5dccd;background:#fffdf8;border-radius:6px;padding:10px 12px}
.count{font:700 24px/1.1 ui-sans-serif,system-ui,sans-serif;display:block}
.legend span,.mark-defined,.mark-hole,.mark-bind{border-radius:2px;padding:0 1px}
.mark-defined{background:#d3f3df;border-bottom:2px solid #0f766e}
.mark-hole{background:#fdf3d7;border-bottom:2px dashed #9a7b1a}
.mark-bind{background:#dde7fb;border-bottom:2px solid #2a4d9a}
pre.paper{white-space:pre-wrap;font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:white;border:1px solid #e5dccd;border-radius:6px;padding:18px;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
th,td{border-bottom:1px solid #eadfce;padding:8px;text-align:left;vertical-align:top}
th{background:#fff4d8}
"""


def render_paper_page(
    *,
    paper_id: str,
    stratum: str,
    entity_id: str,
    text: str,
    eprint_meta: dict,
    fresh_scope_row: dict,
    tex_envs: list[dict],
    repairs: list[Repair],
    definitions: list[Definition],
    marks: list[Mark],
) -> str:
    counts = Counter(mark.kind for mark in marks)
    title = title_from_tex(text) or paper_id
    repair_html = "".join(
        "<div>"
        f"<b>{html.escape(r.damaged)}</b> → <b>{html.escape(r.replacement)}</b><br>"
        f"{r.count} repaired sites, {r.attestations} attestations"
        "</div>"
        for r in repairs
    ) or "<div>No evidence-backed source repairs.</div>"
    defs_html = "".join(
        f"<li id=\"def-{i}\">{html.escape(d.term)} <span class=\"meta\">@{d.position}, {html.escape(d.source)}</span></li>"
        for i, d in enumerate(definitions[:80], 1)
    ) or "<li>No in-paper definitions mined.</li>"
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(paper_id)} golden anatomy</title>
<style>{CSS}</style>
<main>
  <section class="top">
    <p class="meta"><a href="index.html">golden index</a> / {html.escape(stratum)}</p>
    <h1>{html.escape(paper_id)} — {html.escape(title)}</h1>
    <p class="meta">entity {html.escape(entity_id)} · eprint {html.escape(str(eprint_meta.get('status')))} · chars {len(text):,}</p>
    <p class="legend"><b>Legend:</b>
      <span class="mark-defined">defined in-paper</span> ·
      <span class="mark-hole">external concept — needs canon link (HOLE)</span> ·
      <span class="mark-bind">appositive bind</span>
    </p>
    <div class="counts">
      <div><span class="count">{counts['defined']}</span>defined marks</div>
      <div><span class="count">{counts['hole']}</span>amber holes</div>
      <div><span class="count">{counts['bind']}</span>appositive binds</div>
      <div><span class="count">{fresh_scope_row.get('count', 0)}</span>fresh scopes</div>
    </div>
  </section>
  {render_title_card(entity_id, text, fresh_scope_row.get('scopes') or [], tex_envs)}
  <section><h2>Repair Log</h2><div class="repair-log">{repair_html}</div></section>
  <section><h2>In-Paper Definitions</h2><ol>{defs_html}</ol></section>
  <section><h2>Marked Source</h2><pre class="paper">{render_marked_text(text, marks)}</pre></section>
</main>
"""


def render_index(rows: list[dict], sample: dict) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(row['file'])}\">{html.escape(row['paper_id'])}</a></td>"
            f"<td>{html.escape(row['stratum'])}</td>"
            f"<td>{row['defined']}</td><td>{row['holes']}</td><td>{row['binds']}</td>"
            f"<td>{row['repairs']}</td><td>{row['fresh_scopes']}</td>"
            f"<td>{html.escape(row.get('title') or '')}</td>"
            "</tr>"
        )
    totals = Counter()
    for row in rows:
        totals["defined"] += row["defined"]
        totals["holes"] += row["holes"]
        totals["binds"] += row["binds"]
        totals["repairs"] += row["repairs"]
    strata = Counter(row["stratum"] for row in rows)
    strata_html = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in sorted(strata.items()))
    return f"""<!doctype html>
<meta charset="utf-8">
<title>GOLDEN-30 math.CT anatomy proofread</title>
<style>{CSS}</style>
<main>
  <section class="top">
    <h1>GOLDEN-30 math.CT anatomy proofread</h1>
    <p class="meta">Frozen {html.escape(str(sample.get('frozen', '?')))} · rendered {len(rows)} papers</p>
    <p class="legend"><b>Legend:</b>
      <span class="mark-defined">defined in-paper</span> ·
      <span class="mark-hole">external concept — needs canon link (HOLE)</span> ·
      <span class="mark-bind">appositive bind</span>
    </p>
    <div class="counts">
      <div><span class="count">{totals['defined']}</span>defined marks</div>
      <div><span class="count">{totals['holes']}</span>amber holes</div>
      <div><span class="count">{totals['binds']}</span>appositive binds</div>
      <div><span class="count">{totals['repairs']}</span>source repairs</div>
    </div>
    <ul>{strata_html}</ul>
  </section>
  <table>
    <thead><tr><th>Paper</th><th>Stratum</th><th>Defined</th><th>Holes</th><th>Binds</th><th>Repairs</th><th>Fresh scopes</th><th>Title</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</main>
"""


def stratum_lookup(sample: dict) -> dict[str, str]:
    out = {}
    for stratum, papers in (sample.get("strata") or {}).items():
        for paper in papers:
            out[paper] = stratum
    return out


def analyze_paper(loader, args, sample: dict, paper_id: str) -> dict:
    stratum = stratum_lookup(sample).get(paper_id, "?")
    raw_text, eprint_meta = load_eprint_text(loader, args.eprints, paper_id, max_chars=args.max_chars)
    entity_id = eprint_meta.get("entity_id") or paper_id_to_entity_id(paper_id)
    text, repairs = repair_truncated_subscripts(raw_text)
    tex_envs = detect_tex_env_scopes(entity_id, text)
    definitions = mine_definitions(text, tex_envs)
    fresh_scope_row = load_fresh_scopes(args.scope_dir, paper_id)
    marks = select_non_overlapping(
        appositive_bind_marks(text)
        + definition_marks(text, definitions)
        + hole_marks(text, definitions)
    )
    out_file = f"{paper_id}.html"
    html_text = render_paper_page(
        paper_id=paper_id,
        stratum=stratum,
        entity_id=entity_id,
        text=text,
        eprint_meta=eprint_meta,
        fresh_scope_row=fresh_scope_row,
        tex_envs=tex_envs,
        repairs=repairs,
        definitions=definitions,
        marks=marks,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / out_file).write_text(html_text)
    counts = Counter(mark.kind for mark in marks)
    return {
        "paper_id": paper_id,
        "entity_id": entity_id,
        "stratum": stratum,
        "file": out_file,
        "title": title_from_tex(text),
        "defined": counts["defined"],
        "holes": counts["hole"],
        "binds": counts["bind"],
        "repairs": sum(r.count for r in repairs),
        "repair_kinds": len(repairs),
        "fresh_scopes": fresh_scope_row.get("count", 0),
        "definitions": len(definitions),
        "chars": len(text),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--eprints", type=Path, default=DEFAULT_EPRINTS)
    parser.add_argument("--scope-dir", type=Path, default=DEFAULT_SCOPES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--superpod-job", type=Path, default=DEFAULT_SUPERPOD)
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument("--max-chars", type=int, default=1_200_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    sample = json.loads(args.sample.read_text())
    loader = load_superpod_loader(args.superpod_job)
    paper_ids = args.paper_ids or sample["papers"]
    rows = [analyze_paper(loader, args, sample, paper_id) for paper_id in paper_ids]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(render_index(rows, sample))
    (args.out_dir / "manifest.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps({"rendered": len(rows), "out_dir": str(args.out_dir), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
