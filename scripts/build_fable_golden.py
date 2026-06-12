#!/usr/bin/env python3
"""build_fable_golden.py — reference-grade golden pages (the Fable workup).

The operator-approved golden-paragraph.html standard at full-paper scale,
embodying the checklist fixes the codex build lacks:
  T1 repair log (explicit, even when empty; evidence-based repairs only)
  T2 definienda with SPAN DISCIPLINE (the term, not its clause)
  T3 every occurrence of each definiendum marked, linked to its def
  T4 concept-shaped holes ONLY (head kind-word or TitleCase; deduped;
     all occurrences of each distinct held term marked)
  T5 binds from the tempered detectors (inline + display typed arrows)
  T7 title card on every page (signature from mined definienda + holes)
Usage: build_fable_golden.py <paper-id> [...]   (e.g. 0809.2517)
"""
import html
import importlib
import importlib.util
import importlib.machinery
import json
import re
import sys
from collections import Counter
from pathlib import Path

FUTON6 = Path("/home/joe/code/futon6")
LOADER = Path("/tmp/futon6-sbs/scripts/superpod-job.py")
EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
OUT_DIR = FUTON6 / "data" / "showcases" / "ct-anatomy" / "golden"

sys.path.insert(0, str(FUTON6 / "scripts"))
spec = importlib.util.spec_from_file_location("sj", str(LOADER))
sj = importlib.util.module_from_spec(spec); spec.loader.exec_module(sj)
nw = importlib.import_module("nlab-wiring")
texenv = importlib.machinery.SourceFileLoader(
    "texenv", str(FUTON6 / "src/futon6/tex_env_scopes.py")).load_module()

esc = html.escape
KIND_WORDS = ("category", "categories", "functor", "functors", "algebra",
              "algebras", "group", "groups", "manifold", "manifolds", "module",
              "modules", "sheaf", "sheaves", "motive", "motives", "complex",
              "complexes", "space", "spaces", "object", "morphism", "monad",
              "topology", "correspondence", "localization", "square", "product",
              "limit", "colimit", "adjunction", "equivalence", "nerve")
STOP_HEADS = ("the", "a", "an", "this", "that", "these", "those", "it", "we",
              "such", "paper", "section", "proof", "result", "case", "work")


def repairs(text):
    log = []
    for m in re.finditer(r"\\mathcal\{([A-Z])\}_\$", text):
        full = f"\\mathcal{{{m.group(1)}}}_{{\\infty}}"
        n_attest = text.count(full)
        if n_attest >= 3:
            broken = f"\\mathcal{{{m.group(1)}}}_$"
            text = text.replace(broken, full + "$")
            log.append(f"{broken} -> {full}$ ({n_attest} attestations)")
            break
    return text, log


def mine_definienda(text, envs):
    """T2 with span discipline: emph/textbf content only, or tight prose head."""
    defs = {}
    for env in envs:
        if not env["hx/type"].startswith(("env-tex/definition",)):
            continue
        c = env["hx/content"]
        body = text[c["position"]:c["end"]][:500]
        cands = re.findall(r"\\(?:emph|textit|textbf|em)\{([^}]{3,50})\}", body)
        m = re.search(r"(?:An?|The)\s+([a-z][a-z-]+(?: [a-z-]+){0,3}?\s(?:%s))\b"
                      % "|".join(KIND_WORDS), body)
        if m:
            cands.append(m.group(1))
        for cand in cands:
            term = re.sub(r"\s+", " ", re.sub(r"[\\${}]", "", cand)).strip().lower()
            term = re.sub(r"\b(if|such that|when|where)\b.*$", "", term).strip(" ,.;:")
            if 3 <= len(term) <= 50 and not term.split()[0] in STOP_HEADS:
                defs.setdefault(term, c["position"])
    # INLINE definitions (T2): "is called an {\em X}", "we say ... {\em X}",
    # "define ... \emph{X}" — proof-papers define without definition envs
    # (0809.2517 defines Azumaya algebra this way; 0 definition envs).
    for m in re.finditer(
            r"\\(?:emph\{|em\s+|textit\{|textbf\{)\s*([^}$\n]{3,50})\}?",
            text):
        head_ok = re.search(r"(?:%s)\s*$" % "|".join(KIND_WORDS),
                            m.group(1).strip().lower())
        ctx = text[m.end():m.end() + 60]
        ctx_ok = re.match(r"[^.]{0,40}(?:is defined|if\b|denoted|is an?\b|we mean)", ctx)
        if not (head_ok or ctx_ok):
            continue
        term = re.sub(r"\s+", " ", re.sub(r"[\\${}]", "", m.group(1))).strip().lower()
        term = re.sub(r"\b(if|such that|when|where)\b.*$", "", term).strip(" ,.;:")
        if 3 <= len(term) <= 50 and term.split()[0] not in STOP_HEADS:
            defs.setdefault(term, m.start())
    return defs


def concept_holes(text, defs):
    """T4: concept-shaped only — kind-word head or TitleCase; deduped."""
    holes = Counter()
    for m in re.finditer(
            r"\b((?:[A-Z][A-Za-z'-]+\s)?(?:[a-z-]+\s){0,2}(?:%s))\b" % "|".join(KIND_WORDS),
            text):
        term = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        words = term.split()
        if (len(words) >= 2 and words[0] not in STOP_HEADS
                and not any(w in STOP_HEADS for w in words[:-1])
                and term not in defs):
            holes[term] += 1
    return {t: n for t, n in holes.items() if n >= 2}


def collect_marks(text, defs, holes, scopes):
    marks = []
    for term, dpos in defs.items():
        for m in re.finditer(re.escape(term), text, re.I):
            marks.append((m.start(), m.end(), "defined",
                          f"defined in this paper @ {dpos}"))
    for term in holes:
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", text, re.I):
            marks.append((m.start(), m.end(), "hole",
                          f"no in-paper definition — needs canon link ({holes[term]} occurrences)"))
    for s in scopes:
        if str(s.get("hx/type")) != "bind/typed":
            continue
        ends = s.get("hx/ends") or []
        sym = next((e.get("latex") for e in ends if e.get("role") == "symbol"), None)
        typ = next((e.get("text") or e.get("latex") for e in ends if e.get("role") == "type"), "")
        c = s.get("hx/content") or {}
        if sym and c.get("position") is not None:
            marks.append((c["position"], min(c.get("end", c["position"] + 40),
                                             c["position"] + 160), "bind",
                          f"bind: ${sym}$ : {(typ or '')[:60]}"))
    marks.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    return marks


SUBTERM_COLORS = {"math/typed-arrow": "#008080", "math/membership": "#7851a9",
                  "math/subscript": "#8b008b", "math/superscript": "#8b008b",
                  "math/constructor-declaration": "#cc5500",
                  "math/grounded-symbol": "#b22222", "math/group": "#704214",
                  "math/macro-call": "#666"}


def math_expression_layer(eid, text):
    """Golden rule 1 (Joe, 2026-06-12): anything between dollar signs is a
    math-expression scope, and it should have subterms. Envelopes from the
    $...$ blocks; typed subterms nested inside at absolute positions."""
    exprs = []
    for m in re.finditer(r"\$([^$]{1,300})\$", text):
        exprs.append({"start": m.start(), "end": m.end(), "subterms": []})
    sub_all = []
    for det in ("detect_math_scopes", "detect_math_scopes_ast"):
        fn = getattr(nw, det, None)
        if fn:
            sub_all.extend(fn(eid, text) or [])
    by_start = sorted(exprs, key=lambda e: e["start"])
    import bisect
    starts = [e["start"] for e in by_start]
    for s_ in sub_all:
        c = s_.get("hx/content") or {}
        pos, end = c.get("position"), c.get("end")
        t = str(s_.get("hx/type", ""))
        if pos is None or end is None or t == "math/envelope":
            continue
        i = bisect.bisect_right(starts, pos) - 1
        if i >= 0 and end <= by_start[i]["end"]:
            by_start[i]["subterms"].append((pos, min(end, pos + 80), t))
    return by_start


CLS = {"defined": "background:#d3f3df;border-bottom:2px solid #0f766e",
       "hole": "background:#fdf3d7;border-bottom:2px dashed #9a7b1a",
       "bind": "background:#dde7fb;border-bottom:2px solid #2a4d9a"}


def render(pid):
    eid = "arxiv-" + pid.replace("_", "/")
    text, meta = sj._load_eprint_text_for_entity(EPRINTS, eid)
    if not text:
        print(f"{pid}: no text ({meta.get('status')})"); return
    text, rep_log = repairs(text)
    envs = texenv.detect_tex_env_scopes(eid, text)
    scopes = nw.detect_scopes(eid, text) or []
    defs = mine_definienda(text, envs)
    holes = concept_holes(text, defs)
    marks = collect_marks(text, defs, holes, scopes)
    exprs = math_expression_layer(eid, text)
    expr_events = {e["start"]: e for e in exprs}
    out, pos, last = [], 0, 0
    n_used = Counter()

    def emit_expr(e):
        seg, p = [], e["start"]
        subs = sorted(set(e["subterms"]))
        s_last = e["start"]
        for sp, se, st in subs:
            if sp < s_last:
                continue
            seg.append(esc(text[p:sp]))
            color = SUBTERM_COLORS.get(st, "#555")
            seg.append(f'<span style="color:{color};font-weight:600" '
                       f'title="{esc(st)}">{esc(text[sp:se])}</span>')
            p = se; s_last = se
        seg.append(esc(text[p:e["end"]]))
        n_used["mexpr"] += 1
        return (f'<span style="background:#eef0f7;border-radius:3px" '
                f'title="math expression ({len(subs)} subterms)">{"".join(seg)}</span>')

    events = sorted(set([m[0] for m in marks] + list(expr_events)))
    midx = {m[0]: m for m in marks}
    for start in events:
        if start < last:
            continue
        out.append(esc(text[pos:start]))
        if start in midx and (start not in expr_events or midx[start][1] >= expr_events[start]["end"]):
            s_, e_, kind, tip = midx[start]
            out.append(f'<span style="{CLS[kind]}" title="{esc(tip)}">{esc(text[s_:e_])}</span>')
            pos = e_; last = e_; n_used[kind] += 1
        elif start in expr_events:
            e = expr_events[start]
            out.append(emit_expr(e))
            pos = e["end"]; last = e["end"]
    out.append(esc(text[pos:]))
    census = Counter(e["hx/type"].removeprefix("env-tex/") for e in envs
                     if e["hx/type"].removeprefix("env-tex/") in
                     ("theorem", "proposition", "lemma", "corollary", "definition"))
    sig = ", ".join(sorted(defs)[:8]) or "—"
    top_holes = ", ".join(t for t, _ in Counter(holes).most_common(6)) or "—"
    rep_html = "<br>".join(esc(r) for r in rep_log) or "no repairs needed (log explicit per T1)"
    page = OUT_DIR / f"fable-{pid}.html"
    page.write_text(f"""<!doctype html><meta charset=utf-8><title>FABLE GOLDEN — {esc(pid)}</title>
<style>body{{font:16px/1.7 Georgia,serif;max-width:980px;margin:30px auto;color:#1d1a16;background:#fffdf8;padding:0 18px}}
pre{{white-space:pre-wrap;font:inherit}}.card{{border:2px solid #0f766e;border-radius:10px;padding:14px 18px;background:#f6fffd;font-family:system-ui;font-size:14px}}</style>
<div class="card"><h2 style="margin:0 0 6px">TITLE CARD — fable-{esc(pid)}</h2>
<b>Defined in-paper ({len(defs)}):</b> {esc(sig)}<br>
<b>Census:</b> {esc(", ".join(f"{k}×{v}" for k, v in census.most_common()))}<br>
<b>Top external concepts (holes, deduped, ≥2 occurrences):</b> {esc(top_holes)}<br>
<b>Repair log:</b> {rep_html}<br>
<b>Marks:</b> {n_used['defined']} defined · {n_used['hole']} holes ({len(holes)} distinct) · {n_used['bind']} binds · {n_used['mexpr']} math expressions</div>
<p><span style="{CLS['defined']}">defined in-paper</span> ·
<span style="{CLS['hole']}">needs canon link</span> ·
<span style="{CLS['bind']}">typed bind</span> ·
<a href="https://arxiv.org/abs/{esc(pid.replace('_','/'))}">arXiv</a></p>
<pre>{''.join(out)}</pre>""")
    print(f"{pid}: defs={len(defs)} holes={len(holes)} distinct "
          f"(marks: {n_used['defined']}g/{n_used['hole']}a/{n_used['bind']}b) -> {page.name}")


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        render(pid)
