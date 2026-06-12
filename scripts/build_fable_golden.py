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


CLS = {"defined": "background:#d3f3df;border-bottom:2px solid #0f766e",
       "hole": "background:#fdf3d7;border-bottom:2px dashed #9a7b1a",
       "bind": "background:#dde7fb;border-bottom:2px solid #2a4d9a",
       "envtex": "background:#e4ecf4;border-bottom:2px solid #1d3a4d"}

FAMILY_STYLE = {
    "bind": "background:#dde7fb;border-bottom:2px solid #2a4d9a",
    "constrain": "background:#efe3f7;border-bottom:2px solid #7a3ba8",
    "quant": "background:#e3eef7;border-bottom:2px solid #1a6a9a",
    "assume": "background:#e3f7ef;border-bottom:2px solid #1a9a6a",
    "env": "background:#f4f0e4;outline:1px solid #c9bd9c",
    "comment": "color:#888;background:#f0f0f0",
}
SUBTERM_COLORS = {"math/typed-arrow": "#008080", "math/membership": "#7851a9",
                  "math/subscript": "#8b008b", "math/superscript": "#8b008b",
                  "math/constructor-declaration": "#cc5500",
                  "math/category-symbol": "#2e8b57", "math/category-symbol-call": "#2e8b57",
                  "math/tensor": "#8a2be2", "math/isomorphism": "#b8860b",
                  "math/equality": "#004225", "math/typed-binding": "#0f766e",
                  "math/named-functor": "#cc5500", "math/group": "#704214",
                  "math/macro-call": "#999", "math/macro-arg": "#bbb"}


def load_base_scopes(pid):
    """The existing system's voice: the fresh extraction (stage-5 parity)."""
    f = Path("/home/joe/code/storage/mark2/ct-fresh-scopes") / (pid.replace("/", "_") + ".json")
    if not f.exists():
        return []
    return json.loads(f.read_text()).get("scopes", [])


def base_marks_and_envelopes(base):
    envelopes, marks = [], []
    for sc in base:
        t = str(sc.get("hx/type", ""))
        c = sc.get("hx/content") or {}
        pos, end = c.get("position"), c.get("end")
        if pos is None or end is None or end <= pos:
            continue
        fam = t.split("/")[0]
        if t == "math/envelope":
            envelopes.append({"start": pos, "end": end, "subterms": []})
        elif fam == "math":
            marks.append((pos, min(end, pos + 80), "SUB:" + t, t))
        elif fam in FAMILY_STYLE:
            ends = sc.get("hx/ends") or []
            sym = next((e.get("latex") or e.get("text") for e in ends
                        if e.get("role") == "symbol"), None)
            typ = next((e.get("text") or e.get("latex") for e in ends
                        if e.get("role") == "type"), None)
            tip = t + (f": ${sym}$" if sym else "") + (f" : {typ[:50]}" if typ else "")
            marks.append((pos, min(end, pos + 200), "FAM:" + fam, tip))
    envelopes.sort(key=lambda e: e["start"])
    import bisect
    starts = [e["start"] for e in envelopes]
    flat = []
    for pos, end, kind, tip in marks:
        if kind.startswith("SUB:"):
            i = bisect.bisect_right(starts, pos) - 1
            if i >= 0 and end <= envelopes[i]["end"]:
                envelopes[i]["subterms"].append((pos, end, kind[4:]))
                continue
        flat.append((pos, end, kind, tip))
    return envelopes, flat


def render_page(text, envelopes, flat_marks, golden_marks):
    """Base layer (the system's voice) + golden ADDITIONS (dashed top border)."""
    events = {}
    for pos, end, kind, tip in flat_marks:
        events.setdefault(pos, []).append((end, "base", kind, tip))
    for pos, end, kind, tip in golden_marks:
        events.setdefault(pos, []).append((end, "golden", kind, tip))
    for e in envelopes:
        events.setdefault(e["start"], []).append((e["end"], "expr", "mexpr", e))
    out, pos, last = [], 0, 0
    n = Counter()
    for start in sorted(events):
        if start < last:
            continue
        out.append(esc(text[pos:start]))
        end, layer, kind, payload = max(events[start], key=lambda x: x[0])
        if layer == "expr":
            e = payload
            seg, p2, s_last = [], e["start"], e["start"]
            for sp, se, st in sorted(set(e["subterms"])):
                if sp < s_last:
                    continue
                seg.append(esc(text[p2:sp]))
                seg.append(f'<span style="color:{SUBTERM_COLORS.get(st, chr(35)+"555")};font-weight:600" title="{esc(st)}">{esc(text[sp:se])}</span>')
                p2 = se; s_last = se
            seg.append(esc(text[p2:e["end"]]))
            out.append(f'<span style="background:#eef0f7;border-radius:3px" title="math expression ({len(e[chr(34)+chr(34) if False else "subterms"])} subterms)">{"".join(seg)}</span>')
            n["mexpr"] += 1
        elif layer == "base":
            fam = kind[4:] if kind.startswith("FAM:") else kind
            style = FAMILY_STYLE.get(fam, "background:#eee")
            out.append(f'<span style="{style}" title="{esc(payload)}">{esc(text[start:end])}</span>')
            n["base:" + fam] += 1
        else:
            style = CLS[kind] + ";border-top:2px dashed #444"
            out.append(f'<span style="{style}" title="GOLDEN ADDITION — {esc(payload)}">{esc(text[start:end])}</span>')
            n["golden:" + kind] += 1
        pos = end; last = end
    out.append(esc(text[pos:]))
    return "".join(out), n


def render(pid):
    eid = "arxiv-" + pid.replace("_", "/")
    text, meta = sj._load_eprint_text_for_entity(EPRINTS, eid)
    if not text:
        print(f"{pid}: no text ({meta.get('status')})"); return
    text, rep_log = repairs(text)
    base = load_base_scopes(pid)
    envelopes, flat = base_marks_and_envelopes(base)
    envs = texenv.detect_tex_env_scopes(eid, text)
    defs = mine_definienda(text, envs)
    holes = concept_holes(text, defs)
    golden = []
    for term, dpos in defs.items():
        for m in re.finditer(re.escape(term), text, re.I):
            golden.append((m.start(), m.end(), "defined",
                           f"defined in this paper @ {dpos}"))
    for term in holes:
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", text, re.I):
            golden.append((m.start(), m.end(), "hole",
                           f"no in-paper definition — needs canon link ({holes[term]} occ)"))
    for env in envs:
        c = env["hx/content"]
        kind = env["hx/type"].removeprefix("env-tex/")
        golden.append((c["position"], min(c["position"] + 60, c["end"]), "envtex",
                       f"REAL TeX environment: {kind}"))
    body, n = render_page(text, envelopes, flat, golden)
    census = Counter(e["hx/type"].removeprefix("env-tex/") for e in envs)
    base_counts = Counter(str(sc.get("hx/type", "?")).split("/")[0] for sc in base)
    sig = ", ".join(sorted(defs)[:8]) or "—"
    top_holes = ", ".join(t for t, _ in Counter(holes).most_common(6)) or "—"
    rep_html = "<br>".join(esc(r) for r in rep_log) or "no repairs needed (explicit empty log)"
    page = OUT_DIR / f"fable-{pid}.html"
    base_str = " · ".join(f"{k}:{v}" for k, v in base_counts.most_common(8))
    n_str = " · ".join(f"{k}:{v}" for k, v in n.most_common(12))
    page.write_text(f"""<!doctype html><meta charset=utf-8><title>FABLE GOLDEN — {esc(pid)}</title>
<style>body{{font:16px/1.7 Georgia,serif;max-width:1000px;margin:30px auto;color:#1d1a16;background:#fffdf8;padding:0 18px}}
pre{{white-space:pre-wrap;font:inherit}}.card{{border:2px solid #0f766e;border-radius:10px;padding:14px 18px;background:#f6fffd;font-family:system-ui;font-size:14px}}</style>
<div class="card"><h2 style="margin:0 0 6px">FABLE GOLDEN — {esc(pid)} (base = the system's existing voice; dashes on top = golden ADDITIONS)</h2>
<b>Base layer (fresh extraction):</b> {esc(base_str)}<br>
<b>Defined in-paper ({len(defs)}):</b> {esc(sig)}<br>
<b>Census (REAL TeX envs):</b> {esc(", ".join(f"{k}×{v}" for k, v in census.most_common(8)))}<br>
<b>Top holes:</b> {esc(top_holes)}<br>
<b>Repair log:</b> {rep_html}<br>
<b>Rendered marks:</b> {esc(n_str)}</div>
<p><span style="background:#dde7fb;border-bottom:2px solid #2a4d9a">bind</span> ·
<span style="background:#efe3f7;border-bottom:2px solid #7a3ba8">constrain</span> ·
<span style="background:#e3eef7;border-bottom:2px solid #1a6a9a">quant</span> ·
<span style="background:#eef0f7">math expression w/ subterms</span> ·
<span style="background:#d3f3df;border-bottom:2px solid #0f766e;border-top:2px dashed #444">+defined</span> ·
<span style="background:#fdf3d7;border-bottom:2px dashed #9a7b1a;border-top:2px dashed #444">+hole</span> ·
<span style="background:#e4ecf4;border-bottom:2px solid #1d3a4d;border-top:2px dashed #444">+real-env</span> ·
<a href="https://arxiv.org/abs/{esc(pid.replace('_','/'))}">arXiv</a></p>
<pre>{body}</pre>""")
    print(f"{pid}: base={len(base)} scopes ({base_str}); golden +{len(golden)} -> {page.name}")


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        render(pid)
