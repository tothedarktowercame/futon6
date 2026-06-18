#!/usr/bin/env python3
"""Build the SUPERPOD MARK-3 MOCKUP for 1005.2653 — a vision artifact showing
what the imagined LLM-scale phase (over all arXiv) would add on top of the
current CPU pipeline. HONEST: layers already shipped by the real pipeline are
marked ✓ real; aspirational layers are marked ⚗ imagined (hand-authored from a
close reading of the paper, NOT produced by any current detector).

    build_superpod_mockup.py   # -> data/showcases/ct-anatomy/dp-demo/1005.2653-superpod-mockup.html
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_anatomy_html as R
import dp_paper_view as dpv

OUT = R.DEFAULT_OUT / "1005.2653-superpod-mockup.html"

# ⚗ Phase-A ground-to-type: gloss per bound-variable symbol. Applied to the REAL
# marks to synthesize the imagined ~98%-grounded body for the side-by-side.
GROUND_MAP = {
    "a": "object of \U0001d49c", "b": "object of \U0001d49c",
    "c": "object of \U0001d49c", "d": "object of \U0001d49c",
    "x": "object of ℋ", "y": "object of ℋ",
    "u": "object of ℋ", "v": "object of ℋ",
    "k": "the ground field", "S": "antipode ℋᵒᵖ→ℋ",
    "K": "Fourier functor K̂ (=[h,1]) / right adjoint",
    "h": "Kleisli / algebroid functor", "p": "promultiplication",
    "j": "promonoidal unit", "f": "module in [ℋ,Vect_k]", "r": "index variable",
}


def _imagine(text, marks):
    """Apply imagined mark-3 ground-to-type: every ungrounded `symbol` becomes
    `symbol-grounded` with its inferred type. Honest: this is the MOCKUP's
    synthetic enrichment, not detector output."""
    out = []
    for m in marks:
        m = dict(m)
        if m.get("kind") == "symbol":
            body = text[m["start"]:m["end"]]
            gloss = GROUND_MAP.get(body, "typed from binding context")
            m["kind"] = "symbol-grounded"
            m["tip"] = f"{body} : {gloss}  ·  ⚗ mark-3 ground-to-type"
            m["fields"] = [["symbol", body], ["bound", gloss]]
        out.append(m)
    return out

# --- ⚗ imagined Phase-A: ground-to-type (every symbol bound to its type) -----
GROUND_TO_TYPE = [
    ("$k$", "the ground field (fixed); base of $k$-linearity and of $Vect_k$"),
    ("$\\mathbb{V}=Vect_k$", "the base monoidal category — finite-dimensional $k$-vector spaces  ✓ partly real (Vect grounded)"),
    ("$\\mathcal{A}$", "a finite representable $k$-linear flock — i.e. a finite $\\mathbb{V}$-herd (the paper's central object)  ✓ bound"),
    ("$a,b,c,d$", "objects of the herd $\\mathcal{A}$"),
    ("$\\mathcal{A}(a,b)$", "a hom-object of $\\mathcal{A}$ — a finite-dimensional $k$-vector space"),
    ("$\\tau$", "the ternary herd / flock operation"),
    ("$\\mathcal{H}$", "the category of pairs $(a,b)\\in\\mathcal{A}^{op}\\otimes\\mathcal{A}$, hom $\\mathcal{H}((a,b),(c,d))=\\mathcal{A}(\\tau(b,a,c),d)$"),
    ("$x,y,u,v$", "objects of $\\mathcal{H}$ / components of pairs in $\\mathcal{A}^{op}\\otimes\\mathcal{A}$"),
    ("$S$", "the antipode — a functor $\\mathcal{H}^{op}\\to\\mathcal{H}$, $(x,y)\\mapsto(y,x)$"),
    ("$p$", "the promultiplication of the promonoidal structure on $\\mathcal{H}$"),
    ("$j$", "the promonoidal unit"),
    ("$\\hat{K}$", "the Fourier transform functor on $[\\mathcal{H},Vect_k]$ $=$ restriction along $h$ $=[h,1]$"),
    ("$\\check{K}$", "right adjoint of $\\hat{K}$  ($\\hat{K}\\dashv\\check{K}$)"),
    ("$h$", "the canonical Kleisli / algebroid functor $\\mathcal{A}^{op}\\otimes\\mathcal{A}\\to\\mathcal{H}$, surjective on objects"),
    ("$f$", "an object of $[\\mathcal{H},Vect_k]$ (a module); duality $f^{*}(x,y)=f(y,x)^{*}$"),
]

# --- ⚗ imagined Phase-C: expository scopes classified into the minted kinds ---
EXPO_CLASSIFIED = [
    ("L105", "is here <b>nothing other than</b> the process of restriction along $h$",
     "connection/transfer", "reads $\\hat{K}$ AS a known construction"),
    ("L113", "there is <b>undoubtedly also a connection</b> with Brzeziński &amp; Vercruysse, <i>Bimodule herds</i>",
     "connection/example-source", "links to external prior work"),
    ("L114–116", "<b>may be related to</b> the Ehresmann Groupoid, <b>even in the more general case</b> where $\\mathcal{A}$ is not finite",
     "connection (speculative) + open-problem/status", "a conjectural connection + a generalization left open"),
    ("L108", "this is a <b>straight forward calculation</b>",
     "computes-invariant/calculation", "a method meta-comment, not a proof step"),
]

# --- ⚗ imagined Phase-B: anaphora resolved -----------------------------------
ANAPHORA = [
    ("“for the above $S$”", "the antipode $S:\\mathcal{H}^{op}\\to\\mathcal{H}$ defined three lines earlier"),
    ("“$[\\mathcal{A}^{op}\\otimes\\mathcal{A},Vect_k]$ is such”", "… is $*$-autonomous monoidal biclosed"),
    ("“this is a straightforward calculation”", "the preservation of the left/right internal homs by $\\hat{K}=[h,1]$"),
    ("“the promultiplication $p$ … mentioned above”", "$p$ of the promonoidal structure on $\\mathcal{H}$ (eq. 1)"),
]

# --- ⚗ imagined Phase-F: citations resolved (targets ARE in the paper's bib) --
CITES = [
    ("[1]", "Booker &amp; Street, <i>Torsors, herds and flocks</i> — arXiv:0912.4551"),
    ("[2]", "Day, <i>Monoidal functor categories and graphic Fourier transforms</i> — math.QA/0612496"),
    ("[3]", "Day &amp; Street, <i>Quantum categories, $*$-autonomy, and quantum groupoids</i> — Fields Inst. 2004"),
    ("[5]", "Szlachányi, <i>Finite quantum groupoids and inclusion of finite type</i> — Fields Inst. 2001"),
]

PHASES = [
    ("A · ground-to-type", "⚗", "Bind every symbol to its <b>type</b> from its binder + usage (“Let $X$ be a $Y$”, $f(a,b)$ ⇒ $a,b$ : objects of the domain). The 25% → ~98% lift below."),
    ("B · anaphora resolution", "⚗", "Resolve “the above”, “this”, “such”, definite descriptions to their antecedents — the general case the classical enumerate-item resolver can't reach."),
    ("C · expository classification + discovery", "⚗", "Classify each expository region into the minted hierarchy (connection/*, rationale/*, obstruction/*…); DP-style mint-pressure proposes new kinds. Vocabulary already seeded by the gh200 vote."),
    ("D · argument-graph reconstruction", "✓", "Warrants made explicit + honest typed holes. <b>Already real</b> — the self-gating Codex pool produced 13 checker-PASS graphs (the panel below is genuine output, not imagined)."),
    ("E · recurring-non-term filtering", "⚗", "Drop “study of categories” / “wishes to study” — recurring descriptive phrases the corpus-df prior can't catch because they recur; needs NP-head structure."),
    ("F · citation / cross-paper resolution", "⚗", "Resolve [N] to the cited arXiv paper and import its claims — a cross-document argument graph spanning all of arXiv."),
]


def _tbl(rows, cols):
    h = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="mk"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def main() -> int:
    d = dpv.build("1005.2653", with_ca=True, with_binders=True,
                  with_scopes=True, with_xref=True)
    text, real_marks = d["text"], d["marks"]
    real_body = R.render_marked_source(text, real_marks)              # ✓ REAL (25%)
    imag_body = R.render_marked_source(text, _imagine(text, real_marks))  # ⚗ ~98%
    arg_panel = R.render_argument_graphs("1005.2653")  # ✓ REAL
    g2t = _tbl([(s, t) for s, t in GROUND_TO_TYPE], ["symbol", "inferred type (⚗ imagined)"])
    expo = _tbl([(ln, q, f"<code>{k}</code>", why) for ln, q, k, why in EXPO_CLASSIFIED],
                ["line", "expository cue", "minted kind", "reading"])
    ana = _tbl(ANAPHORA, ["anaphor", "resolved referent (⚗)"])
    cites = _tbl(CITES, ["cite", "resolved target"])
    phases = "".join(
        f'<div class="ph"><h4><span class="badge {"real" if b=="✓" else "imag"}">{b}</span> {t}</h4><p>{d}</p></div>'
        for t, b, d in PHASES)
    doc = f"""<!doctype html><meta charset="utf-8">
<title>1005.2653 — SUPERPOD MARK-3 mockup (imagined)</title>
<style>
body{{font:16px/1.6 Georgia,serif;margin:0;color:#1d1a16;background:#fffdf8}}
main{{max-width:1100px;margin:0 auto;padding:0 28px 70px}}
.banner{{background:#3a1d5e;color:#fbeffd;padding:16px 28px;margin:0 -28px 22px;
  font-family:ui-sans-serif,system-ui,sans-serif}}
.banner b{{color:#ffd9a8}}
.banner .tag{{font-size:12.5px;opacity:.92}}
h2{{font-size:18px;border-bottom:2px solid #e8dfcf;padding-top:18px;margin-top:26px}}
.badge{{font:700 11px/1 ui-sans-serif,system-ui,sans-serif;padding:3px 6px;border-radius:4px;vertical-align:middle}}
.badge.real{{background:#0f766e;color:#fff}} .badge.imag{{background:#7c3aed;color:#fff}}
table.mk{{border-collapse:collapse;width:100%;font:13.5px/1.5 ui-sans-serif,system-ui,sans-serif;margin:10px 0}}
table.mk th,table.mk td{{border-bottom:1px solid #eadfce;padding:7px 10px;text-align:left;vertical-align:top}}
table.mk th{{background:#f3ecfb}}
code{{background:#f3ecfb;border-radius:3px;padding:1px 5px;color:#5b21b6;font-weight:600}}
.stat{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;font-family:ui-sans-serif,system-ui,sans-serif}}
.stat div{{border:1px solid #e9e0d0;background:#fff;border-radius:7px;padding:9px 13px;font-size:12.5px;color:#6a5f4f}}
.stat b{{display:block;font-size:17px;color:#1d1a16}}
.stat .now{{color:#9a3412}} .stat .mk3{{color:#0f766e}}
.ph{{border-left:3px solid #c9bfae;padding:2px 0 2px 13px;margin:11px 0}}
.ph h4{{margin:0 0 2px;font:600 14px/1.3 ui-sans-serif,system-ui,sans-serif}}
.ph p{{margin:0;font-size:13px;color:#4a4337;font-family:ui-sans-serif,system-ui,sans-serif}}
/* ONE shared scroll container -> a single scrollbar drives BOTH columns, so
   they stay locked side by side (same source, same line breaks -> line N aligns
   across both). Column headers stay pinned while you scroll. */
.twoup-scroll{{max-height:80vh;overflow:auto;border:1px solid #d9cdbd;border-radius:7px}}
.twoup{{display:grid;grid-template-columns:1fr 1fr;gap:0;align-items:start}}
.twoup .col{{min-width:0}}
.twoup .col:first-child{{border-right:1px solid #e5dccd}}
.twoup .col-h{{position:sticky;top:0;z-index:3;font:700 12px/1 ui-sans-serif,system-ui,sans-serif;
  padding:8px 11px;color:#fff}}
.twoup .col-h.now{{background:#9a3412}} .twoup .col-h.mk3{{background:#0f766e}}
.twoup .paper{{padding:12px;font-size:12px}}
{R.STYLE}
</style>
<main>
<div class="banner">
  ⚗ <b>SUPERPOD MARK-3 — MOCKUP / imagined output.</b>
  <div class="tag">Hand-authored from a close reading of 1005.2653 to illustrate the target of the
  LLM-scale phase (run over <b>all of arXiv</b>). <span class="badge real">✓</span> = layer already
  shipped by the real CPU pipeline; <span class="badge imag">⚗</span> = imagined (NOT produced by any
  current detector). Compare the real page:
  <a style="color:#ffd9a8" href="1005.2653.html">1005.2653.html</a> (25% grounded).</div>
</div>

<h2>What changes, at a glance</h2>
<div class="stat">
  <div>symbol grounding<b><span class="now">25%</span> &rarr; <span class="mk3">~98%</span></b>ground-to-type ⚗</div>
  <div>anaphora resolved<b><span class="now">0</span> &rarr; <span class="mk3">all</span></b>⚗</div>
  <div>expository scopes<b><span class="now">14 unlabelled</span> &rarr; <span class="mk3">classified</span></b>✓→⚗</div>
  <div>argument graphs<b><span class="mk3">4 / 5 holes</span></b>✓ real (Codex pool)</div>
  <div>citations resolved<b><span class="now">0</span> &rarr; <span class="mk3">5</span></b>⚗</div>
</div>

<h2>Side by side — same source, two annotation depths</h2>
<p style="font-size:14px;color:#4a4337">Left: the <b>real CPU page</b> (debt-heavy, 25% grounded — ungrounded
symbols <span class="k-sym">amber-dashed</span>). Right: the <b>imagined mark-3</b> body with ground-to-type
applied — the same bound variables now <span class="k-symg">teal-grounded</span> to their inferred type
(hover any symbol). Same offsets, same line numbers; only the grounding depth differs.</p>
<div class="twoup-scroll">
  <div class="twoup">
    <div class="col"><div class="col-h now">✓ real · CPU · 25% grounded</div><div class="paper">{real_body}</div></div>
    <div class="col"><div class="col-h mk3">⚗ imagined · MARK-3 · ~98% grounded</div><div class="paper">{imag_body}</div></div>
  </div>
</div>

<h2><span class="badge imag">⚗</span> Phase A — ground-to-type (25% &rarr; ~98%)</h2>
<p style="font-size:14px;color:#4a4337">The classical grounder handles named operators (<code>Vect</code> ✓) and
binder symbols. The remaining ungrounded marks are <b>bound variables</b> — and, as you noted, those aren't
noise: each carries a type from its binder. The superpod infers it:</p>
{g2t}

<h2><span class="badge imag">⚗</span> Phase C — expository scopes, classified</h2>
{expo}

<h2><span class="badge imag">⚗</span> Phase B — anaphora resolved</h2>
{ana}

<h2><span class="badge imag">⚗</span> Phase F — citations resolved <span style="font-size:13px;font-weight:400;color:#6a5f4f">(targets are real, from the paper's bibliography)</span></h2>
{cites}

<h2><span class="badge real">✓</span> Phase D — argument reconstruction (REAL, already shipped)</h2>
<p style="font-size:14px;color:#4a4337">This panel is genuine output from the self-gating Codex pool — not imagined.
It is what the superpod stage produces at arXiv scale.</p>
{arg_panel}

<h2>The MARK-3 run — phases (over all arXiv)</h2>
{phases}
<p style="font-size:13px;color:#6a5f4f;font-family:ui-sans-serif,system-ui,sans-serif">
Scope: the CPU layer-(a) + corpus-df term-prior are per-MSC and already generalize (re-point the corpus);
the mark-3 phases are MSC-agnostic — they read mathematical prose, so the run extends from math.CT to all of arXiv.
Hitlist anchors: ground-to-type, anaphora, expository-classification + discovery, argument-graph reconstruction,
recurring-non-term filtering, cross-paper citation resolution.</p>
</main>
"""
    OUT.write_text(doc, encoding="utf8")
    print(f"wrote {OUT}  ({len(doc):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
