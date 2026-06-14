#!/usr/bin/env python3
"""Mine structured definition-scopes from mathlib4 (.lean) — the easiest,
formal source (E-definition-scopes / definition-scope-mining.md).

A Lean `structure`/`class`/`def` IS a definition-scope: definiendum = the
name; the binders/instances = its parametrization ("in C" = the
[MonoidalCategory C] context + an `X : C` field); the fields = the definiens
structure (data morphisms + axioms), each with its doc-comment gloss. This
resolves operator semantics questions (Joe: does "in $\\C$" mean element-of?)
against the formal text, not by guessing.

    mine_mathlib_defs.py <file-or-dir> [--name Mon] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MATHLIB = Path("/home/joe/code/mathlib4")
DECL_RE = re.compile(
    r"^(?P<kw>structure|class|def|abbrev|inductive)\s+(?P<name>[A-Za-z_][\w'.]*)"
    r"(?P<rest>.*)$")
# a field line inside a structure/class `where` block: `name : type`
FIELD_RE = re.compile(r"^\s+(?:\[[^\]]+\]\s*)?(?P<fname>[A-Za-z_][\w']*)\s*"
                      r"(?P<args>(?:\([^)]*\)\s*)*):\s*(?P<ftype>.+?)(?::=.*)?$")
PARAM_RE = re.compile(r"(\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\})")


def parse_params(rest: str):
    """Extract (x : T) / [Inst] / {x : T} binders before `where`/`:=`/`extends`."""
    head = re.split(r"\bwhere\b|:=|\bextends\b", rest, maxsplit=1)[0]
    return [m.group(0) for m in PARAM_RE.finditer(head)]


def mine_file(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    defs, i, n = [], 0, len(lines)
    pending_doc = None
    ctx_ambient = []  # file-level `variable ... [Category C] [MonoidalCategory C]`
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("variable") and ("Category" in line or "Monoidal" in line):
            ctx_ambient = [pm.group(0) for pm in PARAM_RE.finditer(line)
                           if "Category" in pm.group(0) or "Monoidal" in pm.group(0)]
        dm = re.match(r"^/--\s*(.*?)\s*-/\s*$", line)
        if dm:
            pending_doc = dm.group(1); i += 1; continue
        if line.startswith("/--"):  # multi-line doc
            buf = [line[3:]]
            while i + 1 < n and "-/" not in lines[i]:
                i += 1; buf.append(lines[i])
            pending_doc = re.sub(r"-/.*$", "", " ".join(buf)).strip(); i += 1; continue
        m = DECL_RE.match(line)
        if m:
            kw, name, rest = m.group("kw"), m.group("name"), m.group("rest")
            params = parse_params(rest)
            fields = []
            has_where = "where" in rest
            j = i + 1
            fdoc = None
            # gather following lines: continued params (no `where` yet) then fields
            while j < n and not has_where and not re.search(r"\bwhere\b|:=", lines[j]):
                params += [pm.group(0) for pm in PARAM_RE.finditer(lines[j])]
                if "where" in lines[j] or ":=" in lines[j]:
                    break
                j += 1
            if j < n and "where" in lines[j]:
                has_where = True
            if has_where:
                k = j + 1 if (j < n and "where" in lines[j] and j != i) else i + 1
                while k < n:
                    ln = lines[k]
                    if ln.strip() == "" or (ln and not ln[0].isspace()
                                            and not ln.lstrip().startswith("/--")):
                        if ln.strip() == "":
                            k += 1
                            if k < n and lines[k] and not lines[k][0].isspace():
                                break
                            continue
                        break
                    dd = re.match(r"^\s*/--\s*(.*?)\s*-/", ln)
                    if dd:
                        fdoc = dd.group(1); k += 1; continue
                    fm = FIELD_RE.match(ln)
                    if fm and "⟶" not in fm.group("fname"):
                        fields.append({"field": fm.group("fname"),
                                       "type": fm.group("ftype").strip()[:120],
                                       "doc": fdoc})
                        fdoc = None
                    k += 1
            defs.append({
                "name": name, "kind": kw, "doc": pending_doc,
                "params": params, "fields": fields,
                "file": str(path.relative_to(MATHLIB)), "line": i + 1,
                # the "in C" question, answered structurally (inline binders
                # + the enclosing file-level monoidal-category context):
                "ambient": ([p for p in params if "Category" in p or "Monoidal" in p]
                            or ctx_ambient),
            })
            pending_doc = None
        else:
            if line.strip() and not line.startswith("--"):
                pending_doc = None
        i += 1
    return defs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target"); ap.add_argument("--name")
    ap.add_argument("--json"); ap.add_argument("--kinds", default="structure,class")
    a = ap.parse_args(argv)
    p = Path(a.target)
    if not p.is_absolute():
        p = MATHLIB / p
    files = [p] if p.is_file() else sorted(p.rglob("*.lean"))
    kinds = set(a.kinds.split(","))
    out = []
    for f in files:
        for d in mine_file(f):
            if d["kind"] in kinds and (not a.name or d["name"] == a.name):
                out.append(d)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"{len(out)} definition-scopes from {len(files)} file(s)")
    for d in out[:12]:
        amb = " ; ".join(d["ambient"]) or "(no monoidal ambient)"
        print(f"\n● {d['kind']} {d['name']}  [{d['file']}:{d['line']}]")
        if d["doc"]:
            print(f"   doc: {d['doc'][:80]}")
        print(f"   ambient (the 'in C'): {amb}")
        for fld in d["fields"][:6]:
            g = f"  — {fld['doc'][:46]}" if fld["doc"] else ""
            print(f"     · {fld['field']} : {fld['type'][:60]}{g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
