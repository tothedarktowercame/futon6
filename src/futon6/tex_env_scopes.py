"""tex_env_scopes.py — REAL LaTeX environment scopes for paper anatomy.

The February CT run's env/* scopes matched bare prose headers ("\\nLemma"),
which turns lemma-reference lists into walls of fake environments (909 in
arxiv-math/0506470) and misses \\newtheorem-renamed real ones. This detector
reads the actual LaTeX structure:

  - \\newtheorem{lem}{Lemma} resolution (incl. starred, optional args)
  - \\begin{X}...\\end{X} spans with positions, properly nested
  - canonical kinds via the newtheorem map + a standard alias table

Emits futon4-compatible hyperedges with hx/type env-tex/<kind> so the
comparison against the old env/* layer stays visible.
"""
from __future__ import annotations

import re

ALIASES = {
    "thm": "theorem", "theo": "theorem", "theorem": "theorem",
    "lem": "lemma", "lemma": "lemma",
    "prop": "proposition", "proposition": "proposition",
    "cor": "corollary", "corollary": "corollary",
    "defn": "definition", "defi": "definition", "dfn": "definition",
    "definition": "definition",
    "rem": "remark", "remark": "remark", "rmk": "remark",
    "ex": "example", "example": "example", "exa": "example",
    "conj": "conjecture", "conjecture": "conjecture",
    "proof": "proof",
    "notation": "notation", "construction": "construction",
}

_NEWTHM = re.compile(
    r"\\newtheorem\*?\s*\{([^}]+)\}\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_BEGIN_END = re.compile(r"\\(begin|end)\s*\{([A-Za-z@*]+)\}")


def newtheorem_map(text: str) -> dict[str, str]:
    out = {}
    for m in _NEWTHM.finditer(text):
        env, title = m.group(1), m.group(2).strip().lower()
        out[env] = ALIASES.get(title.split()[0] if title else "", None) or \
            re.sub(r"[^a-z]+", "-", title) or env
    return out


def detect_tex_env_scopes(entity_id: str, text: str) -> list[dict]:
    envmap = newtheorem_map(text)

    def kind_of(env: str):
        base = env.rstrip("*")
        if base in envmap:
            return envmap[base]
        return ALIASES.get(base)

    scopes = []
    stack = []
    for m in _BEGIN_END.finditer(text):
        which, env = m.group(1), m.group(2)
        if which == "begin":
            stack.append((env, m.start(), m.end()))
        else:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == env:
                    _, bpos, bend = stack.pop(i)
                    kind = kind_of(env)
                    if kind:
                        body = text[bend:m.start()]
                        scopes.append({
                            "hx/id": f"{entity_id}:texenv-{len(scopes):04d}",
                            "hx/type": f"env-tex/{kind}",
                            "hx/role": "environment",
                            "hx/parent": None,
                            "hx/ends": [
                                {"role": "entity", "ident": entity_id},
                                {"role": "environment", "name": kind,
                                 "env": env, "depth": len(stack) + 1},
                            ],
                            "hx/content": {
                                "match": text[bpos:bend] + body[:80],
                                "position": bpos,
                                "end": m.end(),
                            },
                            "hx/labels": ["scope", "env-tex", kind],
                        })
                    break
    return scopes
