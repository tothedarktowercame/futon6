"""Thread hypergraph assembler — Stage 9a of the futon6 pipeline.

Takes a math.SE thread with its wiring annotations and expression parses,
produces a typed hypergraph suitable for graph neural network embedding.

A typed hypergraph H = (N, E) where:
  N = typed nodes  {post, term, expression, scope}
  E = typed hyperedges  {iatc, mention, discourse, scope, surface, categorical}

Each edge connects an ordered list of node IDs ("ends").

    >>> from futon6.hypergraph import assemble
    >>> hg = assemble(raw_thread, wiring, parses)
    >>> len(hg['nodes']), len(hg['edges'])
    (48, 62)
"""

from __future__ import annotations

import hashlib
import html as html_mod
import re
from typing import Any

from futon6.latex_sexp import parse as sexp_parse, parse_all


# ---------------------------------------------------------------------------
# Node / edge types
# ---------------------------------------------------------------------------

POST_TYPES = {'question', 'answer', 'comment'}

NODE_TYPES = {
    'post',         # question, answer, or comment
    'term',         # NER concept (canonical name)
    'expression',   # a parsed LaTeX expression surface
    'scope',        # a scope binding (let, forall, where)
}

EDGE_TYPES = {
    'iatc',         # illocutionary act: (source_post, target_post)
    'mention',      # NER mention: (post, term)
    'discourse',    # rhetorical marker: (post,) with attrs
    'scope',        # scope binding: (scope_node, post)
    'surface',      # expression surface: (expression, post)
    'categorical',  # categorical badge: (post,) with attrs
}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(raw: dict, wiring: dict, parses: dict | None = None) -> dict:
    """Assemble a typed hypergraph from thread data + wiring annotations.

    Parameters
    ----------
    raw : dict
        Raw StackExchange thread with 'question', 'answers', 'comments_q',
        'comments_a' keys.
    wiring : dict
        Wiring diagram with 'nodes' and 'edges' arrays.
    parses : dict or None
        Optional map of LaTeX -> s-exp string (from the demo PARSES or
        from running latex_sexp.parse on the thread).  If None, parses
        are computed on the fly from the raw HTML.

    Returns
    -------
    dict with keys:
        'thread_id': int
        'nodes': list[dict]  — each has 'id', 'type', 'subtype', 'attrs'
        'edges': list[dict]  — each has 'type', 'ends', 'attrs'
        'meta': dict         — statistics
    """
    nodes: dict[str, dict] = {}   # id -> node dict
    edges: list[dict] = []

    thread_id = raw.get('question', {}).get('id', 0)

    # -----------------------------------------------------------------------
    # 1. Post nodes
    # -----------------------------------------------------------------------
    q = raw['question']
    _add_node(nodes, q_id(q), 'post', 'question', {
        'score': q.get('score', 0),
        'title': q.get('title', ''),
        'tags': q.get('tags', []),
    })

    for ans in raw.get('answers', []):
        _add_node(nodes, a_id(ans), 'post', 'answer', {
            'score': ans.get('score', 0),
            'is_accepted': ans.get('is_accepted', False),
        })

    # Comments on question
    for c in raw.get('comments_q', []):
        _add_node(nodes, c_id(c), 'post', 'comment', {
            'score': c.get('score', 0),
            'parent': q_id(q),
        })

    # Comments on answers
    for aid_str, clist in raw.get('comments_a', {}).items():
        for c in clist:
            _add_node(nodes, c_id(c), 'post', 'comment', {
                'score': c.get('score', 0),
                'parent': f'a-{aid_str}',
            })

    # -----------------------------------------------------------------------
    # 2. IATC edges (from wiring)
    # -----------------------------------------------------------------------
    for e in wiring.get('edges', []):
        edges.append({
            'type': 'iatc',
            'ends': [e['from'], e['to']],
            'attrs': {'act': e.get('iatc', 'unknown')},
        })

    # -----------------------------------------------------------------------
    # 3. NER term nodes + mention edges
    # -----------------------------------------------------------------------
    for wn in wiring.get('nodes', []):
        post_id = wn['id']
        for t in wn.get('ner_terms', []):
            canon = t.get('canon', t['term'])
            term_nid = f'term:{canon}'
            _add_node(nodes, term_nid, 'term', canon, {
                'surface_forms': set(),
            })
            # Accumulate surface forms
            nodes[term_nid]['attrs']['surface_forms'].add(t['term'])
            edges.append({
                'type': 'mention',
                'ends': [post_id, term_nid],
                'attrs': {'surface': t['term']},
            })

    # -----------------------------------------------------------------------
    # 4. Discourse edges
    # -----------------------------------------------------------------------
    for wn in wiring.get('nodes', []):
        post_id = wn['id']
        for d in wn.get('discourse', []):
            role = d.get('hx/role', '')
            dtype = d.get('hx/type', '')

            if role == 'component':
                # Scope binding — create a scope node
                scope_nid = d.get('hx/id', f'scope:{post_id}:{dtype}')
                ends_data = d.get('hx/ends', [])
                _add_node(nodes, scope_nid, 'scope', dtype, {
                    'match': d.get('hx/content', {}).get('match', ''),
                    'ends': ends_data,
                })
                edges.append({
                    'type': 'scope',
                    'ends': [scope_nid, post_id],
                    'attrs': {'binding_type': dtype},
                })
            else:
                # Wire, label, port — discourse marker on the post
                edges.append({
                    'type': 'discourse',
                    'ends': [post_id],
                    'attrs': {
                        'hx_id': d.get('hx/id', ''),
                        'role': role,
                        'dtype': dtype,
                        'match': d.get('hx/content', {}).get('match', ''),
                    },
                })

    # -----------------------------------------------------------------------
    # 5. Categorical edges
    # -----------------------------------------------------------------------
    for wn in wiring.get('nodes', []):
        post_id = wn['id']
        for cat in wn.get('categorical', []):
            edges.append({
                'type': 'categorical',
                'ends': [post_id],
                'attrs': {
                    'concept': cat.get('hx/type', ''),
                    'score': cat.get('hx/content', {}).get('score', 0),
                },
            })

    # -----------------------------------------------------------------------
    # 6. Expression surface nodes + edges
    # -----------------------------------------------------------------------
    # Gather all HTML bodies
    bodies = []
    bodies.append((q_id(q), q.get('body_html', '')))
    for ans in raw.get('answers', []):
        bodies.append((a_id(ans), ans.get('body_html', '')))
    for c in raw.get('comments_q', []):
        bodies.append((c_id(c), c.get('text', '')))
    for aid_str, clist in raw.get('comments_a', {}).items():
        for c in clist:
            bodies.append((c_id(c), c.get('text', '')))

    for post_id, html in bodies:
        expressions = _extract_expressions(html, parses)
        for expr in expressions:
            expr_nid = _expr_id(post_id, expr['latex'])
            _add_node(nodes, expr_nid, 'expression', 'math', {
                'latex': expr['latex'],
                'sexp': expr['sexp'],
                'display': expr.get('display', False),
            })
            edges.append({
                'type': 'surface',
                'ends': [expr_nid, post_id],
                'attrs': {'position': expr.get('position', 0)},
            })

    # -----------------------------------------------------------------------
    # 7. Finalize
    # -----------------------------------------------------------------------
    # Convert sets to lists for JSON serialization
    for n in nodes.values():
        for k, v in n.get('attrs', {}).items():
            if isinstance(v, set):
                n['attrs'][k] = sorted(v)

    node_list = sorted(nodes.values(), key=lambda n: n['id'])

    return {
        'thread_id': thread_id,
        'nodes': node_list,
        'edges': edges,
        'meta': {
            'n_nodes': len(node_list),
            'n_edges': len(edges),
            'n_posts': sum(1 for n in node_list if n['type'] == 'post'),
            'n_terms': sum(1 for n in node_list if n['type'] == 'term'),
            'n_expressions': sum(1 for n in node_list if n['type'] == 'expression'),
            'n_scopes': sum(1 for n in node_list if n['type'] == 'scope'),
            'edge_types': _count_by(edges, 'type'),
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def q_id(q: dict) -> str:
    return f"q-{q['id']}"

def a_id(a: dict) -> str:
    return f"a-{a['id']}"

def c_id(c: dict) -> str:
    return f"c-{c['id']}"

def _add_node(nodes: dict, nid: str, ntype: str, subtype: str,
              attrs: dict) -> None:
    if nid not in nodes:
        nodes[nid] = {'id': nid, 'type': ntype, 'subtype': subtype,
                       'attrs': attrs}

def _expr_id(post_id: str, latex: str) -> str:
    """Deterministic expression node ID."""
    h = hashlib.sha1(latex.encode()).hexdigest()[:8]
    return f"expr:{post_id}:{h}"

def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        v = item.get(key, '?')
        counts[v] = counts.get(v, 0) + 1
    return counts


def _extract_expressions(text: str, parses: dict | None) -> list[dict]:
    """Extract math expressions from HTML/text, parse each to s-exp."""
    results = []
    seen = set()

    # Display math: $$...$$
    for m in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
        tex = _clean_tex(m.group(1))
        if tex not in seen:
            seen.add(tex)
            sexp = _lookup_or_parse(tex, parses)
            results.append({
                'latex': tex, 'sexp': sexp,
                'display': True, 'position': m.start(),
            })

    # Inline math in body: $...$
    display_ranges = [(m.start(), m.end())
                      for m in re.finditer(r'\$\$.+?\$\$', text, re.DOTALL)]
    for m in re.finditer(r'\$([^$\n]+?)\$', text):
        if any(s <= m.start() < e for s, e in display_ranges):
            continue
        tex = _clean_tex(m.group(1))
        if tex not in seen:
            seen.add(tex)
            sexp = _lookup_or_parse(tex, parses)
            results.append({
                'latex': tex, 'sexp': sexp,
                'display': False, 'position': m.start(),
            })

    # Also handle \(...\) notation
    for m in re.finditer(r'\\\((.+?)\\\)', text, re.DOTALL):
        tex = _clean_tex(m.group(1))
        if tex not in seen:
            seen.add(tex)
            sexp = _lookup_or_parse(tex, parses)
            results.append({
                'latex': tex, 'sexp': sexp,
                'display': False, 'position': m.start(),
            })

    results.sort(key=lambda r: r['position'])
    return results


def _clean_tex(raw: str) -> str:
    """Decode HTML entities and normalize whitespace in extracted LaTeX."""
    tex = html_mod.unescape(raw).strip()
    return tex


def _lookup_or_parse(tex: str, parses: dict | None) -> str:
    """Look up a curated parse first; fall back to automatic parsing."""
    if parses:
        # Try exact match
        if tex in parses:
            return parses[tex]
        # Try whitespace-normalized match
        norm = re.sub(r'\s+', ' ', tex).strip()
        for k, v in parses.items():
            if re.sub(r'\s+', ' ', k).strip() == norm:
                return v
    # Automatic parse
    return sexp_parse(tex)


# ---------------------------------------------------------------------------
# Convenience: run on a thread JSON file
# ---------------------------------------------------------------------------

def assemble_from_files(raw_path: str, wiring_path: str,
                        parses: dict | None = None) -> dict:
    """Load thread and wiring from JSON files, assemble hypergraph."""
    import json
    with open(raw_path) as f:
        raw = json.load(f)
    with open(wiring_path) as f:
        wiring = json.load(f)
    return assemble(raw, wiring, parses)


if __name__ == '__main__':
    import json
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m futon6.hypergraph RAW.json WIRING.json [output.json]")
        sys.exit(1)

    raw_path = sys.argv[1]
    wiring_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    hg = assemble_from_files(raw_path, wiring_path)
    result = json.dumps(hg, indent=2, ensure_ascii=False)

    if out_path:
        with open(out_path, 'w') as f:
            f.write(result)
        print(f"Wrote {out_path}: {hg['meta']}")
    else:
        print(result)
