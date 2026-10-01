#!/usr/bin/env python3
"""Attach run-owned annotations to an oxide/Tufte page from the exact marks text.

Usage: .venv/bin/python scripts/render_scope_view.py RUN PAPER TYPESET_DIR OUTPUT
TYPESET_DIR must contain PAPER.tex, PAPER-tufte.html and conversion.log.
Positions are source-line anchors, not claims of exact rendered glyph coverage.
"""
import argparse
from bisect import bisect_right
from collections.abc import Mapping
import json
from pathlib import Path
import re

import edn_format
from edn_compat import edn_safe
from run_artifacts import proof_graphs


def plain(value):
    if isinstance(value, Mapping):
        return {str(k).lstrip(':'): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, edn_format.ImmutableList)):
        return [plain(v) for v in value]
    if isinstance(value, edn_format.Keyword):
        return str(value).lstrip(':')
    return value


def build(run, paper, typeset):
    marks = json.loads((run / 'artifacts/marks' / f'fable-{paper}-dp-emacs.json').read_text())
    source = marks['text']
    if (typeset / f'{paper}.tex').read_text() != source:
        raise ValueError('Typeset source differs from the run-owned marks text')
    source_files = dict(re.findall(r'^Info:source-map:source\s+\[(\d+)\]\s+(.+)$',
                                  (typeset / 'conversion.log').read_text(), re.M))
    indices = [int(i) for i, name in source_files.items()
               if Path(name).resolve() == (typeset / f'{paper}.tex').resolve()]
    if len(indices) != 1:
        raise ValueError('Converter log must identify exactly one matching source file')
    starts = [0] + [m.end() for m in re.finditer('\n', source)]
    lines = source.splitlines(keepends=True)
    records = []

    def add(layer, kind, lo, hi, detail, excerpt, artifact):
        if not (1 <= lo <= hi <= len(starts)):
            raise ValueError(f'Invalid source range {lo}–{hi} in {artifact}')
        records.append(dict(id=f'scope-{len(records)}', layer=layer, kind=kind,
                            lo=lo, hi=hi, detail=detail, excerpt=excerpt,
                            artifact=artifact))

    for m in marks['marks']:
        a, b = m['start'], m['end']
        if not (0 <= a < b <= len(source)):
            raise ValueError(f'Invalid mark offsets: {m}')
        add('Source marks', m['kind'], bisect_right(starts, a),
            bisect_right(starts, b - 1), m, source[a:b],
            f'artifacts/marks/fable-{paper}-dp-emacs.json')

    graphs = [Path(p) for p in proof_graphs(str(run / 'artifacts/graphs'))
              if Path(p).name == f'{paper}.edn' or Path(p).name.startswith(f'{paper}__p')]
    expos = sorted((run / 'artifacts/expo').glob(f'{paper}_*.edn'))
    for path in graphs + expos:
        data = plain(edn_format.loads(edn_safe(path.read_text())))
        if data['paper/id'] != paper:
            raise ValueError(f'Wrong paper in {path}')
        items = data['scopes'] if path in expos else [data]
        for item in items:
            lo, hi = item['source']['lines']
            add('Expository scopes' if path in expos else 'Proof graphs',
                item.get('kind', 'proof'), lo, hi, item,
                ''.join(lines[lo - 1:hi]), str(path.relative_to(run)))
    page = (typeset / f'{paper}-tufte.html').read_text()
    if 'ltx_ERROR' in page or 'data-sourcepos=' not in page:
        raise ValueError('Typeset page has conversion errors or no source positions')
    payload = json.dumps(dict(file=indices[0], records=records), ensure_ascii=False).replace('<', '\\u003c')
    assets = Path(__file__).with_name('scope_view')
    page = page.replace('</head>', '<style>' + (assets / 'view.css').read_text() + '</style></head>', 1)
    interface = (assets / 'panel.html').read_text()
    page = page.replace('</body>', interface + '<script id="scope-data" type="application/json">' + payload
                        + '</script><script>' + (assets / 'view.js').read_text() + '</script></body>', 1)
    return page, records


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('paper')
    ap.add_argument('typeset', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    page, records = build(a.run, a.paper, a.typeset)
    a.output.write_text(page)
    print(f'{a.output}: {len(records)} annotations')
