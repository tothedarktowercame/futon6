#!/usr/bin/env python3
"""Paper-disjoint, fixed-K recognition experiment (not inference validation).

Run --self-test for invariant/negative controls. Run --bundle RUN --k 10
--output result.json for leave-one-producing-paper-out evaluation. Optional
--graphs DIR [DIR ...] substitutes clean graphs; retained candidate windows
and full paper texts still come from RUN. No input artifacts are modified.
The runnable acceptance check is summarize(): positive macro endpoint change
AND an advantage over the mean random-cue endpoint change. No significance or
corpus-wide claim is implied. All folds and random draws are retained.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import statistics
import tempfile

import iatc_lexicon_harvest as lh
from iatc_move_reground import cluster_cues, score
import strategy_recognizer as sr

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paper(path):
    return Path(path).name.split('.candidate.json')[0].split('.edn')[0].rsplit('__p', 1)[0]


@contextmanager
def retained_papers(texts):
    """Use retained full paper bytes, never today's external dp_paper_view.

    This replaces only the harvester's source loader, not extraction, confidence,
    tokenizer or ranking. Its existing line-coordinate limitations remain.
    """
    original = lh._text_lines
    lh._text_lines = lambda pid, cache: texts[pid].split('\n')
    try:
        yield
    finally:
        lh._text_lines = original


def merge_lex(target, source):
    for phrase, entry in source.items():
        dst = target.setdefault(phrase, {'count': 0, 'conf': []})
        dst['count'] += entry['count']
        dst['conf'].extend(entry['conf'])


class Scorer:
    """Cache exact sentence matches; same bucket arithmetic as score(), unrounded."""
    def __init__(self, vocab, windows):
        import re
        self.counts = Counter()
        self.unrecognized = []
        self.hits = {}
        for window in windows:
            for sent in re.split(r'(?<=[.!?])\s+|\n', sr.strip_latex(window)):
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                bucket, _ = sr.classify(sent, vocab)
                self.counts[bucket] += 1
                if bucket == 'ungrounded':
                    self.unrecognized.append(sent.lower())

    def __call__(self, cues):
        matched = set()
        for cue in cues:
            if cue not in self.hits:
                self.hits[cue] = {i for i, s in enumerate(self.unrecognized)
                                  if sr.matches(s, [cue]) is not None}
            matched.update(self.hits[cue])
        g = self.counts['grounded']
        t = self.counts['thin'] + len(matched)
        u = self.counts['ungrounded'] - len(matched)
        total = g + t + u
        return {'score': (g + .5*t)/total if total else None,
                'grounded': g, 'thin': t, 'ungrounded': u, 'denominator': total}


def choose(cues, k, seed=None):
    if len(cues) < k:
        return None
    if seed is None:
        return cues[:k]
    # Stable pseudorandom priorities: uniform subset, coupled across checkpoints.
    return sorted(cues, key=lambda c: hashlib.sha256(f'{seed}:{c}'.encode()).digest())[:k]


def check_partition(a, b):
    if not a or not b or set(a) & set(b):
        raise ValueError('Empty partition or paper leakage.')


def spread(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def summarize(folds):
    """Predeclared endpoint comparison; never compare different folds at different n."""
    common = sorted(set.intersection(*[
        {p['n'] for p in f['points'] if p['frequency'] is not None
         and p['frequency']['B']['score'] is not None
         and p['frequency']['A']['score'] is not None} for f in folds]))
    if len(common) < 2:
        return {'verdict': 'insufficient-fixed-K-checkpoints', 'common_checkpoints': common}
    lo, hi = common[0], common[-1]
    effects = []
    for f in folds:
        a, b = [next(p for p in f['points'] if p['n'] == n) for n in (lo, hi)]
        delta = b['frequency']['B']['score'] - a['frequency']['B']['score']
        rd = [y['B']['score'] - x['B']['score'] for x, y in zip(a['random'], b['random'])]
        effects.append({'heldout_paper': f['B'][0], 'frequency_delta': delta,
                        'random_delta': spread(rd),
                        'advantage': delta - statistics.mean(rd)})
    fd = [e['frequency_delta'] for e in effects]
    adv = [e['advantage'] for e in effects]
    rises = statistics.mean(fd) > 0
    beats = statistics.mean(adv) > 0
    return {'common_checkpoints': common, 'from_n': lo, 'to_n': hi,
            'heldout_rises': rises, 'beats_random': beats,
            'acceptance': rises and beats,
            'frequency_delta': spread(fd), 'frequency_advantage_over_random': spread(adv),
            'positive_folds': sum(d > 0 for d in fd), 'effects': effects,
            'interpretation': 'Descriptive endpoint check, not a significance test or mathematical validity.'}


def evaluate(bundle, graph_dirs, k, seed, repeats):
    artifacts = bundle / 'artifacts'
    graphs = sorted(p for d in graph_dirs for p in d.glob('*.edn')
                    if not p.name.endswith('.rung2.edn'))
    if not graphs or len({p.name for p in graphs}) != len(graphs):
        raise ValueError('No graphs or duplicate graph identities; do not pool versions.')
    groups = sorted({paper(p) for p in graphs})
    if len(groups) < 2:
        raise ValueError('At least two producing papers required.')
    candidates = sorted(artifacts.joinpath('candidates').glob('*.candidate.json'))
    windows = {pid: [] for pid in groups}
    inputs = list(graphs)
    names = set()
    for p in candidates:
        if paper(p) in groups:
            c = json.loads(p.read_text())
            if c['paper-id'] != paper(p) or not c['source-window'].strip():
                raise ValueError(f'Candidate identity/window invalid: {p}')
            windows[paper(p)].append(c['source-window'])
            names.add(p.name.removesuffix('.candidate.json'))
            inputs.append(p)
    if any(p.stem not in names for p in graphs):
        raise ValueError('A graph lacks a retained candidate window.')
    texts = {}
    for pid in groups:
        p = artifacts / 'marks' / f'fable-{pid}-dp-emacs.json'
        texts[pid] = json.loads(p.read_text())['text']
        inputs.append(p)
    vocab_path = ROOT / 'holes/clean/tactic-gesture-vocab.edn'
    inputs += [vocab_path, Path(__file__), Path(lh.__file__), Path(sr.__file__),
               ROOT / 'scripts/iatc_move_reground.py']
    hashes = {str(p): digest(p) for p in inputs}
    vocab = sr.load_vocab(str(vocab_path))
    lexicons = {}
    with retained_papers(texts), tempfile.TemporaryDirectory(prefix='heldout-harvest-') as td:
        for p in graphs:
            link = Path(td) / p.name
            link.symlink_to(p.resolve())
            lexicons[str(p)] = lh.harvest(td)[0]
            link.unlink()
    common_max = min(sum(paper(g) != pid for g in graphs) for pid in groups)
    checkpoints = sorted({n for n in (1, 3, 5, 10, 20, 30, 50, 100, common_max)
                          if n <= common_max})
    folds = []
    for pid in groups:
        train = [g for g in graphs if paper(g) != pid]
        random.Random(f'{seed}:{pid}').shuffle(train)
        a_ids = [p for p in groups if p != pid]
        check_partition(a_ids, [pid])
        sa = Scorer(vocab, [w for p in a_ids for w in windows[p]])
        sb = Scorer(vocab, windows[pid])
        lex, points = {}, []
        for n, g in enumerate(train, 1):
            merge_lex(lex, lexicons[str(g)])
            if n not in checkpoints:
                continue
            cues = cluster_cues(lex, top=None)
            picked = choose(cues, k)
            point = {'n': n, 'cues_available': len(cues), 'K_used': k if picked else 0,
                     'shortfall': max(0, k-len(cues)), 'frequency': None, 'random': []}
            if picked:
                point['frequency'] = {'cues': picked, 'A': sa(picked), 'B': sb(picked)}
                # Independently exercise the original scorer on real held-out windows.
                aug = {**vocab, 'heuristic': {**vocab['heuristic'], 'corpus-move': picked}}
                reference = score(aug, windows[pid])
                assert all(point['frequency']['B'][key] == reference[key]
                           for key in ('grounded', 'thin', 'ungrounded'))
                for r in range(repeats):
                    rc = choose(cues, k, f'{seed}:{pid}:{r}')
                    point['random'].append({'cues': rc, 'A': sa(rc), 'B': sb(rc)})
            points.append(point)
        folds.append({'A': a_ids, 'B': [pid], 'graph_order': [g.name for g in train],
                      'A_windows': sum(len(windows[p]) for p in a_ids),
                      'B_windows': len(windows[pid]), 'points': points})
    if any(digest(p) != h for p, h in hashes.items()):
        raise RuntimeError('Input/code changed during evaluation; rerun with stable inputs.')
    return {'schema': 'accretion-heldout-v1', 'K': k, 'seed': seed, 'random_repeats': repeats,
            'papers': groups, 'graphs': len(graphs), 'common_max_n': common_max,
            'candidate_windows': {p: len(windows[p]) for p in groups},
            'input_sha256': hashes, 'folds': folds, 'acceptance_check': summarize(folds)}


def self_test():
    vocab = sr.load_vocab(str(ROOT / 'holes/clean/tactic-gesture-vocab.edn'))
    windows = ['The zephyr conclusion lacks any explanation here.',
               'The quasar statement is another unknown sentence.']
    s = Scorer(vocab, windows)
    for cues in ([], ['zephyr'], ['zephyr', 'quasar'], ['absentword']):
        aug = {**vocab, 'heuristic': {**vocab['heuristic'], 'corpus-move': cues}}
        reference = score(aug, windows)
        actual = s(cues)
        assert all(actual[k] == reference[k] for k in ('grounded', 'thin', 'ungrounded'))
        assert round(actual['score'], 3) == reference['proof-move-grounding']
    assert choose(['one'], 2) is None
    assert len(choose(['one', 'two', 'three'], 2, 7)) == 2
    assert paper('math__0409598__p12.edn') == 'math__0409598'
    try:
        check_partition(['same-paper'], ['same-paper'])
    except ValueError:
        pass
    else:
        raise AssertionError('Deliberate paper leakage was not rejected.')
    # Deliberately bad accretion claim: same cue count, worse held-out matches.
    assert s(['absentword'])['score'] < s(['zephyr'])['score']
    def point(n, x, r):
        return {'n': n, 'frequency': {'A': {'score': x}, 'B': {'score': x}},
                'random': [{'B': {'score': r}}]}
    f = [{'B': ['heldout'], 'points': [point(1, .3, .2), point(2, .2, .3)]}]
    assert summarize(f)['acceptance'] is False
    f[0]['points'] = [point(1, .2, .2), point(2, .3, .3)]
    assert summarize(f)['acceptance'] is False  # random rises just as much
    f[0]['points'] = [point(1, .2, .2), point(2, .3, .2)]
    assert summarize(f)['acceptance'] is True
    print('PASS: exact scorer parity, K shortfall, old-style paper identity, declining and random-matched controls.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--graphs', type=Path, nargs='+')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--random-repeats', type=int, default=32)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    if args.bundle:
        if args.k < 1 or args.random_repeats < 1 or not args.output:
            ap.error('Positive K/repeats and --output required.')
        result = evaluate(args.bundle, args.graphs or [args.bundle/'artifacts/graphs'],
                          args.k, args.seed, args.random_repeats)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['acceptance_check'], indent=2))
    elif not args.self_test:
        ap.error('--bundle or --self-test required.')


if __name__ == '__main__':
    main()
