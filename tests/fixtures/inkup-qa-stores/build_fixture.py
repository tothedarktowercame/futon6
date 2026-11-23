"""Deterministic stand-in for the BGE model, used to build and test the fixture store.

A text's vector is the normalised sum of unit buckets, one bucket per token (the retrieval
module's tokenizer). Fixture tokens each own a bucket, so no model download is needed, the
fixture's embeddings.npy regenerates byte for byte, and similarity means shared tokens.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from futon6.inkup_question_retrieval import embedding_text, normalize_rows, tokenize  # noqa: E402

DIM = 1024  # the real store's width; wide enough that fixture tokens do not collide


HALF = DIM // 2


def fixture_vocabulary(root: Path | None = None) -> set[str]:
    """Every token of every stored text in the fixture stores next to this file."""
    root = root or Path(__file__).resolve().parent
    import json
    vocab = set()
    for path in sorted(root.glob("*/entities.json")):
        for entity in json.loads(path.read_text(encoding="utf-8")):
            vocab.update(tokenize(embedding_text(entity)))
    return vocab


class FakeEmbedder:
    """Implements encode(texts) and encode_query(text), like the production embedder.

    Collision-free by construction where it matters: every fixture vocabulary token owns one
    bucket in the lower half, so two stored texts are similar exactly when they share a token.
    Tokens the fixture never saw (query-side words such as a category label) hash into the
    upper half, where they cannot lend a stored text similarity it did not earn.
    """

    dim = DIM

    def __init__(self, vocab: set[str]):
        if len(vocab) > HALF:
            raise ValueError(f"fixture vocabulary of {len(vocab)} exceeds the {HALF} owned buckets")
        self.index = {token: i for i, token in enumerate(sorted(vocab))}

    def _bucket(self, token: str) -> int:
        owned = self.index.get(token)
        if owned is not None:
            return owned
        return HALF + zlib.crc32(token.encode("utf-8")) % (DIM - HALF)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                out[i, self._bucket(token)] += 1.0
        return normalize_rows(out)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def make_embedder() -> FakeEmbedder:
    """An embedder over the vocabulary of the built fixture stores."""
    return FakeEmbedder(fixture_vocabulary())


def entity(thread: int, site: str, title: str, question: str, answer: str, tags: list[str],
           created: int) -> dict:
    return {
        "entity/id": f"se-{site}-{thread}",
        "entity/source": f"{site}.stackexchange",
        "title": title,
        "question-body": question,
        "answer-body": answer,
        "tags": tags,
        "created": created,
    }


MATH_SE = [
    (1001, "Schwarz lemma for automorphisms of the unit disk",
     "<p>How do I use the Schwarz lemma to classify automorphisms of the disk? "
     "I applied the Schwarz lemma to the map but the Schwarz lemma gives no uniqueness.</p>",
     "<p>Compose with a disk automorphism; the Schwarz–Pick form settles it.</p>",
     ["complex-analysis", "schwarz-lemma"], 1_500_000_001),
    (1002, "Automorphisms of the disk",
     "<p>Why is every automorphism of the unit disk a Möbius map? A hint says the Schwarz lemma applies.</p>",
     "<p>Normalise a value and a derivative, then apply uniqueness.</p>",
     ["complex-analysis"], 1_500_000_002),
    (1003, "Polynomial bounds on the disk",
     "<p>Let p be a polynomial with sup norm 1. Is every polynomial of degree n bounded by a polynomial "
     "growth estimate on every smaller disk? My polynomial coefficient count seems off.</p>",
     "<p>Use the Bernstein inequality for the polynomial.</p>",
     ["polynomials", "inequality"], 1_500_000_003),
    (1004, "Rescaling the variable in an integral",
     "<p>After rescaling x to t/a the integral simplifies. Does the rescaling change the constant?</p>",
     "<p>No, the rescaling is absorbed by the Jacobian.</p>",
     ["integration"], 1_500_000_004),
    (1005, "Conformal maps between planar domains",
     "<p>When is there a conformal map from one simply connected planar domain onto another?</p>",
     "<p>Riemann mapping theorem; in higher dimension the quasiconformal analogue fails.</p>",
     ["complex-geometry"], 1_500_000_005),
    (1006, "Lemmas about holomorphic functions",
     "<p>Is there a lemma that bounds a holomorphic function by its value at one point?</p>",
     "<p>Yes, the maximum principle.</p>",
     ["complex-analysis"], 1_500_000_006),
    (1007, "Normal families and the disk algebra",
     "<p>Why is a uniformly bounded sequence of holomorphic maps into the disk a normal family?</p>",
     "<p>Montel's theorem.</p>",
     ["complex-analysis"], 1_500_000_007),
]
FILLER_WORDS = sorted(
    f"{stem}{ending}"
    for stem in ("amber", "basalt", "cinder", "dune", "estuary", "fjord", "garnet", "harbour",
                 "island", "juniper", "kelp", "lagoon", "marble", "nectar", "opal", "prairie",
                 "quarry", "reef", "slate", "tundra", "umber", "velvet", "willow", "xenon",
                 "yarrow", "zephyr")
    for ending in ("ly", "ness", "wise", "hood")
)
MATH_SE += [
    (1010 + i, f"Matrix question {words[0]}",
     f"<p>Consider a {words[0]} matrix, a {words[1]} form and a {words[2]} module. "
     f"Diagonalise the {words[0]} matrix and bound the {words[1]} norm.</p>",
     f"<p>Use the spectral theorem for {words[2]} modules.</p>",
     ["linear-algebra"], 1_500_000_100 + i)
    for i, words in enumerate(zip(FILLER_WORDS[:23], FILLER_WORDS[23:46], FILLER_WORDS[46:69]))
]
MATHOVERFLOW = [
    (2001, "Schwarz lemma in several variables",
     "<p>What replaces the Schwarz lemma for holomorphic maps of the polydisc?</p>",
     "<p>There are several generalisations, none as clean.</p>",
     ["cv.complex-variables"], 1_400_000_001),
    (2002, "Pickrell lemma for measures on a path space",
     "<p>Where does the Pickrell lemma on invariant measures fail for infinite dimensions?</p>",
     "<p>It fails for the full path space; see the restricted setting.</p>",
     ["measure-theory"], 1_400_000_002),
] + [
    (2010 + i, f"Category theory note {words[0]}",
     f"<p>Does the adjoint functor theorem apply to the category of {words[0]} diagrams "
     f"over a {words[1]} site with {words[2]} sheaves?</p>",
     f"<p>Check the solution set condition for {words[1]} sites.</p>",
     ["category-theory"], 1_400_000_100 + i)
    for i, words in enumerate(zip(FILLER_WORDS[69:79], FILLER_WORDS[79:89], FILLER_WORDS[89:99]))
]


def build(root: Path) -> list[Path]:
    """Write entities.json and embeddings.npy for each site; return the store directories."""
    made = []
    for site, rows in (("math", MATH_SE), ("mathoverflow", MATHOVERFLOW)):
        store = root / site
        store.mkdir(parents=True, exist_ok=True)
        entities = [entity(row[0], site, *row[1:]) for row in rows]
        (store / "entities.json").write_text(json_dump(entities), encoding="utf-8")
        vectors = make_embedder().encode([embedding_text(e) for e in entities])
        np.save(store / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
        made.append(store)
    return made


def json_dump(entities: list[dict]) -> str:
    import json

    # One object per line: readable in diffs, still a single JSON array.
    inner = ",\n ".join(json.dumps(e, ensure_ascii=False) for e in entities)
    return "[\n " + inner + "\n]\n"


if __name__ == "__main__":
    for path in build(Path(__file__).resolve().parent):
        print(path)
