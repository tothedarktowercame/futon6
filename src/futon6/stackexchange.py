"""StackExchange data processing pipeline.

Parses SE XML data dumps (Posts.xml, Tags.xml), extracts Q&A pairs with
mathematical content, computes embeddings, and outputs enriched entries
for F6 graph integration.

Designed for GPU-accelerated batch processing on large datasets.

SE data dump format: https://meta.stackexchange.com/q/2677
"""

import re
import html
from pathlib import Path
from xml.etree.ElementTree import iterparse
from dataclasses import dataclass, field, asdict


# --- Data structures ---

@dataclass
class SEPost:
    """A StackExchange post (question or answer)."""
    id: int
    post_type: str  # "question" or "answer"
    title: str = ""
    body: str = ""          # raw HTML
    body_text: str = ""     # stripped text
    body_latex: list = field(default_factory=list)  # extracted LaTeX fragments
    score: int = 0
    tags: list = field(default_factory=list)
    answer_count: int = 0
    accepted_answer_id: int | None = None
    parent_id: int | None = None  # for answers: the question ID
    creation_date: str = ""


@dataclass
class SEQAPair:
    """A question with its accepted/top answer."""
    question: SEPost
    answer: SEPost
    tags: list = field(default_factory=list)


# --- HTML / LaTeX extraction ---

_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$([^\$]+?)\$")
_SE_TAGS_RE = re.compile(r"<([^>]+)>")


def strip_html(html_str: str) -> str:
    """Remove HTML tags and decode entities."""
    text = _TAG_RE.sub(" ", html_str)
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_latex(html_str: str) -> list[str]:
    """Extract LaTeX fragments from SE post HTML.

    SE math sites use MathJax: $...$ for inline, $$...$$ for display.
    """
    text = html.unescape(html_str)
    fragments = []
    # Display math first (greedy over inline)
    for m in _LATEX_DISPLAY_RE.finditer(text):
        fragments.append(m.group(1).strip())
    # Inline math
    for m in _LATEX_INLINE_RE.finditer(text):
        frag = m.group(1).strip()
        # Skip if already captured as part of display math
        if frag and frag not in fragments:
            fragments.append(frag)
    return fragments


def parse_se_tags(tag_str: str) -> list[str]:
    """Parse SE tag string like '<algebra><group-theory>' into list."""
    if not tag_str:
        return []
    return _SE_TAGS_RE.findall(tag_str)


# --- XML streaming parser ---

def iter_posts(xml_path: str, min_score: int = 0):
    """Stream-parse Posts.xml yielding SEPost objects.

    Uses iterparse for constant memory usage regardless of file size.
    PostTypeId: 1 = question, 2 = answer.

    Args:
        xml_path: path to Posts.xml
        min_score: skip posts below this score (default 0 = keep all non-negative)
    """
    for event, elem in iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        attrs = elem.attrib
        post_type_id = int(attrs.get("PostTypeId", 0))
        if post_type_id not in (1, 2):
            elem.clear()
            continue

        score = int(attrs.get("Score", 0))
        if score < min_score:
            elem.clear()
            continue

        body_html = attrs.get("Body", "")
        post = SEPost(
            id=int(attrs["Id"]),
            post_type="question" if post_type_id == 1 else "answer",
            title=attrs.get("Title", ""),
            body=body_html,
            body_text=strip_html(body_html),
            body_latex=extract_latex(body_html),
            score=score,
            tags=parse_se_tags(attrs.get("Tags", "")),
            answer_count=int(attrs.get("AnswerCount", 0)),
            accepted_answer_id=int(a) if (a := attrs.get("AcceptedAnswerId")) else None,
            parent_id=int(p) if (p := attrs.get("ParentId")) else None,
            creation_date=attrs.get("CreationDate", ""),
        )
        yield post
        elem.clear()


def load_posts(xml_path: str, min_score: int = 0) -> dict[int, SEPost]:
    """Load all posts into a dict keyed by post ID."""
    return {p.id: p for p in iter_posts(xml_path, min_score)}


def build_qa_pairs(posts: dict[int, SEPost]) -> list[SEQAPair]:
    """Match questions with their accepted or highest-scored answer.

    Returns QA pairs where both question and answer exist.
    """
    questions = {pid: p for pid, p in posts.items() if p.post_type == "question"}

    # Group answers by parent question
    answers_by_q: dict[int, list[SEPost]] = {}
    for p in posts.values():
        if p.post_type == "answer" and p.parent_id:
            answers_by_q.setdefault(p.parent_id, []).append(p)

    pairs = []
    for qid, q in questions.items():
        candidates = answers_by_q.get(qid, [])
        if not candidates:
            continue

        # Prefer accepted answer, then highest score
        answer = None
        if q.accepted_answer_id and q.accepted_answer_id in posts:
            answer = posts[q.accepted_answer_id]
        else:
            answer = max(candidates, key=lambda a: a.score)

        pairs.append(SEQAPair(
            question=q,
            answer=answer,
            tags=q.tags,
        ))

    return pairs


# --- Conversion to F6 entities ---

def qa_to_entity(pair: SEQAPair) -> dict:
    """Convert a QA pair to an F6 entity dict.

    Shape matches planetmath.entries_to_entities() output.
    """
    q = pair.question
    a = pair.answer
    return {
        "entity/id": f"se-physics-{q.id}",
        "entity/type": "QAPair",
        "entity/source": "physics.stackexchange",
        "title": q.title,
        "question-body": q.body_text,
        "answer-body": a.body_text,
        "question-latex": q.body_latex,
        "answer-latex": a.body_latex,
        "tags": pair.tags,
        "score": q.score,
        "answer-score": a.score,
        "created": q.creation_date,
    }


def qa_to_relations(pair: SEQAPair) -> list[dict]:
    """Extract relations from a QA pair.

    Generates:
    - tagged-with relations (from SE tags)
    """
    q = pair.question
    relations = []
    for tag in pair.tags:
        relations.append({
            "from": f"se-physics-{q.id}",
            "to": f"se-tag-{tag}",
            "type": "tagged-with",
        })
    return relations


def tag_entities(pairs: list[SEQAPair]) -> list[dict]:
    """Create tag entities from all QA pairs."""
    tags = set()
    for pair in pairs:
        tags.update(pair.tags)
    return [
        {"entity/id": f"se-tag-{tag}", "entity/type": "SETag", "name": tag}
        for tag in sorted(tags)
    ]


# --- Batch embedding (GPU-ready) ---

def compute_qa_embeddings(
    pairs: list[SEQAPair],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str | None = None,
):
    """Compute embeddings for QA pairs.

    On GPU machines, set device="cuda" and increase batch_size.
    For multi-GPU, the model auto-distributes with device="cuda".

    Args:
        pairs: list of QA pairs
        model_name: sentence-transformers model. Options:
            - "all-MiniLM-L6-v2" (fast, 384d, good baseline)
            - "all-mpnet-base-v2" (better quality, 768d)
            - "BAAI/bge-large-en-v1.5" (strong, 1024d, needs GPU)
        batch_size: encoding batch size (256 for GPU, 32 for CPU)
        device: "cuda", "cpu", or None (auto-detect)

    Returns:
        numpy array of shape (len(pairs), embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    texts = []
    for pair in pairs:
        q = pair.question
        a = pair.answer
        # Combine question title + body + answer for embedding
        parts = [q.title]
        if q.body_text:
            parts.append(q.body_text[:500])  # truncate long bodies
        if a.body_text:
            parts.append(a.body_text[:500])
        texts.append(". ".join(parts))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


# --- Stats ---

def corpus_stats(pairs: list[SEQAPair]) -> dict:
    """Summary statistics for a SE corpus."""
    all_tags = set()
    total_latex = 0
    for pair in pairs:
        all_tags.update(pair.tags)
        total_latex += len(pair.question.body_latex) + len(pair.answer.body_latex)

    return {
        "qa_pairs": len(pairs),
        "unique_tags": len(all_tags),
        "total_latex_fragments": total_latex,
        "avg_q_score": (
            sum(p.question.score for p in pairs) / len(pairs) if pairs else 0
        ),
        "avg_a_score": (
            sum(p.answer.score for p in pairs) / len(pairs) if pairs else 0
        ),
        "with_latex": sum(
            1 for p in pairs
            if p.question.body_latex or p.answer.body_latex
        ),
    }
