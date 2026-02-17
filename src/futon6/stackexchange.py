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
class SEComment:
    """A StackExchange comment on a post."""
    id: int
    post_id: int       # which post (Q or A) this is on
    text: str = ""
    score: int = 0
    creation_date: str = ""
    user_id: int | None = None


@dataclass
class SEThread:
    """A full thread: question + ALL answers + comments."""
    question: SEPost
    answers: list[SEPost] = field(default_factory=list)
    comments: dict[int, list[SEComment]] = field(default_factory=dict)  # post_id -> comments
    tags: list[str] = field(default_factory=list)


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
    """Parse SE tag string into list.

    Handles both formats:
    - Angle-bracket: '<algebra><group-theory>'
    - Pipe-delimited: '|algebra|group-theory|'
    """
    if not tag_str:
        return []
    # Pipe-delimited (real SE dumps)
    if "|" in tag_str:
        return [t for t in tag_str.split("|") if t]
    # Angle-bracket (older format / some dumps)
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
    """Load all posts into a dict keyed by post ID.

    WARNING: holds all posts in memory. For large dumps use
    build_qa_pairs_streaming() instead.
    """
    return {p.id: p for p in iter_posts(xml_path, min_score)}


def build_qa_pairs(posts: dict[int, SEPost]) -> list[SEQAPair]:
    """Match questions with their accepted or highest-scored answer.

    Returns QA pairs where both question and answer exist.
    WARNING: requires all posts in memory. For large dumps use
    build_qa_pairs_streaming() instead.
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


def build_qa_pairs_streaming(xml_path: str, min_score: int = 0,
                             question_limit: int | None = None,
                             shard_index: int | None = None,
                             num_shards: int | None = None) -> list[SEQAPair]:
    """Build QA pairs with two streaming passes over the XML.

    Pass 1: collect only pairing metadata (IDs, scores, accepted answer).
             ~50 bytes per post instead of ~2KB.
    Pass 2: stream again, loading only posts that are part of a pair.

    Memory: O(pairs) not O(all posts). Safe on 4GB machines for multi-GB dumps.

    Shard mode: if shard_index and num_shards are set, only keep questions
    where question_id % num_shards == shard_index.
    """
    import sys

    # --- Pass 1: collect pairing metadata ---
    # For questions: {qid: (accepted_answer_id, tags)}
    questions: dict[int, tuple[int | None, list[str]]] = {}
    # For answers: {parent_id: [(answer_id, score)]}
    answers_by_q: dict[int, list[tuple[int, int]]] = {}

    for event, elem in iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue
        attrs = elem.attrib
        post_type_id = int(attrs.get("PostTypeId", 0))
        score = int(attrs.get("Score", 0))

        if post_type_id == 1 and score >= min_score:
            qid = int(attrs["Id"])
            if num_shards is not None and (qid % num_shards) != shard_index:
                elem.clear()
                continue
            acc = int(a) if (a := attrs.get("AcceptedAnswerId")) else None
            tags = parse_se_tags(attrs.get("Tags", ""))
            questions[qid] = (acc, tags)
        elif post_type_id == 2 and score >= min_score:
            parent = int(p) if (p := attrs.get("ParentId")) else None
            if parent:
                answers_by_q.setdefault(parent, []).append(
                    (int(attrs["Id"]), score))
        elem.clear()

    # Decide which answer to pick for each question
    # {qid: answer_id_to_load}
    pair_map: dict[int, int] = {}
    for qid, (accepted, _tags) in questions.items():
        candidates = answers_by_q.get(qid, [])
        if not candidates:
            continue
        if accepted and any(aid == accepted for aid, _ in candidates):
            pair_map[qid] = accepted
        else:
            pair_map[qid] = max(candidates, key=lambda x: x[1])[0]

    # Apply question limit if requested
    if question_limit is not None and len(pair_map) > question_limit:
        pair_map = dict(list(pair_map.items())[:question_limit])

    # IDs we need to load in pass 2
    needed_ids = set(pair_map.keys()) | set(pair_map.values())

    print(f"       Pass 1: {len(questions)} questions, "
          f"{sum(len(v) for v in answers_by_q.values())} answers, "
          f"{len(pair_map)} pairs to load", file=sys.stderr)

    # Free pass-1 structures we no longer need
    del answers_by_q

    # --- Pass 2: load only the posts we need ---
    loaded: dict[int, SEPost] = {}

    for event, elem in iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue
        attrs = elem.attrib
        pid = int(attrs.get("Id", 0))
        if pid not in needed_ids:
            elem.clear()
            continue

        post_type_id = int(attrs.get("PostTypeId", 0))
        body_html = attrs.get("Body", "")
        loaded[pid] = SEPost(
            id=pid,
            post_type="question" if post_type_id == 1 else "answer",
            title=attrs.get("Title", ""),
            body=body_html,
            body_text=strip_html(body_html),
            body_latex=extract_latex(body_html),
            score=int(attrs.get("Score", 0)),
            tags=parse_se_tags(attrs.get("Tags", "")),
            answer_count=int(attrs.get("AnswerCount", 0)),
            accepted_answer_id=int(a) if (a := attrs.get("AcceptedAnswerId")) else None,
            parent_id=int(p) if (p := attrs.get("ParentId")) else None,
            creation_date=attrs.get("CreationDate", ""),
        )
        elem.clear()

        # Stop early if we have everything
        if len(loaded) == len(needed_ids):
            break

    print(f"       Pass 2: loaded {len(loaded)}/{len(needed_ids)} posts",
          file=sys.stderr)

    # --- Assemble pairs ---
    pairs = []
    for qid, aid in pair_map.items():
        q = loaded.get(qid)
        a = loaded.get(aid)
        if q and a:
            pairs.append(SEQAPair(
                question=q, answer=a,
                tags=questions[qid][1],
            ))

    return pairs


# --- Comments XML streaming parser ---

def iter_comments(xml_path: str):
    """Stream-parse Comments.xml yielding SEComment objects.

    Uses iterparse for constant memory usage.
    """
    for event, elem in iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        attrs = elem.attrib
        comment = SEComment(
            id=int(attrs["Id"]),
            post_id=int(attrs.get("PostId", 0)),
            text=attrs.get("Text", ""),
            score=int(attrs.get("Score", 0)),
            creation_date=attrs.get("CreationDate", ""),
            user_id=int(u) if (u := attrs.get("UserId")) else None,
        )
        yield comment
        elem.clear()


# --- Thread construction (3-pass streaming) ---

def build_threads_streaming(
    posts_xml_path: str,
    comments_xml_path: str | None = None,
    min_score: int = 0,
    thread_limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[SEThread]:
    """Build full thread objects with a 3-pass streaming approach.

    Pass 1 (Posts.xml): metadata index — question IDs, answer-to-question
            mappings, keep ALL answers (not just accepted/top).
    Pass 2 (Posts.xml): load full post bodies for indexed posts.
    Pass 3 (Comments.xml): stream comments, keep only those on indexed posts.

    Assembles SEThread objects, one per question.

    Memory: O(threads) not O(all posts). Safe for large dumps.

    Shard mode: if shard_index and num_shards are set, only keep questions
    where question_id % num_shards == shard_index.
    """
    import sys

    # --- Pass 1: collect thread metadata ---
    # {qid: (accepted_answer_id, tags)}
    questions: dict[int, tuple[int | None, list[str]]] = {}
    # {answer_id: (parent_qid, score)}
    answer_to_q: dict[int, int] = {}
    # {qid: [answer_id, ...]}
    answers_by_q: dict[int, list[int]] = {}

    for event, elem in iterparse(posts_xml_path, events=("end",)):
        if elem.tag != "row":
            continue
        attrs = elem.attrib
        post_type_id = int(attrs.get("PostTypeId", 0))
        score = int(attrs.get("Score", 0))

        if post_type_id == 1 and score >= min_score:
            qid = int(attrs["Id"])
            if num_shards is not None and (qid % num_shards) != shard_index:
                elem.clear()
                continue
            acc = int(a) if (a := attrs.get("AcceptedAnswerId")) else None
            tags = parse_se_tags(attrs.get("Tags", ""))
            questions[qid] = (acc, tags)
        elif post_type_id == 2:
            # Keep ALL answers (no min_score filter on answers — thread completeness)
            parent = int(p) if (p := attrs.get("ParentId")) else None
            if parent and parent in questions or parent:
                aid = int(attrs["Id"])
                answer_to_q[aid] = parent
                answers_by_q.setdefault(parent, []).append(aid)
        elem.clear()

    # Apply thread_limit
    if thread_limit and len(questions) > thread_limit:
        # Keep only the first thread_limit questions (by ID order)
        kept_qids = sorted(questions.keys())[:thread_limit]
        questions = {qid: questions[qid] for qid in kept_qids}

    # Filter answers to only those belonging to kept questions
    answer_to_q = {aid: qid for aid, qid in answer_to_q.items()
                   if qid in questions}
    answers_by_q = {qid: aids for qid, aids in answers_by_q.items()
                    if qid in questions}

    # All post IDs we need to load
    needed_ids = set(questions.keys()) | set(answer_to_q.keys())

    print(f"       Pass 1: {len(questions)} questions, "
          f"{len(answer_to_q)} answers, "
          f"{len(needed_ids)} posts to load", file=sys.stderr)

    # --- Pass 2: load full post bodies ---
    loaded: dict[int, SEPost] = {}

    for event, elem in iterparse(posts_xml_path, events=("end",)):
        if elem.tag != "row":
            continue
        attrs = elem.attrib
        pid = int(attrs.get("Id", 0))
        if pid not in needed_ids:
            elem.clear()
            continue

        post_type_id = int(attrs.get("PostTypeId", 0))
        body_html = attrs.get("Body", "")
        loaded[pid] = SEPost(
            id=pid,
            post_type="question" if post_type_id == 1 else "answer",
            title=attrs.get("Title", ""),
            body=body_html,
            body_text=strip_html(body_html),
            body_latex=extract_latex(body_html),
            score=int(attrs.get("Score", 0)),
            tags=parse_se_tags(attrs.get("Tags", "")),
            answer_count=int(attrs.get("AnswerCount", 0)),
            accepted_answer_id=int(a) if (a := attrs.get("AcceptedAnswerId")) else None,
            parent_id=int(p) if (p := attrs.get("ParentId")) else None,
            creation_date=attrs.get("CreationDate", ""),
        )
        elem.clear()

        if len(loaded) == len(needed_ids):
            break

    print(f"       Pass 2: loaded {len(loaded)}/{len(needed_ids)} posts",
          file=sys.stderr)

    # --- Pass 3: load comments (if available) ---
    comments_by_post: dict[int, list[SEComment]] = {}

    if comments_xml_path and Path(comments_xml_path).exists():
        comment_count = 0
        for comment in iter_comments(comments_xml_path):
            if comment.post_id in needed_ids:
                comments_by_post.setdefault(comment.post_id, []).append(comment)
                comment_count += 1
        print(f"       Pass 3: {comment_count} comments on "
              f"{len(comments_by_post)} posts", file=sys.stderr)
    else:
        print(f"       Pass 3: no Comments.xml provided, skipping",
              file=sys.stderr)

    # --- Assemble threads ---
    threads = []
    for qid in sorted(questions.keys()):
        q = loaded.get(qid)
        if not q:
            continue

        _, tags = questions[qid]
        answer_ids = answers_by_q.get(qid, [])
        answer_posts = [loaded[aid] for aid in answer_ids if aid in loaded]
        # Sort answers by score descending
        answer_posts.sort(key=lambda a: a.score, reverse=True)

        # Collect comments for this thread (question + all answers)
        thread_comments: dict[int, list[SEComment]] = {}
        if qid in comments_by_post:
            thread_comments[qid] = comments_by_post[qid]
        for aid in answer_ids:
            if aid in comments_by_post:
                thread_comments[aid] = comments_by_post[aid]

        threads.append(SEThread(
            question=q,
            answers=answer_posts,
            comments=thread_comments,
            tags=tags,
        ))

    print(f"       Assembled {len(threads)} threads", file=sys.stderr)
    return threads


# --- Conversion to F6 entities ---

def _site_slug(site: str | None) -> str:
    """Return stable site slug used in SE entity IDs."""
    if not site:
        return "stackexchange"
    return site.split(".")[0]


def make_se_entity_id(question_id: int, site: str = "physics.stackexchange") -> str:
    """Build canonical SE entity ID for a question."""
    return f"se-{_site_slug(site)}-{question_id}"


def qa_to_entity(pair: SEQAPair, site: str = "physics.stackexchange",
                 entity_id: str | None = None) -> dict:
    """Convert a QA pair to an F6 entity dict.

    Shape matches planetmath.entries_to_entities() output.
    """
    q = pair.question
    a = pair.answer
    eid = entity_id or make_se_entity_id(q.id, site)
    return {
        "entity/id": eid,
        "entity/type": "QAPair",
        "entity/source": site,
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


def qa_to_relations(pair: SEQAPair, site: str = "physics.stackexchange",
                    entity_id: str | None = None) -> list[dict]:
    """Extract relations from a QA pair.

    Generates:
    - tagged-with relations (from SE tags)
    """
    q = pair.question
    source_id = entity_id or make_se_entity_id(q.id, site)
    relations = []
    for tag in pair.tags:
        relations.append({
            "from": source_id,
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
