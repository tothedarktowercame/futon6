#!/usr/bin/env python3
"""Extract deterministic StackExchange thread samples (Q&A + comments).

Produces four sets:
- math.stackexchange.com / category-theory
- mathoverflow.net / category-theory
- math.stackexchange.com / mathematical-physics
- mathoverflow.net / mathematical-physics

Each set is sampled deterministically from tagged questions.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from xml.etree.ElementTree import iterparse


SITE_TOPIC_TAGS: Dict[str, Dict[str, str]] = {
    "math.stackexchange.com": {
        "category-theory": "category-theory",
        "mathematical-physics": "mathematical-physics",
    },
    "mathoverflow.net": {
        "category-theory": "ct.category-theory",
        "mathematical-physics": "mp.mathematical-physics",
    },
}


TAG_RE = re.compile(r"<([^>]+)>")
HTML_RE = re.compile(r"<[^>]+>")


@dataclass
class Post:
    id: int
    post_type: str
    title: str
    body_html: str
    body_text: str
    score: int
    tags: List[str]
    creation_date: str
    parent_id: int | None = None
    accepted_answer_id: int | None = None
    owner_user_id: int | None = None


def parse_tags(tag_str: str) -> List[str]:
    if not tag_str:
        return []
    if "|" in tag_str:
        return [tag for tag in tag_str.split("|") if tag]
    return TAG_RE.findall(tag_str)


def strip_html(html_str: str) -> str:
    text = html.unescape(html_str or "")
    text = HTML_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_rank_key(seed: str, site: str, topic: str, qid: int) -> str:
    payload = f"{seed}|{site}|{topic}|{qid}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_eligible_questions(
    posts_xml: Path,
    topic_tags: Dict[str, str],
    min_question_score: int,
) -> Dict[str, List[int]]:
    eligible: Dict[str, List[int]] = {topic: [] for topic in topic_tags}

    for event, elem in iterparse(posts_xml, events=("end",)):
        if elem.tag != "row":
            continue

        attrs = elem.attrib
        if attrs.get("PostTypeId") != "1":
            elem.clear()
            continue

        score = int(attrs.get("Score", 0))
        answer_count = int(attrs.get("AnswerCount", 0))
        if score < min_question_score or answer_count <= 0:
            elem.clear()
            continue

        qid = int(attrs["Id"])
        tags = set(parse_tags(attrs.get("Tags", "")))
        for topic, topic_tag in topic_tags.items():
            if topic_tag in tags:
                eligible[topic].append(qid)

        elem.clear()

    return eligible


def load_posts_for_questions(
    posts_xml: Path,
    qids: Iterable[int],
) -> tuple[Dict[int, Post], Dict[int, List[Post]]]:
    qid_set = set(qids)
    questions: Dict[int, Post] = {}
    answers_by_qid: Dict[int, List[Post]] = {}

    for event, elem in iterparse(posts_xml, events=("end",)):
        if elem.tag != "row":
            continue

        attrs = elem.attrib
        post_type_id = attrs.get("PostTypeId")
        if post_type_id == "1":
            qid = int(attrs["Id"])
            if qid in qid_set:
                body_html = attrs.get("Body", "")
                questions[qid] = Post(
                    id=qid,
                    post_type="question",
                    title=attrs.get("Title", ""),
                    body_html=body_html,
                    body_text=strip_html(body_html),
                    score=int(attrs.get("Score", 0)),
                    tags=parse_tags(attrs.get("Tags", "")),
                    creation_date=attrs.get("CreationDate", ""),
                    accepted_answer_id=(
                        int(attrs["AcceptedAnswerId"])
                        if attrs.get("AcceptedAnswerId")
                        else None
                    ),
                    owner_user_id=(
                        int(attrs["OwnerUserId"])
                        if attrs.get("OwnerUserId")
                        else None
                    ),
                )
        elif post_type_id == "2":
            parent_id = int(attrs["ParentId"]) if attrs.get("ParentId") else None
            if parent_id and parent_id in qid_set:
                aid = int(attrs["Id"])
                body_html = attrs.get("Body", "")
                answer = Post(
                    id=aid,
                    post_type="answer",
                    title="",
                    body_html=body_html,
                    body_text=strip_html(body_html),
                    score=int(attrs.get("Score", 0)),
                    tags=[],
                    creation_date=attrs.get("CreationDate", ""),
                    parent_id=parent_id,
                    owner_user_id=(
                        int(attrs["OwnerUserId"])
                        if attrs.get("OwnerUserId")
                        else None
                    ),
                )
                answers_by_qid.setdefault(parent_id, []).append(answer)

        elem.clear()

    for qid, answers in answers_by_qid.items():
        answers.sort(key=lambda post: (-post.score, post.id))

    return questions, answers_by_qid


def load_comments(
    comments_xml: Path,
    needed_post_ids: set[int],
) -> Dict[int, List[dict]]:
    comments_by_post: Dict[int, List[dict]] = {}

    for event, elem in iterparse(comments_xml, events=("end",)):
        if elem.tag != "row":
            continue

        attrs = elem.attrib
        post_id = int(attrs.get("PostId", 0))
        if post_id in needed_post_ids:
            comment = {
                "id": int(attrs["Id"]),
                "post_id": post_id,
                "score": int(attrs.get("Score", 0)),
                "creation_date": attrs.get("CreationDate", ""),
                "user_id": int(attrs["UserId"]) if attrs.get("UserId") else None,
                "text": attrs.get("Text", ""),
            }
            comments_by_post.setdefault(post_id, []).append(comment)

        elem.clear()

    for post_id in comments_by_post:
        comments_by_post[post_id].sort(key=lambda c: c["id"])

    return comments_by_post


def select_qids(
    site: str,
    topic: str,
    candidate_qids: List[int],
    sample_size: int,
    seed: str,
) -> List[int]:
    ranked = sorted(
        candidate_qids,
        key=lambda qid: (stable_rank_key(seed, site, topic, qid), qid),
    )
    return ranked[:sample_size]


def question_url(site: str, qid: int) -> str:
    return f"https://{site}/questions/{qid}"


def answer_url(site: str, aid: int) -> str:
    return f"https://{site}/a/{aid}"


def build_thread_record(
    site: str,
    topic: str,
    topic_tag: str,
    qid: int,
    selection_rank: int,
    question: Post,
    answers: List[Post],
    comments_by_post: Dict[int, List[dict]],
    seed: str,
) -> dict:
    answer_dicts = []
    answer_comments: Dict[str, List[dict]] = {}
    for answer in answers:
        answer_dicts.append(
            {
                "id": answer.id,
                "url": answer_url(site, answer.id),
                "parent_id": answer.parent_id,
                "score": answer.score,
                "creation_date": answer.creation_date,
                "owner_user_id": answer.owner_user_id,
                "body_html": answer.body_html,
                "body_text": answer.body_text,
            }
        )
        answer_comments[str(answer.id)] = comments_by_post.get(answer.id, [])

    q_comments = comments_by_post.get(qid, [])
    total_comment_count = len(q_comments) + sum(
        len(answer_comments[str(answer.id)]) for answer in answers
    )

    return {
        "thread_id": f"{site}:{qid}",
        "site": site,
        "topic": topic,
        "topic_tag": topic_tag,
        "selection": {
            "seed": seed,
            "rank": selection_rank,
        },
        "question": {
            "id": question.id,
            "url": question_url(site, question.id),
            "title": question.title,
            "score": question.score,
            "creation_date": question.creation_date,
            "owner_user_id": question.owner_user_id,
            "tags": question.tags,
            "accepted_answer_id": question.accepted_answer_id,
            "body_html": question.body_html,
            "body_text": question.body_text,
        },
        "answers": answer_dicts,
        "comments": {
            "question": q_comments,
            "answers": answer_comments,
            "total": total_comment_count,
        },
    }


def process_site(
    site: str,
    topic_tags: Dict[str, str],
    posts_xml: Path,
    comments_xml: Path,
    sample_size: int,
    min_question_score: int,
    seed: str,
) -> tuple[dict, Dict[str, List[dict]]]:
    print(f"[{site}] pass 1: scanning eligible questions...", file=sys.stderr)
    eligible_by_topic = scan_eligible_questions(
        posts_xml=posts_xml,
        topic_tags=topic_tags,
        min_question_score=min_question_score,
    )

    all_qids = sorted({qid for qids in eligible_by_topic.values() for qid in qids})
    print(
        f"[{site}] pass 1: {len(all_qids)} unique eligible questions",
        file=sys.stderr,
    )
    for topic, qids in eligible_by_topic.items():
        print(f"[{site}]   {topic}: {len(qids)} tagged questions", file=sys.stderr)

    print(f"[{site}] pass 2: loading posts for eligible questions...", file=sys.stderr)
    questions, answers_by_qid = load_posts_for_questions(posts_xml, all_qids)

    with_answers_by_topic: Dict[str, List[int]] = {}
    selected_by_topic: Dict[str, List[int]] = {}
    for topic in topic_tags:
        with_answers = [
            qid for qid in eligible_by_topic[topic]
            if qid in questions and answers_by_qid.get(qid)
        ]
        with_answers_by_topic[topic] = with_answers
        if len(with_answers) < sample_size:
            raise ValueError(
                f"{site}/{topic}: only {len(with_answers)} threads with answers, "
                f"need {sample_size}"
            )
        selected_by_topic[topic] = select_qids(
            site=site,
            topic=topic,
            candidate_qids=with_answers,
            sample_size=sample_size,
            seed=seed,
        )
        print(
            f"[{site}]   selected {len(selected_by_topic[topic])} for {topic}",
            file=sys.stderr,
        )

    needed_post_ids: set[int] = set()
    for topic in topic_tags:
        for qid in selected_by_topic[topic]:
            needed_post_ids.add(qid)
            for answer in answers_by_qid.get(qid, []):
                needed_post_ids.add(answer.id)

    print(f"[{site}] pass 3: loading comments...", file=sys.stderr)
    comments_by_post = load_comments(comments_xml, needed_post_ids)
    print(
        f"[{site}] pass 3: comments loaded for {len(comments_by_post)} posts",
        file=sys.stderr,
    )

    records_by_topic: Dict[str, List[dict]] = {}
    summary = {
        "site": site,
        "posts_xml": str(posts_xml),
        "comments_xml": str(comments_xml),
        "eligible_questions": {topic: len(eligible_by_topic[topic]) for topic in topic_tags},
        "eligible_with_answers": {topic: len(with_answers_by_topic[topic]) for topic in topic_tags},
        "selected_threads": {},
    }

    for topic, topic_tag in topic_tags.items():
        selected_qids = selected_by_topic[topic]
        records: List[dict] = []
        for rank, qid in enumerate(selected_qids, start=1):
            records.append(
                build_thread_record(
                    site=site,
                    topic=topic,
                    topic_tag=topic_tag,
                    qid=qid,
                    selection_rank=rank,
                    question=questions[qid],
                    answers=answers_by_qid[qid],
                    comments_by_post=comments_by_post,
                    seed=seed,
                )
            )
        records_by_topic[topic] = records
        summary["selected_threads"][topic] = len(records)

    return summary, records_by_topic


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--se-root",
        type=Path,
        default=Path("/home/joe/code/futon6/se-data"),
        help="Directory containing site dump folders with Posts.xml and Comments.xml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stackexchange-samples"),
        help="Where extracted JSONL files and manifest are written",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Threads per site/topic",
    )
    parser.add_argument(
        "--min-question-score",
        type=int,
        default=0,
        help="Minimum question score to consider for sampling",
    )
    parser.add_argument(
        "--seed",
        default="futon5-se-sample-v1",
        help="Deterministic seed used for hash-ranking",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output directory contents",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory {args.output_dir} is non-empty. "
            "Use --overwrite to replace files."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "min_question_score": args.min_question_score,
        "source_root": str(args.se_root),
        "sets": [],
    }

    for site, topic_tags in SITE_TOPIC_TAGS.items():
        site_dir = args.se_root / site
        posts_xml = site_dir / "Posts.xml"
        comments_xml = site_dir / "Comments.xml"
        if not posts_xml.exists() or not comments_xml.exists():
            raise FileNotFoundError(
                f"Missing required files for {site}: {posts_xml} / {comments_xml}"
            )

        summary, records_by_topic = process_site(
            site=site,
            topic_tags=topic_tags,
            posts_xml=posts_xml,
            comments_xml=comments_xml,
            sample_size=args.sample_size,
            min_question_score=args.min_question_score,
            seed=args.seed,
        )

        for topic, records in records_by_topic.items():
            out_file = args.output_dir / f"{site}__{topic}.jsonl"
            write_jsonl(out_file, records)
            comment_total = sum(record["comments"]["total"] for record in records)
            answer_total = sum(len(record["answers"]) for record in records)
            manifest["sets"].append(
                {
                    "site": site,
                    "topic": topic,
                    "topic_tag": SITE_TOPIC_TAGS[site][topic],
                    "threads": len(records),
                    "answers": answer_total,
                    "comments": comment_total,
                    "output_file": str(out_file),
                    "eligible_tagged_questions": summary["eligible_questions"][topic],
                    "eligible_questions_with_answers": summary["eligible_with_answers"][topic],
                }
            )

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote manifest: {manifest_path}", file=sys.stderr)
    for item in manifest["sets"]:
        print(
            f"{item['site']} / {item['topic']}: "
            f"{item['threads']} threads, {item['answers']} answers, {item['comments']} comments",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
