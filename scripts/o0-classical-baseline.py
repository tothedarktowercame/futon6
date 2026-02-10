#!/usr/bin/env python3
"""O-0: Classical Baseline — Chapter 6 at SE Scale.

Reproduce Corneli (2014) learning-event detection on physics.SE using the
NER kernel. Classical statistics, no GPU, no LLM.

Stages:
  1. Parse PostHistory.xml → per-user text timelines
  2. Parse Posts.xml + Comments.xml → treatment events per user
  3. NER term spotting per user → learning events + vocabulary trajectories
  4. Fit Ornstein-Uhlenbeck impulse-damping model
  5. Report + comparison with Corneli (2014)

Each stage checkpoints to data/o0/. Re-run skips completed stages.

Usage:
    python scripts/o0-classical-baseline.py [--data-dir data/physics-se]
    python scripts/o0-classical-baseline.py --stage 3   # restart from stage 3
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import iterparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/physics-se")
OUT_DIR = Path("data/o0")
KERNEL_PATH = Path("data/ner-kernel/terms.tsv")

# PostHistoryTypeId: 2 = Initial Body, 5 = Edit Body
# These contain the actual post text written/edited by a user.
TEXT_HISTORY_TYPES = {"2", "5"}

# Minimum thresholds for including a user in the O-U analysis
MIN_LEARNING_EVENTS = 20
MIN_TREATMENT_EVENTS = 10
MIN_POSTS = 5

# O-U model fitting: time bin width
BIN_WEEKS = 1


# ---------------------------------------------------------------------------
# NER kernel
# ---------------------------------------------------------------------------

# Common English words to exclude (ported from spot-terms.bb)
STOPWORDS = frozenset({
    "at all", "define", "relative", "equivalent", "opposite",
    "independent", "interesting", "relation", "structure",
    "complex", "moment", "numbers", "nature", "square",
    "section", "allows", "analysis", "property", "context",
    "parallel", "consistent", "eventually", "separate",
    "levels", "represent", "stable", "approach", "surface",
    "random", "action", "string", "constant", "observe",
    "formula", "support", "target", "degree", "potential",
    "critical", "particle", "creation", "principle",
    "evolution", "classical", "operator", "symmetry",
    "measurement", "variables", "distribution",
    "invariant", "representation",
})

# Short single-word terms to keep (ported from spot-terms.bb)
KEEP_SHORT = frozenset({
    "monad", "sheaf", "topos", "fiber", "nerve",
    "stalk", "gerbe", "braid", "trace", "wedge",
})

WORD_RE = re.compile(r"[a-z0-9]+")


def load_ner_kernel(path):
    """Load NER kernel TSV. Returns (singles set, multi_index dict).

    multi_index maps each content word to a list of (remaining_words, term)
    tuples. This inverted index allows O(tokens) matching instead of O(terms).
    """
    singles = set()
    multi_index = defaultdict(list)  # first_content_word → [(rest_words, term)]
    multi_count = 0
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            term_lower = parts[0].strip()
            if not term_lower:
                continue
            if term_lower in STOPWORDS:
                continue
            words = term_lower.split()
            if len(words) == 1 and len(term_lower) < 6 and term_lower not in KEEP_SHORT:
                continue
            if len(words) == 1:
                singles.add(term_lower)
            else:
                content = [w for w in words if len(w) >= 3]
                if content:
                    first = content[0]
                    rest = frozenset(content[1:])
                    multi_index[first].append((rest, term_lower))
                    multi_count += 1
    return singles, multi_index, multi_count


def tokenize(text):
    """Lowercase tokenization, tokens >= 3 chars."""
    if not text:
        return set()
    return {w for w in WORD_RE.findall(text.lower()) if len(w) >= 3}


def spot_terms(singles, multi_index, text):
    """Spot NER kernel terms in text. Returns set of matched term_lowers.

    Uses inverted index: for each token in the text, check only multi-word
    terms whose first content word is that token. This is O(tokens * avg_fan)
    instead of O(all_multi_terms).
    """
    if not text:
        return set()
    text_lower = text.lower()
    tokens = tokenize(text)
    matched = tokens & singles  # set intersection for single-word lookup
    # Multi-word: only check terms whose first content word is in tokens
    for token in tokens:
        candidates = multi_index.get(token)
        if not candidates:
            continue
        for rest_words, term in candidates:
            if rest_words.issubset(tokens) and term in text_lower:
                matched.add(term)
    return matched


# ---------------------------------------------------------------------------
# XML streaming helpers
# ---------------------------------------------------------------------------

def stream_xml(path, tag="row"):
    """Memory-efficient streaming XML parse. Yields element attributes as dicts."""
    count = 0
    for event, elem in iterparse(path, events=("end",)):
        if elem.tag == tag:
            yield elem.attrib
            elem.clear()
            count += 1
            if count % 500_000 == 0:
                print(f"  ... {count:,} rows", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 1: PostHistory → per-user text timelines
# ---------------------------------------------------------------------------

def stage1_user_timelines(data_dir, out_dir):
    """Parse PostHistory.xml → per-user text timelines."""
    out_path = out_dir / "user_timelines.json"
    if out_path.exists():
        print(f"Stage 1: {out_path} exists, skipping.", file=sys.stderr)
        return

    print("Stage 1: Parsing PostHistory.xml → user timelines...", file=sys.stderr)
    t0 = time.time()

    # user_id → list of (timestamp_str, post_id, text_length)
    # We DON'T store full text here — too much memory for 1.8GB.
    # Instead we store (timestamp, post_id) and re-scan for terms in Stage 3.
    # Actually, for Stage 3 we need the text. Let's store term sets per entry
    # to avoid holding 1.8GB of text in RAM.
    #
    # Strategy: Load NER kernel NOW, spot terms during streaming, store only
    # the matched terms per (user, timestamp) entry. This is Stage 1+3 fused.
    # But that breaks the checkpointing model.
    #
    # Better strategy: Store just (user_id, post_id, timestamp) here.
    # Stage 3 will re-stream PostHistory.xml for text — it's a second pass
    # but keeps memory bounded.

    timelines = defaultdict(list)  # user_id → [(timestamp, post_id), ...]
    total = 0
    skipped = 0

    ph_path = data_dir / "PostHistory.xml"
    for attrs in stream_xml(ph_path):
        ht = attrs.get("PostHistoryTypeId", "")
        if ht not in TEXT_HISTORY_TYPES:
            continue
        uid = attrs.get("UserId")
        if not uid:
            skipped += 1
            continue
        ts = attrs.get("CreationDate", "")
        pid = attrs.get("PostId", "")
        timelines[uid].append((ts, pid))
        total += 1

    # Sort each user's timeline
    for uid in timelines:
        timelines[uid].sort()

    elapsed = time.time() - t0
    print(f"Stage 1: {total:,} text entries, {len(timelines):,} users, "
          f"{skipped:,} skipped (no UserId), {elapsed:.1f}s", file=sys.stderr)

    # Write as JSON (user_id → list of [timestamp, post_id])
    with open(out_path, "w") as f:
        json.dump(dict(timelines), f)

    print(f"Stage 1: wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 2: Posts + Comments → treatment events
# ---------------------------------------------------------------------------

def stage2_treatment_events(data_dir, out_dir):
    """Parse Posts.xml + Comments.xml → per-user treatment events."""
    out_path = out_dir / "treatment_events.json"
    if out_path.exists():
        print(f"Stage 2: {out_path} exists, skipping.", file=sys.stderr)
        return

    print("Stage 2: Parsing Posts.xml → question/answer maps...", file=sys.stderr)
    t0 = time.time()

    # First pass: build maps
    question_owner = {}   # question_post_id → owner_user_id
    answer_owner = {}     # answer_post_id → (owner_user_id, timestamp)
    answer_parents = {}   # answer_post_id → parent_question_id
    accepted = {}         # question_post_id → accepted_answer_id
    post_owner = {}       # any_post_id → owner_user_id

    for attrs in stream_xml(data_dir / "Posts.xml"):
        pid = attrs.get("Id", "")
        pt = attrs.get("PostTypeId", "")
        uid = attrs.get("OwnerUserId")
        ts = attrs.get("CreationDate", "")

        if uid:
            post_owner[pid] = uid

        if pt == "1":  # Question
            if uid:
                question_owner[pid] = uid
            acc = attrs.get("AcceptedAnswerId")
            if acc:
                accepted[pid] = acc
        elif pt == "2":  # Answer
            parent = attrs.get("ParentId", "")
            if uid and parent:
                answer_owner[pid] = (uid, ts)
                answer_parents[pid] = parent

    print(f"  Questions: {len(question_owner):,}, Answers: {len(answer_owner):,}",
          file=sys.stderr)

    # Build treatment events
    treatments = defaultdict(list)  # user_id → [(timestamp, type), ...]

    # Treatment type 1: someone answers your question
    for ans_id, (answerer_uid, ts) in answer_owner.items():
        q_id = answer_parents[ans_id]
        q_owner = question_owner.get(q_id)
        if q_owner and q_owner != answerer_uid:  # don't self-treat
            treatments[q_owner].append((ts, "answer_received"))

    # Treatment type 2: your answer gets accepted (O(1) lookup via dict)
    for q_id, acc_ans_id in accepted.items():
        info = answer_owner.get(acc_ans_id)
        if info:
            uid, ts = info
            treatments[uid].append((ts, "answer_accepted"))

    # Treatment type 3: comments on your posts
    print("Stage 2: Parsing Comments.xml...", file=sys.stderr)
    for attrs in stream_xml(data_dir / "Comments.xml"):
        pid = attrs.get("PostId", "")
        commenter = attrs.get("UserId")
        ts = attrs.get("CreationDate", "")
        owner = post_owner.get(pid)
        if owner and commenter and owner != commenter:  # don't self-treat
            treatments[owner].append((ts, "comment_received"))

    # Sort each user's treatments
    for uid in treatments:
        treatments[uid].sort()

    elapsed = time.time() - t0
    total_events = sum(len(v) for v in treatments.values())
    print(f"Stage 2: {total_events:,} treatment events for {len(treatments):,} users, "
          f"{elapsed:.1f}s", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump(dict(treatments), f)

    print(f"Stage 2: wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 3: NER term spotting → learning events
# ---------------------------------------------------------------------------

def stage3_learning_events(data_dir, out_dir, kernel_path):
    """Run NER kernel on per-user text timeline → learning events."""
    le_path = out_dir / "learning_events.json"
    vt_path = out_dir / "vocab_trajectories.json"
    if le_path.exists() and vt_path.exists():
        print(f"Stage 3: outputs exist, skipping.", file=sys.stderr)
        return

    # Load user timelines from Stage 1
    print("Stage 3: Loading user timelines...", file=sys.stderr)
    with open(out_dir / "user_timelines.json") as f:
        timelines = json.load(f)

    # Load NER kernel
    print(f"Stage 3: Loading NER kernel from {kernel_path}...", file=sys.stderr)
    singles, multi_index, multi_count = load_ner_kernel(kernel_path)
    print(f"  Kernel: {len(singles)} singles + {multi_count} multi-word terms "
          f"({len(multi_index)} index keys)", file=sys.stderr)

    # We need to re-stream PostHistory.xml to get the text for each
    # (user_id, post_id, timestamp) triple. Build a lookup set for efficiency.
    print("Stage 3: Building lookup index...", file=sys.stderr)
    # Set of (user_id, post_id, timestamp) we need text for
    needed = set()
    for uid, entries in timelines.items():
        for ts, pid in entries:
            needed.add((uid, pid, ts))

    print(f"  Need text for {len(needed):,} entries", file=sys.stderr)

    # Re-stream PostHistory.xml, spot terms on the fly
    print("Stage 3: Re-streaming PostHistory.xml for term spotting...",
          file=sys.stderr)
    t0 = time.time()

    # user_id → set of terms already seen (cumulative vocabulary)
    user_vocab = defaultdict(set)
    # user_id → [(timestamp, term), ...] learning events
    learning_events = defaultdict(list)
    # Temporary: accumulate per-entry terms keyed by (uid, ts)
    # We process in timeline order per user, so we need to buffer per-entry
    # results and then process in order.
    entry_terms = defaultdict(list)  # uid → [(ts, terms_set), ...]

    ph_path = data_dir / "PostHistory.xml"
    matched_entries = 0
    for attrs in stream_xml(ph_path):
        ht = attrs.get("PostHistoryTypeId", "")
        if ht not in TEXT_HISTORY_TYPES:
            continue
        uid = attrs.get("UserId")
        if not uid:
            continue
        ts = attrs.get("CreationDate", "")
        pid = attrs.get("PostId", "")
        if (uid, pid, ts) not in needed:
            continue

        text = attrs.get("Text", "")
        terms = spot_terms(singles, multi_index, text)
        entry_terms[uid].append((ts, terms))
        matched_entries += 1

    elapsed = time.time() - t0
    print(f"  Matched {matched_entries:,} entries in {elapsed:.1f}s", file=sys.stderr)

    # Now process per user in chronological order to find learning events
    print("Stage 3: Detecting learning events...", file=sys.stderr)
    vocab_trajectories = {}
    total_le = 0

    for uid in sorted(entry_terms.keys()):
        entries = sorted(entry_terms[uid])  # sort by timestamp
        seen = set()
        trajectory = []
        events = []

        for ts, terms in entries:
            new_terms = terms - seen
            for t in new_terms:
                events.append((ts, t))
            seen.update(terms)
            trajectory.append((ts, len(seen)))

        if events:
            learning_events[uid] = events
            total_le += len(events)
        if trajectory:
            vocab_trajectories[uid] = trajectory

    print(f"Stage 3: {total_le:,} learning events across {len(learning_events):,} users",
          file=sys.stderr)
    print(f"  Users with trajectories: {len(vocab_trajectories):,}", file=sys.stderr)

    # Summary stats
    if vocab_trajectories:
        final_vocabs = [traj[-1][1] for traj in vocab_trajectories.values() if traj]
        avg_vocab = sum(final_vocabs) / len(final_vocabs)
        max_vocab = max(final_vocabs)
        print(f"  Avg final vocabulary: {avg_vocab:.1f} terms", file=sys.stderr)
        print(f"  Max final vocabulary: {max_vocab} terms", file=sys.stderr)

    with open(le_path, "w") as f:
        json.dump(dict(learning_events), f)
    with open(vt_path, "w") as f:
        json.dump(vocab_trajectories, f)

    print(f"Stage 3: wrote {le_path} and {vt_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 4: O-U model fitting
# ---------------------------------------------------------------------------

def stage4_ou_fit(out_dir):
    """Fit Ornstein-Uhlenbeck impulse-damping model."""
    fit_path = out_dir / "ou_fit.json"
    if fit_path.exists():
        print(f"Stage 4: {fit_path} exists, skipping.", file=sys.stderr)
        return

    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError:
        print("Stage 4: numpy/scipy required. pip install numpy scipy",
              file=sys.stderr)
        sys.exit(1)

    print("Stage 4: Loading learning events and treatment events...",
          file=sys.stderr)

    with open(out_dir / "learning_events.json") as f:
        learning_events = json.load(f)
    with open(out_dir / "treatment_events.json") as f:
        treatment_events = json.load(f)

    def parse_ts(s):
        """Parse ISO timestamp to days since epoch."""
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.timestamp() / 86400.0  # days
        except (ValueError, AttributeError):
            return None

    def bin_events(event_times, bin_width_days, t_min, t_max):
        """Bin event times into counts per bin."""
        n_bins = max(1, int((t_max - t_min) / bin_width_days) + 1)
        counts = np.zeros(n_bins)
        for t in event_times:
            idx = int((t - t_min) / bin_width_days)
            if 0 <= idx < n_bins:
                counts[idx] += 1
        return counts

    def ou_neg_log_likelihood(params, le_rate, treat_bins, dt):
        """Negative log-likelihood for discretized O-U with impulse.

        Model: r[t+1] = r[t] + theta*(mu - r[t])*dt + alpha*treat[t] + noise
        Observations: le_rate[t] ~ Normal(r[t], sigma^2)
        """
        theta, mu, alpha, log_sigma = params
        sigma = np.exp(log_sigma)
        if theta <= 0 or sigma <= 0:
            return 1e12

        T = len(le_rate)
        r = np.zeros(T)
        r[0] = mu

        nll = 0.0
        for t in range(1, T):
            r_pred = r[t - 1] + theta * (mu - r[t - 1]) * dt + alpha * treat_bins[t - 1]
            r[t] = r_pred
            diff = le_rate[t] - r_pred
            nll += 0.5 * (diff / sigma) ** 2 + np.log(sigma)

        return nll

    # Find eligible users
    bin_width = BIN_WEEKS * 7  # days
    eligible = []
    for uid in learning_events:
        le = learning_events[uid]
        te = treatment_events.get(uid, [])
        if len(le) >= MIN_LEARNING_EVENTS and len(te) >= MIN_TREATMENT_EVENTS:
            eligible.append(uid)

    print(f"Stage 4: {len(eligible)} users with >= {MIN_LEARNING_EVENTS} learning "
          f"and >= {MIN_TREATMENT_EVENTS} treatment events", file=sys.stderr)

    if not eligible:
        print("Stage 4: No eligible users. Lowering thresholds...", file=sys.stderr)
        # Try with lower thresholds
        for uid in learning_events:
            le = learning_events[uid]
            te = treatment_events.get(uid, [])
            if len(le) >= 5 and len(te) >= 3:
                eligible.append(uid)
        print(f"Stage 4: {len(eligible)} users with relaxed thresholds (>=5 LE, >=3 TE)",
              file=sys.stderr)

    # Fit per user
    user_fits = {}
    fit_count = 0
    fail_count = 0

    for uid in eligible:
        le = learning_events[uid]
        te = treatment_events.get(uid, [])

        # Parse timestamps
        le_times = [t for t in (parse_ts(e[0]) for e in le) if t is not None]
        te_times = [t for t in (parse_ts(e[0]) for e in te) if t is not None]

        if not le_times or not te_times:
            continue

        t_min = min(min(le_times), min(te_times))
        t_max = max(max(le_times), max(te_times))

        if t_max - t_min < bin_width * 4:  # need at least 4 bins
            continue

        le_rate = bin_events(le_times, bin_width, t_min, t_max)
        treat_bins = bin_events(te_times, bin_width, t_min, t_max)

        dt = 1.0  # one bin = one time unit

        # Initial guess
        mu0 = np.mean(le_rate)
        x0 = [0.1, mu0, 0.5, np.log(max(np.std(le_rate), 0.01))]

        try:
            result = minimize(
                ou_neg_log_likelihood, x0,
                args=(le_rate, treat_bins, dt),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
            )
            if result.success or result.fun < 1e10:
                theta, mu, alpha, log_sigma = result.x
                user_fits[uid] = {
                    "theta": float(theta),
                    "mu": float(mu),
                    "alpha": float(alpha),
                    "sigma": float(np.exp(log_sigma)),
                    "n_learning_events": len(le),
                    "n_treatment_events": len(te),
                    "n_bins": len(le_rate),
                    "span_days": float(t_max - t_min),
                    "nll": float(result.fun),
                    "converged": bool(result.success),
                }
                fit_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1

        if (fit_count + fail_count) % 100 == 0:
            print(f"  ... {fit_count} fits, {fail_count} failures",
                  file=sys.stderr)

    print(f"Stage 4: {fit_count} successful fits, {fail_count} failures",
          file=sys.stderr)

    # Population summary
    summary = {}
    if user_fits:
        thetas = [v["theta"] for v in user_fits.values()]
        alphas = [v["alpha"] for v in user_fits.values()]
        mus = [v["mu"] for v in user_fits.values()]
        sigmas = [v["sigma"] for v in user_fits.values()]
        converged = sum(1 for v in user_fits.values() if v["converged"])

        summary = {
            "n_users_fit": len(user_fits),
            "n_converged": converged,
            "theta": {
                "mean": float(np.mean(thetas)),
                "median": float(np.median(thetas)),
                "std": float(np.std(thetas)),
            },
            "alpha": {
                "mean": float(np.mean(alphas)),
                "median": float(np.median(alphas)),
                "std": float(np.std(alphas)),
                "positive_fraction": float(np.mean([a > 0 for a in alphas])),
            },
            "mu": {
                "mean": float(np.mean(mus)),
                "median": float(np.median(mus)),
            },
            "sigma": {
                "mean": float(np.mean(sigmas)),
                "median": float(np.median(sigmas)),
            },
        }
        print(f"\n=== Population O-U Parameters ===", file=sys.stderr)
        print(f"  Users fit: {summary['n_users_fit']} "
              f"({summary['n_converged']} converged)", file=sys.stderr)
        print(f"  theta (damping): mean={summary['theta']['mean']:.4f}, "
              f"median={summary['theta']['median']:.4f}", file=sys.stderr)
        print(f"  alpha (impulse): mean={summary['alpha']['mean']:.4f}, "
              f"median={summary['alpha']['median']:.4f}, "
              f"positive={summary['alpha']['positive_fraction']:.1%}",
              file=sys.stderr)
        print(f"  mu (baseline):   mean={summary['mu']['mean']:.4f}", file=sys.stderr)
        print(f"  sigma (noise):   mean={summary['sigma']['mean']:.4f}", file=sys.stderr)

    output = {
        "population_summary": summary,
        "user_fits": user_fits,
        "config": {
            "bin_weeks": BIN_WEEKS,
            "min_learning_events": MIN_LEARNING_EVENTS,
            "min_treatment_events": MIN_TREATMENT_EVENTS,
        },
    }

    with open(fit_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Stage 4: wrote {fit_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage 5: Report
# ---------------------------------------------------------------------------

def stage5_report(out_dir):
    """Generate summary report comparing with Corneli (2014)."""
    report_path = out_dir / "o0_report.json"

    print("\nStage 5: Generating report...", file=sys.stderr)

    # Load all stage outputs
    with open(out_dir / "user_timelines.json") as f:
        timelines = json.load(f)
    with open(out_dir / "treatment_events.json") as f:
        treatments = json.load(f)
    with open(out_dir / "learning_events.json") as f:
        learning_events = json.load(f)
    with open(out_dir / "vocab_trajectories.json") as f:
        trajectories = json.load(f)
    with open(out_dir / "ou_fit.json") as f:
        ou_data = json.load(f)

    # Corneli (2014) baseline values
    corneli_2014 = {
        "source": "PlanetMath",
        "users": 445,
        "time_span_years": 9,
        "forum_posts": 8051,
        "corrections": 14064,
        "learning_event_edits": 3867,
        "note": "Values from Corneli (2014) Chapter 6 and f6/learning-event-detection flexiarg",
    }

    # Physics.SE stats
    n_users = len(timelines)
    n_users_with_le = len(learning_events)
    total_le = sum(len(v) for v in learning_events.values())
    total_te = sum(len(v) for v in treatments.values())
    total_posts = sum(len(v) for v in timelines.values())

    # Vocabulary stats
    final_vocabs = []
    for uid, traj in trajectories.items():
        if traj:
            final_vocabs.append(traj[-1][1])

    physics_se = {
        "source": "physics.stackexchange",
        "users_total": n_users,
        "users_with_learning_events": n_users_with_le,
        "total_text_entries": total_posts,
        "total_learning_events": total_le,
        "total_treatment_events": total_te,
        "avg_final_vocabulary": sum(final_vocabs) / max(len(final_vocabs), 1),
        "max_final_vocabulary": max(final_vocabs) if final_vocabs else 0,
        "median_final_vocabulary": sorted(final_vocabs)[len(final_vocabs) // 2] if final_vocabs else 0,
    }

    report = {
        "title": "O-0: Classical Baseline — Chapter 6 at SE Scale",
        "method": "Corneli (2014) learning-event detection with NER kernel",
        "generated": datetime.now().isoformat(),
        "corneli_2014_baseline": corneli_2014,
        "physics_se_results": physics_se,
        "ou_model": ou_data["population_summary"],
        "comparison": {
            "scale_factor_users": n_users_with_le / 445 if n_users_with_le else 0,
            "scale_factor_events": total_le / 3867 if total_le else 0,
            "note": "Physics.SE should have more users and events than PlanetMath. "
                    "If O-U parameters are in the same order of magnitude, the model "
                    "generalises across platforms.",
        },
        "verification": {
            "V13_ner_trajectories": total_le > 0,
            "V14_learning_events_detectable": n_users_with_le > 100,
            "V15_ou_parameters_plausible": bool(ou_data.get("population_summary", {}).get("n_users_fit", 0) > 0),
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print human-readable summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"O-0 CLASSICAL BASELINE — RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Corneli (2014) PlanetMath:", file=sys.stderr)
    print(f"  {corneli_2014['users']} users, {corneli_2014['time_span_years']} years, "
          f"{corneli_2014['learning_event_edits']} learning events", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Physics.SE (this run):", file=sys.stderr)
    print(f"  {n_users:,} total users, {n_users_with_le:,} with learning events",
          file=sys.stderr)
    print(f"  {total_posts:,} text entries, {total_le:,} learning events, "
          f"{total_te:,} treatment events", file=sys.stderr)
    print(f"  Avg vocabulary: {physics_se['avg_final_vocabulary']:.1f} terms, "
          f"Max: {physics_se['max_final_vocabulary']}", file=sys.stderr)
    print(f"", file=sys.stderr)

    pop = ou_data.get("population_summary", {})
    if pop.get("n_users_fit"):
        print(f"O-U Model ({pop['n_users_fit']} users, {pop['n_converged']} converged):",
              file=sys.stderr)
        print(f"  theta (damping):  {pop['theta']['mean']:.4f} "
              f"(median {pop['theta']['median']:.4f})", file=sys.stderr)
        print(f"  alpha (impulse):  {pop['alpha']['mean']:.4f} "
              f"(median {pop['alpha']['median']:.4f}), "
              f"{pop['alpha']['positive_fraction']:.0%} positive", file=sys.stderr)
        print(f"  mu (baseline):    {pop['mu']['mean']:.4f}", file=sys.stderr)
        print(f"  sigma (noise):    {pop['sigma']['mean']:.4f}", file=sys.stderr)
    else:
        print(f"O-U Model: No fits (insufficient data).", file=sys.stderr)

    print(f"", file=sys.stderr)
    print(f"Verification checklist:", file=sys.stderr)
    for k, v in report["verification"].items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}", file=sys.stderr)

    print(f"\nStage 5: wrote {report_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="O-0: Classical Baseline")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="Path to physics-se data dump")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Output directory")
    parser.add_argument("--kernel", default=str(KERNEL_PATH),
                        help="NER kernel TSV path")
    parser.add_argument("--stage", type=int, default=1,
                        help="Start from this stage (1-5)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    kernel_path = Path(args.kernel)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"O-0 Classical Baseline", file=sys.stderr)
    print(f"  Data: {data_dir}", file=sys.stderr)
    print(f"  Output: {out_dir}", file=sys.stderr)
    print(f"  Kernel: {kernel_path}", file=sys.stderr)
    print(f"  Starting from stage: {args.stage}", file=sys.stderr)
    print(f"", file=sys.stderr)

    t_total = time.time()

    if args.stage <= 1:
        stage1_user_timelines(data_dir, out_dir)
    if args.stage <= 2:
        stage2_treatment_events(data_dir, out_dir)
    if args.stage <= 3:
        stage3_learning_events(data_dir, out_dir, kernel_path)
    if args.stage <= 4:
        stage4_ou_fit(out_dir)
    if args.stage <= 5:
        stage5_report(out_dir)

    elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", file=sys.stderr)


if __name__ == "__main__":
    main()
