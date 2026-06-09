#!/usr/bin/env python3
"""Emit differentiable-substrate policy-prior moves.

Two stages, mirroring code_diff_jax_pilot.py:
  futon6/.venv/bin/python scripts/diffsub_emit.py --embed
  futon5/.venv-tpg/bin/python scripts/diffsub_emit.py --jax

G2 v1 grain: mission + capability nodes. /tmp/scopes.json has no stable
per-scope ids, so scope-grain is deferred rather than fabricating endpoints.
The JAX adjacency is sparse N x k over BGE nearest-neighbour candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/joe/code")
HERE = ROOT / "futon6"
OUTDIR = HERE / "resources/differentiable-substrate"
EMB = OUTDIR / "emb.npy"
NODES = OUTDIR / "nodes.json"
SUMMARY = OUTDIR / "diffsub-jax-summary.json"
MOVES = HERE / "data/diffsub-moves.edn"
SCOPES_PATH = Path("/tmp/scopes.json")
CAPS_PATH = HERE / "data/capability-graph.json"
WHOLE_PATH = HERE / "data/mission-wholeness.edn"
PHYLO_PATH = HERE / "data/mission-phylogeny.edn"

KNN = int(os.environ.get("DIFFSUB_KNN", "20"))
TOP_K = int(os.environ.get("DIFFSUB_TOP_K", "24"))
BAND_CENTER = float(os.environ.get("DIFFSUB_BAND_CENTER", "0.55"))
BAND_WIDTH = float(os.environ.get("DIFFSUB_BAND_WIDTH", "0.18"))

FRONTIER_BINDERS = {"capability-scope", "pattern", "psr", "pur"}
CLASSW = {"mess": 1.00, "pipeline": 0.65, "alive": 0.40, "stub": 0.55}


def bare_mission(m: str) -> str:
    return m[2:] if m.startswith("M-") else m


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def mission_classes() -> dict[str, str]:
    text = WHOLE_PATH.read_text()
    return {m: c for m, c in re.findall(r':mission "M-([^"]+)" :class :(\w+)', text)}


def mission_generativity() -> dict[str, int]:
    text = PHYLO_PATH.read_text()
    block = re.search(r":generativity-index \{([^}]*)\}", text)
    if not block:
        return {}
    return {bare_mission(m): int(g) for m, g in re.findall(r'"(M-[^"]+)" (\d+)', block.group(1))}


def mission_doc_text(stem: str) -> str:
    for path in sorted(ROOT.glob(f"futon*/holes/missions/M-{stem}.md")):
        try:
            return path.read_text(errors="ignore")[:6000]
        except Exception:
            pass
    return stem


def scope_aggregates(scopes: list[dict]) -> dict[str, dict]:
    by_m: dict[str, dict] = defaultdict(lambda: {"n": 0, "det": 0, "binders": Counter()})
    for sc in scopes:
        m = sc.get("m")
        if not m:
            continue
        by_m[m]["n"] += 1
        if sc.get("det"):
            by_m[m]["det"] += 1
            by_m[m]["binders"][sc.get("binder", "scope")] += 1
    return by_m


def mission_metric(stem: str, agg: dict, cls: str) -> float:
    n = max(1.0, float(agg.get("n", 1)))
    det = float(agg.get("det", 0))
    frontier = sum(v for k, v in agg.get("binders", {}).items() if k in FRONTIER_BINDERS)
    cweight = CLASSW.get(cls, 0.45)
    per_scope = (0.18 * n + det + 0.30 * frontier) / n
    return per_scope * (0.70 + cweight)


def capability_metric(cap: str, info: dict, caps: dict) -> float:
    if info.get("claimed"):
        return 0.10
    parents = info.get("scope", [])
    claimed_parents = [p for p in parents if caps.get(p, {}).get("claimed")]
    if claimed_parents:
        return 0.35 if not info.get("frontier") else 0.55
    return 2.20 if info.get("frontier") else 1.40


def build_nodes() -> list[dict]:
    scopes = read_json(SCOPES_PATH)
    caps = read_json(CAPS_PATH)
    cls = mission_classes()
    gen = mission_generativity()
    by_m = scope_aggregates(scopes)
    nodes = []
    for stem, agg in sorted(by_m.items()):
        c = cls.get(stem, "neutral")
        binders = ", ".join(f"{k}:{v}" for k, v in agg["binders"].most_common(8))
        nodes.append({
            "id": f"scope/mission/{stem}",
            "kind": "mission",
            "mission": stem,
            "metric": mission_metric(stem, agg, c),
            "degree": int(agg["n"] + gen.get(stem, 0)),
            "class": c,
            "det": int(agg["det"]),
            "text": f"mission M-{stem} class {c} scopes {agg['n']} detached {agg['det']} binders {binders}\n{mission_doc_text(stem)}",
        })
    for cap, info in sorted(caps.items()):
        parents = info.get("scope", [])
        nodes.append({
            "id": f"scope/capability/{cap}",
            "kind": "capability",
            "cap": cap,
            "metric": capability_metric(cap, info, caps),
            "degree": len(parents) + len(info.get("minted_by", [])),
            "claimed": bool(info.get("claimed")),
            "frontier": bool(info.get("frontier")),
            "status": info.get("status"),
            "parents": parents,
            "text": f"capability {cap} status {info.get('status')} frontier {info.get('frontier')} claimed {info.get('claimed')} parents {' '.join(parents)} title {info.get('title', '')}",
        })
    return nodes


def do_embed() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    nodes = build_nodes()
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    vecs = model.encode([n["text"] for n in nodes], normalize_embeddings=True, show_progress_bar=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB, np.asarray(vecs, dtype=np.float32))
    NODES.write_text(json.dumps(nodes, indent=2))
    print(f"embedded {len(nodes)} mission/capability nodes -> {EMB} shape={np.asarray(vecs).shape}")


def softmax(xs: list[float]) -> list[float]:
    mx = max(xs) if xs else 0.0
    exps = [math.exp(x - mx) for x in xs]
    z = sum(exps) or 1.0
    return [x / z for x in exps]


def edn_float(x: float) -> str:
    return f"{float(x):.6g}"


def edn_move(d: dict) -> str:
    adv = f'"{d["adv"]}"' if d.get("adv") else "nil"
    term = "true" if d.get("terminal") else "false"
    note = str(d.get("note", "")).replace('"', "'")
    return (f'  {{:move/id "{d["have"]}->{d["want"]}" :move/class :{d["cls"]}'
            f' :have "{d["have"]}" :want "{d["want"]}" :advances-cap {adv}'
            f' :score {edn_float(d["score"])} :prior {edn_float(d["prior"])}'
            f' :delta-g {edn_float(d["delta_g"])} :confidence :{d["conf"]}'
            f' :rank {d["rank"]} :move/terminal? {term} :note "{note}"}}')


def emit_edn(moves: list[dict]) -> None:
    edn = (";; diffsub-moves.edn — M-differentiable-substrate gradient-scored move-set.\n"
           ";; Same locked shape as diffsub-moves-stub.edn; generated by diffsub_emit.py.\n"
           f"{{:emit/at {int(time.time())}\n"
           " :emit/metric {:compose :additive :epistemic :C-holes :pragmatic :cap-ascent :C-variant :salingaros}\n"
           f" :emit/k {len(moves)}\n"
           " :emit/stub? false\n"
           " :moves [\n" + "\n".join(edn_move(d) for d in moves) + "\n ]}\n")
    MOVES.write_text(edn)


def do_jax() -> None:
    import jax
    import jax.numpy as jnp
    from jax import grad

    emb = np.load(EMB)
    nodes = json.loads(NODES.read_text())
    fresh_nodes = {n["id"]: n for n in build_nodes()}
    nodes = [{**n, **{k: v for k, v in fresh_nodes.get(n["id"], {}).items() if k != "text"}}
             for n in nodes]
    caps = read_json(CAPS_PATH)
    n = len(nodes)
    k = min(KNN, max(1, n - 1))
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    cos = e @ e.T
    np.fill_diagonal(cos, -np.inf)
    cand = np.argsort(-cos, axis=1)[:, :k]
    cand_cos = np.take_along_axis(cos, cand, axis=1)
    metric = np.array([float(nd.get("metric", 0.0)) for nd in nodes], dtype=np.float32)
    degree = np.array([float(nd.get("degree", 0.0)) for nd in nodes], dtype=np.float32)
    node_by_id = {nd["id"]: i for i, nd in enumerate(nodes)}

    cap_goal = np.zeros(n, dtype=np.float32)
    for cap, info in caps.items():
        idx = node_by_id.get(f"scope/capability/{cap}")
        if idx is None or info.get("claimed"):
            continue
        if any(caps.get(p, {}).get("claimed") for p in info.get("scope", [])):
            cap_goal[idx] = 1.0 if info.get("frontier") else 0.7

    g_i = metric[:, None]
    g_j = metric[cand]
    goal_j = cap_goal[cand]
    band = np.exp(-((cand_cos - BAND_CENTER) / BAND_WIDTH) ** 2)
    descent = np.maximum(g_i - g_j, 0.0)
    sat_np = band * (1.0 + descent) + 1.4 * goal_j
    row_min = sat_np.min(axis=1, keepdims=True)
    row_max = sat_np.max(axis=1, keepdims=True)
    sat_np = (sat_np - row_min) / np.maximum(row_max - row_min, 1e-6)
    sat = jnp.asarray(sat_np)

    def loss(a):
        p = jax.nn.softmax(a, axis=1)
        return -jnp.mean(jnp.sum(p * sat, axis=1))

    g = grad(loss)
    a0 = jnp.zeros((n, k))
    g0 = g(a0)
    gnorm = np.asarray(jnp.sqrt(jnp.sum(g0 ** 2, axis=1)))
    med = float(np.median(gnorm)) or 1e-12
    corr = float(np.corrcoef(gnorm, degree)[0, 1]) if np.std(degree) > 0 and np.std(gnorm) > 0 else 0.0

    lr = float(os.environ.get("DIFFSUB_LR", "60.0"))
    steps = int(os.environ.get("DIFFSUB_STEPS", "300"))
    a = a0
    sat_before = float(-loss(a))
    for _ in range(steps):
        a = a - lr * g(a)
    sat_after = float(-loss(a))
    p0 = np.asarray(jax.nn.softmax(a0, axis=1))
    p1 = np.asarray(jax.nn.softmax(a, axis=1))
    gain = p1 - p0

    moves = []
    for cap, info in caps.items():
        if info.get("claimed"):
            continue
        idx = node_by_id.get(f"scope/capability/{cap}")
        if idx is None:
            continue
        parents = [p for p in info.get("scope", []) if caps.get(p, {}).get("claimed")]
        if parents:
            have = f"scope/capability/{parents[0]}"
            conf = "claimed-substrate"
            base = float(gnorm[idx])
        else:
            have = f"scope/conjectural/{cap}-foothold"
            conf = "conjectural"
            base = float(0.15 * gnorm[idx])
        score = max(1e-6, base * 1000.0)
        moves.append({"cls": "advance-capability", "have": have, "want": f"scope/capability/{cap}",
                      "adv": cap, "score": score, "conf": conf, "terminal": False,
                      "delta_g": -score * 0.08,
                      "note": f"gradient toward capability anchor: {info.get('title', '')[:80]}"})

    by_m = {nd["mission"]: nd for nd in nodes if nd["kind"] == "mission"}
    for nd in sorted(by_m.values(), key=lambda x: -x.get("det", 0))[:8]:
        if nd.get("det", 0) <= 0:
            continue
        i = node_by_id[nd["id"]]
        stem = nd["mission"]
        score = max(1e-6, float(gnorm[i]) * 900.0)
        moves.append({"cls": "close-hole", "have": f"scope/{stem}/detached#open",
                      "want": f"scope/{stem}/detached#closed", "adv": None,
                      "score": score, "conf": "claimed-substrate", "terminal": False,
                      "delta_g": -score * 0.08,
                      "note": f"{nd.get('det', 0)} detached holes in {stem} ({nd.get('class')})"})

    mess = [nd for nd in nodes if nd.get("kind") == "mission" and nd.get("class") == "mess" and nd.get("det", 0) > 0]
    if mess:
        nd = max(mess, key=lambda x: x.get("det", 0))
        i = node_by_id[nd["id"]]
        score = max(1e-6, float(gnorm[i]) * 650.0)
        stem = nd["mission"]
        moves.append({"cls": "centre-mess", "have": f"scope/{stem}/cluster",
                      "want": f"scope/{stem}/centred", "adv": None, "score": score,
                      "conf": "claimed-substrate", "terminal": True,
                      "delta_g": -score * 0.08,
                      "note": f"terminal v1 centre-mess for {stem}"})

    moves.sort(key=lambda d: -d["score"])
    moves = moves[:TOP_K]
    priors = softmax([m["score"] for m in moves])
    for rank, (m, p) in enumerate(zip(moves, priors), 1):
        m["rank"] = rank
        m["prior"] = p
        m["score"] = round(float(m["score"]), 6)
        m["delta_g"] = round(float(m["delta_g"]), 6)
    emit_edn(moves)

    cap_moves = [m for m in moves if m["cls"] == "advance-capability"]
    island_scores = [m["score"] for m in cap_moves if m["conf"] == "conjectural"]
    summit_scores = [m["score"] for m in cap_moves if m["conf"] == "claimed-substrate"]
    summary = {
        "grain": "mission+capability",
        "adjacency": f"sparse-knn N={n} k={k}",
        "n": n,
        "k": k,
        "sat_before": sat_before,
        "sat_after": sat_after,
        "gradnorm_min": float(gnorm.min()),
        "gradnorm_med": float(np.median(gnorm)),
        "gradnorm_max": float(gnorm.max()),
        "gradnorm_max_med_ratio": float(gnorm.max() / med),
        "corr_gradnorm_degree": corr,
        "summit_score_med": float(np.median(summit_scores)) if summit_scores else 0.0,
        "island_score_med": float(np.median(island_scores)) if island_scores else 0.0,
        "moves": len(moves),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"[G2] grain={summary['grain']} adjacency={summary['adjacency']}")
    print(f"[conditioning] grad-norm min={summary['gradnorm_min']:.3e} med={summary['gradnorm_med']:.3e} "
          f"max={summary['gradnorm_max']:.3e} max/med={summary['gradnorm_max_med_ratio']:.2f} "
          f"corr(degree)={corr:+.3f}")
    print(f"[loop] mean satisfaction {sat_before:.4f} -> {sat_after:.4f}")
    print(f"[anchors] summit-score-med={summary['summit_score_med']:.6f} island-score-med={summary['island_score_med']:.6f}")
    print(f"wrote {MOVES} — {len(moves)} moves")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--jax", action="store_true")
    args = ap.parse_args()
    if args.embed:
        do_embed()
    elif args.jax:
        do_jax()
    else:
        ap.error("use --embed then --jax")


if __name__ == "__main__":
    main()
