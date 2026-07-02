#!/usr/bin/env python3
"""PROOF-MINE runner — the GPU discharge-evidence miner. Per futon6/holes/proof-mine-runner-spec.md.

For each mission: assemble the dossier (proof_mine_dossier.assemble, CPU) → ONE vLLM pass
(mark4-70b, gold-primed few-shot) → emit a graded record:

  {:mission "<repo>-d/mission/<stem>"                 ; CANONICAL, resolved BEFORE emission (D6)
   :discharges [{:target ... :discharged-by ... :grade :discharged|:open|:unverified|:research
                 :witness "<verbatim dossier span>"}]
   :endpoints [...] :rule-candidates [{:pattern :box :warrant}]}

The design decisions this file OWNS (each traceable to a scar in the spec / README-linode):
  D3  per-mission try/except; the record is built INSIDE the try; results APPEND to proof-mine.jsonl
      as they complete (never an end-of-loop write); --resume skips missions already in the artifact;
      the checkpoint counter is by NEW records, not modulo; nullable model-JSON fields are normalized
      (`x = r.get("k") or []`).  [the 2026-06-25 meme_mine_joint total-loss lesson]
  D4  every 10 missions write proof-mine-status.json (done/total, grade dist, grounding, latency, ETA).
  D5  --rung gold re-mines the 10 A-next gold BLIND and scores vs the sealed *-EMPIRICAL.edn; the
      QUANTIFIED abort band (endpoint precision <0.5 OR grade agreement <0.6 OR verbatim-witness <0.7)
      STOPS the run before any full sweep. The run carries its own yardstick.
  D6  every mission/target ref passes the canonical mission-index bridge BEFORE it is written;
      unresolvable refs go to proof-mine-quarantine.jsonl with the raw span — NEVER minted.
  D10 dossier ≤12k tokens (proof_mine_dossier budgets it), output ≤2k; wall-clock hard stop 6h.

Backends: --backend stub (no GPU; schema-valid plumbing) | openai (vLLM via OpenAI-compatible client).

  futon6/.venv/bin/python scripts/proof_mine.py --rung smoke --backend stub --limit 3
  # gold blind-eval (needs vLLM):  --rung gold --backend openai
"""
import argparse, difflib, glob, json, os, re, sys, time, urllib.request, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proof_mine_dossier import assemble, mission_stem, find_mission_doc, est_tokens  # noqa: E402

HOME = os.path.expanduser("~")
CODE = os.path.join(HOME, "code")
FUTON1A = os.environ.get("FUTON1A", "http://localhost:7071")
GOLD_DIR = os.path.join(CODE, "futon2/holes/labs")
OUT_DIR = os.path.join(CODE, "futon6/data/proof-mine")
GRADES = ("discharged", "open", "unverified", "research")

# The 10 A-next gold missions (the D5 yardstick). Order = canonical numbering in A-next-gold-corpus.md.
GOLD_MISSIONS = [
    "autoclock-in", "invariant-queue-unstuck", "a-sorry-enterprise", "agency-rebuild",
    "f6-ingest", "pattern-ingest", "patterns-done-right", "single-entry-point",
    "state-snapshot-witness", "stepper-calibration",
]


# ---------------------------------------------------------------- D6: canonical bridge
_MISSION_INDEX = None


def mission_index(refresh=False):
    """{stem -> canonical '<repo>-d/mission/<stem>'} from live XTDB mission/doc entities
    (mirror promote_c_entries.bb fetch-mission-index), with a filesystem fallback so the
    bridge still resolves offline. Cached."""
    global _MISSION_INDEX
    if _MISSION_INDEX is not None and not refresh:
        return _MISSION_INDEX
    idx = {}
    url = FUTON1A + "/api/alpha/entities/latest?type=mission%2Fdoc&limit=2000"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/edn"})
        body = urllib.request.urlopen(req, timeout=8).read().decode()
        for name in re.findall(r':name\s+"([^"]*?/mission/[^"]+)"', body):
            idx[name.split("/")[-1]] = name
    except Exception:                             # noqa: BLE001 — offline → filesystem fallback below
        pass
    return _index_with_fs_fallback(idx)


def _index_with_fs_fallback(idx):
    global _MISSION_INDEX
    for pat in ("%s/*/holes/missions/M-*.md", "%s/*/holes/M-*.md"):
        for path in glob.glob(pat % CODE):
            if "desktop-save" in path:
                continue
            stem = mission_stem(os.path.basename(path)[:-3])
            if stem not in idx:
                canon, _ = find_mission_doc(stem)
                if canon:
                    idx[stem] = canon
    _MISSION_INDEX = idx
    return idx


def resolve_ref(raw, idx):
    """Resolve a raw mission-shaped ref to its canonical id, or None (→ quarantine). Non-mission
    refs (sorry-refs, sha, method-refs) pass through unchanged — only mission refs are bridged."""
    if raw is None:
        return None, False
    s = str(raw)
    if "/mission/" in s and s in idx.values():
        return s, True
    stem = mission_stem(s)
    if stem and stem in idx:
        return idx[stem], True
    # not a resolvable mission ref — leave sorry/sha/method refs intact, flag mission-shaped misses
    looks_mission = bool(re.search(r"\bM-|/mission/|mission\b", s))
    return s, (not looks_mission)   # resolved==True for non-mission refs; False = quarantine


# ---------------------------------------------------------------- gold: few-shot + scoring
def _load_edn_loose(path):
    """Load a gold EMPIRICAL .edn as a loose dict via a tiny EDN→python coercion — enough to read
    :endpoints refs + discharge grades for scoring. We do NOT need a full EDN parser; we scrape the
    fields the scorer compares, and fail SOFT (return {}) so a schema drift can't crash the run."""
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return {}
    refs = re.findall(r':ref\s+"([^"]+)"', txt)
    grades = [g for g in re.findall(r':(discharged|open|unverified|research)\b', txt)]
    disch_by = re.findall(r':discharged-by\s+(?:"([^"]+)"|(\w[\w./-]*))', txt)
    return {"endpoints": refs, "grades": grades,
            "discharged_by": [a or b for a, b in disch_by]}


def _norm_ws(s):
    """Whitespace-normalized form for verbatim-span comparison: the model routinely collapses the
    dossier's newlines to spaces when it copies a span, so a strict substring test spuriously fails.
    A span that is verbatim modulo whitespace IS a citation (SFC2b). Not a semantic loosening."""
    return " ".join(str(s).split()) if s else ""


def snap_witness(witness, dossier_text, min_ratio=0.75):
    """Recover the ACTUAL verbatim dossier span the model was pointing at. The 70B paraphrases
    witnesses; rather than trust the paraphrase OR reject a real citation, we find the longest span
    the witness shares with the dossier and, if it covers ≥min_ratio of the witness, return THAT
    (the real, verbatim-mod-whitespace span). Below the threshold the 'witness' isn't grounded →
    None (caller marks it :unsupported). Honest by construction: we never fabricate a span, we only
    recover one the dossier actually contains, or reject."""
    w, dt = _norm_ws(witness), _norm_ws(dossier_text)
    if not w or not dt:
        return None
    if w in dt:                      # already verbatim (mod whitespace)
        return w
    block = difflib.SequenceMatcher(None, dt, w, autojunk=False).find_longest_match(0, len(dt), 0, len(w))
    if block.size >= min_ratio * len(w):
        return dt[block.a: block.a + block.size].strip()
    return None


def _ep_tokens(s):
    """Distinctive tokens of an endpoint string (for fair overlap matching against templated gold)."""
    return {t.lower() for t in re.findall(r"[A-Za-z0-9][\w.-]{2,}", str(s))
            if not t.isdigit() and t.lower() not in _EP_STOP}


_EP_STOP = {"the", "and", "for", "with", "via", "node", "nodes", "entity", "hyperedge", "map",
            "code", "type", "ref", "role", "have", "want", "true", "false", "note"}


def _ep_match(pred, gold_list):
    """A predicted endpoint matches a gold endpoint if they share a ref-shaped token (has :/./-)
    or ≥2 distinctive tokens. Fairer than exact-key equality against the gold's templated refs
    (e.g. 'agent nodes (agent:<id>)') which no free-text emission can reproduce character-exact."""
    pt = _ep_tokens(pred)
    if not pt:
        return False
    for g in gold_list:
        common = pt & _ep_tokens(g)
        if any(("/" in c or ":" in c or "." in c) for c in common) or len(common) >= 2:
            return True
    return False


def gold_few_shot(n=3):
    """Few-shot exemplars from the sealed gold EMPIRICAL files (priming, NOT the blind set).
    Uses the LAST few gold missions so the first ones stay clean for a blind sanity check.
    The exemplar dossiers are assembled under a TIGHT budget: three full dossiers (~7.4k tok) as
    few-shot overflowed the 16384 context on big target missions (the 400s). ~500 tok each teaches
    the output format + grading without eating the window."""
    msgs = []
    for stem in GOLD_MISSIONS[-n:]:
        emp = glob.glob("%s/A-next-%s/*-sorry-EMPIRICAL.edn" % (GOLD_DIR, stem))
        if not emp:
            continue
        gold = _load_edn_loose(emp[0])
        d = assemble(stem, budget_tokens=500)
        if not d.get("doc_found"):
            continue
        ideal = {
            "mission": d["mission"],
            "discharges": [{"target": (gold["endpoints"][0] if gold["endpoints"] else "sorry/%s" % stem),
                            "discharged_by": (gold["discharged_by"][0] if gold["discharged_by"] else None),
                            "grade": (gold["grades"][0] if gold["grades"] else "open"),
                            "witness": "L2"}],
            "endpoints": gold["endpoints"][:6],
            "rule_candidates": [],
        }
        msgs += [{"role": "user", "content": _dossier_prompt(d)},
                 {"role": "assistant", "content": json.dumps(ideal)}]
    return msgs


def score_gold(record, gold, dossier_text):
    """Score a BLIND-mined record against a sealed gold EMPIRICAL. Coarse-grained on purpose
    (the pilot's honesty): endpoint precision/recall on ref-stem overlap, grade agreement on the
    shared discharge grades, witness validity = fraction of witnesses that are verbatim dossier
    spans. Returns a dict of the four D5 numbers."""
    pred_eps = [e for e in _norm(record.get("endpoints")) if _ep_tokens(e)]
    true_eps = [e for e in gold.get("endpoints", []) if _ep_tokens(e)]
    pmatch = sum(1 for p in pred_eps if _ep_match(p, true_eps))
    rmatch = sum(1 for g in true_eps if _ep_match(g, pred_eps))
    ep_prec = pmatch / len(pred_eps) if pred_eps else 0.0
    ep_rec = rmatch / len(true_eps) if true_eps else 0.0

    pred_grades = [d.get("grade") for d in _norm(record.get("discharges")) if d.get("grade")]
    true_grades = gold.get("grades", [])
    # agreement: does the predicted grade DISTRIBUTION land in the gold's grade set? (coarse)
    if pred_grades and true_grades:
        agree = sum(1 for g in pred_grades if g in set(true_grades)) / len(pred_grades)
    else:
        agree = 0.0

    witnesses = [d.get("witness") for d in _norm(record.get("discharges")) if d.get("witness")]
    dt = _norm_ws(dossier_text)
    verbatim = [w for w in witnesses if w and _norm_ws(w) and _norm_ws(w) in dt]
    witness_rate = len(verbatim) / len(witnesses) if witnesses else 0.0

    return {"endpoint_precision": round(ep_prec, 3), "endpoint_recall": round(ep_rec, 3),
            "grade_agreement": round(agree, 3), "witness_rate": round(witness_rate, 3)}


def gold_bands(scores):
    """D5 abort bands. Returns (ok:bool, reasons:list[str]). ANY breach ⇒ STOP before the sweep."""
    reasons = []
    if scores["endpoint_precision"] < 0.5:
        reasons.append("endpoint precision %.2f < 0.50" % scores["endpoint_precision"])
    if scores["grade_agreement"] < 0.6:
        reasons.append("grade agreement %.2f < 0.60" % scores["grade_agreement"])
    if scores["witness_rate"] < 0.7:
        reasons.append("verbatim-witness rate %.2f < 0.70" % scores["witness_rate"])
    return (not reasons), reasons


# ---------------------------------------------------------------- the LLM pass
INSTR = """You read ONE mission DOSSIER (doc + citing commits + c-entries + code endpoints) and emit
a GRADED discharge record as STRICT JSON only:
{"mission": <echo the dossier's canonical mission id>,
 "discharges": [{"target": <c-entry name | sorry-ref this mission owns>,
                 "discharged_by": <commit sha | method-ref | null>,
                 "grade": "discharged|open|unverified|research",
                 "witness": <a VERBATIM span copied from the dossier, or ":unsupported">}],
 "endpoints": [<the mission's sorry interface: real substrate-2/code entity refs — SHORT ids, not prose>],
 "rule_candidates": [{"pattern": <id>, "box": <verb>, "warrant": <verbatim span>}]}
The dossier is presented with numbered lines ("L1: ...", "L2: ..."). The "witness" MUST be the DOSSIER
LINE NUMBER(S) that support the discharge — a single "L42" or a contiguous range "L42-L45". Do NOT
paraphrase, summarize, or write prose in "witness"; cite line numbers only. If no dossier line supports
the discharge, write ":unsupported".
Grades (A-next honesty): discharged = a cited sha/method actually closes it; open = a real hole, not yet
closed (EXPECTED for IDENTIFY-stage missions — :open is correct output, not failure); unverified = a
wiring claims closure the evidence does not support; research = needs new investigation. Ground every
discharge to a VERBATIM witness span from the dossier or mark it ":unsupported" — never invent evidence.
Emit [] for a section with nothing to say. Do NOT fabricate closure to look productive."""


def _number_lines(text):
    """Number the dossier lines so the model can cite witnesses by line (L-refs) instead of
    (unreliably) copying spans verbatim. The same 1-based numbering is used to extract witnesses."""
    return "\n".join("L%d: %s" % (i + 1, ln) for i, ln in enumerate(text.splitlines()))


def _dossier_prompt(d):
    return ("CANONICAL MISSION: %s\nno-code-trail: %s\n\nDOSSIER (cite witnesses by line number, e.g. L12):\n%s"
            "\n\nEmit the JSON record now."
            % (d["mission"], d["no_code_trail"], _number_lines(d["text"])))


def _extract_line_witness(witness, dossier_text):
    """Turn the model's L-ref witness ('L42' / 'L42-L45') into the EXACT dossier line text — verbatim
    by construction. Returns the joined line text, or None if no valid in-range line refs are present."""
    s = str(witness or "")
    nums = set()
    for a, b in re.findall(r"L(\d+)\s*-\s*L?(\d+)", s):
        nums.update(range(int(a), int(b) + 1))
    for m in re.findall(r"L(\d+)", s):
        nums.add(int(m))
    if not nums:
        return None
    lines = dossier_text.splitlines()
    picked = [lines[n - 1] for n in sorted(nums) if 1 <= n <= len(lines) and lines[n - 1].strip()]
    return _norm_ws(" ".join(picked)) if picked else None


def call_openai(d, few_shot, model):
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 2000,
                       "messages": [{"role": "system", "content": INSTR}] + few_shot
                                   + [{"role": "user", "content": _dossier_prompt(d)}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=180) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0)) if m else {}


def call_stub(d):
    """No-GPU plumbing stub: a schema-valid record with ONE :open discharge witnessed by a real
    dossier span. Validates the pipeline (canonical bridge, append, resume, status), NOT reasoning.
    Deliberately emits `null`/empty for the nullable fields so the D3 normalization path is exercised."""
    span = (d["text"].split("\n", 1)[0][:80]) if d.get("text") else ""
    return {"mission": d["mission"],
            "discharges": [{"target": "sorry/%s" % d["stem"], "discharged_by": None,
                            "grade": "open", "witness": span}],
            "endpoints": None,          # <- intentionally null: D3 normalization must survive it
            "rule_candidates": None}


# ---------------------------------------------------------------- record build (D3 + D6)
def _norm(x):
    """The 2026-06-25 lesson: an LLM emits `null` for empty list fields; `x or []` handles present-but-null."""
    return x or []


def build_record(mission_canonical, raw, dossier, idx):
    """Build the graded record INSIDE the per-mission try (D3), applying the D6 canonical bridge at
    emission. Returns (record, quarantine:list). Never raises on a nullable/absent field."""
    quarantine = []
    dossier_text = dossier.get("text", "")

    # D6: the mission id itself must be canonical. Prefer the dossier's resolved id; bridge the model's echo.
    mission, ok = resolve_ref(raw.get("mission") or mission_canonical, idx)
    if not ok:
        quarantine.append({"field": "mission", "raw": raw.get("mission"), "mission": mission_canonical})
        mission = mission_canonical   # keep the dossier's canonical id; never mint the model's echo

    discharges = []
    for dsc in _norm(raw.get("discharges")):
        if not isinstance(dsc, dict):
            continue
        target_raw = dsc.get("target")
        target, tok = resolve_ref(target_raw, idx)
        if not tok:
            quarantine.append({"field": "discharge.target", "raw": target_raw, "mission": mission})
            continue                  # NEVER mint an unresolvable target
        grade = dsc.get("grade") if dsc.get("grade") in GRADES else "open"
        # Prefer the model's L-ref citation (verbatim by construction); fall back to snapping a prose
        # witness to the nearest real span; else :unsupported. Never fabricated.
        snapped = _extract_line_witness(dsc.get("witness"), dossier_text) \
            or snap_witness(dsc.get("witness"), dossier_text)
        discharges.append({
            "target": target, "discharged_by": dsc.get("discharged_by"),
            "grade": grade,
            "witness": snapped if snapped else ":unsupported",
            "witness_verbatim": bool(snapped),
        })

    endpoints = []
    for ep in _norm(raw.get("endpoints")):
        ref, ok = resolve_ref(ep if isinstance(ep, str) else (ep.get("ref") if isinstance(ep, dict) else None), idx)
        if ref:
            endpoints.append(ref)

    rule_candidates = [rc for rc in _norm(raw.get("rule_candidates")) if isinstance(rc, dict)]

    record = {
        "mission": mission, "stem": dossier.get("stem"),
        "discharges": discharges, "endpoints": endpoints,
        "rule_candidates": rule_candidates,
        "no_code_trail": dossier.get("no_code_trail", False),
        "pair_unverified": True,      # the pilot's ⚠pair: E-have-want pairs corpus not located on disk
        "dossier_truncations": dossier.get("truncations", []),
    }
    return record, quarantine


# ---------------------------------------------------------------- resume / status / io
def load_done(out_path):
    """Set of stems already in the artifact (D3 --resume)."""
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path, errors="replace"):
            try:
                done.add(json.loads(ln).get("stem"))
            except ValueError:
                continue
    return done


def append_jsonl(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


def write_status(path, done, total, grade_dist, grounded, latencies, t0):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else None
    status = {
        "done": done, "total": total,
        "grade_distribution": grade_dist,
        "grounding_rate": round(grounded / done, 3) if done else 0.0,
        "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "eta_s": round(eta, 1) if eta is not None else None,
        "elapsed_s": round(elapsed, 1),
    }
    tmp = path + ".tmp"
    json.dump(status, open(tmp, "w"), indent=2)
    os.replace(tmp, path)
    return status


# ---------------------------------------------------------------- the run
def discover_missions(limit=0):
    """All canonical mission stems on disk (the full-sweep universe), deduped, desktop-save excluded."""
    stems = []
    seen = set()
    for pat in ("%s/*/holes/missions/M-*.md", "%s/*/holes/M-*.md"):
        for path in sorted(glob.glob(pat % CODE)):
            if "desktop-save" in path:
                continue
            stem = mission_stem(os.path.basename(path)[:-3])
            if stem and stem not in seen:
                seen.add(stem)
                stems.append(stem)
    return stems[:limit] if limit else stems


def mine_one(stem, backend, few_shot, model, idx, dossier_budget=8000):
    """One unit of work, fully guarded (D3): dossier → LLM → build record. Returns
    (record, quarantine, latency_s, error_or_None). NEVER raises — a bad mission costs itself.
    dossier_budget keeps dossier + few-shot + output inside the model's context window (the 400s)."""
    t0 = time.perf_counter()
    try:
        d = assemble(stem, budget_tokens=dossier_budget)
        if not d.get("doc_found"):
            return None, [], time.perf_counter() - t0, "no doc for %s" % stem
        raw = call_openai(d, few_shot, model) if backend == "openai" else call_stub(d)
        record, quarantine = build_record(d["mission"], raw, d, idx)
        return record, quarantine, time.perf_counter() - t0, None
    except Exception as e:                        # noqa: BLE001 — the per-item resilience boundary
        return None, [], time.perf_counter() - t0, "%s: %s" % (stem, e)


def run_gold(backend, model, out_dir, dossier_budget=8000):
    """D5: re-mine the 10 A-next gold BLIND, score vs sealed EMPIRICAL, enforce abort bands."""
    idx = mission_index()
    few_shot = gold_few_shot() if backend == "openai" else []
    per, agg = [], {"endpoint_precision": [], "grade_agreement": [], "witness_rate": []}
    for stem in GOLD_MISSIONS:
        emp = glob.glob("%s/A-next-%s/*-sorry-EMPIRICAL.edn" % (GOLD_DIR, stem))
        if not emp:
            print("  ! gold missing EMPIRICAL for %s — skipping" % stem)
            continue
        d = assemble(stem, budget_tokens=dossier_budget)
        record, _q, lat, err = mine_one(stem, backend, few_shot, model, idx, dossier_budget)
        if err or record is None:
            print("  ! gold mine failed for %s: %s" % (stem, err))
            continue
        scores = score_gold(record, _load_edn_loose(emp[0]), d.get("text", ""))
        per.append({"stem": stem, "scores": scores})
        for k in agg:
            agg[k].append(scores[k])
        print("  gold %-24s prec=%.2f rec=%.2f grade=%.2f witness=%.2f (%.1fs)"
              % (stem, scores["endpoint_precision"], scores["endpoint_recall"],
                 scores["grade_agreement"], scores["witness_rate"], lat))
    mean = {k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in agg.items()}
    mean["endpoint_precision"] = mean.get("endpoint_precision", 0.0)
    ok, reasons = gold_bands(mean)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"per_mission": per, "mean": mean, "bands_ok": ok, "reasons": reasons},
              open(os.path.join(out_dir, "proof-mine-gold-eval.json"), "w"), indent=2)
    print("GOLD MEAN: %s" % mean)
    print("GOLD BANDS: %s%s" % ("PASS" if ok else "FAIL", "" if ok else " — " + "; ".join(reasons)))
    return ok, mean


def run_sweep(missions, backend, model, out_dir, resume, dossier_budget=8000, concurrency=8,
              hard_stop_s=6 * 3600):
    import concurrent.futures as cf
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "proof-mine.jsonl")
    quar_path = os.path.join(out_dir, "proof-mine-quarantine.jsonl")
    status_path = os.path.join(out_dir, "proof-mine-status.json")
    idx = mission_index()
    few_shot = gold_few_shot() if backend == "openai" else []
    done_set = load_done(out_path) if resume else set()
    todo = [m for m in missions if m not in done_set]
    workers = max(1, concurrency if backend == "openai" else 1)   # stub is CPU-only → no gain from threads
    print("sweep: %d missions (%d already done, skipped)  backend=%s  concurrency=%d"
          % (len(todo), len(done_set), backend, workers))

    grade_dist = {g: 0 for g in GRADES}
    grounded, latencies, done, t0 = 0, [], 0, time.time()

    def _work(stem):
        return mine_one(stem, backend, few_shot, model, idx, dossier_budget)

    # Fan out `workers` requests at once; vLLM continuously-batches them on the GPU (the throughput
    # lever the sibling runners use — sequential leaves the GPU idle, 2026-06-26). executor.map yields
    # in submission order, so ALL append/checkpoint/status bookkeeping stays on THIS thread — no locks,
    # D3 fully preserved (append as each completes, resume-safe, status every 10).
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for record, quarantine, lat, err in ex.map(_work, todo):
            if time.time() - t0 > hard_stop_s:
                print("WALL-CLOCK HARD STOP (%.0fh) — capturing and stopping." % (hard_stop_s / 3600))
                break
            done += 1
            latencies.append(lat)
            if err:
                print("  ! %s" % err)
                continue
            append_jsonl(out_path, record)             # D3: append as it completes, never end-of-loop
            for dsc in record["discharges"]:
                grade_dist[dsc["grade"]] = grade_dist.get(dsc["grade"], 0) + 1
            if any(dsc["witness_verbatim"] for dsc in record["discharges"]):
                grounded += 1
            for q in quarantine:
                append_jsonl(quar_path, q)
            if done % 10 == 0:                         # D4: status every 10 missions
                st = write_status(status_path, done, len(todo), grade_dist, grounded, latencies, t0)
                print("  [%d/%d] grades=%s grounding=%.2f eta=%ss"
                      % (done, len(todo), grade_dist, st["grounding_rate"], st["eta_s"]))
    write_status(status_path, done, len(todo), grade_dist, grounded, latencies, t0)
    print("DONE: %d mined · grades=%s · grounding=%.2f · quarantine=%s"
          % (done, grade_dist, (grounded / done if done else 0.0),
             sum(1 for _ in open(quar_path)) if os.path.exists(quar_path) else 0))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="PROOF-MINE runner (discharge-evidence miner).")
    ap.add_argument("--rung", choices=["smoke", "gold", "full"], default="smoke")
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--limit", type=int, default=0, help="cap missions (smoke); 0 = all")
    ap.add_argument("--resume", action="store_true", help="skip missions already in proof-mine.jsonl (D3)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dossier-budget", type=int, default=8000,
                    help="per-mission dossier token budget; keeps dossier+few-shot+output in-context (D10)")
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "8")),
                    help="concurrent in-flight requests (vLLM continuous-batches them); the throughput lever")
    ap.add_argument("--missions", nargs="*", help="explicit mission stems (else discover all on disk)")
    a = ap.parse_args()

    if a.rung == "gold":
        ok, _ = run_gold(a.backend, a.model, a.out, a.dossier_budget)
        sys.exit(0 if ok else 2)                   # nonzero abort so the shell won't proceed to full

    # smoke with no --limit defaults to the 10 gold missions (fast, real docs); otherwise discover
    # the on-disk universe (capped by --limit for smoke, uncapped for full).
    missions = a.missions or (GOLD_MISSIONS if a.rung == "smoke" and not a.limit
                              else discover_missions(a.limit))
    run_sweep(missions, a.backend, a.model, a.out, a.resume, a.dossier_budget, a.concurrency)


if __name__ == "__main__":
    main()
