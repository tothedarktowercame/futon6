#!/usr/bin/env python3
"""Build the M-reader-clear findings table from completed kit directories.

Usage: reader-clear-findings.py --head SHA --kits DIR --classes classes.tsv --out DIR

A kit is complete when it has answers.jsonl and a round2.<reader>.jsonl for
every reader with a questions.<reader>.jsonl. One TSV row per question.
An optional diagnosis-overrides.json in a kit ({"<reader>:<id>": "<diagnosis>"})
records the owner's reclassification; the answerer's diagnosis is kept in a column.
"""
import argparse, collections, csv, glob, json, os, sys

TYPE = {"printer-drops-content": "printer", "printer-wording": "printer",
        "lean-proof-choice": "lean-proof",
        "statement-unused-hypothesis": "statement", "statement-missing-hypothesis": "statement",
        "known-statement-repair": "statement-known",
        "informal-skips-or-loose": "informal", "reader-gap": "reader-error"}
# Answerers sometimes coin different names for one proposed class; merge them here.
CLASS_ALIAS = {"NEW:omitted-instance-hypotheses": "NEW:lost-instance-hypotheses",
               "NEW:omitted-typeclass-hypotheses": "NEW:lost-instance-hypotheses",
               "NEW:named-cauchy-estimate": "P-unnamed-library-result",
               "NEW:simple-pole-definition": "NEW:undefined-statement-predicate",
               "NEW:image-witness": "NEW:wrong-witness-term"}
COLS = ["head", "module", "declaration", "steps", "printed_line", "reader", "question",
        "reader_verdict", "diagnosis_type", "diagnosis", "answerer_diagnosis", "proposed_class",
        "class_known", "settlement", "print_would_prevent", "evidence"]
MARK_COLS = ["head", "module", "declaration", "step", "printed_line", "reader", "verdict",
             "note", "question", "answerer_diagnosis", "proposed_class", "other_readers"]

def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def one(s):
    return " ".join(str(s).split())

def complete_rows(path):
    """Read a JSONL file atomically, warning and returning None if truncated."""
    try:
        return rows(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skipping incomplete JSONL {path}: {exc}", file=sys.stderr)
        return None

def question_steps(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []

def mark_rows(head, kits):
    marks = []
    for kit in sorted(glob.glob(os.path.join(kits, "*"))):
        module = os.path.basename(kit)
        printed_path = os.path.join(kit, "steps.jsonl")
        if not os.path.exists(printed_path):
            continue
        printed = rows(printed_path)
        by_step = {row["step"]: (index, row) for index, row in enumerate(printed)}
        verdicts = {}
        for path in sorted(glob.glob(os.path.join(kit, "steps.*.jsonl"))):
            reader = os.path.basename(path)[len("steps."):-len(".jsonl")]
            parsed = complete_rows(path)
            if parsed is not None:
                verdicts[reader] = {row["id"]: row for row in parsed}
        questions = {}
        for reader in verdicts:
            path = os.path.join(kit, f"questions.{reader}.jsonl")
            if not os.path.exists(path):
                continue
            parsed = complete_rows(path)
            if parsed is None:
                continue
            for question in parsed:
                for step in question_steps(question.get("steps", [])):
                    questions.setdefault((reader, step), question["id"])
        answers_path = os.path.join(kit, "answers.jsonl")
        answers = {}
        if os.path.exists(answers_path):
            parsed = complete_rows(answers_path)
            if parsed is not None:
                answers = {(row["reader"], row["id"]): row for row in parsed}
        for reader, reader_verdicts in verdicts.items():
            for step, mark in reader_verdicts.items():
                if mark.get("verdict") not in {"wrong-as-printed", "not-justified"} or step not in by_step:
                    continue
                order, printed_row = by_step[step]
                question = questions.get((reader, step))
                answer = answers.get((reader, question)) if question else None
                others = [f"{other}={other_marks[step]['verdict']}"
                          for other, other_marks in sorted(verdicts.items())
                          if other != reader and step in other_marks]
                marks.append({"head": head, "module": module,
                              "declaration": printed_row.get("declaration", "-"),
                              "step": step, "printed_line": one(printed_row.get("text", "")),
                              "reader": reader, "verdict": mark["verdict"],
                              "note": one(mark.get("note", "")),
                              "question": question or "-",
                              "answerer_diagnosis": answer.get("diagnosis", "-") if answer else "-",
                              "proposed_class": answer.get("proposed_class", "-") or "-" if answer else "-",
                              "other_readers": ";".join(others) or "-",
                              "_step_order": order})
    marks.sort(key=lambda row: (row["module"], row["_step_order"], row["reader"]))
    return marks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True); ap.add_argument("--kits", required=True)
    ap.add_argument("--classes", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.classes) as f:
        known = {l.split("\t")[0] for l in f if l.strip()} - {"id"}
    out, skipped = [], []
    for kit in sorted(glob.glob(os.path.join(a.kits, "*"))):
        p = os.path.basename(kit)
        qfiles = sorted(glob.glob(os.path.join(kit, "questions.*.jsonl")))
        readers = [os.path.basename(q)[len("questions."):-len(".jsonl")] for q in qfiles]
        if len(readers) < 2 or not os.path.exists(os.path.join(kit, "answers.jsonl")) or \
           not all(os.path.exists(os.path.join(kit, f"round2.{r}.jsonl")) for r in readers):
            skipped.append(p); continue
        decls = [d["declaration"] for d in json.load(open(os.path.join(kit, "kit.json")))["declarations"]]
        ans = {(x["reader"], x["id"]): x for x in rows(os.path.join(kit, "answers.jsonl"))}
        ov_path = os.path.join(kit, "diagnosis-overrides.json")
        ov = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
        for r in readers:
            r2 = {x["id"]: x for x in rows(os.path.join(kit, f"round2.{r}.jsonl"))}
            for q in rows(os.path.join(kit, f"questions.{r}.jsonl")):
                an = ans.get((r, q["id"]))
                if an is None:
                    sys.exit(f"{p}: no answer for {r} {q['id']}")
                qsteps = question_steps(q["steps"])
                ds = sorted({decls[int(s.split('.')[0][1:]) - 1] for s in qsteps})
                diag = ov.get(f"{r}:{q['id']}", an["diagnosis"])
                cls = an.get("proposed_class", "") if TYPE[diag] == "printer" else ""
                cls = CLASS_ALIAS.get(cls, cls)
                if cls.startswith("NEW:omitted-") and cls.endswith("-hypotheses"):
                    cls = "NEW:lost-instance-hypotheses"
                rr = r2.get(q["id"], {})
                out.append({"head": a.head, "module": p, "declaration": ";".join(ds) or "-",
                            "steps": ";".join(qsteps) or "-", "printed_line": one(q["locus_P"]),
                            "reader": r, "question": q["id"],
                            "reader_verdict": "not-justified" if qsteps else "question-only",
                            "diagnosis_type": TYPE[diag], "diagnosis": diag,
                            "answerer_diagnosis": an["diagnosis"], "proposed_class": cls,
                            "class_known": "" if not cls else ("yes" if cls in known else "new" if cls.startswith("NEW:") else "UNKNOWN"),
                            "settlement": rr.get("status", "missing"),
                            "print_would_prevent": rr.get("would_the_print_have_prevented_it", ""),
                            "evidence": one(an["answer"]).split(". ")[0].rstrip(".") + "."})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"findings.head-{a.head}.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", quotechar=None)
        w.writeheader(); w.writerows(out)
    marks = mark_rows(a.head, a.kits)
    with open(os.path.join(a.out, f"marks.head-{a.head}.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, MARK_COLS, delimiter="\t", quoting=csv.QUOTE_NONE,
                           escapechar="\\", quotechar=None, extrasaction="ignore")
        w.writeheader(); w.writerows(marks)
    c = collections.Counter((x["diagnosis_type"], x["proposed_class"] or "-") for x in out)
    with open(os.path.join(a.out, f"class-counts.head-{a.head}.tsv"), "w") as f:
        f.write("diagnosis_type\tproposed_class\tfindings\tmodules\n")
        for (t, k), n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
            m = len({x["module"] for x in out if (x["diagnosis_type"], x["proposed_class"] or "-") == (t, k)})
            f.write(f"{t}\t{k}\t{n}\t{m}\n")
    print(f"{len(out)} findings from {len({x['module'] for x in out})} problems; not complete: {len(skipped)}")
    verdict_counts = collections.Counter(row["verdict"] for row in marks)
    counts = ", ".join(f"{verdict}={verdict_counts[verdict]}"
                       for verdict in ("wrong-as-printed", "not-justified"))
    print(f"{len(marks)} marks from {len({x['module'] for x in marks})} modules; {counts}")
    bad = [x for x in out if x["class_known"] == "UNKNOWN"]
    if bad:
        print("unknown class ids:", sorted({x["proposed_class"] for x in bad})); return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
