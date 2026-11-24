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

def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def one(s):
    return " ".join(str(s).split())

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
                ds = sorted({decls[int(s.split('.')[0][1:]) - 1] for s in q["steps"]})
                diag = ov.get(f"{r}:{q['id']}", an["diagnosis"])
                cls = an.get("proposed_class", "") if TYPE[diag] == "printer" else ""
                cls = CLASS_ALIAS.get(cls, cls)
                rr = r2.get(q["id"], {})
                out.append({"head": a.head, "module": p, "declaration": ";".join(ds) or "-",
                            "steps": ";".join(q["steps"]) or "-", "printed_line": one(q["locus_P"]),
                            "reader": r, "question": q["id"],
                            "reader_verdict": "not-justified" if q["steps"] else "question-only",
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
    c = collections.Counter((x["diagnosis_type"], x["proposed_class"] or "-") for x in out)
    with open(os.path.join(a.out, f"class-counts.head-{a.head}.tsv"), "w") as f:
        f.write("diagnosis_type\tproposed_class\tfindings\tmodules\n")
        for (t, k), n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
            m = len({x["module"] for x in out if (x["diagnosis_type"], x["proposed_class"] or "-") == (t, k)})
            f.write(f"{t}\t{k}\t{n}\t{m}\n")
    print(f"{len(out)} findings from {len({x['module'] for x in out})} problems; not complete: {len(skipped)}")
    bad = [x for x in out if x["class_known"] == "UNKNOWN"]
    if bad:
        print("unknown class ids:", sorted({x["proposed_class"] for x in bad})); return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
