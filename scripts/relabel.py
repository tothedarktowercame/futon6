import json
p = "data/runs/mark7z/phase-ledger.jsonl"
prov = ("artifacts identical to those ledgered under math-ct-e2e-12: the same 98 proof "
        "graphs / 280 expository scopes / 88 CLean typings, from the same 16 papers. The "
        "prior corpus_id understated the paper count (12 declared, 16 actual); the label "
        "was corrected, the work was not repeated. This is NOT a clean single-corpus "
        "execution of this stage - see the capability proof's A12 caveat.")
rows = [("S3", ["mark7z-s3-shard-a.log", "mark7z-s3-shard-b.log",
                "mark7z-s3-fin-a.log", "mark7z-s3-fin-b.log"]),
        ("S4", ["mark7z-s4.log", "mark7z-s4-redo.log"]),
        ("S7", ["mark7z-s7.log", "mark7z-s7-retry.log", "mark7z-s7-sweep.log"])]
with open(p, "a") as fh:
    for st, ev in rows:
        fh.write(json.dumps({"stage": st, "corpus_id": "math-ct-e2e-16", "run_id": "mark7z",
                             "gate": "pass", "provenance": prov,
                             "relabelled_from": "math-ct-e2e-12", "evidence": ev}) + "\n")
s = set()
for ln in open(p):
    r = json.loads(ln)
    if r["corpus_id"] == "math-ct-e2e-16":
        s.add(r["stage"])
print("e2e-16 now:", len(s), sorted(s, key=lambda x: int(x[1:])))
