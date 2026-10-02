import json, shutil, tempfile, unittest
from pathlib import Path
from scripts import mark8_recovery_queue as q

class RecoveryQueueTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup); self.run=Path(self.t.name)/"run"
 def write(self,stage,items,name="a001"):
  p=self.run/"accounting"/stage/f"{stage}-{name}"/f"{stage}.loop.json"; p.parent.mkdir(parents=True,exist_ok=True)
  p.write_text(json.dumps({"schema":q.ACCOUNTING_SCHEMA,"stage":stage,"producer":"loop","items":items})); return p
 def item(self,i,status,reason="",response=False):
  attempts=[]
  if response:
   rel=f"artifacts/.attempts/{i}.json"; x=self.run/rel; x.parent.mkdir(parents=True,exist_ok=True); x.write_text("response")
   attempts=[{"attempt":0,"response":rel,"result":status}]
  return {"id":i,"paper":"math/0301001" if i.startswith("old") else "A","status":status,"reason":reason,"artifacts":[],"attempts":attempts}
 def test_classes_exact_real_reasons_and_reason_preserved(self):
  reasons=[
   ("no clause spans in the proof, required by mark7-v4","deterministic-input-defect",False),
   ("contract: the steps derive node 7 -> node 8 -> node 7, so the argument assumes what it proves; an equivalence is one iff step","post-call-contract-failure",False),
   ("endpoint returned non-JSON despite the schema (Invalid control character at: line 6 column 61 (char 156)); check serving conformance","sanitize-reparse-candidate",False),
   ("nodes: TimeoutError: timed out","retryable-transport",True),
   ("nodes: output truncated at max_tokens=8192","scoped-retry-candidate",True),
   ("contract: something new","manual-review",False)]
  items=[self.item(str(i),"errored",r,response=(i==2)) for i,(r,_,_) in enumerate(reasons)]
  out=q.build([self.write("S3",items)])
  self.assertEqual([(x["reason-class"],x["retry-eligible"]) for x in out["queue"]],[(c,r) for _,c,r in reasons])
  self.assertEqual([x["reason"] for x in out["queue"]],[r for r,_,_ in reasons])
 def test_failures_once_successes_excluded_but_counted_and_ids_exact(self):
  items=[self.item("old_math", "accepted"),self.item("d","deferred"),self.item("r","rejected","no clause units in the region, required by mark7-v4"),self.item("e","errored","unknown")]
  out=q.build([self.write("S4",items)])
  self.assertEqual([x["item-id"] for x in out["queue"]],["e","r"])
  self.assertEqual(out["counts"]["status"],{"accepted":1,"deferred":1,"errored":1,"rejected":1})
  self.assertEqual(out["queue"][0]["paper-id"],"A")
 def test_duplicate_and_refused_contracts_fail(self):
  a=self.write("S3",[self.item("x","errored","unknown")]); b=self.write("S4",[self.item("x","rejected","unknown")])
  with self.assertRaisesRegex(ValueError,"duplicate"): q.build([a,b])
  d=json.loads(a.read_text()); d["producer"]="model"; a.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,"refused"): q.build([a])
 def test_deterministic_bytes_under_file_and_item_reorder(self):
  xs=[self.item("b","errored","unknown"),self.item("a","rejected","no clause spans in the proof, required by mark7-v4")]
  a=self.write("S3",xs); one=q.encode(q.build([a])); a=self.write("S3",list(reversed(xs))); two=q.encode(q.build([a]))
  self.assertEqual(one,two)
 def test_identical_run_trees_at_different_roots_are_byte_identical(self):
  item=self.item("x","errored","nodes: TimeoutError: timed out",response=True)
  first=self.write("S3",[item])
  other=Path(self.t.name)/"other"/"run"
  shutil.copytree(self.run,other)
  second=other/"accounting/S3/S3-a001/S3.loop.json"
  self.assertEqual(q.encode(q.build([first])),q.encode(q.build([second])))
 def test_evidence_path_escape_is_refused(self):
  item=self.item("x","errored","unknown"); item["artifacts"]=["../outside.json"]
  with self.assertRaisesRegex(ValueError,"escapes run"): q.build([self.write("S3",[item])])
 def test_control_character_requires_completed_response(self):
  reason="endpoint returned non-JSON despite the schema (Invalid control character at: line 1 column 2 (char 1))"
  out=q.build([self.write("S4",[self.item("x","errored",reason)])])
  self.assertEqual(out["queue"][0]["reason-class"],"manual-review")

if __name__ == "__main__": unittest.main()
