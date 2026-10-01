import json,math,shutil,tempfile,unittest
from pathlib import Path
from scripts import mark8_s4_policy_compare as c

class ComparatorTests(unittest.TestCase):
 def tree(self,root,reverse=False):
  run=Path(root)/"run"; d=run/"accounting/S4/S4-a001"; d.mkdir(parents=True)
  papers=["math/0301001","math__0301001"]
  ids={p:[f"{p}:r{i}" for i in range(4 if i==0 else 2)] for i,p in enumerate(papers)}
  ex=[];sel=[]
  for p in papers:
   ex.append({"id":p,"paper":p,"status":"accepted","outputs":ids[p]})
   for j,x in enumerate(ids[p]): sel.append({"id":x,"paper":p,"status":"accepted" if j<1 else "deferred"})
  if reverse: ex.reverse();sel.reverse()
  base={"schema":c.ACCT,"stage":"S4","invocation":"S4-a001"}
  (d/"S4.extract.json").write_text(json.dumps({**base,"producer":"extract","items":ex}))
  (d/"S4.select.json").write_text(json.dumps({**base,"producer":"select","items":sel}))
  (run/"run-manifest.json").write_text(json.dumps({"papers":papers,"selection":{"expository-selection":"archived-policy/v1","expository-cap":1,"expository-cap-rule":None}}))
  return run,d/"S4.extract.json",d/"S4.select.json",run/"run-manifest.json"
 def test_exact_baseline_same_budget_and_old_ids(self):
  with tempfile.TemporaryDirectory() as t:
   _,e,s,m=self.tree(t); x=c.compare(e,s,m)
   self.assertEqual(x["global"]["budget"],2);self.assertEqual(x["global"]["counterfactual"]["selected"],2);self.assertTrue(x["global"]["conserved"])
   self.assertEqual(x["selections"]["baseline"],["math/0301001:r0","math__0301001:r0"])
   self.assertEqual([r["paper-id"] for r in x["per-paper"]],["math/0301001","math__0301001"])
   self.assertEqual(x["archived-policy"]["name"],"archived-policy/v1")
 def test_reordered_and_relocated_are_byte_deterministic(self):
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   _,e1,s1,m1=self.tree(a);_,e2,s2,m2=self.tree(b,True)
   self.assertEqual(c.enc(c.compare(e1,s1,m1)),c.enc(c.compare(e2,s2,m2)))
 def test_duplicates_foreign_inconsistent_and_bad_schema_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   _,e,s,m=self.tree(t); d=json.loads(s.read_text());d["items"].append(d["items"][0]);s.write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"duplicate"):c.compare(e,s,m)
   _,e,s,m=self.tree(Path(t)/"again");d=json.loads(s.read_text());d["items"][0]["id"]="foreign";s.write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"foreign"):c.compare(e,s,m)
   d=json.loads(s.read_text());d["schema"]="bad";s.write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"refused"):c.compare(e,s,m)
 def test_nonfinite_and_path_mismatch_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   run,e,s,m=self.tree(t); d=json.loads(e.read_text());d["score"]=math.inf;e.write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"nonfinite"):c.compare(e,s,m)
   other=run/"copy.json";shutil.copy(s,other)
   with self.assertRaisesRegex(ValueError,"path mismatch"):c.compare(e,other,m)
 def test_missing_archived_name_is_honest(self):
  with tempfile.TemporaryDirectory() as t:
   _,e,s,m=self.tree(t);d=json.loads(m.read_text());del d["selection"]["expository-selection"];m.write_text(json.dumps(d))
   self.assertEqual(c.compare(e,s,m)["archived-policy"]["name"],"archived-observed-selection")
if __name__=="__main__":unittest.main()
