import json,math,os,shutil,tempfile,unittest
from pathlib import Path
from scripts import mark8_browser_boundary as b

class BoundaryTests(unittest.TestCase):
 def make(self,root):
  run=Path(root)/"run"; (run/"accounting/S3/x").mkdir(parents=True);(run/"accounting/S4/x").mkdir(parents=True)
  (run/"artifacts/graphs").mkdir(parents=True);(run/"artifacts/steps").mkdir();(run/"artifacts/expo").mkdir()
  graph='{:paper/id "math__0301001" :passage/id "math__0301001:proof0:L1-2" :nodes [{:id :n1}] :edges [{:id :e1 :warrant {:kind :citation}}]}'
  expo='{:paper/id "math/0301001" :passage/id "math/0301001:r0" :scopes [{:id :s1}]}'
  (run/"artifacts/graphs/math__0301001__p0.edn").write_text(graph)
  (run/"artifacts/steps/math__0301001__p0.steps.json").write_text(json.dumps({"paper_id":"math__0301001","steps":[{"id":"s1"}]}))
  (run/"artifacts/expo/x.edn").write_text(expo);(run/"run-manifest.json").write_text('{"schema-version":1}')
  base={"schema":b.ACCT,"producer":"loop"}
  s3={**base,"stage":"S3","items":[{"id":"math__0301001__p0","paper":"math__0301001","status":"accepted","artifacts":["artifacts/graphs/math__0301001__p0.edn"]}]}
  s4={**base,"stage":"S4","items":[{"id":"math/0301001:r0","paper":"math/0301001","status":"accepted","artifacts":["artifacts/expo/x.edn"]}]}
  for stage,doc in (("S3",s3),("S4",s4)):
   doc["invocation"]=f"{stage}-a001";doc["expected"]=[doc["items"][0]["id"]];doc["counts"]={"accepted":1,"rejected":0,"errored":0,"deferred":0,"expected":1,"unaccounted":0}
  (run/"accounting/S3/x/a.json").write_text(json.dumps(s3));(run/"accounting/S4/x/a.json").write_text(json.dumps(s4))
  return run,Path("accounting/S3/x/a.json"),Path("accounting/S4/x/a.json"),Path("run-manifest.json")
 def test_real_shape_exact_ids_and_allowlist(self):
  with tempfile.TemporaryDirectory() as t:
   run,s3,s4,m=self.make(t);(run/"artifacts/graphs/unaccounted.edn").write_text("{:model-output true}")
   x=b.build(run,s3,s4,m);self.assertEqual(x["summary"]["files"],3)
   self.assertEqual({e["paper-id"] for e in x["entities"] if e["kind"]=="paper"},{"math/0301001","math__0301001"})
   self.assertNotIn("unaccounted.edn",json.dumps(x));self.assertEqual(x["skipped-roles"][2]["role"],"embeddings")
 def test_relocated_identical_tree_is_deterministic(self):
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as c:
   r1,s31,s41,m1=self.make(a);r2,s32,s42,m2=self.make(c)
   self.assertEqual(b.enc(b.build(r1,s31,s41,m1)),b.enc(b.build(r2,s32,s42,m2)))
 def test_absolute_dotdot_symlink_missing_and_empty_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   run,s3,s4,m=self.make(t)
   with self.assertRaisesRegex(ValueError,"absolute"):b.build(run,Path("/tmp/x"),s4,m)
   with self.assertRaisesRegex(ValueError,"escape"):b.build(run,Path("../x"),s4,m)
   outside=Path(t)/"outside";outside.write_text("x");(run/"link").symlink_to(outside)
   with self.assertRaisesRegex(ValueError,"symlink escape"):b.inside(run,Path("link"),"test")
   with self.assertRaisesRegex(ValueError,"missing/empty"):b.inside(run,Path("missing"),"test")
   (run/"empty").write_text("")
   with self.assertRaisesRegex(ValueError,"missing/empty"):b.inside(run,Path("empty"),"test")
 def test_duplicate_unaccepted_foreign_and_inconsistent_identity_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   run,s3,s4,m=self.make(t);d=json.loads((run/s3).read_text());d["items"].append(d["items"][0]);(run/s3).write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"duplicate"):b.build(run,s3,s4,m)
   run,s3,s4,m=self.make(Path(t)/"b");d=json.loads((run/s3).read_text());d["items"][0]["paper"]="foreign";(run/s3).write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"identity"):b.build(run,s3,s4,m)
   run,s3,s4,m=self.make(Path(t)/"c");d=json.loads((run/s3).read_text());d["items"][0]["status"]="rejected";d["counts"]["accepted"]=0;d["counts"]["rejected"]=1;(run/s3).write_text(json.dumps(d));x=b.build(run,s3,s4,m)
   self.assertFalse(any(f["producer-stage"]=="S3" for f in x["files"]))
 def test_malformed_nonfinite_and_dangling_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   run,s3,s4,m=self.make(t);(run/"artifacts/steps/math__0301001__p0.steps.json").write_text("{")
   with self.assertRaises(Exception):b.build(run,s3,s4,m)
   run,s3,s4,m=self.make(Path(t)/"b");d=json.loads((run/s3).read_text());d["bad"]=math.inf;(run/s3).write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,"nonfinite"):b.build(run,s3,s4,m)
   run,s3,s4,m=self.make(Path(t)/"c");old=b.MAX_FILES;b.MAX_FILES=1
   try:
    with self.assertRaisesRegex(ValueError,"budget"):b.build(run,s3,s4,m)
   finally:b.MAX_FILES=old
 def test_false_invocation_counts_and_expected_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   for n,(field,value,pattern) in enumerate((("invocation","S3-a999","invocation"),("counts",{"accepted":999},"counts"),("expected",["foreign"],"expected"))):
    run,s3,s4,m=self.make(Path(t)/str(n));d=json.loads((run/s3).read_text());d[field]=value;(run/s3).write_text(json.dumps(d))
    with self.assertRaisesRegex(ValueError,pattern):b.build(run,s3,s4,m)
 def test_duplicate_and_empty_navigation_ids_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   run,s3,s4,m=self.make(t);p=run/"artifacts/graphs/math__0301001__p0.edn";p.write_text('{:paper/id "math__0301001" :passage/id "p" :nodes [{:id :n1} {:id :n1}] :edges []}')
   with self.assertRaisesRegex(ValueError,"node id"):b.build(run,s3,s4,m)
   run,s3,s4,m=self.make(Path(t)/"step");p=run/"artifacts/steps/math__0301001__p0.steps.json";p.write_text(json.dumps({"paper_id":"math__0301001","steps":[{"id":""}]}))
   with self.assertRaisesRegex(ValueError,"step id"):b.build(run,s3,s4,m)
 def test_producer_native_passage_node_and_warrant_types(self):
  bad=[
   ('{:paper/id "math__0301001" :passage/id 17 :nodes [{:id :n1}] :edges []}',"graph schema"),
   ('{:paper/id "math__0301001" :passage/id "p" :nodes [{:id 17}] :edges []}',"node id"),
   ('{:paper/id "math__0301001" :passage/id "p" :nodes [{:id true}] :edges []}',"node id"),
   ('{:paper/id "math__0301001" :passage/id "p" :nodes [{:id :n1}] :edges [{:warrant {}}]}',"warrant map"),
   ('{:paper/id "math__0301001" :passage/id "p" :nodes [{:id :n1}] :edges [{:warrant {:kind 17}}]}',"warrant kind")]
  with tempfile.TemporaryDirectory() as t:
   for i,(text,pattern) in enumerate(bad):
    run,s3,s4,m=self.make(Path(t)/str(i));(run/"artifacts/graphs/math__0301001__p0.edn").write_text(text)
    with self.assertRaisesRegex(ValueError,pattern):b.build(run,s3,s4,m)
   run,s3,s4,m=self.make(Path(t)/"valid");x=b.build(run,s3,s4,m)
   proof=next(e for e in x["entities"] if e["kind"]=="accepted-proof")
   self.assertEqual(proof["passage-id"],"math__0301001:proof0:L1-2")
   self.assertEqual(proof["node-ids"],[":n1"]);self.assertEqual(proof["warrant-kinds"],[":citation"])
 def test_expanded_metadata_hits_actual_encoded_output_budget(self):
  with tempfile.TemporaryDirectory() as t:
   run,s3,s4,m=self.make(t);normal=b.build(run,s3,s4,m);old=b.MAX_OUTPUT_BYTES
   try:
    b.MAX_OUTPUT_BYTES=len(b.enc(normal))+1000
    p=run/"artifacts/graphs/math__0301001__p0.edn";p.write_text('{:paper/id "math__0301001" :passage/id "p" :nodes [{:id :'+('x'*3000)+'}] :edges []}')
    with self.assertRaisesRegex(ValueError,"encoded.*budget"):b.build(run,s3,s4,m)
   finally:b.MAX_OUTPUT_BYTES=old
if __name__=="__main__":unittest.main()
