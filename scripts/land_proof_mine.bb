#!/usr/bin/env bb
;; land_proof_mine.bb — D7: the SEPARATE, gated, CPU landing step for the PROOF-MINE run.
;; Per futon6/holes/proof-mine-runner-spec.md §D7. The GPU box writes NOTHING to :7071; this reads
;; the run artifact (proof-mine.jsonl) on dev and lands the graded discharges as PROOF-layer relations,
;; reusing promote_c_entries.bb's write path (POST /relation, x-penholder: api, run-write! pipeline).
;;
;;   step 1: `:discharged-by` RELATIONS  target(c-entry|sorry) -> sha/<x> | method/<x>
;;           for every discharge graded :discharged with a non-null discharged_by;
;;   step 2: rule-candidates are emitted as a fold_engine rule-table PROPOSAL file
;;           (data/proof-mine/fold-rule-candidates.edn) — NOT a silent fold_engine edit.
;;
;; DRY-RUN by DEFAULT (author != reviewer): prints what it WOULD write. Pass --execute to write.
;; Idempotent: relations are name-keyed upserts. Nothing here mints an entity the runner quarantined.
(require '[babashka.http-client :as http]
         '[cheshire.core :as json]
         '[clojure.string :as str]
         '[clojure.java.io :as io])

(def base "http://localhost:7071/api/alpha")
(def headers {"Content-Type" "application/json" "x-penholder" "api"})
(def execute? (some #{"--execute"} *command-line-args*))
(def out-dir (or (second (drop-while #(not= % "--out") *command-line-args*))
                 (str (System/getProperty "user.home") "/code/futon6/data/proof-mine")))
(def jsonl (str out-dir "/proof-mine.jsonl"))

(defn- read-records []
  (when-not (.exists (io/file jsonl))
    (throw (ex-info (str "no run artifact: " jsonl " — run the miner first (RUNG=full).") {})))
  (->> (line-seq (io/reader jsonl))
       (remove str/blank?)
       (map #(json/parse-string % true))))

(defn- dsc-node
  "The relation dst for a discharged_by value: a sha or a method class."
  [db]
  (let [s (str db)]
    (if (re-matches #"[0-9a-f]{7,40}" s) (str "sha/" s) (str "method/" s))))

(defn- post! [path payload]
  (let [resp (http/post (str base path)
                        {:headers headers :timeout 20000 :throw false
                         :body (json/generate-string payload)})]
    (when-not (<= 200 (:status resp) 299)
      (throw (ex-info (str "write failed " path " " (:status resp))
                      {:body (subs (str (:body resp)) 0 (min 300 (count (str (:body resp)))))})))
    (:status resp)))

(defn -main []
  (let [records (read-records)
        ;; step 1 — discharged-by relations, ONLY for graded :discharged with a real discharged_by.
        rels (for [r records
                   d (:discharges r)
                   :when (and (= "discharged" (:grade d)) (:discharged_by d) (:target d))]
               {:type "discharged-by" :src (:target d) :dst (dsc-node (:discharged_by d))
                :props {:label "discharged-by" :source "proof-mine"
                        :mission (:mission r) :grade (:grade d)
                        :witness-verbatim (:witness_verbatim d)}})
        methods (into #{} (map :dst rels))
        ;; step 2 — fold_engine rule-table candidates as a PROPOSAL file (no silent edit).
        rule-cands (for [r records rc (:rule_candidates r)]
                     (assoc rc :mission (:mission r)))]
    (println (str (if execute? "EXECUTE" "DRY-RUN")
                  " · records " (count records)
                  " · discharged-by rels " (count rels)
                  " · target nodes " (count methods)
                  " · rule-candidates " (count rule-cands)))
    (println "rel samples:" (vec (take 3 (map #(str (:src %) " -> " (:dst %)) rels))))
    ;; rule-candidates always written as a REVIEWABLE proposal file (both modes) — never a live edit.
    (spit (str out-dir "/fold-rule-candidates.edn") (pr-str (vec rule-cands)))
    (println "wrote" (count rule-cands) "rule-candidates ->" (str out-dir "/fold-rule-candidates.edn")
             "(review → fold_engine rule-table PR; NOT auto-landed)")
    (if execute?
      (do
        (doseq [m methods]
          (post! "/entity" {:name m :type (if (str/starts-with? m "sha/") "commit" "method/class")
                            :external-id m :source "proof-mine" :props {}}))
        (doseq [r rels] (post! "/relation" r))
        (println "WROTE" (+ (count methods) (count rels)) "docs to :7071."))
      (println "(dry-run — pass --execute to write; author != reviewer, so review the counts first)"))))

(-main)
