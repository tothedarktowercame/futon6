#!/usr/bin/env bb
;; land_proof_mine.bb — D7: the SEPARATE, gated, CPU landing step for the PROOF-MINE run.
;; Per futon6/holes/proof-mine-runner-spec.md §D7. The GPU box writes NOTHING to :7071; this reads the
;; CLEANED, VALIDATED landing file (proof-mine-landing.jsonl, produced by proof_mine_land_prep.py) and
;; lands each as a PROOF-layer relation, reusing promote_c_entries.bb's write path (POST /relation,
;; x-penholder: api).
;;
;;   relation:  <canonical mission node>  --:discharged-by-->  sha/<clean-sha>
;;              props {grade, witness, raw-target, raw-discharged-by, source "proof-mine"}
;;   + a commit ENTITY per sha (relation dst target), source "proof-mine".
;;   + rule-candidates emitted as a REVIEWABLE proposal file (never a live fold_engine edit).
;;
;; NONDESTRUCTIVE by construction:
;;   - reads the PRE-VALIDATED landing file (git-checked shas, verbatim witnesses, canonical missions);
;;   - re-checks the mission node EXISTS in :7071 against the live mission-index — SKIPS (never mints) a
;;     mission that is absent;
;;   - idempotent name-keyed upserts (re-run = no duplicates);
;;   - everything tagged source "proof-mine" → the whole landing is reversible (delete by source).
;; DRY-RUN by DEFAULT (author != reviewer). Pass --execute to write.
(require '[babashka.http-client :as http]
         '[cheshire.core :as json]
         '[clojure.string :as str]
         '[clojure.java.io :as io]
         '[clojure.edn :as edn])

(def base "http://localhost:7071/api/alpha")
(def headers {"Content-Type" "application/json" "x-penholder" "api"})
(def execute? (some #{"--execute"} *command-line-args*))
(def out-dir (or (second (drop-while #(not= % "--out") *command-line-args*))
                 (str (System/getProperty "user.home") "/code/futon6/data/proof-mine")))
(def landing-file (str out-dir "/proof-mine-landing.jsonl"))
(def raw-file (str out-dir "/proof-mine.jsonl"))

(defn- read-jsonl [path]
  (when (.exists (io/file path))
    (->> (line-seq (io/reader path)) (remove str/blank?) (map #(json/parse-string % true)))))

(defn- fetch-mission-index
  "Set of canonical mission node names live in :7071 (mirror promote_c_entries.bb). The nondestructive
   existence check: we only land relations whose mission is actually present — never mint a mission."
  []
  (let [resp (http/get (str base "/entities/latest?type=mission%2Fdoc&limit=2000")
                       {:headers {"Accept" "application/edn"} :timeout 20000 :throw false})]
    (if (<= 200 (:status resp) 299)
      (into #{} (keep #(when (str/includes? (str (:name %)) "/mission/") (:name %))
                      (:entities (edn/read-string (:body resp)))))
      (do (println "WARN: could not fetch mission-index (status" (:status resp)
                   ") — cannot verify mission nodes; refusing to land.") #{}))))

(defn- post! [path payload]
  (let [resp (http/post (str base path)
                        {:headers headers :timeout 20000 :throw false
                         :body (json/generate-string payload)})]
    (when-not (<= 200 (:status resp) 299)
      (throw (ex-info (str "write failed " path " " (:status resp))
                      {:body (subs (str (:body resp)) 0 (min 300 (count (str (:body resp)))))})))
    (:status resp)))

(defn -main []
  (let [landing (read-jsonl landing-file)
        _ (when-not landing
            (throw (ex-info (str "no landing file: " landing-file
                                 " — run scripts/proof_mine_land_prep.py first.") {})))
        mindex (fetch-mission-index)
        {present true absent false} (group-by #(contains? mindex (:mission %)) landing)
        shas (into #{} (map :sha present))
        rels (for [r present]
               {:type "discharged-by" :src (:mission r) :dst (str "sha/" (:sha r))
                :props {:label "discharged-by" :source "proof-mine" :grade (:grade r)
                        :witness (:witness r) :raw-target (:raw-target r)
                        :raw-discharged-by (:raw-discharged-by r)}})
        ;; rule-candidates (from the raw run) → reviewable proposal file, both modes; never a live edit.
        rule-cands (for [r (read-jsonl raw-file) rc (:rule_candidates r)] (assoc rc :mission (:mission r)))]
    (println (str (if execute? "EXECUTE" "DRY-RUN")
                  " · landing rels " (count present)
                  " · mission absent in :7071 (skipped) " (count absent)
                  " · commit nodes " (count shas)
                  " · rule-candidates " (count rule-cands)))
    (println "rel samples:" (vec (take 3 (map #(str (:src %) " -> " (:dst %)) rels))))
    (when (seq absent)
      (println "skipped (mission node absent — NOT minted):"
               (vec (take 5 (distinct (map :mission absent))))))
    (spit (str out-dir "/fold-rule-candidates.edn") (pr-str (vec rule-cands)))
    (println "wrote" (count rule-cands) "rule-candidates ->" (str out-dir "/fold-rule-candidates.edn")
             "(review → fold_engine rule-table PR; NOT auto-landed)")
    (if execute?
      (do
        (doseq [s shas]
          (post! "/entity" {:name (str "sha/" s) :type "commit" :external-id (str "sha/" s)
                            :source "proof-mine" :props {:sha s}}))
        (doseq [r rels] (post! "/relation" r))
        (println "WROTE" (+ (count shas) (count rels)) "docs to :7071 (source \"proof-mine\", reversible)."))
      (println "(dry-run — pass --execute to write; author != reviewer, so review the counts first)"))))

(-main)
