#!/usr/bin/env bb
;; Rung-2 semantic harness for IATC argument graphs.
;;
;; Composes:
;;   R2a scripts/iatc_anchor_faithfulness.bb
;;   R2b/R2c scripts/iatc_closure_check.bb
;;   R2d scripts/r2d_concept_coverage.py
;;
;; Defaults to final graph files only; use --include-attempts to inspect retry
;; intermediates under .attempts/.

(require '[babashka.process :as p]
         '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def default-opts
  {:marks-dir "data/showcases/ct-anatomy/golden"
   :anchor-k 2
   :anchor-tau 0.45
   :anchor-floor 0.30
   ;; Observed loop-run-70b final spread includes 0.0 rates and aggregates to
   ;; 6/28. Keep R2c report-only until a stricter floor is calibrated.
   :warrant-floor 0.0
   :include-attempts false
   :gate false})

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/iatc_semcheck.bb [--include-attempts] [--gate] [--out FILE] [--md-out FILE] [--warrant-floor F] <graph.edn-or-dir> [...]"))
  (System/exit 2))

(defn parse-decimal [s flag]
  (try
    (Double/parseDouble s)
    (catch Exception _
      (binding [*out* *err*]
        (println "Bad number for" flag ":" s))
      (System/exit 2))))

(defn parse-args [args]
  (loop [opts default-opts
         paths []
         xs args]
    (if-not (seq xs)
      (do
        (when (empty? paths) (usage!))
        {:opts opts :paths paths})
      (let [[x & more] xs]
        (case x
          "--include-attempts" (recur (assoc opts :include-attempts true) paths more)
          "--gate" (recur (assoc opts :gate true) paths more)
          "--out" (do (when-not (seq more) (usage!))
                      (recur (assoc opts :out (first more)) paths (rest more)))
          "--md-out" (do (when-not (seq more) (usage!))
                         (recur (assoc opts :md-out (first more)) paths (rest more)))
          "--marks-dir" (do (when-not (seq more) (usage!))
                            (recur (assoc opts :marks-dir (first more)) paths (rest more)))
          "--warrant-floor" (do (when-not (seq more) (usage!))
                                (recur (assoc opts :warrant-floor
                                              (parse-decimal (first more) x))
                                       paths (rest more)))
          "--anchor-floor" (do (when-not (seq more) (usage!))
                               (recur (assoc opts :anchor-floor
                                             (parse-decimal (first more) x))
                                      paths (rest more)))
          "--help" (usage!)
          "-h" (usage!)
          (recur opts (conj paths x) more))))))

(defn attempts-path? [file]
  (some #(= ".attempts" (str %))
        (iterator-seq (.iterator (.toPath (io/file file))))))

(defn edn-files [include-attempts? path]
  (let [f (io/file path)]
    (cond
      (not (.exists f)) []
      (.isDirectory f) (->> (file-seq f)
                            (filter #(.isFile %))
                            (filter #(str/ends-with? (.getName %) ".edn"))
                            (filter #(or include-attempts?
                                         (not (attempts-path? %))))
                            (sort-by #(.getPath %)))
      (and (str/ends-with? (.getName f) ".edn")
           (or include-attempts? (not (attempts-path? f)))) [f]
      :else [])))

(defn load-script-into! [ns-sym path]
  (let [ns-obj (or (find-ns ns-sym) (create-ns ns-sym))]
    (binding [*ns* ns-obj]
      (clojure.core/refer 'clojure.core)
      (load-file path))
    ns-obj))

(def anchor-ns
  (load-script-into! 'iatc-semcheck.anchor "scripts/iatc_anchor_faithfulness.bb"))

(def closure-ns
  (load-script-into! 'iatc-semcheck.closure "scripts/iatc_closure_check.bb"))

(def anchor-read-one-edn (ns-resolve anchor-ns 'read-one-edn))
(def anchor-load-lines (ns-resolve anchor-ns 'load-lines))
(def anchor-check-graph (ns-resolve anchor-ns 'check-graph))
(def closure-check-graph (ns-resolve closure-ns 'check-graph))

(defn graph-paper-id [graph file]
  (or (:paper/id graph)
      (some-> (:passage/id graph) (str/split #":") first)
      (str/replace (.getName (io/file file)) #"\.edn$" "")))

(defn seqify [x]
  (cond
    (nil? x) []
    (sequential? x) x
    :else [x]))

(defn endpoint-ids [edge ks]
  (->> ks
       (mapcat #(seqify (get edge %)))
       (filter keyword?)
       vec))

(defn source-lines [x]
  (get-in x [:source :lines]))

(defn node-kind-counts [nodes]
  (into (sorted-map)
        (map (fn [[k v]] [k (count v)])
             (group-by :kind nodes))))

(defn warrant-profile [edge]
  (let [w (:warrant edge)]
    (cond
      (nil? w) {:status :absent}
      (= :missing-warrant w) {:status :missing :kind :missing-warrant}
      (= :missing-warrant (:kind w)) {:status :missing
                                      :kind :missing-warrant
                                      :wanted (or (:wanted w) (:text w))}
      :else {:status :resolved
             :kind (:kind w)
             :label (or (:label w) (:target w) (:text w))})))

(defn profile [file graph anchor-result closure-results concept-result]
  (let [nodes (vec (:nodes graph))
        edges (vec (:edges graph))
        holes (vec (:holes graph))
        anchor-items (:per-item anchor-result)
        imported-terms (->> anchor-items
                            (mapcat :terms)
                            (remove str/blank?)
                            distinct
                            sort
                            vec)
        spans (->> (concat nodes edges)
                   (map source-lines)
                   (filter #(and (vector? %) (= 2 (count %)))))
        lines (when (seq spans)
                [(apply min (map first spans))
                 (apply max (map second spans))])]
    {:paper-id (graph-paper-id graph file)
     :file (.getPath (io/file file))
     :passage-id (:passage/id graph)
     :imported-terms imported-terms
     :region-skeleton {:source-lines lines
                       :node-count (count nodes)
                       :edge-count (count edges)
                       :hole-count (count holes)
                       :node-kinds (node-kind-counts nodes)}
     :reasoning (mapv (fn [edge]
                        {:id (:id edge)
                         :relation (or (:relation edge) (:kind edge) (:role edge))
                         :premises (endpoint-ids edge [:from :given :premise :assume
                                                       :depends-on :contradicts :meta])
                         :conclusions (endpoint-ids edge [:to :conclusion])
                         :warrant (warrant-profile edge)
                         :source {:lines (source-lines edge)}})
                      edges)
     :concept-coverage {:rate (:rate concept-result)
                        :buckets (:buckets concept-result)
                        :undefined (:undefined concept-result)
                        :imported (:imported concept-result)
                        :concept-source (:concept-source concept-result)}
     :certified-by (mapv :check closure-results)}))

(defn absent-structure? [check graph]
  (case check
    :anchor-faithfulness (empty? (:nodes graph))
    :closure (empty? (:edges graph))
    :warrant-resolution (empty? (:edges graph))
    false))

;; A check whose floor admits every rate cannot fail, so printing it as PASS in
;; the same column as three gating rungs makes any aggregate over that column a
;; mixture of "verified" and "not checked". :warrant-resolution is configured
;; report-only (:warrant-floor 0.0) until a stricter floor is calibrated, and on
;; the 98-graph corpus it duly reported 98/98 with 31 of those at rate 0.000.
;; The gating semantics are unchanged -- only the label it prints under.
(defn report-only? [check opts]
  (and (= :warrant-resolution check)
       (zero? (double (or (:warrant-floor opts) 0.0)))))

(defn normalize-check [graph result & [opts]]
  (let [na? (or (absent-structure? (:check result) graph)
                (and (= :concept-coverage (:check result))
                     (empty? (:per-item result))))]
    (cond-> result
      true (assoc :status (cond na? :na
                                (report-only? (:check result) opts) :report
                                (:pass result) :pass
                                :else :fail))
      na? (assoc :pass true
                 :rate nil
                 :reasons ["N/A: required structure absent at this resolution"]))))

;; Prefer the repo venv's interpreter. Invoking bare "python3" picked up the
;; system Python, which has no edn_format, so R2d raised ModuleNotFoundError and
;; the composer reported "R2d concept coverage failed" — rung-2 FAILing on every
;; graph in the corpus while R2d succeeded when run directly. Same class as the
;; LaTeXML gap: the dependency exists, but not where the caller looks.
(def python-bin
  (let [venv (io/file "/.venv/bin/python")
        local (io/file ".venv/bin/python")]
    (cond (.exists local) (.getPath local)
          (.exists venv) (.getPath venv)
          :else "python3")))

(defn concept-check-file [file]
  (let [result @(p/process [python-bin "scripts/r2d_concept_coverage.py"
                            "--edn" (.getPath (io/file file))]
                           {:out :string :err :string})]
    (when-not (zero? (:exit result))
      (throw (ex-info "R2d concept coverage failed"
                      {:file (.getPath (io/file file))
                       :stderr (:err result)})))
    (edn/read-string (:out result))))

(defn check-file [opts file]
  (try
    (let [graph (anchor-read-one-edn (io/file file))
          anchor-opts {:marks-dir (:marks-dir opts)
                       :k (:anchor-k opts)
                       :tau (:anchor-tau opts)
                       :floor (:anchor-floor opts)}
          ctx (anchor-load-lines graph (io/file file) anchor-opts)
          anchor-result (normalize-check graph
                                         (anchor-check-graph graph ctx anchor-opts))
          closure-results (mapv #(normalize-check graph % opts)
                                (closure-check-graph graph
                                                     {:file (.getPath (io/file file))
                                                      :paper-id (graph-paper-id graph file)
                                                      :warrant-floor (:warrant-floor opts)}))
          concept-result (normalize-check graph (concept-check-file file))
          checks (into [anchor-result] (conj closure-results concept-result))
          pass? (every? :pass checks)]
      {:file (.getPath (io/file file))
       :paper-id (graph-paper-id graph file)
       :pass pass?
       :profile (profile file graph anchor-result closure-results concept-result)
       :checks checks})
    (catch Exception e
      {:file (.getPath (io/file file))
       :paper-id nil
       :pass false
       :profile nil
       :checks [{:check :semcheck-load
                 :status :fail
                 :pass false
                 :rate nil
                 :reasons [(.getMessage e)]
                 :per-item []}]})))

(defn aggregate [results opts]
  (let [checks (mapcat :checks results)
        by-check (group-by :check checks)
        summarize (fn [[check rows]]
                    [check {:pass (count (filter #(= :pass (:status %)) rows))
                            :fail (count (filter #(= :fail (:status %)) rows))
                            :na (count (filter #(= :na (:status %)) rows))
                            ;; report-only rows are neither verified nor failed;
                            ;; kept out of :pass so no aggregate mixes them in
                            :report (count (filter #(= :report (:status %)) rows))
                            :rates (vec (keep :rate rows))}])
        failures (vec (filter (complement :pass) results))]
    {:schema :futon6.iatc-semcheck.v1
     :description-first true
     :na-not-fail true
     :thresholds {:anchor-faithfulness {:floor (:anchor-floor opts)
                                        :rate-label "lexical lower bound"
                                        :k (:anchor-k opts)
                                        :tau (:anchor-tau opts)}
                  :warrant-resolution {:floor (:warrant-floor opts)
                                       :justification "loop-run-70b final spread includes 0.0; aggregate 6/28, so default is report-only until calibrated"}}
     :include-attempts (:include-attempts opts)
     :graph-count (count results)
     :pass (empty? failures)
     :failures (mapv #(select-keys % [:file :paper-id]) failures)
     :check-summary (into (sorted-map) (map summarize by-check))
     :graphs results}))

(defn fmt-rate [x]
  (if (some? x) (format "%.3f" (double x)) "n/a"))

(defn print-check [check]
  (println (format "  R2%-1s %-23s %s rate=%s"
                   (case (:check check)
                     :anchor-faithfulness "a"
                     :closure "b"
                     :warrant-resolution "c"
                     :concept-coverage "d"
                     "?")
                   (str (name (:check check))
                        (when (= :anchor-faithfulness (:check check))
                          " (lexical lower bound)"))
                   (str/upper-case (name (:status check)))
                   (fmt-rate (:rate check))))
  (doseq [reason (take 4 (:reasons check))]
    (println "    -" reason)))

(defn print-summary [report]
  (doseq [{:keys [file pass checks profile]} (:graphs report)]
    (println (str (if pass "PASS " "FAIL ") file))
    (doseq [check checks] (print-check check))
    (let [sk (:region-skeleton profile)]
      (println (format "  profile: terms=%d nodes=%d edges=%d holes=%d lines=%s"
                       (count (:imported-terms profile))
                       (:node-count sk)
                       (:edge-count sk)
                       (:hole-count sk)
                       (pr-str (:source-lines sk))))))
  (println)
  (println (format "iatc-semcheck: %d graph(s), %d failing graph(s) -- %s"
                   (:graph-count report)
                   (count (:failures report))
                   (if (:pass report) "PASS" "FAIL"))))

(defn md-check-row [graph check]
  (format "| `%s` | `%s` | `%s` | %s | %d |\n"
          (:paper-id graph)
          (:check check)
          (:status check)
          (fmt-rate (:rate check))
          (count (:reasons check))))

(defn check-by [graph-id check-id report]
  (some (fn [g]
          (when (= graph-id (:paper-id g))
            (some #(when (= check-id (:check %)) %) (:checks g))))
        (:graphs report)))

(defn markdown-report [report]
  (let [rows (apply str
                    (for [g (:graphs report)
                          c (:checks g)]
                      (md-check-row g c)))]
    (str "# IATC semcheck report — loop-run-70b\n\n"
         "- Description-first: yes\n"
         "- N/A != FAIL: yes\n"
         "- Graphs: `" (:graph-count report) "`"
         (when (:include-attempts report) " (includes explicitly requested attempt graph)")
         "\n"
         "- R2a rate label: lexical lower bound\n"
         "- R2c warrant floor: `" (get-in report [:thresholds :warrant-resolution :floor]) "`; "
         (get-in report [:thresholds :warrant-resolution :justification]) "\n"
         "- Overall verdict: **" (if (:pass report) "PASS" "FAIL") "**\n\n"
         "## Ground-truth Anchors\n\n"
         "- `0706.1286`: clean anchor; R2a lexical lower bound `"
         (fmt-rate (:rate (check-by "0706.1286" :anchor-faithfulness report)))
         "`, R2b `" (:status (check-by "0706.1286" :closure report)) "`.\n"
         "- `0709.0248`: R2a-flagged at the proposition anchor; R2a reasons `"
         (count (:reasons (check-by "0709.0248" :anchor-faithfulness report))) "`.\n"
         "- `0708.2185`: R2b-flagged attempt; closure status `"
         (:status (check-by "0708.2185" :closure report)) "`.\n\n"
         "## Check Summary\n\n"
         "| paper | check | status | rate | reasons |\n"
         "|---|---|---:|---:|---:|\n"
         rows
         "\n## Paper-description profiles\n\n"
         (apply str
                (for [g (:graphs report)
                      :let [p (:profile g)
                            sk (:region-skeleton p)]]
                  (str "### `" (:paper-id g) "`\n\n"
                       "- file: `" (:file g) "`\n"
                       "- skeleton: nodes `" (:node-count sk) "`, edges `" (:edge-count sk)
                       "`, holes `" (:hole-count sk) "`, lines `" (pr-str (:source-lines sk)) "`\n"
                       "- imported terms: " (str/join ", " (take 20 (:imported-terms p)))
                       (when (> (count (:imported-terms p)) 20) ", ...")
                       "\n"
                       "- reasoning edges: `" (count (:reasoning p)) "`\n\n"))))))

(defn write-report! [path content]
  (let [f (io/file path)]
    (io/make-parents f)
    (spit f content)))

(defn -main [args]
  (let [{:keys [opts paths]} (parse-args args)
        files (mapcat #(edn-files (:include-attempts opts) %) paths)]
    (when (empty? files)
      (binding [*out* *err*]
        (println "No .edn files found in input paths:" (str/join " " paths)))
      (System/exit 2))
    (let [results (mapv #(check-file opts %) files)
          report (aggregate results opts)]
      (print-summary report)
      (when-let [out (:out opts)]
        (write-report! out (pr-str report)))
      (when-let [out (:md-out opts)]
        (write-report! out (markdown-report report)))
      (System/exit (if (and (:gate opts) (not (:pass report))) 1 0)))))

(when (= *file* (System/getProperty "babashka.file"))
  (-main *command-line-args*))
