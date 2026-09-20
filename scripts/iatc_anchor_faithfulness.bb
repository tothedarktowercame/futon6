#!/usr/bin/env bb
;; Anchor-faithfulness checker for IATC argument graphs.
;;
;; Usage:
;;   bb scripts/iatc_anchor_faithfulness.bb data/iatc-argument-graphs/loop-run-70b
;;   bb scripts/iatc_anchor_faithfulness.bb --k 2 --tau 0.45 --floor 0.30 graph.edn
;;   bb scripts/iatc_anchor_faithfulness.bb --candidates-dir artifacts/candidates graph.edn
;;   bb scripts/iatc_anchor_faithfulness.bb --source paper.json graph.edn

(require '[cheshire.core :as json]
         '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def default-opts
  {:k 2
   :tau 0.45
   :floor 0.30
   :format :text})

(def stopwords
  #{"a" "all" "an" "and" "any" "are" "as" "at" "be" "been" "being" "between" "by"
    "can" "case" "cases" "claim" "claims" "conclusion" "definition" "does"
    "each" "every" "exists" "for" "from" "given" "has" "have" "having" "if"
    "in" "into" "is" "it" "its" "lemma" "let" "like" "line" "lines" "may"
    "include" "not" "of" "on" "or" "ought" "over" "proof" "proposition" "show" "shown"
    "shows" "so" "some" "such" "suppose" "than" "that" "the" "then"
    "there" "these" "this" "those" "to" "using" "we" "where" "which"
    "with" "within"})

(def math-command-stopwords
  #{"begin" "end" "cite" "citep" "citet" "cref" "eqref" "emph" "ensuremath"
    "label" "mathcal" "mathfrak" "mathit" "mathbf" "mathrm" "mathsf"
    "operatorname" "ref" "section" "subsection" "text" "textbf" "textit"
    "textrm"})

(def latex-wrapper-commands
  #{"emph" "ensuremath" "mathbb" "mathcal" "mathfrak" "mathit" "mathbf"
    "mathrm" "mathsf" "operatorname" "text" "textbf" "textit" "textrm"})

(def latex-macro-expansions
  {"G" "G group"
   "Ob" "Ob object"
   "Mor" "Mor morphism"
   "maps" "maps to"})

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/iatc_anchor_faithfulness.bb [--candidates-dir DIR] [--source FILE (explicit 1-based source)] [--k N] [--tau X] [--floor X] [--edn] <graph.edn-or-dir> [...]"))
  (System/exit 2))

(defn parse-int [s flag]
  (try
    (Integer/parseInt s)
    (catch Exception _
      (binding [*out* *err*]
        (println "Bad integer for" flag ":" s))
      (System/exit 2))))

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
    (if (empty? xs)
      (when (empty? paths) (usage!))
      nil)
    (if (empty? xs)
      {:opts opts :paths paths}
      (let [[x & more] xs]
        (case x
          "--marks-dir" (throw (ex-info "Use --candidates-dir for candidate windows, or --source for explicit 1-based source text" {}))
          "--candidates-dir" (do (when (empty? more) (usage!))
                                 (recur (assoc opts :candidates-dir (first more)) paths (rest more)))
          "--source" (do (when (empty? more) (usage!))
                         (recur (assoc opts :source (first more)) paths (rest more)))
          "--k" (do (when (empty? more) (usage!))
                    (recur (assoc opts :k (parse-int (first more) x)) paths (rest more)))
          "--tau" (do (when (empty? more) (usage!))
                      (recur (assoc opts :tau (parse-decimal (first more) x)) paths (rest more)))
          "--floor" (do (when (empty? more) (usage!))
                        (recur (assoc opts :floor (parse-decimal (first more) x)) paths (rest more)))
          "--edn" (recur (assoc opts :format :edn) paths more)
          "--help" (usage!)
          "-h" (usage!)
          (recur opts (conj paths x) more))))))

(defn hidden-attempts-path? [file]
  (some #(= ".attempts" (str %))
        (iterator-seq (.iterator (.toPath (io/file file))))))

(defn edn-files [path]
  (let [f (io/file path)]
    (cond
      (not (.exists f)) []
      (.isDirectory f) (->> (file-seq f)
                            (filter #(.isFile %))
                            (filter #(str/ends-with? (.getName %) ".edn"))
                            ;; H3: .rung2.edn are semcheck REPORTS, not graphs.
                            ;; Scanning them doubled the corpus (98 -> 196) and
                            ;; every sidecar failed with "source text not found",
                            ;; which would have halved the apparent pass rate.
                            (remove #(str/ends-with? (.getName %) ".rung2.edn"))
                            (remove hidden-attempts-path?)
                            (sort-by #(.getPath %)))
      (str/ends-with? (.getName f) ".edn") [f]
      :else [])))

(defn read-one-edn [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (let [form (edn/read {:eof ::eof} r)]
      (if (= ::eof form)
        (throw (ex-info "empty EDN file" {}))
        (let [tail (edn/read {:eof ::eof} r)]
          (when-not (= ::eof tail)
            (throw (ex-info "trailing EDN forms after graph" {:tail tail})))
          form)))))

(defn paper-id [graph file]
  (or (:paper/id graph)
      (some-> (:passage/id graph) (str/split #":") first)
      (str/replace (.getName (io/file file)) #"\.edn$" "")))

(defn read-source-text [source]
  (let [text (slurp source)]
    (if (str/ends-with? (str/lower-case (.getName (io/file source))) ".json")
      (let [parsed (json/parse-string text true)]
        (or (:text parsed)
            (throw (ex-info "JSON source lacks \"text\" field" {:source source}))))
      text)))

(def coordinate-convention
  "Inclusive candidate line labels: source-window split on LF; index 0 has window-lines[0]. No reconstructed-file coordinates.")

(defn candidate-context [graph candidate source]
  (let [[lo hi :as bounds] (:window-lines candidate)
        text (:source-window candidate)]
    (when-not (and (= (:paper/id graph) (:paper-id candidate))
                   (= (:passage/id graph) (:passage-id candidate)))
      (throw (ex-info "candidate identity does not match graph" {:source source})))
    (when-not (and (vector? bounds) (= 2 (count bounds))
                   (every? int? bounds) (<= lo hi) (string? text))
      (throw (ex-info "candidate lacks valid window-lines/source-window" {:source source})))
    (let [lines (vec (str/split text #"\n" -1))]
      (when (> (- hi lo) (dec (count lines)))
        (throw (ex-info "candidate window bounds exceed source-window" {:source source})))
      {:paper-id (:paper-id candidate) :source (str source)
       :line-base lo :line-end hi :lines lines
       :coordinate-convention coordinate-convention})))

(defn load-lines [graph file opts]
  (let [pid (paper-id graph file)]
    (if-let [source (:source opts)]
      ;; Explicit legacy input only: caller certifies these are 1-based file lines.
      {:paper-id pid :source (str source) :line-base 1
       :coordinate-convention "Inclusive 1-based lines in explicitly supplied source"
       :lines (vec (str/split (read-source-text source) #"\n" -1))}
      (let [dir (or (:candidates-dir opts)
                    (io/file (.getParentFile (.getParentFile (io/file file))) "candidates"))
            candidate (io/file dir (str/replace (.getName (io/file file)) #"\.edn$" ".candidate.json"))]
        (when-not (.isFile candidate)
          (throw (ex-info "candidate source-window not found" {:candidate (str candidate)})))
        (candidate-context graph (json/parse-string (slurp candidate) true) candidate)))))

(defn normalize-token [tok]
  (-> tok
      (str/replace #"^\\+" "")
      (str/replace #"[^A-Za-z0-9]+" "")
      str/lower-case))

(defn normalize-latex [text]
  (loop [s (or text "")
         n 0]
    (let [unwrapped (str/replace s #"\\([A-Za-z]+)\{([^{}]*)\}"
                                  (fn [[_ cmd body]]
                                    (if (contains? latex-wrapper-commands cmd)
                                      body
                                      (str " " (get latex-macro-expansions cmd cmd) " " body " "))))]
      (if (or (= s unwrapped) (>= n 6))
        (-> unwrapped
            (str/replace #"\\([A-Za-z]+)"
                         (fn [[_ cmd]]
                           (str " " (get latex-macro-expansions cmd cmd) " ")))
            (str/replace #"[$^_{}]" " ")
            (str/replace #"[\u2192\u27f6\u27f5\u2190]" " to "))
        (recur unwrapped (inc n))))))

(defn keep-token? [tok]
  (let [n (normalize-token tok)]
    (and (>= (count n) 3)
         (not (contains? stopwords n))
         (not (contains? math-command-stopwords n)))))

(defn text-terms [text]
  (let [raw (re-seq #"[A-Za-z][A-Za-z0-9]*" (normalize-latex text))]
    (->> raw
         (map normalize-token)
         (filter keep-token?)
         distinct
         vec)))

(defn span-text
  ([lines line-range]
   (span-text lines line-range 0))
  ([lines [a b] pad]
  (let [n (count lines)
        lo (max 1 (- (or a 1) pad))
        hi (min n (+ (or b n) pad))]
    (if (> lo hi)
      ""
      (str/join "\n" (subvec lines (dec lo) hi))))))

(defn valid-lines? [lines]
  (and (vector? lines)
       (= 2 (count lines))
       (every? int? lines)
       (<= (first lines) (second lines))))

(defn node-item [ctx opts node]
  (let [src-lines (get-in node [:source :lines])
        base (or (:line-base ctx) 1)
        end (or (:line-end ctx) (+ base (dec (count (:lines ctx)))))
        in-window? (and (valid-lines? src-lines)
                        (<= base (first src-lines) (second src-lines) end))
        local-lines (when in-window? (mapv #(+ 1 (- % base)) src-lines))
        terms (text-terms (:text node))
        exact-text (when in-window?
                     (span-text (:lines ctx) local-lines))
        source-text (when in-window?
                      (span-text (:lines ctx) local-lines 1))
        exact-terms (set (text-terms exact-text))
        source-terms (set (text-terms source-text))
        matched (vec (filter source-terms terms))
        missing (vec (remove source-terms terms))
        n-terms (count terms)
        n-matched (count matched)
        fraction (if (pos? n-terms) (/ n-matched n-terms) 0.0)
        scorable? (or (not in-window?) (>= n-terms (:k opts)))
        exact-matched? (boolean (some exact-terms terms))
        faithful? (and in-window? scorable?
                       (or (>= fraction (:tau opts))
                           (and exact-matched?
                                (<= n-terms (:k opts))
                                (>= n-matched (:k opts)))))
        status (cond
                 (not in-window?) :fail
                 (< n-terms (:k opts)) :na
                 faithful? :pass
                 :else :fail)]
    {:id (:id node)
     :kind (:kind node)
     :source {:lines src-lines}
     :anchor-valid in-window?
     :text (:text node)
     :terms terms
     :matched matched
     :missing missing
     :n_terms n-terms
     :n_matched n-matched
     :fraction (double fraction)
     :scorable scorable?
     :status status
     :faithful faithful?}))

(defn check-graph
  "Return {:check :anchor-faithfulness :pass :rate :reasons :per-item}.

  graph may be an EDN map or a graph file. ctx accepts {:paper-id :lines :source};
  File graphs resolve sibling candidates/<stem>.candidate.json by default.
  Candidate labels are inclusive; window text index 0 is window-lines[0].
  In-memory contexts default to inclusive 1-based lines, or specify :line-base.
  Optional opts: {:k 2 :tau 0.45 :floor 0.30 :candidates-dir ... :source ...}."
  ([graph ctx]
   (check-graph graph ctx default-opts))
  ([graph ctx opts]
   (let [opts (merge default-opts opts)
         file (when-not (map? graph) (io/file graph))
         graph-map (if (map? graph) graph (read-one-edn file))
         ctx (if (:lines ctx)
               ctx
               (if file
                 (merge ctx (load-lines graph-map file opts))
                 (throw (ex-info "ctx must include :lines when graph is a map" {}))))
         items (->> (:nodes graph-map)
                    (mapv #(node-item ctx opts %)))
         scored (filter :scorable items)
         faithful (filter :faithful scored)
         rate (if (seq scored) (/ (count faithful) (count scored)) 1.0)
         flagged (vec (remove :faithful scored))
         reasons (mapv (fn [item]
                         {:id (:id item)
                          :source (:source item)
                          :reason (if-not (:anchor-valid item)
                                    "anchor is outside candidate/source line bounds"
                                    (str "matched " (:n_matched item) "/" (:n_terms item)
                                       " key terms below v2 faithfulness thresholds"
                                       " k=" (:k opts) " tau=" (:tau opts)))
                          :missing (:missing item)})
                       flagged)]
     {:check :anchor-faithfulness
      :paper-id (:paper-id ctx)
      :coordinate-convention (or (:coordinate-convention ctx) "Inclusive 1-based lines in supplied context")
      :source (:source ctx)
      :pass (and (every? :anchor-valid items) (>= rate (:floor opts)))
      :rate (double rate)
      :reasons reasons
      :per-item items})))

(defn check-file [opts file]
  (try
    (let [graph (read-one-edn file)
          ctx (load-lines graph file opts)
          result (check-graph graph ctx opts)]
      (assoc result :file (.getPath file)))
    (catch Exception e
      {:check :anchor-faithfulness
       :file (.getPath file)
       :pass false
       :rate nil
       :status :input-error
       :reasons [{:reason (.getMessage e)
                  :data (ex-data e)}]
       :per-item []})))

(defn print-result [result]
  (println (format "%s %s rate=%s flagged=%d"
                   (if (:pass result) "PASS" "FAIL")
                   (:file result)
                   (if (number? (:rate result)) (format "%.3f" (:rate result)) "unscored")
                   (count (:reasons result))))
  (doseq [{:keys [id source reason missing]} (:reasons result)]
    (println (str "  " id " " (pr-str (:lines source)) " :: " reason
                  " missing=" (pr-str (take 8 missing))))))

(defn -main [args]
  (let [{:keys [opts paths]} (parse-args args)
        files (mapcat edn-files paths)]
    (when (empty? files)
      (binding [*out* *err*]
        (println "No .edn files found in input paths:" (str/join " " paths)))
      (System/exit 2))
    (let [results (mapv #(check-file opts %) files)
          rates (keep :rate results)]
      (if (= :edn (:format opts))
        (prn results)
        (do
          (doseq [r results] (print-result r))
          (println)
          (println (format "anchor-faithfulness: %d graph(s), %d scored, %d input errors, min=%s max=%s floor=%.3f -- %s"
                           (count results) (count rates) (- (count results) (count rates))
                           (if (seq rates) (format "%.3f" (double (apply min rates))) "unscored")
                           (if (seq rates) (format "%.3f" (double (apply max rates))) "unscored")
                           (double (:floor opts))
                           (if (every? :pass results) "PASS" "FAIL")))
          (println "Coordinates: candidate source-window labels (inclusive); explicit --source uses 1-based file lines.")
          (println "Scores measure token overlap with one neighbor line of tolerance, not exact highlighting.")))
      (System/exit (if (every? :pass results) 0 1)))))

(when (= *file* (System/getProperty "babashka.file"))
  (-main *command-line-args*))
