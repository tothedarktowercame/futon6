#!/usr/bin/env bb
;; Anchor-faithfulness checker for IATC argument graphs.
;;
;; Usage:
;;   bb scripts/iatc_anchor_faithfulness.bb data/iatc-argument-graphs/loop-run-70b
;;   bb scripts/iatc_anchor_faithfulness.bb --k 2 --tau 0.45 --floor 0.30 graph.edn
;;   bb scripts/iatc_anchor_faithfulness.bb --marks-dir data/showcases/ct-anatomy/golden graph.edn
;;   bb scripts/iatc_anchor_faithfulness.bb --source paper.json graph.edn

(require '[cheshire.core :as json]
         '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def default-opts
  {:marks-dir "data/showcases/ct-anatomy/golden"
   :k 2
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
    (println "Usage: bb scripts/iatc_anchor_faithfulness.bb [--marks-dir DIR] [--source FILE] [--k N] [--tau X] [--floor X] [--edn] <graph.edn-or-dir> [...]"))
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
          "--marks-dir" (do (when (empty? more) (usage!))
                            (recur (assoc opts :marks-dir (first more)) paths (rest more)))
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

(defn source-file [marks-dir pid]
  (io/file marks-dir (str "fable-" pid "-dp-emacs.json")))

(defn read-source-text [source]
  (let [text (slurp source)]
    (if (str/ends-with? (str/lower-case (.getName (io/file source))) ".json")
      (let [parsed (json/parse-string text true)]
        (or (:text parsed)
            (throw (ex-info "JSON source lacks \"text\" field" {:source source}))))
      text)))

(defn load-lines [graph file opts]
  (let [pid (paper-id graph file)
        source (or (:source opts) (.getPath (source-file (:marks-dir opts) pid)))
        f (io/file source)]
    (when-not (.exists f)
      (throw (ex-info "source text not found" {:source source :paper-id pid})))
    {:paper-id pid
     :source (.getPath f)
     :lines (vec (str/split-lines (read-source-text f)))}))

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
        terms (text-terms (:text node))
        exact-text (when (valid-lines? src-lines)
                     (span-text (:lines ctx) src-lines))
        source-text (when (valid-lines? src-lines)
                      (span-text (:lines ctx) src-lines 1))
        exact-terms (set (text-terms exact-text))
        source-terms (set (text-terms source-text))
        matched (vec (filter source-terms terms))
        missing (vec (remove source-terms terms))
        n-terms (count terms)
        n-matched (count matched)
        fraction (if (pos? n-terms) (/ n-matched n-terms) 0.0)
        scorable? (and (valid-lines? src-lines) (>= n-terms (:k opts)))
        exact-matched? (boolean (some exact-terms terms))
        faithful? (and scorable?
                       (or (>= fraction (:tau opts))
                           (and exact-matched?
                                (<= n-terms (:k opts))
                                (>= n-matched (:k opts)))))
        status (cond
                 (not (valid-lines? src-lines)) :na
                 (< n-terms (:k opts)) :na
                 faithful? :pass
                 :else :fail)]
    {:id (:id node)
     :kind (:kind node)
     :source {:lines src-lines}
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
  when graph is a file and ctx lacks :lines, the source text is resolved from
  data/showcases/ct-anatomy/golden/fable-<id>-dp-emacs.json unless overridden.
  Optional opts: {:k 2 :tau 0.45 :floor 0.30 :marks-dir ... :source ...}."
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
                          :reason (str "matched " (:n_matched item) "/" (:n_terms item)
                                       " key terms below v2 faithfulness thresholds"
                                       " k=" (:k opts) " tau=" (:tau opts))
                          :missing (:missing item)})
                       flagged)]
     {:check :anchor-faithfulness
      :paper-id (:paper-id ctx)
      :source (:source ctx)
      :pass (>= rate (:floor opts))
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
       :rate 0.0
       :reasons [{:reason (.getMessage e)
                  :data (ex-data e)}]
       :per-item []})))

(defn print-result [result]
  (println (format "%s %s rate=%.3f flagged=%d"
                   (if (:pass result) "PASS" "FAIL")
                   (:file result)
                   (:rate result)
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
    (let [results (mapv #(check-file opts %) files)]
      (if (= :edn (:format opts))
        (prn results)
        (do
          (doseq [r results] (print-result r))
          (println)
          (println (format "anchor-faithfulness: %d graph(s), min=%.3f max=%.3f floor=%.3f -- %s"
                           (count results)
                           (double (apply min (map :rate results)))
                           (double (apply max (map :rate results)))
                           (double (:floor opts))
                           (if (every? :pass results) "PASS" "FAIL")))))
      (System/exit (if (every? :pass results) 0 1)))))

(when (= *file* (System/getProperty "babashka.file"))
  (-main *command-line-args*))
