#!/usr/bin/env bb
;; tag-patterns.bb — Tag PlanetMath .tex entries with informal reasoning patterns
;;
;; Reads patterns-index.tsv (math-informal/* namespace) and PlanetMath .tex files,
;; scores each entry body against each pattern's hotwords, outputs EDN with matches.
;;
;; Usage:
;;   bb scripts/tag-patterns.bb <tex-dir> [<tex-dir2> ...]
;;   bb scripts/tag-patterns.bb ~/code/planetmath/18_Category_theory_homological_algebra/
;;
;; Scoring uses the same logic as futon3a notions.clj:
;;   score = 2.0 * |hotword hits| + 0.5 * |rationale token hits|
;;
;; Output: EDN to stdout, one map per entry with :entry, :matches, :file

(require '[clojure.string :as str]
         '[clojure.java.io :as io])

;; --- Config ---

(def ^:private tsv-path
  (or (System/getenv "PATTERNS_INDEX")
      (str (System/getenv "HOME") "/code/futon3/resources/sigils/patterns-index.tsv")))

(def ^:private min-score 4.0)
(def ^:private namespace-filter "math-informal/")

;; --- TSV loading ---

(defn parse-tsv-line [line]
  (let [parts (str/split line #"\t" -1)]
    (when (>= (count parts) 5)
      {:id       (nth parts 0)
       :tokipona (nth parts 1)
       :sigil    (nth parts 2)
       :rationale (nth parts 3)
       :hotwords (set (map str/trim (str/split (nth parts 4) #",")))})))

(defn load-patterns [path ns-filter]
  (->> (str/split-lines (slurp path))
       (drop 1)
       (map parse-tsv-line)
       (remove nil?)
       (filter #(str/starts-with? (:id %) ns-filter))
       vec))

;; --- Tokenization (matches notions.clj) ---

(defn tokenize [text]
  (->> (str/split (str/lower-case (or text "")) #"[^a-z0-9]+")
       (remove str/blank?)
       (remove #(< (count %) 3))
       set))

;; --- Scoring ---

(defn score-pattern [tokens pattern]
  (let [hotword-hits   (count (clojure.set/intersection tokens (:hotwords pattern)))
        rat-tokens     (tokenize (:rationale pattern))
        rat-hits       (count (clojure.set/intersection tokens rat-tokens))
        score          (+ (* 2.0 hotword-hits) (* 0.5 rat-hits))]
    (when (pos? score)
      {:pattern  (:id pattern)
       :score    score
       :hotword-hits hotword-hits
       :rationale-hits rat-hits})))

;; --- .tex parsing ---

(defn parse-tex [path]
  (let [content (slurp path)
        lines   (str/split-lines content)
        ;; Extract canonical name
        canon   (->> lines
                     (some #(when-let [[_ name] (re-matches #"\\pmcanonicalname\{(.+)\}" (str/trim %))]
                              name)))
        title   (->> lines
                     (some #(when-let [[_ t] (re-matches #"\\pmtitle\{(.+)\}" (str/trim %))]
                              t)))
        pm-type (->> lines
                     (some #(when-let [[_ t] (re-matches #"\\pmtype\{(.+)\}" (str/trim %))]
                              t)))
        ;; Extract body between \begin{document} and \end{document}
        in-doc  (->> lines
                     (drop-while #(not (str/includes? % "\\begin{document}")))
                     rest
                     (take-while #(not (str/includes? % "\\end{document}")))
                     (str/join "\n"))]
    {:file  (.getName (io/file path))
     :canon canon
     :title title
     :type  pm-type
     :body  in-doc}))

;; --- Main ---

(defn tag-entry [patterns entry]
  (let [tokens  (tokenize (:body entry))
        matches (->> patterns
                     (keep #(score-pattern tokens %))
                     (filter #(>= (:score %) min-score))
                     (sort-by :score >)
                     vec)]
    (when (seq matches)
      {:entry  (or (:canon entry) (:file entry))
       :title  (:title entry)
       :type   (:type entry)
       :file   (:file entry)
       :matches matches})))

(defn process-dir [patterns dir]
  (let [tex-files (->> (file-seq (io/file dir))
                       (filter #(str/ends-with? (.getName %) ".tex"))
                       sort)]
    (->> tex-files
         (map #(parse-tex (.getPath %)))
         (keep #(tag-entry patterns %))
         vec)))

(let [dirs    (or (seq *command-line-args*)
                  [(str (System/getenv "HOME") "/code/planetmath/18_Category_theory_homological_algebra/")])
      patterns (load-patterns tsv-path namespace-filter)
      _        (binding [*out* *err*]
                 (println (str "Loaded " (count patterns) " math-informal patterns")))
      results  (mapcat #(process-dir patterns %) dirs)
      _        (binding [*out* *err*]
                 (println (str "Tagged " (count results) " entries across " (count dirs) " directories")))]

  ;; Summary stats to stderr
  (binding [*out* *err*]
    (let [pattern-freq (->> results
                            (mapcat :matches)
                            (map :pattern)
                            frequencies
                            (sort-by val >))]
      (println "\nPattern frequency:")
      (doseq [[p freq] pattern-freq]
        (println (str "  " p ": " freq)))))

  ;; EDN to stdout
  (prn {:generated   (str (java.time.Instant/now))
        :source-dirs (vec dirs)
        :pattern-ns  namespace-filter
        :min-score   min-score
        :entry-count (count results)
        :entries     (vec results)}))
