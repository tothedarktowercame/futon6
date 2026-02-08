#!/usr/bin/env bb
;; spot-terms.bb — Classical mathematical term spotter ("better NNexus")
;;
;; Reads a term dictionary (TSV) and spots terms in text documents.
;; Works on PlanetMath .tex files or StackExchange JSON entities.
;;
;; Usage:
;;   # Spot terms in PlanetMath .tex files
;;   bb scripts/spot-terms.bb --mode tex ~/code/planetmath/18_Category*/
;;
;;   # Spot terms in SE JSON (from process-stackexchange.py output)
;;   bb scripts/spot-terms.bb --mode se-json data/se-physics.json
;;
;;   # Use a custom dictionary
;;   bb scripts/spot-terms.bb --dict data/pm-terms.tsv --mode tex ~/code/planetmath/18*/
;;
;; The "negative space" — text regions not covered by any term match —
;; is where informal reasoning patterns live. This script reports both
;; what it finds and what's left over.
;;
;; Output: EDN to stdout with term hits and negative-space stats.

(require '[clojure.string :as str]
         '[clojure.java.io :as io])

;; --- Arg parsing ---

(defn parse-args [args]
  (loop [args args
         opts {:mode "tex"
               :dict (str (System/getenv "HOME")
                          "/code/futon6/data/pm-terms.tsv")
               :limit nil
               :paths []}]
    (if (empty? args)
      opts
      (let [[a & rest] args]
        (cond
          (= a "--mode")   (recur (next rest) (assoc opts :mode (first rest)))
          (= a "--dict")   (recur (next rest) (assoc opts :dict (first rest)))
          (= a "--limit")  (recur (next rest) (assoc opts :limit (parse-long (first rest))))
          :else            (recur rest (update opts :paths conj a)))))))

;; --- Term dictionary ---

(defn load-term-dict [path]
  "Load TSV dictionary: term_lower, term, confidence, defined_in"
  (let [lines (str/split-lines (slurp path))
        rows  (drop 1 lines)] ; skip header
    (->> rows
         (map #(str/split % #"\t" -1))
         (filter #(>= (count %) 3))
         (map (fn [[term-lower term confidence & rest]]
                {:term-lower (str/trim term-lower)
                 :term       (str/trim term)
                 :confidence (str/trim confidence)
                 :defined-in (when (seq rest) (str/trim (first rest)))}))
         ;; Filter out overly generic single-word terms
         (remove #(let [words (str/split (:term-lower %) #"\s+")]
                    (and (= 1 (count words))
                         (< (count (:term-lower %)) 6)
                         (not (#{"monad" "sheaf" "topos" "fiber" "nerve"
                                 "stalk" "gerbe" "braid" "trace" "wedge"}
                               (:term-lower %))))))
         ;; Filter common English
         (remove #(#{"at all" "define" "relative" "equivalent" "opposite"
                     "independent" "interesting" "relation" "structure"
                     "complex" "moment" "numbers" "nature" "square"
                     "section" "allows" "analysis" "property" "context"
                     "parallel" "consistent" "eventually" "separate"
                     "levels" "represent" "stable" "approach" "surface"
                     "random" "action" "string" "constant" "observe"
                     "formula" "support" "target" "degree" "potential"
                     "critical" "particle" "creation" "principle"
                     "evolution" "classical" "operator" "symmetry"
                     "measurement" "variables" "distribution"
                     "invariant" "representation"}
                   (:term-lower %)))
         vec)))

;; --- Tokenization ---

(defn tokenize [text]
  (->> (str/split (str/lower-case (or text "")) #"[^a-z0-9]+")
       (remove str/blank?)
       (remove #(< (count %) 3))
       set))

;; --- Multi-word term matching ---

(defn term-words [term-lower]
  (str/split term-lower #"\s+"))

(defn spot-terms
  "Spot dictionary terms in text. Returns set of matched term-lowers."
  [terms text]
  (let [text-lower (str/lower-case (or text ""))
        text-tokens (tokenize text)
        ;; Partition terms into single-word and multi-word
        singles (filter #(= 1 (count (term-words (:term-lower %)))) terms)
        multis  (filter #(> (count (term-words (:term-lower %))) 1) terms)]
    (into
     ;; Single-word: direct token lookup
     (->> singles
          (filter #(contains? text-tokens (:term-lower %)))
          (map :term-lower)
          set)
     ;; Multi-word: check all content words present, then phrase search
     (->> multis
          (filter (fn [t]
                    (let [words (term-words (:term-lower t))
                          content-words (filter #(>= (count %) 3) words)]
                      (and (every? #(contains? text-tokens %) content-words)
                           (str/includes? text-lower (:term-lower t))))))
          (map :term-lower)))))

;; --- .tex parsing ---

(defn parse-tex [path]
  (let [content (slurp path)
        lines   (str/split-lines content)
        canon   (->> lines
                     (some #(when-let [[_ name] (re-matches #"\\pmcanonicalname\{(.+)\}" (str/trim %))]
                              name)))
        title   (->> lines
                     (some #(when-let [[_ t] (re-matches #"\\pmtitle\{(.+)\}" (str/trim %))]
                              t)))
        body    (->> lines
                     (drop-while #(not (str/includes? % "\\begin{document}")))
                     rest
                     (take-while #(not (str/includes? % "\\end{document}")))
                     (str/join "\n"))]
    {:id    (or canon (.getName (io/file path)))
     :title title
     :body  body
     :file  (.getName (io/file path))}))

;; --- SE JSON parsing ---

(defn load-se-json [path limit]
  "Load SE JSON and extract entities. Uses streaming for memory."
  (binding [*out* *err*]
    (println (str "Loading SE JSON from " path "...")))
  (let [data    (-> (slurp path)
                    (cheshire.core/parse-string true))
        entities (:entities data)
        entities (if limit (take limit entities) entities)]
    (->> entities
         (map (fn [e]
                {:id    (get e (keyword "entity/id") "?")
                 :title (or (:title e) "")
                 :body  (str (or (:question-body e) "")
                             "\n"
                             (or (:answer-body e) ""))
                 :tags  (or (:tags e) [])})))))

;; --- Negative space analysis ---

(defn negative-space-tokens
  "Return tokens from text that are NOT part of any matched term."
  [text matched-terms]
  (let [all-tokens (tokenize text)
        term-tokens (->> matched-terms
                         (mapcat #(str/split % #"\s+"))
                         (map str/lower-case)
                         set)]
    (clojure.set/difference all-tokens term-tokens)))

;; --- Main ---

(try
  (require '[cheshire.core])
  (catch Exception _))

(let [opts     (parse-args *command-line-args*)
      dict     (load-term-dict (:dict opts))
      _        (binding [*out* *err*]
                 (println (str "Loaded " (count dict) " terms from " (:dict opts))))

      entries  (case (:mode opts)
                 "tex"     (->> (:paths opts)
                                (mapcat (fn [dir]
                                          (->> (file-seq (io/file dir))
                                               (filter #(str/ends-with? (.getName %) ".tex"))
                                               sort
                                               (map #(parse-tex (.getPath %))))))
                                vec)
                 "se-json" (vec (load-se-json (first (:paths opts)) (:limit opts)))
                 (do (binding [*out* *err*]
                       (println "Unknown mode:" (:mode opts)))
                     (System/exit 1)))

      _        (binding [*out* *err*]
                 (println (str "Processing " (count entries) " entries...")))

      ;; Spot terms
      results  (->> entries
                    (pmap (fn [entry]
                            (let [matched (spot-terms dict (:body entry))
                                  neg-tokens (negative-space-tokens (:body entry) matched)]
                              {:id        (:id entry)
                               :title     (:title entry)
                               :terms     (vec (sort matched))
                               :term-count (count matched)
                               :neg-token-count (count neg-tokens)})))
                    vec)

      ;; Stats
      total    (count results)
      with-terms (count (filter #(pos? (:term-count %)) results))
      term-freq  (->> results
                      (mapcat :terms)
                      frequencies
                      (sort-by val >))]

  ;; Summary to stderr
  (binding [*out* *err*]
    (println (str "\n=== Term Spotting Results ==="))
    (println (str "Entries processed: " total))
    (println (str "With term matches: " with-terms " (" (Math/round (* 100.0 (/ with-terms total))) "%)"))
    (println (str "No matches (negative space): " (- total with-terms)))
    (println (str "Distinct terms spotted: " (count term-freq)))
    (println (str "\nTop 30 terms:"))
    (doseq [[term freq] (take 30 term-freq)]
      (println (str "  " (format "%5d" freq) "  " term)))

    ;; Negative space distribution
    (let [neg-counts (map :neg-token-count results)
          avg-neg (/ (reduce + neg-counts) (max 1 total))]
      (println (str "\nNegative space: avg " (Math/round (double avg-neg)) " unmatched tokens/entry"))))

  ;; EDN to stdout
  (prn {:generated   (str (java.time.Instant/now))
        :dict        (:dict opts)
        :mode        (:mode opts)
        :term-count  (count dict)
        :entry-count total
        :matched     with-terms
        :unmatched   (- total with-terms)
        :top-terms   (vec (take 50 term-freq))
        :entries     (vec results)}))
