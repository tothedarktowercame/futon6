#!/usr/bin/env bb
;; build-ner-kernel.bb — Build the NER kernel from multiple sources
;;
;; Merges term dictionaries from PlanetMath, StackExchange tags, and
;; SE body extraction into a single kernel for the superpod run.
;;
;; The kernel is the classical (non-neural) term dictionary that will
;; be sent alongside the SE data dump. On the superpod, it drives:
;;   1. Classical term spotting (CPU, fast) — Tier 1
;;   2. Scope detection (CPU, regex) — Let/Define/Assume openers
;;   3. Negative space identification — text NOT covered by terms
;;
;; The negative space is where informal reasoning patterns live.
;; The superpod's GPU stages then handle:
;;   4. Binding type classification (GPU, Llama-3) — Tier 2
;;   5. Relation extraction within scopes (GPU) — Tier 3
;;
;; Usage:
;;   bb scripts/build-ner-kernel.bb [--pm-dir DIR] [--se-json FILE] [--out DIR]
;;
;; Output:
;;   kernel/terms.tsv          — merged term dictionary
;;   kernel/scope-patterns.edn — scope-opening regex patterns
;;   kernel/stats.edn          — kernel statistics

(require '[clojure.string :as str]
         '[clojure.java.io :as io]
         '[cheshire.core :as ch])

(def default-pm-dir (str (System/getenv "HOME") "/code/planetmath"))
(def default-out-dir "data/ner-kernel")

;; --- Arg parsing ---
(defn parse-args [args]
  (loop [args args
         opts {:pm-dir default-pm-dir
               :se-json nil
               :out    default-out-dir}]
    (if (empty? args)
      opts
      (let [[a & rest] args]
        (cond
          (= a "--pm-dir")  (recur (next rest) (assoc opts :pm-dir (first rest)))
          (= a "--se-json") (recur (next rest) (assoc opts :se-json (first rest)))
          (= a "--out")     (recur (next rest) (assoc opts :out (first rest)))
          :else             (recur rest opts))))))

;; --- PlanetMath extraction ---
(defn extract-pm-terms [pm-dir]
  (let [dirs (->> (file-seq (io/file pm-dir))
                  (filter #(.isDirectory %))
                  (filter #(re-matches #"\d+_.*" (.getName %))))
        terms (atom {})]
    (doseq [d dirs]
      (doseq [f (->> (file-seq d)
                     (filter #(str/ends-with? (.getName %) ".tex")))]
        (let [raw (slurp (.getPath f))
              ;; pmdefines
              defines (re-seq #"\\pmdefines\{([^}]+)\}" raw)
              ;; pmsynonym
              synonyms (re-seq #"\\pmsynonym\{([^}]*)\}\{[^}]*\}" raw)
              ;; pmtitle
              title (second (re-find #"\\pmtitle\{([^}]+)\}" raw))
              ;; pmcanonicalname
              canon (second (re-find #"\\pmcanonicalname\{([^}]+)\}" raw))]
          ;; Add title
          (when (and title (>= (count title) 3))
            (swap! terms assoc (str/lower-case title)
                   {:term title :source "pm-title" :canon canon}))
          ;; Add defines
          (doseq [[_ term] defines]
            (when (>= (count (str/trim term)) 3)
              (swap! terms assoc (str/lower-case (str/trim term))
                     {:term (str/trim term) :source "pm-defines" :canon canon})))
          ;; Add synonyms
          (doseq [[_ syn] synonyms]
            (when (and syn (>= (count (str/trim syn)) 3))
              (swap! terms assoc (str/lower-case (str/trim syn))
                     {:term (str/trim syn) :source "pm-synonym" :canon canon}))))))
    @terms))

;; --- SE tag extraction ---
(defn extract-se-tags [se-json-path]
  (try
    (let [data (ch/parse-string (slurp se-json-path) true)
          entities (:entities data)
          tag-freq (frequencies (mapcat #(or (:tags %) []) entities))
          ;; meta-tags to skip
          skip? #{"homework-and-exercises" "resource-recommendations" "faq"
                  "soft-question" "popular-science" "terminology" "notation"
                  "conventions" "units" "estimation" "everyday-life"
                  "history" "education" "reference-frames"}]
      (->> tag-freq
           (filter (fn [[tag count]] (and (>= count 5) (not (skip? tag)))))
           (map (fn [[tag count]]
                  [(str/replace tag "-" " ")
                   {:term (str/replace tag "-" " ")
                    :source "se-tag"
                    :count count}]))
           (into {})))
    (catch Exception e
      (binding [*out* *err*]
        (println (str "Warning: could not load SE JSON: " (.getMessage e))))
      {})))

;; --- Scope patterns ---
(def scope-patterns
  [{:type     "let-binding"
    :regex    "\\bLet\\s+\\$[^$]+\\$\\s+(be|denote)"
    :captures ["symbol" "type"]
    :example  "Let $G$ be a Lie group"}
   {:type     "define"
    :regex    "\\bDefine\\s+\\$[^$]+\\$\\s*(:=|=|\\\\equiv)"
    :captures ["symbol" "value"]
    :example  "Define $T_a := R(t_a)$"}
   {:type     "assume"
    :regex    "\\b(Assume|Suppose)\\s+(that\\s+)?\\$"
    :captures ["condition"]
    :example  "Assume that $f$ is continuous"}
   {:type     "consider"
    :regex    "\\bConsider\\s+(a|an|the|some)?\\s*\\$?[^$.]{0,60}"
    :captures ["object"]
    :example  "Consider a vector space $V$"}
   {:type     "for-any"
    :regex    "\\b(for\\s+)?(any|every|each|all)\\s+\\$[^$]+\\$"
    :captures ["quantifier" "symbol"]
    :example  "for every $\\epsilon > 0$"}
   {:type     "where-binding"
    :regex    "\\bwhere\\s+\\$[^$]+\\$\\s+(is|denotes|represents)"
    :captures ["symbol" "description"]
    :example  "where $\\hbar$ is the reduced Planck constant"}
   {:type     "set-notation"
    :regex    "\\$[^$]*\\\\in\\s+[^$]+\\$"
    :captures ["element" "set"]
    :example  "$x \\in X$"}])

;; --- Junk filter ---
(def stop-terms
  #{"the" "set" "via" "see" "note" "remark" "remarks" "example" "examples"
    "definition" "proof" "references" "bibliography" "properties"
    "introduction" "background" "notation" "at all"})

(defn junk? [term]
  (or (< (count term) 3)
      (> (count term) 100)
      (re-matches #"^\d+$" term)
      (contains? stop-terms (str/lower-case term))
      (str/starts-with? term "$")
      (str/starts-with? term "\\")))

;; --- Main ---
(let [opts    (parse-args *command-line-args*)
      out-dir (io/file (:out opts))
      _       (.mkdirs out-dir)

      ;; Extract from PlanetMath
      pm-terms (extract-pm-terms (:pm-dir opts))
      _        (binding [*out* *err*]
                 (println (str "PlanetMath terms: " (count pm-terms))))

      ;; Extract from SE (if provided)
      se-terms (if (:se-json opts)
                 (extract-se-tags (:se-json opts))
                 {})
      _        (binding [*out* *err*]
                 (println (str "SE tag terms: " (count se-terms))))

      ;; Merge (PM takes precedence)
      merged   (merge se-terms pm-terms)
      cleaned  (->> merged
                    (remove (fn [[k v]] (junk? k)))
                    (into (sorted-map)))
      _        (binding [*out* *err*]
                 (println (str "Merged kernel: " (count cleaned) " terms")))]

  ;; Write terms TSV
  (let [tsv-path (io/file out-dir "terms.tsv")]
    (spit tsv-path
          (str "term_lower\tterm\tsource\tcanon_or_count\n"
               (str/join "\n"
                 (map (fn [[k v]]
                        (str k "\t"
                             (:term v) "\t"
                             (:source v) "\t"
                             (or (:canon v) (:count v) "")))
                      cleaned))))
    (binding [*out* *err*]
      (println (str "Written " tsv-path))))

  ;; Write scope patterns
  (let [edn-path (io/file out-dir "scope-patterns.edn")]
    (spit edn-path (pr-str scope-patterns))
    (binding [*out* *err*]
      (println (str "Written " edn-path))))

  ;; Write stats
  (let [stats {:generated    (str (java.time.Instant/now))
               :pm-dir       (:pm-dir opts)
               :se-json      (:se-json opts)
               :total-terms  (count cleaned)
               :pm-terms     (count (filter #(str/starts-with? (or (:source (val %)) "") "pm") cleaned))
               :se-terms     (count (filter #(= "se-tag" (:source (val %))) cleaned))
               :scope-patterns (count scope-patterns)}
        stats-path (io/file out-dir "stats.edn")]
    (spit stats-path (pr-str stats))
    (binding [*out* *err*]
      (println (str "Written " stats-path)))
    (prn stats)))
