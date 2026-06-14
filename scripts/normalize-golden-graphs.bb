#!/usr/bin/env bb
;; Normalize golden anatomy graphs (holes/golden-graphs/*.edn) to SCHEMA.md v1:
;; 1. harmonize edge kinds to the controlled vocabulary (:kind-original kept);
;; 2. assign node :satiety (recognizers = scopes; hungry/full).
;; Idempotent; preserves the ;; comment header of each file.

(require '[clojure.edn :as edn]
         '[clojure.pprint :as pp]
         '[clojure.string :as str]
         '[clojure.java.io :as io])

(def kind-map
  {:applies :parses, :applies-operator :parses, :subscript-parametrizes :parses
   :exists :bind, :considers-as :bind
   :display-definition :defines, :definition :defines, :denotes :defines
   :designates :defines, :definition-condition :defines, :defines-on :defines
   :requires :equips
   :constrain :constrains, :membership :constrains, :typed-arrow :constrains
   :statement :states, :claims-equation :states
   :iff :logical, :iff-expansion :logical
   :obligation-statement :decomposes-into
   :equational-step :proof-step, :proves-by-diagram-chain :proof-step
   :monomorphism-cancel :proof-step, :weakens-local-hypothesis :proof-step
   :concludes :proof-step, :supports :proof-step
   :references :refers, :refers-to :refers})

(def connective-of {:iff :iff, :iff-expansion :iff})

(defn harmonize-edge [e]
  (let [k (:kind e)
        k' (get kind-map k k)]
    (cond-> e
      (not= k k') (assoc :kind k' :kind-original k)
      (connective-of k) (assoc :connective (connective-of k)))))

(defn promise-node-ids [edges]
  (set (for [e edges
             :when (= :promises (:kind e))
             end (:ends e)
             :when (#{:heading :promise} (:role end))]
         (:node end))))

(defn node-satiety [n promised?]
  (cond
    (:satiety n) (:satiety n)
    (promised? (:id n)) {:hungry-for :payoff}
    (= :pointer (:canon n)) {:hungry-for :canon}
    (= :unresolved (:bundling n)) {:hungry-for :bundling}
    (true? (:parse-incomplete n)) {:hungry-for :parse}
    :else :full))

(defn normalize [g]
  (let [edges (mapv harmonize-edge (:hyperedges g))
        promised? (promise-node-ids edges)]
    (assoc g
           :schema "golden-graphs/SCHEMA.md v1"
           :hyperedges edges
           :nodes (mapv #(assoc % :satiety (node-satiety % promised?))
                        (:nodes g)))))

(defn comment-header [text]
  (->> (str/split-lines text)
       (take-while #(or (str/starts-with? % ";;") (str/blank? %)))
       (str/join "\n")))

(defn process! [f]
  (let [text (slurp f)
        g (edn/read-string text)
        g' (normalize g)
        header (comment-header text)]
    (spit f (str header (when (seq header) "\n")
                 (with-out-str (pp/pprint g'))))
    (let [hungry (count (filter #(not= :full (:satiety %)) (:nodes g')))
          mapped (count (filter :kind-original (:hyperedges g')))]
      (println (format "%-50s kinds-mapped=%d hungry=%d/%d"
                       (.getName (io/file f)) mapped hungry
                       (count (:nodes g')))))))

(let [dir (or (first *command-line-args*) "holes/golden-graphs")]
  (doseq [f (sort (filter #(str/ends-with? (.getName %) ".edn")
                          (file-seq (io/file dir))))]
    (process! (.getPath f))))
