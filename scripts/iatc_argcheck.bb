#!/usr/bin/env bb
;; Strict checker for IATC argument-graph EDN files.
;;
;; Usage:
;;   bb scripts/iatc_argcheck.bb path/to/graph.edn
;;   bb scripts/iatc_argcheck.bb path/to/graphs/
;;   bb scripts/iatc_argcheck.bb --include-attempts path/to/graphs/
;;
;; The checker accepts the canonical edge shape from the §6 spec
;;   {:from :x :to :y :role :conclusion ...}
;; and the hand-built seed infer-edge shape
;;   {:premise [:p] :warrant {...} :conclusion :c ...}.
;; It validates references uniformly over both encodings and exits nonzero
;; when any graph fails any gate.

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def node-kinds #{:claim :ref :object :definition :warrant :meta})

(def endpoint-keys
  [:from :to :given :premise :conclusion :assume :contradicts :depends-on :meta])

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/iatc_argcheck.bb [--include-attempts] <file-or-directory> [...]"))
  (System/exit 2))

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

(defn read-one-edn [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (let [form (edn/read {:eof ::eof} r)]
      (if (= ::eof form)
        (throw (ex-info "empty EDN file" {}))
        (let [tail (edn/read {:eof ::eof} r)]
          (when-not (= ::eof tail)
            (throw (ex-info "trailing EDN forms after graph" {:tail tail})))
          form)))))

(defn locus [file graph x]
  (let [src (:source x)
        lines (:lines src)]
    (str (.getPath file)
         (when-let [pid (:passage/id graph)] (str " passage=" pid))
         (when (and (vector? lines) (seq lines))
           (str " lines=" (pr-str lines)))
         (when-let [id (:id x)] (str " id=" id)))))

(defn fail-entry [gate file graph x reason]
  {:gate gate
   :file (.getPath file)
   :locus (locus file graph x)
   :reason reason})

(defn valid-lines? [lines]
  (and (vector? lines)
       (= 2 (count lines))
       (every? int? lines)
       (<= (first lines) (second lines))))

(defn valid-source? [x]
  (let [src (:source x)]
    (and (map? src) (valid-lines? (:lines src)))))

(defn seqify [x]
  (cond
    (nil? x) []
    (sequential? x) x
    :else [x]))

(defn endpoint-ids-from-value [v]
  (cond
    (keyword? v) [v]
    (sequential? v) (mapcat endpoint-ids-from-value v)
    ;; Inline assumptions/warrants may be maps rather than node ids.
    (map? v) []
    :else []))

(defn edge-endpoint-ids [edge]
  (->> endpoint-keys
       (mapcat #(endpoint-ids-from-value (get edge %)))
       vec))

(defn hole-mentions? [holes node-id]
  (some (fn [h]
          (some #(= node-id (get h %))
                [:id :node :ref :target :edge :wanted]))
        holes))

(defn ref-resolved? [holes node]
  (or (:label node)
      (:target node)
      (:citation node)
      (:item node)
      (:theorem node)
      (:local/id node)
      (hole-mentions? holes (:id node))))

(defn missing-warrant-edge? [edge]
  (let [w (:warrant edge)]
    (or (= :missing-warrant w)
        (= :missing-warrant (:kind w))
        (true? (:missing-warrant? edge))
        (= :depends-on (:role edge))
        (= :depends-on (:relation edge)))))

(defn matching-missing-warrant-hole? [holes edge]
  (let [eid (:id edge)]
    (some (fn [h]
            (and (= :missing-warrant (:kind h))
                 (or (nil? eid)
                     (= eid (:edge h))
                     (= eid (:id h))
                     (= eid (:target h)))))
          holes)))

(defn interval [lines]
  (when (valid-lines? lines) [(first lines) (second lines)]))

(defn scope-id [scope]
  (cond
    (keyword? scope) scope
    (map? scope) (:id scope)
    :else nil))

(defn scope-lines [edge]
  (or (interval (get-in edge [:scope :lines]))
      (interval (:scope-lines edge))
      (interval (get-in edge [:assume :source :lines]))
      (interval (get-in edge [:source :lines]))))

(defn assumption-edge? [edge]
  (or (= :assume (:role edge))
      (contains? edge :assume)))

(defn conclusion-edge? [edge]
  (or (= :conclusion (:role edge))
      (contains? edge :conclusion)))

(defn overlapping-not-nested? [[a1 a2] [b1 b2]]
  (or (and (< a1 b1) (<= b1 a2) (< a2 b2))
      (and (< b1 a1) (<= a1 b2) (< b2 a2))))

(defn check-graph [file graph]
  (let [nodes (vec (:nodes graph))
        edges (vec (:edges graph))
        holes (vec (:holes graph))
        node-ids (set (map :id nodes))
        node-by-id (into {} (map (juxt :id identity) nodes))
        failures (atom [])]
    (doseq [n nodes]
      (when-not (valid-source? n)
        (swap! failures conj
               (fail-entry :missing-source file graph n
                           "node lacks :source {:lines [a b]}")))
      (when-not (contains? node-kinds (:kind n))
        (swap! failures conj
               (fail-entry :node-kind file graph n
                           (str "unknown node :kind " (pr-str (:kind n)))))))
    (doseq [e edges]
      (when-not (valid-source? e)
        (swap! failures conj
               (fail-entry :missing-source file graph e
                           "edge lacks :source {:lines [a b]}")))
      (doseq [eid (edge-endpoint-ids e)]
        (when-not (contains? node-ids eid)
          (swap! failures conj
                 (fail-entry :dangling-endpoint file graph e
                             (str "edge endpoint " eid " does not resolve to a node :id")))))
      (when (and (missing-warrant-edge? e)
                 (not (matching-missing-warrant-hole? holes e)))
        (swap! failures conj
               (fail-entry :missing-warrant file graph e
                           "missing warrant is not mirrored by {:kind :missing-warrant ...} in :holes"))))
    (doseq [n nodes
            :when (= :ref (:kind n))]
      (when-not (ref-resolved? holes n)
        (swap! failures conj
               (fail-entry :unresolved-ref file graph n
                           ":ref node has no :label/:target/:citation/:item/:theorem/:local/id and is not listed in :holes"))))
    (doseq [e edges
            cid (endpoint-ids-from-value (:conclusion e))
            :let [n (node-by-id cid)]
            :when (= :meta (:kind n))]
      (swap! failures conj
             (fail-entry :meta-conclusion file graph e
                         (str ":meta node " cid " is used as an object-layer conclusion"))))
    (doseq [e edges
            :when (and (= :conclusion (:role e))
                       (contains? node-ids (:to e))
                       (= :meta (:kind (node-by-id (:to e)))))]
      (swap! failures conj
             (fail-entry :meta-conclusion file graph e
                         (str ":meta node " (:to e) " is the :to of a :conclusion edge"))))
    (let [assumes (->> edges
                       (filter assumption-edge?)
                       (map (fn [e] {:edge e
                                     :id (scope-id (:scope e))
                                     :lines (scope-lines e)}))
                       vec)
          scoped-conclusions (->> edges
                                  (filter conclusion-edge?)
                                  (map (fn [e] (scope-id (:scope e))))
                                  (remove nil?)
                                  set)]
      (doseq [{:keys [edge lines id]} assumes]
        (when-not lines
          (swap! failures conj
                 (fail-entry :bad-subproof-scope file graph edge
                             ":assume edge has no line interval for its subproof scope")))
        (when (and id (not (contains? scoped-conclusions id)))
          (swap! failures conj
                 (fail-entry :bad-subproof-scope file graph edge
                             (str ":assume scope " id " has no conclusion edge in the same scope")))))
      (doseq [[a b] (for [a assumes b assumes :when (neg? (compare (str (:id (:edge a))) (str (:id (:edge b)))))] [a b])]
        (when (and (:lines a) (:lines b)
                   (overlapping-not-nested? (:lines a) (:lines b)))
          (swap! failures conj
                 (fail-entry :bad-subproof-scope file graph (:edge b)
                             (str "subproof scope " (:lines b)
                                  " overlaps but is not nested in " (:lines a)))))))
    @failures))

(defn check-file [file]
  (try
    (let [graph (read-one-edn file)
          failures (check-graph file graph)]
      {:file (.getPath file)
       :ok? (empty? failures)
       :failures failures})
    (catch Exception e
      {:file (.getPath file)
       :ok? false
       :failures [{:gate :edn-parse
                   :file (.getPath file)
                   :locus (.getPath file)
                   :reason (.getMessage e)}]})))

(defn print-result [{:keys [file ok? failures]}]
  (println (str (if ok? "PASS " "FAIL ") file))
  (doseq [{:keys [gate locus reason]} failures]
    (println (str "  [" (name gate) "] " locus " :: " reason))))

(let [args *command-line-args*
      include-attempts? (boolean (some #{"--include-attempts"} args))
      paths (remove #{"--include-attempts"} args)]
  (when (empty? paths) (usage!))
  (let [files (mapcat #(edn-files include-attempts? %) paths)]
    (when (empty? files)
      (binding [*out* *err*]
        (println "No .edn files found in input paths:" (str/join " " args)))
      (System/exit 2))
    (let [results (mapv check-file files)]
      (doseq [r results] (print-result r))
      (when (some (complement :ok?) results)
        (System/exit 1)))))
