#!/usr/bin/env bb
;; Strict checker for Phase 5.4 expository-scope graph EDN files.
;;
;; Usage:
;;   bb scripts/expository_argcheck.bb path/to/graph.edn
;;   bb scripts/expository_argcheck.bb path/to/graphs/

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def repo (.getParentFile (.getParentFile (io/file *file*))))
(def vocab-file (io/file repo "holes/excursions/expository-superpod-vocab.edn"))

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/expository_argcheck.bb <file-or-directory> [...]"))
  (System/exit 2))

(defn edn-files [path]
  (let [f (io/file path)]
    (cond
      (not (.exists f)) []
      (.isDirectory f) (->> (file-seq f)
                            (filter #(.isFile %))
                            (filter #(str/ends-with? (.getName %) ".edn"))
                            (sort-by #(.getPath %)))
      :else [f])))

(defn read-one-edn [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (let [form (edn/read {:eof ::eof} r)]
      (if (= ::eof form)
        (throw (ex-info "empty EDN file" {}))
        (let [tail (edn/read {:eof ::eof} r)]
          (when-not (= ::eof tail)
            (throw (ex-info "trailing EDN forms after graph" {:tail tail})))
          form)))))

(defn load-vocab []
  (let [vocab (read-one-edn vocab-file)]
    {:kinds (set (map :kind (:scopes vocab)))
     :out-of-scope (set (map :iatc (:out-of-scope-arxiv vocab)))}))

(defn valid-lines? [lines]
  (and (vector? lines)
       (= 2 (count lines))
       (every? int? lines)
       (<= (first lines) (second lines))))

(defn valid-source? [x]
  (let [src (:source x)]
    (and (map? src) (valid-lines? (:lines src)))))

(defn nonblank-string? [x]
  (and (string? x) (not (str/blank? x))))

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

(defn has-slot-fill? [scope]
  (let [fill (:slot-fill scope)]
    (cond
      (string? fill) (not (str/blank? fill))
      (nil? fill) false
      :else true)))

(defn held-reason [scope]
  (get-in scope [:held :reason]))

(defn check-graph [vocab file graph]
  (let [scopes (vec (:scopes graph))
        failures (atom [])]
    (when-not (valid-source? graph)
      (swap! failures conj
             (fail-entry :missing-source file graph graph
                         "graph lacks :source {:lines [a b]}")))
    (when-not (vector? scopes)
      (swap! failures conj
             (fail-entry :schema file graph graph
                         "graph lacks a vector :scopes field")))
    (doseq [scope scopes]
      (when-not (valid-source? scope)
        (swap! failures conj
               (fail-entry :missing-source file graph scope
                           "scope lacks :source {:lines [a b]}")))
      (when-not (contains? (:kinds vocab) (:kind scope))
        (swap! failures conj
               (fail-entry :unknown-kind file graph scope
                           (str "scope :kind " (pr-str (:kind scope))
                                " does not resolve to expository-superpod-vocab.edn"))))
      (when (contains? (:out-of-scope vocab) (:kind scope))
        (swap! failures conj
               (fail-entry :out-of-scope-kind file graph scope
                           (str "scope :kind " (pr-str (:kind scope))
                                " is listed under :out-of-scope-arxiv"))))
      (let [reason (held-reason scope)
            held? (contains? scope :held)]
        (when (and held? (not (nonblank-string? reason)))
          (swap! failures conj
                 (fail-entry :empty-held-reason file graph scope
                             ":held reason must be a non-empty string")))
        (when-not (or (has-slot-fill? scope) (nonblank-string? reason))
          (swap! failures conj
                 (fail-entry :missing-slot-fill file graph scope
                             "scope needs :slot-fill or :held {:reason ...}")))))
    @failures))

(defn check-file [vocab file]
  (try
    (let [graph (read-one-edn file)
          failures (check-graph vocab file graph)]
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

(let [args *command-line-args*]
  (when (empty? args) (usage!))
  (let [files (mapcat edn-files args)]
    (when (empty? files)
      (binding [*out* *err*]
        (println "No .edn files found in input paths:" (str/join " " args)))
      (System/exit 2))
    (let [vocab (load-vocab)
          results (mapv #(check-file vocab %) files)]
      (doseq [r results] (print-result r))
      (let [n-fail (count (remove :ok? results))]
        (println)
        (println (str "expository-argcheck: " (count results) " file(s), "
                      n-fail " failing file(s) -- "
                      (if (zero? n-fail) "PASS" "FAIL")))
        (System/exit (if (zero? n-fail) 0 1))))))
