#!/usr/bin/env bb
;; R2b/R2c semantic checks over IATC argument-graph EDN.
;;
;; Usage:
;;   bb scripts/iatc_closure_check.bb path/to/graph-or-dir [...]
;;   bb scripts/iatc_closure_check.bb --warrant-floor 0.20 path/to/graph-or-dir [...]

(require '[babashka.process :as p]
         '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def default-warrant-floor 0.0)

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/iatc_closure_check.bb [--warrant-floor F] <file-or-directory> [...]"))
  (System/exit 2))

(defn edn-files [path]
  (let [f (io/file path)]
    (cond
      (not (.exists f)) []
      (.isDirectory f) (->> (.listFiles f)
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

(defn seqify [x]
  (cond
    (nil? x) []
    (sequential? x) x
    :else [x]))

(defn endpoint-ids [v]
  (cond
    (keyword? v) [v]
    (sequential? v) (mapcat endpoint-ids v)
    :else []))

(defn edge-sources [edge]
  (vec (mapcat #(endpoint-ids (get edge %))
               [:from :given :premise :assume :depends-on :contradicts :meta])))

(defn edge-targets [edge]
  (vec (mapcat #(endpoint-ids (get edge %))
               [:to :conclusion])))

(defn graph-arcs [graph]
  (vec
   (for [edge (:edges graph)
         src (edge-sources edge)
         dst (edge-targets edge)]
     {:edge-id (:id edge)
      :from src
      :to dst})))

(defn adjacency [arcs]
  (reduce (fn [m {:keys [from] :as arc}]
            (update m from (fnil conj []) arc))
          {}
          arcs))

(defn incident-counts [node-ids arcs]
  (let [base (zipmap node-ids (repeat {:in 0 :out 0}))]
    (reduce (fn [m {:keys [from to]}]
              (-> m
                  (update-in [from :out] (fnil inc 0))
                  (update-in [to :in] (fnil inc 0))))
            base
            arcs)))

(defn self-loop [arcs]
  (first (filter #(= (:from %) (:to %)) arcs)))

(defn cycle-from [arcs]
  (if-let [loop (self-loop arcs)]
    {:kind :self-loop
     :edge-id (:edge-id loop)
     :nodes [(:from loop) (:to loop)]}
    (let [adj (adjacency arcs)
          seen (atom #{})
          stack (atom [])
          in-stack (atom #{})
          found (atom nil)]
      (letfn [(visit [node]
                (when-not @found
                  (swap! seen conj node)
                  (swap! stack conj node)
                  (swap! in-stack conj node)
                  (doseq [{:keys [to]} (get adj node)]
                    (cond
                      @found nil
                      (contains? @in-stack to)
                      (let [path @stack
                            idx (.indexOf path to)]
                        (reset! found {:kind :cycle
                                       :nodes (conj (subvec path idx) to)}))
                      (not (contains? @seen to)) (visit to)))
                  (swap! stack pop)
                  (swap! in-stack disj node)))]
        (doseq [{:keys [from]} arcs]
          (when-not (contains? @seen from)
            (visit from)))
        @found))))

(defn reachable-from [adj roots]
  (loop [frontier (seq roots)
         reached #{}]
    (if-not frontier
      reached
      (let [node (first frontier)]
        (if (contains? reached node)
          (recur (next frontier) reached)
          (let [next-nodes (map :to (get adj node))]
            (recur (concat (next frontier) next-nodes)
                   (conj reached node))))))))

(defn check-closure [graph ctx]
  (let [nodes (vec (:nodes graph))
        node-ids (set (map :id nodes))
        arcs (graph-arcs graph)
        counts (incident-counts node-ids arcs)
        roots (->> counts
                   (filter (fn [[_ c]] (and (zero? (:in c)) (pos? (:out c)))))
                   (map first)
                   set)
        terminals (->> counts
                       (filter (fn [[_ c]] (and (pos? (:in c)) (zero? (:out c)))))
                       (map first)
                       set)
        reached (reachable-from (adjacency arcs) roots)
        reachable-terminals (set (filter reached terminals))
        orphan-nodes (->> counts
                          (filter (fn [[_ c]] (and (zero? (:in c)) (zero? (:out c)))))
                          (map first)
                          sort)
        cycle (cycle-from arcs)
        reasons (cond-> []
                  (seq cycle)
                  (conj (if (= :self-loop (:kind cycle))
                          (str "cycle: self-loop at node " (first (:nodes cycle))
                               " via edge " (:edge-id cycle))
                          (str "cycle: " (str/join " -> " (:nodes cycle)))))
                  (seq orphan-nodes)
                  (conj (str "orphan node(s): " (str/join ", " orphan-nodes)))
                  (empty? roots)
                  (conj "no root premise node exists (in-degree 0, out-degree > 0)")
                  (empty? terminals)
                  (conj "no terminal conclusion node exists (in-degree > 0, out-degree 0)")
                  (and (seq terminals) (empty? reachable-terminals))
                  (conj (str "terminal conclusion node(s) not reachable from premises: "
                             (str/join ", " (sort terminals)))))]
    {:check :closure
     :pass (empty? reasons)
     :rate (if (seq node-ids)
             (/ (- (count node-ids) (count orphan-nodes)) (double (count node-ids)))
             nil)
     :reasons reasons
     :per-item [{:nodes (count node-ids)
                 :edges (count arcs)
                 :roots (vec (sort roots))
                 :terminals (vec (sort terminals))
                 :reachable-terminals (vec (sort reachable-terminals))
                 :orphan-nodes (vec orphan-nodes)
                 :cycle cycle
                 :file (:file ctx)}]}))

(defn python-warrant-counts [file]
  (let [code (str "import importlib.util, pathlib\n"
                  "root=pathlib.Path('.').resolve()\n"
                  "spec=importlib.util.spec_from_file_location('h', root/'scripts/mark3_eval_harness.py')\n"
                  "h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
                  "r,t=h.warrant_resolution_counts([pathlib.Path(" (pr-str (.getPath file)) ")])\n"
                  "print(f'{r} {t}')\n")
        result @(p/process ["python3" "-c" code]
                           {:out :string :err :string})]
    (if (zero? (:exit result))
      (mapv parse-long (str/split (str/trim (:out result)) #"\s+"))
      (throw (ex-info "mark3_eval_harness warrant count failed"
                      {:stderr (:err result)})))))

(defn check-warrant-resolution [_graph ctx]
  (let [[resolved total] (python-warrant-counts (io/file (:file ctx)))
        rate (if (pos? total) (/ resolved (double total)) nil)
        floor (:warrant-floor ctx default-warrant-floor)
        ;; No inference edges => no warrants to resolve => N/A, NOT a failure.
        ;; (N/A != FAIL: the gate fails only on present-but-wrong structure, never
        ;; on absent structure — a coarse/edgeless graph must stay useful.)
        na? (zero? total)
        below? (and (some? rate) (< rate floor))
        pass? (or na? (not below?))
        status (cond na? :na below? :fail :else :pass)
        reasons (cond-> []
                  na? (conj "no inference edges — N/A (warrant-resolution not applicable)")
                  below? (conj (format "warrant-resolution %.3f below floor %.3f (%d/%d real warrants)"
                                       rate floor resolved total)))]
    {:check :warrant-resolution
     :pass pass?
     :status status
     :rate rate
     :reasons reasons
     :per-item [{:resolved-warrant-edges resolved
                 :total-edges total
                 :floor floor
                 :file (:file ctx)}]}))

(defn check-graph [graph ctx]
  [(check-closure graph ctx)
   (check-warrant-resolution graph ctx)])

(defn parse-args [args]
  (loop [args args
         opts {:warrant-floor default-warrant-floor}
         paths []]
    (if-not (seq args)
      {:opts opts :paths paths}
      (let [[a & more] args]
        (case a
          "--warrant-floor"
          (if-let [v (first more)]
            (recur (rest more) (assoc opts :warrant-floor (Double/parseDouble v)) paths)
            (usage!))
          (recur more opts (conj paths a)))))))

(defn print-check [result]
  (println (format "  [%s] %s rate=%s"
                   (name (:check result))
                   (if (:pass result) "PASS" "FAIL")
                   (if (some? (:rate result)) (format "%.3f" (:rate result)) "n/a")))
  (doseq [reason (:reasons result)]
    (println (str "    - " reason))))

(defn check-file [file opts]
  (try
    (let [graph (read-one-edn file)
          ctx (assoc opts
                     :file (.getPath file)
                     :paper-id (:paper/id graph))
          results (check-graph graph ctx)]
      {:file (.getPath file)
       :pass (every? :pass results)
       :checks results})
    (catch Exception e
      {:file (.getPath file)
       :pass false
       :checks [{:check :edn-parse
                 :pass false
                 :rate nil
                 :reasons [(.getMessage e)]
                 :per-item []}]})))

(defn -main [args]
  (let [{:keys [opts paths]} (parse-args args)]
    (when (empty? paths) (usage!))
    (let [files (mapcat edn-files paths)]
      (when (empty? files)
        (binding [*out* *err*]
          (println "No .edn files found in input paths:" (str/join " " paths)))
        (System/exit 2))
      (let [results (mapv #(check-file % opts) files)]
        (doseq [{:keys [file checks]} results]
          (println (str (if (every? :pass checks) "PASS " "FAIL ") file))
          (doseq [check checks]
            (print-check check)))
        (let [failures (count (remove :pass results))]
          (println)
          (println (format "iatc-closure-check: %d file(s), %d failing file(s) -- %s"
                           (count results) failures (if (zero? failures) "PASS" "FAIL")))
          (System/exit (if (zero? failures) 0 1)))))))

(when (= *file* (System/getProperty "babashka.file"))
  (-main *command-line-args*))
