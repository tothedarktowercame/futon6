(require '[babashka.fs :as fs]
         '[babashka.process :as p]
         '[clojure.string :as str]
         '[clojure.test :refer [deftest is run-tests]])

(def root (fs/cwd))
(def checker (str (fs/path root "scripts/iatc_closure_check.bb")))

(defn run-check [& args]
  @(p/process (into ["bb" checker] args)
              {:dir (str root)
               :out :string
               :err :string
               :continue true}))

(defn write-edn [dir name body]
  (let [path (fs/path dir name)]
    (spit (str path) body)
    (str path)))

(deftest closure-pass-and-warrant-pass
  (let [dir (fs/create-temp-dir)
        graph (write-edn dir "ok.edn"
                         "{:paper/id \"ok\" :passage/id \"ok:L1-3\"
                           :source {:lines [1 3] :kind :proof}
                           :nodes [{:id :a :kind :claim :source {:lines [1 1]}}
                                   {:id :b :kind :claim :source {:lines [2 2]}}]
                           :edges [{:id :e :kind :infer :premise :a :conclusion :b
                                    :warrant {:kind :citation :target \"T\"}
                                    :source {:lines [1 2]}}]
                           :holes []}")
        result (run-check "--warrant-floor" "1.0" graph)]
    (is (zero? (:exit result)) (:out result))
    (is (str/includes? (:out result) "[closure] PASS"))
    (is (str/includes? (:out result) "[warrant-resolution] PASS rate=1.000"))))

(deftest closure-flags-self-loop-with-edge-id
  (let [dir (fs/create-temp-dir)
        graph (write-edn dir "loop.edn"
                         "{:paper/id \"bad\" :passage/id \"bad:L1-2\"
                           :source {:lines [1 2] :kind :proof}
                           :nodes [{:id :a :kind :claim :source {:lines [1 1]}}]
                           :edges [{:id :e-loop :kind :infer :premise :a :conclusion :a
                                    :warrant {:kind :citation :target \"T\"}
                                    :source {:lines [1 1]}}]
                           :holes []}")
        result (run-check graph)]
    (is (= 1 (:exit result)))
    (is (str/includes? (:out result) "self-loop at node :a via edge :e-loop"))))

(deftest warrant-floor-fails-unresolved-edge
  (let [dir (fs/create-temp-dir)
        graph (write-edn dir "missing.edn"
                         "{:paper/id \"bad\" :passage/id \"bad:L1-2\"
                           :source {:lines [1 2] :kind :proof}
                           :nodes [{:id :a :kind :claim :source {:lines [1 1]}}
                                   {:id :b :kind :claim :source {:lines [2 2]}}]
                           :edges [{:id :e :kind :infer :premise :a :conclusion :b
                                    :warrant {:kind :missing-warrant :wanted :x}
                                    :source {:lines [1 2]}}]
                           :holes [{:kind :missing-warrant :edge :e :wanted :x}]}")
        result (run-check "--warrant-floor" "0.5" graph)]
    (is (= 1 (:exit result)))
    (is (str/includes? (:out result) "[closure] PASS"))
    (is (str/includes? (:out result) "[warrant-resolution] FAIL rate=0.000"))))

(deftest warrant-resolution-na-on-zero-edges
  ;; No inference edges => no warrants to resolve => N/A, NOT a warrant FAIL,
  ;; even with a positive floor (N/A != FAIL — a coarse/edgeless graph stays useful).
  (let [dir (fs/create-temp-dir)
        graph (write-edn dir "edgeless.edn"
                         "{:paper/id \"coarse\" :passage/id \"coarse:L1-1\"
                           :source {:lines [1 1] :kind :proof}
                           :nodes [{:id :a :kind :claim :source {:lines [1 1]}}]
                           :edges []
                           :holes []}")
        result (run-check "--warrant-floor" "0.5" graph)]
    (is (str/includes? (:out result) "[warrant-resolution] PASS rate=n/a"))
    (is (str/includes? (:out result) "N/A"))))

(let [summary (run-tests)]
  (when (pos? (+ (:fail summary) (:error summary)))
    (System/exit 1)))
