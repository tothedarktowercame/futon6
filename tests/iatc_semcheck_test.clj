(ns iatc-semcheck-test
  (:require [clojure.test :refer [deftest is run-tests]]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(declare edn-files check-file default-opts)

(load-file "scripts/iatc_semcheck.bb")

(defn temp-dir []
  (.toFile (java.nio.file.Files/createTempDirectory "iatc-semcheck-test" (make-array java.nio.file.attribute.FileAttribute 0))))

(defn spit-file [path text]
  (io/make-parents (io/file path))
  (spit path text)
  (io/file path))

(deftest finals-only-collection
  (let [d (temp-dir)
        top (spit-file (io/file d "graph.edn") "{:nodes [] :edges [] :holes []}")
        attempt (spit-file (io/file d ".attempts" "graph.attempt0.edn")
                           "{:nodes [] :edges [] :holes []}")]
    (is (= [(.getPath top)]
           (mapv #(.getPath %) (edn-files false (.getPath d)))))
    (is (= (sort [(.getPath attempt) (.getPath top)])
           (sort (mapv #(.getPath %) (edn-files true (.getPath d))))))))

(deftest absent-edge-structure-is-na-not-fail
  (let [d (temp-dir)
        marks (io/file d "golden")
        graph (spit-file
               (io/file d "9999.0001.edn")
               "{:paper/id \"9999.0001\"
                 :passage/id \"9999.0001:p\"
                 :nodes [{:id :n1 :kind :claim :text \"abelian category\"
                          :source {:lines [1 1]}}]
                 :edges []
                 :holes []}")]
    (spit-file (io/file marks "fable-9999.0001-dp-emacs.json")
               "{\"text\":\"An abelian category appears here.\"}")
    (let [result (check-file (assoc default-opts :marks-dir (.getPath marks)) graph)
          by-check (into {} (map (juxt :check identity) (:checks result)))]
      (is (:pass result))
      (is (= :pass (get-in by-check [:anchor-faithfulness :status])))
      (is (= :na (get-in by-check [:closure :status])))
      (is (= :na (get-in by-check [:warrant-resolution :status])))
      (is (str/includes? (first (get-in by-check [:closure :reasons])) "N/A")))))

(when (= *file* (System/getProperty "babashka.file"))
  (let [{:keys [fail error]} (run-tests)]
    (System/exit (if (zero? (+ fail error)) 0 1))))
