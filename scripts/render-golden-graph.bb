#!/usr/bin/env bb
;; Render a golden anatomy graph EDN -> dot. W2 DISCIPLINE: consumes the EDN
;; file ALONE — no source text, no judgment. If the picture is poor, the
;; graph is poor. Satiety coloring: full = filled; hungry = red dashed,
;; labeled with hunger type. Usage: render-golden-graph.bb <file.edn>
(require '[clojure.edn :as edn] '[clojure.string :as str])
(def role-fill {:symbol "#dde7fb" :compound "#e4ecf4" :concept "#d3f3df"
                :decorator "#f5e6c8" :capability "#fdf3d7" :mexpr "#eef0f7"
                :display "#eef0f7" :macro "#e8e8f4" :anchor "#d8d8d8"
                :component "#f3e3f7" :operator "#e3eef7" :scope "#f0f0f0"})
(defn esc [s] (str/replace (str s) "\"" "\\\""))
(defn trunc [s n] (if (> (count (str s)) n) (str (subs (str s) 0 n) "…") (str s)))
(let [f (first *command-line-args*)
      g (edn/read-string (slurp f))
      out (str/replace f #"\.edn$" ".svg")]
  (println "digraph g { rankdir=TB; bgcolor=white; node[fontname=Helvetica,fontsize=10]; edge[fontname=Helvetica,fontsize=8,color=\"#666666\"];")
  (println (format "label=\"%s — %s (rendered from EDN alone)\"; labelloc=t; fontname=\"Helvetica-Bold\";"
                   (esc (:paper g)) (esc (get-in g [:region :label]))))
  (doseq [n (:nodes g)]
    (let [hungry (when (map? (:satiety n)) (name (:hungry-for (:satiety n))))
          fill (get role-fill (:role n) "#ffffff")
          lab (str (trunc (:form n) 34) "\\n" (name (or (:role n) :node))
                   (when hungry (str "\\nHUNGRY: " hungry)))]
      (println (format "%s [label=\"%s\", shape=box, style=\"filled%s\", fillcolor=\"%s\"%s];"
                       (str "n" (Math/abs (hash (:id n)))) (esc lab)
                       (if hungry ",dashed,bold" ",rounded") fill
                       (if hungry ", color=\"#c0392b\", penwidth=2" "")))))
  (doseq [[i e] (map-indexed vector (:hyperedges g))]
    (let [eid (str "e" i)
          lab (str (name (:kind e))
                   (when-let [j (:justification e)] (str "\\n[" (trunc (or (:form j) j) 16) "]")))]
      (println (format "%s [label=\"%s\", shape=ellipse, style=filled, fillcolor=\"#ffffff\", fontsize=8, color=\"#999999\"];" eid (esc lab)))
      (doseq [end (:ends e)]
        (when (some #(= (:id %) (:node end)) (:nodes g))
          (println (format "%s -> n%s [label=\"%s\"];" eid (Math/abs (hash (:node end))) (esc (name (or (:role end) :end)))))))))
  (println "}"))
