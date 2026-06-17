#!/usr/bin/env bb
;; Deterministic LaTeXML Content-MathML -> Clojure :structure transducer.
;;
;; Usage:
;;   bb scripts/sfc_def_structure.bb '<latex formula>'
;;   printf '%s' '<latex formula>' | bb scripts/sfc_def_structure.bb -

(require '[babashka.process :as p]
         '[clojure.edn :as edn]
         '[clojure.string :as str])

(import '[java.text Normalizer Normalizer$Form])

(def symbol-dictionary
  {"approx" "cong"
   "evaluated-at" "restrict"
   "conditional-set" "conditional-set"
   "formulae-sequence" "formulae-sequence"
   "for-all" "forall"
   "⇒" "implies"
   "→" "→"
   "⋅" "·"})

(def mathbb-symbols
  {"A" "𝔸" "B" "𝔹" "C" "ℂ" "D" "𝔻" "E" "𝔼" "F" "𝔽" "G" "𝔾"
   "H" "ℍ" "I" "𝕀" "J" "𝕁" "K" "𝕂" "L" "𝕃" "M" "𝕄" "N" "ℕ"
   "O" "𝕆" "P" "ℙ" "Q" "ℚ" "R" "ℝ" "S" "𝕊" "T" "𝕋" "U" "𝕌"
   "V" "𝕍" "W" "𝕎" "X" "𝕏" "Y" "𝕐" "Z" "ℤ"})

(def mathcal-symbols
  {"A" "𝒜" "B" "ℬ" "C" "𝒞" "D" "𝒟" "E" "ℰ" "F" "ℱ" "G" "𝒢"
   "H" "ℋ" "I" "ℐ" "J" "𝒥" "K" "𝒦" "L" "ℒ" "M" "ℳ" "N" "𝒩"
   "O" "𝒪" "P" "𝒫" "Q" "𝒬" "R" "ℛ" "S" "𝒮" "T" "𝒯" "U" "𝒰"
   "V" "𝒱" "W" "𝒲" "X" "𝒳" "Y" "𝒴" "Z" "𝒵"})

(def styled-symbols
  (set (concat (vals mathbb-symbols) (vals mathcal-symbols))))

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/sfc_def_structure.bb '<latex formula>' | -"))
  (System/exit 2))

(defn nfkc [s]
  (Normalizer/normalize (str s) Normalizer$Form/NFKC))

(defn clean-symbol [s]
  (let [s (str/trim (str s))]
    (if (contains? styled-symbols s)
      s
      (-> s nfkc str/trim))))

(defn normalize-font-macros [formula]
  (letfn [(replace-font [s command dictionary]
            (str/replace s
                         (re-pattern (str "\\\\" command "\\s*\\{\\s*([A-Za-z])\\s*\\}"))
                         (fn [[_ letter]]
                           (get dictionary letter (str "\\" command "{" letter "}")))))]
    (-> formula
        (replace-font "mathbb" mathbb-symbols)
        (replace-font "mathcal" mathcal-symbols))))

(defn read-formula [args]
  (cond
    (= ["-"] args) (slurp *in*)
    (seq args) (str/join " " args)
    :else (usage!)))

(defn latexml-cmml [formula]
  (let [result @(p/process ["latexmlmath" "--cmml=-" "-"]
                           {:in formula :out :string :err :string})]
    (when-not (zero? (:exit result))
      (throw (ex-info "latexmlmath failed" {:stderr (:err result)})))
    (:out result)))

(def token-re #"(?s)<(/?)([A-Za-z0-9:]+)([^>]*)>|([^<]+)")

(defn decode-xml [s]
  (-> s
      (str/replace "&lt;" "<")
      (str/replace "&gt;" ">")
      (str/replace "&amp;" "&")
      (str/replace "&quot;" "\"")
      (str/replace #"&#10;" "\n")))

(defn tokenize-xml [xml]
  (->> (re-seq token-re xml)
       (map (fn [[raw close tag attrs text]]
              (if tag
                {:raw raw
                 :kind (cond
                         (= "/" close) :close
                         (str/ends-with? attrs "/") :self
                         :else :open)
                 :tag (keyword (last (str/split tag #":")))}
                {:kind :text :text (decode-xml text)})))
       (remove #(and (= :text (:kind %)) (str/blank? (:text %))))
       vec))

(declare parse-node)

(defn parse-children [tokens i close-tag]
  (loop [i i
         children []]
    (let [tok (get tokens i)]
      (cond
        (nil? tok) [children i]
        (and (= :close (:kind tok)) (= close-tag (:tag tok))) [children (inc i)]
        :else (let [[node next-i] (parse-node tokens i)]
                (recur next-i (conj children node)))))))

(defn parse-node [tokens i]
  (let [tok (get tokens i)]
    (case (:kind tok)
      :text [{:tag :text :text (:text tok)} (inc i)]
      :self [{:tag (:tag tok) :children []} (inc i)]
      :open (let [[children next-i] (parse-children tokens (inc i) (:tag tok))]
              [{:tag (:tag tok) :children children} next-i])
      :close [{:tag :unexpected-close :children []} (inc i)])))

(defn parse-xml [xml]
  (first (parse-node (tokenize-xml xml) 0)))

(defn element-children [node]
  (vec (remove #(= :text (:tag %)) (:children node))))

(defn text-content [node]
  (apply str (map #(if (= :text (:tag %)) (:text %) (text-content %)) (:children node))))

(defn local-name [node]
  (name (:tag node)))

(declare cmml->sexpr)

(defn first-element [node]
  (first (element-children node)))

(defn math-root [doc]
  (let [children (element-children doc)]
    (or (first children) doc)))

(defn operator-symbol [node]
  (let [tag (local-name node)
        text (clean-symbol (text-content node))]
    (case tag
      "eq" "="
      "in" "∈"
      "approx" "cong"
      "times" "*"
      "and" "and"
      "exists" "exists"
      "ci" (get symbol-dictionary text text)
      "csymbol" (get symbol-dictionary text text)
      text)))

(defn normalize-apply [op args]
  (let [op (get symbol-dictionary op op)]
    (cond
      (= op "evaluated-at") (cons 'restrict args)
      (= op "conditional-set") (cons 'conditional-set args)
      (= op "for-all") (cons 'forall args)
      (= op "exists") (cons 'exists args)
      (= op "approx") (cons 'cong args)
      (= op "cong") (cons 'cong args)
      (= op "⇒") (cons 'implies args)
      (= op "implies") (cons 'implies args)
      (= op "·") (cons '· args)
      (= op "⋅") (cons '· args)
      (= op "→") (cons '→ args)
      (= op ":") (cons (symbol ":") args)
      (= op "∈") (cons '∈ args)
      (= op "=") (cons '= args)
      :else (cons (symbol op) args))))

(defn quantifier-form? [expr]
  (and (seq? expr)
       (#{'forall 'exists} (first expr))
       (= 2 (count expr))))

(defn quantifier->binder [expr body]
  (let [quantifier (first expr)
        bound (second expr)]
    (list quantifier [bound] body)))

(declare canonicalize-sexpr)

(defn canonicalize-formulae-sequence [items]
  (let [[quantifiers body-items] (split-with quantifier-form? items)
        body (case (count body-items)
               0 (list (symbol ":hole") "missing-formulae-sequence-body")
               1 (first body-items)
               (list (symbol ":hole") "unhandled-formulae-sequence" (vec body-items)))]
    (if (seq quantifiers)
      (reduce (fn [inner quantifier]
                (quantifier->binder quantifier inner))
              body
              (reverse quantifiers))
      (if (= 1 (count items))
        (first items)
        (list (symbol ":hole") "unhandled-formulae-sequence" (vec items))))))

(defn math-font-times? [expr font-symbol]
  (and (seq? expr)
       (= '* (first expr))
       (= 3 (count expr))
       (= font-symbol (second expr))
       (symbol? (nth expr 2))))

(defn canonicalize-font-times [expr]
  (cond
    (math-font-times? expr (symbol "\\mathbb"))
    (symbol (get mathbb-symbols (name (nth expr 2)) (name (nth expr 2))))

    (math-font-times? expr (symbol "\\mathcal"))
    (symbol (get mathcal-symbols (name (nth expr 2)) (name (nth expr 2))))

    :else expr))

(defn canonicalize-sexpr [expr]
  (cond
    (seq? expr)
    (let [expr (apply list (map canonicalize-sexpr expr))
          expr (canonicalize-font-times expr)]
      (if (and (seq? expr) (= 'formulae-sequence (first expr)))
        (canonicalize-formulae-sequence (vec (rest expr)))
        expr))

    (vector? expr) (mapv canonicalize-sexpr expr)
    :else expr))

(defn cmml->sexpr [node]
  (let [tag (local-name node)]
    (case tag
      "math" (cmml->sexpr (math-root node))
      "apply" (let [[op-node & arg-nodes] (element-children node)
                    op (operator-symbol op-node)
                    args (map cmml->sexpr arg-nodes)]
                (normalize-apply op args))
      "ci" (symbol (clean-symbol (text-content node)))
      "cn" (edn/read-string (clean-symbol (text-content node)))
      "csymbol" (symbol (get symbol-dictionary
                              (clean-symbol (text-content node))
                              (clean-symbol (text-content node))))
      "share" 'share
      (symbol (clean-symbol (text-content node))))))

(defn rewrite-rw [formula]
  (str/replace formula #"\\Rw\b" (constantly "\\Rightarrow")))

(def typed-binder-re
  #"\\(forall|exists)\s+([A-Za-z]\s*(?:,\s*[A-Za-z]\s*)*)\s*:\s*(.+?)\s*(?:\\,)?\s*\\?\.")

(def multi-binder-re
  #"\\(forall|exists)\s+([A-Za-z]\s*(?:,\s*[A-Za-z]\s*)+)\s*(?:\\,)?\s*\\?\.")

(defn normalize-quantifier-name [name]
  (case name
    "forall" "forall"
    "exists" "exists"))

(defn quantifier-command [name]
  (str "\\" (normalize-quantifier-name name)))

(defn binder-normalize [formula]
  (let [capture (atom [])]
    [(-> formula
         (str/replace typed-binder-re
                      (fn [[_ quantifier vars type]]
                        (let [vars (mapv str/trim (str/split vars #","))
                              type (str/trim type)]
                          (swap! capture conj {:vars vars :type type})
                          (str/join " "
                                    (map #(str (quantifier-command quantifier)
                                               " " % "\\,.")
                                         vars)))))
         (str/replace multi-binder-re
                      (fn [[_ quantifier vars]]
                        (let [vars (mapv str/trim (str/split vars #","))]
                          (str/join " "
                                    (map #(str (quantifier-command quantifier)
                                               " " % "\\,.")
                                         vars))))))
     @capture]))

(defn parse-arrow-type [type]
  (let [[_ a b] (re-find #"([A-Za-z]+)\s*\\to\s*([A-Za-z]+)" type)]
    (list '→ (symbol a) (symbol b))))

(defn parse-restrict-eq [s]
  (let [[_ a m b n] (re-find #"([A-Za-z])\s*\|_\{?([A-Za-z])\}?\s*=\s*([A-Za-z])\s*\|_\{?([A-Za-z])\}?" s)]
    (list '= (list 'restrict (symbol a) (symbol m))
          (list 'restrict (symbol b) (symbol n)))))

(defn parse-action-cong [s]
  (let [[_ a x b y] (re-find #"([A-Za-z])\s*\\cdot\s*([A-Za-z])\s*\\cong\s*([A-Za-z])\s*\\cdot\s*([A-Za-z])" s)]
    (list 'cong (list '· (symbol a) (symbol x))
          (list '· (symbol b) (symbol y)))))

(defn l-closure-structure [formula captures]
  (when-let [[_ lhs set-var set-carrier body]
             (re-find #"\\overline\{([A-Za-z])\}\s*=\s*\\\{\s*([A-Za-z])\s*\\in\s*([A-Za-z])\s*\\mid\s*(.+)\s*\\\}" formula)]
    (let [[antecedent consequent] (str/split body #"\\Rightarrow" 2)
          cap (first captures)
          vars (mapv symbol (:vars cap))
          type (parse-arrow-type (:type cap))]
      (list '= (list 'overline (symbol lhs))
            (list 'conditional-set
                  (list '∈ (symbol set-var) (symbol set-carrier))
                  (list 'forall vars (list (symbol ":") type)
                        (list 'implies
                              (parse-restrict-eq antecedent)
                              (parse-action-cong consequent))))))))

(defn relational-chain-regroup [expr]
  (if (and (seq? expr)
           (= 'and (first expr))
           (= 3 (count (rest expr))))
    (let [[a b c] (rest expr)]
      (if (and (seq? b) (= 'implies (first b))
               (seq? c) (= 'cong (first c)))
        (list 'implies a (list 'cong (second b) (nth c 2)))
        expr))
    expr))

(def grounded-operators #{'= 'conditional-set '∈ 'forall 'exists (symbol ":") (symbol ":hole")
                          '→ 'implies 'restrict 'cong 'overline})

(defn collect-symbols [expr]
  (let [seen (atom #{})]
    (letfn [(walk [x head?]
              (cond
                (symbol? x) (when-not (and head? (contains? grounded-operators x))
                              (swap! seen conj x))
                (seq? x) (do (walk (first x) true)
                             (doseq [y (rest x)] (walk y false)))
                (vector? x) (doseq [y x] (walk y false))
                :else nil))]
      (walk expr false)
      (->> @seen
           (map name)
           sort
           (mapv (fn [s] {:symbol s :grounding :hole}))))))

(defn transduce-formula [formula]
  (let [formula (rewrite-rw formula)
        [normalized captures] (binder-normalize formula)
        normalized (normalize-font-macros normalized)
        cmml (latexml-cmml normalized)
        parsed (try
                 (-> (cmml->sexpr (parse-xml cmml))
                     canonicalize-sexpr
                     relational-chain-regroup)
                 (catch Exception e
                   (symbol (str "parse-error:" (.getMessage e)))))
        structure (or (l-closure-structure formula captures) parsed)]
    {:schema "sfc-def-structure/v1"
     :formula formula
     :normalized-formula normalized
     :binder-captures captures
     :structure structure
     :ungrounded (collect-symbols structure)
     :cmml cmml}))

(defn -main [args]
  (let [formula (str/trim (read-formula args))
        result (transduce-formula formula)]
    (prn (select-keys result [:schema :formula :normalized-formula
                              :binder-captures :structure :ungrounded]))
    0))

(when (= *file* (System/getProperty "babashka.file"))
  (System/exit (-main *command-line-args*)))
