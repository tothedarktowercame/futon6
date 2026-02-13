-- Convert math-like markdown text/code into LaTeX math during pandoc export.

local greek_words = {
  {"Gamma", "\\Gamma"}, {"Delta", "\\Delta"}, {"Theta", "\\Theta"},
  {"Lambda", "\\Lambda"}, {"Xi", "\\Xi"}, {"Pi", "\\Pi"}, {"Sigma", "\\Sigma"},
  {"Upsilon", "\\Upsilon"}, {"Phi", "\\Phi"}, {"Psi", "\\Psi"}, {"Omega", "\\Omega"},
  {"alpha", "\\alpha"}, {"beta", "\\beta"}, {"gamma", "\\gamma"},
  {"delta", "\\delta"}, {"epsilon", "\\epsilon"}, {"varepsilon", "\\varepsilon"},
  {"zeta", "\\zeta"}, {"eta", "\\eta"}, {"theta", "\\theta"}, {"vartheta", "\\vartheta"},
  {"iota", "\\iota"}, {"kappa", "\\kappa"}, {"lambda", "\\lambda"}, {"mu", "\\mu"},
  {"nu", "\\nu"}, {"xi", "\\xi"}, {"pi", "\\pi"}, {"rho", "\\rho"},
  {"sigma", "\\sigma"}, {"tau", "\\tau"}, {"upsilon", "\\upsilon"},
  {"phi", "\\phi"}, {"chi", "\\chi"}, {"omega", "\\omega"},
  {"psi", "\\psi"},
}

local greek_token_names = {
  alpha = true, beta = true, gamma = true, delta = true, epsilon = true,
  varepsilon = true, zeta = true, eta = true, theta = true, vartheta = true,
  iota = true, kappa = true, lambda = true, mu = true, nu = true, xi = true,
  pi = true, rho = true, sigma = true, tau = true, upsilon = true, phi = true,
  chi = true, psi = true, omega = true,
  Alpha = true, Beta = true, Gamma = true, Delta = true, Theta = true,
  Lambda = true, Xi = true, Pi = true, Sigma = true, Upsilon = true,
  Phi = true, Psi = true, Omega = true,
}

local unicode_ops = {
  {"⊞", "\\boxplus "}, {"∧", "\\wedge "}, {"∩", "\\cap "},
  {"⊂", "\\subset "}, {"⊕", "\\oplus "}, {"⊗", "\\otimes "},
  {"≠", "\\neq "}, {"≤", "\\le "}, {"≥", "\\ge "},
  {"→", "\\to "}, {"←", "\\leftarrow "}, {"∞", "\\infty "},
  {"×", "\\times "}, {"π", "\\pi "}, {"Γ", "\\Gamma "},
  {"Φ", "\\Phi "}, {"μ", "\\mu "}, {"λ", "\\lambda "},
  {"ω", "\\omega "}, {"χ", "\\chi "}, {"√", "\\sqrt "},
}

local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function normalize_escaped_script_artifacts(s)
  -- Undo escaped-text artifacts when a token should be parsed as math.
  s = s:gsub("\\textasciicircum%s*{}", "^")
  s = s:gsub("\\textasciicircum", "^")
  s = s:gsub("\\%^%s*%{%s*%}", "^")
  s = s:gsub("\\%^", "^")
  s = s:gsub("\\_", "_")
  s = s:gsub("([_%^])\\%{([^{}]-)\\%}", "%1{%2}")
  s = s:gsub("\\%(", "(")
  s = s:gsub("\\%)", ")")
  return s
end

local function find_matching_brace(s, open_idx)
  if s:sub(open_idx, open_idx) ~= "{" then
    return nil
  end
  local depth = 1
  local i = open_idx + 1
  while i <= #s do
    local ch = s:sub(i, i)
    if ch == "{" then
      depth = depth + 1
    elseif ch == "}" then
      depth = depth - 1
      if depth == 0 then
        return i
      end
    end
    i = i + 1
  end
  return nil
end

local function mark_integer_literals(s)
  local out = {}
  local i = 1
  local cmd = "\\mNumber"
  local cmd_len = #cmd

  while i <= #s do
    if s:sub(i, i + cmd_len - 1) == cmd and s:sub(i + cmd_len, i + cmd_len) == "{" then
      local close_idx = find_matching_brace(s, i + cmd_len)
      if close_idx ~= nil then
        table.insert(out, s:sub(i, close_idx))
        i = close_idx + 1
      else
        table.insert(out, s:sub(i, i))
        i = i + 1
      end
    else
      local ch = s:sub(i, i)
      if ch:match("%d") then
        local j = i
        while j <= #s and s:sub(j, j):match("%d") do
          j = j + 1
        end
        table.insert(out, "\\mNumber{" .. s:sub(i, j - 1) .. "}")
        i = j
      else
        table.insert(out, ch)
        i = i + 1
      end
    end
  end

  return table.concat(out)
end

local function looks_like_algorithm_line(line)
  if line:find("@", 1, true) then
    return true
  end
  if line:find("#", 1, true) then
    return true
  end
  local low = line:lower()
  if low:match("^%s*if%s+") or low:match("^%s*for%s+") or low:match("^%s*while%s+") then
    return true
  end
  if low:match("^%s*return%s+") or low:match("^%s*solve%s+") then
    return true
  end
  if low:find("break", 1, true) then
    return true
  end
  if low:find("sparse(", 1, true) then
    return true
  end
  return false
end

local function replace_word(s, w, repl)
  s = s:gsub("^" .. w .. "%f[%A]", repl)
  s = s:gsub("([^\\%a])" .. w .. "%f[%A]", "%1" .. repl)
  return s
end

local function wrap_prose_word(s, w)
  return replace_word(s, w, "\\text{" .. w .. "}")
end

local function mask_text_macros(s)
  local slots = {}
  local masked = s:gsub("\\text%b{}", function(m)
    table.insert(slots, m)
    return "@@MPTEXT" .. tostring(#slots) .. "@@"
  end)
  return masked, slots
end

local function unmask_text_macros(s, slots)
  return s:gsub("@@MPTEXT(%d+)@@", function(idx)
    return slots[tonumber(idx)] or ""
  end)
end

local function flatten_nested_text_macros(s)
  local prev = nil
  while prev ~= s do
    prev = s
    s = s:gsub("\\text%s*{%s*\\text%s*{([^{}]*)}%s*}", "\\text{%1}")
    s = s:gsub("\\mathup%s*{%s*\\mathup%s*{([^{}]*)}%s*}", "\\mathup{%1}")
    s = s:gsub("\\mOpName%s*{%s*\\mOpName%s*{([^{}]*)}%s*}", "\\mOpName{%1}")
  end
  return s
end

local function is_latex_cmd_atom(atom)
  return atom:match("^\\[%a]+$") ~= nil
end

local math_word_stop = {
  ["and"] = true, ["or"] = true, ["for"] = true, ["all"] = true, ["the"] = true,
  ["is"] = true, ["are"] = true, ["mod"] = true, ["let"] = true,
}

local relation_cmds = {
  ["\\in"] = true, ["\\notin"] = true, ["\\le"] = true, ["\\leq"] = true,
  ["\\ge"] = true, ["\\geq"] = true, ["\\neq"] = true, ["\\subseteq"] = true,
  ["\\approx"] = true, ["\\to"] = true, ["\\leftarrow"] = true,
  ["\\Rightarrow"] = true, ["\\mapsto"] = true, ["\\leftrightarrow"] = true,
  ["\\Leftrightarrow"] = true,
}

local function is_relation_cmd(atom)
  return relation_cmds[atom] == true
end

local function is_math_atom(atom)
  if atom == nil or atom == "" then
    return false
  end
  if is_latex_cmd_atom(atom) then
    return true
  end
  if atom:match("^\\mathup%b{}_%b{}$") then
    return true
  end
  if atom:match("^[A-Za-z][A-Za-z0-9]*%b()$") then
    return true
  end
  if atom:match("^[%d]+$") then
    return true
  end
  if atom:match("^[%d]+/[%d]+$") then
    return true
  end
  if atom:match("^[A-Za-z]$") then
    return true
  end
  if greek_token_names[atom] then
    return true
  end
  if atom:match("^[A-Za-z][A-Za-z0-9]*_[{%w]") or atom:match("^[A-Za-z][A-Za-z0-9]*%^") then
    return true
  end
  if atom:match("^GL_%b{}$") or atom:match("^GL_[A-Za-z0-9%+%-]+$") or atom:match("^GLn[%+%-]?%d*$") then
    return true
  end
  if atom:match("^[A-Za-z][A-Za-z0-9]?[A-Za-z0-9]?$") then
    local low = atom:lower()
    if not math_word_stop[low] then
      return true
    end
  end
  if atom:match("^%b()$") or atom:match("^%b[]$") then
    return true
  end
  if atom:match("[\\{}_^]") then
    return true
  end
  return false
end

local function normalize_infix_ops(s)
  local function strip_unmatched_trailing(core, close_ch, open_ch)
    local suffix = ""
    while core:sub(-1) == close_ch do
      if core:find(open_ch, 1, true) then
        break
      end
      core = core:sub(1, -2)
      suffix = close_ch .. suffix
    end
    return core, suffix
  end

  local function split_atom_edges(atom)
    local prefix = atom:match("^([%(%[%{]*)") or ""
    local core = atom:sub(#prefix + 1)
    local punct = core:match("([%.,;:]*)$") or ""
    core = core:sub(1, #core - #punct)
    local s1, s2, s3
    core, s1 = strip_unmatched_trailing(core, ")", "(")
    core, s2 = strip_unmatched_trailing(core, "]", "[")
    core, s3 = strip_unmatched_trailing(core, "}", "{")
    local suffix = (s1 or "") .. (s2 or "") .. (s3 or "") .. punct
    if core == "" then
      core = atom
      prefix = ""
      suffix = ""
    end
    return prefix, core, suffix
  end

  local function repl(op_tex)
    return function(a, b)
      local ap, ac, as = split_atom_edges(a)
      local bp, bc, bs = split_atom_edges(b)
      local function is_operator_atom(atom)
        if atom == "\\ast" or atom == "\\times" or atom == "\\cdot" then
          return true
        end
        if atom:match("^\\mBridgeOperator%b{}$") then
          return true
        end
        return false
      end
      if op_tex == "\\times" and (
          is_relation_cmd(ac) or is_relation_cmd(bc) or
          is_operator_atom(ac) or is_operator_atom(bc) or
          bc == "is"
        ) then
        return a .. " x " .. b
      end
      if is_math_atom(ac) and is_math_atom(bc) then
        return ap .. ac .. as .. " " .. op_tex .. " " .. bp .. bc .. bs
      end
      return a .. " " .. (op_tex == "\\times" and "x" or (op_tex == "\\ast" and "*" or "in")) .. " " .. b
    end
  end

  -- Tight products without spaces (exclude star-subscript forms like F*_mu).
  s = s:gsub("([A-Za-z0-9%)%}])%*([A-Za-z0-9%(%{\\])", "%1 \\ast %2")
  s = s:gsub("([A-Za-z0-9%)%}])%s*%*%s*([A-Za-z0-9%(%{\\])", "%1 \\ast %2")

  -- Compact dimension form: 3x3 -> 3 \times 3
  s = s:gsub("(%f[%d]%d+)%s*[xX]%s*(%d+%f[%D])", "%1 \\times %2")

  local prev = nil
  while prev ~= s do
    prev = s
    s = s:gsub("(%S+)%s+[xX]%s+(%S+)", repl("\\times"))
    s = s:gsub("(%S+)%s*%*%s*(%S+)", repl("\\ast"))
    s = s:gsub("(%S+)%s+in%s+(%S+)", repl("\\in"))
  end
  return s
end

local function has_balanced_delimiters(s)
  local braces = 0
  local parens = 0
  for i = 1, #s do
    local ch = s:sub(i, i)
    if ch == "{" then
      braces = braces + 1
    elseif ch == "}" then
      braces = braces - 1
      if braces < 0 then
        return false
      end
    elseif ch == "(" then
      parens = parens + 1
    elseif ch == ")" then
      parens = parens - 1
      if parens < 0 then
        return false
      end
    end
  end
  return braces == 0 and parens == 0
end

local function normalize_expr(s)
  s = trim(s)
  s = s:gsub("%$", "")
  s = normalize_escaped_script_artifacts(s)

  s = s:gsub("{%[}", "[")
  s = s:gsub("{%]}", "]")
  s = s:gsub("\\textbar%s*_%s*", "|_")
  s = s:gsub("\\textbar%s*%^%s*", "|^")
  s = s:gsub("\\textbar%s*{}", "|")
  s = s:gsub("\\textbar", "|")
  s = s:gsub("\\textless%s*{}", "<")
  s = s:gsub("\\textgreater%s*{}", ">")
  s = s:gsub("\\textless", "<")
  s = s:gsub("\\textgreater", ">")
  s = s:gsub("(\\+)integral", "\\integral")

  -- Handle literal backslash separators like N_n\GL_n, but avoid LaTeX commands.
  s = s:gsub("([%w%)%}])\\([A-Z])", "%1 \\backslash %2")

  for _, pair in ipairs(unicode_ops) do
    s = s:gsub(pair[1], pair[2])
  end

  s = s:gsub("<=>", "\\Leftrightarrow ")
  s = s:gsub("!=", "\\neq ")
  s = s:gsub(">=", "\\ge ")
  s = s:gsub("<=", "\\le ")
  s = s:gsub(">>", "\\gg ")
  s = s:gsub("<<", "\\ll ")
  s = s:gsub("%-%>", "\\to ")
  s = s:gsub("<%-", "\\leftarrow ")
  s = replace_word(s, "eps", "\\epsilon")
  s = replace_word(s, "int", "\\Integral")
  s = replace_word(s, "det", "\\det")
  s = replace_word(s, "max", "\\max")
  s = replace_word(s, "inf", "\\inf")
  s = replace_word(s, "sqrt", "\\sqrt")
  s = replace_word(s, "sin", "\\sin")
  s = replace_word(s, "cos", "\\cos")
  s = replace_word(s, "log", "\\log")
  s = replace_word(s, "bigcup", "\\bigcup")
  s = replace_word(s, "subseteq", "\\subseteq")
  s = replace_word(s, "wedge", "\\wedge")
  s = replace_word(s, "char", "\\mOpName{char}")
  s = replace_word(s, "disc", "\\mOpName{disc}")
  s = replace_word(s, "dx", "\\mathrm{d}x")
  s = replace_word(s, "dy", "\\mathrm{d}y")
  s = replace_word(s, "dt", "\\mathrm{d}t")
  s = replace_word(s, "ds", "\\mathrm{d}s")
  s = replace_word(s, "dg", "\\mathrm{d}g")
  s = s:gsub("%f[%a]d([A-Z])", "\\mathrm{d}%1")
  s = s:gsub("([_%^])%s*{?infinity}?", "%1{\\infty}")
  s = replace_word(s, "infinity", "\\infty")
  s = s:gsub("%f[%a]int%s+(.-)%s+dx%f[%A]", function(body)
    return "\\Integral " .. trim(body) .. "\\,dx"
  end)
  s = replace_word(s, "SO", "\\mathup{SO}")
  s = replace_word(s, "Isom", "\\mathup{Isom}")
  s = replace_word(s, "FH", "\\mathup{FH}")
  s = replace_word(s, "SH", "\\mathup{SH}")
  s = replace_word(s, "EG", "\\mathup{EG}")
  s = replace_word(s, "Bpi", "B_{\\pi}")
  s = replace_word(s, "BPI", "\\mathup{BPI}")
  s = replace_word(s, "TITU", "\\mathup{TITU}")
  s = replace_word(s, "UBU", "\\mathup{UBU}")
  s = replace_word(s, "Symp", "\\mathup{Symp}")
  s = replace_word(s, "Ham", "\\mathup{Ham}")
  s = replace_word(s, "Fix", "\\mOpName{Fix}")
  s = replace_word(s, "vec", "\\mOpName{vec}")
  s = replace_word(s, "tr", "\\mOpName{tr}")
  s = s:gsub("%f[%a]diag%s*%(", "\\operatorname{diag}(")
  s = s:gsub("%f[%a]span%s*%(", "\\mOpName{span}(")
  s = s:gsub("%f[%a]ker%s*%(", "\\mOpName{ker}(")
  s = s:gsub("%f[%a]rank%s*%(", "\\mOpName{rank}(")
  s = s:gsub("%f[%a]dim%s*%(", "\\mOpName{dim}(")
  s = s:gsub("%f[%a]codim%s*%(", "\\mOpName{codim}(")
  s = s:gsub("%f[%a]Tr%s*%(", "\\mOpName{Tr}(")
  s = s:gsub("%f[%a]trace%s*%(", "\\mOpName{trace}(")
  s = s:gsub("^int%s*_", "\\int_")
  s = s:gsub("([^\\%a])int%s*_", "%1\\int_")
  s = s:gsub("^integral%s*_", "\\Integral_")
  s = s:gsub("([^\\%a])integral%s*_", "%1\\Integral_")
  s = s:gsub("^integral%f[%A]", "\\Integral")
  s = s:gsub("([^\\%a])integral%f[%A]", "%1\\Integral")
  s = s:gsub("^sum%s*_", "\\sum_")
  s = s:gsub("([^\\%a])sum%s*_", "%1\\sum_")
  s = s:gsub("^prod%s*_", "\\prod_")
  s = s:gsub("([^\\%a])prod%s*_", "%1\\prod_")
  s = s:gsub("%f[%a]exp%s*%(", "\\exp(")
  s = s:gsub("%f[%a]log%s*%(", "\\log(")
  s = s:gsub("%f[%a]det%s*%(", "\\det(")
  s = s:gsub("%f[%a]mod%s+(\\mNumber{%d+})", "\\mOpName{mod} %1")
  s = s:gsub("%f[%a]mod%s+(%d+)", "\\mOpName{mod} \\mNumber{%1}")

  -- Measure notation: dmu, dmu_0 -> d\mu, d\mu_0.
  s = s:gsub("%f[%a]dmu_%{([^}]+)%}", "d\\mu_{%1}")
  s = s:gsub("%f[%a]dmu_([A-Za-z0-9%+%-]+)", "d\\mu_%1")
  s = s:gsub("%f[%a]dmu%f[%A]", "d\\mu")

  -- Parenthesized exponents/subscripts (Q^(abgd) -> Q^{abgd}).
  s = s:gsub("%^%{%}%(([^()]+)%)", "^{%1}")
  s = s:gsub("_%{%}%(([^()]+)%)", "_{%1}")
  s = s:gsub("%^%(([^()]+)%)", "^{%1}")
  s = s:gsub("_%(([^()]+)%)", "_{%1}")
  s = s:gsub("(\\[%a]+)%s+([_%^])", "%1%2")

  -- Normalize bare subscripts/superscripts from pseudo-code (A_tau -> A_{tau}).
  s = s:gsub("([%a%)%}])_([A-Za-z][A-Za-z0-9]*)", "%1_{%2}")
  s = s:gsub("([%a%)%}])%^([A-Za-z][A-Za-z0-9]*)", "%1^{%2}")
  s = s:gsub("}_([A-Za-z][A-Za-z0-9]*)", "}_{%1}")
  s = s:gsub("}%^([A-Za-z][A-Za-z0-9]*)", "}^{%1}")
  s = s:gsub("([%a%)%}])_\\([A-Za-z]+)(%b{})", "%1_{\\%2%3}")
  s = s:gsub("([%a%)%}])%^\\([A-Za-z]+)(%b{})", "%1^{\\%2%3}")
  s = s:gsub("}_\\([A-Za-z]+)(%b{})", "}_{\\%1%2}")
  s = s:gsub("}%^\\([A-Za-z]+)(%b{})", "}^{\\%1%2}")
  s = s:gsub("([%a%)%}])_\\([A-Za-z]+)", "%1_{\\%2}")
  s = s:gsub("([%a%)%}])%^\\([A-Za-z]+)", "%1^{\\%2}")
  s = s:gsub("}_\\([A-Za-z]+)", "}_{\\%1}")
  s = s:gsub("}%^\\([A-Za-z]+)", "}^{\\%1}")
  s = s:gsub("%^\\%*", "^{\\mDualStar}")
  s = s:gsub("%^%*", "^{\\mDualStar}")
  s = s:gsub("_\\%*", "_{\\mDualStar}")
  s = s:gsub("_%*", "_{\\mDualStar}")
  s = s:gsub("([A-Za-z][A-Za-z0-9]*)_{([^}]+)}_{([^}]+)}", "%1_{%2,%3}")

  -- Group names: GL_n, GL_{n+1}, GLn+1.
  s = s:gsub("GL_%{([^}]+)%}", "\\mathup{GL}_{%1}")
  s = s:gsub("GL_([A-Za-z0-9%+%-]+)", "\\mathup{GL}_{%1}")
  s = s:gsub("GLn([%+%-]%d+)", "\\mathup{GL}_{n%1}")
  s = s:gsub("GLn(%f[%A])", "\\mathup{GL}_{n}%1")

  -- Treat a bare '*' suffix as ascii-art prime in symbolic tokens.
  s = s:gsub("([A-Za-z\\][A-Za-z0-9\\]*)%*_%{([^}]+)%}", "%1^{\\prime}_{%2}")
  s = s:gsub("([A-Za-z\\][A-Za-z0-9\\]*)%*_([A-Za-z0-9]+)", "%1^{\\prime}_{%2}")
  s = s:gsub("([A-Za-z\\][A-Za-z0-9\\]*)%*(%f[%A])", "%1^{\\prime}%2")

  -- Parenthesized products: (A x B), (A * B)
  s = s:gsub("%(([%w\\{}_%^%+%-]+)%s+[xX]%s+([%w\\{}_%^%+%-]+)%)", "(%1 \\times %2)")
  s = s:gsub("%(([%w\\{}_%^%+%-]+)%s*%*%s*([%w\\{}_%^%+%-]+)%)", "(%1 \\ast %2)")

  s = normalize_infix_ops(s)

  s = replace_word(s, "in", "\\in")
  s = replace_word(s, "notin", "\\notin")
  s = replace_word(s, "rtimes", "\\rtimes")

  for _, pair in ipairs(greek_words) do
    s = replace_word(s, pair[1], pair[2])
  end

  -- Explicit hat-token normalization (domain vocabulary).
  s = replace_word(s, "Ahat", "\\hat{A}")
  s = replace_word(s, "Phat", "\\hat{P}")
  s = replace_word(s, "phat", "\\hat{p}")
  s = replace_word(s, "ahat", "\\hat{a}")

  -- Prose connectors should not remain as bare words in math.
  local prose_words = {
    "for", "and", "such", "that", "there", "exists", "where", "with",
    "all", "every", "each", "is", "so", "to", "can", "one", "outside", "through",
    "internal", "the", "The", "There", "transfer", "Transfer", "allowed", "Allowed",
    "assumption", "exact", "eg", "fix", "sobolev", "hadamard", "symp", "ham",
    "titu", "ubu", "bpi", "cholesky", "Assumption", "EXACT", "Sobolev", "Cholesky",
    "Hadamard", "Titu", "additive", "admissible", "analogous", "axes", "choose",
    "connected", "const", "conv", "correction", "corrections", "cross", "curve",
    "equivalent", "graph", "lemma", "light", "local", "match", "measures", "nr",
    "of", "order", "origin", "poly", "positive", "renorm", "reparameterized",
    "sector", "selection", "smooth", "space", "subgroups", "surplus", "tensor",
    "term", "theorem", "transition", "vertex", "weighted", "ceil",
  }
  local prose_masked, prose_slots = mask_text_macros(s)
  for _, w in ipairs(prose_words) do
    prose_masked = wrap_prose_word(prose_masked, w)
  end
  s = unmask_text_macros(prose_masked, prose_slots)
  s = flatten_nested_text_macros(s)

  -- Prevent glued prose after Greek commands: \muare -> \mu are.
  local greek_cmds = {
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
  }
  for _, cmd in ipairs(greek_cmds) do
    s = s:gsub("\\" .. cmd .. "([A-Za-z])", "\\" .. cmd .. " %1")
  end

  s = mark_integer_literals(s)

  s = s:gsub("#", "\\#")
  s = s:gsub("%%", "\\%%")

  s = s:gsub("%s+", " ")
  return trim(s)
end

local function keep_inline_code(s)
  if s:match("^/") or s:match("^scripts/") or s:match("^data/") then
    return true
  end
  if s:match("%.md$") or s:match("%.py$") or s:match("%.jsonl$") then
    return true
  end
  return false
end

local function is_mathy_inline(s)
  if s:match("[_%^=<>%+%-%*/{}%[%]%(%)]") then
    return true
  end
  if s:match("FH%(") then
    return true
  end
  if s:match("Gamma") or s:match("pi") or s:match("tau") or s:match("sigma") then
    return true
  end
  return false
end

local function is_compound_math_token(s)
  if not s:match("[_%^]") then
    return false
  end
  if not s:match("[%a%d]") then
    return false
  end
  -- Keep prose suffixes outside math: GL_n-equivariant -> $GL_n$-equivariant.
  if s:match("%-[A-Za-z][A-Za-z][A-Za-z]") then
    return false
  end
  if not has_balanced_delimiters(s) then
    return false
  end
  return s:match("^[^%s]+$") ~= nil
end

local function push_str(out, s)
  if s == "" then
    return
  end
  local n = #out
  if n > 0 and out[n].t == "Str" then
    out[n].text = out[n].text .. s
  else
    table.insert(out, pandoc.Str(s))
  end
end

local function consume_group(s, i, open_ch, close_ch)
  if s:sub(i, i) ~= open_ch then
    return nil
  end
  local depth = 1
  local j = i + 1
  while j <= #s do
    local ch = s:sub(j, j)
    if ch == open_ch then
      depth = depth + 1
    elseif ch == close_ch then
      depth = depth - 1
      if depth == 0 then
        return j
      end
    end
    j = j + 1
  end
  return nil
end

local function consume_script(s, i)
  if i > #s then
    return nil
  end
  local ch = s:sub(i, i)
  if ch == "{" then
    return consume_group(s, i, "{", "}")
  end
  if ch == "(" then
    return consume_group(s, i, "(", ")")
  end
  if ch == "\\" then
    local j = i + 1
    while j <= #s and s:sub(j, j):match("[%a]") do
      j = j + 1
    end
    if j == i + 1 then
      return nil
    end
    return j - 1
  end
  if ch:match("[%w]") then
    local j = i
    while j <= #s and s:sub(j, j):match("[%w]") do
      j = j + 1
    end
    return j - 1
  end
  return nil
end

local function find_next_mathy_segment(s, from_idx)
  local i = from_idx
  while i <= #s do
    if s:sub(i, i):match("[%a]") then
      local j = i
      while j <= #s and s:sub(j, j):match("[%w]") do
        j = j + 1
      end
      local k = j
      local has_script = false
      while k <= #s do
        local op = s:sub(k, k)
        if op == "^" or op == "_" then
          local end_idx = consume_script(s, k + 1)
          if end_idx == nil then
            break
          end
          has_script = true
          k = end_idx + 1
        else
          break
        end
      end
      if has_script then
        return i, k - 1
      end
    end
    i = i + 1
  end
  return nil
end

local function convert_mathy_segments(token)
  local out = {}
  local pos = 1
  local converted = false
  while true do
    local a, b = find_next_mathy_segment(token, pos)
    if a == nil then
      break
    end
    push_str(out, token:sub(pos, a - 1))
    table.insert(out, pandoc.Math("InlineMath", normalize_expr(token:sub(a, b))))
    converted = true
    pos = b + 1
  end
  push_str(out, token:sub(pos))
  return out, converted
end

local function brace_delta(s)
  local d = 0
  for i = 1, #s do
    local ch = s:sub(i, i)
    if ch == "{" then
      d = d + 1
    elseif ch == "}" then
      d = d - 1
    end
  end
  return d
end

local function paren_delta(s)
  local d = 0
  for i = 1, #s do
    local ch = s:sub(i, i)
    if ch == "(" then
      d = d + 1
    elseif ch == ")" then
      d = d - 1
    end
  end
  return d
end

local function extract_unclosed_script_start_token(s)
  -- Accept prefixes like "(", "1/(", "-" before a split script token.
  -- Example: "(sum_{j" -> prefix "(", core "sum_{j".
  local normalized = normalize_escaped_script_artifacts(s)
  local prefix, core = normalized:match("^(.-)([%a][%w]-[_%^][{].*)$")
  if core == nil then
    return nil, nil
  end
  if core:find("}", 1, true) then
    return nil, nil
  end
  if prefix:match("%a") then
    return nil, nil
  end
  return prefix, core
end

local function is_comma_varlist(s)
  local saw_comma = false
  local expect_letter = true
  for i = 1, #s do
    local ch = s:sub(i, i)
    if expect_letter then
      if not ch:match("[a-z]") then
        return false
      end
      expect_letter = false
    else
      if ch ~= "," then
        return false
      end
      saw_comma = true
      expect_letter = true
    end
  end
  return saw_comma and (not expect_letter)
end

local function append_inlines(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

local function is_space_like(el)
  return el.t == "Space" or el.t == "SoftBreak"
end

local function parse_token_with_punct(s)
  local core, punct = s:match("^(.-)([%.,;:]*)$")
  if core == nil then
    return nil, nil
  end
  return core, punct
end

local function extract_binary_side(el)
  if el.t == "Math" and el.mathtype == "InlineMath" then
    return el.text, "", true
  end
  if el.t ~= "Str" then
    return nil, nil, nil
  end
  local core, punct = parse_token_with_punct(el.text)
  if core == nil or core == "" then
    return nil, nil, nil
  end
  if core:match("^[A-Za-z]$") then
    return core, punct, false
  end
  if core:match("^%d+$") then
    return core, punct, false
  end
  if core:match("^[A-Za-z0-9]+/[A-Za-z0-9]+$") then
    return core, punct, false
  end
  if core == "infinity" or core == "Infinity" or core == "inf" then
    return core, punct, false
  end
  if is_compound_math_token(core) then
    return core, punct, false
  end
  if core:match("^GL_%b{}$") or core:match("^GL_[A-Za-z0-9%+%-]+$") or core:match("^GLn[%+%-]?%d*$") then
    return core, punct, false
  end
  return nil, nil, nil
end

local function normalize_binary_side(raw, already_math)
  if already_math then
    return raw
  end
  if raw:match("^[A-Za-z]$") then
    return raw
  end
  return normalize_expr(raw)
end

local function build_binary_expr(lhs, lhs_is_math, op, rhs, rhs_is_math)
  local op_tex = nil
  if op == "in" then
    op_tex = "\\in"
  elseif op == "x" or op == "X" then
    op_tex = "\\times"
  elseif op == "+" then
    op_tex = "\\mBridgeOperator{+}"
  elseif op == "*" then
    op_tex = "\\ast"
  elseif op == "=" then
    op_tex = "="
  elseif op == "<" then
    op_tex = "<"
  elseif op == ">" then
    op_tex = ">"
  elseif op == "<=" then
    op_tex = "\\le"
  elseif op == ">=" then
    op_tex = "\\ge"
  elseif op == "<<" then
    op_tex = "\\ll"
  elseif op == ">>" then
    op_tex = "\\gg"
  elseif op == "->" then
    op_tex = "\\to"
  elseif op == "<-" then
    op_tex = "\\leftarrow"
  elseif op == "<=>" then
    op_tex = "\\Leftrightarrow"
  elseif op == "!=" then
    op_tex = "\\neq"
  elseif op == "approx" then
    op_tex = "\\approx"
  else
    return nil
  end
  return normalize_binary_side(lhs, lhs_is_math) .. " " .. op_tex .. " " .. normalize_binary_side(rhs, rhs_is_math)
end

local function extract_binary_operator(el)
  if el == nil then
    return nil
  end
  if el.t == "Str" then
    if el.text == "in" or el.text == "x" or el.text == "X" or el.text == "+" or el.text == "*" then
      return el.text
    end
    if el.text == "=" or el.text == "<" or el.text == ">" or el.text == "<=" or el.text == ">=" or
       el.text == "!=" or el.text == "<<" or el.text == ">>" or el.text == "->" or el.text == "<-" or
       el.text == "<=>" then
      return el.text
    end
    if el.text == "≈" then
      return "approx"
    end
    return nil
  end
  if el.t == "Math" and el.mathtype == "InlineMath" then
    local txt = trim(el.text)
    if txt == "+" or txt == "=" or txt == "<" or txt == ">" then
      return txt
    end
    if txt == "\\le" or txt == "\\leq" then
      return "<="
    end
    if txt == "\\ge" or txt == "\\geq" then
      return ">="
    end
    if txt == "\\ll" then
      return "<<"
    end
    if txt == "\\gg" then
      return ">>"
    end
    if txt == "\\to" then
      return "->"
    end
    if txt == "\\leftarrow" then
      return "<-"
    end
    if txt == "\\Leftrightarrow" then
      return "<=>"
    end
    if txt == "\\neq" then
      return "!="
    end
    if txt == "\\approx" then
      return "approx"
    end
    if txt == "\\in" then
      return "in"
    end
    if txt == "\\times" then
      return "x"
    end
    if txt == "\\ast" then
      return "*"
    end
  end
  return nil
end

local function convert_binary_operator_sequences_once(inlines)
  local out = {}
  local i = 1
  local changed = false
  while i <= #inlines do
    local a = inlines[i]
    local b = inlines[i + 1]
    local c = inlines[i + 2]
    local d = inlines[i + 3]
    local e = inlines[i + 4]
    local op = extract_binary_operator(c)
    if a ~= nil and b ~= nil and c ~= nil and d ~= nil and e ~= nil
      and is_space_like(b) and is_space_like(d)
      and op ~= nil then
      local lhs, lpunct, lhs_is_math = extract_binary_side(a)
      local rhs, rpunct, rhs_is_math = extract_binary_side(e)
      if lhs ~= nil and rhs ~= nil and lpunct == "" then
        local expr = build_binary_expr(lhs, lhs_is_math, op, rhs, rhs_is_math)
        if expr ~= nil then
          table.insert(out, pandoc.Math("InlineMath", expr))
          if rpunct ~= "" then
            table.insert(out, pandoc.Str(rpunct))
          end
          i = i + 5
          changed = true
        else
          table.insert(out, a)
          i = i + 1
        end
      else
        table.insert(out, a)
        i = i + 1
      end
    else
      table.insert(out, a)
      i = i + 1
    end
  end
  return out, changed
end

local function convert_binary_operator_sequences(inlines)
  local cur = inlines
  while true do
    local nxt, changed = convert_binary_operator_sequences_once(cur)
    if not changed then
      return nxt
    end
    cur = nxt
  end
end

local function is_symbolic_glue_str(s)
  if s == nil or s == "" then
    return false
  end
  if s:match("^%d+$") then
    return false
  end
  -- Allow compact fraction starters like "1/(x" to glue into adjacent math.
  if s:match("^%d+/%([A-Za-z]$") then
    return true
  end
  if s:match("[A-Za-z]") then
    return false
  end
  return s:match("^[%s%(%[%{%)%]%,%+%-%*/=<>'`|\\%^_%d%.]+$") ~= nil
end

local function merge_adjacent_math_fragments(inlines)
  local out = {}
  local i = 1
  while i <= #inlines do
    local cur = inlines[i]
    if cur ~= nil and cur.t == "Math" and cur.mathtype == "InlineMath" then
      local pieces = {cur.text}
      local j = i + 1
      local math_count = 1
      while j <= #inlines do
        local nxt = inlines[j]
        if nxt.t == "Math" and nxt.mathtype == "InlineMath" then
          table.insert(pieces, nxt.text)
          math_count = math_count + 1
          j = j + 1
        elseif nxt.t == "Space" or nxt.t == "SoftBreak" then
          local look = inlines[j + 1]
          if look ~= nil and ((look.t == "Math" and look.mathtype == "InlineMath") or (look.t == "Str" and is_symbolic_glue_str(look.text))) then
            table.insert(pieces, " ")
            j = j + 1
          else
            break
          end
        elseif nxt.t == "Str" and is_symbolic_glue_str(nxt.text) then
          table.insert(pieces, nxt.text)
          j = j + 1
        else
          break
        end
      end

      if math_count >= 2 then
        table.insert(out, pandoc.Math("InlineMath", normalize_expr(table.concat(pieces))))
        i = j
      else
        table.insert(out, cur)
        i = i + 1
      end
    else
      table.insert(out, cur)
      i = i + 1
    end
  end
  return out
end

local function convert_simple_str_token_to_inlines(s)
  local op_tex = ({
    ["->"] = "\\to",
    ["<-"] = "\\leftarrow",
    ["<=>"] = "\\Leftrightarrow",
    ["<<"] = "\\ll",
    [">>"] = "\\gg",
    ["<"] = "<",
    [">"] = ">",
    ["="] = "=",
    ["<="] = "\\le",
    [">="] = "\\ge",
    ["!="] = "\\neq",
    ["≈"] = "\\approx",
  })[s]
  if op_tex ~= nil then
    return {pandoc.Math("InlineMath", op_tex)}
  end

  local abs_inner, abs_punct = s:match("^|([^|]+)|([%.,;:]*)$")
  if abs_inner ~= nil and abs_inner:match("^[%w\\_%^%+%-%s]+$") then
    local out = {pandoc.Math("InlineMath", "|" .. normalize_expr(trim(abs_inner)) .. "|")}
    if abs_punct ~= "" then
      table.insert(out, pandoc.Str(abs_punct))
    end
    return out
  end

  local compact_core, compact_punct = s:match("^(.-)([%.,;:]*)$")
  if compact_core ~= nil then
    local compact_ops = {"<=>", "->", "<-", "<<", ">>", "<=", ">=", "!=", "=", "<", ">"}
    for _, op in ipairs(compact_ops) do
      local idx = compact_core:find(op, 1, true)
      if idx ~= nil and idx > 1 and idx <= (#compact_core - #op + 1) then
        local lhs = compact_core:sub(1, idx - 1)
        local rhs = compact_core:sub(idx + #op)
        if lhs ~= "" and rhs ~= "" and lhs:match("^[%w\\{}_^%(%)[%]/|%-]+$") and rhs:match("^[%w\\{}_^%(%)[%]/|%-]+$") then
          local expr = build_binary_expr(lhs, false, op, rhs, false)
          if expr ~= nil then
            local out = {pandoc.Math("InlineMath", expr)}
            if compact_punct ~= "" then
              table.insert(out, pandoc.Str(compact_punct))
            end
            return out
          end
        end
      end
    end
  end

  local hat_core, hat_punct = s:match("^([A-Za-z]+)([%.,;:]*)$")
  local hat_map = {
    Ahat = "\\hat{A}",
    Phat = "\\hat{P}",
    phat = "\\hat{p}",
    ahat = "\\hat{a}",
  }
  if hat_core ~= nil and hat_map[hat_core] ~= nil then
    local out = {pandoc.Math("InlineMath", hat_map[hat_core])}
    if hat_punct ~= "" then
      table.insert(out, pandoc.Str(hat_punct))
    end
    return out
  end

  local esc_core, esc_punct = s:match("^(.-)([%.,;:]*)$")
  if esc_core ~= nil and (
      esc_core:find("\\_", 1, true) or
      esc_core:find("\\^", 1, true) or
      esc_core:find("\\textasciicircum", 1, true)
    ) then
    local normalized = normalize_escaped_script_artifacts(esc_core)
    if normalized:match("[_%^]") then
      local out = {pandoc.Math("InlineMath", normalize_expr(normalized))}
      if esc_punct ~= "" then
        table.insert(out, pandoc.Str(esc_punct))
      end
      return out
    end
  end

  local relcore, relpunct = s:match("^(.-)([%.,;:]*)$")
  if relcore ~= nil and (
      relcore:find("\\textless", 1, true) or
      relcore:find("\\textgreater", 1, true) or
      relcore:find("\\textbar", 1, true)
    ) then
    local out = {pandoc.Math("InlineMath", normalize_expr(relcore))}
    if relpunct ~= "" then
      table.insert(out, pandoc.Str(relpunct))
    end
    return out
  end

  local binner, bpunct = s:match("^%[([^%[%]]+)%]([%.,;:]*)$")
  if binner ~= nil and binner:match("[%w\\_%^%+%-%s,]+") then
    local out = {pandoc.Math("InlineMath", "[" .. normalize_expr(trim(binner)) .. "]")}
    if bpunct ~= "" then
      table.insert(out, pandoc.Str(bpunct))
    end
    return out
  end

  local sinner, spunct = s:match("^%{([^{}]+)%}([%.,;:]*)$")
  if sinner ~= nil and sinner:match("[%w\\_%^%+%-%s,]+") then
    local out = {pandoc.Math("InlineMath", "\\{" .. normalize_expr(trim(sinner)) .. "\\}")}
    if spunct ~= "" then
      table.insert(out, pandoc.Str(spunct))
    end
    return out
  end

  local gfn, gargs, gpunct = s:match("^([A-Za-z]+)%(([^()]*)%)([%.,;:]*)$")
  if gfn ~= nil and greek_token_names[gfn] then
    local out = {pandoc.Math("InlineMath", normalize_expr(gfn .. "(" .. gargs .. ")"))}
    if gpunct ~= "" then
      table.insert(out, pandoc.Str(gpunct))
    end
    return out
  end

  local gopen, gtail = s:match("^([A-Za-z]+)%((.*)$")
  if gopen ~= nil and greek_token_names[gopen] then
    return {pandoc.Math("InlineMath", normalize_expr(gopen)), pandoc.Str("(" .. gtail)}
  end

  local pre_diag, diag_args_any, post_diag = s:match("^(.-)diag%(([^()]*)%)(.*)$")
  if diag_args_any ~= nil and (pre_diag == "" or not pre_diag:match("[%a]$")) then
    local out = {}
    if pre_diag ~= "" then
      table.insert(out, pandoc.Str(pre_diag))
    end
    table.insert(out, pandoc.Math("InlineMath", "\\operatorname{diag}(" .. normalize_expr(diag_args_any) .. ")"))
    if post_diag ~= "" then
      table.insert(out, pandoc.Str(post_diag))
    end
    return out
  end

  local pdiag_args, pdiag_punct = s:match("^%((diag%([^()]*%))([%.,;:]*)$")
  if pdiag_args ~= nil then
    local inner = pdiag_args:match("^diag%(([^()]*)%)$")
    local out = {pandoc.Str("("), pandoc.Math("InlineMath", "\\operatorname{diag}(" .. normalize_expr(inner) .. ")")}
    if pdiag_punct ~= "" then
      table.insert(out, pandoc.Str(pdiag_punct))
    end
    return out
  end

  local diag_args, dpunct = s:match("^diag%(([^()]*)%)([%.,;:]*)$")
  if diag_args ~= nil then
    local out = {pandoc.Math("InlineMath", "\\operatorname{diag}(" .. normalize_expr(diag_args) .. ")")}
    if dpunct ~= "" then
      table.insert(out, pandoc.Str(dpunct))
    end
    return out
  end

  local greek, punct = s:match("^([A-Za-z]+)([%.,;:]*)$")
  if greek ~= nil and greek_token_names[greek] then
    local out = {pandoc.Math("InlineMath", normalize_expr(greek))}
    if punct ~= "" then
      table.insert(out, pandoc.Str(punct))
    end
    return out
  end

  local varlist, vpunct = s:match("^([a-z][a-z,]*)([%.,;:]*)$")
  if varlist ~= nil and is_comma_varlist(varlist) then
    local out = {pandoc.Math("InlineMath", normalize_expr(varlist))}
    if vpunct ~= "" then
      table.insert(out, pandoc.Str(vpunct))
    end
    return out
  end
  local pvarlist, pvpunct = s:match("^%(([a-z][a-z,]*)%)([%.,;:]*)$")
  if pvarlist ~= nil and is_comma_varlist(pvarlist) then
    local out = {pandoc.Str("("), pandoc.Math("InlineMath", normalize_expr(pvarlist)), pandoc.Str(")")}
    if pvpunct ~= "" then
      table.insert(out, pandoc.Str(pvpunct))
    end
    return out
  end

  local hy_math, hy_word, hy_punct = s:match("^(.-)%-([A-Za-z][A-Za-z][A-Za-z][A-Za-z%-]*)([%.,;:]*)$")
  if hy_math ~= nil and hy_math ~= "" then
    local mathy_prefix = false
    if greek_token_names[hy_math] then
      mathy_prefix = true
    elseif is_compound_math_token(hy_math) then
      mathy_prefix = true
    elseif hy_math:match("^GL_%b{}$") or hy_math:match("^GL_[A-Za-z0-9%+%-]+$") or hy_math:match("^GLn[%+%-]?%d*$") then
      mathy_prefix = true
    end
    if mathy_prefix then
      local out = {
        pandoc.Math("InlineMath", normalize_expr(hy_math)),
        pandoc.Str("-" .. hy_word),
      }
      if hy_punct ~= "" then
        table.insert(out, pandoc.Str(hy_punct))
      end
      return out
    end
  end

  -- Let block-level combiner handle split brace patterns like lambda_{alpha_m,
  if s:match("^[A-Za-z][A-Za-z0-9]*[%^_]%{[^}]*$") then
    return nil
  end

  local comp_core, comp_punct = parse_token_with_punct(s)
  if comp_core ~= nil and comp_core ~= "" and is_compound_math_token(comp_core) then
    local out = {pandoc.Math("InlineMath", normalize_expr(comp_core))}
    if comp_punct ~= "" then
      table.insert(out, pandoc.Str(comp_punct))
    end
    return out
  end

  if s:match("[_%^]") then
    local out, converted = convert_mathy_segments(s)
    if converted then
      return out
    end
  end

  return nil
end

local function convert_split_brace_math_inlines(inlines)
  local out = {}
  local i = 1
  while i <= #inlines do
    local el = inlines[i]
    local start_prefix = nil
    local start_core = nil
    if el.t == "Str" then
      start_prefix, start_core = extract_unclosed_script_start_token(el.text)
    end

    if start_core ~= nil then
      local first = (start_prefix or "") .. start_core
      local parts = {first}
      local j = i + 1
      local depth = brace_delta(first)
      local pdepth = paren_delta(first)
      local closed = (depth <= 0 and pdepth <= 0)

      while j <= #inlines and not closed do
        local ej = inlines[j]
        if ej.t == "Space" or ej.t == "SoftBreak" then
          table.insert(parts, " ")
        elseif ej.t == "Str" then
          local piece = normalize_escaped_script_artifacts(ej.text)
          table.insert(parts, piece)
          depth = depth + brace_delta(piece)
          pdepth = pdepth + paren_delta(piece)
          if depth <= 0 and pdepth <= 0 then
            closed = true
          end
        elseif ej.t == "Math" and ej.mathtype == "InlineMath" then
          table.insert(parts, ej.text)
          depth = depth + brace_delta(ej.text)
          pdepth = pdepth + paren_delta(ej.text)
          if depth <= 0 and pdepth <= 0 then
            closed = true
          end
        else
          break
        end
        j = j + 1
      end

      if closed then
        local raw = table.concat(parts)
        if has_balanced_delimiters(raw) and raw:match("[_%^]") and raw:match("[%a]") then
          local core, tail = raw:match("^(.-})([%.,;:]*)$")
          if core ~= nil then
            raw = core
          else
            tail = ""
          end
          table.insert(out, pandoc.Math("InlineMath", normalize_expr(raw)))
          if tail ~= "" then
            table.insert(out, pandoc.Str(tail))
          end
          i = j
        else
          local conv = convert_simple_str_token_to_inlines(el.text)
          if conv ~= nil then
            append_inlines(out, conv)
          else
            table.insert(out, el)
          end
          i = i + 1
        end
      else
        local conv = convert_simple_str_token_to_inlines(el.text)
        if conv ~= nil then
          append_inlines(out, conv)
        else
          table.insert(out, el)
        end
        i = i + 1
      end
    else
      if el.t == "Str" then
        local conv = convert_simple_str_token_to_inlines(el.text)
        if conv ~= nil then
          append_inlines(out, conv)
        else
          table.insert(out, el)
        end
      else
        table.insert(out, el)
      end
      i = i + 1
    end
  end
  return out
end

function Str(el)
  local s = el.text
  if keep_inline_code(s) then
    return nil
  end
  if s:find("://", 1, true) then
    return nil
  end

  local conv = convert_simple_str_token_to_inlines(s)
  if conv ~= nil then
    return conv
  end
  return nil
end

function Code(el)
  local s = el.text
  if keep_inline_code(s) then
    return nil
  end
  if is_mathy_inline(s) then
    return pandoc.Math("InlineMath", normalize_expr(s))
  end
  return nil
end

function Math(el)
  return pandoc.Math(el.mathtype, normalize_expr(el.text))
end

function CodeBlock(el)
  local txt = trim(el.text)
  if txt == "" then
    return nil
  end

  -- Keep genuine code blocks untouched.
  if txt:match("^def%s+") or txt:match("^import%s+") or txt:match("^class%s+") or txt:match("^#!/") then
    return nil
  end

  for line in txt:gmatch("[^\n]+") do
    if looks_like_algorithm_line(line) then
      return nil
    end
  end

  local lines = {}
  for line in txt:gmatch("[^\n]+") do
    line = trim(line)
    if line ~= "" then
      table.insert(lines, normalize_expr(line))
    end
  end

  if #lines == 0 then
    return nil
  end

  if #lines == 1 then
    return pandoc.Para({pandoc.Math("DisplayMath", lines[1])})
  end

  local body = table.concat(lines, " \\\\\n")
  return pandoc.Para({pandoc.Math("DisplayMath", "\\begin{aligned}\n" .. body .. "\n\\end{aligned}")})
end

function Para(el)
  el.content = convert_split_brace_math_inlines(el.content)
  el.content = convert_binary_operator_sequences(el.content)
  el.content = merge_adjacent_math_fragments(el.content)
  return el
end

function Plain(el)
  el.content = convert_split_brace_math_inlines(el.content)
  el.content = convert_binary_operator_sequences(el.content)
  el.content = merge_adjacent_math_fragments(el.content)
  return el
end

local function normalize_inline_container(el)
  el.content = convert_split_brace_math_inlines(el.content)
  el.content = convert_binary_operator_sequences(el.content)
  el.content = merge_adjacent_math_fragments(el.content)
  return el
end

function Header(el)
  return normalize_inline_container(el)
end

function Emph(el)
  return normalize_inline_container(el)
end

function Strong(el)
  return normalize_inline_container(el)
end

function Strikeout(el)
  return normalize_inline_container(el)
end

function Superscript(el)
  return normalize_inline_container(el)
end

function Subscript(el)
  return normalize_inline_container(el)
end

function SmallCaps(el)
  return normalize_inline_container(el)
end

function Span(el)
  return normalize_inline_container(el)
end
