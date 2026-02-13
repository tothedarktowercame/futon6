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

local function replace_word(s, w, repl)
  s = s:gsub("^" .. w .. "%f[%A]", repl)
  s = s:gsub("([^\\%a])" .. w .. "%f[%A]", "%1" .. repl)
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
      if op_tex == "\\times" and (is_relation_cmd(ac) or is_relation_cmd(bc) or bc == "is") then
        return a .. " x " .. b
      end
      if is_math_atom(ac) and is_math_atom(bc) then
        return ap .. ac .. as .. " " .. op_tex .. " " .. bp .. bc .. bs
      end
      return a .. " " .. (op_tex == "\\times" and "x" or (op_tex == "\\ast" and "*" or "in")) .. " " .. b
    end
  end

  -- Tight products without spaces.
  s = s:gsub("([%w%)%}])%*([%w%(%{\\])", "%1 \\ast %2")

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

  -- Handle literal backslash separators like N_n\GL_n.
  s = s:gsub("([%w%)%}])\\([%w%(])", "%1 \\backslash %2")

  for _, pair in ipairs(unicode_ops) do
    s = s:gsub(pair[1], pair[2])
  end

  s = s:gsub("<=>", "\\Leftrightarrow ")
  s = s:gsub("!=", "\\neq ")
  s = s:gsub(">=", "\\ge ")
  s = s:gsub("<=", "\\le ")
  s = s:gsub("%-%>", "\\to ")
  s = s:gsub("<%-", "\\leftarrow ")
  s = s:gsub("%f[%a]diag%s*%(", "\\operatorname{diag}(")

  -- Parenthesized exponents/subscripts (Q^(abgd) -> Q^{abgd}).
  s = s:gsub("%^%(([^()]+)%)", "^{%1}")
  s = s:gsub("_%(([^()]+)%)", "_{%1}")

  -- Normalize bare subscripts/superscripts from pseudo-code (A_tau -> A_{tau}).
  s = s:gsub("([%a%)%}])_([A-Za-z][A-Za-z0-9]*)", "%1_{%2}")
  s = s:gsub("([%a%)%}])%^([A-Za-z][A-Za-z0-9]*)", "%1^{%2}")
  s = s:gsub("}_([A-Za-z][A-Za-z0-9]*)", "}_{%1}")
  s = s:gsub("}%^([A-Za-z][A-Za-z0-9]*)", "}^{%1}")
  s = s:gsub("([%a%)%}])_\\([A-Za-z]+)", "%1_{\\%2}")
  s = s:gsub("([%a%)%}])%^\\([A-Za-z]+)", "%1^{\\%2}")
  s = s:gsub("}_\\([A-Za-z]+)", "}_{\\%1}")
  s = s:gsub("}%^\\([A-Za-z]+)", "}^{\\%1}")
  s = s:gsub("([A-Za-z][A-Za-z0-9]*)_{([^}]+)}_{([^}]+)}", "%1_{%2,%3}")

  -- Group names: GL_n, GL_{n+1}, GLn+1.
  s = s:gsub("GL_%{([^}]+)%}", "\\mathup{GL}_{%1}")
  s = s:gsub("GL_([A-Za-z0-9%+%-]+)", "\\mathup{GL}_{%1}")
  s = s:gsub("GLn([%+%-]%d+)", "\\mathup{GL}_{n%1}")
  s = s:gsub("GLn(%f[%A])", "\\mathup{GL}_{n}%1")

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
  if not s:match("[%a]") then
    return false
  end
  if not has_balanced_delimiters(s) then
    return false
  end
  return s:match("^[A-Za-z0-9_%^{}%*%+%-%/%.:,()]+$") ~= nil
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
  elseif op == "*" then
    op_tex = "\\ast"
  else
    return nil
  end
  return normalize_binary_side(lhs, lhs_is_math) .. " " .. op_tex .. " " .. normalize_binary_side(rhs, rhs_is_math)
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
    if a ~= nil and b ~= nil and c ~= nil and d ~= nil and e ~= nil
      and is_space_like(b) and is_space_like(d)
      and c.t == "Str" and (c.text == "in" or c.text == "x" or c.text == "X" or c.text == "*") then
      local lhs, lpunct, lhs_is_math = extract_binary_side(a)
      local rhs, rpunct, rhs_is_math = extract_binary_side(e)
      if lhs ~= nil and rhs ~= nil and lpunct == "" then
        local expr = build_binary_expr(lhs, lhs_is_math, c.text, rhs, rhs_is_math)
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

local function convert_simple_str_token_to_inlines(s)
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

  -- Let block-level combiner handle split brace patterns like lambda_{alpha_m,
  if s:match("^[A-Za-z][A-Za-z0-9]*[%^_]%{[^}]*$") then
    return nil
  end

  if is_compound_math_token(s) then
    return {pandoc.Math("InlineMath", normalize_expr(s))}
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

    if el.t == "Str" and (el.text:match("^[A-Za-z][A-Za-z0-9]*%^%{[^}]*$") or el.text:match("^[A-Za-z][A-Za-z0-9]*_%{[^}]*$")) then
      local parts = {el.text}
      local j = i + 1
      local depth = brace_delta(el.text)
      local closed = depth <= 0

      while j <= #inlines and not closed do
        local ej = inlines[j]
        if ej.t == "Space" or ej.t == "SoftBreak" then
          table.insert(parts, " ")
        elseif ej.t == "Str" then
          table.insert(parts, ej.text)
          depth = depth + brace_delta(ej.text)
          if depth <= 0 then
            closed = true
          end
        elseif ej.t == "Math" and ej.mathtype == "InlineMath" then
          table.insert(parts, ej.text)
          depth = depth + brace_delta(ej.text)
          if depth <= 0 then
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
  return el
end

function Plain(el)
  el.content = convert_split_brace_math_inlines(el.content)
  el.content = convert_binary_operator_sequences(el.content)
  return el
end
