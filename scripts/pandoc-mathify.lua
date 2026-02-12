-- Convert math-like markdown text/code into LaTeX math during pandoc export.

local greek_words = {
  {"Gamma", "\\Gamma"}, {"Delta", "\\Delta"}, {"Theta", "\\Theta"},
  {"Lambda", "\\Lambda"}, {"Pi", "\\Pi"}, {"Sigma", "\\Sigma"},
  {"Phi", "\\Phi"}, {"Omega", "\\Omega"},
  {"alpha", "\\alpha"}, {"beta", "\\beta"}, {"gamma", "\\gamma"},
  {"delta", "\\delta"}, {"theta", "\\theta"}, {"lambda", "\\lambda"},
  {"pi", "\\pi"}, {"sigma", "\\sigma"}, {"tau", "\\tau"},
  {"phi", "\\phi"}, {"chi", "\\chi"}, {"omega", "\\omega"},
  {"mu", "\\mu"}, {"kappa", "\\kappa"}, {"psi", "\\psi"},
}

local greek_token_names = {
  alpha = true, beta = true, gamma = true, delta = true, theta = true,
  lambda = true, pi = true, sigma = true, tau = true, phi = true,
  chi = true, omega = true, mu = true, kappa = true, psi = true,
  Alpha = true, Beta = true, Gamma = true, Delta = true, Theta = true,
  Lambda = true, Pi = true, Sigma = true, Phi = true, Omega = true,
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
  return s:gsub("%f[%a]" .. w .. "%f[%A]", repl)
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
  s = s:gsub("([%w%)%}])%s*%*%s*([%w%(%{\\])", "%1 \\ast %2")

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

  -- Dimension separators: n x n -> n \times n
  local prev = nil
  while prev ~= s do
    prev = s
    s = s:gsub("([A-Za-z0-9{}])%s+[xX]%s+([A-Za-z0-9{])", "%1 \\times %2")
  end

  s = replace_word(s, "in", "\\in")
  s = replace_word(s, "notin", "\\notin")
  s = replace_word(s, "rtimes", "\\rtimes")

  for _, pair in ipairs(greek_words) do
    s = replace_word(s, pair[1], pair[2])
  end

  s = replace_word(s, "Q", "\\mathbb{Q}")
  s = replace_word(s, "R", "\\mathbb{R}")
  s = replace_word(s, "Z", "\\mathbb{Z}")
  s = replace_word(s, "C", "\\mathbb{C}")

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

local function convert_simple_str_token_to_inlines(s)
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
  return el
end

function Plain(el)
  el.content = convert_split_brace_math_inlines(el.content)
  return el
end
