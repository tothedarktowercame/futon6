'use strict';
// UTF-16 → Unicode-codepoint conversion at the client edge (D5-DR1, R4).
//
// Authority locations are Unicode-codepoint, zero-based, half-open. VS Code
// positions are UTF-16 code units. Conversion is legal ONLY against the
// exact revision bytes whose digest matched the named artifact revision;
// the caller supplies those bytes. A position inside a surrogate pair is
// rejected — it does not name a codepoint.

class MidSurrogateError extends Error {
  constructor(utf16Offset) {
    super(`UTF-16 offset ${utf16Offset} falls inside a surrogate pair; it names no Unicode codepoint`);
    this.code = 'arxana.client/mid-surrogate';
    this.utf16Offset = utf16Offset;
  }
}

/** True when the code unit at utf16Offset is the low half of a surrogate
 * pair, i.e. the offset sits between a high surrogate and its low half. */
function isMidSurrogate(text, utf16Offset) {
  if (typeof text !== 'string') throw new TypeError('text must be a string');
  if (!Number.isInteger(utf16Offset) || utf16Offset < 0 || utf16Offset > text.length) {
    throw new TypeError(`utf16Offset out of range: ${utf16Offset}`);
  }
  const prev = text.charCodeAt(utf16Offset - 1);
  return utf16Offset > 0
    && prev >= 0xd800 && prev <= 0xdbff   // previous unit is a high surrogate
    && utf16Offset < text.length;         // and this offset is not the end
}

/** Convert one UTF-16 offset to a codepoint offset against TEXT. */
function utf16ToCodepoint(text, utf16Offset) {
  if (isMidSurrogate(text, utf16Offset)) throw new MidSurrogateError(utf16Offset);
  let codepoints = 0;
  for (let i = 0; i < utf16Offset; i++) {
    const unit = text.charCodeAt(i);
    if (unit >= 0xdc00 && unit <= 0xdfff) continue; // low surrogate: same codepoint as its high half
    codepoints++;
  }
  return codepoints;
}

/** Convert a UTF-16 half-open range to a codepoint half-open range.
 * Rejects a boundary inside a surrogate pair. */
function utf16RangeToCodepointRange(text, start, end) {
  if (!(start <= end)) throw new TypeError(`inverted range ${start}..${end}`);
  return { start: utf16ToCodepoint(text, start), end: utf16ToCodepoint(text, end) };
}

/** Convert a codepoint half-open range back to UTF-16 offsets (for painting
 * authority-returned candidate ranges in the editor). */
function codepointRangeToUtf16Range(text, start, end) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) {
    throw new TypeError(`invalid codepoint range ${start}..${end}`);
  }
  // boundary[k] = UTF-16 offset where codepoint k begins (only computed at
  // codepoint edges, never inside a surrogate pair).
  const boundary = [0];
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) i++; // skip the low half
    boundary.push(i + 1);
  }
  const total = boundary.length - 1;
  if (start > total || end > total) {
    throw new TypeError(`codepoint range ${start}..${end} exceeds text (${total} codepoints)`);
  }
  return { start: boundary[start], end: boundary[end] };
}

module.exports = { isMidSurrogate, utf16ToCodepoint, utf16RangeToCodepointRange, codepointRangeToUtf16Range, MidSurrogateError };
