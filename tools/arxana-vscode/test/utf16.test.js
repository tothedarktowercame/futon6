'use strict';
const test = require('node:test');
const assert = require('node:assert');
const {
  utf16RangeToCodepointRange, codepointRangeToUtf16Range, isMidSurrogate, MidSurrogateError,
} = require('../lib/utf16');

// Astral-heavy text: U+1D400 (𝐀), 'objects', U+1F9EA (🧪), 'end', U+104B0 (𐒰)
const astral = '\u{1D400}objects \u{1F9EA}end \u{104B0}';

test('astral UTF-16 range converts to codepoint half-open range', () => {
  // '\u{1D400}' is 2 UTF-16 units, 1 codepoint.
  assert.deepEqual(utf16RangeToCodepointRange(astral, 0, 9), { start: 0, end: 8 }); // "𝐀objects"
  // range covering only the second astral char
  const uStart = astral.indexOf('\u{1F9EA}');
  const uEnd = uStart + 2;
  assert.deepEqual(utf16RangeToCodepointRange(astral, uStart, uEnd), { start: 9, end: 10 });
  // full text
  const cps = [...astral].length;
  assert.deepEqual(utf16RangeToCodepointRange(astral, 0, astral.length), { start: 0, end: cps });
});

test('mid-surrogate UTF-16 coordinate is refused, never rounded', () => {
  const hi = astral.indexOf('\u{1D400}'); // high surrogate at hi, low at hi+1
  assert.equal(isMidSurrogate(astral, hi + 1), true);
  assert.throws(() => utf16RangeToCodepointRange(astral, hi + 1, astral.length), MidSurrogateError);
  const err = (() => { try { utf16RangeToCodepointRange(astral, 0, hi + 1); return null; } catch (e) { return e; } })();
  assert.ok(err instanceof MidSurrogateError);
  assert.equal(err.code, 'arxana.client/mid-surrogate');
  // boundaries at code unit edges are fine
  assert.equal(isMidSurrogate(astral, hi), false);
  assert.equal(isMidSurrogate(astral, hi + 2), false);
  assert.equal(isMidSurrogate(astral, 0), false);
  assert.equal(isMidSurrogate(astral, astral.length), false);
});

test('authority codepoint range converts back to UTF-16 for painting', () => {
  assert.deepEqual(codepointRangeToUtf16Range(astral, 0, 8), { start: 0, end: 9 });
  assert.deepEqual(codepointRangeToUtf16Range(astral, 9, 10), { start: 10, end: 12 });
  assert.deepEqual(codepointRangeToUtf16Range(astral, 0, [...astral].length), { start: 0, end: astral.length });
});

test('real fixture range 792..830 round-trips against intro-generated.tex bytes', () => {
  const fs = require('fs');
  const fx = require('../fixtures/draft8-v1.json');
  const text = fs.readFileSync(fx.artifacts['intro-generated.tex'].path, 'utf8');
  const u = codepointRangeToUtf16Range(text, 792, 830);
  assert.deepEqual(utf16RangeToCodepointRange(text, u.start, u.end), { start: 792, end: 830 });
  // the authority evidence quote is exactly this slice
  assert.equal([...text].slice(792, 830).join(''), 'The object of study is a finite family');
});
