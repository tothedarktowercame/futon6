'use strict';
// Revision identity is authority-owned (R3): the client never mints
// `arxana:*` IDs. It only checks whether the exact bytes of an open
// document digest-match a revision the fixture names, so position
// conversion runs against known revision bytes (R4).

const crypto = require('crypto');

function sha256Hex(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function revisionIdFromSha(shaHex) {
  return `arxana:revision:sha256:${shaHex}`;
}

/** Match document bytes against a fixture map { logicalId -> revisionId }.
 * Returns { kind:'matched', logicalId, revisionId, sha256 } or a typed
 * refusal { kind:'unmatched', sha256 } — never a guessed revision. */
function matchRevision(utf8Bytes, revisionsByLogicalId) {
  const sha = sha256Hex(utf8Bytes);
  for (const [logicalId, revisionId] of Object.entries(revisionsByLogicalId)) {
    if (revisionIdFromSha(sha) === revisionId) {
      return { kind: 'matched', logicalId, revisionId, sha256: sha };
    }
  }
  return { kind: 'unmatched', sha256: sha };
}

module.exports = { sha256Hex, revisionIdFromSha, matchRevision };
