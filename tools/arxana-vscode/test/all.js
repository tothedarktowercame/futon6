'use strict';
// Node 24 no longer discovers a directory passed to `node --test test/`.
// Keep the declared gate meaningful by loading every suite explicitly.
require('./client.test');
require('./digest.test');
require('./f2.test');
require('./live.test');
require('./navigation.test');
require('./utf16.test');
require('./view.test');
