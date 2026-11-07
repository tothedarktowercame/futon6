# Arxana code-server demo — start/stop/rollback

A projection-only VS Code/code-server extension (T-codeserver-arxana-demo,
M-jvm-arxana). It consumes `arxana.v1` (`capabilities/handshake`,
`scope/resolve`, `correspondence/query`) from the isolated loopback JVM
surface (`futon3c.arxana.demo`) over the draft8 LaTeX/Oxide fixture. No
futon1/futon1b calls, no `arxana:*` id minting, no first-candidate selection,
no store writes, no firewall/nginx/live-store changes.

## Files

- `extension.js` — commands + webview panel + decorations (IDs only)
- `lib/client.js` — arxana.v1 HTTP client with typed client refusals
- `lib/utf16.js` — UTF-16 ↔ codepoint half-open ranges, mid-surrogate refusal
- `lib/digest.js` — digest-match of open document bytes to fixture revisions
- `lib/view.js` — pure view models + webview HTML (all candidates, no auto-select)
- `fixtures/draft8-v1.json` — fixture stand-in config (see its `provenance`)
- `scripts/live-demo.js` — live acceptance run, writes `demo-captures/*.html`
- `test/*.test.js` — 18 node tests (`node --test test/*.test.js`)

## Start

1. Fresh-census and start the isolated JVM service (loopback only):

   ```sh
   bash /home/joe/code/futon4a/scripts/check-demo-ports.sh   # must say AVAILABLE
   cd /home/joe/code/futon3c
   nohup clojure -M -m futon3c.arxana.demo 7084 > /tmp/arxana-codeserver-demo-jvm.log 2>&1 &
   echo $! > /tmp/arxana-codeserver-demo-jvm.pid
   curl -sf http://127.0.0.1:7084/arxana/v1/capabilities   # wait until it answers
   ```

2. Run the headless acceptance flow through the extension's own code:

   ```sh
   cd /home/joe/code/futon6/tools/arxana-vscode
   node scripts/live-demo.js        # prints PASS/FAIL, writes demo-captures/
   node --test test/*.test.js       # 18 tests
   ```

3. Browser demo (does NOT touch the shared code-server on :8090):

   ```sh
   ss -ltn 'sport = :8097' | tail -n +2 | grep -q . && echo REFUSED || echo ok
   mkdir -p /tmp/arxana-cs-demo/ext/arxana-codeserver-demo-0.1.0 /tmp/arxana-cs-demo/user
   cp -r {package.json,extension.js,lib,fixtures} /tmp/arxana-cs-demo/ext/arxana-codeserver-demo-0.1.0/  # from this dir
   code-server --extensions-dir /tmp/arxana-cs-demo/ext --user-data-dir /tmp/arxana-cs-demo/user \
     --bind-addr 127.0.0.1:8097 --auth none
   # open http://127.0.0.1:8097, open intro-generated.tex from
   # futon5 .../wysiwyg/sandbox/, then run the commands:
   #   Arxana: Show capabilities (handshake)
   #   Arxana: Resolve scope for active fixture document
   #   Arxana: Query correspondence
   #   Arxana: Open rendered target (navigate correspondence)
   ```

   To install into the shared code-server instead: copy the same directory
   under `~/.local/share/code-server/extensions/` and restart code-server
   (its nREPL/Marimo extensions are unaffected; not done during this demo).

## Stop / rollback

```sh
kill $(cat /tmp/arxana-codeserver-demo-jvm.pid)            # JVM service
pkill -f 'bind-addr 127.0.0.1:8097'                        # private code-server
rm -rf /tmp/arxana-cs-demo                                 # private instance data
bash /home/joe/code/futon4a/scripts/check-demo-ports.sh    # all AVAILABLE again
```

The extension is additive: deleting `tools/arxana-vscode` and the copied
extension directory rolls everything back. No store, fixture or service file
was modified.

## Notes

- `draft8.html` must be read from `arxana-fixture/draft8.html` (the gated
  copy the authority ingests); `page/draft8.html` has drifted and is not the
  revision.
- Extension host difference (code-server vs desktop): none for authority
  behavior — the extension uses only `fetch` against loopback and the webview
  API; identical code runs on desktop VS Code.
