# futon-relay-channel — Agency bells pushed into a running Claude session

Built 2026-11-10 by claude-cs (stage 2), for the claude-cs seat in Joe's
code-server. It replaces `tools/futon-relay-vscode` (the relay-folder +
clipboard extension), which is retired. Its source stays in git, but it has
been removed from code-server.

## How it is wired

1. **Seat.** claude-cs is a pull-only Agency seat (`:delivery-mode :inbox`,
   futon3c `scripts/register_claude_cs.clj`). A bell to it is written to
   `~/.claude/agency-inbox/claude-cs/<job-id>.json` and the job stays
   `delivered`. Agency never starts a claude process for it, so nothing forks
   the conversation.
2. **Server.** `server.mjs` is a stdio MCP server that declares
   `experimental["claude/channel"]`. It watches that inbox (fs.watch plus a
   3 s poll) and sends each delivery once per process as
   `notifications/claude/channel`. The `content` is a two-line header (job id,
   sender, the ack command) followed by the prompt, and the `meta` is
   `{job-id, from, ack-url}`. The server never acks.
3. **Ack.** The session answers, then runs
   `python3 ~/code/futon3c/scripts/clink-inbox.py --agent claude-cs ack <job-id> --result-file <answer>`.
   That moves the file to `consumed/`, sets the job to `done`, and returns the
   answer to the sender as the job result. An unacked delivery is pushed again
   the next time the session starts.
4. **Launcher.** `claude-cs.sh` runs `claude` with:
   - `--mcp-config relay-mcp.json`
   - `--dangerously-load-development-channels server:futon-relay-channel`
   - an `--append-system-prompt` that tells the session these channel
     messages are its own bells. Without it, a session treats channel text as
     third-party content and does not act on it.
   - `--resume <id>` from `~/.local/state/futon3c/session-id-claude-cs`
     (`--new` starts a fresh session instead).

   Only this conversation gets the channel. Every other Claude conversation
   starts normally.
5. **code-server.** `claudeCode.useTerminal` is true. Run `claude-cs.sh` in a
   code-server terminal.
6. **Emacs.** In futon4a `lisp/futon4a-subprompt.el`, ㊡ subprompts are now
   sent as a bell from `joe` by default (`futon4a-subprompt-delivery`), so they
   take this same path.

## Constraints found (Claude Code 2.1.322)

- **Interactive only.** Channels register only in the interactive TUI. In the
  SDK / stream-json mode that the VS Code panel uses, a development channel
  server is never registered. In that mode only the `channel_enable` control
  request enables a channel, and only for a marketplace plugin.
  `claudeCode.claudeProcessWrapper` applies only to that panel mode;
  terminal mode runs `claude` from PATH.
- **One confirmation per launch.** Each launch asks "I am using this for local
  development". Press Enter. The answer is not remembered, and no flag skips
  it for a development server.
- **One session at a time.** Run only one channel session for the seat. Two
  running sessions would each push every delivery.
- **Bells queue.** A bell that arrives mid-turn is queued and shown after the
  turn. This was tested on 2026-11-10: bell 4 was pushed at 18:54:50, during
  turn 3, and was delivered after it.

## Files

- **`server.mjs`:** the channel server.
- **`server.test.mjs`:** tests, run with `node --test server.test.mjs`.
- **`relay-mcp.json`:** the MCP config the launcher passes.
- **`claude-cs.sh`:** the launcher.
- **`probe.mjs`, `probe-mcp.json`:** the stage 1 probe.

The server logs to `~/.local/state/futon3c/relay/channel-claude-cs.log`.
