# MARK2 systemd units

Install the MARK2 watchdog as a user timer on the Chicago staging host to
detect a stalled lane and send an email alert.

```bash
mkdir -p ~/.config/systemd/user
cp systemd/user-mark2-watchdog.service ~/.config/systemd/user/mark2-watchdog.service
cp systemd/user-mark2-watchdog.timer ~/.config/systemd/user/mark2-watchdog.timer

cat > ~/.config/mark2-watchdog.env <<'EOF'
MARK2_ALERT_TO=joe
MARK2_ALERT_REPEAT_HOURS=12
EOF

systemctl --user daemon-reload
systemctl --user enable --now mark2-watchdog.timer
```

For this to remain active without an interactive login, `joe` must have
linger enabled:

```bash
sudo loginctl enable-linger joe
```

If `loginctl show-user joe -p Linger` reports `Linger=no` and you do not want
to change that, use a root-owned system timer instead. A user timer without
linger is not a reliable watchdog surface on a headless host.

The watchdog does **not** check for an always-running `mark2` process. That
would be the wrong invariant because the coordinator is expected to be idle
whenever the inbox already meets the ready target or the manifest is drained.
Instead it alerts when all of the following are true:

- the manifest still has pending work
- the inbox has fewer ready batches than `MARK2_READY_TARGET`
- there is no active `mark2 fill` or `mark2 build` process

It also records a small state file at `~/mark2/watchdog-state.json` so the same
alert is not re-emailed every timer tick.

Useful checks:

```bash
systemctl --user status mark2-watchdog.timer
journalctl --user -u mark2-watchdog.service -b --no-pager | tail -n 80
python3 ~/mark2/mark2 watchdog
```
