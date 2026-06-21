# Running The Proxy In The Background

These examples assume:

```bash
model-router init --preset lmstudio --yes
```

and a config at:

```text
~/.model-router/routing_proxy.yaml
```

## macOS launchd

Create `~/Library/LaunchAgents/com.hermes-router.proxy.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.hermes-router.proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>model-router-proxy --config ~/.model-router/routing_proxy.yaml</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/hermes-router-proxy.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/hermes-router-proxy.err.log</string>
</dict>
</plist>
```

Commands:

```bash
launchctl load ~/Library/LaunchAgents/com.hermes-router.proxy.plist
launchctl unload ~/Library/LaunchAgents/com.hermes-router.proxy.plist
launchctl list | grep hermes-router
```

## Linux systemd

Create `~/.config/systemd/user/hermes-router-proxy.service`:

```ini
[Unit]
Description=Hermes Router OpenAI-compatible proxy
After=network-online.target

[Service]
ExecStart=model-router-proxy --config %h/.model-router/routing_proxy.yaml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

Commands:

```bash
systemctl --user daemon-reload
systemctl --user enable --now hermes-router-proxy
systemctl --user status hermes-router-proxy
journalctl --user -u hermes-router-proxy -f
```
