# Running The Proxy In The Background

These examples assume:

```bash
model-router init --preset lmstudio --yes
```

and a config at:

```text
~/.model-router/routing_proxy.yaml
```

Before installing a background service, start the upstream server and check the
config:

```bash
model-router doctor --config ~/.model-router/routing_proxy.yaml
```

For LM Studio, start the local server on `http://127.0.0.1:1234/v1` and edit
the generated backend model ids to match LM Studio. For Ollama, use:

```bash
ollama pull qwen3:0.6b
ollama pull qwen3:4b
ollama pull qwen3:14b
ollama pull qwen2.5-coder:7b
model-router init --preset ollama --yes
```

Point the agent at:

```text
Base URL: http://127.0.0.1:8082/v1
Model: model-router
API key: leave blank unless proxy auth is configured
```

## macOS launchd

Resolve the installed proxy command and write a user LaunchAgent:

```bash
PROXY_BIN="$(command -v model-router-proxy)"
test -n "$PROXY_BIN"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.model-router/logs"
cat > "$HOME/Library/LaunchAgents/com.modelrouter.proxy.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.modelrouter.proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PROXY_BIN}</string>
    <string>--config</string>
    <string>${HOME}/.model-router/routing_proxy.yaml</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${HOME}/.model-router/logs/proxy.out.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/.model-router/logs/proxy.err.log</string>
</dict>
</plist>
EOF
```

The generated plist should look like this, with your absolute
`model-router-proxy` path in the first `ProgramArguments` entry:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.modelrouter.proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/model-router-proxy</string>
    <string>--config</string>
    <string>/Users/you/.model-router/routing_proxy.yaml</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/you/.model-router/logs/proxy.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/you/.model-router/logs/proxy.err.log</string>
</dict>
</plist>
```

Commands:

```bash
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.modelrouter.proxy.plist"
launchctl kickstart -k "gui/$(id -u)/com.modelrouter.proxy"
launchctl print "gui/$(id -u)/com.modelrouter.proxy"
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.modelrouter.proxy.plist"
tail -f "$HOME/.model-router/logs/proxy.err.log"
```

## Linux systemd

Resolve the installed proxy command and write a user service:

```bash
PROXY_BIN="$(command -v model-router-proxy)"
test -n "$PROXY_BIN"
mkdir -p "$HOME/.config/systemd/user" "$HOME/.model-router/logs"
cat > "$HOME/.config/systemd/user/model-router-proxy.service" <<EOF
[Unit]
Description=ModelRouter OpenAI-compatible proxy
After=network-online.target

[Service]
ExecStart=${PROXY_BIN} --config %h/.model-router/routing_proxy.yaml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
```

The generated unit should look like this, with your absolute
`model-router-proxy` path in `ExecStart`:

```ini
[Unit]
Description=ModelRouter OpenAI-compatible proxy
After=network-online.target

[Service]
ExecStart=/home/you/.local/bin/model-router-proxy --config %h/.model-router/routing_proxy.yaml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

Commands:

```bash
systemctl --user daemon-reload
systemctl --user enable --now model-router-proxy
systemctl --user status model-router-proxy
journalctl --user -u model-router-proxy -f
```

If the service should keep running after logout, enable lingering once:

```bash
loginctl enable-linger "$USER"
```
