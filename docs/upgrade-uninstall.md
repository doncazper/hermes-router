# Upgrade And Uninstall

Upgrade the current PyPI package with the proxy extra:

```bash
python -m pip install --upgrade "hermes-router[proxy]"
```

Install the optional TUI surface only if you want it:

```bash
python -m pip install --upgrade "hermes-router[proxy,tui]"
```

## After Upgrade

Validate config before restarting a proxy that agents use:

```bash
model-router validate-proxy-config --config ~/.model-router/routing_proxy.yaml
model-router doctor --config ~/.model-router/routing_proxy.yaml
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml
```

Run live dogfood only when the local proxy and intended backend are running:

```bash
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml --execute
```

## Config Migration Notes

Recent config changes are additive. Existing decision-mode proxy configs remain
the default and should continue to route without opting into new surfaces.

Newer configs may include:

- `proxy.routing_mode`, default `decision`.
- `proxy.default_backend` and `proxy.default_model` for manual/basic mode.
- `proxy.respect_client_model`, `proxy.unknown_model_behavior`, and
  `proxy.safety_gate_mode`.
- Capability hints exposed through `/v1/models`.
- Optional runtime adapter, model library, benchmark, TUI, and settings state.

ModelRouter should not silently download models, enable hosted providers,
overwrite config, mutate routing policy, or switch routing modes. Use
`model-router settings` or explicit CLI actions to save, apply, and restart.

Before changing config, keep a copy:

```bash
cp ~/.model-router/routing_proxy.yaml ~/.model-router/routing_proxy.yaml.bak
```

If a catalog update is available, inspect it before applying:

```bash
model-router catalog diff --config ~/.model-router/model_router.yaml
model-router catalog apply --config ~/.model-router/model_router.yaml --yes
```

## Rollback

Reinstall the previous published package version if a new beta or experimental
surface blocks your workflow:

```bash
python -m pip install "hermes-router[proxy]==<previous-version>"
```

Then restore the config backup if needed:

```bash
cp ~/.model-router/routing_proxy.yaml.bak ~/.model-router/routing_proxy.yaml
```

Decision-mode proxy routing is the stable default. Experimental failures in the
TUI or other admin surfaces should not require config rollback unless you saved
new settings explicitly.

## Uninstall

Stop any running proxy or settings process first. Then uninstall the package:

```bash
python -m pip uninstall hermes-router
```

Local config, telemetry, feedback, benchmarks, and runtime logs are left in
`~/.model-router` so you do not lose routing history by accident. Remove them
only if you intentionally want a clean local state:

```bash
rm -rf ~/.model-router
```
