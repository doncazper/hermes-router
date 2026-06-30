# Shared admin state contract

This design contract exists so the web UI, TUI, installer, and future admin API are wired to the same backend state/actions instead of each inventing their own data model.

## Required principle

Every rendered field must read from shared admin state or from a documented action result. Every mutating action must require explicit confirmation.

## Required top-level state keys

```text
app
proxy
routes
model_aliases
backends
model_library
installer
telemetry
latest_receipt
logs
actions
```

## Proxy mode fields

```text
proxy.routing_mode = decision | manual | model_map | passthrough
proxy.decision_layer_enabled = true | false
proxy.default_backend
proxy.default_model
proxy.respect_client_model
proxy.unknown_model_behavior = fallback_to_default | reject_404
proxy.safety_gate_mode = decision_only | always_static | off
```

## Required action ids

```text
proxy.start
proxy.stop
proxy.restart
config.save_proxy_patch
config.set_routing_mode
model.scan
model.discover
model.download.plan
model.download.run
model.assign_route
runtime.load_model
runtime.unload_model
doctor.run
benchmark.plan
benchmark.run
telemetry.feedback.write
```

## Required action result shape

```text
ok: boolean
action_id: string
message: string
error: string or null
restart_recommended: boolean or null
refreshed_state: boolean
```

The full field mapping lives in `ui-tui-wireframes.md` because it is easier to review alongside the intended screens.
