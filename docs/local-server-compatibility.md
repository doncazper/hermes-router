# Local Server Compatibility Matrix

ModelRouter exposes one local OpenAI-compatible `/v1` endpoint for clients that
need routing policy, provider/backend selection, receipts, safety gates, and
privacy-safe telemetry above local or hosted model servers. This compatibility
layer is a proxy/control-plane boundary. It does not become an inference engine,
does not run tools, does not translate every provider-native API, and does not
route based on live pricing, runtime discovery, or telemetry side effects.

Status meanings:

- **Supported**: ModelRouter implements the local endpoint shape and forwards to
  a configured backend when policy selects one.
- **Partial**: ModelRouter preserves the request/stream shape, but actual
  behavior depends on the selected backend and its current server version.
- **Unsupported**: ModelRouter returns a shaped error or capability reason and
  does not fake the feature.
- **Deferred**: Product direction exists, but the endpoint is not implemented.

## ModelRouter Endpoint Contract

| Surface | ModelRouter status | Behavior |
| --- | --- | --- |
| `/v1/models` | Supported | Returns configured ModelRouter aliases and backend model ids. It includes legacy boolean `capabilities` plus `capability_details` with status and reasons for partial, unsupported, or deferred capabilities. It does not call upstreams. |
| `/v1/chat/completions` | Supported | Routes from recent user message text, overwrites the outgoing backend `model`, preserves the request body, and forwards to the selected OpenAI-compatible backend. |
| Streaming chat | Partial | Preserves upstream server-sent-event bytes and does not buffer streams for verification or usage discovery. Requires the selected backend to support streaming. |
| Tool calls | Partial | Preserves `tools`, `tool_choice`, `parallel_tool_calls`, and legacy `functions` by default. Per-backend `strip_tools: true` removes tool fields before forwarding. ModelRouter does not execute tools. |
| Structured output | Partial | Preserves structured-output fields such as `response_format`. ModelRouter does not validate backend schema support or translate unsupported structured-output dialects. |
| `/v1/responses` | Partial | Routes from `input`, preserves common Responses API fields, and forwards only when the selected backend supports `/v1/responses`. |
| `/v1/embeddings` | Partial | Routes from bounded string input text, preserves the request body, and forwards non-streaming embedding requests when the selected backend supports `/v1/embeddings`. |
| `/v1/completions` | Partial | Routes from `prompt`, preserves the request body, and forwards to a compatible upstream. |
| `/v1/messages` | Deferred | Returns a shaped `unsupported_endpoint` response. Anthropic Messages compatibility needs explicit capability plumbing before it is supported. |
| Unknown `/v1/*` endpoints | Unsupported | Return a shaped `unsupported_endpoint` response and never call upstreams. |
| `/health` | Supported | Reports proxy configuration, backend reachability, runtime state, routing mode, policy, verifier mode, and observability settings without exposing secrets. |

## Runtime And Backend Notes

This matrix describes ModelRouter behavior when a backend is configured for the
named runtime family. Exact upstream support can vary by runtime version, server
flags, loaded model, and provider gateway.

| Backend family | `/v1/models` | Chat completions | Streaming | Tools | Structured output | Responses | Embeddings | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LM Studio | Supported by ModelRouter listing; upstream used by `doctor`/health when configured | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Best used when LM Studio already owns local model downloads and loading. Configure exact model ids advertised by LM Studio. |
| Ollama OpenAI-compatible server | Supported by ModelRouter listing; upstream used by `doctor`/health when configured | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough if upstream supports it | Partial passthrough if upstream supports it | Presets target `http://127.0.0.1:11434/v1`; model pulls remain an Ollama/operator action. |
| llama.cpp / `llama-server` | Supported by ModelRouter listing; upstream used by readiness checks | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough if upstream supports it | Partial passthrough if upstream supports it | Managed runtime support starts/stops configured argv commands only. It does not infer model paths or download GGUF files. |
| MLX-LM managed server | Supported for listing/readiness | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Unsupported for managed `mlx-lm` backends | Unsupported for managed `mlx-lm` backends | Current managed MLX-LM support is chat/models-first. Use another upstream when clients need Responses or embeddings. |
| LocalAI | Supported by ModelRouter listing; upstream used by `doctor`/health when configured | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough if upstream supports it | Partial passthrough if upstream supports it | Treated as a generic OpenAI-compatible backend unless a future native adapter adds explicit capabilities. |
| vLLM OpenAI-compatible server | Supported by ModelRouter listing; upstream used by `doctor`/health when configured | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough if upstream supports it | Partial passthrough if upstream supports it | Treated as a generic OpenAI-compatible backend unless a future native adapter adds explicit capabilities. |
| Hosted OpenAI-compatible backends | Supported by ModelRouter listing; upstream used by `doctor`/health when configured | Supported passthrough | Partial passthrough | Partial passthrough | Partial passthrough | Partial passthrough if upstream supports it | Partial passthrough if upstream supports it | API keys are configured per backend and forwarded as `Authorization: Bearer ...`; secrets are never included in model lists, health output, or telemetry logs. |

## Auth And Error Shape

If `proxy.api_key` or `proxy.api_key_env` is configured, routed `/v1/*`
endpoints and `/v1/models` require `Authorization: Bearer <token>`. Authentication
failures return:

```json
{
  "error": {
    "message": "Missing or invalid bearer token.",
    "type": "authentication_error"
  }
}
```

Unsupported or deferred compatibility endpoints return OpenAI-style error
objects with a `modelrouter` metadata block describing the endpoint and known
supported/planned endpoints. Backend-specific capability gaps, such as managed
MLX-LM not supporting `/v1/responses`, are exposed through `/v1/models`
`capability_details` and through shaped proxy errors when a request targets that
unsupported path.

## Non-Goals

- No live pricing fetch in `route_fast(...)`, `route(...)`, proxy forwarding, or
  endpoint compatibility checks.
- No success inference from latency, token usage, cost, or verifier output.
- No hidden planner/worker orchestration and no tool execution.
- No custom inference engine from scratch when proven runtimes already exist.
- No claim that every OpenAI-compatible backend supports every endpoint or
  dialect listed here.
