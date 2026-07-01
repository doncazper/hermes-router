# Versioned pricing catalog

ModelRouter estimates token cost only in reporting paths. Routing decisions still
use configured route/backend metadata such as `cost_tier`, provider policy, and
latency tier. `route_fast(...)`, `route(...)`, proxy forwarding, and verifier
flows must never fetch live prices, scrape provider pages, or call pricing APIs.

## Boundaries

- **Cost tiers** are routing policy metadata. They are stable enough for route
  selection and do not claim exact spend.
- **Actual usage** is upstream metadata the proxy records when a provider already
  returns it: prompt tokens, completion tokens, total tokens, cached input tokens,
  and upstream model id.
- **Pricing catalog metadata** is local, versioned, packaged with ModelRouter,
  and optionally overridden by the operator.
- **Estimated cost** is reporting metadata derived from usage plus catalog
  entries. It is not a success claim and it must not influence routing.

Missing usage or a missing catalog match produces a clear unavailable status
instead of an invented estimate.

## Catalog schema

The packaged catalog lives in package data and uses YAML:

```yaml
catalog_version: 1
updated_at: "2026-06-30T00:00:00Z"
notes:
  - Packaged catalog uses local defaults and non-authoritative examples.
entries:
  - provider: openai
    model: example-hosted-model
    input_per_1m: 1.0
    output_per_1m: 3.0
    cached_input_per_1m: 0.25
    currency: USD
    effective_date: "2026-06-30"
    source: example-placeholder
    notes: Replace with operator-confirmed provider pricing before reporting spend.
```

Entry fields:

| Field | Meaning |
| --- | --- |
| `provider` | Provider or runtime family, such as `openai`, `anthropic`, `local`, or `example`. |
| `model` | Exact model id used for lookup. |
| `input_per_1m` | Price for uncached input tokens per 1 million tokens. |
| `output_per_1m` | Price for completion/output tokens per 1 million tokens. |
| `cached_input_per_1m` | Price for cached input tokens per 1 million tokens. |
| `currency` | ISO-like currency label, usually `USD`. |
| `effective_date` | Date the operator/source says the price became effective. |
| `source` | Local source label or URL/reference string. |
| `notes` | Human-readable caveat. |

The top-level `catalog_version` and `updated_at` identify the catalog file. The
version is not provider pricing truth; it is ModelRouter's local metadata
version.

## Overrides

Operators may create a local override catalog, normally:

```text
~/.model-router/pricing_catalog.yaml
```

Overrides are explicit local files. ModelRouter does not refresh them while
routing or forwarding requests. A later maintenance command can preview and
apply packaged defaults, but provider price updates should remain an operator
action.

Inspect the active catalog and the packaged-vs-override diff without mutating
anything:

```bash
model-router pricing status --override ~/.model-router/pricing_catalog.yaml
model-router pricing diff --override ~/.model-router/pricing_catalog.yaml
```

`status` reports the active catalog version/source, override path, validation
state, validation errors, and warnings for placeholder/example pricing entries.
`diff` previews what the packaged catalog would write. Both commands read local
package data and the local override only; they perform no network checks and do
not create or update files.

Create the default override file from packaged metadata only after explicit
confirmation:

```bash
model-router pricing apply --override ~/.model-router/pricing_catalog.yaml
```

Use `--yes` only for deliberate non-interactive maintenance:

```bash
model-router pricing apply --override ~/.model-router/pricing_catalog.yaml --yes
```

`apply` writes packaged metadata to the local override path, backing up an
existing override first. It never fetches provider pricing, and it is not called
from `route_fast(...)`, `route(...)`, proxy forwarding, or default routing.

Then edit `~/.model-router/pricing_catalog.yaml` with prices you have verified
against your provider contract, invoice, or official pricing page. Do not rely
on the packaged example hosted entry for spend reporting; it exists only to show
the schema and exercise estimates.

Example operator override entry:

```yaml
catalog_version: 2
updated_at: "2026-06-30T00:00:00Z"
notes:
  - Operator-verified from provider pricing terms on 2026-06-30.
entries:
  - provider: your-provider
    model: your-model-id
    input_per_1m: 1.25
    output_per_1m: 5.00
    cached_input_per_1m: 0.25
    currency: USD
    effective_date: "2026-06-30"
    source: operator-verified-provider-pricing
    notes: Replace this example with pricing verified for your account/region/plan.
```

If telemetry rows do not include provider identity, ModelRouter can still match
by model id only when exactly one catalog entry has that model id. If multiple
providers share the same model id, add provider metadata to the telemetry source
or use distinct model ids in your override.

Override semantics:

- Packaged entries load first.
- Local override entries replace packaged entries with the same
  `(provider, model)` pair.
- Local override entries may add new models.
- Invalid override files should fail validation in maintenance commands and
  produce no invented estimates.

## Lookup semantics

Telemetry uses the best model identity available:

1. `upstream_model`
2. `backend_model`
3. `selected_model`

If provider identity is available, lookup uses `(provider, model)`. If provider
identity is absent, lookup may use model-only matching only when exactly one
catalog entry has that model id. Multiple providers with the same model id are
reported as ambiguous. Missing usage, missing model ids, missing price entries,
and ambiguous model-only matches are all reporting statuses, not errors in the
routing path.

Cached input tokens are treated as part of prompt/input tokens. Estimated input
cost uses:

```text
uncached_input = max(prompt_tokens - cached_input_tokens, 0)
input_cost = uncached_input * input_per_1m / 1_000_000
cached_input_cost = cached_input_tokens * cached_input_per_1m / 1_000_000
output_cost = completion_tokens * output_per_1m / 1_000_000
```

## Reporting fields

Telemetry summaries and review rows may include:

- `estimated_input_cost`
- `estimated_output_cost`
- `estimated_cached_input_cost`
- `estimated_total_cost`
- `estimated_cost_currency`
- `pricing_catalog_version`
- `pricing_catalog_source`
- `pricing_source`
- `pricing_effective_date`
- `pricing_provider`
- `pricing_model`
- `pricing_match_status`

These fields are JSON-safe and privacy-safe. They must not expose prompts,
request bodies, secrets, or response text.
