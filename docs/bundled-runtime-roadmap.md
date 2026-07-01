# Bundled Runtime Roadmap

This document describes a future optional bundled-runtime path for ModelRouter.
It is a packaging and first-run convenience roadmap, not an implementation plan
for a custom inference engine.

ModelRouter should continue to own control-plane concerns: routing policy,
runtime coordination, local operator UX, route receipts, telemetry, cost/outcome
reporting, safety gates, and OpenAI-compatible proxy behavior. Proven runtimes
should continue to own low-level model execution, kernels, schedulers, GPU
integration, quantization details, and provider-native behavior.

The runtime mode strategy remains:

- `external_managed`: another app or service owns the runtime, such as LM
  Studio, an existing Ollama daemon, LocalAI, vLLM, or a hosted
  OpenAI-compatible gateway.
- `external_cli`: the operator installed a runtime and ModelRouter may start or
  stop a configured command only through explicit adapter actions.
- `bundled_future`: ModelRouter may later package a proven runtime to simplify
  setup, while keeping the same adapter contract and all external runtime paths.

## Product Stance

Bundled runtime support is optional. ModelRouter must keep working as a routing
and control plane above LM Studio, Ollama, llama.cpp, MLX-LM, LocalAI, vLLM,
hosted providers, and generic OpenAI-compatible servers. Users should be able to
choose a ModelRouter-managed convenience path without being locked into it.

Bundling should become interesting only when it removes real setup friction that
adapters, guided connect flows, and external CLI management cannot solve.

## Likely First Bundled Runtime: llama.cpp

`llama.cpp` is the likely first bundled runtime because it is broad, mature, and
runtime-neutral:

- It supports the GGUF model ecosystem that many local users already understand.
- It has a server mode that fits ModelRouter's OpenAI-compatible proxy boundary.
- It can run on CPU and common local acceleration paths, with platform-specific
  builds when needed.
- It is useful on macOS, Windows, and Linux, which makes it a better first
  bundled candidate than a single-platform runtime.
- It already maps to the existing `external_cli` managed process shape:
  configured argv, readiness URL, logs, start, stop, load by process start, and
  unload by process stop.

Bundling llama.cpp would mean packaging or downloading a known upstream build
with provenance. It would not mean forking llama.cpp kernels or building a
ModelRouter inference engine.

## Later Apple Silicon Path: MLX

MLX or MLX-LM may become the Apple Silicon optimized path once the external CLI
adapter and packaging story are stable. It is attractive because Apple Silicon
users are a major local-model audience, and MLX can offer a native path that
feels appropriate on macOS.

MLX should come later because:

- It is platform-specific.
- It carries Python environment and dependency management questions.
- It needs clear model-cache, Hugging Face, and local-path behavior before
  ModelRouter can present it as a smooth bundled path.
- macOS signing, notarization, and update testing should be proven on a simpler
  bundled runtime first.

MLX should complement llama.cpp. It should not replace the external LM Studio,
Ollama, llama.cpp, or hosted-provider paths.

## Binary Packaging Strategy

The first packaging target should be a separately versioned runtime payload, not
a hidden dependency inside routing code.

Recommended shape:

- Package runtime binaries per platform and architecture.
- Store runtime metadata separately from ModelRouter package metadata.
- Verify checksums before use.
- Preserve upstream provenance, version, build flags, source URL, license, and
  update channel.
- Prefer argv-only process launch. Avoid shell strings.
- Keep binary install/update actions behind explicit preview and confirmation.
- Do not place runtime discovery, download, or update checks in
  `route_fast(...)`, `route(...)`, proxy forwarding, or default request
  handling.

Possible artifact layout:

```text
~/.model-router/
  runtimes/
    llama.cpp/
      0.0.0-upstream-build-id/
        bin/
        manifest.json
        checksums.txt
  runtime-cache/
  models/
  logs/
```

Packaged desktop builds may later choose platform-native application support
directories, but CLI installs should keep the user-overridable
`~/.model-router` convention.

## Versioning And Updates

Bundled runtime versions should be separate from ModelRouter versions.

ModelRouter can ship with a default known-good runtime manifest, but operators
should see runtime updates as explicit maintenance:

- `runtime_version`: upstream runtime version or build id.
- `runtime_package_version`: ModelRouter packaging metadata version.
- `runtime_manifest_version`: schema version for bundled runtime metadata.
- `modelrouter_min_version`: minimum ModelRouter version tested with the
  runtime package.
- `source`, `license`, `checksums`, `signed_at`, and `updated_at`.

Updates should be preview-first and rollback-friendly. ModelRouter should keep
at least one previous bundled runtime version until the operator confirms
cleanup.

## Model Storage

Bundled runtime support should not take ownership of every local model file.

Recommended defaults:

- ModelRouter-managed model files: `~/.model-router/models`.
- Bundled runtime binaries: `~/.model-router/runtimes`.
- Runtime logs: `~/.model-router/logs`.
- Imported external runtime models: registry metadata only unless the operator
  explicitly copies or moves files.

Model files owned by LM Studio, Ollama, LocalAI, vLLM, or another runtime should
remain where that runtime expects them. ModelRouter can import metadata and
route to the runtime without relocating the files.

## User Override Path

Operators must be able to bypass bundled binaries:

- Use an externally managed runtime instead.
- Point an `external_cli` backend at a specific binary path.
- Configure a custom model directory.
- Pin a previous bundled runtime version.
- Disable bundled runtime recommendations.
- Remove bundled runtime artifacts without deleting external runtime config.

The override path is important for trust. Advanced users should never feel
trapped behind a ModelRouter-only runtime.

## Sandboxing And Security

Bundled runtime processes should follow the same safety model as configured
managed runtimes, with stricter defaults:

- Bind local development servers to `127.0.0.1` by default.
- Launch with explicit argv arrays, not shell strings.
- Store PID markers only for ModelRouter-owned processes.
- Stop only ModelRouter-owned processes unless the operator explicitly confirms
  otherwise.
- Redact secrets from config, logs, telemetry, and UI.
- Keep runtime logs local and mark that runtime output may include generated
  text.
- Warn before exposing ports on `0.0.0.0`.
- Prefer per-user installs over system-wide writes.
- Avoid privileged service installation in the first bundled version.

Future desktop packaging can explore OS sandboxing, resource limits, and
network controls, but the first bundled path should stay conservative and easy
to inspect.

## Signing, Notarization, And Platform Differences

### macOS

- Ship signed and notarized app artifacts when ModelRouter distributes binaries
  inside a desktop package.
- Use hardened runtime settings appropriate for launching helper processes.
- Separate arm64 and x86_64 binaries unless universal builds are tested.
- Treat MLX as Apple Silicon only until there is tested support elsewhere.
- Document Gatekeeper behavior and how users can remove bundled artifacts.

### Windows

- Prefer signed binaries or a signed installer to reduce SmartScreen friction.
- Avoid assuming WSL, Docker Desktop, Visual Studio Build Tools, CUDA, or Vulkan
  are installed.
- Surface firewall prompts and service installation as explicit operator steps.
- Keep per-user runtime storage unless an installer explicitly chooses a shared
  location.

### Linux

- Expect more variation: glibc versions, distro packages, CUDA/ROCm, container
  runtimes, systemd, and filesystem permissions.
- Prefer tarball or package-manager-neutral artifacts first.
- Offer container or systemd guidance as explicit external setup, not silent
  mutation.
- Preserve checksums and provenance because binary trust varies by distro.

## Telemetry And Privacy Boundaries

Bundled runtime telemetry should stay local-first and privacy-safe:

- Record runtime id, runtime version, endpoint, health, capability state,
  selected backend/model, latency, token usage when upstream reports it, and
  cost/catalog coverage when available.
- Do not record raw prompts, request bodies, secrets, response text, or raw
  runtime logs in telemetry summaries.
- Do not infer success from runtime health, token usage, latency, or cost.
- Keep outcome labels manually supplied by operators/users.
- Do not fetch live pricing or runtime updates during routing or proxy
  forwarding.

Bundled runtime logs are operational diagnostics. The UI should warn that logs
may include upstream runtime output and should avoid copying them into telemetry
by default.

## Rollback And Uninstall

Every bundled runtime update should have a rollback path:

- Stop ModelRouter-owned processes.
- Switch the backend to the previous runtime package.
- Restore the previous runtime manifest.
- Keep model files unless the operator explicitly deletes them.
- Keep local config, telemetry, feedback, and route receipts unless the
  operator intentionally removes them.
- Remove only bundled runtime artifacts during bundled-runtime uninstall.

Uninstall should not remove LM Studio, Ollama, LocalAI, vLLM, external
llama.cpp builds, external MLX environments, hosted provider config, or model
directories owned by other tools.

## Go/No-Go Criteria

Bundling is worth doing when all of these are true:

- External adapter and guided install flows are stable enough to show the real
  remaining setup friction.
- The bundled runtime can use the existing runtime adapter contract without
  creating a second execution path.
- Packaging, checksums, signing, notarization, update, rollback, and uninstall
  are testable for at least one target platform.
- Licensing and upstream provenance are clear.
- Support burden is bounded by a narrow runtime/version matrix.
- The UI can explain bundled, external managed, and external CLI ownership
  without confusing users.
- Hot-path tests prove routing and proxy forwarding do not depend on bundled
  runtime discovery or maintenance.

Bundling should wait if any of these are true:

- The feature would imply ModelRouter owns low-level inference kernels.
- It would weaken LM Studio, Ollama, LocalAI, vLLM, hosted, or generic
  OpenAI-compatible interoperability.
- It would require silent installs, model downloads, system service mutation, or
  hidden route-decision side effects.
- The support matrix is too broad for the team to test.
- External runtimes and guided setup solve the common workflow well enough.

## Suggested Milestones

1. Keep hardening `external_managed` and `external_cli` adapters.
2. Improve settings/CLI runtime status, disabled reasons, model import, and log
   surfaces.
3. Run a packaging spike for a single llama.cpp version on one platform.
4. Add a bundled runtime manifest format and validation tests.
5. Add preview-only install/update/uninstall plans.
6. Add explicit operator-confirmed install/update/rollback.
7. Expand to additional platforms only after the first platform is supportable.
8. Evaluate MLX-LM as an Apple Silicon optimized bundled path.

## Related Docs

- [Runtime strategy](runtime-strategy.md)
- [Runtime adapter contract](runtime-adapter-contract.md)
- [Runtime install flow](runtime-install-flow.md)
- [LM Studio parity roadmap](lm-studio-parity-roadmap.md)
- [Upgrade and uninstall](upgrade-uninstall.md)
- [Release checklist](release-checklist.md)
