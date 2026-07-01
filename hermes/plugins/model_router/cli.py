"""Command line interface for ModelRouter decisions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from hermes.plugins.model_router.admin.actions import AdminActionError, run_admin_action
from hermes.plugins.model_router.availability import validate_router_availability
from hermes.plugins.model_router.catalog_update import (
    apply_catalog_update,
    catalog_diff,
    catalog_status,
)
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.dispatch import build_dispatch_plan, dispatch_plan_to_json
from hermes.plugins.model_router.eval_runner import (
    DEFAULT_EVAL_RESULTS_PATH,
    eval_evidence_for_model,
    eval_fixture_summaries,
    eval_report,
    execute_eval_run,
)
from hermes.plugins.model_router.evals import EvalFixtureError
from hermes.plugins.model_router.installer import (
    build_install_plan,
    options_from_namespace,
)
from hermes.plugins.model_router.models import ModelEngine, RouterConfig
from hermes.plugins.model_router.model_benchmark import (
    DEFAULT_BENCHMARK_PATH,
    benchmark_summary,
    execute_benchmark_plan,
    load_benchmark_results,
    plan_backend_benchmarks,
)
from hermes.plugins.model_router.policy import ModelRouter, route_prompt
from hermes.plugins.model_router.pricing_catalog import (
    DEFAULT_PRICING_CATALOG_NAME,
    apply_pricing_catalog,
    default_pricing_override_path,
    pricing_diff,
    pricing_status,
)
from hermes.plugins.model_router.profiles import ROUTING_PROFILE_VALUES
from hermes.plugins.model_router.product import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PROXY_PORT,
    PRESETS,
    doctor_proxy_config,
    initialize_product_config,
    validate_proxy_config,
)
from hermes.plugins.model_router.proxy_dogfood import run_proxy_dogfood
from hermes.plugins.model_router.proxy_config import ProxyConfigError
from hermes.plugins.model_router.receipts import decision_to_receipt, receipt_to_json
from hermes.plugins.model_router.routing_log import (
    DEFAULT_LOG_PATH,
    DEFAULT_FEEDBACK_PATH,
    OUTCOME_LABELS,
    RoutingLogWriter,
    build_feedback,
)
from hermes.plugins.model_router.runtime_install import (
    RuntimeConnectRequest,
    RuntimeInstallError,
    build_runtime_connect_plan,
    runtime_doctor_report,
    runtime_status_report,
)
from hermes.plugins.model_router.setup_assistant import (
    DiscoveredModel,
    DownloadPlan,
    DownloadSuggestion,
    SetupRecommendation,
    engine_override_for_download,
    engine_override_for_local_model,
    execute_download_plan,
    execute_prereq_install_plan,
    plan_model_downloads,
    plan_prereq_installs,
    recommend_setup,
    scan_local_environment,
    write_config_from_recommendation,
    write_recommended_config,
)
from hermes.plugins.model_router.telemetry import (
    feedback_summary,
    pricing_override_skeleton_from_gaps,
    replay_events,
    review_queue,
)
from hermes.plugins.model_router.workflow_benchmark import (
    run_workflow_benchmarks,
    workflow_case_names,
    workflow_cases_by_name,
)

ROUTE_WIZARD_LABELS = (
    ("simple", "Simple rewrites/extraction"),
    ("balanced", "General chat and summarization"),
    ("reasoning", "Deep reasoning and planning"),
    ("coding", "Coding and repository work"),
    ("research", "Web research and RAG"),
    ("vision", "Vision, screenshots, OCR"),
    ("image_generation", "Image generation"),
)

ROUTE_LOCAL_ENGINES = {
    "simple": "fast_local",
    "balanced": "balanced_local",
    "reasoning": "reasoning_local",
    "coding": "code_agent",
    "research": "web_research",
    "vision": "multimodal_vision",
    "image_generation": "image_generation",
}

HF_CLI_INSTALL_COMMAND = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--upgrade",
    "huggingface_hub[cli]",
)


@dataclass(frozen=True)
class WizardChoice:
    label: str
    engine: str
    engine_override: dict[str, Any] | None = None
    download_suggestion: DownloadSuggestion | None = None


@dataclass(frozen=True)
class WizardSelections:
    routing_targets: dict[str, str]
    engine_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    download_suggestions: tuple[DownloadSuggestion, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-router",
        description="Decide which engine category should handle a prompt.",
    )
    configure_parser(parser)
    return parser


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init",
        help="Create ready-to-run local proxy configs",
    )
    init.add_argument(
        "--preset",
        choices=PRESETS,
        default=None,
        help="Provider template to use",
    )
    init.add_argument(
        "--auto",
        action="store_true",
        help="Choose a preset from local Ollama/LM Studio signals",
    )
    init.add_argument(
        "--auto-models",
        action="store_true",
        help="Scan local models and fill managed mlx-lm/llamacpp backend models",
    )
    init.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=None,
        help="Additional or replacement local model directory to scan",
    )
    init.add_argument(
        "--profile",
        default="balanced",
        choices=("balanced", "lightweight", "quality"),
        help="Recommendation profile for model suggestions",
    )
    init.add_argument(
        "--config-dir",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR),
        help="Directory for generated configs",
    )
    init.add_argument(
        "--proxy-port",
        type=int,
        default=DEFAULT_PROXY_PORT,
        help="Local proxy port",
    )
    init.add_argument(
        "--yes",
        action="store_true",
        help="Use defaults without interactive prompts",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated files",
    )
    init.add_argument("--json", action="store_true", help="Emit JSON output")
    init.set_defaults(func=_cmd_init)

    install = subparsers.add_parser(
        "install",
        help="Plan deterministic first-run onboarding",
    )
    install.add_argument(
        "--config-dir",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR),
        help="Directory for ModelRouter configs and local telemetry",
    )
    install.add_argument(
        "--quick",
        action="store_true",
        help="Prefer the shortest safe onboarding path",
    )
    install.add_argument(
        "--auto",
        action="store_true",
        help="Choose a preset from local runtime signals",
    )
    install.add_argument(
        "--local-only",
        action="store_true",
        help="Keep recommendations local-only; do not enable hosted providers",
    )
    install.add_argument("--lmstudio", action="store_true", help="Prefer LM Studio")
    install.add_argument("--ollama", action="store_true", help="Prefer Ollama")
    install.add_argument("--mlx-lm", action="store_true", help="Prefer MLX-LM")
    install.add_argument("--llamacpp", action="store_true", help="Prefer llama.cpp")
    install.add_argument(
        "--developer",
        action="store_true",
        help="Include developer setup/check commands",
    )
    install.add_argument("--json", action="store_true", help="Emit JSON output")
    install.add_argument(
        "--yes",
        action="store_true",
        help="Record confirmation intent; M2 still prints a plan only",
    )
    install.set_defaults(func=_cmd_install)

    decide = subparsers.add_parser("decide", help="Score and route a prompt")
    decide.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON routing receipt",
    )
    decide.add_argument(
        "--explain",
        action="store_true",
        help="Emit a concise human-readable receipt explanation",
    )
    decide.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
    )
    decide.add_argument(
        "--profile",
        dest="routing_profile",
        default="balanced",
        choices=ROUTING_PROFILE_VALUES,
        help="Routing profile to compile into routing constraints",
    )
    decide.add_argument(
        "--force-engine",
        default=None,
        help="Prefer a specific engine by name; high-risk actions still confirm",
    )
    decide.add_argument(
        "--attachment",
        action="append",
        choices=("image", "pdf", "audio", "code"),
        default=None,
        help="Declare an attachment modality for routing constraints",
    )
    decide.add_argument(
        "--max-cost-tier",
        default=None,
        help="Reject engines above this cost tier",
    )
    decide.add_argument(
        "--max-latency-tier",
        default=None,
        help="Reject engines above this latency tier",
    )
    decide.add_argument(
        "--latency-sensitive",
        action="store_true",
        help="Prefer lower-latency engines when possible",
    )
    decide.add_argument(
        "--provider-allow",
        action="append",
        default=None,
        help="Only allow this provider; repeat for multiple providers",
    )
    decide.add_argument(
        "--provider-deny",
        action="append",
        default=None,
        help="Deny this provider; repeat for multiple providers",
    )
    decide.add_argument(
        "--local-only",
        action="store_true",
        help="Reject hosted/API providers for this decision",
    )
    decide.add_argument(
        "--no-hosted",
        action="store_true",
        help="Alias for --local-only",
    )
    decide.add_argument("prompt", nargs="+", help="Prompt text to route")
    decide.set_defaults(func=_cmd_decide)

    dispatch = subparsers.add_parser(
        "dispatch-plan",
        help="Build a safe dry-run dispatch plan without executing adapters",
    )
    _add_routing_hint_args(dispatch)
    dispatch.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON dry-run dispatch plan",
    )
    dispatch.add_argument(
        "--include-alternatives",
        action="store_true",
        help="Include ranked alternative engines in the embedded receipt",
    )
    dispatch.add_argument("prompt", nargs="+", help="Prompt text to plan")
    dispatch.set_defaults(func=_cmd_dispatch_plan)

    workflow_benchmark = subparsers.add_parser(
        "workflow-benchmark",
        help="Run offline workflow routing correctness benchmarks",
    )
    workflow_benchmark.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
    )
    workflow_benchmark.add_argument(
        "--case",
        action="append",
        choices=workflow_case_names(),
        default=None,
        help="Run one workflow fixture by name; repeat for multiple cases",
    )
    workflow_benchmark.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when any expected route changes",
    )
    workflow_benchmark.add_argument("--json", action="store_true", help="Emit JSON output")
    workflow_benchmark.set_defaults(func=_cmd_workflow_benchmark)

    eval_cmd = subparsers.add_parser(
        "eval",
        help="Run privacy-safe local suitability evals against a configured backend",
    )
    eval_subparsers = eval_cmd.add_subparsers(dest="eval_command", required=True)
    eval_list = eval_subparsers.add_parser(
        "list",
        help="List built-in eval fixtures without printing prompt bodies",
    )
    eval_list.add_argument(
        "--category",
        default=None,
        help="Only list fixtures from one category",
    )
    eval_list.add_argument("--json", action="store_true", help="Emit JSON output")
    eval_list.set_defaults(func=_cmd_eval_list)

    eval_run = eval_subparsers.add_parser(
        "run",
        help="Execute eval fixtures against one explicitly selected backend",
    )
    eval_run.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    eval_run.add_argument(
        "--backend",
        required=True,
        help="Backend name from routing_proxy.yaml",
    )
    eval_run.add_argument(
        "--model",
        default=None,
        help="Model id to send upstream; defaults to the configured backend model",
    )
    fixture_selection = eval_run.add_mutually_exclusive_group(required=True)
    fixture_selection.add_argument(
        "--fixture",
        default=None,
        help="Fixture id or category to run",
    )
    fixture_selection.add_argument(
        "--all",
        "--all-fixtures",
        dest="all",
        action="store_true",
        help="Run all built-in eval fixtures; requires --confirm-large-run",
    )
    eval_run.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_EVAL_RESULTS_PATH),
        help="Path for eval result JSONL",
    )
    eval_run.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-fixture backend timeout in seconds",
    )
    eval_run.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for automation; defaults to a timestamped id",
    )
    eval_run.add_argument(
        "--confirm-large-run",
        action="store_true",
        help=(
            "Confirm a broad eval run after reviewing time/cost risk; "
            "required with --all-fixtures"
        ),
    )
    eval_run.add_argument("--json", action="store_true", help="Emit JSON output")
    eval_run.set_defaults(func=_cmd_eval_run)

    eval_report_cmd = eval_subparsers.add_parser(
        "report",
        help="Summarize eval JSONL results without raw prompts or outputs",
    )
    eval_report_cmd.add_argument("run_id", help="Run id to report, or latest")
    eval_report_cmd.add_argument(
        "--results",
        type=Path,
        default=Path(DEFAULT_EVAL_RESULTS_PATH),
        help="Path to eval result JSONL",
    )
    eval_report_cmd.add_argument("--json", action="store_true", help="Emit JSON output")
    eval_report_cmd.set_defaults(func=_cmd_eval_report)

    eval_evidence = eval_subparsers.add_parser(
        "evidence",
        help="Show cached advisory eval evidence for one model",
    )
    eval_evidence.add_argument("--model", required=True, help="Model id to inspect")
    eval_evidence.add_argument(
        "--backend",
        default=None,
        help="Optional backend filter from routing_proxy.yaml",
    )
    eval_evidence.add_argument(
        "--results",
        type=Path,
        default=Path(DEFAULT_EVAL_RESULTS_PATH),
        help="Path to eval result JSONL",
    )
    eval_evidence.add_argument("--json", action="store_true", help="Emit JSON output")
    eval_evidence.set_defaults(func=_cmd_eval_evidence)

    catalog = subparsers.add_parser(
        "catalog",
        help="Inspect and apply packaged catalog updates",
    )
    catalog_subparsers = catalog.add_subparsers(
        dest="catalog_command",
        required=True,
    )
    catalog_status_parser = catalog_subparsers.add_parser(
        "status",
        help="Show packaged catalog and local config status",
    )
    catalog_status_parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "model_router.yaml",
        help="Path to local model_router.yaml",
    )
    catalog_status_parser.add_argument(
        "--migration-log",
        type=Path,
        default=None,
        help="Path to catalog migration JSONL",
    )
    catalog_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )
    catalog_status_parser.set_defaults(func=_cmd_catalog_status)

    catalog_diff_parser = catalog_subparsers.add_parser(
        "diff",
        help="Preview packaged router catalog changes without applying them",
    )
    catalog_diff_parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "model_router.yaml",
        help="Path to local model_router.yaml",
    )
    catalog_diff_parser.add_argument(
        "--migration-log",
        type=Path,
        default=None,
        help="Path to catalog migration JSONL",
    )
    catalog_diff_parser.add_argument(
        "--context",
        type=_positive_int,
        default=3,
        help="Unified diff context lines",
    )
    catalog_diff_parser.add_argument(
        "--max-lines",
        type=_positive_int,
        default=240,
        help="Maximum diff lines to print",
    )
    catalog_diff_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    catalog_diff_parser.set_defaults(func=_cmd_catalog_diff)

    catalog_apply_parser = catalog_subparsers.add_parser(
        "apply",
        help="Apply packaged router catalog defaults after confirmation",
    )
    catalog_apply_parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "model_router.yaml",
        help="Path to local model_router.yaml",
    )
    catalog_apply_parser.add_argument(
        "--migration-log",
        type=Path,
        default=None,
        help="Path to catalog migration JSONL",
    )
    catalog_apply_parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply without an interactive prompt",
    )
    catalog_apply_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    catalog_apply_parser.set_defaults(func=_cmd_catalog_apply)

    dogfood = subparsers.add_parser(
        "dogfood",
        help="Run opt-in local dogfood checks",
    )
    dogfood_subparsers = dogfood.add_subparsers(
        dest="dogfood_command",
        required=True,
    )
    dogfood_proxy = dogfood_subparsers.add_parser(
        "proxy",
        help="Plan or run live local proxy dogfood checks",
    )
    dogfood_proxy.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    dogfood_proxy.add_argument(
        "--endpoint",
        default=None,
        help="Proxy endpoint root such as http://127.0.0.1:8082",
    )
    dogfood_proxy.add_argument(
        "--execute",
        action="store_true",
        help="Run live local HTTP checks against the proxy",
    )
    dogfood_proxy.add_argument(
        "--require-running",
        action="store_true",
        help="Treat an unavailable proxy as a failure instead of a skip",
    )
    dogfood_proxy.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds",
    )
    dogfood_proxy.add_argument("--json", action="store_true", help="Emit JSON output")
    dogfood_proxy.set_defaults(func=_cmd_dogfood_proxy)

    validate = subparsers.add_parser(
        "validate-config",
        help="Validate catalog shape and non-executing availability checks",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON validation report",
    )
    validate.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
    )
    validate.set_defaults(func=_cmd_validate_config)

    validate_proxy = subparsers.add_parser(
        "validate-proxy-config",
        help="Validate routing proxy config shape",
    )
    validate_proxy.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    validate_proxy.add_argument("--json", action="store_true", help="Emit JSON output")
    validate_proxy.set_defaults(func=_cmd_validate_proxy_config)

    doctor = subparsers.add_parser(
        "doctor",
        help="Validate configs and check backend reachability",
    )
    doctor.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    doctor.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Backend health timeout in seconds",
    )
    doctor.add_argument("--json", action="store_true", help="Emit JSON output")
    doctor.set_defaults(func=_cmd_doctor)

    settings = subparsers.add_parser(
        "settings",
        help="Run the local ModelRouter admin settings UI",
    )
    settings.add_argument(
        "--config-dir",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR),
        help="Directory containing routing_proxy.yaml and telemetry files",
    )
    settings.add_argument(
        "--host",
        default="127.0.0.1",
        help="Settings UI bind host. Defaults to localhost.",
    )
    settings.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Settings UI port",
    )
    settings.add_argument(
        "--no-open",
        action="store_true",
        help="Print the URL without opening a browser",
    )
    settings.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="Uvicorn log level",
    )
    settings.set_defaults(func=_cmd_settings)

    tui = subparsers.add_parser(
        "tui",
        help="Run the terminal ModelRouter control center",
        description="Run the terminal ModelRouter control center.",
    )
    tui.add_argument(
        "--config-dir",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR),
        help="Directory containing routing_proxy.yaml and telemetry files",
    )
    tui.set_defaults(func=_cmd_tui)

    runtime = subparsers.add_parser(
        "runtime",
        help="Inspect or explicitly invoke runtime adapter maintenance actions",
        description=(
            "Inspect or explicitly invoke runtime adapter maintenance actions. "
            "Runtime adapters coordinate external/proven runtimes for operators; "
            "they are not required for route_fast or proxy forwarding."
        ),
    )
    runtime_subparsers = runtime.add_subparsers(
        dest="runtime_command",
        required=True,
    )
    runtime_status = runtime_subparsers.add_parser(
        "status",
        help="Show runtime adapter detection, health, and capabilities",
    )
    _add_runtime_common_args(runtime_status, backend_required=False)
    runtime_status.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.status",
    )
    runtime_models = runtime_subparsers.add_parser(
        "models",
        help="List models visible to a runtime adapter",
    )
    _add_runtime_common_args(runtime_models)
    runtime_models.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.models",
    )
    runtime_loaded = runtime_subparsers.add_parser(
        "loaded",
        help="List loaded models when the runtime supports it",
    )
    _add_runtime_common_args(runtime_loaded)
    runtime_loaded.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.loaded_models",
    )
    runtime_start = runtime_subparsers.add_parser(
        "start",
        help="Explicitly start a runtime only when the adapter supports it",
    )
    _add_runtime_common_args(runtime_start, mutating=True)
    runtime_start.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.start_server",
    )
    runtime_stop = runtime_subparsers.add_parser(
        "stop",
        help="Explicitly stop a runtime only when the adapter supports it",
    )
    _add_runtime_common_args(runtime_stop, mutating=True)
    runtime_stop.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.stop_server",
    )
    runtime_load = runtime_subparsers.add_parser(
        "load",
        help="Explicitly load a model only when the adapter supports it",
    )
    _add_runtime_common_args(runtime_load, mutating=True)
    runtime_load.add_argument("--model", required=True, help="Runtime model id")
    runtime_load.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.load_model",
    )
    runtime_unload = runtime_subparsers.add_parser(
        "unload",
        help="Explicitly unload a model only when the adapter supports it",
    )
    _add_runtime_common_args(runtime_unload, mutating=True)
    runtime_unload.add_argument("--model", required=True, help="Runtime model id")
    runtime_unload.set_defaults(
        func=_cmd_runtime_action,
        runtime_action_id="runtime.unload_model",
    )

    runtimes = subparsers.add_parser(
        "runtimes",
        help="Guided runtime status, doctor, and connect assistant",
        description=(
            "Guided runtime status, doctor, and connect assistant. Plans are "
            "preview-only by default; config writes require --write --yes."
        ),
    )
    runtimes_subparsers = runtimes.add_subparsers(
        dest="runtimes_command",
        required=True,
    )
    runtimes_status = runtimes_subparsers.add_parser(
        "status",
        help="Show configured runtime status and guidance",
    )
    _add_guided_runtime_common_args(runtimes_status)
    runtimes_status.set_defaults(func=_cmd_runtimes_status)

    runtimes_doctor = runtimes_subparsers.add_parser(
        "doctor",
        help="Diagnose runtime connectivity and show next actions",
    )
    _add_guided_runtime_common_args(runtimes_doctor)
    runtimes_doctor.set_defaults(func=_cmd_runtimes_doctor)

    runtimes_connect = runtimes_subparsers.add_parser(
        "connect",
        help="Preview or apply a safe backend connection for a runtime",
    )
    runtimes_connect.add_argument(
        "runtime_id",
        choices=("lmstudio", "ollama", "llamacpp"),
        help="Runtime to connect",
    )
    _add_guided_runtime_common_args(runtimes_connect)
    runtimes_connect.add_argument(
        "--backend",
        default="fast",
        help="Existing backend to update when --write --yes is supplied",
    )
    runtimes_connect.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI-compatible endpoint; defaults depend on runtime",
    )
    runtimes_connect.add_argument(
        "--model",
        default=None,
        help="Model id/tag to write into the selected backend",
    )
    runtimes_connect.add_argument(
        "--write",
        action="store_true",
        help="Write the previewed backend config patch",
    )
    runtimes_connect.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the explicit config write",
    )
    runtimes_connect.set_defaults(func=_cmd_runtimes_connect)

    feedback = subparsers.add_parser(
        "feedback",
        help="Append a hindsight label for a logged routing event",
    )
    feedback.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_FEEDBACK_PATH),
        help="Path to routing-feedback JSONL output",
    )
    feedback.add_argument(
        "--notes",
        default=None,
        help="Optional short note explaining the correction",
    )
    feedback.add_argument(
        "--outcome",
        choices=OUTCOME_LABELS,
        default=None,
        help="Optional manually supplied outcome label",
    )
    feedback.add_argument("request_id", help="Request id from routing-events JSONL")
    feedback.add_argument("expected_engine", help="Correct engine for this event")
    feedback.set_defaults(func=_cmd_feedback)

    telemetry = subparsers.add_parser(
        "telemetry",
        help="Inspect routing events, feedback labels, and replay coverage",
    )
    telemetry_subparsers = telemetry.add_subparsers(
        dest="telemetry_command",
        required=True,
    )
    telemetry_summary = telemetry_subparsers.add_parser(
        "summary",
        help="Summarize routing telemetry and replay mismatches",
    )
    _add_telemetry_paths(telemetry_summary)
    telemetry_summary.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
    )
    telemetry_summary.add_argument(
        "--max-examples",
        type=_positive_int,
        default=10,
        help="Maximum request ids to show per example list",
    )
    _add_pricing_catalog_arg(telemetry_summary)
    telemetry_summary.add_argument(
        "--pricing-override-skeleton",
        action="store_true",
        help="Include an operator-editable pricing override skeleton for catalog gaps",
    )
    telemetry_summary.add_argument("--json", action="store_true", help="Emit JSON")
    telemetry_summary.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any feedback-labeled event now routes incorrectly",
    )
    telemetry_summary.set_defaults(func=_cmd_telemetry_summary)

    telemetry_feedback = telemetry_subparsers.add_parser(
        "feedback",
        help="List feedback labels without printing prompts",
    )
    _add_telemetry_paths(telemetry_feedback)
    telemetry_feedback.add_argument(
        "--include-notes",
        action="store_true",
        help="Include feedback notes in output",
    )
    telemetry_feedback.add_argument(
        "--max-rows",
        type=_positive_int,
        default=50,
        help="Maximum labels to show",
    )
    telemetry_feedback.add_argument("--json", action="store_true", help="Emit JSON")
    telemetry_feedback.set_defaults(func=_cmd_telemetry_feedback)

    telemetry_review = telemetry_subparsers.add_parser(
        "review",
        help="Show a privacy-safe wrong-route review queue",
    )
    _add_telemetry_paths(telemetry_review)
    telemetry_review.add_argument(
        "--max-rows",
        type=_positive_int,
        default=20,
        help="Maximum unlabeled events to show",
    )
    _add_pricing_catalog_arg(telemetry_review)
    telemetry_review.add_argument(
        "--pricing-override-skeleton",
        action="store_true",
        help="Include an operator-editable pricing override skeleton for catalog gaps",
    )
    telemetry_review.add_argument("--json", action="store_true", help="Emit JSON")
    telemetry_review.set_defaults(func=_cmd_telemetry_review)

    pricing = subparsers.add_parser(
        "pricing",
        help="Inspect and maintain local pricing catalog metadata",
    )
    pricing_subparsers = pricing.add_subparsers(
        dest="pricing_command",
        required=True,
    )
    pricing_status_parser = pricing_subparsers.add_parser(
        "status",
        help="Show packaged and local pricing catalog status",
    )
    _add_pricing_maintenance_args(pricing_status_parser)
    pricing_status_parser.add_argument("--json", action="store_true", help="Emit JSON")
    pricing_status_parser.set_defaults(func=_cmd_pricing_status)

    pricing_diff_parser = pricing_subparsers.add_parser(
        "diff",
        help="Preview local pricing catalog override changes",
    )
    _add_pricing_maintenance_args(pricing_diff_parser)
    pricing_diff_parser.add_argument(
        "--context",
        type=_positive_int,
        default=3,
        help="Unified diff context lines",
    )
    pricing_diff_parser.add_argument(
        "--max-lines",
        type=_positive_int,
        default=240,
        help="Maximum diff lines to print",
    )
    pricing_diff_parser.add_argument("--json", action="store_true", help="Emit JSON")
    pricing_diff_parser.set_defaults(func=_cmd_pricing_diff)

    pricing_apply_parser = pricing_subparsers.add_parser(
        "apply",
        help="Write packaged pricing metadata to the local override after confirmation",
    )
    _add_pricing_maintenance_args(pricing_apply_parser)
    pricing_apply_parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply without an interactive prompt",
    )
    pricing_apply_parser.add_argument("--json", action="store_true", help="Emit JSON")
    pricing_apply_parser.set_defaults(func=_cmd_pricing_apply)

    setup = subparsers.add_parser(
        "setup",
        help="Scan local models and generate recommended router config",
    )
    setup_subparsers = setup.add_subparsers(dest="setup_command", required=True)

    scan = setup_subparsers.add_parser(
        "scan",
        help="Scan safe local signals such as model cache dirs and commands",
    )
    _add_setup_scan_args(scan)
    scan.set_defaults(func=_cmd_setup_scan)

    recommend = setup_subparsers.add_parser(
        "recommend",
        help="Recommend routing targets, engine overrides, and download plans",
    )
    _add_setup_scan_args(recommend)
    recommend.add_argument(
        "--profile",
        default="balanced",
        choices=("balanced", "lightweight", "quality"),
        help="Recommendation profile for future model download plans",
    )
    recommend.add_argument(
        "--download-alternatives",
        type=_positive_int,
        default=2,
        help="Recommended download candidates per route",
    )
    recommend.add_argument(
        "--benchmark-results",
        type=Path,
        default=Path(DEFAULT_BENCHMARK_PATH),
        help="Path to privacy-safe local benchmark results",
    )
    recommend.set_defaults(func=_cmd_setup_recommend)

    prereqs = setup_subparsers.add_parser(
        "install-prereqs",
        help="Plan or install Python prerequisites into the active environment",
    )
    prereqs.add_argument(
        "--preset",
        default="proxy",
        choices=("proxy", "mlx-lm", "llamacpp", "all"),
        help="Prerequisite set to plan or install",
    )
    prereqs.add_argument(
        "--execute",
        action="store_true",
        help="Run the planned pip install commands",
    )
    prereqs.add_argument(
        "--yes",
        action="store_true",
        help="Confirm execution without an interactive prompt",
    )
    prereqs.add_argument("--json", action="store_true", help="Emit JSON output")
    prereqs.set_defaults(func=_cmd_setup_install_prereqs)

    download = setup_subparsers.add_parser(
        "download",
        help="Plan or execute approved Hugging Face model downloads",
    )
    _add_setup_scan_args(download)
    download.add_argument(
        "--profile",
        default="balanced",
        choices=("balanced", "lightweight", "quality"),
        help="Recommendation profile for model download plans",
    )
    download.add_argument(
        "--route",
        action="append",
        default=None,
        help="Only include a route such as fast_local or multimodal_vision",
    )
    download.add_argument(
        "--repo-id",
        default=None,
        help="Custom Hugging Face repo id to download for the selected route",
    )
    download.add_argument(
        "--adapter",
        default=None,
        help="Adapter name for a custom repo download plan",
    )
    download.add_argument(
        "--alternatives",
        type=_positive_int,
        default=1,
        help="Recommended download candidates per route",
    )
    download.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Root directory for downloaded model folders",
    )
    download.add_argument(
        "--execute",
        action="store_true",
        help="Run the planned hf download commands",
    )
    download.add_argument(
        "--yes",
        action="store_true",
        help="Confirm execution without an interactive prompt",
    )
    download.add_argument(
        "--benchmark-results",
        type=Path,
        default=Path(DEFAULT_BENCHMARK_PATH),
        help="Path to privacy-safe local benchmark results",
    )
    download.set_defaults(func=_cmd_setup_download)

    benchmark = setup_subparsers.add_parser(
        "benchmark",
        help="Plan or run privacy-safe local backend smoke benchmarks",
    )
    benchmark.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    benchmark.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Only benchmark a configured backend name",
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_BENCHMARK_PATH),
        help="Path for benchmark result JSON",
    )
    benchmark.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-backend benchmark timeout in seconds",
    )
    benchmark.add_argument(
        "--execute",
        action="store_true",
        help="Run the local benchmark requests",
    )
    benchmark.add_argument(
        "--yes",
        action="store_true",
        help="Confirm execution without an interactive prompt",
    )
    benchmark.add_argument("--json", action="store_true", help="Emit JSON output")
    benchmark.set_defaults(func=_cmd_setup_benchmark)

    write = setup_subparsers.add_parser(
        "write",
        help="Write a recommended model-router config file",
    )
    _add_setup_scan_args(write)
    write.add_argument(
        "--profile",
        default="balanced",
        choices=("balanced", "lightweight", "quality"),
        help="Recommendation profile for future model download plans",
    )
    write.add_argument(
        "--download-alternatives",
        type=_positive_int,
        default=2,
        help="Recommended download candidates per route",
    )
    write.add_argument(
        "--output",
        type=Path,
        default=Path("configs/model_router.local.yaml"),
        help="Path for the generated config",
    )
    write.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file",
    )
    write.set_defaults(func=_cmd_setup_write)

    wizard = setup_subparsers.add_parser(
        "wizard",
        help="Interactive setup flow that asks before writing config",
    )
    _add_setup_scan_args(wizard)
    wizard.add_argument(
        "--profile",
        default="balanced",
        choices=("balanced", "lightweight", "quality"),
        help="Recommendation profile for future model download plans",
    )
    wizard.add_argument(
        "--output",
        type=Path,
        default=Path("configs/model_router.local.yaml"),
        help="Path for the generated config",
    )
    wizard.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file",
    )
    wizard.set_defaults(func=_cmd_setup_wizard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _cmd_decide(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    hints = _routing_hints_from_args(args)
    try:
        decision = ModelRouter.from_config(args.config).route(prompt, hints=hints)
    except RouterConfigError:
        decision = route_prompt(prompt, config_path=args.config, hints=hints)
    receipt = decision_to_receipt(decision)
    if args.json:
        print(receipt_to_json(receipt))
    elif args.explain:
        _print_explain(receipt)
    else:
        _print_readable(receipt)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        result = initialize_product_config(
            preset=args.preset,
            auto_detect=args.auto,
            auto_models=args.auto_models,
            model_dirs=_model_dirs_from_args(args),
            profile=args.profile,
            config_dir=args.config_dir,
            proxy_port=args.proxy_port,
            force=args.force,
            interactive=not args.yes,
        )
    except ValueError as exc:
        print(f"Init failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        for message in result.messages:
            print(message)
        if result.written:
            print("Written:")
            for path in result.written:
                print(f"- {path}")
        if result.skipped:
            print("Skipped:")
            for path in result.skipped:
                print(f"- {path}")
    return 0 if result.ok else 1


def _cmd_install(args: argparse.Namespace) -> int:
    try:
        plan = build_install_plan(options_from_namespace(args))
    except ValueError as exc:
        print(f"Install planning failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        _print_install_plan(plan.to_dict())
    return 0 if plan.ok else 1


def _cmd_dispatch_plan(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    hints = _routing_hints_from_args(args)
    plan = build_dispatch_plan(
        prompt,
        config_path=args.config,
        hints=hints,
        include_alternatives=args.include_alternatives,
    )
    if args.json:
        print(dispatch_plan_to_json(plan))
    else:
        _print_dispatch_plan(plan)
    return 0


def _cmd_workflow_benchmark(args: argparse.Namespace) -> int:
    report = run_workflow_benchmarks(
        config_path=args.config,
        cases=workflow_cases_by_name(args.case),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_workflow_benchmark_report(report)
    return 1 if args.fail_on_mismatch and not report.ok else 0


def _cmd_eval_list(args: argparse.Namespace) -> int:
    try:
        fixtures = eval_fixture_summaries(category=args.category)
    except EvalFixtureError as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Eval fixture list failed: {exc}", file=sys.stderr)
        return 1
    payload = {
        "ok": True,
        "fixtures": list(fixtures),
        "privacy": "prompt bodies are not printed by eval list",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_eval_fixture_list(payload)
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    try:
        execution = execute_eval_run(
            config_path=args.config,
            backend=args.backend,
            model=args.model,
            fixture_selector=args.fixture,
            all_fixtures=args.all,
            output_path=args.output,
            timeout_seconds=args.timeout,
            run_id=args.run_id,
            confirm_large_run=args.confirm_large_run,
        )
    except (EvalFixtureError, ProxyConfigError, OSError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Eval run failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(execution.to_dict(), indent=2, sort_keys=True))
    else:
        _print_eval_run_execution(execution)
    return 0 if execution.ok else 1


def _cmd_eval_report(args: argparse.Namespace) -> int:
    report = eval_report(args.run_id, result_path=args.results)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_eval_report(report)
    return 0


def _cmd_eval_evidence(args: argparse.Namespace) -> int:
    evidence = eval_evidence_for_model(
        args.model,
        result_path=args.results,
        backend=args.backend,
    )
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        _print_eval_evidence(evidence)
    return 0


def _cmd_catalog_status(args: argparse.Namespace) -> int:
    status = catalog_status(args.config, migration_log=args.migration_log)
    if args.json:
        print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    else:
        _print_catalog_status(status)
    return 0


def _cmd_catalog_diff(args: argparse.Namespace) -> int:
    diff = catalog_diff(
        args.config,
        migration_log=args.migration_log,
        context_lines=args.context,
        max_lines=args.max_lines,
    )
    if args.json:
        print(json.dumps(diff.to_dict(), indent=2, sort_keys=True))
    else:
        _print_catalog_diff(diff)
    return 0


def _cmd_catalog_apply(args: argparse.Namespace) -> int:
    confirmed = args.yes
    if not confirmed and not args.json:
        diff = catalog_diff(args.config, migration_log=args.migration_log)
        _print_catalog_diff(diff)
        if diff.has_changes:
            answer = input("Apply packaged catalog update? [y/N] ").strip().lower()
            confirmed = answer in {"y", "yes"}
        else:
            confirmed = True

    result = apply_catalog_update(
        args.config,
        confirmed=confirmed,
        migration_log=args.migration_log,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_catalog_apply_result(result)
    return 0 if result.ok else 1


def _cmd_dogfood_proxy(args: argparse.Namespace) -> int:
    report = run_proxy_dogfood(
        config_path=args.config,
        endpoint=args.endpoint,
        execute=args.execute,
        require_running=args.require_running,
        timeout_seconds=args.timeout,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_dogfood_report(report)
    return 0 if report.ok else 1


def _add_routing_hint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
    )
    parser.add_argument(
        "--profile",
        dest="routing_profile",
        default="balanced",
        choices=ROUTING_PROFILE_VALUES,
        help="Routing profile to compile into routing constraints",
    )
    parser.add_argument(
        "--force-engine",
        default=None,
        help="Prefer a specific engine by name; high-risk actions still confirm",
    )
    parser.add_argument(
        "--attachment",
        action="append",
        choices=("image", "pdf", "audio", "code"),
        default=None,
        help="Declare an attachment modality for routing constraints",
    )
    parser.add_argument(
        "--max-cost-tier",
        default=None,
        help="Reject engines above this cost tier",
    )
    parser.add_argument(
        "--max-latency-tier",
        default=None,
        help="Reject engines above this latency tier",
    )
    parser.add_argument(
        "--latency-sensitive",
        action="store_true",
        help="Prefer lower-latency engines when possible",
    )
    parser.add_argument(
        "--provider-allow",
        action="append",
        default=None,
        help="Only allow this provider; repeat for multiple providers",
    )
    parser.add_argument(
        "--provider-deny",
        action="append",
        default=None,
        help="Deny this provider; repeat for multiple providers",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Reject hosted/API providers for this decision",
    )
    parser.add_argument(
        "--no-hosted",
        action="store_true",
        help="Alias for --local-only",
    )


def _routing_hints_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "force_engine": args.force_engine,
        "profile": getattr(args, "routing_profile", "balanced"),
        "attachments": args.attachment or [],
        "max_cost_tier": args.max_cost_tier,
        "max_latency_tier": args.max_latency_tier,
        "latency_sensitive": args.latency_sensitive,
        "provider_allowlist": getattr(args, "provider_allow", None) or [],
        "provider_denylist": getattr(args, "provider_deny", None) or [],
        "local_only": bool(getattr(args, "local_only", False))
        or bool(getattr(args, "no_hosted", False)),
        "hosted_allowed": False if getattr(args, "no_hosted", False) else None,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _add_telemetry_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(DEFAULT_LOG_PATH),
        help="Path to routing-events JSONL",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path(DEFAULT_FEEDBACK_PATH),
        help="Path to routing-feedback JSONL",
    )


def _add_pricing_catalog_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pricing-catalog",
        type=Path,
        default=None,
        help=(
            "Optional local pricing catalog override for reporting estimates; "
            "no network pricing checks are made"
        ),
    )


def _add_pricing_maintenance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--override",
        type=Path,
        default=default_pricing_override_path(),
        help=(
            f"Local pricing override path, defaults to "
            f"~/.model-router/{DEFAULT_PRICING_CATALOG_NAME}"
        ),
    )


def _add_runtime_common_args(
    parser: argparse.ArgumentParser,
    *,
    backend_required: bool = True,
    mutating: bool = False,
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    parser.add_argument(
        "--backend",
        required=backend_required,
        help="Backend name from routing_proxy.yaml",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.25,
        help="Bounded runtime status/model-list timeout in seconds",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    if mutating:
        parser.add_argument(
            "--yes",
            action="store_true",
            help="Confirm the explicit runtime maintenance operation.",
        )


def _add_guided_runtime_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_DIR) / "routing_proxy.yaml",
        help="Path to routing_proxy.yaml",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.25,
        help="Bounded local runtime health timeout in seconds",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")


def _runtime_cli_paths(config_path: Path) -> dict[str, Path]:
    config_dir = config_path.expanduser().parent
    return {
        "proxy_config": config_path,
        "model_router_config": config_dir / "model_router.yaml",
        "benchmarks": config_dir / "benchmarks.json",
        "models": config_dir / "models",
        "pricing": config_dir / DEFAULT_PRICING_CATALOG_NAME,
        "feedback": Path(DEFAULT_FEEDBACK_PATH),
    }


def _cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        config = load_router_config(args.config)
    except RouterConfigError as exc:
        payload = {
            "config_valid": False,
            "error": str(exc),
            "all_available": False,
            "engines": {},
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Config valid: false")
            print(f"Error: {exc}")
        return 1

    report = validate_router_availability(config)
    payload = {"config_valid": True, **report.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Config valid: true")
        print(f"All enabled engines available: {str(report.all_available).lower()}")
        for name, result in sorted(report.engines.items()):
            print(f"- {name}: {'available' if result.available else 'unavailable'}")
            for reason in result.reasons:
                print(f"  - {reason}")
    return 0 if report.all_available else 1


def _cmd_validate_proxy_config(args: argparse.Namespace) -> int:
    try:
        config = validate_proxy_config(args.config)
    except ProxyConfigError as exc:
        payload = {"config_valid": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Proxy config valid: false")
            print(f"Error: {exc}")
        return 1

    payload = {
        "config_valid": True,
        "proxy_config": config.source_path,
        "router_config": config.router_config or "default",
        "routing_profile": config.proxy.routing_profile,
        "backends": sorted(config.backends),
        "backend_policy": config.backend_policy.to_dict(),
        "verifier": config.verifier.to_dict(),
        "engine_backends": dict(sorted(config.engine_backends.items())),
        "observability": {
            "enabled": config.observability.enabled,
            "prompt_capture": config.observability.prompt_capture,
            "max_bytes": config.observability.max_bytes,
            "backups": config.observability.backups,
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Proxy config valid: true")
        print(f"Proxy config: {config.source_path}")
        print(f"Router config: {config.router_config or 'default'}")
        print(f"Routing profile: {config.proxy.routing_profile}")
        print("Backends: " + ", ".join(sorted(config.backends)))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_proxy_config(args.config, timeout_seconds=args.timeout)
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Proxy config valid: {str(report.proxy_config_valid).lower()}")
        print(f"Router config valid: {str(report.router_config_valid).lower()}")
        print(f"Overall ok: {str(report.ok).lower()}")
        if report.proxy_endpoint:
            print(f"Agent endpoint: {report.proxy_endpoint}")
        if report.telemetry_log_path:
            print(f"Telemetry log: {report.telemetry_log_path}")
        features = payload.get("maturity", {}).get("features", [])
        if features:
            print("Feature maturity:")
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                print(
                    "- "
                    f"{feature.get('label', feature.get('feature_id', 'unknown'))}: "
                    f"{feature.get('maturity', 'unknown')}"
                )
        if report.errors:
            print("Errors:")
            for error in report.errors:
                print(f"- {error}")
        print("Backends:")
        for backend in report.backends:
            status = "reachable" if backend.reachable else "unreachable"
            print(f"- {backend.backend}: {status} ({backend.detail})")
        if report.remediation:
            print("Next steps:")
            for item in report.remediation:
                print(f"- {item}")
    return 0 if report.ok else 1


def _cmd_settings(args: argparse.Namespace) -> int:
    from hermes.plugins.model_router.settings_ui import (
        SettingsDependencyError,
        run_settings_server,
    )

    try:
        return run_settings_server(
            config_dir=args.config_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            log_level=args.log_level,
        )
    except SettingsDependencyError as exc:
        print(f"Settings UI failed: {exc}", file=sys.stderr)
        return 1


def _cmd_tui(args: argparse.Namespace) -> int:
    from hermes.plugins.model_router.tui import run_tui

    return run_tui(config_dir=args.config_dir)


def _cmd_runtime_action(args: argparse.Namespace) -> int:
    action_id = args.runtime_action_id
    mutating = action_id in {
        "runtime.start_server",
        "runtime.stop_server",
        "runtime.load_model",
        "runtime.unload_model",
    }
    confirmed = bool(getattr(args, "yes", False))
    if mutating and not confirmed and not args.json:
        backend = getattr(args, "backend", "")
        answer = input(f"Run {action_id} for backend {backend}? [y/N] ").strip().lower()
        confirmed = answer in {"y", "yes"}
    payload: dict[str, Any] = {
        "backend": getattr(args, "backend", "") or "",
        "timeout_seconds": getattr(args, "timeout", 0.25),
    }
    model = getattr(args, "model", None)
    if model:
        payload["model"] = model
    if mutating:
        payload["confirm"] = confirmed
    try:
        result = run_admin_action(action_id, _runtime_cli_paths(args.config), payload)
    except AdminActionError as exc:
        error_payload = {"ok": False, "error": str(exc), "details": exc.details}
        if args.json:
            print(json.dumps(error_payload, indent=2, sort_keys=True))
        else:
            print(f"Runtime action failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_runtime_action_result(result)
    body = result.get("payload")
    ok = isinstance(body, dict) and body.get("ok", result.get("ok")) is not False
    return 0 if ok else 1


def _cmd_runtimes_status(args: argparse.Namespace) -> int:
    try:
        report = runtime_status_report(
            args.config,
            timeout_seconds=args.timeout,
        )
    except (RuntimeInstallError, ProxyConfigError, OSError, ValueError) as exc:
        return _print_runtime_install_error(exc, json_output=args.json)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_runtimes_status(report)
    return 0


def _cmd_runtimes_doctor(args: argparse.Namespace) -> int:
    try:
        report = runtime_doctor_report(
            args.config,
            timeout_seconds=args.timeout,
        )
    except (RuntimeInstallError, ProxyConfigError, OSError, ValueError) as exc:
        return _print_runtime_install_error(exc, json_output=args.json)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_runtimes_doctor(report)
    return 0


def _cmd_runtimes_connect(args: argparse.Namespace) -> int:
    confirmed = bool(args.yes)
    if args.write and not confirmed and not args.json:
        try:
            preview = build_runtime_connect_plan(
                RuntimeConnectRequest(
                    runtime_id=args.runtime_id,
                    config_path=args.config,
                    backend=args.backend,
                    endpoint=args.endpoint,
                    model=args.model,
                    write=False,
                    confirmed=False,
                    timeout_seconds=args.timeout,
                )
            )
        except (RuntimeInstallError, ProxyConfigError, OSError, ValueError) as exc:
            return _print_runtime_install_error(exc, json_output=args.json)
        _print_runtimes_connect_plan(preview)
        answer = input("Write this runtime backend config patch? [y/N] ").strip().lower()
        confirmed = answer in {"y", "yes"}
    try:
        plan = build_runtime_connect_plan(
            RuntimeConnectRequest(
                runtime_id=args.runtime_id,
                config_path=args.config,
                backend=args.backend,
                endpoint=args.endpoint,
                model=args.model,
                write=args.write,
                confirmed=confirmed,
                timeout_seconds=args.timeout,
            )
        )
    except (RuntimeInstallError, ProxyConfigError, OSError, ValueError) as exc:
        return _print_runtime_install_error(exc, json_output=args.json)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        _print_runtimes_connect_plan(plan)
    return 0 if plan.get("ok") else 1


def _print_runtime_install_error(exc: Exception, *, json_output: bool) -> int:
    if json_output:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
    else:
        print(f"Runtime guidance failed: {exc}", file=sys.stderr)
    return 1


def _cmd_feedback(args: argparse.Namespace) -> int:
    writer = RoutingLogWriter(args.output)
    payload = build_feedback(
        request_id=args.request_id,
        expected_engine=args.expected_engine,
        outcome_label=args.outcome,
        notes=args.notes,
    )
    if not writer.write(payload):
        print(f"Failed to write feedback to {args.output}", file=sys.stderr)
        return 1
    print(f"Feedback written to {args.output}")
    return 0


def _cmd_telemetry_summary(args: argparse.Namespace) -> int:
    summary = replay_events(
        events_path=args.events,
        feedback_path=args.feedback,
        config_path=args.config,
        pricing_catalog_path=args.pricing_catalog,
        max_examples=args.max_examples,
    )
    _attach_pricing_override_skeleton(summary, enabled=args.pricing_override_skeleton)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_telemetry_summary(summary)
    if args.fail_on_regression and summary["expected_mismatch_count"]:
        return 1
    return 0


def _cmd_telemetry_feedback(args: argparse.Namespace) -> int:
    summary = feedback_summary(
        feedback_path=args.feedback,
        events_path=args.events,
        include_notes=args.include_notes,
        max_rows=args.max_rows,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_feedback_summary(summary, include_notes=args.include_notes)
    return 0


def _cmd_telemetry_review(args: argparse.Namespace) -> int:
    summary = review_queue(
        events_path=args.events,
        feedback_path=args.feedback,
        pricing_catalog_path=args.pricing_catalog,
        max_rows=args.max_rows,
    )
    _attach_pricing_override_skeleton(summary, enabled=args.pricing_override_skeleton)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_telemetry_review(summary)
    return 0


def _cmd_pricing_status(args: argparse.Namespace) -> int:
    status = pricing_status(args.override)
    if args.json:
        print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    else:
        _print_pricing_status(status)
    return 0 if status.override_valid and not status.validation_errors else 1


def _attach_pricing_override_skeleton(
    summary: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    summary["pricing_override_skeleton"] = pricing_override_skeleton_from_gaps(
        summary.get("catalog_coverage_gaps") or [],
    )


def _cmd_pricing_diff(args: argparse.Namespace) -> int:
    diff = pricing_diff(
        args.override,
        context_lines=args.context,
        max_lines=args.max_lines,
    )
    if args.json:
        print(json.dumps(diff.to_dict(), indent=2, sort_keys=True))
    else:
        _print_pricing_diff(diff)
    return 0 if diff.status.override_valid and not diff.status.validation_errors else 1


def _cmd_pricing_apply(args: argparse.Namespace) -> int:
    confirmed = args.yes
    if not confirmed and not args.json:
        diff = pricing_diff(args.override)
        _print_pricing_diff(diff)
        if diff.has_changes:
            answer = input("Apply packaged pricing metadata locally? [y/N] ").strip().lower()
            confirmed = answer in {"y", "yes"}
        else:
            confirmed = True
    result = apply_pricing_catalog(args.override, confirmed=confirmed)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_pricing_apply_result(result)
    return 0 if result.ok else 1


def _cmd_setup_scan(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    if args.json:
        print(json.dumps(discovery.to_dict(), indent=2, sort_keys=True))
    else:
        _print_discovery(discovery)
    return 0


def _cmd_setup_recommend(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    recommendation = recommend_setup(
        discovery,
        profile=args.profile,
        download_alternatives=args.download_alternatives,
        benchmark_results=load_benchmark_results(args.benchmark_results),
    )
    if args.json:
        print(json.dumps(recommendation.to_dict(), indent=2, sort_keys=True))
    else:
        _print_recommendation(recommendation)
    return 0


def _cmd_setup_download(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    plan = plan_model_downloads(
        discovery=discovery,
        profile=args.profile,
        routes=args.route,
        local_root=args.local_root,
        repo_id=args.repo_id,
        adapter=args.adapter,
        alternatives=args.alternatives,
        benchmark_results=load_benchmark_results(args.benchmark_results),
    )
    confirmed = args.yes
    if args.execute and not confirmed and not args.json:
        _print_download_plan(plan)
        answer = input("Run these hf download commands? [y/N] ").strip().lower()
        confirmed = answer in {"y", "yes"}

    result = execute_download_plan(
        plan,
        execute=args.execute,
        confirmed=confirmed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_download_result(result)
    return 0 if result.ok else 1


def _cmd_setup_benchmark(args: argparse.Namespace) -> int:
    targets = plan_backend_benchmarks(args.config, backends=args.backend)
    confirmed = args.yes
    if args.execute and not confirmed and not args.json:
        _print_benchmark_plan(targets, args.output)
        answer = input("Run these local benchmark requests? [y/N] ").strip().lower()
        confirmed = answer in {"y", "yes"}

    result = execute_benchmark_plan(
        targets,
        output_path=args.output,
        execute=args.execute,
        confirmed=confirmed,
        timeout_seconds=args.timeout,
    )
    if args.json:
        payload = result.to_dict()
        payload["summary"] = benchmark_summary(args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_benchmark_result(result)
    return 0 if result.ok else 1


def _cmd_setup_install_prereqs(args: argparse.Namespace) -> int:
    plan = plan_prereq_installs(preset=args.preset)
    confirmed = args.yes
    if args.execute and not confirmed and not args.json:
        _print_prereq_plan(plan)
        answer = input("Run these pip install commands? [y/N] ").strip().lower()
        confirmed = answer in {"y", "yes"}

    result = execute_prereq_install_plan(
        plan,
        execute=args.execute,
        confirmed=confirmed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_prereq_result(result)
    return 0 if result.ok else 1


def _cmd_setup_write(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    result = write_recommended_config(
        args.output,
        discovery=discovery,
        force=args.force,
        profile=args.profile,
        download_alternatives=args.download_alternatives,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(result.message)
        _print_recommendation(result.recommendation)
    return 0 if result.written else 1


def _cmd_setup_wizard(args: argparse.Namespace) -> int:
    model_dirs = _model_dirs_from_args(args)
    discovery = scan_local_environment(model_dirs=model_dirs)
    recommendation = recommend_setup(discovery, profile=args.profile)
    output = Path(args.output).expanduser()
    output_exists = output.exists()
    current_config = _load_wizard_config(output)
    print("ModelRouter setup wizard")
    print("")
    discovery, recommendation = _maybe_install_hf_cli_for_wizard(
        discovery=discovery,
        recommendation=recommendation,
        model_dirs=model_dirs,
        profile=args.profile,
    )
    _print_discovery(discovery)
    print("")
    mode = _ask_setup_mode()
    known_engines = _known_engine_names()
    selections = _ask_route_targets(
        mode=mode,
        recommendation=recommendation,
        discovery=discovery,
        known_engines=known_engines,
        current_config=current_config,
        use_current_defaults=output_exists,
    )
    wizard_recommendation = _build_wizard_recommendation(
        recommendation=recommendation,
        selections=selections,
        mode=mode,
    )
    print("")
    _print_recommendation(wizard_recommendation)
    print("")
    answer = input(f"Write this config to {args.output}? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("No config written.")
        return 0

    force = args.force
    if output.exists() and not force:
        overwrite_answer = input(
            f"Config already exists: {output}. Overwrite? [y/N] "
        ).strip().lower()
        if overwrite_answer not in {"y", "yes"}:
            print("No config written.")
            return 0
        force = True

    result = write_config_from_recommendation(
        output,
        recommendation=wizard_recommendation,
        force=force,
        base_config_path=output if output_exists and current_config is not None else None,
    )
    print(result.message)
    if not result.written:
        return 1

    if wizard_recommendation.download_suggestions:
        print("")
        download_answer = input(
            "Download selected recommended models now? [y/N] "
        ).strip().lower()
        if download_answer in {"y", "yes"}:
            download_result = execute_download_plan(
                DownloadPlan(
                    suggestions=wizard_recommendation.download_suggestions,
                    notes=("Selected during setup wizard.",),
                ),
                execute=True,
                confirmed=True,
            )
            _print_download_result(download_result)
            return 0 if download_result.ok else 1
        print("Downloads skipped.")

    return 0


def _maybe_install_hf_cli_for_wizard(
    *,
    discovery,
    recommendation: SetupRecommendation,
    model_dirs,
    profile: str,
):
    if discovery.commands.get("hf") or not recommendation.download_suggestions:
        return discovery, recommendation

    print("Hugging Face `hf` CLI is missing.")
    print("Recommended model downloads use `hf download`.")
    answer = input("Install it into this Python environment now? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Skipping hf CLI install for now.")
        print("")
        return discovery, recommendation

    print("Installing Hugging Face `hf` CLI...")
    returncode = _run_hf_cli_install()
    if returncode != 0:
        print(f"hf CLI install failed with return code {returncode}.")
        print("Continuing without hf; downloads can be run after installing it.")
        print("")
        return discovery, recommendation

    print("hf CLI install completed.")
    refreshed = scan_local_environment(model_dirs=model_dirs)
    refreshed_recommendation = recommend_setup(refreshed, profile=profile)
    print("")
    return refreshed, refreshed_recommendation


def _run_hf_cli_install() -> int:
    completed = subprocess.run(HF_CLI_INSTALL_COMMAND, check=False)
    return int(completed.returncode)


def _add_setup_scan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=None,
        help="Additional or replacement local model directory to scan",
    )
    parser.add_argument(
        "--no-default-dirs",
        action="store_true",
        help="Do not scan default local model cache directories",
    )


def _model_dirs_from_args(args: argparse.Namespace):
    if getattr(args, "no_default_dirs", False):
        return args.model_dir or []
    return getattr(args, "model_dir", None)


def _load_wizard_config(output: Path) -> RouterConfig | None:
    try:
        return load_router_config(output if output.exists() else None)
    except RouterConfigError:
        return None


def _print_discovery(discovery) -> None:
    print("Commands:")
    for name, available in sorted(discovery.commands.items()):
        print(f"- {name}: {'available' if available else 'missing'}")
    print("API keys:")
    for name, present in sorted(discovery.env_vars.items()):
        print(f"- {name}: {'present' if present else 'missing'}")
    print("Model directories:")
    for path in discovery.model_dirs:
        print(f"- {path}")
    print("Models:")
    if not discovery.models:
        print("- none discovered")
    for model in discovery.models:
        roles = ", ".join(model.roles) if model.roles else "unclassified"
        print(f"- {model.repo_id} ({model.source}; {roles})")


def _print_recommendation(recommendation) -> None:
    print("Routing targets:")
    for route, engine in sorted(recommendation.routing_targets.items()):
        print(f"- {route}: {engine}")
    print("Engine overrides:")
    if not recommendation.engine_overrides:
        print("- none")
    for engine in sorted(recommendation.engine_overrides):
        print(f"- {engine}")
    print("Local model recommendations:")
    if not recommendation.local_model_recommendations:
        print("- none")
    for item in recommendation.local_model_recommendations[:12]:
        print(
            f"- {item.route}: {item.repo_id} "
            f"({item.score.label}, score {item.score.overall_score})"
        )
        if item.score.warnings:
            print("  warnings: " + ", ".join(item.score.warnings))
    print("Download suggestions:")
    if not recommendation.download_suggestions:
        print("- none")
    for suggestion in recommendation.download_suggestions:
        if suggestion.score is not None:
            print(
                f"- {suggestion.route}: {suggestion.repo_id} "
                f"({suggestion.score.label}, score {suggestion.score.overall_score})"
            )
            if suggestion.score.warnings:
                print("  warnings: " + ", ".join(suggestion.score.warnings))
        else:
            print(f"- {suggestion.route}: {suggestion.repo_id}")
        print(f"  command: {' '.join(suggestion.command)}")
    print("Notes:")
    if not recommendation.notes:
        print("- none")
    for note in recommendation.notes:
        print(f"- {note}")


def _ask_setup_mode() -> str:
    print("Model source mode:")
    print("1. Local LLMs only")
    print("2. API keys / hosted models")
    print("3. Mix of local + API/agent tools")
    answer = input("Choose model source mode [3]: ").strip().lower()
    if answer in {"1", "local", "local llms", "local only"}:
        return "local"
    if answer in {"2", "api", "api keys", "hosted"}:
        return "api"
    return "mixed"


def _ask_route_targets(
    *,
    mode: str,
    recommendation: SetupRecommendation,
    discovery,
    known_engines: set[str],
    current_config: RouterConfig | None,
    use_current_defaults: bool,
) -> WizardSelections:
    print("")
    print("Choose a model or engine for each route. Press Enter to keep the default.")
    print("Type a number from the list, or type any known engine name directly.")
    print("Known engines: " + ", ".join(sorted(known_engines)))
    targets: dict[str, str] = {}
    engine_overrides: dict[str, dict[str, Any]] = {}
    download_suggestions: list[DownloadSuggestion] = []
    for route, label in ROUTE_WIZARD_LABELS:
        default = _wizard_default_engine(
            route=route,
            mode=mode,
            recommendation=recommendation,
            discovery=discovery,
            current_config=current_config,
            use_current_defaults=use_current_defaults,
        )
        choices = _wizard_choices_for_route(
            route=route,
            discovery=discovery,
            recommendation=recommendation,
        )
        print("")
        print(f"{label} ({route})")
        print("  0. " + _keep_engine_label(default, current_config))
        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice.label}")
        note = _wizard_recommendation_note(
            route=route,
            discovery=discovery,
            recommendation=recommendation,
        )
        if note:
            print(f"  {note}")
        answer = input(f"Select model/engine for {route} [0]: ").strip()
        selected = default
        choice = _resolve_wizard_choice(answer, choices, known_engines)
        if isinstance(choice, WizardChoice):
            selected = choice.engine
            if choice.engine_override:
                engine_overrides[selected] = choice.engine_override
            if choice.download_suggestion:
                download_suggestions.append(choice.download_suggestion)
        elif isinstance(choice, str):
            selected = choice
        elif answer and answer != "0":
            print(f"Unknown choice {answer!r}; keeping {default}.")
        targets[route] = selected
    targets["confirmation"] = "human_confirm"
    return WizardSelections(
        routing_targets=targets,
        engine_overrides=engine_overrides,
        download_suggestions=tuple(download_suggestions),
    )


def _wizard_recommendation_note(
    *,
    route: str,
    discovery,
    recommendation: SetupRecommendation,
) -> str:
    local_engine = ROUTE_LOCAL_ENGINES[route]
    has_exact_local_match = any(local_engine in model.roles for model in discovery.models)
    has_download = (
        _download_suggestion_for_route(
            local_engine,
            recommendation.download_suggestions,
        )
        is not None
    )
    has_api_or_agent_choice = bool(_api_and_agent_choices(route, discovery))
    if has_exact_local_match or has_download or has_api_or_agent_choice:
        return ""
    return (
        "No exact recommendation for this route; keep the default or choose a "
        "known compatible engine."
    )


def _wizard_default_engine(
    *,
    route: str,
    mode: str,
    recommendation: SetupRecommendation,
    discovery,
    current_config: RouterConfig | None,
    use_current_defaults: bool,
) -> str:
    if use_current_defaults and current_config is not None:
        current = current_config.target_engine(route)
        if current:
            return current
    if mode == "local":
        return ROUTE_LOCAL_ENGINES[route]
    if mode == "api":
        return _api_default_engine(route, discovery) or ROUTE_LOCAL_ENGINES[route]
    return recommendation.routing_targets.get(route, ROUTE_LOCAL_ENGINES[route])


def _keep_engine_label(engine_name: str, current_config: RouterConfig | None) -> str:
    engine = current_config.get_engine(engine_name) if current_config is not None else None
    if engine is None:
        return f"Keep engine {engine_name}"
    details = _engine_details(engine)
    if not details:
        return f"Keep engine {engine_name}"
    return f"Keep engine {engine_name} ({details})"


def _engine_details(engine: ModelEngine) -> str:
    details = [f"model: {engine.model}", f"adapter: {engine.adapter}"]
    paths = engine.availability.required_paths
    if paths:
        details.append("path: " + paths[0])
    return "; ".join(details)


def _api_default_engine(route: str, discovery) -> str | None:
    if route == "coding":
        if discovery.commands.get("claude"):
            return "claude_code"
        if discovery.commands.get("codex"):
            return "codex"
    if route in {"balanced", "simple"} and discovery.env_vars.get("OPENAI_API_KEY"):
        return "openai_api"
    if route == "reasoning":
        if discovery.env_vars.get("ANTHROPIC_API_KEY"):
            return "anthropic_api"
        if discovery.env_vars.get("OPENAI_API_KEY"):
            return "openai_api"
    return None


def _wizard_choices_for_route(
    *,
    route: str,
    discovery,
    recommendation: SetupRecommendation,
) -> tuple[WizardChoice, ...]:
    local_engine = ROUTE_LOCAL_ENGINES[route]
    choices: list[WizardChoice] = []
    for model in _models_for_route(local_engine, discovery.models):
        roles = ", ".join(model.roles) if model.roles else "unclassified"
        model_label = (
            "Local model" if local_engine in model.roles else "Other local model"
        )
        choices.append(
            WizardChoice(
                label=(
                    f"{model_label} {model.repo_id} "
                    f"({model.source}; {roles}; path: {model.path})"
                ),
                engine=local_engine,
                engine_override=engine_override_for_local_model(local_engine, model),
            )
        )
    suggestion = _download_suggestion_for_route(
        local_engine,
        recommendation.download_suggestions,
    )
    if suggestion is not None:
        choices.append(
            WizardChoice(
                label=(
                    f"Recommended download {suggestion.repo_id} "
                    f"({suggestion.reason}; download offered after save)"
                ),
                engine=local_engine,
                engine_override=engine_override_for_download(suggestion),
                download_suggestion=suggestion,
            )
        )
    choices.extend(_api_and_agent_choices(route, discovery))
    return tuple(choices)


def _models_for_route(
    local_engine: str,
    models: tuple[DiscoveredModel, ...],
) -> tuple[DiscoveredModel, ...]:
    matching = [model for model in models if local_engine in model.roles]
    other = [model for model in models if local_engine not in model.roles]
    return tuple(matching + other)


def _download_suggestion_for_route(
    local_engine: str,
    suggestions: tuple[DownloadSuggestion, ...],
) -> DownloadSuggestion | None:
    for suggestion in suggestions:
        if suggestion.route == local_engine:
            return suggestion
    return None


def _api_and_agent_choices(route: str, discovery) -> tuple[WizardChoice, ...]:
    choices: list[WizardChoice] = []
    if route == "coding":
        if discovery.commands.get("claude"):
            choices.append(WizardChoice(label="Agent tool claude_code", engine="claude_code"))
        if discovery.commands.get("codex"):
            choices.append(WizardChoice(label="Agent tool codex", engine="codex"))
    if discovery.env_vars.get("OPENAI_API_KEY") and route in {
        "simple",
        "balanced",
        "reasoning",
    }:
        choices.append(WizardChoice(label="Hosted API openai_api", engine="openai_api"))
    if discovery.env_vars.get("ANTHROPIC_API_KEY") and route in {
        "balanced",
        "reasoning",
    }:
        choices.append(
            WizardChoice(label="Hosted API anthropic_api", engine="anthropic_api")
        )
    return tuple(choices)


def _resolve_wizard_choice(
    answer: str,
    choices: tuple[WizardChoice, ...],
    known_engines: set[str],
) -> WizardChoice | str | None:
    if not answer or answer == "0":
        return None
    if answer.isdecimal():
        index = int(answer)
        if 1 <= index <= len(choices):
            return choices[index - 1]
        return None
    if answer in known_engines:
        return answer
    lowered = answer.lower()
    matches = [choice for choice in choices if lowered in choice.label.lower()]
    if len(matches) == 1:
        print(f"Matched {answer!r} to {matches[0].label}.")
        return matches[0]
    return None


def _selected_api_overrides(routing_targets: dict[str, str]) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    if "openai_api" in routing_targets.values():
        overrides["openai_api"] = {
            "enabled": True,
            "availability": {
                "status": "auto",
                "required_env": ["OPENAI_API_KEY"],
            },
        }
    if "anthropic_api" in routing_targets.values():
        overrides["anthropic_api"] = {
            "enabled": True,
            "availability": {
                "status": "auto",
                "required_env": ["ANTHROPIC_API_KEY"],
            },
        }
    return overrides


def _selected_command_overrides(routing_targets: dict[str, str]) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    if "claude_code" in routing_targets.values():
        overrides["claude_code"] = {
            "enabled": True,
            "availability": {
                "status": "auto",
                "required_commands": ["claude"],
            },
        }
    if "codex" in routing_targets.values():
        overrides["codex"] = {
            "enabled": True,
            "availability": {
                "status": "auto",
                "required_commands": ["codex"],
            },
        }
    return overrides


def _build_wizard_recommendation(
    *,
    recommendation: SetupRecommendation,
    selections: WizardSelections,
    mode: str,
) -> SetupRecommendation:
    engine_overrides = dict(selections.engine_overrides)
    engine_overrides.update(_selected_command_overrides(selections.routing_targets))
    engine_overrides.update(_selected_api_overrides(selections.routing_targets))
    return SetupRecommendation(
        routing_targets=selections.routing_targets,
        engine_overrides=engine_overrides,
        download_suggestions=selections.download_suggestions,
        notes=_wizard_notes(
            recommendation=recommendation,
            routing_targets=selections.routing_targets,
            mode=mode,
        ),
    )


def _wizard_notes(
    *,
    recommendation: SetupRecommendation,
    routing_targets: dict[str, str],
    mode: str,
) -> tuple[str, ...]:
    notes: list[str] = []
    for note in recommendation.notes:
        if (
            "coding route set to claude_code" in note
            and routing_targets.get("coding") != "claude_code"
        ):
            continue
        if (
            "coding route set to codex" in note
            and routing_targets.get("coding") != "codex"
        ):
            continue
        notes.append(note)
    notes.append(f"Wizard mode selected: {mode}.")
    notes.append("Route selections were confirmed interactively.")
    return tuple(notes)


def _known_engine_names() -> set[str]:
    try:
        return set(load_router_config().engines)
    except RouterConfigError:
        return {
            "fast_local",
            "balanced_local",
            "reasoning_local",
            "code_agent",
            "claude_code",
            "codex",
            "openai_api",
            "anthropic_api",
            "web_research",
            "multimodal_vision",
            "image_generation",
            "human_confirm",
        }


def _print_download_plan(plan) -> None:
    print("Download plan:")
    if not plan.suggestions:
        print("- none")
    for suggestion in plan.suggestions:
        print(f"- {suggestion.route}: {suggestion.repo_id}")
        print(f"  command: {' '.join(suggestion.command)}")
    print("Notes:")
    if not plan.notes:
        print("- none")
    for note in plan.notes:
        print(f"- {note}")


def _print_benchmark_plan(targets, output_path: Path) -> None:
    print("Benchmark plan:")
    if not targets:
        print("- none")
    for target in targets:
        managed = "managed runtime" if target.managed_runtime else "unmanaged"
        print(f"- {target.backend}: {target.model} ({target.route}; {managed})")
        print(f"  base_url: {target.base_url}")
    print(f"Output: {output_path}")
    print("Prompt: fixed synthetic smoke prompt; prompt body is not stored.")


def _print_benchmark_result(result) -> None:
    print(f"Executed: {str(result.executed).lower()}")
    print(f"OK: {str(result.ok).lower()}")
    print(f"Output: {result.output_path}")
    print("Results:")
    if not result.results:
        print("- none")
    for item in result.results:
        print(f"- {item.backend}: {item.status}")
        print(f"  model: {item.model}")
        if item.tokens_per_second is not None:
            print(f"  tokens/sec: {item.tokens_per_second}")
        if item.total_latency_ms is not None:
            print(f"  total latency ms: {item.total_latency_ms}")
        if item.error:
            print(f"  error: {item.error}")
    print("Notes:")
    if not result.notes:
        print("- none")
    for note in result.notes:
        print(f"- {note}")


def _print_workflow_benchmark_report(report) -> None:
    print("Workflow Routing Benchmark")
    print(f"OK: {str(report.ok).lower()}")
    print(
        "Summary: "
        f"{report.passed}/{report.total} passed; "
        f"{report.route_changes} route changes"
    )
    print("Results:")
    if not report.results:
        print("- none")
    for item in report.results:
        status = "pass" if item.passed else "fail"
        print(
            f"- {item.name}: {status} "
            f"({item.expected_engine} -> {item.selected_engine}; "
            f"{item.route_latency_us} us)"
        )
        print(f"  profile: {item.routing_profile}")
        if item.reason_codes:
            print(f"  codes: {', '.join(item.reason_codes[:8])}")
        for reason in item.failure_reasons:
            print(f"  failure: {reason}")
    print("Notes:")
    if not report.notes:
        print("- none")
    for note in report.notes:
        print(f"- {note}")


def _print_eval_fixture_list(payload: Mapping[str, Any]) -> None:
    print("ModelRouter Evals")
    fixtures = payload.get("fixtures") if isinstance(payload.get("fixtures"), list) else []
    print(f"Fixtures: {len(fixtures)}")
    if not fixtures:
        print("- none")
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            continue
        print(
            "- "
            f"{fixture.get('id')}: {fixture.get('category')} "
            f"({fixture.get('task_profile')})"
        )
    print(f"Privacy: {payload.get('privacy')}")


def _print_eval_run_execution(execution) -> None:
    print("ModelRouter Eval Run")
    print(f"Run id: {execution.run_id}")
    print(f"Created: {execution.created_at}")
    print(f"Backend: {execution.backend}")
    print(f"Model: {execution.model}")
    print(f"Output: {execution.output_path}")
    print(f"OK: {str(execution.ok).lower()}")
    print(f"Passed: {str(execution.passed).lower()}")
    print("Results:")
    if not execution.results:
        print("- none")
    for result in execution.results:
        print(
            f"- {result.fixture_id}: {result.exit_status} "
            f"score={result.score_percent}% status={result.status}"
        )
        if result.latency_ms is not None:
            print(f"  latency ms: {result.latency_ms}")
        if result.failure_reasons:
            print(f"  failures: {'; '.join(result.failure_reasons[:3])}")
    print("Notes:")
    if not execution.notes:
        print("- none")
    for note in execution.notes:
        print(f"- {note}")


def _print_eval_report(report) -> None:
    print("ModelRouter Eval Report")
    print(f"Run id: {report.run_id or 'none'}")
    print(f"Results: {report.result_path}")
    print(f"Backend: {report.backend or 'unknown'}")
    print(f"Model: {report.model or 'unknown'}")
    print(f"Selected engine: {report.selected_engine or 'unknown'}")
    print(
        "Summary: "
        f"total={report.total}, completed={report.completed}, "
        f"passed={report.passed}, failed={report.failed}, "
        f"timeouts={report.timeouts}, unknown={report.unknown}, "
        f"mean_score={report.score_mean_percent if report.score_mean_percent is not None else 'n/a'}, "
        f"weighted={report.weighted_score_mean if report.weighted_score_mean is not None else 'n/a'}"
    )
    latency = report.latency_summary
    print(
        "Latency: "
        f"count={latency.get('count', 0)} "
        f"missing={latency.get('missing', 0)} "
        f"mean_ms={latency.get('mean_ms') or 'n/a'} "
        f"median_ms={latency.get('median_ms') or 'n/a'} "
        f"min_ms={latency.get('min_ms') or 'n/a'} "
        f"max_ms={latency.get('max_ms') or 'n/a'}"
    )
    usage = report.usage_summary
    print(
        "Usage: "
        f"rows={usage.get('rows_with_usage', 0)} "
        f"missing={usage.get('rows_missing_usage', 0)} "
        f"prompt={usage.get('usage_prompt_tokens', 0)} "
        f"completion={usage.get('usage_completion_tokens', 0)} "
        f"total={usage.get('usage_total_tokens', 0)}"
    )
    print("By category:")
    if not report.by_category:
        print("- none")
    for category, group in report.by_category.items():
        print(
            f"- {category}: total={group.get('total', 0)} "
            f"passed={group.get('passed', 0)} failed={group.get('failed', 0)} "
            f"timeouts={group.get('timeouts', 0)} "
            f"mean={group.get('score_mean_percent')} "
            f"weighted={group.get('weighted_score_mean')}"
        )
    print("Top failure reasons:")
    if not report.top_failure_reasons:
        print("- none")
    for reason in report.top_failure_reasons:
        print(f"- {reason.get('reason')}: {reason.get('count')}")
    print("Suitability notes:")
    if not report.suitability_notes:
        print("- none")
    for note in report.suitability_notes:
        print(f"- {note}")
    privacy = report.privacy
    print(
        "Privacy: "
        f"prompt={privacy.get('prompt_retention', 'unknown')} "
        f"output={privacy.get('output_retention', 'unknown')} "
        f"artifacts={privacy.get('artifact_retention', 'unknown')}"
    )
    print("Results:")
    if not report.results:
        print("- none")
    for row in report.results:
        print(
            f"- {row.get('fixture_id')}: {row.get('exit_status')} "
            f"score={row.get('score_percent')} status={row.get('status')}"
        )
    print("Notes:")
    if not report.notes:
        print("- none")
    for note in report.notes:
        print(f"- {note}")


def _print_eval_evidence(evidence: Mapping[str, Any]) -> None:
    print("ModelRouter Eval Evidence")
    print(f"Model: {evidence.get('model') or 'unknown'}")
    print(f"Backend: {evidence.get('backend') or 'any'}")
    print(f"Status: {evidence.get('status') or 'unknown'}")
    print(f"Latest run: {evidence.get('latest_run_id') or 'none'}")
    print(f"Last evaluated: {evidence.get('last_evaluated_at') or 'never'}")
    print(
        "Summary: "
        f"fixtures={evidence.get('fixture_count', 0)} "
        f"passed={evidence.get('passed', 0)} "
        f"failed={evidence.get('failed', 0)} "
        f"timeouts={evidence.get('timeouts', 0)} "
        f"mean_score={evidence.get('score_mean_percent') or 'n/a'} "
        f"weighted={evidence.get('weighted_score_mean') or 'n/a'}"
    )
    print(f"Stale: {str(bool(evidence.get('stale'))).lower()}")
    stale_reasons = (
        evidence.get("stale_reasons")
        if isinstance(evidence.get("stale_reasons"), list)
        else []
    )
    if stale_reasons:
        print("Stale reasons:")
        for reason in stale_reasons:
            print(f"- {reason}")
    categories = (
        evidence.get("by_category")
        if isinstance(evidence.get("by_category"), Mapping)
        else {}
    )
    print("By category:")
    if not categories:
        print("- not evaluated")
    for category, group in categories.items():
        if not isinstance(group, Mapping):
            continue
        print(
            f"- {category}: total={group.get('total', 0)} "
            f"passed={group.get('passed', 0)} failed={group.get('failed', 0)} "
            f"mean={group.get('score_mean_percent')}"
        )
    failures = (
        evidence.get("top_failure_reasons")
        if isinstance(evidence.get("top_failure_reasons"), list)
        else []
    )
    print("Top failure reasons:")
    if not failures:
        print("- none")
    for reason in failures:
        if isinstance(reason, Mapping):
            print(f"- {reason.get('reason')}: {reason.get('count')}")
    print("Notes:")
    notes = evidence.get("notes") if isinstance(evidence.get("notes"), list) else []
    if not notes:
        print("- Eval evidence is advisory and does not change routing automatically.")
    for note in notes:
        print(f"- {note}")


def _print_catalog_status(status) -> None:
    local_state = (
        "missing"
        if not status.local_exists
        else ("matches packaged" if status.local_matches_packaged else "customized")
    )
    print("Catalog Status")
    print(f"Packaged model catalog: v{status.packaged_model_catalog_version}")
    print(f"Packaged router config: {status.packaged_router_config_source}")
    print(f"Local config: {status.local_config} ({local_state})")
    print(f"Migration log: {status.migration_log}")
    print(f"Remote checks: {str(status.remote_checks_enabled).lower()}")
    if status.last_applied_model_catalog_version is not None:
        print(f"Last applied catalog: v{status.last_applied_model_catalog_version}")
    print("Overrides:")
    if not status.overrides:
        print("- none")
    for item in status.overrides:
        print(f"- {item}")
    print("Notes:")
    for note in status.notes:
        print(f"- {note}")


def _print_catalog_diff(diff) -> None:
    print("Catalog Diff")
    print(f"Action: {diff.action}")
    print(f"Changes: {str(diff.has_changes).lower()}")
    if diff.diff_lines:
        print("Diff:")
        for line in diff.diff_lines:
            print(line)
    else:
        print("Diff: none")
    print("Notes:")
    if not diff.notes:
        print("- none")
    for note in diff.notes:
        print(f"- {note}")


def _print_catalog_apply_result(result) -> None:
    print("Catalog Apply")
    print(f"OK: {str(result.ok).lower()}")
    print(f"Executed: {str(result.executed).lower()}")
    print(f"Action: {result.action}")
    print(f"Config: {result.config_path}")
    if result.backup_path:
        print(f"Backup: {result.backup_path}")
    print(f"Migration log: {result.migration_log}")
    print("Notes:")
    if not result.notes:
        print("- none")
    for note in result.notes:
        print(f"- {note}")


def _print_pricing_status(status) -> None:
    override_state = (
        "missing"
        if not status.override_exists
        else ("valid" if status.override_valid else "invalid")
    )
    print("Pricing Catalog Status")
    print(f"Packaged pricing catalog: v{status.packaged_catalog_version}")
    print(f"Packaged entries: {status.packaged_entry_count}")
    print(f"Override: {status.override_path} ({override_state})")
    if status.override_catalog_version is not None:
        print(f"Override catalog: v{status.override_catalog_version}")
    print(f"Active catalog: v{status.active_catalog_version}")
    print(f"Active source: {status.active_catalog_source}")
    print(f"Active entries: {status.active_entry_count}")
    print(f"Remote checks: {str(status.remote_checks_enabled).lower()}")
    print(f"Validation state: {status.validation_state}")
    print("Warnings:")
    if not status.warnings:
        print("- none")
    for warning in status.warnings:
        print(f"- {warning}")
    print("Validation:")
    if not status.validation_errors:
        print("- none")
    for error in status.validation_errors:
        print(f"- {error}")
    print("Notes:")
    for note in status.notes:
        print(f"- {note}")


def _print_pricing_diff(diff) -> None:
    print("Pricing Catalog Diff")
    print(f"Action: {diff.action}")
    print(f"Changes: {str(diff.has_changes).lower()}")
    print(f"Override: {diff.status.override_path}")
    if diff.diff_lines:
        print("Diff:")
        for line in diff.diff_lines:
            print(line)
    else:
        print("Diff: none")
    print("Notes:")
    if not diff.notes:
        print("- none")
    for note in diff.notes:
        print(f"- {note}")


def _print_pricing_apply_result(result) -> None:
    print("Pricing Catalog Apply")
    print(f"OK: {str(result.ok).lower()}")
    print(f"Executed: {str(result.executed).lower()}")
    print(f"Action: {result.action}")
    print(f"Override: {result.override_path}")
    if result.backup_path:
        print(f"Backup: {result.backup_path}")
    print("Notes:")
    if not result.notes:
        print("- none")
    for note in result.notes:
        print(f"- {note}")


def _print_runtime_action_result(result: Mapping[str, Any]) -> None:
    body = result.get("payload")
    body = body if isinstance(body, Mapping) else {}
    print("Runtime Action")
    print(f"Action: {result.get('action_id', 'unknown')}")
    print(f"OK: {str(body.get('ok', result.get('ok', False))).lower()}")
    if body.get("backend"):
        print(f"Backend: {body['backend']}")
    if body.get("status"):
        print(f"Status: {body['status']}")
    if body.get("disabled_reason"):
        print(f"Disabled: {body['disabled_reason']}")
    runtime = body.get("runtime")
    if isinstance(runtime, Mapping):
        print(f"Runtime id: {runtime.get('runtime_id', runtime.get('provider', 'unknown'))}")
        print(f"Provider: {runtime.get('provider', 'unknown')}")
        print(f"Runtime kind: {runtime.get('runtime_kind', 'unknown')}")
        if runtime.get("runtime_mode"):
            print(f"Runtime mode: {runtime['runtime_mode']}")
        if "detected" in runtime:
            print(f"Detected: {str(runtime.get('detected')).lower()}")
        if runtime.get("endpoint"):
            print(f"Endpoint: {runtime['endpoint']}")
        health = runtime.get("health")
        if isinstance(health, Mapping):
            print(f"Health: {health.get('status', 'unknown')} ({health.get('detail', '')})")
        if runtime.get("missing_dependency"):
            print(f"Missing dependency: {runtime['missing_dependency']}")
        if runtime.get("install_hint"):
            print(f"Install hint: {runtime['install_hint']}")
        if runtime.get("last_checked_at"):
            print(f"Last checked: {runtime['last_checked_at']}")
    backends = body.get("backends")
    if isinstance(backends, list):
        print("Backends:")
        for item in backends:
            if not isinstance(item, Mapping):
                continue
            health = item.get("health") if isinstance(item.get("health"), Mapping) else {}
            detected = item.get("detected")
            suffix = f", detected={str(detected).lower()}" if detected is not None else ""
            print(
                f"- {item.get('backend', item.get('provider', 'unknown'))}: "
                f"{health.get('status', 'unknown')}{suffix}"
            )
    models = body.get("models") or body.get("loaded_models")
    if isinstance(models, list):
        print("Models:")
        if not models:
            print("- none")
        for item in models:
            if isinstance(item, Mapping):
                loaded = item.get("loaded")
                suffix = "" if loaded is None else f" loaded={str(loaded).lower()}"
                print(f"- {item.get('model_id', 'unknown')}{suffix}")
    action_result = body.get("result")
    if isinstance(action_result, Mapping):
        print(f"Result: {action_result.get('status', 'unknown')}")
        if body.get("process_owner") or body.get("runtime_mode"):
            print(
                "Owner: "
                f"{body.get('process_owner', 'unknown')} "
                f"({body.get('runtime_mode', 'unknown')})"
            )
        if action_result.get("message"):
            print(f"Message: {action_result['message']}")


def _print_runtimes_status(report: Mapping[str, Any]) -> None:
    print("Runtime Status")
    print(f"Config: {report.get('config_path')}")
    print(f"Ready: {report.get('ready_count', 0)}/{report.get('runtime_count', 0)}")
    print(f"Imported models: {report.get('imported_model_count', 0)}")
    print("Runtimes:")
    runtimes = report.get("runtimes") if isinstance(report.get("runtimes"), list) else []
    if not runtimes:
        print("- none")
    for item in runtimes:
        if not isinstance(item, Mapping):
            continue
        detected = item.get("detected")
        detected_text = "unknown" if detected is None else str(detected).lower()
        print(
            f"- {item.get('backend')}: {item.get('runtime_id')} "
            f"health={item.get('health_status', 'unknown')} "
            f"detected={detected_text}"
        )
        if item.get("endpoint"):
            print(f"  endpoint: {item['endpoint']}")
        if item.get("install_hint"):
            print(f"  install hint: {item['install_hint']}")
    imported = (
        report.get("imported_models")
        if isinstance(report.get("imported_models"), list)
        else []
    )
    if imported:
        print("Imported Models:")
        for item in imported[:20]:
            if not isinstance(item, Mapping):
                continue
            suffix = ""
            if item.get("load_state") not in {None, "unknown"}:
                suffix = f" ({item.get('load_state')})"
            print(
                f"- {item.get('backend')}: {item.get('model_id')}{suffix}"
            )
    _print_notes(report)


def _print_runtimes_doctor(report: Mapping[str, Any]) -> None:
    _print_runtimes_status(report)
    print("Guidance:")
    guidance = report.get("guidance") if isinstance(report.get("guidance"), list) else []
    if not guidance:
        print("- none")
    for item in guidance:
        if not isinstance(item, Mapping):
            continue
        print(
            f"- {item.get('backend')}: {item.get('next_action')} "
            f"({item.get('runtime_id')})"
        )
        if item.get("message"):
            print(f"  {item['message']}")


def _print_runtimes_connect_plan(plan: Mapping[str, Any]) -> None:
    print("Runtime Connect Plan")
    print(f"Runtime: {plan.get('runtime_id')}")
    print(f"Backend: {plan.get('backend')}")
    print(f"Endpoint: {plan.get('endpoint')}")
    print(f"Model: {plan.get('model')}")
    print(f"Dry run: {str(plan.get('dry_run', True)).lower()}")
    print(f"Config written: {str(plan.get('config_written', False)).lower()}")
    if plan.get("backup_path"):
        print(f"Backup: {plan['backup_path']}")
    health = plan.get("health") if isinstance(plan.get("health"), Mapping) else {}
    if health:
        print(
            "Health: "
            f"{health.get('health_status', 'unknown')} "
            f"detected={str(health.get('detected')).lower()}"
        )
        if health.get("install_hint"):
            print(f"Install hint: {health['install_hint']}")
    if plan.get("error"):
        print(f"Error: {plan['error']}")
    warnings = plan.get("warnings") if isinstance(plan.get("warnings"), list) else []
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("Config patch preview:")
    print(str(plan.get("config_diff") or "none").rstrip())
    print("Guidance:")
    guidance = plan.get("guidance") if isinstance(plan.get("guidance"), list) else []
    if not guidance:
        print("- none")
    for item in guidance:
        print(f"- {item}")
    print("Actions:")
    actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    if not actions:
        print("- none")
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        print(f"- {action.get('label')} ({action.get('kind')})")
        if action.get("preview"):
            print(f"  {action['preview']}")
        elif action.get("url"):
            print(f"  {action['url']}")
    _print_notes(plan)


def _print_notes(payload: Mapping[str, Any]) -> None:
    notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


def _print_dogfood_report(report) -> None:
    print("Proxy Dogfood")
    print(f"Executed: {str(report.executed).lower()}")
    print(f"OK: {str(report.ok).lower()}")
    print(f"Endpoint: {report.endpoint}")
    print(f"Config: {report.config_path}")
    print(
        "Summary: "
        f"{report.passed} passed; {report.failed} failed; "
        f"{report.skipped} skipped; {report.planned} planned"
    )
    print("Checks:")
    if not report.checks:
        print("- none")
    for check in report.checks:
        suffix = (
            f" (HTTP {check.status_code})"
            if check.status_code is not None
            else ""
        )
        print(f"- {check.name}: {check.status}{suffix}")
        print(f"  {check.detail}")
    print("Notes:")
    if not report.notes:
        print("- none")
    for note in report.notes:
        print(f"- {note}")


def _print_download_result(result) -> None:
    print(f"Executed: {str(result.executed).lower()}")
    print(f"OK: {str(result.ok).lower()}")
    print("Results:")
    if not result.results:
        print("- none")
    for item in result.results:
        print(f"- {item.route}: {item.status}")
        print(f"  repo: {item.repo_id}")
        print(f"  command: {' '.join(item.command)}")
        if item.returncode is not None:
            print(f"  returncode: {item.returncode}")
    print("Notes:")
    if not result.notes:
        print("- none")
    for note in result.notes:
        print(f"- {note}")


def _print_prereq_plan(plan) -> None:
    print("Prerequisite install plan:")
    if not plan.steps:
        print("- none")
    for step in plan.steps:
        print(f"- {step.name}: {' '.join(step.command)}")
        print(f"  reason: {step.reason}")
    print("Notes:")
    if not plan.notes:
        print("- none")
    for note in plan.notes:
        print(f"- {note}")


def _print_prereq_result(result) -> None:
    print(f"Executed: {str(result.executed).lower()}")
    print(f"OK: {str(result.ok).lower()}")
    print("Results:")
    if not result.statuses:
        print("- none")
    for item in result.statuses:
        print(f"- {item.route}: {item.status}")
        print(f"  package: {item.repo_id}")
        print(f"  command: {' '.join(item.command)}")
        if item.returncode is not None:
            print(f"  returncode: {item.returncode}")
    print("Notes:")
    if not result.notes:
        print("- none")
    for note in result.notes:
        print(f"- {note}")


def _print_install_plan(plan: dict[str, Any]) -> None:
    installer = plan.get("installer") if isinstance(plan.get("installer"), dict) else {}
    print("ModelRouter installer plan")
    print(f"Config dir: {plan.get('config_dir')}")
    print(f"Selected preset: {plan.get('selected_preset')} ({plan.get('preset_reason')})")
    print(f"Package version: {installer.get('package_version') or 'unknown'}")
    print(f"Install method: {installer.get('install_method') or 'unknown'}")
    print(f"Python: {installer.get('python_version') or 'unknown'}")
    print("")
    print("Detected:")
    runtimes = installer.get("detected_runtimes")
    if isinstance(runtimes, dict) and runtimes:
        for key, value in sorted(runtimes.items()):
            print(f"- {key}: {value}")
    else:
        print("- none")
    print("")
    if plan.get("existing_config"):
        print("Existing config detected; no overwrite is planned.")
    else:
        print("No routing_proxy.yaml detected; init is planned as an explicit command.")
    warnings = plan.get("warnings") if isinstance(plan.get("warnings"), list) else []
    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("")
    print("Next commands:")
    for command in plan.get("next_commands", []):
        if not isinstance(command, dict):
            continue
        status = "" if command.get("available", True) else " (not available)"
        command_text = " ".join(str(part) for part in command.get("command", []))
        print(f"- {command.get('label')}{status}: {command_text}")
        if command.get("reason"):
            print(f"  {command['reason']}")
    notes = plan.get("notes") if isinstance(plan.get("notes"), list) else []
    if notes:
        print("")
        print("Notes:")
        for note in notes:
            print(f"- {note}")


def _print_dispatch_plan(plan) -> None:
    print(f"Selected engine: {plan.selected_engine}")
    print(f"Provider: {plan.provider}")
    print(f"Model: {plan.model}")
    print(f"Adapter: {plan.adapter}")
    print(f"Dry run: {str(plan.dry_run).lower()}")
    print(f"Can dispatch: {str(plan.can_dispatch).lower()}")
    print(f"Blocked: {str(plan.blocked).lower()}")
    print(f"Requires confirmation: {str(plan.requires_confirmation).lower()}")
    print("Reasons:")
    for reason in plan.reasons:
        print(f"- {reason}")


def _print_telemetry_summary(summary: dict[str, Any]) -> None:
    print("Routing telemetry summary")
    print(f"Events: {summary['events']}")
    print(f"Routing events: {summary['routing_events']}")
    print(f"Replayable events: {summary['replayed']}")
    print(f"Skipped without full prompt: {summary['skipped_no_prompt']}")
    print(f"Feedback labels: {summary['feedback_labels']}")
    print(f"Labeled replayable events: {summary['labeled_replayable']}")
    print(f"Unlabeled replayable events: {summary['unlabeled_replayable']}")
    print(f"Route changes: {summary['route_change_count']}")
    print(f"Expected mismatches: {summary['expected_mismatch_count']}")
    print(f"Replay mean latency: {summary['replay_route_latency_mean_ms']} ms")
    print(f"Usage events: {summary.get('usage_events', 0)}")
    print(
        "Usage tokens: "
        f"prompt={summary.get('usage_prompt_tokens', 0)}, "
        f"completion={summary.get('usage_completion_tokens', 0)}, "
        f"total={summary.get('usage_total_tokens', 0)}, "
        f"cached_input={summary.get('usage_cached_input_tokens', 0)}"
    )
    print(f"Estimated cost events: {summary.get('estimated_cost_events', 0)}")
    cost = _format_cost_summary(summary)
    print(f"Estimated cost: {cost or 'none'}")
    print(f"Pricing catalog: {summary.get('pricing_catalog_source', 'unknown')}")
    print(f"Catalog coverage: {_format_catalog_coverage(summary.get('catalog_coverage'))}")
    _print_catalog_coverage_gaps(summary.get("catalog_coverage_gaps"))
    _print_pricing_override_skeleton(summary.get("pricing_override_skeleton"))
    _print_counter("Outcome labels", summary.get("outcome_label_counts", {}))
    _print_counter("Pricing matches", summary.get("pricing_match_counts", {}))
    _print_counter("Selected engines", summary["selected_engine_counts"])
    _print_counter("Statuses", summary["status_counts"])
    _print_usage_groups("Usage by route", summary.get("usage_by_selected_engine"))
    _print_usage_groups("Usage by backend", summary.get("usage_by_backend"))
    _print_usage_groups("Usage by model", summary.get("usage_by_model"))
    _print_counter("Mismatch groups", summary["mismatch_groups"])
    _print_id_list(
        "Unlabeled replayable request ids",
        summary["unlabeled_replayable_request_ids"],
    )
    _print_id_list(
        "Skipped private/no-prompt request ids",
        summary["skipped_no_prompt_request_ids"],
    )
    _print_id_list(
        "Feedback labels without matching events",
        summary["feedback_without_event_request_ids"],
    )
    _print_id_list(
        "Feedback labels for private/no-prompt events",
        summary["feedback_for_private_event_request_ids"],
    )


def _print_feedback_summary(summary: dict[str, Any], *, include_notes: bool) -> None:
    print(f"Feedback labels: {summary['feedback_labels']}")
    _print_counter("Expected engines", summary["expected_engine_counts"])
    _print_counter("Outcome labels", summary.get("outcome_label_counts", {}))
    print("Labels:")
    if not summary["labels"]:
        print("- none")
    for label in summary["labels"]:
        found = "yes" if label["event_found"] else "no"
        replayable = "yes" if label["replayable"] else "no"
        historical = label.get("historical_engine") or "unknown"
        outcome = label.get("outcome_label") or "unlabeled"
        print(
            f"- {label['request_id']}: {label['expected_engine']} "
            f"(outcome: {outcome}, event: {found}, replayable: {replayable}, "
            f"historical: {historical})"
        )
        if include_notes and label.get("notes"):
            print(f"  notes: {label['notes']}")
    if summary["truncated"]:
        print("- output truncated; increase --max-rows to show more")


def _print_telemetry_review(summary: dict[str, Any]) -> None:
    print("Telemetry review queue")
    print(f"Reviewable: {summary['reviewable']}")
    print(f"Skipped labeled: {summary['skipped_labeled']}")
    print(f"Skipped private/no-prompt: {summary['skipped_private']}")
    print(f"Catalog coverage: {_format_catalog_coverage(summary.get('catalog_coverage'))}")
    _print_catalog_coverage_gaps(summary.get("catalog_coverage_gaps"))
    _print_pricing_override_skeleton(summary.get("pricing_override_skeleton"))
    print(f"Privacy: {summary['privacy']}")
    print("Items:")
    if not summary["items"]:
        print("- none")
    for item in summary["items"]:
        reason_codes = ", ".join(item.get("reason_codes") or []) or "none"
        usage = _format_usage_summary(item.get("usage"))
        print(
            f"- {item['request_id']}: selected={item.get('selected_engine') or 'unknown'} "
            f"status={item.get('status') or 'unknown'} replayable={str(item.get('replayable')).lower()}"
        )
        if item.get("receipt_summary"):
            print(f"  summary: {item['receipt_summary']}")
        if usage:
            print(f"  usage: {usage}")
        cost = _format_cost_summary(item.get("cost"))
        if cost:
            print(f"  cost: {cost}")
        elif isinstance(item.get("cost"), dict):
            print(f"  cost: {item['cost'].get('pricing_match_status', 'unknown')}")
        print(f"  reason codes: {reason_codes}")
        print(f"  feedback: {item['suggested_feedback_command']}")
    if summary["truncated"]:
        print("- output truncated; increase --max-rows to show more")


def _print_counter(title: str, values: dict[str, int]) -> None:
    print(f"{title}:")
    if not values:
        print("- none")
    for key, value in values.items():
        print(f"- {key}: {value}")


def _print_usage_groups(title: str, values: Any) -> None:
    print(f"{title}:")
    if not isinstance(values, dict) or not values:
        print("- none")
        return
    for key, usage in values.items():
        formatted = _format_usage_summary(usage)
        cost = _format_cost_summary(usage)
        suffix = f"; cost={cost}" if cost else ""
        print(f"- {key}: {formatted or 'no usage'}{suffix}")


def _print_catalog_coverage_gaps(values: Any) -> None:
    print("Catalog coverage gaps:")
    if not isinstance(values, list) or not values:
        print("- none")
        return
    for gap in values:
        if not isinstance(gap, dict):
            continue
        usage = _format_usage_summary(gap)
        print(
            "- "
            f"{gap.get('pricing_match_status') or 'unknown'}: "
            f"provider={gap.get('provider') or 'unknown'} "
            f"model={gap.get('model') or 'unknown'} "
            f"backend={gap.get('backend') or 'unknown'} "
            f"backend_model={gap.get('backend_model') or 'unknown'} "
            f"upstream_model={gap.get('upstream_model') or 'unknown'} "
            f"route={gap.get('selected_engine') or 'unknown'} "
            f"events={_safe_usage_int(gap.get('events'))}"
            + (f" usage={usage}" if usage else "")
        )


def _print_pricing_override_skeleton(value: Any) -> None:
    if not isinstance(value, str):
        return
    print("Pricing override skeleton:")
    if not value.strip():
        print("- none")
        return
    print(value.rstrip())


def _format_usage_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    prompt = _safe_usage_int(value.get("usage_prompt_tokens"))
    completion = _safe_usage_int(value.get("usage_completion_tokens"))
    total = _safe_usage_int(value.get("usage_total_tokens"))
    cached = _safe_usage_int(value.get("usage_cached_input_tokens"))
    if prompt == completion == total == cached == 0:
        return ""
    parts = [
        f"prompt={prompt}",
        f"completion={completion}",
        f"total={total}",
    ]
    if cached:
        parts.append(f"cached_input={cached}")
    upstream_model = value.get("upstream_model")
    if isinstance(upstream_model, str) and upstream_model:
        parts.append(f"upstream_model={upstream_model}")
    return ", ".join(parts)


def _format_cost_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    if value.get("pricing_match_status") not in {None, "matched"}:
        return ""
    events = _safe_usage_int(value.get("estimated_cost_events"))
    total = value.get("estimated_total_cost")
    currency = value.get("estimated_cost_currency") or ""
    if not isinstance(total, (int, float)) or total < 0:
        return ""
    if events == 0 and total == 0:
        return ""
    parts = [f"{total:.8f}".rstrip("0").rstrip(".") or "0"]
    if isinstance(currency, str) and currency:
        parts.append(currency)
    if events:
        parts.append(f"events={events}")
    if value.get("pricing_is_placeholder") is True:
        parts.append("placeholder")
    return " ".join(parts)


def _format_catalog_coverage(value: Any) -> str:
    if not isinstance(value, dict):
        return "usage_rows=0 matched=0 missing=0 placeholder=0 estimated=0 no_usage=0"
    parts = [
        f"usage_rows={_safe_usage_int(value.get('total_rows_with_usage'))}",
        f"matched={_safe_usage_int(value.get('rows_with_catalog_match'))}",
        "missing="
        f"{_safe_usage_int(value.get('rows_missing_provider_model_catalog_match'))}",
        f"placeholder={_safe_usage_int(value.get('rows_using_placeholder_pricing'))}",
        f"estimated={_safe_usage_int(value.get('rows_with_estimated_cost'))}",
        f"no_usage={_safe_usage_int(value.get('rows_without_enough_usage_data'))}",
    ]
    version = value.get("active_catalog_version")
    source = value.get("active_catalog_source")
    if isinstance(version, int) and version > 0:
        parts.append(f"catalog=v{version}")
    if isinstance(source, str) and source:
        parts.append(f"source={source}")
    confidence = value.get("cost_confidence")
    if isinstance(confidence, str) and confidence:
        parts.append(f"confidence={confidence}")
    return " ".join(parts)


def _safe_usage_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _print_id_list(title: str, values: list[str]) -> None:
    print(f"{title}:")
    if not values:
        print("- none")
    for value in values:
        print(f"- {value}")


def _print_readable(receipt) -> None:
    print(f"Summary: {receipt.summary}")
    print(f"Selected engine: {receipt.selected_engine}")
    print(f"Routing profile: {receipt.routing_profile}")
    print(f"Fallback engine: {receipt.fallback_engine or 'none'}")
    print(f"Complexity: {receipt.complexity_score}/100")
    print(f"Risk: {receipt.risk_score}/100")
    print(f"Confidence: {receipt.confidence_score}/100")
    print(f"Requires confirmation: {str(receipt.requires_confirmation).lower()}")
    print(f"Requires tools: {str(receipt.requires_tools).lower()}")
    print(f"Requires freshness: {str(receipt.requires_freshness).lower()}")
    print(f"Requires code execution: {str(receipt.requires_code_execution).lower()}")
    print(f"Requires vision: {str(receipt.requires_vision).lower()}")
    print(
        "Requires image generation: "
        f"{str(receipt.requires_image_generation).lower()}"
    )
    print(f"Config valid: {str(receipt.config_valid).lower()}")
    print(f"Availability valid: {str(receipt.availability_valid).lower()}")
    print(f"Fallback used: {str(receipt.fallback_used).lower()}")
    print("Requirements:")
    requirements = receipt.requirements.to_dict()
    for key in sorted(requirements):
        print(f"- {key}: {requirements[key]}")
    print("Rejected engines:")
    if not receipt.rejected_engines:
        print("- none")
    for rejection in receipt.rejected_engines:
        print(f"- {rejection.engine}: {rejection.reason}")
    print("Alternatives:")
    if not receipt.alternatives:
        print("- none")
    for alternative in receipt.alternatives:
        print(f"- {alternative.engine}: rank {alternative.rank_score}/100")
    print("Availability:")
    for reason in receipt.availability_reasons:
        print(f"- {reason}")
    print("Reasons:")
    for reason in receipt.reasons:
        print(f"- {reason}")


def _print_explain(receipt) -> None:
    print("Route Receipt")
    print(f"Summary: {receipt.summary}")
    print(f"Selected: {receipt.selected_engine}")
    print(f"Profile: {receipt.routing_profile}")
    print(f"Reason codes: {', '.join(receipt.reason_codes) or 'none'}")
    print(f"Selected route: {receipt.selected_route_explanation}")
    print(f"Policy: {receipt.policy_explanation}")
    print(f"Rejected: {receipt.rejection_explanation}")
    print(f"Fallback: {receipt.fallback_explanation}")
    print(f"Safety: {receipt.safety_explanation}")
    print(f"Privacy: {receipt.privacy_explanation}")
    print(f"Wrong route: {receipt.wrong_route_next_action}")


if __name__ == "__main__":
    raise SystemExit(main())
