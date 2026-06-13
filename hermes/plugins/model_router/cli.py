"""Command line interface for Hermes model-router decisions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from hermes.plugins.model_router.availability import validate_router_availability
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.dispatch import build_dispatch_plan, dispatch_plan_to_json
from hermes.plugins.model_router.models import ModelEngine, RouterConfig
from hermes.plugins.model_router.policy import ModelRouter, route_prompt
from hermes.plugins.model_router.receipts import decision_to_receipt, receipt_to_json
from hermes.plugins.model_router.setup_assistant import (
    DiscoveredModel,
    DownloadPlan,
    DownloadSuggestion,
    SetupRecommendation,
    engine_override_for_download,
    engine_override_for_local_model,
    execute_download_plan,
    plan_model_downloads,
    recommend_setup,
    scan_local_environment,
    write_config_from_recommendation,
    write_recommended_config,
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
        prog="python -m hermes.plugins.model_router.cli",
        description="Decide which Hermes engine category should handle a prompt.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide = subparsers.add_parser("decide", help="Score and route a prompt")
    decide.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON routing receipt",
    )
    decide.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
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
    dispatch.add_argument("prompt", nargs="+", help="Prompt text to plan")
    dispatch.set_defaults(func=_cmd_dispatch_plan)

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
    recommend.set_defaults(func=_cmd_setup_recommend)

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
    download.set_defaults(func=_cmd_setup_download)

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
    else:
        _print_readable(receipt)
    return 0


def _cmd_dispatch_plan(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    hints = _routing_hints_from_args(args)
    plan = build_dispatch_plan(
        prompt,
        config_path=args.config,
        hints=hints,
    )
    if args.json:
        print(dispatch_plan_to_json(plan))
    else:
        _print_dispatch_plan(plan)
    return 0


def _add_routing_hint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model_router.yaml catalog",
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


def _routing_hints_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "force_engine": args.force_engine,
        "attachments": args.attachment or [],
        "max_cost_tier": args.max_cost_tier,
        "max_latency_tier": args.max_latency_tier,
        "latency_sensitive": args.latency_sensitive,
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


def _cmd_setup_scan(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    if args.json:
        print(json.dumps(discovery.to_dict(), indent=2, sort_keys=True))
    else:
        _print_discovery(discovery)
    return 0


def _cmd_setup_recommend(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    recommendation = recommend_setup(discovery, profile=args.profile)
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


def _cmd_setup_write(args: argparse.Namespace) -> int:
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    result = write_recommended_config(
        args.output,
        discovery=discovery,
        force=args.force,
        profile=args.profile,
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
    print("Hermes model-router setup wizard")
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
    if args.no_default_dirs:
        return args.model_dir or []
    return args.model_dir


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
    print("Download suggestions:")
    if not recommendation.download_suggestions:
        print("- none")
    for suggestion in recommendation.download_suggestions:
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


def _print_readable(receipt) -> None:
    print(f"Selected engine: {receipt.selected_engine}")
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


if __name__ == "__main__":
    raise SystemExit(main())
