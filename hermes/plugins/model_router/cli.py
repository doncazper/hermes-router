"""Command line interface for Hermes model-router decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes.plugins.model_router.availability import validate_router_availability
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.policy import route_prompt
from hermes.plugins.model_router.receipts import decision_to_receipt, receipt_to_json
from hermes.plugins.model_router.setup_assistant import (
    SetupRecommendation,
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
    decide.add_argument("prompt", nargs="+", help="Prompt text to route")
    decide.set_defaults(func=_cmd_decide)

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
    decision = route_prompt(prompt, config_path=args.config)
    receipt = decision_to_receipt(decision)
    if args.json:
        print(receipt_to_json(receipt))
    else:
        _print_readable(receipt)
    return 0


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
    discovery = scan_local_environment(model_dirs=_model_dirs_from_args(args))
    recommendation = recommend_setup(discovery, profile=args.profile)
    print("Hermes model-router setup wizard")
    print("")
    _print_discovery(discovery)
    print("")
    mode = _ask_setup_mode()
    known_engines = _known_engine_names()
    routing_targets = _ask_route_targets(
        mode=mode,
        recommendation=recommendation,
        discovery=discovery,
        known_engines=known_engines,
    )
    wizard_recommendation = _build_wizard_recommendation(
        recommendation=recommendation,
        routing_targets=routing_targets,
        mode=mode,
    )
    print("")
    _print_recommendation(wizard_recommendation)
    print("")
    answer = input(f"Write this config to {args.output}? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("No config written.")
        return 0

    result = write_config_from_recommendation(
        args.output,
        recommendation=wizard_recommendation,
        force=args.force,
    )
    print(result.message)
    return 0 if result.written else 1


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
) -> dict[str, str]:
    print("")
    print("Choose an engine for each route. Press Enter to accept the default.")
    print("You can type any known engine name to override a route.")
    print("Known engines: " + ", ".join(sorted(known_engines)))
    targets: dict[str, str] = {}
    for route, label in ROUTE_WIZARD_LABELS:
        default = _wizard_default_engine(
            route=route,
            mode=mode,
            recommendation=recommendation,
            discovery=discovery,
        )
        answer = input(f"{label} ({route}) [{default}]: ").strip()
        selected = answer or default
        if selected not in known_engines:
            print(f"Unknown engine {selected!r}; keeping {default}.")
            selected = default
        targets[route] = selected
    targets["confirmation"] = "human_confirm"
    return targets


def _wizard_default_engine(
    *,
    route: str,
    mode: str,
    recommendation: SetupRecommendation,
    discovery,
) -> str:
    local_defaults = {
        "simple": "fast_local",
        "balanced": "balanced_local",
        "reasoning": "reasoning_local",
        "coding": "code_agent",
        "research": "web_research",
        "vision": "multimodal_vision",
        "image_generation": "image_generation",
    }
    if mode == "local":
        return local_defaults[route]
    if mode == "api":
        return _api_default_engine(route, discovery) or local_defaults[route]
    return recommendation.routing_targets.get(route, local_defaults[route])


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
    routing_targets: dict[str, str],
    mode: str,
) -> SetupRecommendation:
    selected_engines = set(routing_targets.values())
    engine_overrides = {
        engine_name: patch
        for engine_name, patch in recommendation.engine_overrides.items()
        if engine_name in selected_engines
    }
    engine_overrides.update(_selected_command_overrides(routing_targets))
    engine_overrides.update(_selected_api_overrides(routing_targets))
    download_suggestions = tuple(
        suggestion
        for suggestion in recommendation.download_suggestions
        if suggestion.route in selected_engines
    )
    return SetupRecommendation(
        routing_targets=routing_targets,
        engine_overrides=engine_overrides,
        download_suggestions=download_suggestions,
        notes=_wizard_notes(
            recommendation=recommendation,
            routing_targets=routing_targets,
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
    print("Availability:")
    for reason in receipt.availability_reasons:
        print(f"- {reason}")
    print("Reasons:")
    for reason in receipt.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
