"""Command line interface for Hermes model-router decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes.plugins.model_router.availability import validate_router_availability
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.policy import route_prompt
from hermes.plugins.model_router.receipts import decision_to_receipt, receipt_to_json


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
