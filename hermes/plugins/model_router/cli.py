"""Command line interface for Hermes model-router decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    print(f"Config valid: {str(receipt.config_valid).lower()}")
    print("Reasons:")
    for reason in receipt.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
