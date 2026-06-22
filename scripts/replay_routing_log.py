#!/usr/bin/env python3
"""Replay routing JSONL logs against the current router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_on_path()

from hermes.plugins.model_router.telemetry import replay_events  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay routing-events JSONL against the current route_fast path.",
    )
    parser.add_argument(
        "--events",
        default="~/.model-router/routing-events.jsonl",
        help="Path to routing-events JSONL.",
    )
    parser.add_argument(
        "--feedback",
        default="~/.model-router/routing-feedback.jsonl",
        help="Path to routing-feedback JSONL.",
    )
    parser.add_argument("--config", default=None, help="Optional router config path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any feedback-labeled event now routes incorrectly.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = replay_events(
        events_path=args.events,
        feedback_path=args.feedback,
        config_path=args.config,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Routing log replay")
        print(f"Events: {summary['events']}")
        print(f"Routing events: {summary['routing_events']}")
        print(f"Replayed: {summary['replayed']}")
        print(f"Skipped without full prompt: {summary['skipped_no_prompt']}")
        print(f"Feedback labels: {summary['feedback_labels']}")
        print(f"Labeled replayable: {summary['labeled_replayable']}")
        print(f"Unlabeled replayable: {summary['unlabeled_replayable']}")
        print(f"Route changes: {summary['route_change_count']}")
        print(f"Expected mismatches: {summary['expected_mismatch_count']}")
        print(f"Mismatch groups: {summary['mismatch_groups'] or {}}")
        print(f"Replay mean latency: {summary['replay_route_latency_mean_ms']} ms")
    if args.fail_on_regression and summary["expected_mismatch_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
