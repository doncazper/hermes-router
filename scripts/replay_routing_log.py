#!/usr/bin/env python3
"""Replay routing JSONL logs against the current router."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def replay_events(
    *,
    events_path: str | Path,
    feedback_path: str | Path | None,
    config_path: str | None,
) -> dict[str, Any]:
    _ensure_repo_on_path()
    from hermes.plugins.model_router import ModelRouter
    from hermes.plugins.model_router.routing_log import read_jsonl

    events = read_jsonl(events_path)
    feedback = _feedback_by_request(read_jsonl(feedback_path) if feedback_path else [])
    router = ModelRouter.from_config(config_path, validate_availability=False)

    route_changes: list[dict[str, str]] = []
    expected_mismatches: list[dict[str, str]] = []
    confusion: Counter[str] = Counter()
    replay_latencies_ms: list[float] = []
    historical_latencies_ms: list[float] = []
    replayed = 0
    skipped_no_prompt = 0

    for event in events:
        if event.get("event_type") != "routing_event":
            continue
        request_id = str(event.get("request_id", ""))
        prompt = event.get("prompt")
        if not isinstance(prompt, str):
            skipped_no_prompt += 1
            continue

        started = perf_counter()
        current_engine = router.route_fast(prompt)
        replay_latency_ms = (perf_counter() - started) * 1000
        replay_latencies_ms.append(replay_latency_ms)
        replayed += 1

        historical_latency = event.get("route_latency_ms")
        if isinstance(historical_latency, (int, float)):
            historical_latencies_ms.append(float(historical_latency))

        historical_engine = str(event.get("selected_engine", ""))
        if current_engine != historical_engine:
            route_changes.append(
                {
                    "request_id": request_id,
                    "historical_engine": historical_engine,
                    "current_engine": current_engine,
                }
            )

        expected_engine = feedback.get(request_id)
        if expected_engine:
            confusion[f"{expected_engine}->{current_engine}"] += 1
            if current_engine != expected_engine:
                expected_mismatches.append(
                    {
                        "request_id": request_id,
                        "expected_engine": expected_engine,
                        "current_engine": current_engine,
                    }
                )

    historical_mean = _mean(historical_latencies_ms)
    replay_mean = _mean(replay_latencies_ms)
    return {
        "events": len(events),
        "feedback_labels": len(feedback),
        "replayed": replayed,
        "skipped_no_prompt": skipped_no_prompt,
        "route_changes": route_changes,
        "route_change_count": len(route_changes),
        "expected_mismatches": expected_mismatches,
        "expected_mismatch_count": len(expected_mismatches),
        "confusion_matrix": dict(sorted(confusion.items())),
        "historical_route_latency_mean_ms": historical_mean,
        "replay_route_latency_mean_ms": replay_mean,
        "route_latency_delta_mean_ms": (
            round(replay_mean - historical_mean, 6)
            if historical_mean is not None and replay_mean is not None
            else None
        ),
    }


def _feedback_by_request(rows: list[dict[str, Any]]) -> dict[str, str]:
    feedback: dict[str, str] = {}
    for row in rows:
        if row.get("event_type") != "routing_feedback":
            continue
        request_id = row.get("request_id")
        expected_engine = row.get("expected_engine")
        if isinstance(request_id, str) and isinstance(expected_engine, str):
            feedback[request_id] = expected_engine
    return feedback


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 6)


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
        print(f"Replayed: {summary['replayed']}")
        print(f"Skipped without full prompt: {summary['skipped_no_prompt']}")
        print(f"Route changes: {summary['route_change_count']}")
        print(f"Expected mismatches: {summary['expected_mismatch_count']}")
        print(f"Replay mean latency: {summary['replay_route_latency_mean_ms']} ms")
    if args.fail_on_regression and summary["expected_mismatch_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
