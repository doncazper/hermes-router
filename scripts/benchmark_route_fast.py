#!/usr/bin/env python3
"""Benchmark the initialized ModelRouter.route_fast hot path."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = (
    "rewrite this text",
    "fix the repo and run tests",
    "search the web for the latest TypeScript release notes",
    "drop the production database",
    (
        "Design a distributed task scheduler with backpressure and exactly-once "
        "delivery semantics."
    ),
)


def _ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Hermes ModelRouter.route_fast().",
    )
    parser.add_argument(
        "--config",
        help="Optional model router YAML config path.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=300_000,
        help="Route calls per repeat. Default: 300000.",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=5,
        help="Number of benchmark repeats. Default: 5.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to include in the benchmark mix. Can be passed more than once.",
    )
    parser.add_argument(
        "--validate-availability",
        action="store_true",
        help="Validate configured engine availability during router initialization.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print benchmark metrics as JSON.",
    )
    return parser.parse_args()


def _benchmark(
    *,
    config_path: str | None,
    iterations: int,
    repeat: int,
    prompts: tuple[str, ...],
    validate_availability: bool,
) -> dict[str, object]:
    _ensure_repo_on_path()
    from hermes.plugins.model_router import ModelRouter

    router = ModelRouter.from_config(
        config_path,
        validate_availability=validate_availability,
    )
    for prompt in prompts:
        router.route_fast(prompt)

    runs_us: list[float] = []
    prompt_count = len(prompts)
    for _ in range(repeat):
        start = perf_counter()
        for index in range(iterations):
            router.route_fast(prompts[index % prompt_count])
        elapsed = perf_counter() - start
        runs_us.append(elapsed / iterations * 1_000_000)

    mean_us = statistics.mean(runs_us)
    best_us = min(runs_us)
    return {
        "iterations": iterations,
        "repeat": repeat,
        "prompts": prompt_count,
        "validate_availability": validate_availability,
        "route_fast_runs_us": [round(value, 4) for value in runs_us],
        "route_fast_mean_us": round(mean_us, 4),
        "route_fast_best_us": round(best_us, 4),
        "routes_per_second_best": round(1_000_000 / best_us),
    }


def main() -> int:
    args = _parse_args()
    prompts = tuple(args.prompts or DEFAULT_PROMPTS)
    metrics = _benchmark(
        config_path=args.config,
        iterations=args.iterations,
        repeat=args.repeat,
        prompts=prompts,
        validate_availability=args.validate_availability,
    )
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("Hermes route_fast benchmark")
        print(f"Iterations: {metrics['iterations']}")
        print(f"Repeats: {metrics['repeat']}")
        print(f"Prompts: {metrics['prompts']}")
        print(f"Mean: {metrics['route_fast_mean_us']} us/route")
        print(f"Best: {metrics['route_fast_best_us']} us/route")
        print(f"Best throughput: {metrics['routes_per_second_best']} routes/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
