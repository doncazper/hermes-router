#!/usr/bin/env python3
"""Fail when the initialized route_fast hot path exceeds latency budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_BEST_US = 25.0
DEFAULT_MAX_MEAN_US = 50.0


def _ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a positive float, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"{name} must be a positive float, got {value!r}"
        )
    return parsed


def _env_float_or_error(
    parser: argparse.ArgumentParser,
    name: str,
    default: float,
) -> float:
    try:
        return _env_float(name, default)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark route_fast() and fail if it exceeds latency budgets.",
    )
    parser.add_argument("--config", help="Optional model router YAML config path.")
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=100_000,
        help="Route calls per repeat. Default: 100000.",
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
        "--max-best-us",
        type=_positive_float,
        default=_env_float_or_error(
            parser,
            "ROUTE_FAST_MAX_BEST_US",
            DEFAULT_MAX_BEST_US,
        ),
        help=(
            "Maximum allowed best route_fast latency in microseconds. "
            "Default: 25 or ROUTE_FAST_MAX_BEST_US."
        ),
    )
    parser.add_argument(
        "--max-mean-us",
        type=_positive_float,
        default=_env_float_or_error(
            parser,
            "ROUTE_FAST_MAX_MEAN_US",
            DEFAULT_MAX_MEAN_US,
        ),
        help=(
            "Maximum allowed mean route_fast latency in microseconds. "
            "Default: 50 or ROUTE_FAST_MAX_MEAN_US."
        ),
    )
    parser.add_argument(
        "--validate-availability",
        action="store_true",
        help="Validate configured engine availability during router initialization.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _ensure_repo_on_path()
    from scripts.benchmark_route_fast import DEFAULT_PROMPTS, _benchmark

    metrics = _benchmark(
        config_path=args.config,
        iterations=args.iterations,
        repeat=args.repeat,
        prompts=tuple(args.prompts or DEFAULT_PROMPTS),
        validate_availability=args.validate_availability,
    )
    best_us = float(metrics["route_fast_best_us"])
    mean_us = float(metrics["route_fast_mean_us"])
    failures = []
    if best_us > args.max_best_us:
        failures.append(
            f"best {best_us:.4f} us exceeds budget {args.max_best_us:.4f} us"
        )
    if mean_us > args.max_mean_us:
        failures.append(
            f"mean {mean_us:.4f} us exceeds budget {args.max_mean_us:.4f} us"
        )

    payload = {
        **metrics,
        "max_best_us": args.max_best_us,
        "max_mean_us": args.max_mean_us,
        "passed": not failures,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("ModelRouter route_fast latency check")
        print(f"Best: {best_us:.4f} us/route (budget {args.max_best_us:.4f})")
        print(f"Mean: {mean_us:.4f} us/route (budget {args.max_mean_us:.4f})")
        if failures:
            print("FAILED")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
