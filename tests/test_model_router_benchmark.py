import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_route_fast_benchmark_script_emits_parseable_metrics():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_route_fast.py",
            "--iterations",
            "10",
            "--repeat",
            "2",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["iterations"] == 10
    assert payload["repeat"] == 2
    assert payload["route_fast_mean_us"] > 0
    assert payload["route_fast_best_us"] > 0
    assert payload["routes_per_second_best"] > 0
    assert payload["prompts"] >= 1
