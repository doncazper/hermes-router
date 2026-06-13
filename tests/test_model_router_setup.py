import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

from hermes.plugins.model_router.setup_assistant import (
    recommend_setup,
    scan_local_environment,
    write_recommended_config,
)
from hermes.plugins.model_router.config import load_router_config


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_cli_with_input(
    *args: str,
    user_input: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        input=user_input,
        capture_output=True,
        check=False,
    )


def test_scan_detects_hugging_face_cache_models_and_commands(tmp_path, monkeypatch):
    hf_cache = tmp_path / "hub"
    (hf_cache / "models--Qwen--Qwen3-0.6B").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text("#!/bin/sh\n", encoding="utf-8")
    claude.chmod(claude.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    discovery = scan_local_environment(model_dirs=[hf_cache], command_names=["claude"])
    payload = discovery.to_dict()

    assert payload["commands"]["claude"] is True
    assert payload["models"][0]["repo_id"] == "Qwen/Qwen3-0.6B"
    assert payload["models"][0]["source"] == "huggingface_cache"


def test_recommend_setup_prefers_available_claude_code(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text("#!/bin/sh\n", encoding="utf-8")
    claude.chmod(claude.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir))
    discovery = scan_local_environment(model_dirs=[], command_names=["claude", "codex"])

    recommendation = recommend_setup(discovery)

    assert recommendation.routing_targets["coding"] == "claude_code"
    assert recommendation.engine_overrides["claude_code"]["enabled"] is True
    assert any("Claude Code" in note for note in recommendation.notes)


def test_recommend_setup_includes_download_plan_for_missing_roles():
    discovery = scan_local_environment(model_dirs=[], command_names=[])

    recommendation = recommend_setup(discovery)
    payload = recommendation.to_dict()
    routes = {item["route"] for item in payload["download_suggestions"]}

    assert {"fast_local", "balanced_local", "multimodal_vision", "image_generation"} <= routes
    assert all(item["command"][0:2] == ["hf", "download"] for item in payload["download_suggestions"])


def test_write_recommended_config_is_safe_by_default(tmp_path):
    output = tmp_path / "model_router.local.yaml"
    output.write_text("existing: true\n", encoding="utf-8")
    discovery = scan_local_environment(model_dirs=[], command_names=[])

    result = write_recommended_config(output, discovery=discovery, force=False)

    assert result.written is False
    assert "already exists" in result.message
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {"existing": True}


def test_write_recommended_config_writes_valid_config_when_forced(tmp_path):
    output = tmp_path / "model_router.local.yaml"
    discovery = scan_local_environment(model_dirs=[], command_names=[])

    result = write_recommended_config(output, discovery=discovery, force=True)
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert result.written is True
    assert data["routing_targets"]["coding"] == "code_agent"
    assert data["engines"]["fast_local"]["model"] == "Qwen/Qwen3-0.6B"
    assert data["engines"]["fast_local"]["availability"]["required_paths"]
    assert "engines" in data
    assert "download_suggestions" not in data


def test_setup_scan_cli_emits_json():
    result = _run_cli("setup", "scan", "--json", "--no-default-dirs")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "commands" in payload
    assert "models" in payload


def test_setup_recommend_cli_emits_download_suggestions():
    result = _run_cli("setup", "recommend", "--json", "--no-default-dirs")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "routing_targets" in payload
    assert payload["download_suggestions"]


def test_setup_write_cli_writes_config(tmp_path):
    output = tmp_path / "model_router.local.yaml"

    result = _run_cli(
        "setup",
        "write",
        "--json",
        "--no-default-dirs",
        "--output",
        str(output),
        "--force",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["written"] is True
    assert output.exists()


def test_setup_wizard_asks_before_writing_config(tmp_path):
    output = tmp_path / "model_router.local.yaml"

    result = _run_cli_with_input(
        "setup",
        "wizard",
        "--no-default-dirs",
        "--output",
        str(output),
        user_input="y\n",
    )

    assert result.returncode == 0
    assert "Write this config" in result.stdout
    assert output.exists()


def test_setup_wizard_can_decline_write(tmp_path):
    output = tmp_path / "model_router.local.yaml"

    result = _run_cli_with_input(
        "setup",
        "wizard",
        "--no-default-dirs",
        "--output",
        str(output),
        user_input="n\n",
    )

    assert result.returncode == 0
    assert "No config written" in result.stdout
    assert not output.exists()


def test_local_example_config_is_structurally_valid():
    config = load_router_config(ROOT / "configs" / "model_router.local.example.yaml")

    assert config.target_engine("coding") == "code_agent"
    assert config.get_engine("multimodal_vision") is not None
    assert config.get_engine("image_generation") is not None
