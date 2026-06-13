import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

from hermes.plugins.model_router.setup_assistant import (
    execute_download_plan,
    plan_model_downloads,
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
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        input=user_input,
        env=env,
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


def test_scan_detects_api_key_presence_without_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")

    discovery = scan_local_environment(model_dirs=[], command_names=[])
    payload = discovery.to_dict()

    assert payload["env_vars"]["OPENAI_API_KEY"] is True
    assert "secret-value" not in json.dumps(payload)


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


def test_recommend_setup_enables_api_engines_when_keys_are_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value")
    discovery = scan_local_environment(model_dirs=[], command_names=[])

    recommendation = recommend_setup(discovery)

    assert recommendation.engine_overrides["openai_api"]["enabled"] is True
    assert recommendation.engine_overrides["anthropic_api"]["enabled"] is True
    assert "secret-value" not in json.dumps(recommendation.to_dict())


def test_plan_model_downloads_filters_routes_and_rewrites_local_root(tmp_path):
    discovery = scan_local_environment(model_dirs=[], command_names=[])

    plan = plan_model_downloads(
        discovery=discovery,
        routes=["fast_local"],
        local_root=tmp_path / "models",
    )

    assert len(plan.suggestions) == 1
    suggestion = plan.suggestions[0]
    assert suggestion.route == "fast_local"
    assert suggestion.command[-1] == str(tmp_path / "models" / "fast_local" / "Qwen--Qwen3-0.6B")


def test_plan_model_downloads_supports_custom_repo_id(tmp_path):
    plan = plan_model_downloads(
        routes=["balanced_local"],
        repo_id="custom-org/custom-model",
        local_root=tmp_path / "models",
    )

    assert len(plan.suggestions) == 1
    suggestion = plan.suggestions[0]
    assert suggestion.route == "balanced_local"
    assert suggestion.repo_id == "custom-org/custom-model"
    assert suggestion.command == (
        "hf",
        "download",
        "custom-org/custom-model",
        "--local-dir",
        str(tmp_path / "models" / "balanced_local" / "custom-org--custom-model"),
    )


def test_execute_download_plan_dry_run_does_not_call_runner(tmp_path):
    plan = plan_model_downloads(
        discovery=scan_local_environment(model_dirs=[], command_names=[]),
        routes=["fast_local"],
        local_root=tmp_path / "models",
    )
    calls: list[tuple[str, ...]] = []

    result = execute_download_plan(
        plan,
        execute=False,
        confirmed=False,
        runner=lambda command: calls.append(command) or 0,
    )

    assert result.executed is False
    assert calls == []
    assert result.results[0].status == "planned"


def test_execute_download_plan_requires_confirmation(tmp_path):
    plan = plan_model_downloads(
        discovery=scan_local_environment(model_dirs=[], command_names=[]),
        routes=["fast_local"],
        local_root=tmp_path / "models",
    )

    result = execute_download_plan(
        plan,
        execute=True,
        confirmed=False,
        runner=lambda command: 0,
    )

    assert result.executed is False
    assert result.results[0].status == "confirmation_required"


def test_execute_download_plan_runs_confirmed_commands(tmp_path):
    plan = plan_model_downloads(
        discovery=scan_local_environment(model_dirs=[], command_names=[]),
        routes=["fast_local"],
        local_root=tmp_path / "models",
    )
    calls: list[tuple[str, ...]] = []

    result = execute_download_plan(
        plan,
        execute=True,
        confirmed=True,
        runner=lambda command: calls.append(command) or 0,
    )

    assert result.executed is True
    assert calls == [plan.suggestions[0].command]
    assert result.results[0].status == "completed"


def test_execute_download_plan_reports_missing_command(tmp_path):
    plan = plan_model_downloads(
        discovery=scan_local_environment(model_dirs=[], command_names=[]),
        routes=["fast_local"],
        local_root=tmp_path / "models",
    )

    result = execute_download_plan(
        plan,
        execute=True,
        confirmed=True,
        runner=lambda command: (_ for _ in ()).throw(FileNotFoundError("hf")),
    )

    assert result.executed is True
    assert result.ok is False
    assert result.results[0].status == "missing_command"
    assert result.results[0].returncode == 127


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


def test_setup_scan_cli_shows_api_key_presence_without_values():
    result = _run_cli_with_input(
        "setup",
        "scan",
        "--no-default-dirs",
        user_input="",
        extra_env={"OPENAI_API_KEY": "secret-value"},
    )

    assert result.returncode == 0
    assert "API keys:" in result.stdout
    assert "- OPENAI_API_KEY: present" in result.stdout
    assert "secret-value" not in result.stdout


def test_setup_recommend_cli_emits_download_suggestions():
    result = _run_cli("setup", "recommend", "--json", "--no-default-dirs")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "routing_targets" in payload
    assert payload["download_suggestions"]


def test_setup_download_cli_defaults_to_dry_run(tmp_path):
    result = _run_cli(
        "setup",
        "download",
        "--json",
        "--no-default-dirs",
        "--route",
        "fast_local",
        "--local-root",
        str(tmp_path / "models"),
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["executed"] is False
    assert payload["results"][0]["status"] == "planned"


def test_setup_download_cli_accepts_custom_repo_id(tmp_path):
    result = _run_cli(
        "setup",
        "download",
        "--json",
        "--no-default-dirs",
        "--route",
        "balanced_local",
        "--repo-id",
        "custom-org/custom-model",
        "--local-root",
        str(tmp_path / "models"),
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["results"][0]["repo_id"] == "custom-org/custom-model"
    assert payload["results"][0]["status"] == "planned"


def test_setup_download_cli_executes_with_yes_and_fake_hf(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "hf-called.txt"
    hf = bin_dir / "hf"
    hf.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {marker}\nexit 0\n",
        encoding="utf-8",
    )
    hf.chmod(hf.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    result = _run_cli(
        "setup",
        "download",
        "--json",
        "--no-default-dirs",
        "--route",
        "fast_local",
        "--local-root",
        str(tmp_path / "models"),
        "--execute",
        "--yes",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["executed"] is True
    assert payload["results"][0]["status"] == "completed"
    assert "download" in marker.read_text(encoding="utf-8")


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
        user_input="3\n" + "\n" * 7 + "y\n",
    )

    assert result.returncode == 0
    assert "Model source mode" in result.stdout
    assert "Coding and repository work" in result.stdout
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
        user_input="3\n" + "\n" * 7 + "n\n",
    )

    assert result.returncode == 0
    assert "No config written" in result.stdout
    assert not output.exists()


def test_setup_wizard_local_mode_writes_local_routes(tmp_path):
    output = tmp_path / "model_router.local.yaml"

    result = _run_cli_with_input(
        "setup",
        "wizard",
        "--no-default-dirs",
        "--output",
        str(output),
        user_input="1\n" + "\n" * 7 + "y\n",
    )
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert data["routing_targets"]["coding"] == "code_agent"
    assert data["routing_targets"]["balanced"] == "balanced_local"
    assert data["engines"]["codex"]["enabled"] is False
    assert "coding route set to codex" not in result.stdout


def test_setup_wizard_api_mode_can_use_api_key_routes(tmp_path):
    output = tmp_path / "model_router.local.yaml"

    result = _run_cli_with_input(
        "setup",
        "wizard",
        "--no-default-dirs",
        "--output",
        str(output),
        user_input="2\n" + "\n" * 7 + "y\n",
        extra_env={"OPENAI_API_KEY": "secret-value"},
    )
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert data["routing_targets"]["balanced"] == "openai_api"
    assert data["engines"]["balanced_local"]["model"] == "hermes-balanced-local"
    assert data["engines"]["openai_api"]["enabled"] is True
    assert data["engines"]["openai_api"]["availability"]["required_env"] == [
        "OPENAI_API_KEY"
    ]
    assert "secret-value" not in result.stdout


def test_setup_wizard_can_explicitly_assign_claude_code(tmp_path):
    output = tmp_path / "model_router.local.yaml"

    result = _run_cli_with_input(
        "setup",
        "wizard",
        "--no-default-dirs",
        "--output",
        str(output),
        user_input="1\n" + "\n" * 3 + "claude_code\n" + "\n" * 3 + "y\n",
        extra_env={"PATH": ""},
    )
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert data["routing_targets"]["coding"] == "claude_code"
    assert data["engines"]["claude_code"]["enabled"] is True
    assert data["engines"]["claude_code"]["availability"]["required_commands"] == [
        "claude"
    ]


def test_local_example_config_is_structurally_valid():
    config = load_router_config(ROOT / "configs" / "model_router.local.example.yaml")

    assert config.target_engine("coding") == "code_agent"
    assert config.get_engine("multimodal_vision") is not None
    assert config.get_engine("image_generation") is not None
