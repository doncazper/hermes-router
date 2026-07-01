import json
from pathlib import Path

import pytest

from hermes.plugins.model_router.admin.state import settings_paths
import hermes.plugins.model_router.installer as installer_module
from hermes.plugins.model_router.installer import (
    InstallerOptions,
    build_install_plan,
    build_installer_state,
)
from hermes.plugins.model_router.product import FirstRunSignals
from hermes.plugins.model_router.setup_assistant import SetupDiscovery


def _discovery() -> SetupDiscovery:
    return SetupDiscovery(
        commands={
            "claude": False,
            "codex": False,
            "hf": False,
            "ollama": True,
            "llama-server": False,
            "mlx_lm.server": False,
            "lmstudio": False,
        },
        model_dirs=(),
        env_vars={
            "OPENAI_API_KEY": False,
            "ANTHROPIC_API_KEY": False,
            "HF_TOKEN": False,
        },
        models=(),
    )


def _signals() -> FirstRunSignals:
    return FirstRunSignals(
        ollama_installed=True,
        ollama_running=False,
        lmstudio_running=False,
        apple_silicon=True,
        mlx_lm_available=False,
        llama_server_available=False,
        recommended_preset="ollama",
        notes=("Ollama is installed but not running.",),
    )


def _admin_state(config_dir: Path, *, config_exists: bool) -> dict:
    return {
        "paths": {
            "config_dir": str(config_dir),
            "proxy_config": str(config_dir / "routing_proxy.yaml"),
            "model_router_config": str(config_dir / "model_router.yaml"),
        },
        "proxy": {
            "endpoint": None,
            "routing_mode": "decision",
            "routing_profile": "balanced",
        },
        "actions": [{"id": "doctor.run"}, {"id": "model.scan"}],
        "config_exists": config_exists,
        "config_valid": False,
        "config_error": None,
    }


def _scan_result() -> dict:
    return {
        "ok": True,
        "payload": {
            "discovery": {"models": []},
            "recommendation": {"notes": ["fake recommendation"]},
            "download_plan": {"suggestions": []},
        },
    }


def test_install_plan_json_is_deterministic_with_injected_state(tmp_path):
    options = InstallerOptions(config_dir=tmp_path / "config", quick=True)

    first = build_install_plan(
        options,
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(options.config_dir, config_exists=False),
        scan_result=_scan_result(),
    ).to_dict()
    second = build_install_plan(
        options,
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(options.config_dir, config_exists=False),
        scan_result=_scan_result(),
    ).to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["dry_run"] is True
    assert first["selected_preset"] == "ollama"
    assert first["installer"]["detected_runtimes"]["ollama_installed"] is True


def test_install_plan_existing_config_does_not_plan_overwrite(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    proxy_config = config_dir / "routing_proxy.yaml"
    proxy_config.write_text("sentinel: true\n", encoding="utf-8")

    plan = build_install_plan(
        InstallerOptions(config_dir=config_dir, ollama=True),
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(config_dir, config_exists=True),
        scan_result=_scan_result(),
    ).to_dict()

    command_ids = {command["id"] for command in plan["next_commands"]}
    assert plan["existing_config"] is True
    assert "init" not in command_ids
    assert "doctor" in command_ids
    assert proxy_config.read_text(encoding="utf-8") == "sentinel: true\n"


def test_install_plan_model_config_only_avoids_partial_init(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model_config = config_dir / "model_router.yaml"
    model_config.write_text("routing_targets: {}\n", encoding="utf-8")

    plan = build_install_plan(
        InstallerOptions(config_dir=config_dir, ollama=True),
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(config_dir, config_exists=True),
        scan_result=_scan_result(),
    ).to_dict()

    commands = {command["id"]: command for command in plan["next_commands"]}
    assert plan["existing_config"] is True
    assert plan["partial_config"] is True
    assert "init" not in commands
    assert "validate_config" in commands
    assert "init_force" in commands
    assert commands["proxy"]["available"] is False
    assert model_config.read_text(encoding="utf-8") == "routing_targets: {}\n"
    assert not (config_dir / "routing_proxy.yaml").exists()


def test_install_plan_proxy_config_only_runs_doctor_without_init(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    proxy_config = config_dir / "routing_proxy.yaml"
    proxy_config.write_text("router_config: missing.yaml\n", encoding="utf-8")

    plan = build_install_plan(
        InstallerOptions(config_dir=config_dir, ollama=True),
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(config_dir, config_exists=True),
        scan_result=_scan_result(),
    ).to_dict()

    command_ids = {command["id"] for command in plan["next_commands"]}
    assert plan["existing_config"] is True
    assert plan["partial_config"] is True
    assert "init" not in command_ids
    assert "doctor" in command_ids
    assert proxy_config.read_text(encoding="utf-8") == "router_config: missing.yaml\n"


def test_install_plan_is_dry_and_does_not_create_config_dir(tmp_path):
    config_dir = tmp_path / "missing"

    plan = build_install_plan(
        InstallerOptions(config_dir=config_dir, auto=True, yes=True),
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(config_dir, config_exists=False),
        scan_result=_scan_result(),
    ).to_dict()

    assert plan["dry_run"] is True
    assert plan["confirmed"] is True
    assert any("does not execute follow-up commands" in item for item in plan["warnings"])
    assert any("does not mutate by default" in item for item in plan["notes"])
    assert not config_dir.exists()
    init = next(command for command in plan["next_commands"] if command["id"] == "init")
    assert init["mutates"] is True
    assert init["requires_confirmation"] is True
    assert "Follow-up command" in init["reason"]


def test_installer_state_reports_config_and_runtime_signals(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "model_router.yaml").write_text("routing_targets: {}\n")

    state = build_installer_state(
        settings_paths(config_dir),
        discovery=_discovery(),
        signals=_signals(),
    )

    assert state["config_files"]["config_dir"] is True
    assert state["config_files"]["model_router_config"] is True
    assert state["config_files"]["routing_proxy_config"] is False
    assert state["detected_runtimes"]["ollama_installed"] is True
    assert "optional_dependencies" in state


def test_install_plan_adds_prereq_command_when_hf_missing_but_proxy_deps_present(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(
        installer_module,
        "_module_available",
        lambda module: module in {"fastapi", "httpx", "uvicorn", "textual"},
    )

    plan = build_install_plan(
        InstallerOptions(config_dir=config_dir, quick=True),
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(config_dir, config_exists=False),
        scan_result=_scan_result(),
    ).to_dict()

    install_prereqs = next(
        command for command in plan["next_commands"] if command["id"] == "install_prereqs"
    )
    assert install_prereqs["command"][:4] == [
        "model-router",
        "setup",
        "install-prereqs",
        "--preset",
    ]
    assert install_prereqs["command"][4] == "proxy"


def test_install_plan_surfaces_pipx_prereq_guidance(tmp_path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(installer_module, "detect_install_method", lambda: "pipx")

    plan = build_install_plan(
        InstallerOptions(config_dir=config_dir, quick=True),
        discovery=_discovery(),
        signals=_signals(),
        admin_state=_admin_state(config_dir, config_exists=False),
        scan_result=_scan_result(),
    ).to_dict()

    assert plan["installer"]["install_method"] == "pipx"
    assert any("pipx inject" in note for note in plan["prereq_plan"]["notes"])
    assert any("pipx inject" in warning for warning in plan["warnings"])
    assert any("pipx inject" in note for note in plan["notes"])
    first_step = plan["prereq_plan"]["steps"][0]
    assert first_step["command"][:4] == [
        "pipx",
        "inject",
        "--include-apps",
        "hermes-router",
    ]
