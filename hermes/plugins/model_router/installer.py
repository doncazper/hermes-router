"""Deterministic first-run installer planner for ModelRouter."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
import json
from pathlib import Path
import platform
import shutil
import socket
import sys
from typing import Any, Mapping

from hermes.plugins.model_router.admin.actions import (
    AdminActionError,
    run_admin_action,
)
from hermes.plugins.model_router.admin.state import build_admin_state, settings_paths
from hermes.plugins.model_router.product import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PROXY_PORT,
    FirstRunSignals,
    detect_first_run_environment,
)
from hermes.plugins.model_router.setup_assistant import (
    SetupDiscovery,
    plan_prereq_installs,
    scan_local_environment,
)


INSTALLABLE_PRESETS = ("lmstudio", "ollama", "mlx-lm", "llamacpp")
OPTIONAL_MODULES = {
    "fastapi": "fastapi",
    "httpx": "httpx",
    "textual": "textual",
    "uvicorn": "uvicorn",
    "huggingface_hub": "huggingface_hub",
    "mlx_lm": "mlx_lm",
}
INSTALLER_COMMANDS = (
    "model-router",
    "model-router-proxy",
    "hermes-router",
    "hf",
    "ollama",
    "llama-server",
    "mlx_lm.server",
    "lmstudio",
)


@dataclass(frozen=True)
class InstallerOptions:
    """Flags that shape the deterministic install plan."""

    config_dir: Path = Path(DEFAULT_CONFIG_DIR)
    quick: bool = False
    auto: bool = False
    local_only: bool = False
    lmstudio: bool = False
    ollama: bool = False
    mlx_lm: bool = False
    llamacpp: bool = False
    developer: bool = False
    yes: bool = False
    proxy_port: int = DEFAULT_PROXY_PORT
    settings_port: int = 8099


@dataclass(frozen=True)
class InstallCommand:
    """A command the user may explicitly run after reviewing the plan."""

    id: str
    label: str
    command: tuple[str, ...]
    reason: str
    mutates: bool = False
    requires_confirmation: bool = False
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "command": list(self.command),
            "reason": self.reason,
            "mutates": self.mutates,
            "requires_confirmation": self.requires_confirmation,
            "available": self.available,
        }


@dataclass(frozen=True)
class InstallPlan:
    """Read-only onboarding plan produced by ``model-router install``."""

    ok: bool
    dry_run: bool
    confirmed: bool
    selected_preset: str | None
    preset_reason: str
    config_dir: str
    existing_config: bool
    partial_config: bool
    installer: dict[str, Any]
    admin: dict[str, Any]
    scan: dict[str, Any]
    prereq_plan: dict[str, Any]
    next_commands: tuple[InstallCommand, ...]
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "confirmed": self.confirmed,
            "selected_preset": self.selected_preset,
            "preset_reason": self.preset_reason,
            "config_dir": self.config_dir,
            "existing_config": self.existing_config,
            "partial_config": self.partial_config,
            "installer": self.installer,
            "admin": self.admin,
            "scan": self.scan,
            "prereq_plan": self.prereq_plan,
            "next_commands": [command.to_dict() for command in self.next_commands],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


def build_installer_state(
    paths: Mapping[str, Path],
    *,
    discovery: SetupDiscovery | None = None,
    signals: FirstRunSignals | None = None,
    proxy_port: int = DEFAULT_PROXY_PORT,
    settings_port: int = 8099,
) -> dict[str, Any]:
    """Build the shared installer state block from safe local signals."""

    discovery = discovery or scan_local_environment()
    command_status = _command_status(discovery)
    detected_runtimes = _detected_runtimes(
        discovery=discovery,
        signals=signals,
    )
    ports = {
        f"127.0.0.1:{proxy_port}_available": _port_available(
            "127.0.0.1",
            proxy_port,
        ),
        f"127.0.0.1:{settings_port}_available": _port_available(
            "127.0.0.1",
            settings_port,
        ),
        "127.0.0.1:11434_open": _port_open("127.0.0.1", 11434),
        "127.0.0.1:1234_open": _port_open("127.0.0.1", 1234),
    }
    optional_dependencies = {
        name: _module_available(module)
        for name, module in sorted(OPTIONAL_MODULES.items())
    }
    config_files = {
        "config_dir": Path(paths["config_dir"]).expanduser().exists(),
        "model_router_config": Path(paths["model_router_config"]).expanduser().exists(),
        "routing_proxy_config": Path(paths["proxy_config"]).expanduser().exists(),
        "events": Path(paths["events"]).expanduser().exists(),
        "feedback": Path(paths["feedback"]).expanduser().exists(),
    }
    warnings = _installer_warnings(
        optional_dependencies=optional_dependencies,
        command_status=command_status,
        config_files=config_files,
        detected_runtimes=detected_runtimes,
    )
    return {
        "install_method": detect_install_method(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_version": package_version(),
        "optional_dependencies": optional_dependencies,
        "path_status": command_status,
        "config_files": config_files,
        "ports": ports,
        "detected_runtimes": detected_runtimes,
        "recommended_next_actions": _recommended_next_action_labels(
            config_files=config_files,
            optional_dependencies=optional_dependencies,
        ),
        "warnings": warnings,
    }


def build_install_plan(
    options: InstallerOptions,
    *,
    discovery: SetupDiscovery | None = None,
    signals: FirstRunSignals | None = None,
    admin_state: Mapping[str, Any] | None = None,
    scan_result: Mapping[str, Any] | None = None,
) -> InstallPlan:
    """Build a deterministic, no-mutation onboarding plan."""

    paths = settings_paths(options.config_dir)
    discovery = discovery or scan_local_environment()
    signals = signals or detect_first_run_environment()
    admin_state = dict(admin_state or _safe_admin_state(paths))
    scan_result = dict(scan_result or _safe_scan_action(paths))
    installer = build_installer_state(
        paths,
        discovery=discovery,
        signals=signals,
        proxy_port=options.proxy_port,
        settings_port=options.settings_port,
    )
    selected_preset, preset_reason = _selected_preset(options, signals)
    config_files = installer["config_files"]
    model_config_exists = bool(config_files["model_router_config"])
    proxy_config_exists = bool(config_files["routing_proxy_config"])
    existing_config = model_config_exists or proxy_config_exists
    partial_config = model_config_exists != proxy_config_exists
    prereq_preset = _prereq_preset(selected_preset, developer=options.developer)
    prereq_plan_object = plan_prereq_installs(
        preset=prereq_preset,
        install_method=str(installer.get("install_method") or "unknown"),
    )
    prereq_plan = prereq_plan_object.to_dict()
    prereqs_needed = _prereq_plan_needs_followup(prereq_plan_object, installer)
    warnings = list(installer["warnings"])
    prereq_notes = prereq_plan_object.notes
    if prereqs_needed:
        warnings.extend(f"Prerequisite plan: {note}" for note in prereq_notes)
    if options.local_only:
        warnings.append("Local-only requested; hosted providers will not be enabled.")
    if partial_config:
        warnings.append(
            "Partial config detected; no first-run init command is planned by default."
        )
    if options.yes:
        warnings.append(
            "--yes records confirmation intent for the installer plan, but "
            "`model-router install` still does not execute follow-up commands."
        )
    commands = _next_commands(
        options=options,
        paths=paths,
        selected_preset=selected_preset,
        existing_config=existing_config,
        config_files=config_files,
        prereq_preset=prereq_preset,
        prereqs_needed=prereqs_needed,
        installer=installer,
    )
    return InstallPlan(
        ok=True,
        dry_run=True,
        confirmed=options.yes,
        selected_preset=selected_preset,
        preset_reason=preset_reason,
        config_dir=str(Path(options.config_dir).expanduser()),
        existing_config=existing_config,
        partial_config=partial_config,
        installer=installer,
        admin=_admin_summary(admin_state),
        scan=_scan_summary(scan_result),
        prereq_plan=prereq_plan,
        next_commands=tuple(commands),
        notes=tuple(
            _install_notes(
                options,
                selected_preset,
                existing_config,
                partial_config=partial_config,
                prereq_notes=prereq_notes if prereqs_needed else (),
            )
        ),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def package_version() -> str | None:
    try:
        return metadata.version("hermes-router")
    except metadata.PackageNotFoundError:
        return None


def detect_install_method() -> str:
    try:
        distribution = metadata.distribution("hermes-router")
    except metadata.PackageNotFoundError:
        return "unknown"
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            payload = json.loads(direct_url)
        except json.JSONDecodeError:
            payload = {}
        dir_info = payload.get("dir_info")
        if isinstance(dir_info, Mapping) and dir_info.get("editable") is True:
            return "editable"
    installer = (distribution.read_text("INSTALLER") or "").strip().lower()
    executable = sys.executable.lower()
    prefix = sys.prefix.lower()
    if "pipx" in executable or "pipx" in prefix:
        return "pipx"
    if "uv" in installer or "uv" in executable or "uv" in prefix:
        return "uv_tool"
    if installer == "pip":
        return "pip"
    return "unknown"


def _selected_preset(
    options: InstallerOptions,
    signals: FirstRunSignals,
) -> tuple[str | None, str]:
    requested = [
        preset
        for enabled, preset in (
            (options.lmstudio, "lmstudio"),
            (options.ollama, "ollama"),
            (options.mlx_lm, "mlx-lm"),
            (options.llamacpp, "llamacpp"),
        )
        if enabled
    ]
    if len(requested) > 1:
        raise ValueError("choose only one preset flag")
    if requested:
        return requested[0], "requested by flag"
    if options.auto or options.quick:
        return signals.recommended_preset, "auto-detected from local signals"
    return signals.recommended_preset, "recommended from local signals"


def _prereq_preset(selected_preset: str | None, *, developer: bool) -> str:
    if developer:
        return "all"
    if selected_preset in {"mlx-lm", "llamacpp"}:
        return selected_preset
    return "proxy"


def _safe_admin_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    try:
        return build_admin_state(paths)
    except Exception as exc:
        return {"error": str(exc)}


def _safe_scan_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    try:
        return run_admin_action("model.scan", paths)
    except AdminActionError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "status_code": exc.status_code,
            "details": exc.details,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _admin_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    if "error" in state:
        return {"ok": False, "error": str(state["error"])}
    paths = state.get("paths") if isinstance(state.get("paths"), Mapping) else {}
    proxy = state.get("proxy") if isinstance(state.get("proxy"), Mapping) else {}
    actions = state.get("actions") if isinstance(state.get("actions"), list) else []
    return {
        "ok": True,
        "config_exists": bool(state.get("config_exists")),
        "config_valid": bool(state.get("config_valid")),
        "config_error": state.get("config_error"),
        "proxy": {
            "endpoint": proxy.get("endpoint"),
            "routing_mode": proxy.get("routing_mode"),
            "routing_profile": proxy.get("routing_profile"),
        },
        "paths": {
            "config_dir": paths.get("config_dir"),
            "routing_proxy_config": paths.get("proxy_config"),
            "model_router_config": paths.get("model_router_config"),
        },
        "actions": [
            action.get("id")
            for action in actions
            if isinstance(action, Mapping) and action.get("id")
        ],
    }


def _scan_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("ok") is False:
        return {"ok": False, "error": result.get("error")}
    payload = result.get("payload")
    if not isinstance(payload, Mapping):
        return {"ok": False, "error": "scan action returned no payload"}
    discovery = payload.get("discovery")
    recommendation = payload.get("recommendation")
    download_plan = payload.get("download_plan")
    models = discovery.get("models", []) if isinstance(discovery, Mapping) else []
    suggestions = (
        download_plan.get("suggestions", [])
        if isinstance(download_plan, Mapping)
        else []
    )
    return {
        "ok": True,
        "model_count": len(models) if isinstance(models, list) else 0,
        "download_suggestion_count": (
            len(suggestions) if isinstance(suggestions, list) else 0
        ),
        "recommendation_notes": (
            recommendation.get("notes", [])
            if isinstance(recommendation, Mapping)
            else []
        ),
    }


def _next_commands(
    *,
    options: InstallerOptions,
    paths: Mapping[str, Path],
    selected_preset: str | None,
    existing_config: bool,
    config_files: Mapping[str, bool],
    prereq_preset: str,
    prereqs_needed: bool,
    installer: Mapping[str, Any],
) -> list[InstallCommand]:
    config_dir = str(Path(options.config_dir).expanduser())
    proxy_config = str(paths["proxy_config"])
    model_config = str(paths["model_router_config"])
    model_config_exists = bool(config_files.get("model_router_config"))
    proxy_config_exists = bool(config_files.get("routing_proxy_config"))
    proxy_available = proxy_config_exists or not existing_config
    proxy_reason = (
        "Follow-up command that runs the local OpenAI-compatible proxy/router."
        if proxy_available
        else "routing_proxy.yaml is missing; repair config before starting the proxy."
    )
    commands: list[InstallCommand] = []
    if prereqs_needed:
        commands.append(
            InstallCommand(
                id="install_prereqs",
                label="Install optional proxy/runtime prerequisites",
                command=(
                    "model-router",
                    "setup",
                    "install-prereqs",
                    "--preset",
                    prereq_preset,
                    "--execute",
                    "--yes",
                ),
                reason=(
                    "Follow-up command that installs optional Python packages "
                    "into the active environment."
                ),
                mutates=True,
                requires_confirmation=True,
            )
        )
    if proxy_config_exists:
        commands.append(
            InstallCommand(
                id="doctor",
                label="Check existing config",
                command=("model-router", "doctor", "--config", proxy_config),
                reason="Existing config detected; validate it instead of overwriting.",
            )
        )
    elif model_config_exists:
        commands.append(
            InstallCommand(
                id="validate_config",
                label="Validate existing router config",
                command=("model-router", "validate-config", "--config", model_config),
                reason=(
                    "Partial config detected; routing_proxy.yaml is missing, "
                    "so no first-run overwrite is planned."
                ),
            )
        )
        if selected_preset:
            commands.append(
                InstallCommand(
                    id="init_force",
                    label="Recreate configs after backup",
                    command=(
                        "model-router",
                        "init",
                        "--preset",
                        selected_preset,
                        "--config-dir",
                        config_dir,
                        "--force",
                        "--yes",
                    ),
                    reason=(
                        "Back up existing config first; this overwrites files "
                        "to repair a partial config directory."
                    ),
                    mutates=True,
                    requires_confirmation=True,
                )
            )
    elif selected_preset:
        init_command: tuple[str, ...] = (
            "model-router",
            "init",
            "--preset",
            selected_preset,
            "--config-dir",
            config_dir,
            "--yes",
        )
        if selected_preset in {"llamacpp", "mlx-lm"}:
            init_command = (*init_command, "--auto-models")
        commands.append(
            InstallCommand(
                id="init",
                label="Create initial config",
                command=init_command,
                reason=(
                    "Follow-up command that writes first-run configs; it will "
                    "not overwrite existing files."
                ),
                mutates=True,
                requires_confirmation=True,
            )
        )
        commands.append(
            InstallCommand(
                id="doctor",
                label="Check config after init",
                command=("model-router", "doctor", "--config", proxy_config),
                reason="Run after creating routing_proxy.yaml.",
            )
        )
    commands.extend(
        [
            InstallCommand(
                id="settings",
                label="Open local admin settings UI",
                command=("model-router", "settings", "--config-dir", config_dir),
                reason="Visual admin/config surface; no chat UI.",
            ),
            InstallCommand(
                id="proxy",
                label="Start proxy",
                command=("model-router-proxy", "--config", proxy_config),
                reason=proxy_reason,
                mutates=True,
                requires_confirmation=True,
                available=proxy_available,
            ),
            InstallCommand(
                id="download_plan",
                label="Review recommended model downloads",
                command=("model-router", "setup", "download"),
                reason="Plans downloads only; add --execute --yes after review.",
            ),
            InstallCommand(
                id="telemetry",
                label="Inspect telemetry after dogfooding",
                command=(
                    "model-router",
                    "telemetry",
                    "summary",
                    "--events",
                    str(paths["events"]),
                    "--feedback",
                    str(paths["feedback"]),
                ),
                reason="Shows privacy-safe routing data and labels.",
            ),
        ]
    )
    if options.developer:
        commands.extend(
            [
                InstallCommand(
                    id="developer_editable_install",
                    label="Install editable developer environment",
                    command=(sys.executable, "-m", "pip", "install", "-e", ".[dev,proxy]"),
                    reason="Developer-only editable install command.",
                    mutates=True,
                    requires_confirmation=True,
                ),
                InstallCommand(
                    id="developer_tests",
                    label="Run developer checks",
                    command=(sys.executable, "-m", "pytest"),
                    reason="Developer validation after local changes.",
                ),
            ]
        )
    optional = installer.get("optional_dependencies")
    optional = optional if isinstance(optional, Mapping) else {}
    tui_available = bool(optional.get("textual"))
    commands.append(
        InstallCommand(
            id="tui",
            label="Open terminal UI",
            command=("model-router", "tui"),
            reason=(
                "Opens the optional terminal control center."
                if tui_available
                else "Install the TUI extra, for example `python -m pip install 'hermes-router[tui]'`."
            ),
            available=tui_available,
        )
    )
    return commands


def _install_notes(
    options: InstallerOptions,
    selected_preset: str | None,
    existing_config: bool,
    *,
    partial_config: bool,
    prereq_notes: tuple[str, ...] = (),
) -> list[str]:
    notes = [
        "Installer plan is deterministic and plan-only; it does not mutate by default.",
        (
            "Dependencies, model downloads, config writes, hosted providers, "
            "runtime starts, and services remain explicit follow-up commands."
        ),
    ]
    if existing_config:
        notes.append("Existing config detected; no overwrite planned.")
    if partial_config:
        notes.append(
            "Partial config detected; installer avoids first-run init by default."
        )
    if options.quick:
        notes.append("--quick selected the shortest path through the same safe plan.")
    if selected_preset:
        notes.append(f"Preset candidate: {selected_preset}.")
    for note in prereq_notes:
        notes.append(f"Prerequisite plan: {note}")
    return notes


def _command_status(discovery: SetupDiscovery) -> dict[str, bool]:
    status = dict(discovery.commands)
    for command in INSTALLER_COMMANDS:
        status[command] = status.get(command, shutil.which(command) is not None)
    return dict(sorted(status.items()))


def _detected_runtimes(
    *,
    discovery: SetupDiscovery,
    signals: FirstRunSignals | None,
) -> dict[str, bool]:
    if signals is not None:
        return {
            "apple_silicon": signals.apple_silicon,
            "ollama_installed": signals.ollama_installed,
            "ollama_running": signals.ollama_running,
            "lmstudio_running": signals.lmstudio_running,
            "mlx_lm_available": signals.mlx_lm_available,
            "llama_server_available": signals.llama_server_available,
            "local_mlx_models": bool(signals.mlx_lm_models),
            "local_gguf_models": bool(signals.gguf_models),
        }
    commands = discovery.commands
    return {
        "apple_silicon": platform.machine().lower() in {"arm64", "aarch64"}
        and platform.system() == "Darwin",
        "ollama_installed": bool(commands.get("ollama")),
        "ollama_running": _port_open("127.0.0.1", 11434),
        "lmstudio_running": _port_open("127.0.0.1", 1234),
        "mlx_lm_available": bool(commands.get("mlx_lm.server")),
        "llama_server_available": bool(commands.get("llama-server")),
        "local_mlx_models": any(model.source == "huggingface" for model in discovery.models),
        "local_gguf_models": any(model.name.lower().endswith(".gguf") for model in discovery.models),
    }


def _installer_warnings(
    *,
    optional_dependencies: Mapping[str, bool],
    command_status: Mapping[str, bool],
    config_files: Mapping[str, bool],
    detected_runtimes: Mapping[str, bool],
) -> list[str]:
    warnings: list[str] = []
    missing_proxy_deps = [
        name for name in ("fastapi", "httpx", "uvicorn") if not optional_dependencies[name]
    ]
    if missing_proxy_deps:
        warnings.append(
            "Missing proxy optional dependencies: " + ", ".join(missing_proxy_deps)
        )
    if not command_status.get("model-router-proxy", False):
        warnings.append("model-router-proxy command is not on PATH.")
    if not config_files.get("routing_proxy_config", False):
        warnings.append("routing_proxy.yaml is not present yet.")
    if detected_runtimes.get("ollama_installed") and not detected_runtimes.get(
        "ollama_running"
    ):
        warnings.append("Ollama command detected but server is not running.")
    if not command_status.get("hf", False):
        warnings.append("Hugging Face hf CLI is missing; downloads cannot run yet.")
    return warnings


def _recommended_next_action_labels(
    *,
    config_files: Mapping[str, bool],
    optional_dependencies: Mapping[str, bool],
) -> list[str]:
    labels: list[str] = []
    if not all(optional_dependencies.get(name, False) for name in ("fastapi", "httpx", "uvicorn")):
        labels.append("Install proxy prerequisites")
    if not config_files.get("routing_proxy_config", False):
        labels.append("Create first-run config")
    labels.extend(["Run doctor", "Open settings UI", "Start proxy"])
    return labels


def _proxy_optional_deps_ready(installer: Mapping[str, Any]) -> bool:
    optional = installer.get("optional_dependencies")
    if not isinstance(optional, Mapping):
        return False
    return all(bool(optional.get(name)) for name in ("fastapi", "httpx", "uvicorn"))


def _prereq_plan_needs_followup(plan: Any, installer: Mapping[str, Any]) -> bool:
    return any(
        not _prereq_step_ready(step, installer)
        for step in getattr(plan, "steps", ())
    )


def _prereq_step_ready(step: Any, installer: Mapping[str, Any]) -> bool:
    package = str(getattr(step, "command", ("",))[-1]).lower()
    optional = installer.get("optional_dependencies")
    commands = installer.get("path_status")
    optional = optional if isinstance(optional, Mapping) else {}
    commands = commands if isinstance(commands, Mapping) else {}
    if package.startswith("fastapi"):
        return bool(optional.get("fastapi"))
    if package.startswith("httpx"):
        return bool(optional.get("httpx"))
    if package.startswith("uvicorn"):
        return bool(optional.get("uvicorn"))
    if package == "mlx-lm":
        return bool(optional.get("mlx_lm"))
    if package.startswith("huggingface_hub"):
        return bool(commands.get("hf"))
    return False


def _module_available(module_name: str) -> bool:
    try:
        return util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        return sock.connect_ex((host, port)) == 0


def options_from_namespace(namespace: Any) -> InstallerOptions:
    return InstallerOptions(
        config_dir=Path(namespace.config_dir),
        quick=bool(namespace.quick),
        auto=bool(namespace.auto),
        local_only=bool(namespace.local_only),
        lmstudio=bool(namespace.lmstudio),
        ollama=bool(namespace.ollama),
        mlx_lm=bool(namespace.mlx_lm),
        llamacpp=bool(namespace.llamacpp),
        developer=bool(namespace.developer),
        yes=bool(namespace.yes),
    )
