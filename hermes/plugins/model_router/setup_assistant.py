"""Setup helpers for configuring Hermes model-router engines."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Sequence
import shutil
import subprocess
import sys

import yaml

from hermes.plugins.model_router.config import (
    DEFAULT_ROUTING_TARGETS,
    default_config_source,
    default_config_text,
)
from hermes.plugins.model_router.model_advisor import (
    HardwareProfile,
    ModelAdvice,
    detect_hardware_profile,
    recommend_catalog_models,
)

DEFAULT_COMMANDS = (
    "claude",
    "codex",
    "hf",
    "ollama",
    "llama-server",
    "lmstudio",
)

DEFAULT_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
)

MODEL_SCAN_MAX_DEPTH = 4
SKIP_MODEL_DIR_NAMES = {
    "blobs",
    "library",
    "manifests",
    "refs",
    "registry.ollama.ai",
    "snapshots",
    "tmp",
}
MODEL_FILE_MARKERS = {
    "config.json",
    "generation_config.json",
    "model_index.json",
    "tokenizer.json",
    "tokenizer.model",
}
MODEL_FILE_SUFFIXES = (
    ".gguf",
    ".safetensors",
    ".bin",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
)
MODEL_NAME_HINTS = (
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi",
    "deepseek",
    "sdxl",
    "flux",
    "clip",
    "embed",
    "bge",
    "vision",
    "text",
    "instruct",
    "chat",
    "diffusion",
)

ROLE_TO_ENGINE = {
    "fast_local": "fast_local",
    "balanced_local": "balanced_local",
    "reasoning_local": "reasoning_local",
    "code_agent": "code_agent",
    "web_research": "web_research",
    "multimodal_vision": "multimodal_vision",
    "image_generation": "image_generation",
}


@dataclass(frozen=True)
class DiscoveredModel:
    name: str
    repo_id: str
    path: str
    source: str
    roles: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "path": self.path,
            "source": self.source,
            "roles": list(self.roles),
        }


@dataclass(frozen=True)
class SetupDiscovery:
    commands: dict[str, bool]
    model_dirs: tuple[str, ...]
    env_vars: dict[str, bool] = field(default_factory=dict)
    models: tuple[DiscoveredModel, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": dict(sorted(self.commands.items())),
            "env_vars": dict(sorted(self.env_vars.items())),
            "model_dirs": list(self.model_dirs),
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True)
class DownloadSuggestion:
    route: str
    repo_id: str
    provider: str
    adapter: str
    reason: str
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "repo_id": self.repo_id,
            "provider": self.provider,
            "adapter": self.adapter,
            "reason": self.reason,
            "command": list(self.command),
        }


@dataclass(frozen=True)
class DownloadPlan:
    suggestions: tuple[DownloadSuggestion, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [
                suggestion.to_dict() for suggestion in self.suggestions
            ],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DownloadResult:
    route: str
    repo_id: str
    command: tuple[str, ...]
    status: str
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "repo_id": self.repo_id,
            "command": list(self.command),
            "status": self.status,
            "returncode": self.returncode,
        }


@dataclass(frozen=True)
class DownloadExecution:
    executed: bool
    results: tuple[DownloadResult, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(result.status in {"planned", "completed"} for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SetupRecommendation:
    routing_targets: dict[str, str]
    engine_overrides: dict[str, dict[str, Any]]
    download_suggestions: tuple[DownloadSuggestion, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    hardware_profile: HardwareProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "routing_targets": dict(sorted(self.routing_targets.items())),
            "engine_overrides": self.engine_overrides,
            "download_suggestions": [
                suggestion.to_dict() for suggestion in self.download_suggestions
            ],
            "notes": list(self.notes),
        }
        if self.hardware_profile is not None:
            payload["hardware_profile"] = self.hardware_profile.to_dict()
        return payload


@dataclass(frozen=True)
class ConfigWriteResult:
    written: bool
    path: str
    message: str
    recommendation: SetupRecommendation

    def to_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "path": self.path,
            "message": self.message,
            "recommendation": self.recommendation.to_dict(),
        }


def default_model_dirs() -> tuple[Path, ...]:
    return (
        Path("models"),
        Path("~/.cache/huggingface/hub").expanduser(),
        Path("~/.ollama/models").expanduser(),
        Path("~/.lmstudio/models").expanduser(),
        Path("~/Library/Application Support/LM Studio/models").expanduser(),
        Path("~/models").expanduser(),
        Path("~/Downloads").expanduser(),
    )


def scan_local_environment(
    *,
    model_dirs: Sequence[str | Path] | None = None,
    command_names: Sequence[str] = DEFAULT_COMMANDS,
    env_var_names: Sequence[str] = DEFAULT_ENV_VARS,
) -> SetupDiscovery:
    paths = tuple(
        Path(path).expanduser()
        for path in (default_model_dirs() if model_dirs is None else model_dirs)
    )
    commands = {
        name: _resolve_command_executable(name) is not None for name in command_names
    }
    env_vars = {name: bool(os.environ.get(name)) for name in env_var_names}
    models = _scan_model_dirs(paths)
    return SetupDiscovery(
        commands=commands,
        env_vars=env_vars,
        model_dirs=tuple(str(path) for path in paths),
        models=models,
    )


def recommend_setup(
    discovery: SetupDiscovery,
    *,
    profile: str = "balanced",
    hardware: HardwareProfile | None = None,
) -> SetupRecommendation:
    routing_targets = dict(DEFAULT_ROUTING_TARGETS)
    engine_overrides: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    hardware = hardware or detect_hardware_profile()
    notes.append(_hardware_note(hardware))

    if discovery.commands.get("claude"):
        routing_targets["coding"] = "claude_code"
        engine_overrides["claude_code"] = {
            "enabled": True,
            "availability": {
                "status": "auto",
                "required_commands": ["claude"],
            },
        }
        notes.append("Claude Code detected; coding route set to claude_code.")
    elif discovery.commands.get("codex"):
        routing_targets["coding"] = "codex"
        engine_overrides["codex"] = {
            "enabled": True,
            "availability": {
                "status": "auto",
                "required_commands": ["codex"],
            },
        }
        notes.append("Codex CLI detected; coding route set to codex.")

    if discovery.env_vars.get("OPENAI_API_KEY"):
        engine_overrides["openai_api"] = _api_engine_override(
            env_var="OPENAI_API_KEY",
            adapter="openai_chat",
        )
        notes.append("OPENAI_API_KEY detected; openai_api can be enabled.")
    if discovery.env_vars.get("ANTHROPIC_API_KEY"):
        engine_overrides["anthropic_api"] = _api_engine_override(
            env_var="ANTHROPIC_API_KEY",
            adapter="anthropic_chat",
        )
        notes.append("ANTHROPIC_API_KEY detected; anthropic_api can be enabled.")

    matched_roles = _local_model_overrides(discovery.models, engine_overrides, notes)
    download_suggestions = tuple(
        suggestion
        for suggestion in _default_download_suggestions(profile, hardware)
        if suggestion.route not in matched_roles
    )
    for suggestion in download_suggestions:
        engine_name = ROLE_TO_ENGINE[suggestion.route]
        engine_overrides.setdefault(
            engine_name,
            _engine_override_for_suggestion(suggestion),
        )

    if not discovery.commands.get("hf"):
        notes.append("Install the Hugging Face `hf` CLI before running download plans.")

    return SetupRecommendation(
        routing_targets=routing_targets,
        engine_overrides=engine_overrides,
        download_suggestions=download_suggestions,
        notes=tuple(notes),
        hardware_profile=hardware,
    )


def write_recommended_config(
    output_path: str | Path,
    *,
    discovery: SetupDiscovery | None = None,
    force: bool = False,
    profile: str = "balanced",
    base_config_path: str | Path | None = None,
) -> ConfigWriteResult:
    output = Path(output_path).expanduser()
    discovery = discovery or scan_local_environment()
    recommendation = recommend_setup(discovery, profile=profile)

    return write_config_from_recommendation(
        output,
        recommendation=recommendation,
        force=force,
        base_config_path=base_config_path,
    )


def write_config_from_recommendation(
    output_path: str | Path,
    *,
    recommendation: SetupRecommendation,
    force: bool = False,
    base_config_path: str | Path | None = None,
) -> ConfigWriteResult:
    output = Path(output_path).expanduser()

    if output.exists() and not force:
        return ConfigWriteResult(
            written=False,
            path=str(output),
            message=f"config already exists: {output}",
            recommendation=recommendation,
        )

    data = _load_config_mapping(base_config_path)
    data["routing_targets"] = {
        **data.get("routing_targets", {}),
        **recommendation.routing_targets,
    }
    engines = data.setdefault("engines", {})
    for engine_name, patch in recommendation.engine_overrides.items():
        if engine_name in engines:
            _deep_update(engines[engine_name], patch)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return ConfigWriteResult(
        written=True,
        path=str(output),
        message=f"wrote recommended config: {output}",
        recommendation=recommendation,
    )


def _hardware_note(hardware: HardwareProfile) -> str:
    memory = (
        f"{hardware.total_memory_gb:.0f} GB RAM"
        if hardware.total_memory_gb is not None
        else "unknown RAM"
    )
    disk = (
        f"{hardware.disk_free_gb:.0f} GB free"
        if hardware.disk_free_gb is not None
        else "unknown disk"
    )
    chip = "Apple Silicon" if hardware.apple_silicon else hardware.machine or "unknown CPU"
    return f"Hardware advisor: {chip}, {memory}, {disk}."


def engine_override_for_local_model(
    role: str,
    model: DiscoveredModel,
) -> dict[str, Any]:
    return _engine_override_for_model(role, model)


def engine_override_for_download(
    suggestion: DownloadSuggestion,
) -> dict[str, Any]:
    return _engine_override_for_suggestion(suggestion)


def _api_engine_override(*, env_var: str, adapter: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "adapter": adapter,
        "availability": {
            "status": "auto",
            "required_env": [env_var],
        },
    }


def plan_model_downloads(
    *,
    discovery: SetupDiscovery | None = None,
    profile: str = "balanced",
    routes: Sequence[str] | None = None,
    local_root: str | Path | None = None,
    repo_id: str | None = None,
    adapter: str | None = None,
) -> DownloadPlan:
    if repo_id is not None:
        route = _custom_download_route(routes)
        suggestion = _with_local_root(
            DownloadSuggestion(
                route=route,
                repo_id=repo_id,
                provider="huggingface",
                adapter=adapter or _adapter_for_route(route),
                reason="User-selected Hugging Face model.",
                command=(
                    "hf",
                    "download",
                    repo_id,
                    "--local-dir",
                    f"models/{route}/{_repo_slug(repo_id)}",
                ),
            ),
            local_root,
        )
        return DownloadPlan(
            suggestions=(suggestion,),
            notes=("Custom Hugging Face repo requested by user.",),
        )

    discovery = discovery or scan_local_environment()
    recommendation = recommend_setup(discovery, profile=profile)
    selected_routes = set(routes or ())
    suggestions = tuple(
        _with_local_root(suggestion, local_root)
        for suggestion in recommendation.download_suggestions
        if not selected_routes or suggestion.route in selected_routes
    )
    notes = recommendation.notes
    if selected_routes:
        missing = selected_routes - {suggestion.route for suggestion in suggestions}
        if missing:
            notes = (
                *notes,
                "No download suggestions for routes: " + ", ".join(sorted(missing)),
            )
    return DownloadPlan(suggestions=suggestions, notes=notes)


def _custom_download_route(routes: Sequence[str] | None) -> str:
    if not routes:
        return "custom"
    if len(routes) != 1:
        raise ValueError("custom repo downloads require exactly one --route")
    return routes[0]


def execute_download_plan(
    plan: DownloadPlan,
    *,
    execute: bool,
    confirmed: bool,
    runner=None,
) -> DownloadExecution:
    if not execute:
        return DownloadExecution(
            executed=False,
            results=tuple(
                DownloadResult(
                    route=suggestion.route,
                    repo_id=suggestion.repo_id,
                    command=suggestion.command,
                    status="planned",
                )
                for suggestion in plan.suggestions
            ),
            notes=plan.notes,
        )

    if not confirmed:
        return DownloadExecution(
            executed=False,
            results=tuple(
                DownloadResult(
                    route=suggestion.route,
                    repo_id=suggestion.repo_id,
                    command=suggestion.command,
                    status="confirmation_required",
                )
                for suggestion in plan.suggestions
            ),
            notes=(*plan.notes, "Pass --yes or confirm interactively to execute."),
        )

    runner = runner or _run_download_command
    results: list[DownloadResult] = []
    for suggestion in plan.suggestions:
        try:
            returncode = runner(suggestion.command)
        except FileNotFoundError:
            results.append(
                DownloadResult(
                    route=suggestion.route,
                    repo_id=suggestion.repo_id,
                    command=suggestion.command,
                    status="missing_command",
                    returncode=127,
                )
            )
            continue
        results.append(
            DownloadResult(
                route=suggestion.route,
                repo_id=suggestion.repo_id,
                command=suggestion.command,
                status="completed" if returncode == 0 else "failed",
                returncode=returncode,
            )
        )
    return DownloadExecution(
        executed=True,
        results=tuple(results),
        notes=plan.notes,
    )


def _scan_model_dirs(paths: Sequence[Path]) -> tuple[DiscoveredModel, ...]:
    models: list[DiscoveredModel] = []
    seen: set[str] = set()
    for root in paths:
        if not root.exists() or not root.is_dir():
            continue
        for model in _ollama_manifest_models(root):
            if model.repo_id in seen:
                continue
            seen.add(model.repo_id)
            models.append(model)
        if _is_lm_studio_model_root(root):
            for model in _lm_studio_models(root):
                if model.repo_id in seen:
                    continue
                seen.add(model.repo_id)
                models.append(model)
            continue
        for candidate in _iter_model_candidates(root):
            model = _model_from_dir(candidate)
            if model.repo_id in seen:
                continue
            seen.add(model.repo_id)
            models.append(model)
    return tuple(models)


def _ollama_manifest_models(root: Path) -> tuple[DiscoveredModel, ...]:
    manifests = root / "manifests"
    if not manifests.is_dir():
        return ()
    models: list[DiscoveredModel] = []
    try:
        registries = sorted(path for path in manifests.iterdir() if path.is_dir())
    except OSError:
        return ()
    for registry in registries:
        try:
            namespaces = sorted(path for path in registry.iterdir() if path.is_dir())
        except OSError:
            continue
        for namespace in namespaces:
            try:
                model_dirs = sorted(path for path in namespace.iterdir() if path.is_dir())
            except OSError:
                continue
            for model_dir in model_dirs:
                try:
                    tags = sorted(path for path in model_dir.iterdir() if path.is_file())
                except OSError:
                    continue
                for tag in tags:
                    repo_id = (
                        f"{model_dir.name}:{tag.name}"
                        if namespace.name == "library"
                        else f"{namespace.name}/{model_dir.name}:{tag.name}"
                    )
                    models.append(
                        DiscoveredModel(
                            name=f"{model_dir.name}:{tag.name}",
                            repo_id=repo_id,
                            path=str(tag),
                            source="ollama",
                            roles=_infer_roles(repo_id),
                        )
                    )
    return tuple(models)


def _is_lm_studio_model_root(root: Path) -> bool:
    return root.name == "models" and root.parent.name in {".lmstudio", "LM Studio"}


def _lm_studio_models(root: Path) -> tuple[DiscoveredModel, ...]:
    models: list[DiscoveredModel] = []
    try:
        owners = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return ()
    for owner in owners:
        try:
            model_dirs = sorted(path for path in owner.iterdir() if path.is_dir())
        except OSError:
            continue
        for model_dir in model_dirs:
            if not _has_model_file_marker(model_dir):
                continue
            repo_id = f"{owner.name}/{model_dir.name}"
            models.append(
                DiscoveredModel(
                    name=model_dir.name,
                    repo_id=repo_id,
                    path=str(model_dir),
                    source="lm_studio",
                    roles=_infer_roles(repo_id),
                )
            )
    return tuple(models)


def _iter_model_candidates(root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > MODEL_SCAN_MAX_DEPTH:
            continue
        if directory != root and _looks_like_model_dir(directory, depth):
            candidates.append(directory)
            continue
        if depth >= MODEL_SCAN_MAX_DEPTH:
            continue
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in reversed(children):
            if child.is_dir() and not child.name.startswith("."):
                stack.append((child, depth + 1))
    return tuple(candidates)


def _looks_like_model_dir(path: Path, depth: int) -> bool:
    name = path.name.lower()
    if name in SKIP_MODEL_DIR_NAMES:
        return False
    if path.name.startswith("models--"):
        return True
    if _has_model_file_marker(path):
        return True
    return depth <= 2 and any(hint in name for hint in MODEL_NAME_HINTS)


def _has_model_file_marker(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        if not child.is_file():
            continue
        name = child.name.lower()
        if name in MODEL_FILE_MARKERS or name.endswith(MODEL_FILE_SUFFIXES):
            return True
    return False


def _model_from_dir(path: Path) -> DiscoveredModel:
    if path.name.startswith("models--"):
        repo_id = _hf_cache_repo_id(path.name)
        source = "huggingface_cache"
    else:
        repo_id = path.name
        source = "local_directory"
    return DiscoveredModel(
        name=path.name,
        repo_id=repo_id,
        path=str(path),
        source=source,
        roles=_infer_roles(repo_id),
    )


def _hf_cache_repo_id(name: str) -> str:
    parts = name.removeprefix("models--").split("--")
    if len(parts) < 2:
        return name
    return f"{parts[0]}/{'--'.join(parts[1:])}"


def _infer_roles(identifier: str) -> tuple[str, ...]:
    text = identifier.lower()
    roles: list[str] = []
    size_b = _model_size_billions(text)
    if any(token in text for token in ("vl", "vision", "image-text", "ocr")):
        roles.append("multimodal_vision")
    if any(token in text for token in ("diffusion", "sdxl", "stable-diffusion", "flux")):
        roles.append("image_generation")
    if any(token in text for token in ("code", "coder", "deepseek-coder")):
        roles.append("code_agent")
    if any(token in text for token in ("embed", "bge", "e5", "rerank")):
        roles.append("web_research")
    if (
        any(token in text for token in ("small", "mini"))
        or size_b is not None
        and size_b <= 1.5
    ):
        roles.append("fast_local")
    if (
        any(token in text for token in ("instruct", "chat"))
        or size_b is not None
        and 2 <= size_b <= 13
    ):
        roles.append("balanced_local")
    if any(token in text for token in ("reason",)) or size_b is not None and size_b >= 7:
        roles.append("reasoning_local")
    return tuple(dict.fromkeys(roles))


def _model_size_billions(text: str) -> float | None:
    sizes = [
        float(match.group(1))
        for match in re.finditer(r"(?<![a-z])(\d+(?:\.\d+)?)b\b", text)
    ]
    return max(sizes) if sizes else None


def _local_model_overrides(
    models: Sequence[DiscoveredModel],
    engine_overrides: dict[str, dict[str, Any]],
    notes: list[str],
) -> set[str]:
    matched_roles: set[str] = set()
    for model in models:
        for role in model.roles:
            if role in matched_roles:
                continue
            engine_name = ROLE_TO_ENGINE.get(role)
            if engine_name is None:
                continue
            matched_roles.add(role)
            engine_overrides[engine_name] = _engine_override_for_model(role, model)
            notes.append(f"Local model {model.repo_id} mapped to {engine_name}.")
    return matched_roles


def _engine_override_for_model(
    role: str,
    model: DiscoveredModel,
) -> dict[str, Any]:
    adapter = {
        "image_generation": "local_image_generation",
        "multimodal_vision": "local_vision",
        "code_agent": "local_code",
    }.get(role, "local_chat")
    return {
        "provider": model.source,
        "model": model.repo_id,
        "adapter": adapter,
        "enabled": True,
        "availability": {
            "status": "auto",
            "required_paths": [model.path],
        },
    }


def _engine_override_for_suggestion(suggestion: DownloadSuggestion) -> dict[str, Any]:
    return {
        "provider": suggestion.provider,
        "model": suggestion.repo_id,
        "adapter": suggestion.adapter,
        "enabled": True,
        "availability": {
            "status": "auto",
            "required_paths": [suggestion.command[-1]],
        },
    }


def _default_download_suggestions(
    profile: str,
    hardware: HardwareProfile,
) -> tuple[DownloadSuggestion, ...]:
    return tuple(
        _download_suggestion_for_advice(advice)
        for advice in recommend_catalog_models(profile=profile, hardware=hardware)
    )


def _download_suggestion_for_advice(advice: ModelAdvice) -> DownloadSuggestion:
    include_args: list[str] = []
    for pattern in advice.include:
        include_args.extend(("--include", pattern))
    return DownloadSuggestion(
        route=advice.route,
        repo_id=advice.repo_id,
        provider=advice.provider,
        adapter=advice.adapter,
        reason=advice.reason,
        command=(
            "hf",
            "download",
            advice.repo_id,
            *include_args,
            "--local-dir",
            f"models/{advice.route}/{_repo_slug(advice.repo_id)}",
        ),
    )


def _with_local_root(
    suggestion: DownloadSuggestion,
    local_root: str | Path | None,
) -> DownloadSuggestion:
    if local_root is None:
        return suggestion
    route_dir = Path(local_root).expanduser() / suggestion.route / _repo_slug(
        suggestion.repo_id
    )
    command = (*suggestion.command[:-1], str(route_dir))
    return DownloadSuggestion(
        route=suggestion.route,
        repo_id=suggestion.repo_id,
        provider=suggestion.provider,
        adapter=suggestion.adapter,
        reason=suggestion.reason,
        command=command,
    )


def _run_download_command(command: tuple[str, ...]) -> int:
    executable = _resolve_command_executable(command[0])
    if executable is None:
        raise FileNotFoundError(command[0])
    completed = subprocess.run((executable, *command[1:]), check=False)
    return int(completed.returncode)


def _resolve_command_executable(command_name: str) -> str | None:
    executable = shutil.which(command_name)
    if executable is not None:
        return executable

    if os.environ.get("HERMES_ROUTER_COMMAND_DISCOVERY_PATH_ONLY"):
        return None

    sibling = Path(sys.executable).with_name(command_name)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def _adapter_for_route(route: str) -> str:
    if route == "image_generation":
        return "local_image_generation"
    if route == "multimodal_vision":
        return "local_vision"
    if route == "reasoning_local":
        return "local_reasoning"
    if route == "web_research":
        return "web_research"
    if route == "code_agent":
        return "local_code"
    return "local_chat"


def _repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _load_config_mapping(config_path: str | Path | None) -> dict[str, Any]:
    if config_path:
        path = Path(config_path).expanduser()
        source = str(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        source = default_config_source()
        data = yaml.safe_load(default_config_text())
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {source}")
    return data


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value
