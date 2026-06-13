"""Setup helpers for configuring Hermes model-router engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import shutil

import yaml

from hermes.plugins.model_router.config import (
    DEFAULT_ROUTING_TARGETS,
    default_config_path,
)

DEFAULT_COMMANDS = (
    "claude",
    "codex",
    "hf",
    "ollama",
    "llama-server",
    "lmstudio",
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
    models: tuple[DiscoveredModel, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": dict(sorted(self.commands.items())),
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
class SetupRecommendation:
    routing_targets: dict[str, str]
    engine_overrides: dict[str, dict[str, Any]]
    download_suggestions: tuple[DownloadSuggestion, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_targets": dict(sorted(self.routing_targets.items())),
            "engine_overrides": self.engine_overrides,
            "download_suggestions": [
                suggestion.to_dict() for suggestion in self.download_suggestions
            ],
            "notes": list(self.notes),
        }


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
        Path("~/.cache/huggingface/hub").expanduser(),
        Path("~/.ollama/models").expanduser(),
        Path("~/Library/Application Support/LM Studio/models").expanduser(),
        Path("~/models").expanduser(),
    )


def scan_local_environment(
    *,
    model_dirs: Sequence[str | Path] | None = None,
    command_names: Sequence[str] = DEFAULT_COMMANDS,
) -> SetupDiscovery:
    paths = tuple(
        Path(path).expanduser()
        for path in (default_model_dirs() if model_dirs is None else model_dirs)
    )
    commands = {name: shutil.which(name) is not None for name in command_names}
    models = _scan_model_dirs(paths)
    return SetupDiscovery(
        commands=commands,
        model_dirs=tuple(str(path) for path in paths),
        models=models,
    )


def recommend_setup(
    discovery: SetupDiscovery,
    *,
    profile: str = "balanced",
) -> SetupRecommendation:
    routing_targets = dict(DEFAULT_ROUTING_TARGETS)
    engine_overrides: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

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

    matched_roles = _local_model_overrides(discovery.models, engine_overrides, notes)
    download_suggestions = tuple(
        suggestion
        for suggestion in _default_download_suggestions(profile)
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


def _scan_model_dirs(paths: Sequence[Path]) -> tuple[DiscoveredModel, ...]:
    models: list[DiscoveredModel] = []
    seen: set[str] = set()
    for root in paths:
        if not root.exists() or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            model = _model_from_dir(child)
            if model.repo_id in seen:
                continue
            seen.add(model.repo_id)
            models.append(model)
    return tuple(models)


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
    if any(token in text for token in ("vl", "vision", "image-text", "ocr")):
        roles.append("multimodal_vision")
    if any(token in text for token in ("diffusion", "sdxl", "stable-diffusion", "flux")):
        roles.append("image_generation")
    if any(token in text for token in ("code", "coder", "deepseek-coder")):
        roles.append("code_agent")
    if any(token in text for token in ("embed", "bge", "e5", "rerank")):
        roles.append("web_research")
    if any(token in text for token in ("0.5b", "0.6b", "1.5b", "small", "mini")):
        roles.append("fast_local")
    if any(token in text for token in ("3b", "4b", "7b", "8b", "instruct", "chat")):
        roles.append("balanced_local")
    if any(token in text for token in ("7b", "8b", "14b", "32b", "reason")):
        roles.append("reasoning_local")
    return tuple(dict.fromkeys(roles))


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


def _default_download_suggestions(profile: str) -> tuple[DownloadSuggestion, ...]:
    del profile
    specs = (
        (
            "fast_local",
            "Qwen/Qwen3-0.6B",
            "Fast local intent/filler/rewrite model.",
        ),
        (
            "balanced_local",
            "Qwen/Qwen2.5-3B-Instruct",
            "Balanced local summarization and ordinary chat model.",
        ),
        (
            "reasoning_local",
            "Qwen/Qwen3-8B",
            "Larger local reasoning seed for planning-heavy work.",
        ),
        (
            "multimodal_vision",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Vision/OCR seed for screenshots, charts, and diagrams.",
        ),
        (
            "image_generation",
            "stabilityai/sdxl-turbo",
            "Fast local diffusion seed for image-generation requests.",
        ),
    )
    return tuple(
        DownloadSuggestion(
            route=route,
            repo_id=repo_id,
            provider="huggingface",
            adapter=_adapter_for_route(route),
            reason=reason,
            command=(
                "hf",
                "download",
                repo_id,
                "--local-dir",
                f"models/{route}/{_repo_slug(repo_id)}",
            ),
        )
        for route, repo_id, reason in specs
    )


def _adapter_for_route(route: str) -> str:
    if route == "image_generation":
        return "local_image_generation"
    if route == "multimodal_vision":
        return "local_vision"
    if route == "reasoning_local":
        return "local_reasoning"
    return "local_chat"


def _repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _load_config_mapping(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path).expanduser() if config_path else default_config_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
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
