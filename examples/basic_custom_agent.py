"""Minimal custom-agent integration example.

This file intentionally does not call a model provider. It shows the boundary
most agent builders need: route once per user turn, map the selected engine to
your app's runtime config, then let your own agent/client execute the turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_router import ModelRouter


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str | None = None


ROUTE_MAP: dict[str, ModelSpec] = {
    "fast_local": ModelSpec(
        provider="lmstudio",
        model="lmstudio-community/Qwen3-0.6B-GGUF",
        base_url="http://localhost:1234/v1",
    ),
    "balanced_local": ModelSpec(
        provider="lmstudio",
        model="lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit",
        base_url="http://localhost:1234/v1",
    ),
    "reasoning_local": ModelSpec(
        provider="lmstudio",
        model="your-reasoning-model",
        base_url="http://localhost:1234/v1",
    ),
    "code_agent": ModelSpec(
        provider="lmstudio",
        model="your-coder-model",
        base_url="http://localhost:1234/v1",
    ),
    "web_research": ModelSpec(provider="current", model="current"),
    "multimodal_vision": ModelSpec(
        provider="lmstudio",
        model="your-vision-model",
        base_url="http://localhost:1234/v1",
    ),
    "image_generation": ModelSpec(provider="current", model="current"),
    "human_confirm": ModelSpec(provider="human", model="confirmation_required"),
}


def select_model_for_turn(router: ModelRouter, prompt: str) -> tuple[str, ModelSpec]:
    """Return the selected engine and host-app model spec for one prompt."""

    engine = router.route_fast(prompt)
    return engine, ROUTE_MAP.get(engine, ROUTE_MAP["human_confirm"])


def main() -> None:
    router = ModelRouter.from_config(validate_availability=False)
    prompt = "fix the repo and run tests"
    engine, spec = select_model_for_turn(router, prompt)
    print(f"engine={engine}")
    print(f"provider={spec.provider}")
    print(f"model={spec.model}")


if __name__ == "__main__":
    main()
