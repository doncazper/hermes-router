"""Feature maturity metadata for ModelRouter product surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureMaturity:
    feature_id: str
    label: str
    maturity: str
    summary: str
    release_gate: str
    docs: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FEATURE_MATURITY: tuple[FeatureMaturity, ...] = (
    FeatureMaturity(
        feature_id="basic_router_mode",
        label="Basic router mode",
        maturity="beta",
        summary=(
            "Decision mode is stable and remains default; manual mode is the "
            "first supported decision-layer-off path."
        ),
        release_gate="Dogfood both decision and manual/basic mode before release.",
        docs="docs/codex/productization-roadmap.md#M1-basic-router-mode",
    ),
    FeatureMaturity(
        feature_id="installer",
        label="Installer onboarding",
        maturity="beta",
        summary="Plan-only onboarding with no silent downloads, services, or config overwrites.",
        release_gate="Run installer JSON tests and confirm first-run next commands.",
        docs="docs/model-router.md",
    ),
    FeatureMaturity(
        feature_id="model_library",
        label="Model library",
        maturity="beta",
        summary="Settings model surfaces are data-backed and confirmation-gated.",
        release_gate="Verify installed/discover/recommended/download/assignment empty states.",
        docs="docs/product-north-star.md",
    ),
    FeatureMaturity(
        feature_id="runtime_adapters",
        label="Runtime adapters",
        maturity="beta",
        summary="Adapters report health, model visibility, capabilities, and disabled reasons.",
        release_gate="Dogfood LM Studio, Ollama, llama.cpp, and MLX-LM where available.",
        docs="docs/model-router.md#optional-proxy-and-future-adapters",
    ),
    FeatureMaturity(
        feature_id="tui",
        label="TUI control center",
        maturity="experimental",
        summary="Read-only Textual control center backed by shared admin state.",
        release_gate="Keep TUI read-only until interactive confirmations are dogfooded.",
        docs="docs/codex/productization-roadmap.md#M5-tui-v1",
    ),
    FeatureMaturity(
        feature_id="compatibility_endpoints",
        label="Compatibility endpoints",
        maturity="beta",
        summary=(
            "Chat, Responses, embeddings, completions, models, and shaped "
            "unsupported endpoint responses share the proxy forwarding path."
        ),
        release_gate="Run proxy dogfood and fake-upstream compatibility tests.",
        docs="docs/model-router.md#optional-proxy-and-future-adapters",
    ),
)


def feature_maturity_state() -> dict[str, Any]:
    """Return shared maturity metadata for doctor, settings, TUI, and releases."""

    features = [feature.to_dict() for feature in FEATURE_MATURITY]
    counts: dict[str, int] = {}
    for feature in FEATURE_MATURITY:
        counts[feature.maturity] = counts.get(feature.maturity, 0) + 1
    return {
        "status": "release_candidate",
        "levels": {
            "stable": "Expected to remain compatible within this product line.",
            "beta": "Usable, tested, and release-gated, but still dogfood-sensitive.",
            "experimental": "Available for dogfood; behavior may change.",
            "planned": "Documented direction, not implemented yet.",
        },
        "features": features,
        "counts": dict(sorted(counts.items())),
        "release_gate": {
            "required_checks": [
                "python -m ruff check .",
                "python -m pytest",
                "python scripts/check_route_fast_latency.py --json",
                "model-router dogfood proxy --config <routing_proxy.yaml>",
                "model-router dogfood proxy --config <routing_proxy.yaml> --execute",
            ],
            "manual_dogfood": [
                "Decision-mode proxy smoke through a real local backend.",
                "Manual-mode proxy smoke through a configured default backend/model.",
                "Settings or TUI review of maturity labels and runtime status.",
            ],
        },
    }


def maturity_label(feature_id: str) -> str:
    """Return a short maturity label for display surfaces."""

    for feature in FEATURE_MATURITY:
        if feature.feature_id == feature_id:
            return feature.maturity
    return "unknown"
