"""Terminal control center for ModelRouter admin state.

The TUI is intentionally an admin/control surface, not a chat UI. The first
slice is read-only and renders shared admin state so future interactive actions
can reuse the same state/action boundary as the web settings UI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
import sys
from typing import Any

from hermes.plugins.model_router.admin import (
    ProxyProcessSupervisor,
    build_admin_state,
    settings_paths,
)
from hermes.plugins.model_router.product import DEFAULT_CONFIG_DIR


TEXTUAL_INSTALL_HINT = 'python -m pip install "hermes-router[tui]"'
TUI_TABS = (
    "Status",
    "Models",
    "Routing",
    "Runtimes",
    "Telemetry",
    "Logs",
    "Settings",
)


class TuiDependencyError(RuntimeError):
    """Raised when optional TUI dependencies are unavailable."""


@dataclass(frozen=True)
class TuiView:
    """Pure render state used by Textual and tests."""

    tabs: dict[str, str]

    def snapshot(self) -> str:
        sections: list[str] = []
        for tab in TUI_TABS:
            sections.append(f"## {tab}\n{self.tabs.get(tab, '').rstrip()}")
        return "\n\n".join(sections).rstrip() + "\n"


def build_tui_state(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    supervisor: ProxyProcessSupervisor | None = None,
) -> dict[str, Any]:
    """Build shared admin state for the TUI."""

    return build_admin_state(settings_paths(config_dir), supervisor)


def build_tui_view(state: Mapping[str, Any]) -> TuiView:
    """Render all TUI tabs from shared admin state."""

    return TuiView(
        tabs={
            "Status": _status_tab(state),
            "Models": _models_tab(state),
            "Routing": _routing_tab(state),
            "Runtimes": _runtimes_tab(state),
            "Telemetry": _telemetry_tab(state),
            "Logs": _logs_tab(state),
            "Settings": _settings_tab(state),
        }
    )


def render_tui_snapshot(state: Mapping[str, Any]) -> str:
    """Return a deterministic text snapshot for tests and fallback debugging."""

    return build_tui_view(state).snapshot()


def run_tui(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output: TextIOBase | None = None,
) -> int:
    """Run the Textual TUI, or print an install hint when Textual is missing."""

    stream = output or sys.stderr
    try:
        app = create_tui_app(config_dir=config_dir)
    except TuiDependencyError as exc:
        print(f"ModelRouter TUI requires Textual. Install it with:\n{exc}", file=stream)
        return 1
    result = app.run()
    return int(result or 0)


def create_tui_app(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    supervisor: ProxyProcessSupervisor | None = None,
) -> Any:
    """Create the Textual app with optional dependencies loaded lazily."""

    symbols = _load_textual_symbols()
    app_cls = symbols["App"]
    header_cls = symbols["Header"]
    footer_cls = symbols["Footer"]
    static_cls = symbols["Static"]
    tabbed_content_cls = symbols["TabbedContent"]
    tab_pane_cls = symbols["TabPane"]
    vertical_scroll_cls = symbols["VerticalScroll"]

    paths = settings_paths(config_dir)
    proxy_supervisor = supervisor or ProxyProcessSupervisor(
        config_path=paths["proxy_config"],
        log_path=paths["settings_proxy_log"],
    )

    class ModelRouterTui(app_cls):  # type: ignore[misc, valid-type]
        CSS = """
        Screen {
            background: $surface;
        }

        TabPane {
            padding: 1 2;
        }

        Static.panel {
            border: round $primary;
            padding: 1 2;
        }
        """
        BINDINGS = [
            ("r", "refresh", "Refresh"),
            ("q", "quit", "Quit"),
        ]
        TITLE = "ModelRouter"
        SUB_TITLE = "Local proxy control center"

        def __init__(self) -> None:
            super().__init__()
            self._state: dict[str, Any] = {}

        def compose(self) -> Any:
            yield header_cls()
            with tabbed_content_cls():
                for tab in TUI_TABS:
                    with tab_pane_cls(tab, id=_tab_id(tab)):
                        with vertical_scroll_cls():
                            yield static_cls("", id=_body_id(tab), classes="panel")
            yield footer_cls()

        def on_mount(self) -> None:
            self.action_refresh()

        def action_refresh(self) -> None:
            self._state = build_admin_state(paths, proxy_supervisor)
            view = build_tui_view(self._state)
            for tab, body in view.tabs.items():
                self.query_one(f"#{_body_id(tab)}", static_cls).update(body)

    return ModelRouterTui()


def _load_textual_symbols() -> dict[str, Any]:
    try:
        from textual.app import App
        from textual.containers import VerticalScroll
        from textual.widgets import Footer, Header, Static, TabbedContent, TabPane
    except ModuleNotFoundError as exc:
        name = str(getattr(exc, "name", "") or "")
        if name == "textual" or name.startswith("textual."):
            raise TuiDependencyError(TEXTUAL_INSTALL_HINT) from exc
        raise
    return {
        "App": App,
        "Footer": Footer,
        "Header": Header,
        "Static": Static,
        "TabbedContent": TabbedContent,
        "TabPane": TabPane,
        "VerticalScroll": VerticalScroll,
    }


def _status_tab(state: Mapping[str, Any]) -> str:
    proxy = _mapping(state.get("proxy"))
    process = _mapping(state.get("proxy_process"))
    observability = _mapping(state.get("observability"))
    receipt = _mapping(state.get("route_receipt"))
    lines = [
        _pair("Product", state.get("product", "ModelRouter")),
        _pair("Config valid", _yes_no(state.get("config_valid"))),
        _pair("Proxy endpoint", proxy.get("endpoint") or "not configured"),
        _pair("Proxy process", process.get("state") or "unknown"),
        _pair("Routing mode", proxy.get("routing_mode") or "decision"),
        _pair("Routing profile", proxy.get("routing_profile") or "balanced"),
        _pair("Decision layer", _yes_no(proxy.get("decision_layer_enabled"))),
        _pair("Telemetry", "on" if observability.get("enabled") else "off"),
    ]
    if receipt:
        lines.extend(
            [
                "",
                "Latest route receipt",
                _pair("Request", receipt.get("request_id") or "none"),
                _pair("Selected", receipt.get("selected") or "none"),
                _pair("Backend", receipt.get("backend") or "none"),
                _pair("Model", receipt.get("model") or "none"),
            ]
        )
    return "\n".join(lines)


def _models_tab(state: Mapping[str, Any]) -> str:
    library = _mapping(state.get("model_library"))
    installed = _list(library.get("installed"))
    recommended = _list(library.get("recommended"))
    downloads = _list(library.get("downloads"))
    assignments = _list(library.get("assignments"))
    lines = ["Installed"]
    if installed:
        lines.extend(
            _bullet(
                f"{item.get('display_name') or item.get('model_id')} "
                f"({item.get('source') or 'local'})"
            )
            for item in installed[:12]
            if isinstance(item, Mapping)
        )
    else:
        lines.append(_bullet("No local models found. Run model-router setup scan."))
    lines.extend(["", "Recommended downloads"])
    if recommended:
        lines.extend(
            _bullet(
                f"{item.get('model_id')} -> {', '.join(_string_list(item.get('route_fit'))) or 'route'} "
                f"[{item.get('score_label') or 'unscored'}]"
            )
            for item in recommended[:8]
            if isinstance(item, Mapping)
        )
    else:
        lines.append(_bullet("No recommendations available yet."))
    lines.extend(["", "Download plans"])
    if downloads:
        lines.extend(
            _bullet(f"{item.get('route')}: {item.get('model_id')} ({item.get('status')})")
            for item in downloads[:8]
            if isinstance(item, Mapping)
        )
    else:
        lines.append(_bullet("No planned downloads. Downloads require explicit confirmation."))
    lines.extend(["", "Assignments"])
    if assignments:
        lines.extend(
            _bullet(
                f"{item.get('route_id')} -> {item.get('backend')} / {item.get('model')}"
            )
            for item in assignments[:12]
            if isinstance(item, Mapping)
        )
    else:
        lines.append(_bullet("No route assignments are available."))
    return "\n".join(lines)


def _routing_tab(state: Mapping[str, Any]) -> str:
    proxy = _mapping(state.get("proxy"))
    route_map = _list(state.get("route_map"))
    lines = [
        _pair("Mode", proxy.get("routing_mode") or "decision"),
        _pair("Default backend", proxy.get("default_backend") or "not set"),
        _pair("Default model", proxy.get("default_model") or "not set"),
        _pair("Respect client model", _yes_no(proxy.get("respect_client_model"))),
        "",
        "Route map",
    ]
    if not route_map:
        lines.append(_bullet("No routing map. Initialize a proxy config first."))
        return "\n".join(lines)
    for row in route_map[:16]:
        if not isinstance(row, Mapping):
            continue
        selected = "*" if row.get("selected") else " "
        lines.append(
            f"{selected} {row.get('route_id')}: {row.get('provider')} -> "
            f"{row.get('target')} | fallback {row.get('fallback')}"
        )
    return "\n".join(lines)


def _runtimes_tab(state: Mapping[str, Any]) -> str:
    backends = _list(state.get("backends"))
    providers = _mapping(state.get("provider_runtime")).get("providers")
    provider_rows = _list(providers)
    lines = ["Backends"]
    if backends:
        for backend in backends[:16]:
            if not isinstance(backend, Mapping):
                continue
            adapter = _mapping(backend.get("runtime_adapter"))
            health = _mapping(adapter.get("health"))
            caps = _mapping(adapter.get("capabilities"))
            load = _mapping(caps.get("load_model"))
            lines.append(
                _bullet(
                    f"{backend.get('name')}: {adapter.get('provider') or 'runtime'} "
                    f"{health.get('status') or 'unknown'}; "
                    f"load={_support(load)}"
                )
            )
    else:
        lines.append(_bullet("No backends configured."))
    lines.extend(["", "Provider status"])
    if provider_rows:
        lines.extend(
            _bullet(f"{item.get('name')}: {item.get('status')} ({item.get('detail')})")
            for item in provider_rows[:16]
            if isinstance(item, Mapping)
        )
    else:
        lines.append(_bullet("No provider status available."))
    return "\n".join(lines)


def _telemetry_tab(state: Mapping[str, Any]) -> str:
    telemetry = _mapping(state.get("telemetry"))
    recent = _list(state.get("recent_events"))
    lines = [
        _pair("Events", telemetry.get("events", 0)),
        _pair("Feedback labels", telemetry.get("feedback_labels", 0)),
        _pair("Outcome labels", _compact_counts(telemetry.get("outcome_label_counts"))),
        _pair("Unlabeled replayable", telemetry.get("unlabeled_replayable", 0)),
        _pair("Mismatches", telemetry.get("expected_mismatch_count", 0)),
        _pair("Fallbacks", telemetry.get("fallback_count", 0)),
        _pair("Usage events", telemetry.get("usage_events", 0)),
        _pair("Usage tokens", _compact_usage(telemetry)),
        _pair("Engines", _compact_counts(telemetry.get("selected_engine_counts"))),
        _pair("Backends", _compact_counts(telemetry.get("backend_counts"))),
        _pair("Usage by backend", _compact_usage_groups(telemetry.get("usage_by_backend"))),
        "",
        "Recent requests",
    ]
    if recent:
        lines.extend(
            _bullet(
                f"{item.get('request_id')}: {item.get('selected_engine')} -> "
                f"{item.get('backend')} ({item.get('status')}; "
                f"tokens={item.get('usage_tokens') or 'none'})"
            )
            for item in recent[:10]
            if isinstance(item, Mapping)
        )
    else:
        lines.append(_bullet("No telemetry yet. Start the proxy and send requests."))
    return "\n".join(lines)


def _logs_tab(state: Mapping[str, Any]) -> str:
    paths = _mapping(state.get("paths"))
    observability = _mapping(state.get("observability"))
    process = _mapping(state.get("proxy_process"))
    backends = _list(state.get("backends"))
    lines = [
        _pair("Telemetry log", observability.get("log_path") or paths.get("events")),
        _pair("Feedback log", paths.get("feedback")),
        _pair("Settings proxy log", process.get("log_path") or paths.get("settings_proxy_log")),
        "",
        "Runtime logs",
    ]
    runtime_logs: list[str] = []
    for backend in backends:
        if not isinstance(backend, Mapping):
            continue
        adapter = _mapping(backend.get("runtime_adapter"))
        logs = _mapping(adapter.get("logs"))
        for path in _string_list(logs.get("paths")):
            runtime_logs.append(f"{backend.get('name')}: {path}")
    if runtime_logs:
        lines.extend(_bullet(item) for item in runtime_logs[:16])
    else:
        lines.append(_bullet("No runtime log paths are configured."))
    return "\n".join(lines)


def _settings_tab(state: Mapping[str, Any]) -> str:
    paths = _mapping(state.get("paths"))
    actions = _list(state.get("actions"))
    proxy = _mapping(state.get("proxy"))
    maturity = _mapping(state.get("maturity"))
    maturity_features = _list(maturity.get("features"))
    lines = [
        "TUI v1 is read-only. Mutating shared actions require confirm=true.",
        "No chat surface. No prompt transcript.",
        "",
        _pair("Config dir", paths.get("config_dir")),
        _pair("Proxy config", paths.get("proxy_config")),
        _pair("Router config", paths.get("model_router_config")),
        _pair("Prompt capture", _mapping(state.get("observability")).get("prompt_capture")),
        _pair("Safety gate mode", proxy.get("safety_gate_mode")),
        "",
        "Feature maturity",
    ]
    if maturity_features:
        for feature in maturity_features:
            if not isinstance(feature, Mapping):
                continue
            label = feature.get("label") or feature.get("feature_id") or "feature"
            lines.append(_bullet(f"{label}: {feature.get('maturity', 'unknown')}"))
    else:
        lines.append(_bullet("No maturity metadata available."))
    lines.extend(
        [
            "",
            "Shared actions",
        ]
    )
    if actions:
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            marker = "requires confirm" if action.get("requires_confirm") else "read-only"
            lines.append(_bullet(f"{action.get('id')}: {action.get('label')} ({marker})"))
    else:
        lines.append(_bullet("No shared action descriptors available."))
    return "\n".join(lines)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _pair(label: str, value: Any) -> str:
    return f"{label}: {_safe(value)}"


def _bullet(value: Any) -> str:
    return f"- {_safe(value)}"


def _safe(value: Any) -> str:
    text = str(value if value is not None else "none")
    text = " ".join(text.split())
    if len(text) > 160:
        return text[:157] + "..."
    return text


def _yes_no(value: Any) -> str:
    return "yes" if value is True else "no"


def _support(value: Mapping[str, Any]) -> str:
    if value.get("supported") is True:
        return "supported"
    return "disabled"


def _compact_counts(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "none"
    return ", ".join(f"{key}={count}" for key, count in sorted(value.items()))


def _compact_usage(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    prompt = _usage_int(value.get("usage_prompt_tokens"))
    completion = _usage_int(value.get("usage_completion_tokens"))
    total = _usage_int(value.get("usage_total_tokens"))
    cached = _usage_int(value.get("usage_cached_input_tokens"))
    if prompt == completion == total == cached == 0:
        return "none"
    parts = [f"prompt={prompt}", f"completion={completion}", f"total={total}"]
    if cached:
        parts.append(f"cached_input={cached}")
    return ", ".join(parts)


def _compact_usage_groups(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "none"
    parts = []
    for key, usage in sorted(value.items()):
        formatted = _compact_usage(usage)
        if formatted != "none":
            parts.append(f"{key}={formatted}")
    return "; ".join(parts) if parts else "none"


def _usage_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _tab_id(tab: str) -> str:
    return tab.lower().replace(" ", "-")


def _body_id(tab: str) -> str:
    return f"{_tab_id(tab)}-body"
