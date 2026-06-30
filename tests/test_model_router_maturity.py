from hermes.plugins.model_router.maturity import feature_maturity_state, maturity_label


def test_feature_maturity_state_covers_release_gated_surfaces():
    state = feature_maturity_state()
    features = {feature["feature_id"]: feature for feature in state["features"]}

    assert set(features) == {
        "basic_router_mode",
        "installer",
        "model_library",
        "runtime_adapters",
        "tui",
        "compatibility_endpoints",
    }
    assert features["basic_router_mode"]["maturity"] == "beta"
    assert features["tui"]["maturity"] == "experimental"
    assert maturity_label("runtime_adapters") == "beta"
    assert maturity_label("unknown") == "unknown"
    assert "python -m pytest" in state["release_gate"]["required_checks"]
    assert any("Manual-mode" in item for item in state["release_gate"]["manual_dogfood"])
