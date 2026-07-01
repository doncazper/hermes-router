import json

import hermes.plugins.model_router.runtime_install as runtime_install
from hermes.plugins.model_router.product import initialize_product_config


def test_runtime_status_report_surfaces_imported_runtime_models(tmp_path, monkeypatch):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    def fake_runtime_state(backend, *, timeout_seconds=0.25):
        del timeout_seconds
        return {
            "runtime_id": "lmstudio",
            "provider": "lmstudio",
            "runtime_kind": "lmstudio",
            "endpoint": backend.base_url,
            "detected": True,
            "last_checked_at": "2026-06-30T12:00:00Z",
            "health": {"status": "ready", "ok": True},
            "models": [
                {
                    "model_id": f"runtime-visible-{backend.name}",
                    "loaded": False,
                    "source": "lmstudio_api",
                }
            ],
            "loaded_models": [],
            "capabilities": {"discover_models": {"supported": True}},
            "logs": {"supported": False},
        }

    monkeypatch.setattr(runtime_install, "runtime_state_for_backend", fake_runtime_state)

    report = runtime_install.runtime_status_report(tmp_path / "routing_proxy.yaml")

    assert report["ok"] is True
    assert report["imported_model_count"] >= 1
    imported = {
        item["model_id"]: item
        for item in report["imported_models"]
        if item["source"] == "runtime_import"
    }
    assert imported["runtime-visible-fast"]["runtime_id"] == "lmstudio"
    assert imported["runtime-visible-fast"]["routing_eligible"] is True
    assert imported["runtime-visible-fast"]["metadata"]["runtime_source"] == "lmstudio_api"
    json.dumps(report)
