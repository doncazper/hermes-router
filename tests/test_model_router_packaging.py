from importlib import resources
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import tomllib

from hermes.plugins.model_router.config import default_config_source, load_router_config


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_console_script_without_unverified_plugin_entry_point():
    with open(ROOT / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    assert (
        project["scripts"]["hermes-router"]
        == "hermes.plugins.model_router.cli:main"
    )
    assert "entry-points" not in project


def test_default_config_loads_from_package_resource_outside_repo_cwd(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    config = load_router_config()

    assert config.source_path == default_config_source()
    assert config.routing_targets["coding"] == "code_agent"
    assert config.get_engine("fast_local") is not None


def test_packaged_default_config_resource_exists():
    config_resource = resources.files("hermes.plugins.model_router").joinpath(
        "data",
        "model_router.yaml",
    )

    assert config_resource.is_file()


def test_packaged_default_config_matches_repo_default_config():
    config_resource = resources.files("hermes.plugins.model_router").joinpath(
        "data",
        "model_router.yaml",
    )

    assert config_resource.read_text(encoding="utf-8") == (
        ROOT / "configs" / "model_router.yaml"
    ).read_text(encoding="utf-8")


def test_wheel_contains_console_script_and_packaged_config(tmp_path):
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(ROOT),
                "--no-deps",
                "--wheel-dir",
                str(tmp_path),
            ],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        shutil.rmtree(ROOT / "build", ignore_errors=True)
        shutil.rmtree(ROOT / "hermes_router.egg-info", ignore_errors=True)

    assert result.returncode == 0, result.stderr
    wheels = sorted(tmp_path.glob("hermes_router-*.whl"))
    assert wheels

    with zipfile.ZipFile(wheels[-1]) as wheel:
        names = set(wheel.namelist())
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = wheel.read(entry_points_name).decode("utf-8")

    assert "hermes/plugins/model_router/data/model_router.yaml" in names
    assert "hermes-router = hermes.plugins.model_router.cli:main" in entry_points
    assert "hermes_agent.plugins" not in entry_points
