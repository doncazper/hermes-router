from importlib import resources
from pathlib import Path
import configparser
import io
import shutil
import subprocess
import sys
import zipfile

import tomllib

import model_router
from hermes.plugins.model_router.config import default_config_source, load_router_config
from hermes.plugins.model_router.product import preset_template_names


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_generic_package_metadata():
    with open(ROOT / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    assert project["name"] == "hermes-router"
    assert project["version"] == "0.6.1"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert (
        project["description"]
        == "ModelRouter: fast deterministic model routing and OpenAI-compatible proxying for custom AI agents"
    )
    assert project["urls"]["Homepage"] == "https://github.com/doncazper/model-router"
    assert project["urls"]["Repository"] == "https://github.com/doncazper/model-router"
    assert project["urls"]["Issues"] == "https://github.com/doncazper/model-router/issues"
    assert (
        project["scripts"]["hermes-router"]
        == "hermes.plugins.model_router.cli:main"
    )
    assert (
        project["scripts"]["model-router"]
        == "hermes.plugins.model_router.cli:main"
    )
    assert (
        project["scripts"]["model-router-proxy"]
        == "hermes.plugins.model_router.proxy:main"
    )
    assert project["optional-dependencies"]["proxy"] == [
        "fastapi>=0.115,<1",
        "httpx>=0.27,<1",
        "uvicorn>=0.30,<1",
    ]
    assert project["optional-dependencies"]["release"] == [
        "build>=1,<2",
        "twine>=5,<7",
    ]
    assert "entry-points" not in project
    assert "model_router*" in pyproject["tool"]["setuptools"]["packages"]["find"][
        "include"
    ]


def test_generic_public_import_path_reexports_router_api():
    assert model_router.ModelRouter.__name__ == "ModelRouter"
    assert callable(model_router.route_prompt)
    assert callable(model_router.score_prompt)
    assert callable(model_router.build_dispatch_plan)


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


def test_packaged_model_catalog_resource_exists():
    catalog_resource = resources.files("hermes.plugins.model_router").joinpath(
        "data",
        "model_catalog.yaml",
    )

    assert catalog_resource.is_file()


def test_packaged_proxy_example_resource_exists():
    proxy_resource = resources.files("hermes.plugins.model_router").joinpath(
        "data",
        "routing_proxy.example.yaml",
    )

    assert proxy_resource.is_file()


def test_packaged_provider_template_resources_exist():
    for template_name in preset_template_names():
        template_resource = resources.files("hermes.plugins.model_router").joinpath(
            "data",
            template_name,
        )

        assert template_resource.is_file(), template_name


def test_packaged_default_config_matches_repo_default_config():
    config_resource = resources.files("hermes.plugins.model_router").joinpath(
        "data",
        "model_router.yaml",
    )

    assert config_resource.read_text(encoding="utf-8") == (
        ROOT / "configs" / "model_router.yaml"
    ).read_text(encoding="utf-8")


def test_wheel_contains_console_scripts_generic_package_and_packaged_config(tmp_path):
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

    parser = configparser.ConfigParser()
    parser.read_file(io.StringIO(entry_points))

    assert "hermes-router = hermes.plugins.model_router.cli:main" in entry_points
    assert "model-router = hermes.plugins.model_router.cli:main" in entry_points
    assert "model-router-proxy = hermes.plugins.model_router.proxy:main" in entry_points
    assert "model_router/__init__.py" in names
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    assert "hermes/plugins/model_router/data/model_router.yaml" in names
    assert "hermes/plugins/model_router/data/model_catalog.yaml" in names
    assert "hermes/plugins/model_router/data/routing_proxy.example.yaml" in names
    for template_name in preset_template_names():
        assert f"hermes/plugins/model_router/data/{template_name}" in names
    assert set(parser.sections()) == {"console_scripts"}
