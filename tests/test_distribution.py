"""Public distributions must contain only the framework packages."""

from pathlib import Path
import tomllib

from setuptools import find_namespace_packages


def test_package_discovery_excludes_evaluation_harness():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    config = project["tool"]["setuptools"]["packages"]["find"]
    packages = find_namespace_packages(
        where=str(root / config["where"][0]),
        include=config["include"],
        exclude=config.get("exclude", []),
    )
    assert {"blocks", "chaperone", "chaperone_cli"} <= set(packages)
    assert packages
    assert all(
        name.split(".")[0] in {"blocks", "chaperone", "chaperone_cli"}
        for name in packages
    )


def test_block_runtime_assets_are_declared_as_package_data():
    from fnmatch import fnmatch
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = project["tool"]["setuptools"]["package-data"]["blocks"]
    assets = ["scripts/ddg_protocol.xml", "scripts/flags_rescore_genpot",
              "scripts/install_usalign.sh", "utils/params/FAD.params"]
    for relative in assets:
        assert (root / "src/blocks" / relative).is_file()
        assert any(fnmatch(relative, pattern) for pattern in patterns), relative
