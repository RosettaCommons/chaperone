"""Prepare BindCraft presets and execute the upstream entry point.

This script is copied into the BindCraft container.  Keeping preset resolution
inside the image lets the Pipelines block expose small override dictionaries
without vendoring BindCraft's large JSON files into Pipelines itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BINDCRAFT_HOME = Path("/opt/BindCraft")


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _preset_path(value: str, category: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return BINDCRAFT_HOME / category / candidate.name


def _merge_config(source: Path, overrides_path: Path, destination: Path) -> Path:
    config = _read_json(source)
    config.update(_read_json(overrides_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BindCraft with merged presets")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--filters", default="default_filters")
    parser.add_argument("--advanced", default="default_4stage_multimer")
    parser.add_argument("--filter-overrides", required=True)
    parser.add_argument("--advanced-overrides", required=True)
    parser.add_argument("--config-dir", required=True)
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    filters = _merge_config(
        _preset_path(args.filters, "settings_filters"),
        Path(args.filter_overrides),
        config_dir / "filters.json",
    )
    advanced = _merge_config(
        _preset_path(args.advanced, "settings_advanced"),
        Path(args.advanced_overrides),
        config_dir / "advanced.json",
    )

    # BindCraft fills blank values similarly, but explicit in-image paths make
    # the generated run configuration self-describing and deterministic.
    advanced_config = _read_json(advanced)
    advanced_config.update(
        {
            "af_params_dir": str(BINDCRAFT_HOME / "params"),
            "dssp_path": str(BINDCRAFT_HOME / "functions" / "dssp"),
            "dalphaball_path": str(BINDCRAFT_HOME / "functions" / "DAlphaBall.gcc"),
        }
    )
    advanced.write_text(json.dumps(advanced_config, indent=2, sort_keys=True) + "\n")

    os.chdir(BINDCRAFT_HOME)
    os.execv(
        "/opt/bindcraft-env/bin/python",
        [
            "/opt/bindcraft-env/bin/python",
            "-u",
            str(BINDCRAFT_HOME / "bindcraft.py"),
            "--settings",
            str(Path(args.settings).resolve()),
            "--filters",
            str(filters),
            "--advanced",
            str(advanced),
        ],
    )


if __name__ == "__main__":
    main()
