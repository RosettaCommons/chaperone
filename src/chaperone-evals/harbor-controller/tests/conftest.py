"""Test configuration: the controller lives inside the Pipelines checkout."""

from __future__ import annotations

import sys
from pathlib import Path

# tests -> harbor-controller -> chaperone-evals -> src -> repository root
PIPELINES_REPO = Path(__file__).resolve().parents[4]
PIPELINES_SOURCE = PIPELINES_REPO / "src"
for site_packages in sorted(
    (PIPELINES_REPO / ".venv" / "lib").glob("python*/site-packages")
):
    if site_packages.is_dir():
        sys.path.append(str(site_packages))
if PIPELINES_SOURCE.is_dir():
    sys.path.insert(0, str(PIPELINES_SOURCE))
