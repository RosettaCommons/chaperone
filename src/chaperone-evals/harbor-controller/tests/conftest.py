"""Test configuration: the controller lives inside the Chaperone checkout."""

from __future__ import annotations

import sys
import pytest
from pathlib import Path

# tests -> harbor-controller -> chaperone-evals -> src -> repository root
CHAPERONE_REPO = Path(__file__).resolve().parents[4]
CHAPERONE_SOURCE = CHAPERONE_REPO / "src"
for site_packages in sorted(
    (CHAPERONE_REPO / ".venv" / "lib").glob("python*/site-packages")
):
    if site_packages.is_dir():
        sys.path.append(str(site_packages))
if CHAPERONE_SOURCE.is_dir():
    sys.path.insert(0, str(CHAPERONE_SOURCE))


@pytest.fixture(autouse=True)
def historical_digs_policy(monkeypatch):
    """Existing fixtures describe digs; portability tests explicitly override it."""
    monkeypatch.setenv("CHAPERONE_EVAL_SITE", "digs")
    for name in ("CHAPERONE_EVAL_SITE_CONFIG", "CHAPERONE_EVAL_AGENT_IMAGE",
                 "CHAPERONE_EVAL_WORKER_IMAGE", "CHAPERONE_EVAL_BASELINE_IMAGE"):
        monkeypatch.delenv(name, raising=False)
