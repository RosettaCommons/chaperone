from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT.parent / "evals_task" / "ppi_pdl1" / "prompt.md"


def _tokens(path: Path) -> int:
    tiktoken = pytest.importorskip("tiktoken")
    return len(tiktoken.get_encoding("o200k_base").encode(path.read_text()))


def test_controller_and_task_contracts_stay_bounded_and_separate() -> None:
    controller = ROOT / "instruction_harness.md"
    assert _tokens(controller) <= 1_150
    assert _tokens(TASK) <= 1_150
    assert "minimum interaction PAE" not in controller.read_text().lower()
    assert 'gres="gpu:a4000' not in TASK.read_text()


def test_sequence_table_is_the_only_model_authored_deliverable() -> None:
    instruction = (ROOT / "instruction_harness.md").read_text()
    assert "only evaluation deliverable" in instruction
    assert "submission/designs.csv" in instruction
    assert "controller measures compute" in instruction
    assert "terminal_report.json" not in instruction
    assert "lineage.jsonl" not in instruction
    assert "submission/run_meta.json" not in instruction
