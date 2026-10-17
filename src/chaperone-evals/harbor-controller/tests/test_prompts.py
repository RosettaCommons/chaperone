"""Task prompts must track --required-n instead of hard-coding a count."""

from __future__ import annotations

import re

import pytest

from run_eval import REQUIRED_N_PLACEHOLDER, render_prompt
from task_paths import PUBLIC_TASKS_ROOT

TASKS = sorted(path.name for path in PUBLIC_TASKS_ROOT.iterdir() if path.is_dir())


def test_every_task_is_covered():
    assert set(TASKS) == {
        "dna_binder",
        "enzyme_glycosidase",
        "enzyme_protease",
        "nanoparticle_c3",
        "peptide_binder",
        "ppi_pdl1",
        "ribozyme_design",
        "rna_binder",
    }


@pytest.mark.parametrize("task", TASKS)
def test_prompt_uses_the_placeholder_not_a_literal_count(task):
    text = (PUBLIC_TASKS_ROOT / task / "prompt.md").read_text()

    assert REQUIRED_N_PLACEHOLDER in text
    assert not re.search(r"^REQUIRED_N\s*=\s*\d+", text, re.MULTILINE)


@pytest.mark.parametrize("task", TASKS)
def test_rendered_prompt_carries_the_requested_count_only(task):
    text = (PUBLIC_TASKS_ROOT / task / "prompt.md").read_text()

    rendered = render_prompt(text, 7, task)

    assert REQUIRED_N_PLACEHOLDER not in rendered
    assert "REQUIRED_N = 7" in rendered
    assert not re.search(r"(?<![\d.])96(?![\d.])", rendered)


def test_a_hard_coded_count_is_refused():
    with pytest.raises(SystemExit, match="hard-codes `REQUIRED_N = 96`"):
        render_prompt("# Task\n\nREQUIRED_N = 96\nSubmit {{REQUIRED_N}} rows.\n", 10)


def test_default_count_renders_without_a_stray_literal():
    assert "96 rows" in render_prompt("Submit {{REQUIRED_N}} rows.", 96)
