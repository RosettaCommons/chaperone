"""Regression coverage ported from upstream PR #356."""

import json

import numpy as np
import pytest
from biotite.structure import AtomArray

from chaperone import get_settings
from chaperone.data import PipelineDataHandler, make_row
from chaperone.data.schema import SELECTIONS, _get_selection_mask
from chaperone.execution_tracking import (
    load_execution_records,
    next_execution_attempt,
    track_execution,
)
from chaperone.manifest import PipelineManifest
from chaperone.orchestration import JobArgs, SlurmTaskRunner


def _write_one_record(step_dir):
    """Run a tracked execution so ``.execution`` holds exactly one real record."""
    sandbox_dir = step_dir / "chunk_0"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    with track_execution(
        step_dir=step_dir,
        sandbox_dir=sandbox_dir,
        block_name="rfd3",
        source="compiled_manifest",
        chunk_id=0,
    ):
        pass
    return sandbox_dir


def test_in_flight_temp_records_are_ignored(tmp_path):
    """A sibling worker's ``.tmp_*.json`` must not be read as a record.

    ``pathlib.Path.glob`` matches dotfiles (``glob.glob`` does not), so a bare
    ``*.json`` also enumerated the temp files ``_write_execution_record`` is
    about to ``os.replace()`` into place -- double-counting the attempt once
    the rename lands.
    """
    step_dir = tmp_path / "rfd3"
    sandbox_dir = _write_one_record(step_dir)

    execution_dir = step_dir / ".execution"
    real = next(iter(execution_dir.glob("*.json")))
    (execution_dir / ".tmp_concurrent.json").write_text(real.read_text())

    assert len(load_execution_records(step_dir)) == 1
    assert next_execution_attempt(step_dir, sandbox_dir) == 2


def test_record_that_cannot_be_opened_is_skipped(tmp_path):
    """A record that is enumerated but unreadable must not kill the worker.

    The observed failure was a ``FileNotFoundError`` on a ``.tmp_*.json`` that
    another process renamed away between the glob and the open, which took down
    all 1,457 tasks of a job array. A dangling symlink reproduces the same
    listed-then-unopenable sequence without a race.
    """
    step_dir = tmp_path / "rfd3"
    sandbox_dir = _write_one_record(step_dir)

    execution_dir = step_dir / ".execution"
    (execution_dir / "vanished.json").symlink_to(execution_dir / "gone.json")

    assert len(load_execution_records(step_dir)) == 1
    assert next_execution_attempt(step_dir, sandbox_dir) == 2


def test_load_tolerates_unknown_step_field(tmp_path):
    """A step field added by a newer version must not break older readers.

    pipeline.json is read from shared rundirs by checkouts other than the
    one that wrote it, so an unknown key has to be dropped, not raised on.
    """
    (tmp_path / "pipeline.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "block_name": "copy",
                        "name": "copy1",
                        "pipeline_input_names": ["input"],
                        "parquet": ".pipeline/copy1.parquet",
                        "some_future_field": 123,
                        "execution_summary": None,
                    }
                ],
                "completed_updates": [],
            }
        )
    )

    loaded = PipelineManifest.load(tmp_path)
    assert loaded.step_names() == ["copy1"]
    step = loaded.get_step("copy1")
    assert step.block_name == "copy"
    assert step.pipeline_input_names == ["input"]
    assert not hasattr(step, "some_future_field")


def test_load_tolerates_unknown_execution_summary_field(tmp_path):
    """Same tolerance for the nested ExecutionSummary dict."""
    (tmp_path / "pipeline.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "block_name": "copy",
                        "name": "copy1",
                        "pipeline_input_names": [],
                        "parquet": ".pipeline/copy1.parquet",
                        "execution_summary": {
                            "records": 2,
                            "wall_seconds": 12.5,
                            "future_metric": 9,
                        },
                    }
                ],
                "completed_updates": [],
            }
        )
    )

    loaded = PipelineManifest.load(tmp_path)
    summary = loaded.get_step("copy1").execution_summary
    assert summary.records == 2
    assert summary.wall_seconds == 12.5


def test_load_tolerates_missing_optional_field(tmp_path):
    """A manifest predating an optional field still loads (default applies)."""
    (tmp_path / "pipeline.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "block_name": "copy",
                        "name": "copy1",
                        "pipeline_input_names": [],
                        "parquet": ".pipeline/copy1.parquet",
                    }
                ],
                "completed_updates": [],
            }
        )
    )

    loaded = PipelineManifest.load(tmp_path)
    assert loaded.get_step("copy1").execution_summary is None


def test_inner_merge_with_empty_child_returns_empty():
    """A block that produced no output must not pass its input through."""
    base = PipelineDataHandler(
        [make_row(name="mpnn", path="/runs/mpnn/design_001.pdb")]
    )
    empty = PipelineDataHandler(base._df.clear())

    merged = PipelineDataHandler.merge(base, empty)
    assert len(merged) == 0
    assert merged._df.schema == base._df.schema


def test_left_merge_with_empty_child_preserves_base():
    """how='left' keeps base rows -- existing behaviour, must not regress."""
    base = PipelineDataHandler(
        [make_row(name="mpnn", path="/runs/mpnn/design_001.pdb")]
    )
    empty = PipelineDataHandler(base._df.clear())

    merged = PipelineDataHandler.merge(base, empty, how="left")
    assert len(merged) == 1


def test_slurm_kwargs_not_shared_across_runners(tmp_path):
    """Two runners built without slurm_kwargs must not share one dict.

    ``slurm_kwargs: Any = {}`` aliased a single dict across every call, and
    __init__ mutates it (``exclude``, and the compute_backend pop), so one
    step's excluded nodes silently applied to every later step of the run.
    """
    from chaperone.orchestration import JobArgs

    ja_excl = JobArgs(partition="gpu", exclude="g2605", time_limit="00:10:00")
    r_excl = SlurmTaskRunner(job_args=ja_excl, log_folder=str(tmp_path))
    assert r_excl.slurm_kwargs.get("exclude") == "g2605"

    ja_plain = JobArgs(partition="cpu", time_limit="00:10:00")
    r_plain = SlurmTaskRunner(job_args=ja_plain, log_folder=str(tmp_path))
    assert "exclude" not in r_plain.slurm_kwargs, (
        f"runner inherited a stale exclude from a prior runner: "
        f"{r_plain.slurm_kwargs!r}"
    )
    assert r_excl.slurm_kwargs is not r_plain.slurm_kwargs


@pytest.mark.parametrize(
    "selection", [SELECTIONS.CA, SELECTIONS.BKBN, SELECTIONS.BKBN_NO_OXYGEN]
)
def test_backbone_selection_excludes_ligand_atoms(selection):
    atoms = AtomArray(8)
    atoms.atom_name = ["N", "CA", "C", "O"] * 2
    atoms.hetero = [False] * 4 + [True] * 4
    mask = _get_selection_mask(selection, atoms, {})
    assert not mask[4:].any()
    expected = {
        SELECTIONS.CA: [False, True, False, False],
        SELECTIONS.BKBN: [True] * 4,
        SELECTIONS.BKBN_NO_OXYGEN: [True, True, True, False],
    }[selection]
    np.testing.assert_array_equal(mask[:4], expected)


def test_dssr_constructs_without_site_configuration(monkeypatch):
    from blocks.metrics.dssr_snap_metrics import DSSRSnapMetricsBlock

    # Keep the real get_setting_field: a permissive stub would hide the bug.
    monkeypatch.setattr(get_settings(), "X3DNA_BIN_DIR", None)
    monkeypatch.setattr(get_settings(), "X3DNA_DIR", None)
    block = DSSRSnapMetricsBlock()
    assert block.x3dna_dssr_bin is None
    assert block.x3dna_home is None


def test_slurm_runner_does_not_mutate_caller_kwargs(tmp_path):
    kwargs = {"constraint": "Ampere"}
    runner = SlurmTaskRunner(
        job_args=JobArgs(exclude="g2605"), slurm_kwargs=kwargs, log_folder=str(tmp_path)
    )
    assert kwargs == {"constraint": "Ampere"}
    assert runner.slurm_kwargs["exclude"] == "g2605"


def test_unreadable_record_does_not_hide_readable_records(tmp_path, monkeypatch):
    import builtins
    from pathlib import Path

    step_dir = tmp_path / "rfd3"
    sandbox_dir = _write_one_record(step_dir)
    unreadable = step_dir / ".execution" / "unreadable.json"
    unreadable.write_text("{}")
    real_open = builtins.open

    def open_record(path, *args, **kwargs):
        if Path(path) == unreadable:
            raise PermissionError("record temporarily unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_record)
    assert len(load_execution_records(step_dir)) == 1
    assert next_execution_attempt(step_dir, sandbox_dir) == 2
