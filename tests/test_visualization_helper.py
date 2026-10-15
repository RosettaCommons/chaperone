"""Plotting and file inspection consume the pipeline's selected rows."""

import gzip

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from blocks.util_blocks.filter import FilterBlock
from chaperone import PipelineDataHandler
from chaperone.data.schema import make_row
from chaperone.visualization_helper import plot_metrics, select_for_viewing


def test_plot_uses_filterblock_selection():
    handler = PipelineDataHandler(
        [
            make_row("input", f"/tmp/{i}.cif", metrics={"score": {"x": i}})
            for i in range(4)
        ]
    )
    selected = FilterBlock(expr="score.x >= 2")(handler)
    fig = plot_metrics(selected, metrics_col="score", bins=2)
    try:
        assert sum(p.get_height() for p in fig.axes[0].patches) == 2
        assert "n=2" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_plot_rejects_missing_values_before_sampling():
    handler = PipelineDataHandler(
        [
            make_row("input", "/tmp/a.cif", metrics={"score": {"x": 1}}),
            make_row("input", "/tmp/b.cif", metrics={"score": {"x": None}}),
        ]
    )
    with pytest.raises(ValueError, match="score.x.*1"):
        plot_metrics(handler, metrics_col="score.x", sample_size=1)


def test_viewing_copies_selected_gzip_and_preserves_existing_directory(tmp_path):
    source = tmp_path / "input.cif.gz"
    with gzip.open(source, "wt") as stream:
        stream.write("data_input\n#\n")
    handler = PipelineDataHandler.from_paths([source], name="input")
    output = tmp_path / "view"
    result = select_for_viewing(handler, output, path_col="input.path")
    from pathlib import Path

    copied = Path(result["copied_to"][0])
    assert copied.read_text() == "data_input\n#\n"
    with pytest.raises(FileExistsError):
        select_for_viewing(handler, output, path_col="input.path")
    assert copied.read_text() == "data_input\n#\n"
