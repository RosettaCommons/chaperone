"""Public block docs hide only fields explicitly marked internal."""

from pathlib import Path
import runpy

from pydantic import Field

from blocks.networks.prediction.alphafold3 import AlphaFold3Block
from blocks.networks.prediction.rosettafold3 import RosettaFold3Block
from blocks.networks.prediction.esmfold2 import ESMFold2Block


class APIOnlyFieldBlock(AlphaFold3Block):
    public_api_setting: dict = Field(
        default_factory=dict, json_schema_extra={"cli_type": None}
    )


def test_docs_hide_internal_revisions_but_keep_public_python_api_fields():
    render = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "docs/site/gen_block_docs.py")
    )["_render_block"]
    for cls, internal in [
        (APIOnlyFieldBlock, "input_preparation_version"),
        (RosettaFold3Block, "output_lineage_version"),
        (ESMFold2Block, "output_validation_version"),
    ]:
        docs = "\n".join(render("example", cls))
        assert internal not in docs
        if cls is APIOnlyFieldBlock:
            assert "public_api_setting" in docs
