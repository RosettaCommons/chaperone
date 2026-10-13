"""Order collection block for packaging design campaign results.

Extracts protein sequences from prediction structures, checks against an
ordered set, and collects prediction metrics — all in a SLURM-parallelizable
chunked block.
"""

import logging
import os
from pathlib import Path
from typing import List

from pydantic import Field

from pipelines import BlockFunction, PipelineDataHandler
from pipelines.data.schema import make_row

logger = logging.getLogger(__name__)


def _seq_fingerprint(seq_dict: dict) -> str:
    """Build a sequence fingerprint from get_sequences output.

    Concatenates protein chain sequences sorted by chain ID.
    Non-protein chains (ligands, ions) return None and are excluded.
    """
    return "|".join(f"{k}:{v}" for k, v in sorted(seq_dict.items()) if v is not None)


class OrderCollectionBlock(BlockFunction):
    """Extract sequences, check ordered status, and collect prediction metrics.

    For each input design, this block:
    1. Loads the structure to extract the protein sequence
    2. Reads prediction metrics from the parquet (no redundant file I/O)
    3. Checks whether the sequence is in the ordered set
    4. Outputs enriched metrics with sequence, ordered flag, and prediction data

    Usage::

        ordered_seqs = [...]  # from final filter
        engine.run(
            OrderCollectionBlock(
                prediction_step="af3",
                ordered_sequences=ordered_seqs,
            ),
            data=engine.compile(steps=["af3"]),
            job_args={"partition": "cpu", "rows_per_job": 500},
        )
    """

    _name = "order_collection"
    _description = "Extract sequences, check ordered status, collect metrics"
    _inline = True
    _chunked = True

    ordered_sequences: List[str] = Field(
        default_factory=list,
        description="Ordered sequence fingerprints (from final filter)",
    )
    prediction_step: str = Field(
        default="",
        description="Prediction step name to extract metrics from",
    )

    def __call__(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        **kwargs,
    ) -> PipelineDataHandler:
        ordered_set = set(self.ordered_sequences)
        rows = []

        for loaded in handler.get_data(which="last", load=True):
            seq_dict = loaded.sequence or {}
            protein_seq = _seq_fingerprint(seq_dict)

            # Metrics from parquet — already deserialized by get_data
            step_name = self.prediction_step or loaded.name
            step_metrics = loaded.metrics.get(step_name, {})

            enriched = {
                "sequence": protein_seq,
                "ordered": protein_seq in ordered_set,
                "design_name": loaded.basename,
                "structure_path": (
                    os.path.relpath(loaded.path, sandbox_dir.parent.parent)
                    if loaded.path
                    else ""
                ),
                "min_pae": step_metrics.get(
                    "min_pae_interaction", step_metrics.get("min_pae")
                ),
                "plddt": step_metrics.get("plddt"),
                "iptm": step_metrics.get("iptm"),
                "ptm": step_metrics.get("ptm"),
                "ranking_score": step_metrics.get("ranking_score"),
                "ligand_plddt": step_metrics.get("ligand_plddt"),
            }

            rows.append(
                make_row(
                    name=self.name,
                    path=loaded.path,
                    metrics={self.name: enriched},
                )
            )

        return PipelineDataHandler(rows)
