"""Parse MPNN's per-sequence ``overall_confidence`` and ``seq_rec`` from the FASTA.

Useful as a pre-AF3 selection step: feed N MPNN sequences per parent, then a
parent-based quantile filter keeps the top-K based on this metric so AF3 only
runs on the highest-confidence sequence(s).

Pattern:

    blocks = [
        ...,
        MPNN(num_seq_per_pdb=8, hbond_fix=True),
        MpnnScore(name="mpnn_score"),
        Filter(name="select_top",
               filters={"mpnn_score.overall_confidence":
                        MetricFilter(quantile=1/8,
                                     quantile_direction="higher",
                                     parent_based=True)}),
        AF3(...),
        ...
    ]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import ClassVar

from chaperone import InlineMetricBlock, JobArgs, PipelineDataHandler
from chaperone.data.schema import Column, dump_metrics, get_deserialized

logger = logging.getLogger(__name__)

_HEADER_ID_RE = re.compile(r"\bid=(\d+)\b")
_CONF_RE = re.compile(r"overall_confidence=([\d.eE+-]+)")
_REC_RE = re.compile(r"seq_rec=([\d.eE+-]+)")
_LIG_CONF_RE = re.compile(r"ligand_confidence=([\d.eE+-]+)")


class MpnnScoreBlock(InlineMetricBlock):
    """Attach MPNN per-sequence confidence (from FASTA) as metrics to each row."""

    _name = "mpnn_score"
    _display_name = "MPNN Score"
    _description = "Parse MPNN FASTA confidences as ranking metrics"
    _creates_new_step = False
    _accepts_sequence_input = True
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu", rows_per_job=2000)

    def get_data_kwargs(self):
        return {"which": "last", "load": False}

    def compute_metrics(self, loaded, *, context=None, index=0):
        return self._extract(loaded.path)

    def execute(self, handler, sandbox_dir=None, chunk_id=None):
        """Read embedded FASTA metrics for sequence-native MPNN outputs."""
        if not handler.has_sequence_artifacts:
            return super().execute(handler, sandbox_dir, chunk_id)

        rows = []
        for row in handler._df.iter_rows(named=True):
            metrics = get_deserialized(row, Column.METRICS)
            source = metrics.get("mpnn") or metrics.get("foundry_mpnn") or {}
            value = {
                "overall_confidence": source.get(
                    "overall_confidence", source.get("confidence")
                ),
                "seq_rec": source.get(
                    "seq_rec", source.get("sequence_recovery")
                ),
                "ligand_confidence": source.get(
                    "ligand_confidence", source.get("ligand_interface_confidence")
                ),
            }
            metric_row = dict(row)
            metric_row[str(Column.METRICS)] = dump_metrics({self.name: value})
            metric_row[str(Column.PARENT_DESIGN_ID)] = None
            rows.append(metric_row)
        return PipelineDataHandler(rows)

    def _extract(self, structure_path: str) -> dict:
        p = Path(structure_path)
        # MPNN packed naming convention: <base>__<seq_id>_<sample>.pdb
        m = re.match(r"^(.+)__(\d+)_(\d+)$", p.stem)
        if not m:
            logger.warning("mpnn_score: cannot parse seq_id from %s", p.stem)
            return self._empty()
        base, seq_id = m.group(1), int(m.group(2))

        # Sibling FASTA lives in ../seqs/<base>.{cif.gz.fa, fa, pdb.fa}
        seqs_dir = p.parent.parent / "seqs"
        candidates = [
            seqs_dir / f"{base}.cif.gz.fa",
            seqs_dir / f"{base}.cif.fa",
            seqs_dir / f"{base}.pdb.fa",
            seqs_dir / f"{base}.fa",
        ]
        fa = next((c for c in candidates if c.exists()), None)
        if fa is None:
            logger.warning(
                "mpnn_score: no FASTA next to %s (looked under %s)", p, seqs_dir
            )
            return self._empty()

        for line in fa.read_text().splitlines():
            if not line.startswith(">"):
                continue
            mid = _HEADER_ID_RE.search(line)
            if not mid or int(mid.group(1)) != seq_id:
                continue
            conf = _CONF_RE.search(line)
            recv = _REC_RE.search(line)
            ligc = _LIG_CONF_RE.search(line)
            return {
                "overall_confidence": float(conf.group(1)) if conf else None,
                "seq_rec": float(recv.group(1)) if recv else None,
                "ligand_confidence": float(ligc.group(1)) if ligc else None,
            }
        logger.warning("mpnn_score: id=%d not found in %s", seq_id, fa)
        return self._empty()

    @staticmethod
    def _empty() -> dict:
        return {"overall_confidence": None, "seq_rec": None, "ligand_confidence": None}
