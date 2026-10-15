import logging
from enum import Enum
from typing import List

from biotite.structure import AtomArray, connect_via_residue_names
import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock

logger = logging.getLogger(__name__)


class PepprMetric(str, Enum):
    RMSD = "rmsd"
    TM_SCORE = "tm_score"
    LDDT = "lddt"
    LDDT_PLI = "lddt_pli"
    FNAT = "fnat"
    IRMSD = "irmsd"
    LRMSD = "lrmsd"
    DOCKQ = "dockq"


# Maps our config names to peppr class names
METRIC_CLASSES = {
    PepprMetric.RMSD: "MonomerRMSD",
    PepprMetric.TM_SCORE: "MonomerTMScore",
    PepprMetric.LDDT: "GlobalLDDTScore",
    PepprMetric.LDDT_PLI: "LDDTPLIScore",
    PepprMetric.FNAT: "ContactFraction",
    PepprMetric.IRMSD: "InterfaceRMSD",
    PepprMetric.LRMSD: "PocketAlignedLigandRMSD",
    PepprMetric.DOCKQ: "DockQScore",
}


def peppr_metrics(
    reference_atoms: AtomArray,
    prediction_atoms: AtomArray,
    *,
    metrics: list[PepprMetric] | None = None,
    rmsd_threshold: float = 2.0,
    tolerate_exceptions: bool = True,
    min_sequence_identity: float = 0.95,
    allow_unmatched_entities: bool = False,
) -> dict[str, float | None]:
    """Compute selected PEPPR metrics directly from two atom arrays."""
    import peppr

    reference_atoms = reference_atoms[np.isfinite(reference_atoms.coord).all(axis=-1)]
    prediction_atoms = prediction_atoms[
        np.isfinite(prediction_atoms.coord).all(axis=-1)
    ]
    if reference_atoms.bonds is None:
        reference_atoms.bonds = connect_via_residue_names(reference_atoms)
    if prediction_atoms.bonds is None:
        prediction_atoms.bonds = connect_via_residue_names(prediction_atoms)

    selected_metrics = list(PepprMetric) if metrics is None else metrics
    result = {}
    for metric in selected_metrics:
        metric = PepprMetric(metric)
        try:
            metric_class = getattr(peppr, METRIC_CLASSES[metric])
            instance = (
                metric_class(threshold=rmsd_threshold)
                if metric == PepprMetric.RMSD
                else metric_class()
            )
            evaluator = peppr.Evaluator(
                [instance],
                min_sequence_identity=min_sequence_identity,
                allow_unmatched_entities=allow_unmatched_entities,
            )
            evaluator.feed("row", reference_atoms, prediction_atoms)
            value = evaluator.tabulate_metrics()[instance.name].iloc[0]
            result[metric.value] = (
                float(value) if value is not None and np.isfinite(value) else None
            )
        except Exception:
            logger.exception("PEPPR metric %s failed", metric.value)
            if not tolerate_exceptions:
                raise
            result[metric.value] = None
    return result


class PepprBlock(InlineMetricBlock):
    """Compute structural evaluation metrics using peppr.

    Computes metrics (RMSD, TM-score, lDDT, lDDT-PLI, fnat, iRMSD, LRMSD, DockQ)
    between selected reference and prediction structures in each row (first
    and last by default).
    """

    _name = "peppr"
    _display_name = "PEPPR"
    _description = "Compute structural evaluation metrics using peppr"
    _creates_new_step = False
    _inline = True
    _chunked = False

    metrics: List[PepprMetric] = Field(
        default=list(PepprMetric),
        description="Which metrics to compute. Options: rmsd, tm_score, lddt, lddt_pli, fnat, irmsd, lrmsd, dockq",
    )
    rmsd_threshold: float = Field(
        default=2.0,
        description="RMSD threshold (Angstroms) for MonomerRMSD classification",
    )
    tolerate_exceptions: bool = Field(
        default=True,
        description="If True, continue when individual metrics fail (e.g., lDDT-PLI with no ligand). Failed metrics are stored as None.",
    )
    which: List[int] = Field(
        default=[0, -1],
        min_length=2,
        max_length=2,
        description="Reference and prediction indices in the structure lineage.",
    )

    def get_data_kwargs(self):
        return {"which": self.which, "load": True}

    def compute_metrics(self, loaded, *, context=None, index=0):
        first_loaded, last_loaded = loaded
        result_metrics = peppr_metrics(
            first_loaded.atom_array,
            last_loaded.atom_array,
            metrics=self.metrics,
            rmsd_threshold=self.rmsd_threshold,
            tolerate_exceptions=self.tolerate_exceptions,
        )

        logger.info(
            "Computed peppr metrics for %s -> %s: %s",
            first_loaded.path,
            last_loaded.path,
            result_metrics,
        )

        return result_metrics
