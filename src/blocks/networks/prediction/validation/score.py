"""In-place metric block for validation predictions."""

from __future__ import annotations

from enum import Enum
from functools import partial

from pydantic import Field

from blocks.metrics.peppr import peppr_metrics
from chaperone import InlineMetricBlock

from .lddt import rf3_allatom_lddt
from .remap import remap_output_to_input


class ValidationMetric(str, Enum):
    RF3_ALLATOM_LDDT = "rf3_allatom_lddt"
    PEPPR = "peppr"


class ScoreValStructuresBlock(InlineMetricBlock):
    """Score the last (predicted) structure against the first (reference) one."""

    _name = "score_val_structures"
    _display_name = "ScoreValStructures"
    _description = "Score predictions against manifested validation references"
    _creates_new_step = False
    _inline = True
    _chunked = False

    metrics: list[ValidationMetric] = Field(
        default_factory=lambda: [ValidationMetric.RF3_ALLATOM_LDDT],
        min_length=1,
        description="Scoring backends to run: rf3_allatom_lddt and/or peppr.",
    )
    remap_outputs_to_input: bool = Field(
        False,
        description=(
            "Map a renumbered monomer prediction onto the single scored protein "
            "pn-unit before scoring. Disabled by default so direct reference/output "
            "comparison remains unchanged for other validation users."
        ),
    )
    min_atom_coverage: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Minimum fraction of resolved reference atoms that a remapped oracle "
            "output must represent. Used only when remap_outputs_to_input is enabled."
        ),
    )

    def get_data_kwargs(self):
        return {"which": "all", "load": True}

    def compute_metrics(self, loaded, *, context=None, index=0):
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise ValueError(
                "score_val_structures requires a reference and a prediction in each row"
            )
        reference, prediction = loaded[0], loaded[-1]
        validation = prediction.metrics.get("validation", {})
        missing = [
            field
            for field in ("interfaces_to_score", "pn_units_to_score")
            if validation.get(field) is None
        ]
        if missing:
            raise ValueError(
                f"validation row {index} is missing carried scoring metadata {missing}"
            )
        reference_atoms = reference.atom_array
        prediction_atoms = prediction.atom_array
        result = {}
        if self.remap_outputs_to_input:
            reference_atoms, prediction_atoms, result = remap_output_to_input(
                reference_atoms,
                prediction_atoms,
                validation,
                min_atom_coverage=self.min_atom_coverage,
            )
        scoring_functions = {
            ValidationMetric.RF3_ALLATOM_LDDT: partial(
                rf3_allatom_lddt,
                interfaces_to_score=validation["interfaces_to_score"],
                pn_units_to_score=validation["pn_units_to_score"],
                pn_unit_id_map=validation.get("pn_unit_id_map"),
            ),
            ValidationMetric.PEPPR: partial(
                peppr_metrics,
                min_sequence_identity=0.0,
                allow_unmatched_entities=True,
            ),
        }
        for metric in self.metrics:
            result.update(scoring_functions[metric](reference_atoms, prediction_atoms))
        return result
