"""Protein-protein interface ddG with FastRelax.

Relaxes the input complex using PyRosetta FastRelax (coordinate-constrained
to the input coordinates), then computes ddG = score(complex) - score(binder
translated 10 000 Å away), using the beta_nov16 all-atom scorefunction.

Intended for protein-protein or protein-peptide complexes of standard amino
acids — no .params files are needed.

Output metrics (under ``<name>.*``):
    ddg          — binding energy kcal/mol (more negative = stronger binder)
    ddg_std      — std across replicates (0.0 when replicates=1)
    n_replicates — number of completed replicates
    relax_score  — total Rosetta score of the relaxed complex (last replicate)
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)


def _relax_and_ddg(
    pose,
    *,
    binder_chain: str,
    relax_rounds: int,
    constrain: bool,
    constraint_weight: float,
) -> tuple[float, float]:
    """One FastRelax cycle followed by a no-repack ddG. Returns (ddg, total_score)."""
    import pyrosetta as pyr
    from blocks.utils.pyrosetta import calc_ddg_norepack

    sfxn = pyr.create_score_function("beta_nov16")
    if constrain:
        sfxn.set_weight(
            pyr.rosetta.core.scoring.coordinate_constraint, constraint_weight
        )

    fr = pyr.rosetta.protocols.relax.FastRelax(sfxn, relax_rounds)
    if constrain:
        fr.constrain_relax_to_start_coords(True)

    p = pose.clone()
    fr.apply(p)

    # Score with plain beta_nov16 (no constraint weight inflating the total)
    sfxn_score = pyr.create_score_function("beta_nov16")
    total_score = float(sfxn_score(p))
    ddg = calc_ddg_norepack(p, sfxn_score, move_chain=binder_chain)
    return ddg, total_score


class PPIDDGBlock(InlineMetricBlock):
    """Protein-protein binding energy via FastRelax + ddG.

    For each input structure:
    1. FastRelax with coordinate constraints (preserves the AF3-predicted
       backbone while optimising sidechains and local geometry).
    2. ddG = score(complex) - score(binder translated 10 000 Å away),
       evaluated with beta_nov16 and no further repacking.

    More negative ddG = stronger predicted binding.
    """

    _name = "ppi_ddg"
    _display_name = "PPI DDG"
    _description = "Protein-protein binding energy with FastRelax + ddG"
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(
        partition="cpu",
        mem_gb=8,
        rows_per_job=10,
    )

    binder_chain: str = Field(
        default="A",
        description="Chain ID of the binder (translated far away for unbound score).",
    )
    target_chain: str = Field(
        default="B",
        description="Chain ID of the target (stays in place).",
    )
    relax_rounds: int = Field(
        default=1,
        description="FastRelax standard_repeats. 1 is fast (~1–3 min/structure on CPU); "
        "3–5 gives lower, more converged energies.",
    )
    replicates: int = Field(
        default=1,
        description="Independent relax + ddG cycles. Mean and std are reported. "
        "Each additional replicate adds one full relax run.",
    )
    constrain_to_start_coords: bool = Field(
        default=True,
        description="Harmonic coordinate constraints during FastRelax — keeps the "
        "backbone close to the AF3-predicted geometry.",
    )
    constraint_weight: float = Field(
        default=1.0,
        description="Coordinate constraint weight. Higher = stays closer to input "
        "coords; lower = more relaxation freedom.",
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        """Init PyRosetta once per chunk; return a pose-loader closure."""
        from blocks.utils.pyrosetta import init_pyrosetta
        from io import StringIO
        import biotite.structure.io.pdb as pdb_io

        init_pyrosetta()
        import pyrosetta as pyr

        def _load_pose(atom_array):
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(atom_array)
            buf = StringIO()
            pdb_file.write(buf)
            pdb_string = buf.getvalue()
            buf.close()
            pose = pyr.Pose()
            pyr.io.pose_from_pdbstring(
                pose, pdb_string, pose.residue_type_set_for_pose(), "temp.pdb"
            )
            return pose

        return _load_pose

    def compute_metrics(self, loaded, *, context=None, index=0):
        load_pose = context
        try:
            pose = load_pose(loaded.atom_array)
            ddg_vals, relax_scores = [], []
            for _ in range(self.replicates):
                ddg, rscore = _relax_and_ddg(
                    pose,
                    binder_chain=self.binder_chain,
                    relax_rounds=self.relax_rounds,
                    constrain=self.constrain_to_start_coords,
                    constraint_weight=self.constraint_weight,
                )
                ddg_vals.append(ddg)
                relax_scores.append(rscore)

            arr = np.array(ddg_vals)
            metrics = {
                "ddg": float(arr.mean()),
                "ddg_std": float(arr.std()),
                "n_replicates": len(arr),
                "relax_score": float(relax_scores[-1]),
            }
            logger.info(
                "ppi_ddg for %s: ddg=%.2f ± %.2f (relax_score=%.2f)",
                loaded.basename,
                metrics["ddg"],
                metrics["ddg_std"],
                metrics["relax_score"],
            )
        except Exception as e:
            logger.error("ppi_ddg failed for %s: %s", loaded.basename, e, exc_info=True)
            metrics = {
                "ddg": None,
                "ddg_std": None,
                "n_replicates": 0,
                "relax_score": None,
                "error": str(e),
            }

        return metrics
