"""Soft-repulsive ligand binding ddG.

Port from /home/bcov/sc/scaffold_comparison/py_protocols/ddg_soft_sm.py.

Packs the binder interface with the **hard** beta_nov16 scorefxn, then scores
with the **soft** beta_nov16_soft scorefxn. Repeated N times; the best and
worst replicates are dropped (when N≥3) and the mean of the rest is reported.

This is **not** the same metric as ``LigandDDGBlock`` (rosetta_scripts XML,
hard-scored, single replicate, pre-made params only). The soft scorefxn is
more forgiving of small steric clashes that would otherwise dominate the
score at an imperfect interface.

Since this uses pyrosetta it will require params file
Params cascade (when ``params_file`` is not given)::

    soft_ddg(atom_array)
      └─ generate_params(ligand)
            └─ ligand has H? → use as-is
            └─ else: protonate_ligand(ligand)   [Dimorphite-DL, pH-aware]

"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)


def soft_ddg(
    atom_array,
    *,
    params_file: str | None = None,
    params_cache_dir: str | Path | None = None,
    replicates: int = 5,
    binder_chain: str = "A",
    ligand_chain: str = "B",
    interesting_atoms: list[str] | None = None,
    ph: float = 7.4,
    ref_smiles: str | None = None,
) -> dict:
    """Compute soft-repulsive binding ddG for a protein–ligand complex.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Complex with binder on ``binder_chain`` and ligand on ``ligand_chain``.
    params_file : str, optional
        Pre-made Rosetta .params file. If given, the protonation / params
        cascade is bypassed entirely — **you are responsible for ensuring the
        ligand hydrogens match the params file's expectations**.
    params_cache_dir : str | Path, optional
        Shared cache directory for generated params files (keyed by ligand
        3-letter code). When None (standalone/notebook use), a per-call
        TemporaryDirectory is created and cleaned up on return.
    replicates : int
        Number of pack+score cycles. When ≥3, the best and worst are dropped
        before averaging (trimmed mean).
    binder_chain, ligand_chain : str
        Chain identifiers. The binder chain is the one translated away to
        compute the unbound score.
    interesting_atoms : list[str], optional
        Subset of ligand heavy-atom names to keep (plus their bonded H).
        Useful for scoring only part of a ligand. Params is generated from
        the **subset**, so this is only valid when ``params_file`` is None.
    ph : float
        Target pH for Dimorphite-DL protonation. Only used when the ligand
        lacks H and no ``params_file`` is given.
    ref_smiles : str, optional
        Reference SMILES for rebuilding the ligand bond graph when it is
        disconnected (diffusion-model outputs with non-CCD atom names).
        Falls back to CCD lookup on the ligand's res_name.

    Returns
    -------
    dict
        ``{"ddg_soft": float, "ddg_soft_std": float, "n_replicates_used": int}``
    """
    from blocks.utils.pyrosetta import (
        atom_array_to_pose,
        beta_pack,
        calc_ddg_norepack,
        init_pyrosetta,
    )
    import pyrosetta as pyr

    init_pyrosetta()

    # Strip binder H (Rosetta rebuilds from its own geometry), keep ligand
    # atoms untouched — mirrors ddg_soft_sm.py:615.
    atom_array = atom_array[
        (atom_array.chain_id == ligand_chain) | (atom_array.element != "H")
    ]
    binder = atom_array[atom_array.chain_id == binder_chain]
    ligand = atom_array[atom_array.chain_id == ligand_chain]
    if len(ligand) == 0:
        raise ValueError(
            f"No atoms on ligand_chain={ligand_chain!r}. "
            f"Chains present: {sorted(set(atom_array.chain_id))}"
        )
    if len(binder) == 0:
        raise ValueError(
            f"No atoms on binder_chain={binder_chain!r}. "
            f"Chains present: {sorted(set(atom_array.chain_id))}"
        )

    # --- params cascade ---------------------------------------------------
    _tmp = None  # held so TemporaryDirectory isn't GC'd before pose construction
    if params_file is None:
        from blocks.utils.params import generate_params

        if params_cache_dir is None:
            _tmp = tempfile.TemporaryDirectory(prefix="ddg_soft_")
            params_cache_dir = _tmp.name
        params_file, ligand = generate_params(
            ligand, params_cache_dir, ph=ph, ref_smiles=ref_smiles
        )

    # --- interesting_atoms subset  (ddg_soft_sm.py:625-626) ---------------
    # Applied after protonation so H bonded to the named heavies are kept.
    if interesting_atoms:
        from blocks.utils.ligand import get_atom_subset_with_h

        ligand = get_atom_subset_with_h(ligand, interesting_atoms)

    # --- pose -------------------------------------------------------------
    complex_array = binder + ligand
    pose = atom_array_to_pose(complex_array, params_files=[params_file])

    if _tmp is not None:
        _tmp.cleanup()

    # --- pack hard, score soft, N replicates  (ddg_soft_sm.py:632-647) ----
    sfxn_soft = pyr.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
        "beta_nov16_soft"
    )
    ddgs = []
    for _ in range(replicates):
        beta_pack(pose, lock_chain=ligand_chain, soft=False)
        ddgs.append(calc_ddg_norepack(pose, sfxn_soft, move_chain=binder_chain))

    ddgs = np.asarray(ddgs, dtype=float)
    idx = np.argsort(ddgs)
    if replicates >= 3:
        idx = idx[1:-1]

    return {
        "ddg_soft": float(ddgs[idx].mean()),
        "ddg_soft_std": float(ddgs[idx].std()),
        "n_replicates_used": int(len(idx)),
    }


class LigandDDGSoftBlock(InlineMetricBlock):
    """Soft-repulsive ddG (pack hard, score soft, trimmed mean over replicates).

    Thin wrapper around :func:`soft_ddg`. Params files are generated on the
    fly and cached at the step level (``{rundir}/{step}/_params_cache/``) so
    sibling chunks share them — chunks run as separate processes, so the
    cache is filesystem-based with atomic writes.

    Set ``override_params_file`` to bypass the protonation/generation cascade
    entirely (matches the ``LigandDDGBlock`` convention).
    """

    _name: ClassVar[str] = "ligand_ddg_soft"
    _display_name: ClassVar[str] = "Ligand DDG Soft"
    _description: ClassVar[str] = (
        "Soft-repulsive ddG (pack hard, score soft, trimmed mean)"
    )
    _creates_new_step: ClassVar[bool] = False
    _resources: ClassVar[JobArgs] = JobArgs(
        partition="cpu", mem_gb=8, rows_per_job=1000
    )

    replicates: int = Field(
        default=5,
        description="Pack+score cycles. ≥3 → trimmed mean (drop best+worst).",
    )
    binder_chain: str = Field(default="A", description="Chain ID of the binder.")
    ligand_chain: str = Field(default="B", description="Chain ID of the ligand.")
    interesting_atoms: list[str] | None = Field(
        default=None,
        description="Ligand heavy-atom names to keep (plus their bonded H). "
        "Params is generated from the subset.",
    )
    ph: float = Field(
        default=7.4,
        description="Target pH for Dimorphite-DL protonation of H-less ligands.",
    )
    override_params_file: str | None = Field(
        default=None,
        description="Pre-made .params file. Bypasses protonation and params "
        "generation — ligand H must already match.",
    )
    ref_smiles: str | None = Field(
        default=None,
        description="Reference SMILES for rebuilding the ligand bond graph "
        "when atom names don't follow CCD convention (common in diffusion "
        "outputs). Defaults to CCD lookup on the ligand's res_name.",
    )
    cleanup_params_cache: bool = Field(
        default=True,
        description="Delete the shared params cache when the last chunk exits.",
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        from blocks.utils.pyrosetta import init_pyrosetta

        init_pyrosetta()

        # Shared cache lives in the step dir (sandbox_dir.parent), not the
        # chunk dir — so all chunks of this step see the same path.
        cache_dir = Path(sandbox_dir).parent / "_params_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # PID-sentinel refcount: last chunk to finish cleans up.
        sentinel = cache_dir / f".pid_{os.getpid()}"
        sentinel.touch()

        return {"cache_dir": cache_dir, "sentinel": sentinel}

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            metrics = soft_ddg(
                loaded.atom_array,
                params_file=self.override_params_file,
                params_cache_dir=context["cache_dir"],
                replicates=self.replicates,
                binder_chain=self.binder_chain,
                ligand_chain=self.ligand_chain,
                interesting_atoms=self.interesting_atoms,
                ph=self.ph,
                ref_smiles=self.ref_smiles,
            )
        except Exception as e:
            logger.error("ligand_ddg_soft failed for %s: %s", loaded.basename, e)
            metrics = {"ddg_soft": None, "error": str(e)}

        return metrics

    def teardown(self, context):
        if context is None:
            return
        cache_dir = context["cache_dir"]
        context["sentinel"].unlink(missing_ok=True)
        if self.cleanup_params_cache and not list(cache_dir.glob(".pid_*")):
            # Last one out. If two chunks race here, both rmtree the same
            # dir — ignore_errors makes the loser a no-op.
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Cleaned params cache: %s", cache_dir)
