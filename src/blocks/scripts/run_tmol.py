#!/usr/bin/env python
"""tmol runner — executed INSIDE the tmol container by the tmol blocks.

Two modes (``--mode``):

* ``interface`` — protein↔ligand interface energetics. Emits the weighted
  block-pair ddG plus a per-term breakdown (fa_atr, fa_rep, fa_sol, fa_elec,
  hbond, lk_ball, ...) for each input structure. Single-point (no minimise /
  repack), matching the tmol golden ddG protocol.
* ``minimize`` — Cartesian, all-atoms-free energy minimisation
  (beta_nov2016_cart). Writes a minimised PDB per input and emits
  energy_start / energy_final / energy_delta (total weighted energy).

Per structure we write ``<basename>.metrics.json``; ``minimize`` also writes
``<basename>.min.pdb``. The driving block reads these back in ``compile``.

LIGAND TYPING (the hard part): real complexes from RFD3/AF3/cofold carry the
ligand's *input* coordinates, which must be SCORED AS-IS. tmol still needs a
chemical residue-type for any non-polymer residue. We obtain that typing
WITHOUT touching the input coordinates:

  1. ``--ligand-params <file.tmol>``  — a pre-made tmol params file (most
     deterministic; reproduces the tmol golden ddGs).
  2. ``--ligand-smiles <SMILES>``     — we generate a ``.tmol`` params file on
     the fly (OpenBabel 3D + MMFF94 charges, ``protonate=False`` to avoid the
     missing dimorphite_dl dependency), then hand it to
     ``pose_stack_from_biotite(prepare_ligands=True, ligand_params_files=[...])``.

In BOTH cases the params file supplies only the residue *type* (atom types,
bonds, partial charges, internal-coordinate tree). The coordinates come from
the input structure — verified: every input ligand heavy atom coincides with
its pose-stack atom to 0.0000 Å. A ligand that cannot be typed is skipped with
a logged warning (the chunk keeps going).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger("run_tmol")

# tmol term name -> readable / Rosetta-style key (keys are flat, no dots).
_TERM_RENAME = {
    "fa_ljatr": "fa_atr",
    "fa_ljrep": "fa_rep",
    "fa_lk": "fa_sol",
}

# Common solvent / buffer residues that are never "the ligand".
_SOLVENT = {
    "HOH",
    "WAT",
    "DOD",
    "TIP",
    "TIP3",
    "SOL",
    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "ZN",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
    "ACT",
}


def _load_structure(path: str):
    import biotite.structure as struc
    import biotite.structure.io

    s = biotite.structure.io.load_structure(str(path), model=1, include_bonds=True)
    if isinstance(s, struc.AtomArrayStack):
        s = s[0]
    return s


def _detect_ligand_resnames(structure) -> list[str]:
    """Non-polymer residue names (the ligand). Excludes amino acids, nucleotides
    and common solvent/ions."""
    import biotite.structure as struc

    polymer = struc.filter_amino_acids(structure)
    try:
        polymer = polymer | struc.filter_nucleotides(structure)
    except Exception:
        pass
    names = []
    for rn in structure.res_name[~polymer]:
        rn = rn.strip()
        if rn and rn.upper() not in _SOLVENT and rn not in names:
            names.append(rn)
    return names


def _generate_params_from_smiles(smiles: str, res_name: str, out_path: Path) -> Path:
    """SMILES -> tmol .tmol params file (chemical typing only; no input coords).

    Atoms are named canonically (1-indexed per element: C1.. N1.. O1..). The
    structure's ligand atom names must match this convention for pose-build to
    place its coordinates onto the typed residue.
    """
    from tmol.ligand.detect import nonstandard_residue_info_from_smiles_via_mol2
    from tmol.ligand.preparation import prepare_single_ligand
    from tmol.ligand.params_io import write_params_file

    info = nonstandard_residue_info_from_smiles_via_mol2(
        smiles, res_name=res_name, protonate=False, conformer_search=False
    )
    prep = prepare_single_ligand(info, sample_proton_chi=False)
    # Atomic write: many chunks share one cache dir on NFS, so write to a
    # pid-unique temp then rename — a reader never sees a half-written .tmol,
    # and the first chunk to finish "wins" (any complete file is valid typing).
    tmp = out_path.parent / f".{out_path.name}.tmp.{os.getpid()}"
    write_params_file(prep, str(tmp), format="tmol")
    os.replace(tmp, out_path)
    return out_path


def _build_pose(structure, ligand_resnames, params_files, dev):
    """Return (pose_stack, score_function, ligand_block_mask_or_None, param_db)."""
    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
    from tmol.score import beta2016_score_function

    # Only add the ligand-typing kwargs when there's a non-polymer residue to type.
    ligand_kwargs = {}
    if ligand_resnames and params_files:
        ligand_kwargs = dict(
            prepare_ligands=True,
            ligand_params_files=[str(p) for p in params_files],
            no_optH=True,
            sample_proton_chi=False,
        )
    ps, ctx = pose_stack_from_biotite(
        structure,
        dev,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
        **ligand_kwargs,
    )

    sfxn = beta2016_score_function(dev, param_db=ctx.parameter_database)

    lig_mask = None
    if ligand_resnames:
        co = ctx.canonical_ordering
        rt = ctx.canonical_form.res_types[0]
        lig_mask = torch.zeros((1, ps.max_n_blocks), dtype=torch.bool, device=dev)
        for b in range(min(ps.max_n_blocks, rt.shape[0])):
            rti = int(rt[b])
            if rti >= 0 and co.restype_io_equiv_classes[rti] in ligand_resnames:
                lig_mask[0, b] = True
        if not bool(lig_mask.any()):
            lig_mask = None
    return ps, sfxn, lig_mask, ctx.parameter_database


def _resolve_params(ligand_resnames, smiles, params_files_arg, cache_dir):
    """Resolve / generate the .tmol params files for this structure's ligand(s)."""
    if params_files_arg:
        return [Path(p) for p in params_files_arg]
    if smiles and ligand_resnames:
        # one ligand smiles -> type the first detected non-polymer residue
        res_name = ligand_resnames[0]
        cache = cache_dir / f"ligand_{res_name}.tmol"
        if not cache.exists():
            _generate_params_from_smiles(smiles, res_name, cache)
        return [cache]
    return []


def _interface(ps, sfxn, lig_mask, ctx_db):
    from tmol.score.score_utils import calculate_block_pair_ddg

    ddg = calculate_block_pair_ddg(
        ps, lig_mask, sfxn=sfxn, minimize=False, pack=False, database=ctx_db
    )
    terms = calculate_block_pair_ddg(
        ps,
        lig_mask,
        sfxn=sfxn,
        sum_terms=False,
        minimize=False,
        pack=False,
        database=ctx_db,
    )
    names = [st.name for st in sfxn.all_score_types()]
    metrics = {"ddg": float(ddg.reshape(-1)[0].detach())}
    for i, name in enumerate(names):
        val = float(terms[i, 0].detach())
        if abs(val) > 1e-6:
            metrics[_TERM_RENAME.get(name, name)] = val
    return metrics


def _total_energy(ps, sfxn) -> float:
    wpsm = sfxn.render_whole_pose_scoring_module(ps)
    return float(wpsm(ps.coords)[0].detach())


def _minimize(ps, sfxn, fix_backbone, out_pdb):
    from tmol.optimization.minimizers import run_cart_min
    from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

    e_start = _total_energy(ps, sfxn)
    coord_mask = None
    if fix_backbone:
        from tmol.score.score_utils import build_sidechain_coord_mask

        coord_mask = build_sidechain_coord_mask(ps)
    ps_min = run_cart_min(ps, sfxn, coord_mask=coord_mask)
    e_final = _total_energy(ps_min, sfxn)
    write_pose_stack_pdb(ps_min, str(out_pdb))
    return {
        "energy_start": e_start,
        "energy_final": e_final,
        "energy_delta": e_final - e_start,
    }


def _process_one(path, args, dev, cache_dir, out_dir):
    basename = Path(path).stem
    structure = _load_structure(path)
    ligand_resnames = _detect_ligand_resnames(structure)
    logger.info("%s: detected ligand residues %s", basename, ligand_resnames or "none")

    if args.mode == "interface" and not ligand_resnames:
        logger.warning("%s: no non-polymer ligand found — skipping interface", basename)
        return {"input_path": str(path), "skipped": "no_ligand"}

    params_files = []
    if ligand_resnames:
        try:
            params_files = _resolve_params(
                ligand_resnames,
                args.ligand_smiles,
                args.ligand_params,
                cache_dir,
            )
        except Exception as e:  # ligand typing failed
            logger.warning("%s: ligand params generation failed (%s)", basename, e)
            params_files = []
        if not params_files:
            msg = "no ligand_smiles / ligand_params supplied or typing failed"
            if args.mode == "interface":
                logger.warning("%s: %s — skipping", basename, msg)
                return {"input_path": str(path), "skipped": "ligand_untyped"}
            logger.warning("%s: %s — minimising without ligand typing", basename, msg)

    try:
        ps, sfxn, lig_mask, param_db = _build_pose(
            structure, ligand_resnames, params_files, dev
        )
    except Exception as e:
        logger.warning("%s: pose build failed (%s) — skipping", basename, e)
        return {"input_path": str(path), "skipped": f"pose_build_failed: {e}"}

    if args.mode == "interface":
        if lig_mask is None:
            logger.warning("%s: ligand block not found in pose — skipping", basename)
            return {"input_path": str(path), "skipped": "no_ligand_block"}
        metrics = _interface(ps, sfxn, lig_mask, param_db)
        return {"input_path": str(path), **metrics}

    # minimize
    out_pdb = out_dir / f"{basename}.min.pdb"
    metrics = _minimize(ps, sfxn, args.fix_backbone, out_pdb)
    return {"input_path": str(path), "structure_path": str(out_pdb), **metrics}


def main():
    p = argparse.ArgumentParser(description="tmol interface / minimize runner")
    p.add_argument("--inputs", help="File with one input structure path per line")
    p.add_argument("--input", action="append", default=[], help="Repeatable input path")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Shared ligand-params cache dir (defaults to <output-dir>/ligand_params). "
        "Point every chunk at one dir so each ligand is typed once per batch.",
    )
    p.add_argument("--mode", choices=["interface", "minimize"], required=True)
    p.add_argument("--ligand-smiles", default=None)
    p.add_argument(
        "--ligand-params",
        action="append",
        default=None,
        help="Pre-made .tmol params file(s)",
    )
    p.add_argument(
        "--fix-backbone",
        action="store_true",
        help="(minimize) freeze backbone, move sidechains only",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Shared across chunks when --cache-dir is passed; else per-chunk (standalone).
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "ligand_params"
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = list(args.input)
    if args.inputs:
        paths += [
            ln.strip()
            for ln in Path(args.inputs).read_text().splitlines()
            if ln.strip()
        ]
    if not paths:
        raise SystemExit("No input structures provided")

    # Use a GPU only if one is actually visible (CPU partition -> no --nv -> CPU).
    # NB: the AOT GPU wheel is Ampere->Hopper only; don't schedule onto b4000.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("tmol device: %s", dev)
    for path in paths:
        basename = Path(path).stem
        try:
            metrics = _process_one(path, args, dev, cache_dir, out_dir)
        except Exception as e:
            logger.exception("%s: unexpected failure", basename)
            metrics = {"input_path": str(path), "skipped": f"error: {e}"}
        (out_dir / f"{basename}.metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info("%s: %s", basename, metrics)


if __name__ == "__main__":
    main()
