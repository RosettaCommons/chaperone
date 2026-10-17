"""Ligand protonation and subset utilities.

Ports of small-molecule helpers used by ddg_soft_sm.py, adapted for
biotite AtomArray and Dimorphite-DL pH-aware protonation (replacing
bcov's frame-based hydrogen transfer).
"""

import logging

import numpy as np
from biotite.structure import AtomArray

logger = logging.getLogger(__name__)


def _generate_h_names(out: AtomArray, used_names: set[str]) -> None:
    """Assign unique ≤4-char names to hydrogen atoms in-place.

    Names each H after its bonded heavy atom's name suffix (e.g. H on C3 → HC3),
    disambiguating with a trailing counter when needed. Port of the naming
    strategy from conformer_protonation.py:generate_hydrogen_names, simplified
    for biotite (no PDBResidueInfo).
    """
    h_mask = out.element == "H"
    h_idx = np.where(h_mask)[0]
    if len(h_idx) == 0 or out.bonds is None:
        return

    h_counts: dict[int, int] = {}
    for idx in h_idx:
        bonded, _ = out.bonds.get_bonds(int(idx))
        parent = None
        for b in bonded:
            if out.element[b] != "H":
                parent = int(b)
                break
        if parent is None:
            # orphan H — just give it a running number
            name = f"H{idx}"
            if len(name) > 4:
                name = name[:4]
            out.atom_name[idx] = name
            used_names.add(name)
            continue

        h_counts[parent] = h_counts.get(parent, 0) + 1
        n = h_counts[parent]
        parent_name = str(out.atom_name[parent]).strip()
        # drop leading element letter(s) from parent name for the suffix
        suffix = parent_name[1:] if len(parent_name) > 1 else parent_name

        for cand in (
            f"H{suffix}" if n == 1 else f"H{suffix}{n}",
            f"H{suffix}{n}",
            f"H{n}{suffix}",
            f"H{parent}{n}",
        ):
            if len(cand) <= 4 and cand not in used_names:
                out.atom_name[idx] = cand
                used_names.add(cand)
                break
        else:
            # last resort: running index
            cand = f"H{len(used_names)}"[:4]
            out.atom_name[idx] = cand
            used_names.add(cand)


def _rebuild_mol_from_geometry(
    ligand_array: AtomArray,
    ref_smiles: str | None = None,
    *,
    ref_mol=None,
):
    """Build an RDKit mol from heavy-atom coords + a reference SMILES.

    Used when the input AtomArray has unusable bonds — typically because
    the ``res_name`` is a CCD code but the ``atom_name``s don't follow CCD
    convention (common in diffusion-model outputs). atomworks' parser then
    matches atoms to the CCD template *by name* and draws bonds between the
    wrong atom pairs; the distance filter drops some, leaving a fragmented
    graph.

    Recovery:
      1. ``biotite.structure.connect_via_distances`` rebuilds connectivity
         from geometry alone (no names) — gets the correct heavy-atom graph
         but with unknown bond orders (all set to SINGLE as a placeholder).
      2. ``AssignBondOrdersFromTemplate(ref_smiles, ...)`` maps the correct
         bond orders/charges onto that graph via substructure match
         (element + connectivity, not names).

    Note: inferring bond orders from heavy-atom geometry alone is a
    chicken-and-egg problem — without H you can't distinguish C-C from C≡C
    at ~1.5 Å. The reference SMILES breaks the tie.
    """
    import biotite.structure as struc
    from atomworks.io.tools.rdkit import atom_array_to_rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if ref_mol is None and ref_smiles is None:
        raise ValueError("one of ref_smiles / ref_mol is required")

    heavy = ligand_array[ligand_array.element != "H"].copy()

    # Resolve the reference mol up front — both the fast path and the geometry
    # fallback need it.
    if ref_mol is None:
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is None:
            raise ValueError(f"Could not parse ref_smiles: {ref_smiles!r}")

    # ── Fast path: index-identity bond transfer ──────────────────────────────
    # Cofold oracles (AF3/Boltz/Chai via atomworks) emit their output ligand in
    # the SAME atom order as the input SMILES, so predicted heavy atom i is the
    # same chemical atom as reference atom i. When the element sequence matches
    # index-for-index we transfer the reference's bond graph onto the predicted
    # coordinates verbatim and skip geometry bond perception entirely.
    #
    # The geometry fallback below is fragile for strained or AF3-distorted
    # scaffolds: a bond stretched past even the widened cutoffs is dropped,
    # fragmenting the graph so AssignBondOrdersFromTemplate fails with "No
    # matching found" (observed on a 69-atom taxane: 2 bonds dropped, 1 atom
    # left fully disconnected). The order-identity path is immune to that.
    #
    # Gate is strict element-order equality; anything else (e.g. a reference CIF
    # in canonical/author order) falls through to the geometry path, where the
    # caller's downstream RMSD check guards against a spurious match.
    ref_elems = [a.GetSymbol() for a in ref_mol.GetAtoms()]
    pred_elems = [str(e).capitalize() for e in heavy.element]
    if ref_elems == pred_elems:
        from rdkit.Geometry import Point3D

        pred_mol = atom_array_to_rdkit(heavy, hydrogen_policy="remove", sanitize=False)
        fixed = Chem.Mol(ref_mol)
        conf = Chem.Conformer(fixed.GetNumAtoms())
        for i, xyz in enumerate(heavy.coord):
            conf.SetAtomPosition(
                i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            )
        fixed.RemoveAllConformers()
        fixed.AddConformer(conf, assignId=True)
        # Self-validate the order assumption. The element-sequence gate is not
        # proof of atom identity (and build_remap_reference passes the *canonical*
        # SMILES), so verify the transferred bond graph is geometrically sane on
        # the predicted coordinates: every bond a plausible length. A mis-ordered
        # overlay yields multi-angstrom "bonds" -> fall through to geometry
        # perception instead of trusting a spurious match.
        pos = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(fixed.GetNumAtoms())]
        )
        if all(
            float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
            <= 2.5
            for b in fixed.GetBonds()
        ):
            Chem.SanitizeMol(fixed)
            fixed._annotations = pred_mol._annotations
            return fixed

    # ── Fallback: geometry bond perception + template bond-order assignment ───
    # Widen biotite's Allen-1987 bond cutoffs to accept diffusion-model bond
    # geometry. Prediction models routinely compress or elongate heavy-atom
    # bonds past the default thresholds, dropping real bonds and failing
    # template match. Observed in the wild (counts from a 3-ligand baseline
    # sweep over ~12.7k AF3 outputs):
    #   • C–N: urea ring compressed to ~1.29Å (< default 1.307Å min).
    #   • C–C: aromatic C=C compressed to ~1.27Å after AF3 inference
    #     (< default 1.30Å min); recovers ~9 of 14 est remap failures.
    #   • C–S: thiolane C–S elongated to 1.89Å (> default 1.88Å max);
    #     no prior override → whole ring dropped. Adding the pair
    #     recovers sulfur-ring bond-perception failures.
    #   • C–O: amide C=O occasionally overshoots similarly; phosphoester
    #     C–O (ribose-C5' → bridging O of a pyrophosphate, e.g. NAD/FAD/
    #     ATP) sits at ~1.61–1.62Å, past the default 1.50Å max → dropped,
    #     so the C–O max is widened to 1.65Å.
    #   • C–C: a curated (non-diffusion-model) reference CIF can also trip this —
    #     a ring-fusion sp3-sp3 C–C bond (e.g. a steroid-conjugate's cyclopentane/
    #     cyclohexane fusion carbon) sat at 1.702Å, just past the prior 1.70Å max,
    #     dropping the bond and fragmenting the ring before AssignBondOrdersFrom-
    #     Template ever saw a diffusion-model output. Widened to 1.75Å.
    # Overrides merge with biotite defaults; unspecified element pairs are
    # unaffected. Truly broken conformers (atoms at <1Å or >3Å from a
    # supposed bond partner) are still correctly rejected.
    bonds = struc.connect_via_distances(
        heavy,
        distance_range={
            ("C", "N"): (1.15, 1.65),
            ("C", "O"): (1.10, 1.65),
            ("C", "C"): (1.15, 1.75),
            ("C", "S"): (1.40, 2.00),
        },
    )
    bond_arr = bonds.as_array()
    bond_arr[:, 2] = struc.BondType.SINGLE  # placeholder; template fixes orders
    heavy.bonds = struc.BondList(len(heavy), bond_arr)

    mol = atom_array_to_rdkit(heavy, hydrogen_policy="remove", sanitize=False)
    fixed = AllChem.AssignBondOrdersFromTemplate(ref_mol, mol)
    Chem.SanitizeMol(fixed)
    fixed._annotations = mol._annotations
    return fixed


def _get_ref_smiles(res_name: str, ref_smiles: str | None) -> str | None:
    """Resolve a reference SMILES: user-supplied, else CCD lookup."""
    if ref_smiles:
        return ref_smiles
    try:
        from atomworks.io.tools.rdkit import ccd_code_to_rdkit
        from rdkit import Chem

        ref_mol = ccd_code_to_rdkit(res_name)
        return Chem.MolToSmiles(Chem.RemoveHs(ref_mol))
    except Exception as e:
        logger.debug("CCD lookup failed for %s: %s", res_name, e)
        return None


def protonate_ligand(
    ligand_array: AtomArray,
    ph: float = 7.4,
    ref_smiles: str | None = None,
) -> AtomArray:
    """Add pH-aware hydrogens to a heavy-atom ligand via Dimorphite-DL.

    Flow:
      1. biotite → RDKit (strip H)
      2. SMILES → dimorphite_dl.protonate_smiles → protonated SMILES
      3. AssignBondOrdersFromTemplate maps protonation state onto 3D coords
      4. AddHs(addCoords=True) places explicit H
      5. RDKit → biotite, name H atoms, restore residue annotations

    Falls back to plain Chem.AddHs (no pH awareness) if dimorphite returns
    nothing or template mapping fails.

    Parameters
    ----------
    ligand_array : AtomArray
        Single-residue ligand. Existing H are stripped before re-protonation.
    ph : float
        Target pH. Dimorphite-DL is queried with ph_min=ph_max=ph to get the
        single most probable protonation state.
    ref_smiles : str, optional
        Reference SMILES used to rebuild the ligand when its bond graph is
        disconnected (CCD-name mismatch — see ``_rebuild_mol_from_geometry``).
        If None, a CCD lookup on ``res_name`` is attempted. Irrelevant when
        the input bonds are already correct.

    Returns
    -------
    AtomArray
        Ligand with explicit hydrogens, correct protonation state, and
        per-atom annotations (res_name/res_id/chain_id/hetero) copied from
        the input.
    """
    from atomworks.io.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if len(ligand_array) == 0:
        raise ValueError("protonate_ligand: empty ligand_array")

    template = ligand_array[0]

    # 1. biotite → RDKit, stripping any existing H so we regenerate from scratch
    mol = atom_array_to_rdkit(ligand_array, hydrogen_policy="remove")
    mol_no_h = Chem.RemoveHs(mol)  # no-op if already H-free; preserves atom order

    # Detect disconnected graph — atom names didn't match the CCD template,
    # so atomworks drew bonds between wrong pairs and the distance filter
    # broke the molecule apart. Rebuild from geometry + a reference SMILES.
    smi_check = Chem.MolToSmiles(mol_no_h)
    if "." in smi_check or mol_no_h.GetNumBonds() == 0:
        resolved_ref = _get_ref_smiles(str(template.res_name), ref_smiles)
        if resolved_ref is None:
            raise ValueError(
                f"Ligand {template.res_name!r} has a disconnected bond graph "
                f"(SMILES={smi_check!r}) and no reference SMILES is available. "
                f"Pass ref_smiles= explicitly, or ensure atom names follow CCD "
                f"convention for res_name={template.res_name!r}."
            )
        logger.info(
            "Ligand %s bond graph is disconnected (%s); rebuilding from "
            "geometry + ref_smiles",
            template.res_name,
            smi_check,
        )
        mol = _rebuild_mol_from_geometry(ligand_array, resolved_ref)
        mol_no_h = Chem.RemoveHs(mol)

    # Keep only atom_name + rdkit_atom_id for the round-trip.
    # Dropping 'element' is required: otherwise H atoms (absent from the
    # annotation arrays) get empty-string elements after restoration.
    anno = {
        k: v for k, v in mol._annotations.items() if k in ("atom_name", "rdkit_atom_id")
    }

    # 2. Dimorphite-DL protonation
    smiles = Chem.MolToSmiles(mol_no_h)
    with_h = None
    try:
        from dimorphite_dl import protonate_smiles

        prot_list = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.5)
    except ImportError:
        logger.warning(
            "dimorphite_dl not installed; falling back to plain AddHs "
            "(no pH-aware protonation). Install with: uv pip install dimorphite-dl"
        )
        prot_list = []

    if prot_list:
        prot_2d = Chem.MolFromSmiles(prot_list[0])
        if prot_2d is not None:
            try:
                # 3. map protonation state back onto 3D heavy-atom coords
                prot_3d = AllChem.AssignBondOrdersFromTemplate(prot_2d, mol_no_h)
                prot_3d._annotations = anno
                # 4. add explicit H with 3D placement
                with_h = Chem.AddHs(prot_3d, addCoords=True)
                with_h._annotations = anno
                logger.debug(
                    "Protonated %s at pH %.1f: %s → %s (+%d H)",
                    template.res_name,
                    ph,
                    smiles,
                    prot_list[0],
                    with_h.GetNumAtoms() - prot_3d.GetNumAtoms(),
                )
            except Exception as e:
                logger.warning(
                    "AssignBondOrdersFromTemplate failed for %s (%s); "
                    "falling back to plain AddHs",
                    template.res_name,
                    e,
                )
        else:
            logger.warning(
                "Could not parse protonated SMILES %r; falling back to plain AddHs",
                prot_list[0],
            )
    else:
        logger.warning(
            "Dimorphite-DL returned no protonation states for %s (SMILES=%r); "
            "falling back to plain AddHs",
            template.res_name,
            smiles,
        )

    if with_h is None:
        # Fallback: plain AddHs on the original 3D mol (no pH adjustment)
        mol_no_h._annotations = anno
        with_h = Chem.AddHs(mol_no_h, addCoords=True)
        with_h._annotations = anno

    # 5. RDKit → biotite
    out = atom_array_from_rdkit(with_h, remove_hydrogens=False)

    # Name the new H atoms — they come back with empty atom_name since they
    # have no entry in _annotations.
    heavy_names = {str(n) for n in out.atom_name if str(n).strip()}
    _generate_h_names(out, heavy_names)

    # Restore residue-level annotations on ALL atoms (RDKit round-trip loses
    # these for H atoms, and we only kept atom_name in _annotations).
    n = len(out)
    out.res_name = np.full(n, template.res_name)
    out.res_id = np.full(n, template.res_id)
    out.chain_id = np.full(n, template.chain_id)
    out.hetero = np.full(n, template.hetero)
    if hasattr(template, "ins_code"):
        out.ins_code = np.full(n, template.ins_code)

    return out


def get_atom_subset_with_h(atom_array: AtomArray, atom_names: list[str]) -> AtomArray:
    """Return a subset of atoms plus their bonded hydrogens.

    Direct port of ddg_soft_sm.py:582-599. For each named heavy atom,
    include it and any H bonded to it.

    Raises AssertionError if any requested name is absent.
    """
    name_to_idx = {str(name): i for i, name in enumerate(atom_array.atom_name)}

    missing = [n for n in atom_names if n not in name_to_idx]
    assert not missing, f"Atoms not found in atom_array: {missing}"

    keep: set[int] = set()
    for name in atom_names:
        idx = name_to_idx[name]
        keep.add(idx)
        if atom_array.bonds is not None:
            bonded, _ = atom_array.bonds.get_bonds(idx)
            for b in bonded:
                if atom_array.element[b] == "H":
                    keep.add(int(b))

    mask = np.zeros(len(atom_array), dtype=bool)
    mask[list(keep)] = True
    return atom_array[mask]
