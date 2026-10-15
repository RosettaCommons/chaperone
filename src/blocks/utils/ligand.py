"""Ligand bond reconstruction and public CCD reference lookup."""

import logging

import numpy as np
from biotite.structure import AtomArray

logger = logging.getLogger(__name__)


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
    #     recovers the btn thiolane class of failures.
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
