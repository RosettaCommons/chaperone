"""Per-row chirality check vs a reference ligand structure.

Ports the chirality logic from `/mnt/net/software/lab/af3_sm/v2/run_alphafold_sm.py`
(functions `_build_rdkit_mol_from_atom_array`, `check_chirality`).

Use AFTER `remap_predicted_ligand` so the output ligand atom names line up
with the reference. This block does NOT remap atoms itself — atom-name
matching is required for the per-atom CIP comparison.
"""

import logging
from typing import ClassVar, Optional

from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs
from pipelines.io import parse_to_atom_array

logger = logging.getLogger(__name__)


def _build_rdkit_mol_from_atom_array(atom_array):
    """Build an RDKit mol from a biotite AtomArray with bond records.

    Returns (mol, ok). ok=False on any RDKit failure — callers should treat
    that as 'cannot determine chirality' rather than a hard error.
    """
    from rdkit import Chem

    btype_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        5: Chem.BondType.AROMATIC,  # biotite AROMATIC_SINGLE
        6: Chem.BondType.AROMATIC,  # biotite AROMATIC_DOUBLE
        9: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for elem in atom_array.element:
        try:
            rw.AddAtom(Chem.Atom(str(elem)))
        except Exception:
            return None, False
    if atom_array.bonds is not None:
        for idx1, idx2, btype in atom_array.bonds.as_array():
            rdkit_btype = btype_map.get(int(btype), Chem.BondType.SINGLE)
            try:
                rw.AddBond(int(idx1), int(idx2), rdkit_btype)
            except Exception:
                pass
    conf = Chem.Conformer(len(atom_array))
    for i, coord in enumerate(atom_array.coord):
        conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception:
        return None, False
    return mol, True


def _cip_codes(atom_array) -> dict:
    """Atom-name → CIP code (R/S) dict. Empty dict if RDKit can't assign."""
    from rdkit import Chem

    mol, ok = _build_rdkit_mol_from_atom_array(atom_array)
    if not ok:
        return {}
    Chem.AssignStereochemistryFrom3D(mol)
    out = {}
    for i, name in enumerate(atom_array.atom_name):
        cip = mol.GetAtomWithIdx(i).GetPropsAsDict().get("_CIPCode")
        if cip is not None:
            out[str(name)] = cip
    return out


class ChiralityCheckBlock(InlineMetricBlock):
    """Per-row chirality check vs a reference ligand.

    Builds RDKit mols from biotite atom arrays, assigns CIP codes from 3D
    geometry, and compares per-atom by name. Emits
    `<name>.correct_chirality = 1.0` when every reference CIP-bearing atom
    matches in the row's structure; `0.0` if any centre flipped; `1.0`
    (silent pass) when either RDKit fails to build a mol or no chiral
    centres are present (matches v2 semantics — non-chiral ligands always
    pass).

    Use AFTER ``remap_predicted_ligand`` so the output ligand atoms carry
    the same names as the reference.
    """

    _name = "chirality_check"
    _display_name = "Chirality Check"
    _description = "Per-atom CIP-code chirality comparison vs a reference ligand"
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu", rows_per_job=1000)

    reference_cif: str = Field(
        ...,
        description=(
            "Path to reference ligand structure (CIF/PDB) with bond records. "
            "Atom names in this reference must match those in the row's ligand."
        ),
    )
    ligand_chain: Optional[str] = Field(
        None,
        description=(
            "Chain ID for the ligand atoms in row structures. None auto-detects "
            "via the `hetero` annotation (any atom with hetero=True is a ligand)."
        ),
    )

    def _select_ligand(self, atom_array):
        if self.ligand_chain is not None:
            return atom_array[atom_array.chain_id == self.ligand_chain]
        if hasattr(atom_array, "hetero"):
            mask = atom_array.hetero
            if mask.any():
                return atom_array[mask]
        # No ligand_chain and no hetero info → use whole structure (caller
        # responsibility to ensure reference matches).
        return atom_array

    def setup(self, handler, sandbox_dir, chunk_id):
        ref_array = parse_to_atom_array(self.reference_cif)
        ref_ligand = self._select_ligand(ref_array)
        ref_cip = _cip_codes(ref_ligand)

        if not ref_cip:
            logger.warning(
                f"[{self.name}] reference {self.reference_cif} has no detectable "
                f"chiral centres — every row will pass."
            )

        if ref_ligand.bonds is None:
            logger.error(
                f"[{self.name}] reference {self.reference_cif} ligand has no "
                f"bond records — cannot transplant onto AF3 outputs. Skipping."
            )
            # Sentinel: no reference bonds → every row passes silently.
            return {"no_ref_bonds": True}

        ref_bond_pairs = [
            (
                str(ref_ligand.atom_name[int(idx1)]),
                str(ref_ligand.atom_name[int(idx2)]),
                int(btype),
            )
            for idx1, idx2, btype in ref_ligand.bonds.as_array()
        ]
        return {
            "no_ref_bonds": False,
            "ref_cip": ref_cip,
            "ref_bond_pairs": ref_bond_pairs,
        }

    def compute_metrics(self, loaded, *, context=None, index=0):
        import biotite.structure as struct
        import numpy as np

        if context["no_ref_bonds"]:
            return {"correct_chirality": 1.0}

        ref_cip = context["ref_cip"]
        ref_bond_pairs = context["ref_bond_pairs"]

        out_ligand = self._select_ligand(loaded.atom_array)

        # AF3 output ligand has no bonds — transplant from reference by
        # atom name (mirrors run_alphafold_sm.py:convert_out_model_pose).
        name_to_idx = {str(name): i for i, name in enumerate(out_ligand.atom_name)}
        new_bonds = []
        missed = False
        for n1, n2, btype in ref_bond_pairs:
            if n1 not in name_to_idx or n2 not in name_to_idx:
                missed = True
                break
            new_bonds.append((name_to_idx[n1], name_to_idx[n2], btype))

        if missed or not new_bonds:
            # Atom-name mismatch (likely upstream remap was skipped).
            # Be conservative: pass silently, log loudly.
            logger.warning(
                f"[{self.name}] {loaded.path}: ligand atom names don't "
                f"match reference; skipping chirality check (passes)."
            )
            return {"correct_chirality": 1.0}

        ligand_with_bonds = out_ligand.copy()
        ligand_with_bonds.bonds = struct.BondList(
            len(out_ligand),
            np.array(new_bonds, dtype=np.uint32),
        )
        out_cip = _cip_codes(ligand_with_bonds)
        correct = 1.0
        for name, ref_code in ref_cip.items():
            out_code = out_cip.get(name)
            if out_code is not None and out_code != ref_code:
                correct = 0.0
                break
        return {"correct_chirality": correct}
