"""Prepare an RFdiffusion C3 scaffold without losing its motif or symmetry axis."""

import json
from pathlib import Path

import biotite.structure as struc
import numpy as np
from pydantic import Field
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from chaperone import BlockFunction, PipelineDataHandler
from chaperone.data import make_row
from chaperone.io import parse_to_atom_array, write_pdb


def ca_lookup(atoms):
    ca = atoms[atoms.atom_name == "CA"]
    keys = [f"{chain}{res}" for chain, res in zip(ca.chain_id, ca.res_id)]
    if len(keys) != len(set(keys)) or not np.isfinite(ca.coord).all():
        raise ValueError("Expected unique, finite protein CA coordinates")
    return dict(zip(keys, ca.coord))


def prepare_c3(atoms, motif_reference, clash_reference, index_map, max_c3_rmsd=2.0):
    """Align the stalk through mapped motif CAs, then orient the C3 axis to z.

    Returns one subunit with its radial displacement intact. RPXdock reconstructs
    the ideal trimer from this subunit. Clash counts exclude the scaffold motif.
    """
    chains = list(struc.get_chains(atoms))
    if len(chains) != 3:
        raise ValueError("Nanoparticle preparation requires exactly three chains")
    source_ca, target_ca = ca_lookup(motif_reference), ca_lookup(atoms)
    if not source_ca or not all(
        key in index_map and index_map[key] in target_ca for key in source_ca
    ):
        raise ValueError(
            "Missing RFdiffusion motif correspondence (diffused_index_map)"
        )
    fixed = np.array([target_ca[index_map[key]] for key in source_ca])
    moving = np.array(list(source_ca.values()))
    _, transform = struc.superimpose(fixed, moving)
    aligned_stalk = transform.apply(
        clash_reference.coord[clash_reference.atom_name == "CA"]
    )
    designed = np.array(
        [xyz for key, xyz in target_ca.items() if key not in set(index_map.values())]
    )
    n_clashes = 0
    if len(designed):
        # Count unique stalk CAs, not all atom pairs (which overcounts contacts).
        distance, _ = cKDTree(designed).query(aligned_stalk)
        n_clashes = int(np.count_nonzero(distance < 3.0))

    subunits = [atoms[atoms.chain_id == chain] for chain in chains]
    cas = [part.coord[part.atom_name == "CA"] for part in subunits]
    if len({len(ca) for ca in cas}) != 1 or len(cas[0]) < 3:
        raise ValueError(
            "C3 subunits must have equal residue counts and at least three residues"
        )
    centroids = np.array([ca.mean(axis=0) for ca in cas])
    center = centroids.mean(axis=0)
    axis = np.cross(centroids[1] - centroids[0], centroids[2] - centroids[0])
    radial = centroids[0] - center
    if np.linalg.norm(axis) < 1e-6 or np.linalg.norm(radial) < 1e-6:
        raise ValueError("Degenerate C3 subunit centroids")
    axis /= np.linalg.norm(axis)
    radial /= np.linalg.norm(radial)
    basis = np.column_stack([radial, np.cross(axis, radial), axis])
    canonical = [(ca - center) @ basis for ca in cas]
    errors = []
    for index in (1, 2):
        predicted = Rotation.from_euler("z", 120 * index, degrees=True).apply(
            canonical[0]
        )
        errors.append(
            float(np.sqrt(np.mean(np.sum((predicted - canonical[index]) ** 2, axis=1))))
        )
    deviation = max(errors)
    if deviation > max_c3_rmsd:
        raise ValueError(f"Input is not C3 symmetric: CA deviation {deviation:.3f} A")
    monomer = subunits[0].copy()
    monomer.coord = (monomer.coord - center) @ basis
    residues = list(dict.fromkeys(monomer.res_id))
    motif = [
        f"A{i + 1}"
        for i, res in enumerate(residues)
        if f"{chains[0]}{res}" in index_map.values()
    ]
    if not motif:
        raise ValueError("The retained subunit has no fixed motif")
    original_ids = monomer.res_id.copy()
    for index, residue in enumerate(residues, 1):
        monomer.res_id[original_ids == residue] = index
    monomer.chain_id[:] = "A"
    if "chain_iid" in monomer.get_annotation_categories():
        monomer.chain_iid[:] = "A"
    return (
        monomer,
        motif,
        {
            "n_clashes": n_clashes,
            "c3_ca_rmsd": deviation,
            "residue_count": len(residues),
        },
    )


class NanoparticlePrepareBlock(BlockFunction):
    _name = "nanoparticle_prepare"
    _description = (
        "Align antigen clash reference and preserve a C3 scaffold's symmetry axis"
    )
    motif_pdb: str = Field(..., description="The original RFdiffusion C3 antigen stub")
    clash_ref_pdb: str = Field(
        ..., description="Full antigen stalk in the stub's coordinate frame"
    )
    max_clashes: int = Field(8, ge=0)
    max_c3_rmsd: float = Field(2.0, gt=0)

    def __call__(self, handler, sandbox_dir, **kwargs):
        root = Path(sandbox_dir)
        root.mkdir(parents=True, exist_ok=True)
        motif_ref = parse_to_atom_array(self.motif_pdb)
        clash_ref = parse_to_atom_array(self.clash_ref_pdb)
        rows = []
        for index, item in enumerate(handler.get_data(which="last", load=True)):
            mapping = item.metadata.get("diffused_index_map") or {}
            if isinstance(mapping, str):
                mapping = json.loads(mapping)
            monomer, motif, metrics = prepare_c3(
                item.atom_array, motif_ref, clash_ref, mapping, self.max_c3_rmsd
            )
            if metrics["n_clashes"] > self.max_clashes:
                continue
            path = root / f"{Path(item.path).stem}_asu{index}.pdb"
            write_pdb(monomer, path)
            rows.append(
                make_row(
                    name=self.name,
                    path=path,
                    path_handle=item.path,
                    parent_design_id=item.design_id,
                    metadata={
                        "motif": motif,
                        "diffused_index_map": "{}",
                        "motif_atom_contigs": {},
                    },
                    metrics={self.name: metrics},
                )
            )
        return (
            PipelineDataHandler.merge(handler, PipelineDataHandler(rows))
            if rows
            else handler[:0]
        )
