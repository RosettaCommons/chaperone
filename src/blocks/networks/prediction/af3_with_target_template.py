"""AF3 subclass that injects the target receptor as a structural template.

The default AF3 block predicts the binder+receptor complex de novo. For binder
design where the receptor structure is known and fixed (it came from an
experimental structure used by RFD3), giving AF3 the receptor as a template
materially improves prediction quality of both the receptor and the interface.

Usage in pipeline.py:

    AF3WithTargetTemplate(
        target_chain="B",
        # Optional: replace the (possibly truncated) receptor coming out of
        # RFD3/MPNN with a full external receptor PDB. Useful when RFD3 was
        # restricted to a sub-contig for speed but AF3 should see the full target.
        full_receptor_pdb="/abs/path/to/full_receptor.pdb",
    )

Implementation:
- Builds the AF3 input JSON from the design AtomArray (binder + receptor as
  produced by RFD3/MPNN).
- If ``full_receptor_pdb`` is set, replaces the target chain in the assembled
  array with the full receptor before serialising to JSON, so chain B's sequence
  AND template are both the full receptor. The binder coordinates / residues are
  unaffected.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal, Optional

from pydantic import Field, model_validator

from blocks.constants import AF3_RELEASE_LINES
from blocks.networks.prediction.helpers import create_af3_json_from_design
from blocks.networks.prediction.alphafold3 import AlphaFold3Block

logger = logging.getLogger(__name__)


def _to_cif_string(chain_atom_array, name: str) -> str:
    """Serialize a single-chain AtomArray to an mmCIF string for AF3 templates."""
    import numpy as np
    from atomworks.io.utils.io_utils import (
        to_cif_string,
    )  # local import; same util the helper uses

    from chaperone.io import ensure_bonds

    template = ensure_bonds(chain_atom_array).copy()
    categories = template.get_annotation_categories()
    if "occupancy" not in categories:
        template.set_annotation("occupancy", np.ones(len(template), dtype=float))
    if "b_factor" not in categories:
        template.set_annotation("b_factor", np.zeros(len(template), dtype=float))
    return to_cif_string(template, id=name) + "".join(AF3_RELEASE_LINES)


def _swap_target_chain(design_atom_array, full_receptor_atom_array, target_chain: str):
    """Return a new atom_array with the target chain replaced by the full receptor's atoms.

    Drops every atom in the design where chain_id == target_chain, then appends
    every atom of the full receptor (relabelled to target_chain in case it isn't).
    """
    import numpy as np
    from biotite.structure import concatenate

    keep_mask = design_atom_array.chain_id != target_chain
    binder_only = design_atom_array[keep_mask].copy()
    full_recv = full_receptor_atom_array.copy()
    new_chain_ids = np.asarray(full_recv.chain_id).copy()
    new_chain_ids[:] = target_chain
    full_recv.chain_id = new_chain_ids
    if "chain_iid" in full_recv.get_annotation_categories():
        full_recv.set_annotation("chain_iid", new_chain_ids.copy())
    assembled = concatenate([binder_only, full_recv])
    # Coordinate-free sequence inputs only carry the minimal atom annotations.
    # Biotite concatenation keeps the intersection of annotation categories, so
    # without restoring these fields a target template can lose B factors and
    # produce an mmCIF that AF3 refuses to parse.
    if "occupancy" not in assembled.get_annotation_categories():
        assembled.set_annotation("occupancy", np.ones(len(assembled), dtype=float))
    if "b_factor" not in assembled.get_annotation_categories():
        assembled.set_annotation("b_factor", np.zeros(len(assembled), dtype=float))
    return assembled


class AF3WithTargetTemplate(AlphaFold3Block):
    """AF3 with the design's target chain attached as a structural template."""

    _name = "af3_with_target_template"
    _description = "AlphaFold3 with the target chain structure as a fixed template"

    input_mode: Literal["json"] = "json"

    target_chain: str = Field(
        default="B",
        description="Chain ID of the receptor / target whose structure should be templated.",
    )
    template_kalign_matches: Optional[int] = Field(
        default=None,
        description="If set, override AF3's `templates[].kalign_matches`. Default lets AF3 decide.",
    )
    full_receptor_pdb: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a PDB/CIF containing the full target chain. When set, "
            "AF3's chain `target_chain` is replaced with this full receptor (sequence + "
            "template), regardless of the truncated receptor that RFD3/MPNN emitted."
        ),
    )

    @model_validator(mode="after")
    def _validate_full_receptor_pdb_exists(self):
        if self.full_receptor_pdb is not None and not os.path.exists(
            self.full_receptor_pdb
        ):
            raise ValueError(
                f"full_receptor_pdb does not exist: {self.full_receptor_pdb}"
            )
        return self

    def _load_full_receptor(self):
        """Load the full receptor atom array, restricted to ``target_chain``."""
        from chaperone.io import parse_to_atom_array

        aa = parse_to_atom_array(self.full_receptor_pdb, remove_hydrogens=True)
        sub = aa[aa.chain_id == self.target_chain]
        if len(sub) == 0:
            raise ValueError(
                f"full_receptor_pdb {self.full_receptor_pdb!r} has no chain "
                f"{self.target_chain!r}; available chains: "
                f"{sorted(set(str(c) for c in aa.chain_id))}"
            )
        return sub

    def _prepare_af3_inputs(self, handler, inputdir):
        full_receptor = self._load_full_receptor() if self.full_receptor_pdb else None
        if hasattr(handler, "get_prediction_inputs"):
            inputs = handler.get_prediction_inputs(which=self.which, load_context=True)
        else:  # Backward-compatible duck-typed handlers used by integrations.
            inputs = handler.get_data(which=self.which, load=True)
        for loaded in inputs:
            json_file = os.path.join(inputdir, f"{loaded.basename}.json")
            if os.path.exists(json_file):
                logger.info("Skipping existing input: %s", json_file)
                continue

            atoms = loaded.atom_array
            if full_receptor is not None:
                atoms = _swap_target_chain(atoms, full_receptor, self.target_chain)

            json_data = create_af3_json_from_design(
                atom_array=atoms,
                name=loaded.basename,
                remove_ligands=self.remove_ligands,
                specified_smiles=self.specified_smiles,
                specified_ccd=self.specified_ccd,
                disable_ccd_detection=self.disable_ccd_detection,
                protein_sequences=getattr(loaded, "sequences", None),
            )
            json_data["modelSeeds"] = self.model_seeds

            self._inject_target_template(json_data, atoms)

            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=4)

    def _inject_target_template(self, json_data: dict, atom_array) -> None:
        """Attach the full target chain as a structural template for that chain."""
        target_entry = None
        for entry in json_data.get("sequences", []):
            protein = entry.get("protein")
            if protein is not None and str(protein.get("id")) == self.target_chain:
                target_entry = protein
                break
        if target_entry is None:
            logger.warning(
                "AF3WithTargetTemplate: chain %r not in JSON; skipping template injection.",
                self.target_chain,
            )
            return

        chain_mask = atom_array.chain_id == self.target_chain
        if not chain_mask.any():
            logger.warning(
                "AF3WithTargetTemplate: chain %r not in atom_array; skipping.",
                self.target_chain,
            )
            return

        chain_atoms = atom_array[chain_mask]
        seq_len = len(target_entry["sequence"])
        cif_string = _to_cif_string(
            chain_atoms, name=f"target_chain_{self.target_chain}"
        )
        template = {
            "mmcif": cif_string,
            "queryIndices": list(range(seq_len)),
            "templateIndices": list(range(seq_len)),
        }
        if self.template_kalign_matches is not None:
            template["kalign_matches"] = self.template_kalign_matches
        target_entry["templates"] = [template]
        logger.info(
            "AF3WithTargetTemplate: attached template (%d residues) to chain %s of %s",
            seq_len,
            self.target_chain,
            json_data.get("name"),
        )
