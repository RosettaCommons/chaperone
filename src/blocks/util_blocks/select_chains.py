"""Select explicit structure chains while retaining parent lineage."""

import json
from pathlib import Path

import numpy as np
from atomworks.io.utils.io_utils import to_cif_file
from pydantic import Field

from chaperone import BlockFunction, PipelineDataHandler
from chaperone.io import ensure_bonds
from chaperone.data import make_row


class SelectChainsBlock(BlockFunction):
    _name = "select_chains"
    _description = "Keep specified chains, preserving residue IDs and parent lineage"
    _inline = True
    _chunked = False
    chains: list[str] = Field(..., min_length=1)

    def __call__(self, handler, sandbox_dir, **kwargs):
        root = Path(sandbox_dir)
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, loaded in enumerate(handler.get_data(load=True)):
            atoms = loaded.atom_array
            missing = set(self.chains) - set(atoms.chain_id)
            if missing:
                raise ValueError(f"Cannot select missing chains: {sorted(missing)}")
            selected = atoms[np.isin(atoms.chain_id, self.chains)]
            keys = {f"{c}{r}" for c, r in zip(selected.chain_id, selected.res_id)}
            metadata = dict(loaded.metadata)
            metadata["motif"] = [k for k in metadata.get("motif") or [] if k in keys]
            mapping = metadata.get("diffused_index_map") or {}
            if isinstance(mapping, str):
                mapping = json.loads(mapping)
            metadata["diffused_index_map"] = json.dumps(
                {k: v for k, v in mapping.items() if v in keys}
            )
            metadata["motif_atom_contigs"] = {
                k: v
                for k, v in (metadata.get("motif_atom_contigs") or {}).items()
                if k in keys
            }
            path = root / f"{loaded.basename}_chains_{index}.cif"
            to_cif_file(ensure_bonds(selected), path)
            rows.append(
                make_row(
                    name=self.name,
                    path=str(path),
                    path_handle=loaded.path,
                    parent_design_id=loaded.design_id,
                    metadata=metadata,
                )
            )
        return (
            PipelineDataHandler.merge(handler, PipelineDataHandler(rows))
            if rows
            else handler[:0]
        )
