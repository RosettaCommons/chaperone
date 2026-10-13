"""Turn sequence-only RNA design submissions into scoreable pipeline rows.

A motif-scaffolding evaluation accepts a *sequence* and a *motif map* - never
coordinates, because coordinates from the designer cannot be trusted and would
make the oracle refold redundant. Two things then have to happen before any
metric block can run:

1. **The sequence must enter as RNA.** Oracles build their entity from the
   chain type, so a sequence spoofed with CA atoms and amino-acid residue names
   is folded as a peptide. ``write_rna_design_cif`` writes one C3' anchor per
   nucleotide with RNA residue names, plus any cofactor as a hetero residue so
   the oracle carries the ion (AF3 reads it as a CCD ligand entity).

2. **The motif map must become ``diffused_index_map``.** Every motif-aware
   block in the repo - ``reference_motif_rmsd``, ``motif_geometry``,
   ``motif_confidence``, ``rna_thermo``, ``sequence_identity_gate`` - addresses
   residues by their *reference* id and resolves them through that metadata
   key, which RFD3 writes for a diffusion output. A submitted map is the same
   information, so writing it under the same key makes every one of those
   blocks work on a submission with no special-casing.

    handler = rna_design_handler(
        [{"design_id": "d017", "sequence": "GGCU...", "motif_map": {"M10": 22}}],
        out_dir, ligands=["MN"],
    )
    engine.run(["af3", RefMotifRMSD(reference=motif_pdb, ligands=[]), ...], data=handler)
"""

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from pipelines.data import make_row
from pipelines.data.handler import PipelineDataHandler

logger = logging.getLogger(__name__)

RNA_BASES = frozenset("ACGU")


def write_rna_design_cif(
    sequence: str,
    path,
    *,
    chain: str = "A",
    ligands: Sequence[str] = (),
    ligand_chain: str = "B",
) -> Path:
    """Write a coordinate-free RNA chain, plus one hetero atom per cofactor.

    Coordinates are NaN: this is an oracle *input*, and any coordinate here
    would be a coordinate the submitter did not provide. Cofactors are written
    as single-atom hetero residues named by their CCD code, which is how the
    AF3/Boltz/Chai input builders pick them up as ligand entities.
    """
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import AtomArray

    from pipelines.io import ensure_bonds

    sequence = sequence.upper()
    invalid = sorted(set(sequence) - RNA_BASES)
    if invalid:
        raise ValueError(f"Sequence has non-RNA residues {invalid}: {sequence[:50]}...")
    if not sequence:
        raise ValueError("Empty sequence")

    atoms = AtomArray(len(sequence) + len(ligands))
    for i, base in enumerate(sequence):
        atoms.coord[i] = [np.nan, np.nan, np.nan]
        atoms.chain_id[i] = chain
        atoms.res_id[i] = i + 1
        atoms.res_name[i] = base
        atoms.atom_name[i] = "C3'"
        atoms.element[i] = "C"
        atoms.hetero[i] = False

    for j, ligand in enumerate(ligands):
        i = len(sequence) + j
        atoms.coord[i] = [np.nan, np.nan, np.nan]
        atoms.chain_id[i] = ligand_chain
        atoms.res_id[i] = j + 1
        atoms.res_name[i] = ligand
        atoms.atom_name[i] = ligand
        atoms.element[i] = ligand.capitalize()
        atoms.hetero[i] = True

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomworks otherwise stamps the current date/time into the mmCIF header.
    # The engine fingerprints input bytes, so identical submissions would miss
    # cache and repeat an expensive oracle fold on every scorer invocation.
    to_cif_file(
        ensure_bonds(atoms),
        str(path),
        id=path.name,
        date="1970-01-01",
        time="00:00:00",
    )
    return path


def motif_index_map(
    motif_map: Mapping[str, int], chain: str = "A", *, one_based: bool = False
) -> dict:
    """``{reference id: sequence index}`` -> ``{reference id: design residue id}``.

    The submission's indices are 0-based positions in the submitted sequence;
    residue ids in a structure are 1-based, so the offset is applied here once
    rather than in every block that consumes the map.
    """
    offset = 0 if one_based else 1
    return {
        str(ref): f"{chain}{int(index) + offset}" for ref, index in motif_map.items()
    }


def rna_design_handler(
    designs: Iterable[Mapping],
    out_dir,
    *,
    chain: str = "A",
    ligands: Sequence[str] = (),
    sequence_key: str = "sequence",
    motif_key: str = "motif_map",
    id_key: str = "design_id",
    one_based_motif: bool = False,
) -> PipelineDataHandler:
    """Handler of scorer-owned RNA inputs, one row per submitted design.

    Each design is a mapping with a sequence, an optional id, and an optional
    ``{reference residue id: sequence index}`` motif map. Every other key is
    carried through as a metric so the submission's own columns survive into
    the scored table.
    """
    out_dir = Path(out_dir)
    rows = []
    for i, design in enumerate(designs):
        design = dict(design)
        sequence = design.pop(sequence_key, None)
        if not sequence:
            raise ValueError(f"Design {i} has no {sequence_key!r}")
        design_id = str(design.pop(id_key, None) or f"design_{i:05d}")
        motif_map = design.pop(motif_key, None)

        cif_path = write_rna_design_cif(
            sequence, out_dir / f"{design_id}.cif", chain=chain, ligands=ligands
        )
        metadata = {}
        if motif_map:
            metadata["diffused_index_map"] = json.dumps(
                motif_index_map(motif_map, chain, one_based=one_based_motif)
            )
        rows.append(
            make_row(
                name="ingest",
                path=str(cif_path),
                metadata=metadata,
                metrics={
                    id_key: design_id,
                    sequence_key: sequence,
                    "sequence_length": len(sequence),
                    **design,
                },
            )
        )

    if not rows:
        raise ValueError("No designs supplied")
    logger.info("Materialized %d RNA designs into %s", len(rows), out_dir)
    return PipelineDataHandler(rows)


def load_submission(
    path,
    *,
    motif_column: Optional[str] = "motif_map",
) -> list[dict]:
    """Read a CSV/parquet submission into design mappings.

    The motif column, when present, holds a JSON object per row -
    ``{"M10": 22, "M11": 23}`` - because a flat table cannot carry a mapping
    any other way and a positional list silently breaks when the target's motif
    order is revised.
    """
    import polars as pl

    path = Path(path)
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    designs = []
    for row in frame.to_dicts():
        if motif_column and isinstance(row.get(motif_column), str):
            row[motif_column] = json.loads(row[motif_column])
        designs.append(row)
    return designs
