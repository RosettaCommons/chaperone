"""Create explicit sequence variants while preserving structure lineage."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pydantic import Field

from pipelines import InlineBlockFunction, JobArgs, PipelineDataHandler
from pipelines.data.schema import make_row


_POSITION = re.compile(r"^(?P<chain>[^:\d]+):?(?P<resid>-?\d+)$")
_BACKBONE_AND_CB = frozenset({"N", "CA", "C", "O", "OXT", "CB"})
_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


def _parse_position(value: str) -> tuple[str, int]:
    match = _POSITION.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid residue position {value!r}; expected A42 or A:42")
    return match.group("chain"), int(match.group("resid"))


def mutate_residues(array, mutations: dict[str, str]):
    """Return a copy carrying the requested residue identities.

    Old side-chain atoms beyond ``CB`` are removed because their names and
    coordinates are not valid for the new residue.  Backbone, ``CB``, ligands,
    chain numbering, and all untouched residues are preserved.  This is enough
    to define a sequence variant for structure predictors; a predictor or
    packing block constructs the new side chain explicitly downstream.
    """

    output = array.copy()
    changed: list[dict[str, object]] = []
    strip = np.zeros(len(output), dtype=bool)

    for position, requested_name in mutations.items():
        chain, residue = _parse_position(position)
        requested_name = requested_name.strip().upper()
        if requested_name not in _AMINO_ACIDS:
            raise ValueError(
                f"Invalid amino-acid name {requested_name!r} for {position}; "
                "use a standard three-letter residue name"
            )
        mask = (
            (output.chain_id == chain)
            & (output.res_id == residue)
            & ~np.asarray(output.hetero, dtype=bool)
        )
        names = np.unique(output.res_name[mask])
        if len(names) != 1:
            raise KeyError(f"Protein residue {position!r} was not found uniquely")
        original_name = str(names[0])
        output.res_name[mask] = requested_name
        strip |= mask & ~np.isin(output.atom_name, tuple(_BACKBONE_AND_CB))
        changed.append(
            {
                "position": position,
                "from": original_name,
                "to": requested_name,
            }
        )

    return output[~strip], changed


class MutateResiduesBlock(InlineBlockFunction):
    """Expand each input structure into named residue-mutation variants."""

    _name = "mutate_residues"
    _display_name = "Mutate Residues"
    _description = "Create named protein sequence variants with explicit lineage"
    _inline = True
    _chunked = True
    _creates_new_step = True
    _resources = JobArgs(partition="cpu", mem_gb=4, rows_per_job=500)

    variants: dict[str, dict[str, str]] = Field(
        ...,
        description=(
            "Named mutation sets, e.g. {'a73f': {'A73': 'PHE'}, "
            "'double': {'A38': 'PHE', 'A73': 'PHE'}}."
        ),
    )

    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int = 0,
    ) -> PipelineDataHandler:
        from atomworks.io.utils.io_utils import to_cif_file
        from pipelines.io import ensure_bonds

        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for source_index, source in enumerate(
            handler.get_data(which="last", load=True)
        ):
            source_stem = Path(source.path).name
            for suffix in (".cif.gz", ".pdb.gz", ".cif", ".pdb", ".gz"):
                if source_stem.endswith(suffix):
                    source_stem = source_stem[: -len(suffix)]
                    break
            for variant, mutations in self.variants.items():
                if not variant or not re.fullmatch(r"[A-Za-z0-9_.-]+", variant):
                    raise ValueError(
                        f"Invalid variant name {variant!r}; use letters, digits, '.', '_' or '-'"
                    )
                mutated, changed = mutate_residues(source.atom_array, mutations)
                # LigandMPNN derives output IDs by splitting the input basename
                # at the first dot. A dot before the variant therefore makes
                # several variants of one source overwrite each other's packed
                # outputs. Keep the variant in the dot-free stem.
                output_path = sandbox_dir / f"{source_stem}__{variant}.cif"
                to_cif_file(ensure_bonds(mutated), str(output_path))
                rows.append(
                    make_row(
                        name=self.name,
                        path=str(output_path),
                        metadata={
                            "variant": variant,
                            "mutations": changed,
                            "source_path": source.path,
                            "source_index": source_index,
                            "chunk_id": chunk_id,
                        },
                    )
                )
        return PipelineDataHandler(rows)
