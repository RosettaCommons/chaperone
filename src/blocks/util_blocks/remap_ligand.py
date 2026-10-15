"""Chunk-splittable block that remaps predicted ligand atom/residue names.

Encapsulates the logic of
:func:`blocks.networks.prediction.helpers.remap_predicted_ligand` as a proper
block so it slots into an ``engine.run([...])`` chain and runs in parallel
across SLURM chunks like every other block.

Key difference from the standalone function: the standalone function does
**global** discovery (scans every structure) and rewrites every output CIF
**in place**. A chunked block never sees all rows at once and must not mutate
shared upstream files, so this block:

  1. Discovers the predicted→reference mapping **per chunk** (the engine slices
     the upstream parquet, so each worker scans only its own rows). Because
     every prediction output of the same ligand shares identical atom naming,
     per-chunk discovery yields the same mapping as the global scan.
  2. Copies each row's structure CIF into the block's own ``sandbox_dir`` and
     rewrites the atom/residue names in that **copy** — the upstream files are
     never touched. The returned handler points at the sandbox copies.
  3. Passes a chunk through unchanged when no un-remapped match is found
     (already-remapped chunks are a no-op, not a failure).

catcif caveat: ``apply_atom_name_remap`` (and the underlying text rewrite) use
raw ``open()`` and cannot dereference an ``archive.catcif:entry_id`` tag. This
block resolves every path via ``LoadedStructure`` (which materialises catcif
structures to a temp CIF) *before* copying into the sandbox, so it works with
either raw CIFs or catcif archives.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from pydantic import Field

from blocks.networks.prediction.helpers import (
    build_remap_reference,
    discover_ligand_remap,
)
from chaperone import InlineBlockFunction, JobArgs, PipelineDataHandler
from chaperone.data.structure import _resolve_path
from chaperone.io import parse_to_atom_array

logger = logging.getLogger(__name__)


def _select_ligand_atoms(atom_array):
    """Select a ligand robustly from either PDB HETATM or mmCIF chain metadata."""
    import numpy as np
    from atomworks.enums import ChainType

    hetero = np.asarray(atom_array.hetero, dtype=bool)
    ligand = atom_array[hetero]
    if len(ligand) == 0:
        ligand = atom_array[atom_array.chain_type == ChainType.NON_POLYMER]
    return ligand[ligand.element != "H"]


def _rewrite_cif_copy(src_path: str, dst_path: str, lookup: dict[str, str]) -> None:
    """Copy ``src_path`` to ``dst_path`` and rewrite ligand atom/residue names.

    A single combined whitespace-bounded regex replaces predicted names with
    reference names (longest key first so ``C10`` matches before ``C1``). When
    ``lookup`` is empty (no-op chunk) the file is copied verbatim.
    """
    if not lookup:
        shutil.copyfile(src_path, dst_path)
        return

    sorted_keys = sorted(lookup.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<=\s)(" + "|".join(re.escape(k) for k in sorted_keys) + r")(?=\s)"
    )

    import gzip

    opener = gzip.open if src_path.endswith(".gz") else open
    with opener(src_path, "rt") as f:
        content = f.read()
    content = pattern.sub(lambda m: lookup[m.group(0)], content)
    with open(dst_path, "w") as f:
        f.write(content)


class RemapLigandBlock(InlineBlockFunction):
    """Rename predicted ligand atom/residue names to match a reference CIF.

    Use inside an ``engine.run`` chain right after a prediction oracle so the
    metric blocks downstream (hbond, contact_ms, rmsd, …) read the canonical
    reference atom names::

        engine.run(
            [
                Blocks.get("remap_ligand")(
                    reference_cif=LIGAND_CIF,
                    target_name=LIGAND_NAME,
                    core_atoms=CORE_ATOMS,
                ),
                HBond(name="af3_hbond"),
                ...,
            ],
            data=af3_handler,
        )

    SMILES is read automatically from the reference CIF's bond records.
    """

    _name = "remap_ligand"
    _display_name = "Remap Ligand"
    _description = "Rename predicted ligand atom/residue names to match a reference CIF"
    _inline = True
    _chunked = True
    _creates_new_step = False
    _resources = JobArgs(partition="cpu", mem_gb=8, rows_per_job=2000)

    reference_cif: str = Field(
        ...,
        description="Path to a ligand-only CIF with canonical atom names and "
        "_chem_comp_bond records.",
    )
    target_name: Optional[str] = Field(
        None,
        description="Desired residue name in outputs (e.g. 'B;N'). Read from the "
        "reference CIF if None.",
    )
    atom_name_mapping: Optional[dict[str, str]] = Field(
        None,
        description=(
            "Optional explicit predicted-to-reference atom-name mapping. When "
            "provided, bypass graph discovery and validate each row against the "
            "mapping's complete source or target name set. This is useful for "
            "large homogeneous predictor outputs with a previously validated "
            "serialization convention."
        ),
    )
    core_atoms: Optional[list[str]] = Field(
        None,
        description=(
            "Reference atom names of the rigid core. When given, the RMSD used "
            "for the substructure-match tiebreak is computed only over these "
            "atoms, ignoring flexible linker tails whose rotamer space would "
            "inflate the full-atom RMSD of correct mappings (and otherwise "
            "prevent the scan from ever clearing rmsd_ok, forcing a slow "
            "full-chunk fall-through)."
        ),
    )
    buried_atoms: Optional[list[str]] = Field(
        None,
        description="Backward-compatible alias for core_atoms.",
    )
    sample_size: int = Field(
        64,
        description="Scan this many structures first before streaming the rest "
        "of the chunk.",
    )
    rmsd_ok: float = Field(
        2.0,
        description="Stop on the first mapping with RMSD below this threshold (Å).",
    )
    num_workers: Optional[int] = Field(
        0,
        description="Parallel workers for the atom-mapping scan. 0 (default) runs "
        "serial — avoids a fork() deadlock when the scan runs inside a 1-CPU SLURM "
        "allocation after torch/numpy have initialised their thread pools, and is "
        "reproducible. Set None → min(cpu_count, 8) for standalone multi-core use.",
        json_schema_extra={"cli_type": int},
    )
    emit_metric: bool = Field(
        True,
        description="Store the match RMSD under '<name>.match_rmsd' on every "
        "remapped row.",
    )

    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int = 0,
    ) -> PipelineDataHandler:
        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Resolve every last-structure path off disk first (materialises
        # catcif structures to a temp CIF), so the scan and the
        # copy operate on a real file rather than an archive tag.
        loaded_rows = list(handler.get_data(load=False, which="last"))
        if not loaded_rows:
            logger.info("RemapLigandBlock[%s]: empty chunk, skipping.", chunk_id)
            return handler

        resolved_paths = [
            _resolve_path(ls.path, ls.storage_format) for ls in loaded_rows
        ]

        # ── Discover the mapping for THIS chunk ──────────────────────────
        reference = None
        if self.atom_name_mapping is not None:
            if not self.target_name:
                raise ValueError(
                    "target_name is required when atom_name_mapping is provided"
                )
            target_name = self.target_name
            atom_remap = dict(self.atom_name_mapping)
            best_rmsd = None
            best_pred_res = "explicit mapping"
            reference_atom_names = set(atom_remap.values())
            mapping_source_names = set(atom_remap)
            n_ref_atoms = len(atom_remap)
        else:
            reference = build_remap_reference(
                self.reference_cif,
                target_name=self.target_name,
                core_atoms=self.core_atoms,
                buried_atoms=self.buried_atoms,
            )
            target_name = reference["target_name"]

            result = discover_ligand_remap(
                resolved_paths,
                reference,
                sample_size=self.sample_size,
                rmsd_ok=self.rmsd_ok,
                num_workers=self.num_workers,
            )

            if result is None:
                expected = set(reference["ref_atom_names"])
                canonical = []
                for resolved_path in resolved_paths:
                    atoms = parse_to_atom_array(resolved_path)
                    ligand = _select_ligand_atoms(atoms)
                    named = ligand[ligand.res_name == target_name]
                    canonical.append(set(named.atom_name) == expected)
                if all(canonical):
                    logger.info(
                        "RemapLigandBlock[%s]: all %d rows already use canonical "
                        "ligand names for %s",
                        chunk_id,
                        len(loaded_rows),
                        target_name,
                    )
                    return handler
                bad = [i for i, value in enumerate(canonical) if not value]
                raise ValueError(
                    "Ligand remap discovery failed and rows are not already "
                    f"canonical for {target_name}: row indices {bad[:10]}"
                )

            best_rmsd, atom_remap, best_pred_res = result
            reference_atom_names = set(reference["ref_atom_names"])
            mapping_source_names = set(atom_remap)
            n_ref_atoms = reference["n_ref_atoms"]
        if (
            len(atom_remap) != n_ref_atoms
            or len(set(atom_remap.values())) != n_ref_atoms
        ):
            raise ValueError(
                "Discovered ligand atom remap is not bijective: "
                f"{len(atom_remap)} source and {len(set(atom_remap.values()))} "
                f"target names for {n_ref_atoms} atoms"
            )

        if best_rmsd is None:
            logger.info(
                "RemapLigandBlock[%s]: applying explicit mapping → %s "
                "(%d atoms) across %d rows",
                chunk_id,
                target_name,
                len(atom_remap),
                len(loaded_rows),
            )
        else:
            rmsd_label = "core" if reference["ref_buried_idx"] is not None else "full"
            logger.info(
                "RemapLigandBlock[%s]: %s → %s (%d atoms, %s-atom RMSD %.3f Å) "
                "across %d rows",
                chunk_id,
                best_pred_res,
                target_name,
                len(atom_remap),
                rmsd_label,
                best_rmsd,
                len(loaded_rows),
            )

        # ── Apply: copy each CIF into the sandbox and rewrite the copy ───
        rows_out = []
        for ls, src in zip(loaded_rows, resolved_paths):
            # A handler may contain multiple source naming conventions (for
            # example canonical A;N/B;N rows together with one-based LIG_B
            # rows). Applying the first discovered mapping to every row can
            # corrupt already-canonical names because the two name sets
            # overlap. Select or discover a mapping per naming convention.
            atom_array = parse_to_atom_array(src, remove_hydrogens=True)
            # PDB inputs lack mmCIF entity metadata, so parsers can classify the
            # entire protein chain as NON_POLYMER.  HETATM is the authoritative
            # ligand selector when present; retain chain_type as the fallback
            # for formats that do not carry a hetero annotation.
            ligand = _select_ligand_atoms(atom_array)
            row_atom_names = set(str(name) for name in ligand.atom_name)
            row_res_names = set(str(name) for name in ligand.res_name)

            row_lookup: dict[str, str]
            row_best_rmsd = best_rmsd
            if row_atom_names == reference_atom_names:
                row_lookup = {}
            elif row_atom_names == mapping_source_names:
                row_lookup = dict(atom_remap)
            else:
                if reference is None:
                    raise ValueError(
                        "Ligand atom names match neither side of the explicit mapping "
                        f"for {ls.path}: names={sorted(row_atom_names)}"
                    )
                row_result = discover_ligand_remap(
                    [src],
                    reference,
                    sample_size=0,
                    rmsd_ok=self.rmsd_ok,
                    num_workers=0,
                )
                if row_result is None:
                    raise ValueError(
                        "No safe ligand atom remap for heterogeneous row "
                        f"{ls.path}: names={sorted(row_atom_names)}"
                    )
                row_best_rmsd, row_remap, _ = row_result
                if (
                    len(row_remap) != reference["n_ref_atoms"]
                    or len(set(row_remap.values())) != reference["n_ref_atoms"]
                ):
                    raise ValueError(
                        "Per-row ligand atom remap is not bijective for " f"{ls.path}"
                    )
                row_lookup = dict(row_remap)

            if len(row_res_names) == 1:
                row_res_name = next(iter(row_res_names))
                if row_res_name != target_name:
                    row_lookup[row_res_name] = target_name

            raw = str(ls.path)
            if ".catcif:" in raw:
                # catcif tag ``archive.catcif:entry_id`` — the entry_id is the
                # per-structure identity. Using the archive basename instead
                # collapses every row in the chunk to the same stem (file
                # collision) and breaks lineage backlinking (the merge key
                # strips ``af3_cycle0.chunk_0.catcif`` down to ``af3_cycle0``).
                stem = raw.rsplit(":", 1)[1]
                if not stem.endswith((".cif", ".cif.gz", ".pdb")):
                    stem += ".cif"
            else:
                stem = Path(raw.split(":", 1)[0]).name
            # Normalise compressed copies to a plain .cif in the sandbox.
            if stem.endswith(".gz"):
                stem = stem[:-3]
            dst = sandbox_dir / stem
            _rewrite_cif_copy(src, str(dst), row_lookup)

            metrics_update = (
                {self.name: {"match_rmsd": float(row_best_rmsd)}}
                if self.emit_metric and row_best_rmsd is not None
                else None
            )
            rows_out.append(
                ls.to_row_with_updates(path=str(dst), metrics=metrics_update)
            )

        return PipelineDataHandler(rows_out)
