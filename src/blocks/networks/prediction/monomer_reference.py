"""Shared helpers for single-sequence protein monomer predictors.

SimpleFold and ColabFold (AlphaFold2) both fold one protein sequence at a time.
Their shared sequence and MSA preparation lives here instead of being copied
per block:

* ``protein_monomer`` / ``residue_sequence`` — crop an input structure to its
  single protein chain and read the FASTA the predictor should be given, using
  the CCD's one-letter code or canonical parent component for modified
  residues (MSE->M, SEP->S, PTR->Y, ...).
* ``stage_monomer_manifest`` — write the predictor sequence/provenance manifest
  used across ``create_command`` and ``compile``.
* ``resolve_msa_path`` / ``stage_query_a3m`` — look up an on-disk protein MSA
  by sequence hash from the same sharded roots RF3 validation uses
  (``PROTEIN_MSA_DIRS``), and stage it as a plain a3m.
"""

from __future__ import annotations

import functools
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from biotite.structure import filter_amino_acids, get_residue_starts
from biotite.structure.info.ccd import get_ccd

from pipelines import get_setting_field
from pipelines.io import RES_NAME_TO_ONE_LETTER

logger = logging.getLogger(__name__)

#: Sharded on-disk MSA roots are described exactly as in RF3/AtomWorks:
#: ``{"dir": ..., "extension": ..., "directory_depth": ...}``.
DEFAULT_MSA_EXTENSION = ".a3m.gz"
DEFAULT_MSA_DIRECTORY_DEPTH = 2


###########################################################################
# Sequence + monomer cropping
###########################################################################


@functools.cache
def _ccd_one_letter_codes() -> dict[str, str]:
    """Return CCD-aware FASTA codes, including modified-residue parents."""
    chem_comp = get_ccd()["chem_comp"]
    codes = [str(value) for value in chem_comp["id"].as_array()]
    one_letter = [str(value) for value in chem_comp["one_letter_code"].as_array()]
    parents = [str(value) for value in chem_comp["mon_nstd_parent_comp_id"].as_array()]
    direct = {
        code: one
        for code, one in zip(codes, one_letter, strict=True)
        if len(one) == 1 and one != "?"
    }
    parent_by_code = dict(zip(codes, parents, strict=True))
    mapping = dict(direct)
    for code in codes:
        current = code
        visited = set()
        while current not in mapping and current not in visited:
            visited.add(current)
            parent = parent_by_code.get(current, "?")
            # Multiple comma-separated parents do not define one FASTA residue.
            if parent == "?" or "," in parent:
                break
            current = parent
        if current in mapping:
            mapping[code] = mapping[current]
    return mapping


def residue_sequence(atom_array) -> str:
    """Return the one-letter sequence of ``atom_array``, one code per residue.

    Modified residues resolve through the CCD's one-letter code or canonical
    parent component (MSE->M, SEP->S, PTR->Y), so the predictor folds a real
    amino acid instead of an unknown. Truly noncanonical residues stay ``X``.
    """
    ccd_mapping = _ccd_one_letter_codes()
    starts = get_residue_starts(atom_array)
    return "".join(
        ccd_mapping.get(str(name), RES_NAME_TO_ONE_LETTER.get(str(name), "X"))
        for name in atom_array.res_name[starts]
    )


def standard_residue_sequence(atom_array) -> str:
    """Return the sequence with every non-standard residue as ``X``.

    This is the encoding the shared MSA stores are keyed on, so it is the
    fallback lookup key when the CCD-canonical sequence misses.
    """
    starts = get_residue_starts(atom_array)
    return "".join(
        RES_NAME_TO_ONE_LETTER.get(str(name), "X")
        for name in atom_array.res_name[starts]
    )


def protein_monomer(atom_array):
    """Return the input's only protein chain."""
    if "is_protein" in atom_array.get_annotation_categories():
        mask = np.asarray(atom_array.is_protein, dtype=bool)
    else:
        mask = filter_amino_acids(atom_array)
    protein = atom_array[mask]
    if len(protein) == 0:
        raise ValueError("A monomer predictor requires at least one protein residue")

    chain_category = (
        "chain_iid"
        if "chain_iid" in protein.get_annotation_categories()
        else "chain_id"
    )
    chains = np.unique(protein.get_annotation(chain_category)).tolist()
    if len(chains) != 1:
        raise ValueError(
            "This is a monomer predictor; prepare exactly one protein "
            f"chain/pn-unit per handler row, found {chains}"
        )
    return protein


###########################################################################
# Monomer manifest staging
###########################################################################


def stage_monomer_manifest(
    loaded,
    *,
    manifest_dir: Path | str,
    extra: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Write the predictor sequence/provenance manifest for one handler row.

    Returns the manifest dict, whose ``sequence`` is what the predictor should
    fold. Pass ``write=False`` when the caller still has fields to add and will
    write the manifest itself once it is complete — the manifest doubles as the
    "this target is fully staged" marker, so a partial one must never land.
    """
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    protein = protein_monomer(loaded.atom_array)
    target = loaded.basename

    manifest = {
        "target": target,
        "source_path": loaded.path,
        "sequence": residue_sequence(protein),
        "msa_lookup_sequence": standard_residue_sequence(protein),
        "dropped_nonprotein_atoms": len(loaded.atom_array) - len(protein),
        **(extra or {}),
    }
    if write:
        write_monomer_manifest(manifest_dir, manifest)
    return manifest


def write_monomer_manifest(manifest_dir: Path | str, manifest: dict[str, Any]) -> None:
    """(Re)write one manifest, e.g. after a block adds its own fields to it."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{manifest['target']}.json").write_text(
        json.dumps(manifest, indent=2)
    )


def load_monomer_references(manifest_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load every manifest written by :func:`stage_monomer_manifest`, by target."""
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted(Path(manifest_dir).glob("*.json"))
    }


###########################################################################
# On-disk protein MSAs
###########################################################################


def normalize_msa_dirs(msa_dirs: Iterable[str | dict]) -> list[dict]:
    """Normalize MSA roots to AtomWorks' ``{dir, extension, directory_depth}``.

    Plain strings are accepted and take the shared-cluster defaults
    (``.a3m.gz`` files sharded two directories deep).
    """
    normalized = []
    for entry in msa_dirs:
        if isinstance(entry, str):
            entry = {"dir": entry}
        if not isinstance(entry, dict) or "dir" not in entry:
            raise TypeError(
                "each msa_dirs entry must be a path or a dict with a 'dir' key, "
                f"got {entry!r}"
            )
        directory = str(entry["dir"])
        if not Path(directory).is_absolute():
            raise ValueError(f"msa_dirs paths must be absolute: {directory}")
        normalized.append(
            {
                "dir": directory,
                "extension": str(entry.get("extension", DEFAULT_MSA_EXTENSION)),
                "directory_depth": int(
                    entry.get("directory_depth", DEFAULT_MSA_DIRECTORY_DEPTH)
                ),
            }
        )
    if not normalized:
        raise ValueError("msa_dirs must contain at least one MSA root")
    return normalized


def default_msa_dirs() -> list[dict] | None:
    """Return the site's configured ``PROTEIN_MSA_DIRS``, if any."""
    configured = get_setting_field("PROTEIN_MSA_DIRS", None)
    return normalize_msa_dirs(configured) if configured else None


def resolve_msa_path(
    sequence: str,
    msa_dirs: Iterable[str | dict],
    *,
    alternates: Iterable[str] = (),
) -> Path | None:
    """Return the sharded on-disk MSA for ``sequence``, or None if absent.

    Roots are searched in order and keyed by AtomWorks' sequence hash, so the
    same MSA store RF3 validation reads is reused verbatim. ``alternates`` are
    tried afterwards: the stores were built from AtomWorks' ``X``-for-modified
    encoding, so a chain containing MSE/SEP/PTR hashes differently from the
    CCD-canonical sequence it is folded as.
    """
    from atomworks.ml.transforms.msa._msa_loading_utils import get_msa_path

    roots = normalize_msa_dirs(msa_dirs)
    for key in (sequence, *alternates):
        if not key:
            continue
        path = get_msa_path(key, roots)
        if path is not None:
            return path
    return None


def read_query_a3m(
    msa_path: Path | str,
    *,
    sequence: str,
    max_sequences: int | None = None,
) -> tuple[list[str], list[str]]:
    """Read a (optionally gzipped) a3m into headers and one-line records."""
    path = Path(msa_path)
    opener = gzip.open if path.name.endswith(".gz") else open
    headers: list[str] = []
    records: list[str] = []
    with opener(path, "rt") as stream:
        for line in stream:
            line = line.strip()
            # ColabFold reads the first FASTA record of the file as the query,
            # so store-level '#' comment headers must not survive staging.
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if max_sequences is not None and len(headers) >= max_sequences:
                    break
                headers.append(line)
                records.append("")
            elif records:
                records[-1] += line
    if not records:
        raise ValueError(f"MSA {path} contains no sequences")
    query = records[0].upper()
    if query != sequence.upper() and not (
        len(query) == len(sequence)
        and all(
            msa_aa == folded_aa or msa_aa == "X"
            for msa_aa, folded_aa in zip(query, sequence.upper(), strict=True)
        )
    ):
        raise ValueError(
            f"MSA {path} query sequence does not match the folded sequence: "
            f"msa={query!r}, folded={sequence!r}"
        )
    records[0] = sequence.upper()
    return headers, records


def stage_query_a3m(
    msa_path: Path | str,
    output_path: Path | str,
    *,
    sequence: str,
    max_sequences: int | None = None,
) -> int:
    """Write a plain a3m whose first record is ``sequence``; return its depth.

    The stored query record is verified against the sequence the predictor is
    being folded on, so a hash collision or a stale MSA cannot be scored as if
    it matched. Positions where the store holds ``X`` (its encoding for a
    modified residue) are accepted against the CCD-canonical residue and
    rewritten to it — same chain, same alignment columns, and the predictor
    then folds a real amino acid rather than an unknown.
    """
    headers, records = read_query_a3m(
        msa_path,
        sequence=sequence,
        max_sequences=max_sequences,
    )
    Path(output_path).write_text(
        "".join(f"{header}\n{record}\n" for header, record in zip(headers, records))
    )
    return len(records)
