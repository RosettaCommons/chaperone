"""Helper functions for prediction oracle input/output conversion.

Each prediction block (AF3, Boltz2, Chai1) needs to convert atomworks
atom arrays into its native input format (JSON, YAML, FASTA). These
helpers are factored out to keep the block files focused on orchestration.
"""

import functools
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import random
import string

import numpy as np
import yaml
from atomworks.constants import DICT_THREE_TO_ONE
from atomworks.enums import ChainType
from atomworks.io.utils.io_utils import to_cif_string
from biotite.structure import get_residue_starts

from blocks.utils.oracles import is_ccd_name

logger = logging.getLogger(__name__)


def to_af3_template_cif_string(chain_atom_array, name: str) -> str:
    """Serialize a protein template with AF3's entity and release metadata."""
    from atomworks.io.utils.io_utils import CIFWriteConfig

    from blocks.constants import AF3_RELEASE_LINES
    from chaperone.io import ensure_bonds

    template = ensure_bonds(chain_atom_array).copy()
    categories = template.get_annotation_categories()
    if "occupancy" not in categories:
        template.set_annotation("occupancy", np.ones(len(template), dtype=float))
    if "b_factor" not in categories:
        template.set_annotation("b_factor", np.zeros(len(template), dtype=float))
    return to_cif_string(
        template,
        config=CIFWriteConfig(
            id=name,
            include_entity_categories=True,
            include_chem_comp=False,
            include_nan_coords=False,
        ),
    ) + "".join(AF3_RELEASE_LINES)


def _extract_protein_sequence_map(
    atom_array, protein_sequences: dict[str, str] | None = None
) -> dict[str, str]:
    """Map ``chain_id -> protein sequence`` for every protein chain in ``atom_array``."""
    if protein_sequences is not None:
        return {str(chain): str(seq) for chain, seq in protein_sequences.items()}
    if atom_array is None:
        return {}
    from biotite.structure import filter_amino_acids, to_sequence
    from biotite.structure.chains import get_chain_starts

    protein_atoms = atom_array[filter_amino_acids(atom_array)]
    chain_starts = get_chain_starts(protein_atoms)
    chain_ids = [str(protein_atoms[cs].chain_id) for cs in chain_starts]
    sequences = to_sequence(protein_atoms)[0]
    return dict(zip(chain_ids, (str(s) for s in sequences)))


###########################################################################
# Shared builder cores (used by every create_*_from_design oracle builder)
#
# These exist so each oracle's input builder is a thin shape-mapper that
# emits its native format. Centralizing the polymer/ligand iteration and
# ligand resolution guarantees the same precedence, chain-id allocation,
# and unknown-residue handling across all oracles.
###########################################################################


def _iter_polymer_chains(atom_array, protein_sequences: dict[str, str] | None = None):
    """Iterate polymer chains (protein, DNA, RNA) of ``atom_array``.

    Yields ``(chain_id, chain_type, sequence)`` for each protein
    (``POLYPEPTIDE_L``), DNA, or RNA chain, in ``np.unique`` chain-id order.
    Non-polymer (ligand/ion) chains are skipped — handle those via
    :func:`_iter_ligand_residues`.

    - Protein sequences use ``DICT_THREE_TO_ONE.get(aa, "X")`` so unknown
      residues become ``X`` rather than being silently dropped.
    - DNA sequences use ``DNA_TWO_TO_ONE`` (residues not in the map are skipped,
      matching biotite/AF3 conventions).
    - RNA sequences are the raw residue names joined (already single-letter).

    The ``chain_id`` is yielded as returned by ``np.unique`` (caller decides
    whether to ``str()`` it for emission / chain-tracking).
    """
    from blocks.constants import DNA_TWO_TO_ONE

    overrides = {
        str(chain): str(seq) for chain, seq in (protein_sequences or {}).items()
    }
    # Every polymer chain the structure already provides, of any type - not just
    # the protein ones. A sequence row for a chain the structure supplies as RNA
    # or DNA describes THAT chain; treating it as a missing protein chain emits a
    # second entity with the same id, and AF3 rejects the whole input.
    seen_chains = set()
    chains = np.unique(atom_array.chain_id) if atom_array is not None else []
    for chain in chains:
        chain_atoms = atom_array[atom_array.chain_id == chain]
        chain_res_starts = get_residue_starts(chain_atoms)
        chain_type = chain_atoms.chain_type[0]

        if chain_type == ChainType.POLYPEPTIDE_L:
            seq = overrides.get(str(chain))
            if seq is None:
                seq = "".join(
                    DICT_THREE_TO_ONE.get(aa, "X")
                    for aa in chain_atoms[chain_res_starts].res_name
                )
        elif chain_type == ChainType.DNA:
            seq = "".join(
                DNA_TWO_TO_ONE[res]
                for res in chain_atoms[chain_res_starts].res_name
                if res in DNA_TWO_TO_ONE
            )
        elif chain_type == ChainType.RNA:
            seq = "".join(chain_atoms[chain_res_starts].res_name)
        else:
            continue

        seen_chains.add(str(chain))
        yield chain, chain_type, seq

    # A context structure is optional and may omit a newly introduced protein
    # chain. Sequence rows remain authoritative, so append such chains here.
    for chain, sequence in overrides.items():
        if chain not in seen_chains:
            yield chain, ChainType.POLYPEPTIDE_L, sequence


def _iter_ligand_residues(atom_array, remove_ligands=False):
    """Iterate ligand residues of ``atom_array``.

    Yields ``(res_name, ligand_atom_array, source_chain)`` for each residue
    whose ``res_name`` is not in ``NON_LIGAND_NAMES``. ``source_chain`` is the
    original chain id of the residue (used to optionally preserve the chain id
    on emission). Yields nothing when ``remove_ligands`` is ``True``.
    """
    from blocks.constants import NON_LIGAND_NAMES

    if remove_ligands or atom_array is None:
        return

    starts = get_residue_starts(atom_array)
    residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    names = atom_array.res_name[starts].tolist()

    for start, end, res_name in zip(residue_starts[:-1], residue_starts[1:], names):
        if res_name in NON_LIGAND_NAMES:
            continue
        yield res_name, atom_array[start:end], atom_array.chain_id[start]


def _resolve_ligand(
    res_name,
    atoms,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
):
    """Resolve a ligand residue to a canonical ``(kind, value)`` descriptor.

    ``kind`` is either ``"ccd"`` or ``"smiles"``; ``value`` is the CCD code or
    SMILES string. Precedence (canonical, CCD-first — matches the AF3 builder):

    1. ``specified_ccd[res_name]`` → ``("ccd", code)``
    2. ``specified_smiles[res_name]`` → ``("smiles", smiles)``
    3. CCD-name auto-detection (``is_ccd_name`` and not ``disable_ccd_detection``)
       → ``("ccd", res_name)``
    4. RDKit geometry → SMILES, falling back to CCD-code → SMILES.

    Oracles that cannot emit a raw CCD code (e.g. Chai-1/ESMFold2 FASTA, which
    are SMILES-only) should pass the result through :func:`_ccd_to_smiles` /
    :func:`_resolve_ligand_smiles`.
    """
    specified_smiles = specified_smiles or {}
    specified_ccd = specified_ccd or {}

    if res_name in specified_ccd:
        return "ccd", specified_ccd[res_name]
    if res_name in specified_smiles:
        return "smiles", specified_smiles[res_name]
    if is_ccd_name(res_name) and not disable_ccd_detection:
        return "ccd", res_name

    import rdkit.Chem as Chem
    from atomworks.io.tools.rdkit import atom_array_to_rdkit, ccd_code_to_rdkit

    try:
        mol = atom_array_to_rdkit(atoms)
        return "smiles", Chem.MolToSmiles(mol)
    except Exception as e:
        logger.warning(f"Error converting {res_name} to SMILES: {e}")
        try:
            mol = ccd_code_to_rdkit(res_name)
            return "smiles", Chem.MolToSmiles(mol)
        except Exception as ccd_error:
            raise RuntimeError(
                f"Failed to get SMILES for ligand {res_name}. "
                f"RDKit: {e}. CCD: {ccd_error}"
            )


def _ccd_to_smiles(ccd_code):
    """Convert a CCD code to a SMILES string, or ``None`` on failure (logged)."""
    import rdkit.Chem as Chem
    from atomworks.io.tools.rdkit import ccd_code_to_rdkit

    try:
        mol = ccd_code_to_rdkit(ccd_code)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.warning(f"Failed to convert CCD {ccd_code} to SMILES: {e}")
        return None


def _resolve_ligand_smiles(
    res_name,
    atoms,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
):
    """Resolve a ligand to a SMILES string for SMILES-only oracles (Chai-1, ESMFold2).

    Uses the same canonical CCD-first precedence as :func:`_resolve_ligand`,
    converting any CCD descriptor to SMILES. Returns ``None`` if no SMILES could
    be produced (mirrors the historical FASTA-builder behaviour of skipping the
    ligand on conversion failure).
    """
    kind, value = _resolve_ligand(
        res_name,
        atoms,
        specified_smiles=specified_smiles,
        specified_ccd=specified_ccd,
        disable_ccd_detection=disable_ccd_detection,
    )
    if kind == "smiles":
        return value
    return _ccd_to_smiles(value)


def _next_ligand_chain_id(added_chains, source_chain=None):
    """Pick a chain id for a ligand, preserving ``source_chain`` when free.

    If ``source_chain`` is supplied and not already used, it is returned
    unchanged (preserves the design's original ligand chain id). Otherwise the
    next unused single-letter id (A–Z) is returned, then double letters
    (AA, AB, …) once the 26 single letters are exhausted — avoiding the
    ``ascii_uppercase[...][0]`` IndexError past 26 chains.
    """
    used = set(str(c) for c in added_chains)
    if source_chain is not None and str(source_chain) not in used:
        return source_chain

    for c in string.ascii_uppercase:
        if c not in used:
            return c
    # Overflow past 26 chains: AA, AB, ... ZZ, then AAA, ...
    for length in range(2, 5):
        for combo in itertools.product(string.ascii_uppercase, repeat=length):
            cid = "".join(combo)
            if cid not in used:
                return cid
    raise ValueError("Exhausted available ligand chain ids.")


def create_colabfold_fasta_from_design(
    atom_array, name, protein_sequences: dict[str, str] | None = None
) -> str:
    """Create a ColabFold (monomer) FASTA string from an atomworks atom array.

    ColabFold's FASTA is minimal: one header line and one sequence line::

        >design_0
        MSEQ...

    Monomer-only: only the protein is folded; ligands and nucleic acids are
    ignored (AlphaFold2 cannot model them). If the design has more than one
    protein chain, the longest is folded and a warning is logged — use a
    multimer-aware oracle (AF3/Boltz2/Chai1) for complexes.
    """
    sequence_map = _extract_protein_sequence_map(atom_array, protein_sequences)
    if not sequence_map:
        raise ValueError(f"No protein chain found in design {name!r} for ColabFold.")
    if len(sequence_map) > 1:
        chain_id, sequence = max(sequence_map.items(), key=lambda kv: len(kv[1]))
        logger.warning(
            "ColabFold is monomer-only: design %r has %d protein chains %s; "
            "folding only the longest (chain %s).",
            name,
            len(sequence_map),
            list(sequence_map),
            chain_id,
        )
    else:
        sequence = next(iter(sequence_map.values()))
    return f">{name}\n{sequence}\n"


def parse_colabfold_fasta_sequence(fasta: str) -> str:
    """Read back the single sequence written by ``create_colabfold_fasta_from_design``.

    Callers that need the sequence itself (to hash it against an MSA store, or
    to score a prediction against it) must not re-derive it independently, or
    the sequence folded and the sequence reasoned about can drift apart.
    """
    records = [line.strip() for line in fasta.splitlines() if not line.startswith(">")]
    sequence = "".join(record for record in records if record)
    if not sequence:
        raise ValueError(f"ColabFold FASTA contains no sequence: {fasta!r}")
    return sequence


###########################################################################
# AF3 JSON helpers
###########################################################################


def fill_af3_json_from_template(
    template_json,
    atom_array,
    name,
    diffused_index_map,
    add_ptm_at_mapped_residues: dict | None = None,
    add_ptm_at_residues: list[dict] | None = None,
    protein_sequences: dict[str, str] | None = None,
):
    """Fill an AF3 JSON template with sequence and optional modifications.

    Reads a template JSON, inserts protein sequences from the upstream design,
    and optionally adds post-translational modifications (PTMs).

    Args:
        template_json: Path to the AF3 template JSON (must contain ``userCCD``
            if using custom ligand modifications).
        atom_array: Biotite AtomArray from the upstream design.
        name: Design name for the JSON ``name`` field.
        diffused_index_map: Mapping from original residue keys (e.g. ``"A10"``)
            to diffused residue identifiers (e.g. ``"A42"``). Required when
            using ``add_ptm_at_mapped_residues``.
        add_ptm_at_mapped_residues: Map of ``{ptmType: original_residue_key}``.
            Looks up the residue key in ``diffused_index_map`` to resolve the
            final 1-indexed position. Used for enzyme design where a substrate
            is covalently attached at a catalytic residue from the theozyme
            motif (e.g. ``{"L:1": "A10"}``).
        add_ptm_at_residues: Direct list of AF3 modification dicts, each with
            ``ptmType`` and ``ptmPosition`` (1-indexed). Positions are used
            as-is without index map lookup (e.g.
            ``[{"ptmType": "CCD_HY3", "ptmPosition": 42}]``).
    """
    json_data = json.load(open(template_json))
    json_data["name"] = name

    sequence_dict = _extract_protein_sequence_map(atom_array, protein_sequences)

    # Add PTMs via diffused index map lookup
    if add_ptm_at_mapped_residues:
        if isinstance(diffused_index_map, str):
            diffused_index_map = json.loads(diffused_index_map)
        json_data["sequences"][0]["protein"]["modifications"].extend(
            [
                {"ptmType": k, "ptmPosition": int(diffused_index_map[v][1:])}
                for k, v in add_ptm_at_mapped_residues.items()
            ]
        )

    # Add PTMs with directly specified positions
    if add_ptm_at_residues:
        json_data["sequences"][0]["protein"]["modifications"].extend(
            add_ptm_at_residues
        )

    # For every sequence in the template, insert the actual input sequence from
    # the upstream designs (skip non-protein entries like ligands)
    for s in json_data["sequences"]:
        if "protein" not in s:
            continue
        chain_id = s["protein"]["id"]
        if isinstance(chain_id, list):
            chain_id = chain_id[0]
        s["protein"]["sequence"] = str(sequence_dict[chain_id])
    return json_data


def create_af3_json_from_design(
    atom_array,
    name,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    only_monomer=False,
    template_resi: None | bool | str = False,
    invert_template_resi=False,
    protein_sequences: dict[str, str] | None = None,
):
    """
    Creates a JSON input file for AF3 based on:
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

    File: Path to pdb or cif file
    remove_ligands: If True, skip ligands
    specified_smiles: A dictionary specified in the input config file by user. Would skip automatic ccd fetching of ligands and match the ligand names to the specified smiles instead.
    If None, will automatically fetch ccd codes for the ligands.
    only_monomer: If True, only include the first protein chain in input pdb.
    template_resi: A string of template indices for each chain. e.g. 'A1,A2,A3,B4,B5,B6'
    invert_template_resi: If True, invert the template indices, i.g. residues other than template_resi will be templated
    """
    # empty JSON start
    json_data = {
        "name": name,
        "sequences": [],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }

    if template_resi and len(template_resi):
        motif_list = template_resi.split(",")
        template_resi = {
            chain: [] for chain in np.unique([str(resi[0]) for resi in motif_list])
        }
        [template_resi[str(resi[0])].append(int(resi[1:])) for resi in motif_list]

    added_chains = []

    # ... Protein / DNA / RNA sequences
    for chain, chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        if chain_type == ChainType.DNA:
            json_data["sequences"].append(dict(dna={"id": chain, "sequence": seq}))
        elif chain_type == ChainType.POLYPEPTIDE_L:
            templates = []
            if template_resi and len(template_resi):
                if atom_array is None:
                    raise ValueError("AF3 residue templates require structure context.")
                chain_atom_array = atom_array[atom_array.chain_id == chain]
                if invert_template_resi:
                    template_resi[chain] = [
                        i
                        for i in range(len(seq))
                        if i not in template_resi.get(chain, [])
                    ]
                templates.append(
                    {
                        "mmcif": to_af3_template_cif_string(
                            chain_atom_array, name=f"{name}_chain_{chain}"
                        ),
                        "queryIndices": template_resi.get(chain, []),
                        "templateIndices": template_resi.get(chain, []),
                    }
                )
            json_data["sequences"].append(
                dict(
                    protein={
                        "id": chain,
                        "sequence": seq,
                        "unpairedMsa": "",  # "",  # or empty string
                        "unpairedMsaPath": None,
                        "pairedMsa": "",
                        "pairedMsaPath": None,
                        "templates": templates,
                    }
                )
            )
        elif chain_type == ChainType.RNA:
            # AF3 validates that every RNA chain carries an unpaired MSA field
            # before featurisation ("RNA chain N is missing unpaired MSA"), and
            # these blocks always run with the data pipeline skipped. Emit the
            # empty single-sequence MSA explicitly, as the protein branch does.
            # DNA needs no equivalent — AF3 builds no DNA MSAs.
            json_data["sequences"].append(
                dict(
                    rna={
                        "id": chain,
                        "sequence": seq,
                        "unpairedMsa": "",
                        "unpairedMsaPath": None,
                    }
                )
            )
        else:  # pragma: no cover - _iter_polymer_chains never yields others
            continue

        added_chains.append(chain)
        if only_monomer:
            break

    # ... Ligand SMILES / CCD
    added_ligands = []

    for res_name, ligand, source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        if only_monomer and res_name in added_ligands:
            # NOTE: assuming that only one ligand with the same name per monomer
            logger.info(f"Skipping {res_name} because it is already added")
            continue
        # Preserve source chain ID when available; fall back to next unused id
        chain = _next_ligand_chain_id(added_chains, source_chain=source_chain)
        logger.info(f"Adding ligand {res_name} to chain {chain}")
        kind, value = _resolve_ligand(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )
        lig_dict = {"ccdCodes": [value]} if kind == "ccd" else {"smiles": value}

        json_data["sequences"].append(
            dict(
                ligand={
                    "id": chain,
                }
                | lig_dict
            )
        )
        added_ligands.append(res_name)
        added_chains.append(chain)
    return json_data


###########################################################################
# Boltz2 YAML helpers
###########################################################################


def create_boltz_yaml_from_design(
    atom_array,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    predict_affinity=False,
    protein_sequences: dict[str, str] | None = None,
):
    """Create a Boltz-2 input YAML dict from an atomworks atom array.

    Converts protein chains, DNA, RNA, and ligands into the Boltz YAML
    input format. Proteins use single-sequence mode (msa: empty) by default;
    use --use_msa_server to auto-generate MSAs.
    """
    yaml_data = {
        "version": 1,
        "sequences": [],
    }

    added_chains = []
    ligand_chain_id = None

    # Add polymer chains (protein, DNA, RNA)
    for chain, chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        if chain_type == ChainType.POLYPEPTIDE_L:
            yaml_data["sequences"].append(
                {
                    "protein": {
                        "id": str(chain),
                        "sequence": seq,
                        "msa": "empty",
                    }
                }
            )
        elif chain_type == ChainType.DNA:
            yaml_data["sequences"].append({"dna": {"id": str(chain), "sequence": seq}})
        elif chain_type == ChainType.RNA:
            yaml_data["sequences"].append({"rna": {"id": str(chain), "sequence": seq}})
        else:  # pragma: no cover
            continue

        added_chains.append(chain)

    # Add ligands
    added_ligands = []
    for res_name, ligand, source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        chain_id = _next_ligand_chain_id(added_chains, source_chain=source_chain)
        logger.info(f"Adding ligand {res_name} to chain {chain_id}")

        lig_entry = {"id": str(chain_id)}
        kind, value = _resolve_ligand(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )
        lig_entry["ccd" if kind == "ccd" else "smiles"] = value

        yaml_data["sequences"].append({"ligand": lig_entry})
        added_chains.append(chain_id)
        added_ligands.append(res_name)

        # Track first ligand for affinity prediction
        if ligand_chain_id is None:
            ligand_chain_id = chain_id

    # Add affinity property if requested
    if predict_affinity and ligand_chain_id is not None:
        yaml_data["properties"] = [{"affinity": {"binder": str(ligand_chain_id)}}]

    return yaml_data


def fill_boltz_yaml_from_template(
    template_yaml, atom_array, protein_sequences: dict[str, str] | None = None
):
    """Fill sequences from a design into a template Boltz YAML.

    Replaces protein sequences in the template while keeping all other
    configuration (constraints, templates, properties) intact.
    """
    with open(template_yaml) as f:
        yaml_data = yaml.safe_load(f)

    sequence_dict = _extract_protein_sequence_map(atom_array, protein_sequences)

    # Fill protein sequences in template
    for entry in yaml_data.get("sequences", []):
        if "protein" not in entry:
            continue
        chain_id = entry["protein"]["id"]
        if isinstance(chain_id, list):
            chain_id = chain_id[0]
        if str(chain_id) in sequence_dict:
            entry["protein"]["sequence"] = str(sequence_dict[str(chain_id)])

    return yaml_data


###########################################################################
# Chai1 FASTA helpers
###########################################################################


def create_chai_fasta_from_design(
    atom_array,
    name,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    protein_sequences: dict[str, str] | None = None,
):
    """Create a Chai-1 FASTA string from an atomworks atom array.

    Chai-1 FASTA format uses headers like:
        >protein|name=chain_A
        MSEQ...
        >ligand|name=lig_1
        CCO
        >dna|name=dna_A
        AGTC
        >rna|name=rna_A
        AGUC
    """
    lines = []

    # Add polymer chains (protein, DNA, RNA)
    for chain, chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        if chain_type == ChainType.POLYPEPTIDE_L:
            lines.append(f">protein|name={name}_chain_{chain}")
        elif chain_type == ChainType.DNA:
            lines.append(f">dna|name={name}_dna_{chain}")
        elif chain_type == ChainType.RNA:
            # Chai-1 expects single-letter RNA: A, U, G, C
            lines.append(f">rna|name={name}_rna_{chain}")
        else:  # pragma: no cover
            continue
        lines.append(seq)

    # Add ligands as SMILES (Chai-1 FASTA is SMILES-only)
    added_ligands = []
    for res_name, ligand, _source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        smiles = _resolve_ligand_smiles(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )

        if smiles:
            lig_id = f"{name}_lig_{res_name}"
            if lig_id in added_ligands:
                lig_id = f"{lig_id}_{len(added_ligands)}"
            lines.append(f">ligand|name={lig_id}")
            lines.append(smiles)
            added_ligands.append(lig_id)

    return "\n".join(lines) + "\n"


###########################################################################
# ESMFold2 spec helpers
###########################################################################


def create_esmfold2_spec_from_design(
    atom_array,
    name,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    protein_sequences: dict[str, str] | None = None,
):
    """Create an ESMFold2 input spec dict from an atomworks atom array.

    ESMFold2's :class:`StructurePredictionInput` is built from
    ``ProteinInput(id, sequence)`` and ``LigandInput(id, smiles)`` entries. This
    helper extracts protein chains (single-sequence, no MSA) and ligand SMILES so
    the inner runner can reconstruct the input without touching biotite/atomworks.

    Ligand detection mirrors :func:`create_chai_fasta_from_design` (specified
    SMILES/CCD override, CCD-code auto-detection, then RDKit geometry → SMILES).
    Non-protein polymers (DNA/RNA) are out of scope for ESMFold2 here and are
    skipped with a warning.

    Returns:
        dict: ``{"name": name, "chains": [{"id", "sequence"}],
        "ligands": [{"id", "smiles"}]}``.
    """
    spec = {"name": name, "chains": [], "ligands": []}
    added_chains = []

    # Protein chains (ESMFold2 ProteinInput, single-sequence)
    for chain, chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        if chain_type == ChainType.POLYPEPTIDE_L:
            spec["chains"].append({"id": str(chain), "sequence": seq})
            added_chains.append(str(chain))
        else:  # DNA / RNA — out of scope for ESMFold2
            logger.warning(
                "ESMFold2 spec for %r: skipping non-protein polymer chain %s "
                "(nucleic acids are out of scope for this block).",
                name,
                chain,
            )

    if not spec["chains"]:
        raise ValueError(f"No protein chain found in design {name!r} for ESMFold2.")

    # Ligands as SMILES (ESMFold2 LigandInput)
    for res_name, ligand, source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        smiles = _resolve_ligand_smiles(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )

        if smiles:
            chain_id = _next_ligand_chain_id(added_chains, source_chain=source_chain)
            spec["ligands"].append({"id": str(chain_id), "smiles": smiles})
            added_chains.append(str(chain_id))

    return spec


###########################################################################
# Protenix JSON helpers
###########################################################################


# Common monatomic ions recognized by Protenix v2's "ion" input type.
# These use CCD codes WITHOUT the "CCD_" prefix.
PROTENIX_ION_CODES = frozenset(
    {
        "MG",
        "CA",
        "ZN",
        "FE",
        "FE2",
        "MN",
        "CO",
        "NI",
        "CU",
        "CU1",
        "K",
        "NA",
        "CL",
    }
)


def create_protenix_json_from_design(
    atom_array,
    name,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    protein_sequences: dict[str, str] | None = None,
):
    """Create a Protenix input JSON list from an atomworks atom array.

    Protenix input format is similar to AF3 but with key differences:
    - Top level is a JSON array ``[{...}]``
    - Protein uses ``proteinChain``, DNA uses ``dnaSequence``, RNA uses ``rnaSequence``
    - Chain IDs are lists: ``"id": ["A"]``
    - Each entity has a ``"count": 1`` field
    - CCD ligands use ``"ligand": "CCD_ATP"`` format (with ``CCD_`` prefix)
    - SMILES ligands use ``"ligand": "CCO"`` (raw SMILES string)

    See https://github.com/bytedance/Protenix for input format docs.
    """
    job = {
        "name": name,
        "sequences": [],
    }

    added_chains = []

    for chain, chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        if chain_type == ChainType.POLYPEPTIDE_L:
            job["sequences"].append(
                {
                    "proteinChain": {
                        "sequence": seq,
                        "count": 1,
                        "id": [str(chain)],
                    }
                }
            )
        elif chain_type == ChainType.DNA:
            job["sequences"].append(
                {
                    "dnaSequence": {
                        "sequence": seq,
                        "count": 1,
                        "id": [str(chain)],
                    }
                }
            )
        elif chain_type == ChainType.RNA:
            job["sequences"].append(
                {
                    "rnaSequence": {
                        "sequence": seq,
                        "count": 1,
                        "id": [str(chain)],
                    }
                }
            )
        else:  # pragma: no cover
            continue

        added_chains.append(chain)

    added_ligands = []
    for res_name, ligand, source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        chain_id = _next_ligand_chain_id(added_chains, source_chain=source_chain)

        # Ions get their own entity type in Protenix v2
        if res_name in PROTENIX_ION_CODES:
            logger.info(f"Adding ion {res_name} to chain {chain_id}")
            job["sequences"].append(
                {
                    "ion": {
                        "ion": res_name,
                        "count": 1,
                    }
                }
            )
            added_chains.append(chain_id)
            added_ligands.append(res_name)
            continue

        logger.info(f"Adding ligand {res_name} to chain {chain_id}")

        kind, value = _resolve_ligand(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )
        lig_value = f"CCD_{value}" if kind == "ccd" else value

        job["sequences"].append(
            {
                "ligand": {
                    "ligand": lig_value,
                    "count": 1,
                    "id": [str(chain_id)],
                }
            }
        )
        added_chains.append(chain_id)
        added_ligands.append(res_name)

    return [job]


def fill_protenix_json_from_template(
    template_json,
    atom_array,
    name,
    protein_sequences: dict[str, str] | None = None,
):
    """Fill sequences from a design into a template Protenix JSON.

    Reads a template JSON, replaces protein sequences from the upstream
    design's atom array while preserving all other configuration (ligands,
    constraints, covalent bonds, etc.).
    """
    with open(template_json) as f:
        json_data = json.load(f)

    if isinstance(json_data, list):
        job = json_data[0]
    else:
        job = json_data

    job["name"] = name

    sequence_dict = _extract_protein_sequence_map(atom_array, protein_sequences)

    for entry in job.get("sequences", []):
        if "proteinChain" not in entry:
            continue
        chain_id = entry["proteinChain"]["id"]
        if isinstance(chain_id, list):
            chain_id = chain_id[0]
        if str(chain_id) in sequence_dict:
            entry["proteinChain"]["sequence"] = str(sequence_dict[str(chain_id)])

    if isinstance(json_data, list):
        return json_data
    return [job]


###########################################################################
# OpenFold3 JSON helpers
###########################################################################


def create_openfold3_json_from_design(
    atom_array,
    name,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    protein_sequences: dict[str, str] | None = None,
):
    """Create an OpenFold3 query dict from an atomworks atom array.

    Returns a dict suitable for inclusion in an InferenceQuerySet's ``queries``
    field. The top-level structure is::

        {name: {"chains": [...]}}

    Each chain has ``molecule_type``, ``chain_ids``, and either ``sequence``
    (protein/DNA/RNA) or ``smiles``/``ccd_codes`` (ligand).
    """
    chains = []
    added_chains = []

    # Add polymer chains (protein, DNA, RNA)
    for chain, chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        if chain_type == ChainType.POLYPEPTIDE_L:
            mol_type = "PROTEIN"
        elif chain_type == ChainType.DNA:
            mol_type = "DNA"
        elif chain_type == ChainType.RNA:
            mol_type = "RNA"
        else:  # pragma: no cover
            continue
        chains.append(
            {
                "molecule_type": mol_type,
                "chain_ids": [str(chain)],
                "sequence": seq,
            }
        )
        added_chains.append(chain)

    # Add ligands
    for res_name, ligand, source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        chain_id = _next_ligand_chain_id(added_chains, source_chain=source_chain)
        logger.info(f"Adding ligand {res_name} to chain {chain_id}")

        lig_entry = {
            "molecule_type": "LIGAND",
            "chain_ids": [str(chain_id)],
        }
        kind, value = _resolve_ligand(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )
        if kind == "ccd":
            lig_entry["ccd_codes"] = [value]
        else:
            lig_entry["smiles"] = value

        chains.append(lig_entry)
        added_chains.append(chain_id)

    return {name: {"chains": chains}}


def fill_openfold3_json_from_template(
    template_json,
    atom_array,
    name,
    protein_sequences: dict[str, str] | None = None,
):
    """Fill protein sequences from a design into an OpenFold3 template JSON.

    Replaces protein sequences in the template while keeping all other
    configuration (ligands, MSA paths, covalent bonds, etc.) intact.
    """
    with open(template_json) as f:
        json_data = json.load(f)

    sequence_dict = _extract_protein_sequence_map(atom_array, protein_sequences)

    # Find the query entry — may be nested under "queries" or at top level
    if "queries" in json_data:
        # Pick the first query entry and rename it
        old_key = next(iter(json_data["queries"]))
        query = json_data["queries"].pop(old_key)
        json_data["queries"][name] = query
    else:
        query = json_data

    # Fill protein sequences
    for chain_entry in query.get("chains", []):
        mol_type = chain_entry.get("molecule_type", "").upper()
        if mol_type != "PROTEIN":
            continue
        chain_id_list = chain_entry.get("chain_ids", [])
        if isinstance(chain_id_list, str):
            chain_id_list = [chain_id_list]
        for cid in chain_id_list:
            if cid in sequence_dict:
                chain_entry["sequence"] = str(sequence_dict[cid])
                break

    return json_data


###########################################################################
# RF3 JSON helpers
###########################################################################


def create_rf3_json_from_design(
    atom_array,
    name,
    remove_ligands=False,
    specified_smiles=None,
    specified_ccd=None,
    disable_ccd_detection=False,
    protein_sequences: dict[str, str] | None = None,
):
    """Create an RF3 component-based JSON entry from an atomworks atom array.

    RF3's ``rf3 fold`` CLI accepts a JSON list of entries, each with a
    ``name`` and ``components`` list. This function builds a single entry.

    Returns:
        dict: ``{"name": name, "components": [...]}``.
    """
    components = []

    # Add polymer chains (protein, DNA, RNA). RF3 components carry their own
    # chain id for polymers; ligand components are chain-id-less.
    for chain, _chain_type, seq in _iter_polymer_chains(atom_array, protein_sequences):
        components.append({"seq": seq, "chain_id": str(chain)})

    # Add ligands
    for res_name, ligand, _source_chain in _iter_ligand_residues(
        atom_array, remove_ligands=remove_ligands
    ):
        kind, value = _resolve_ligand(
            res_name,
            ligand,
            specified_smiles=specified_smiles,
            specified_ccd=specified_ccd,
            disable_ccd_detection=disable_ccd_detection,
        )
        components.append({"ccd_code": value} if kind == "ccd" else {"smiles": value})

    return {"name": name, "components": components}


def fill_rf3_json_from_template(
    template_json,
    atom_array,
    name,
    protein_sequences: dict[str, str] | None = None,
):
    """Fill protein sequences from a design into an RF3 JSON template.

    Reads a template JSON, replaces protein sequences (components that have
    both ``seq`` and ``chain_id`` keys) from the upstream design's atom array
    while preserving everything else (ligand components, bonds,
    template_selection, etc.).

    Returns:
        dict: The updated entry dict (caller wraps in a list for multi-design JSON).
    """
    with open(template_json) as f:
        json_data = json.load(f)

    # Handle both list format and single-entry format
    if isinstance(json_data, list):
        entry = json_data[0]
    else:
        entry = json_data

    entry["name"] = name

    sequence_dict = _extract_protein_sequence_map(atom_array, protein_sequences)

    # Replace protein sequences in template components
    for component in entry.get("components", []):
        if "seq" in component and "chain_id" in component:
            chain_id = component["chain_id"]
            if str(chain_id) in sequence_dict:
                component["seq"] = str(sequence_dict[str(chain_id)])

    return entry


def is_pdb_file(path) -> bool:
    """Check if a file is a PDB (lacks bond information)."""
    return str(path).lower().endswith(".pdb")


def pdb_assign_ligand_bonds(atom_array, specified_smiles):
    """Assign bonds to ligand atoms in PDB format using SMILES templates.

    For each ligand in specified_smiles, builds an RDKit mol from the
    ligand's 3D coordinates, infers connectivity from geometry, then
    assigns bond orders from the SMILES template.

    Args:
        atom_array: biotite AtomArray (modified in place).
        specified_smiles: dict mapping ligand residue name → SMILES string.

    Returns:
        The atom_array with bonds assigned.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    from biotite.structure import BondList, BondType

    if atom_array.bonds is None:
        atom_array.bonds = BondList(len(atom_array))

    rdkit_to_biotite = {
        Chem.BondType.SINGLE: BondType.SINGLE,
        Chem.BondType.DOUBLE: BondType.DOUBLE,
        Chem.BondType.TRIPLE: BondType.TRIPLE,
        Chem.BondType.AROMATIC: BondType.AROMATIC_DOUBLE,
    }

    unique_res_names = np.unique(atom_array.res_name).tolist()
    assigned_any = False

    for lig_name, smiles in specified_smiles.items():
        lig_mask = atom_array.res_name == lig_name
        lig_indices = np.where(lig_mask)[0]
        if len(lig_indices) == 0:
            # Skip resnames not present in THIS structure. specified_smiles is
            # often the union of resname variants across a heterogeneous input
            # set (e.g. LIG_B and T;4 for the same ligand from different
            # authors); any single structure holds only one. This mirrors the
            # leniency AF3/Protenix already have. The `assigned_any` guard below
            # still errors if none of the keys match.
            continue

        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise ValueError(
                f"create_bond_graph: RDKit cannot parse SMILES for "
                f"'{lig_name}': {smiles}"
            )
        template = Chem.RemoveHs(template)

        if len(lig_indices) != template.GetNumAtoms():
            raise ValueError(
                f"create_bond_graph: atom count mismatch for '{lig_name}': "
                f"structure has {len(lig_indices)} heavy atoms, "
                f"SMILES has {template.GetNumAtoms()} heavy atoms. "
                f"Structure elements: "
                f"{atom_array.element[lig_mask].tolist()}. "
                f"The SMILES must represent exactly the atoms present "
                f"in the structure (no more, no fewer)."
            )

        lig_elements = atom_array.element[lig_mask]
        raw_mol = Chem.RWMol()
        conf = Chem.Conformer(len(lig_indices))
        for i, idx in enumerate(lig_indices):
            raw_mol.AddAtom(Chem.Atom(lig_elements[i]))
            x, y, z = atom_array.coord[idx]
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        raw_mol.AddConformer(conf)

        try:
            rdDetermineBonds.DetermineConnectivity(raw_mol)
            mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
        except Exception as e:
            logger.warning(
                f"create_bond_graph: bond order assignment failed for "
                f"'{lig_name}' ({e}), using connectivity-only bonds"
            )
            mol = raw_mol

        for bond in mol.GetBonds():
            i_global = lig_indices[bond.GetBeginAtomIdx()]
            j_global = lig_indices[bond.GetEndAtomIdx()]
            bt = rdkit_to_biotite.get(bond.GetBondType(), BondType.SINGLE)
            atom_array.bonds.add_bond(i_global, j_global, bt)

        logger.info(
            f"create_bond_graph: assigned {mol.GetNumBonds()} bonds for '{lig_name}'"
        )
        assigned_any = True

    if not assigned_any:
        raise RuntimeError(
            "create_bond_graph: no bonds were assigned for any ligand. "
            f"specified_smiles keys: {list(specified_smiles.keys())}, "
            f"structure residue names: {unique_res_names}"
        )

    return atom_array


###########################################################################
# Ligand atom name remapping
###########################################################################


def _match_aligned_rmsd(match, ref_coords, pred_coords, ref_core_idx=None):
    """Return the rigid-body-aligned RMSD for one RDKit substructure match.

    ``match[pred_idx] == ref_idx`` because the predicted molecule is the
    query passed to ``ref_mol.GetSubstructMatches(pred_mol)``.  Core atoms are
    therefore selected by their *values* in ``match``, not by indexing the
    match tuple with reference indices.

    Prediction oracles do not preserve the input coordinate frame, so raw
    coordinate RMSD cannot distinguish symmetric atom mappings.  Centering
    and Kabsch-aligning each candidate makes the tiebreak SE(3)-invariant.
    """
    match = np.asarray(match, dtype=int)
    pred_indices = np.arange(len(match), dtype=int)
    if ref_core_idx is not None:
        core = set(np.asarray(ref_core_idx, dtype=int).tolist())
        keep = np.asarray([ref_idx in core for ref_idx in match], dtype=bool)
        pred_indices = pred_indices[keep]
        ref_indices = match[keep]
    else:
        ref_indices = match

    reference = np.asarray(ref_coords, dtype=float)[ref_indices]
    mobile = np.asarray(pred_coords, dtype=float)[pred_indices]
    reference_centered = reference - reference.mean(axis=0)
    mobile_centered = mobile - mobile.mean(axis=0)

    covariance = mobile_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    fitted = mobile_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((reference_centered - fitted) ** 2, axis=1))))


def _scan_one_structure(
    path,
    *,
    ref_smiles,
    ref_mol_bytes,
    ref_coords,
    ref_atom_names,
    ref_buried_idx,
    target_name,
    n_ref_atoms,
):
    """Scan a single CIF for a predicted→reference atom-name mapping.

    Returns ``(rmsd, mapping, pred_res)`` for the best substructure match in
    this structure, or ``None`` if the structure yields no usable mapping.
    The RMSD is computed over *buried* ref atoms only when ``ref_buried_idx``
    is supplied (rigid-core tiebreak), else over all atoms.

    Designed as a ``multiprocessing.Pool`` worker — all inputs are picklable.
    ``ref_mol_bytes`` is the binary serialization of the geometry-rebuilt
    reference Mol (preserves atom ordering to match ``ref_atom_names``).
    """
    from atomworks.enums import ChainType
    from rdkit import Chem

    from blocks.utils.ligand import _rebuild_mol_from_geometry
    from chaperone.io import parse_to_atom_array

    try:
        arr = parse_to_atom_array(path, remove_hydrogens=True)
    except Exception:
        return None

    pred_lig = arr[arr.chain_type == ChainType.NON_POLYMER]
    pred_lig = pred_lig[pred_lig.element != "H"]
    if len(pred_lig) == 0 or len(pred_lig) != n_ref_atoms:
        return None

    pred_res = str(np.unique(pred_lig.res_name)[0])
    pred_atom_names = set(str(n) for n in pred_lig.atom_name)
    ref_atom_name_set = set(ref_atom_names)
    if len(pred_atom_names) != n_ref_atoms or len(ref_atom_name_set) != n_ref_atoms:
        return None
    if pred_res == target_name and pred_atom_names == ref_atom_name_set:
        return None  # already remapped

    ref_mol = Chem.Mol(ref_mol_bytes)
    try:
        # Pass the pre-built reference mol (deserialized once above) instead of
        # ``ref_smiles`` so ``_rebuild_mol_from_geometry`` skips a redundant
        # ``Chem.MolFromSmiles`` on every scanned structure. ``ref_smiles`` is
        # still forwarded as a fallback.
        pred_mol = _rebuild_mol_from_geometry(pred_lig, ref_smiles, ref_mol=ref_mol)
    except Exception:
        return None

    matches = ref_mol.GetSubstructMatches(pred_mol)
    if not matches:
        return None

    best_rmsd = math.inf
    best_map = None
    for m in matches:
        rmsd = _match_aligned_rmsd(
            m,
            ref_coords,
            pred_lig.coord,
            ref_core_idx=ref_buried_idx,
        )
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_map = {
                str(pred_lig.atom_name[pi]): ref_atom_names[ri]
                for pi, ri in enumerate(m)
            }

    if best_map is None:
        return None
    return best_rmsd, best_map, pred_res


def build_remap_reference(
    reference_cif, target_name=None, *, core_atoms=None, buried_atoms=None
):
    """Parse a reference ligand CIF into the picklable bundle the scan needs.

    Factored out of :func:`remap_predicted_ligand` so both the standalone
    function and :class:`blocks.util_blocks.remap_ligand.RemapLigandBlock`
    derive the SMILES / ref-mol / core-index exactly the same way.

    ``core_atoms`` names the rigid-core reference atoms used for the RMSD
    tiebreak; a flexible linker/tail whose rotamer space would otherwise
    inflate the full-atom RMSD of a correct mapping is excluded. ``buried_atoms``
    is a backward-compatible alias (superseded by ``core_atoms``).

    Returns a dict with keys consumed by :func:`discover_ligand_remap`:
    ``target_name``, ``smiles``, ``ref_mol_bytes``, ``ref_coords``,
    ``ref_atom_names``, ``ref_buried_idx``, ``n_ref_atoms``.
    """
    from atomworks.enums import ChainType
    from atomworks.io.tools.rdkit import atom_array_to_rdkit
    from rdkit import Chem

    from blocks.utils.ligand import _rebuild_mol_from_geometry
    from chaperone.io import parse_to_atom_array

    core_atoms = core_atoms if core_atoms is not None else buried_atoms

    ref_arr = parse_to_atom_array(reference_cif, remove_hydrogens=True)
    ref_lig = ref_arr[ref_arr.chain_type == ChainType.NON_POLYMER]
    if len(ref_lig) == 0:
        ref_lig = ref_arr  # standalone ligand CIF

    if target_name is None:
        target_name = str(np.unique(ref_lig.res_name)[0])

    # Extract SMILES from reference CIF bond records
    ref_mol_base = atom_array_to_rdkit(ref_lig, hydrogen_policy="remove")
    smiles = Chem.MolToSmiles(ref_mol_base)
    ref_mol = _rebuild_mol_from_geometry(ref_lig, smiles)

    ref_atom_names = [str(n) for n in ref_lig.atom_name]

    # Core-atom index for rigid-core RMSD tiebreak (ignores flexible linker).
    ref_buried_idx = None
    if core_atoms:
        core_set = set(core_atoms)
        idx = [i for i, n in enumerate(ref_atom_names) if n in core_set]
        if len(idx) >= 3:
            ref_buried_idx = np.asarray(idx, dtype=int)
        else:
            logger.warning(
                "build_remap_reference: only %d of %d core_atoms found in "
                "reference; falling back to full-atom RMSD.",
                len(idx),
                len(core_atoms),
            )

    return {
        "target_name": target_name,
        "smiles": smiles,
        "ref_mol_bytes": ref_mol.ToBinary(),
        "ref_coords": np.asarray(ref_lig.coord),
        "ref_atom_names": ref_atom_names,
        "ref_buried_idx": ref_buried_idx,
        "n_ref_atoms": len(ref_lig),
    }


def discover_ligand_remap(
    paths,
    reference,
    *,
    sample_size=64,
    rmsd_ok=2.0,
    num_workers=None,
):
    """Scan ``paths`` for the best predicted→reference ligand-name mapping.

    Shared discovery core for :func:`remap_predicted_ligand` (global, over the
    whole handler) and the chunk-splittable
    :class:`blocks.util_blocks.remap_ligand.RemapLigandBlock` (per chunk).
    Because every prediction output of the same ligand shares identical atom
    naming, per-chunk discovery returns the same mapping as global discovery.

    Args:
        paths: List of CIF paths to scan (already resolved off-disk; no catcif
            tags — materialize those first).
        reference: Bundle from :func:`build_remap_reference`.
        sample_size: Scan a seeded random sample of up to this many paths first;
            if none clears ``rmsd_ok`` continue streaming the rest. 0/None →
            scan in given order (shuffled with a fixed seed for reproducibility).
        rmsd_ok: Stop on the first mapping with RMSD below this threshold.
        num_workers: Scan parallelism. None → ``min(cpu_count, 8)``. 0 → serial.

    Returns:
        ``(best_rmsd, atom_remap, pred_res)`` for the best match, or ``None``
        when no un-remapped structure yields a usable mapping (e.g. the chunk
        is already remapped).
    """
    if not paths:
        return None

    rng = random.Random(0)
    if sample_size and len(paths) > sample_size:
        sampled = rng.sample(paths, sample_size)
        sampled_set = set(sampled)
        rest = [p for p in paths if p not in sampled_set]
        rng.shuffle(rest)
        ordered_paths = sampled + rest
    else:
        ordered_paths = list(paths)
        rng.shuffle(ordered_paths)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    worker_fn = functools.partial(
        _scan_one_structure,
        ref_smiles=reference["smiles"],
        ref_mol_bytes=reference["ref_mol_bytes"],
        ref_coords=reference["ref_coords"],
        ref_atom_names=reference["ref_atom_names"],
        ref_buried_idx=reference["ref_buried_idx"],
        target_name=reference["target_name"],
        n_ref_atoms=reference["n_ref_atoms"],
    )

    best_rmsd = math.inf
    best_map = None
    best_pred_res = None
    pool = None
    try:
        if num_workers == 0:
            iterator = map(worker_fn, ordered_paths)
        else:
            pool = mp.Pool(num_workers)
            iterator = pool.imap_unordered(worker_fn, ordered_paths, chunksize=4)

        for res in iterator:
            if res is None:
                continue
            rmsd, m, pred_res = res
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_map = m
                best_pred_res = pred_res
            if best_rmsd < rmsd_ok:
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    if best_map is None:
        return None
    return best_rmsd, best_map, best_pred_res


def remap_predicted_ligand(
    handler,
    reference_cif,
    target_name=None,
    *,
    core_atoms=None,
    buried_atoms=None,
    sample_size=64,
    rmsd_ok=2.0,
    num_workers=None,
):
    """Remap ligand atom and residue names in prediction outputs to match a reference.

    Streams through ``handler`` structures until one yields a substructure
    match with RMSD < ``rmsd_ok``, then applies that mapping to every output
    CIF on disk. If ``core_atoms`` is supplied, RMSD is computed only over
    those rigid-core atoms (ignores flexible linker tails).

    SMILES is extracted automatically from the reference CIF's bond records.

    Args:
        handler: PipelineDataHandler with prediction outputs (from ``engine.run()``).
        reference_cif: Path to a ligand-only CIF with canonical atom names
            and ``_chem_comp_bond`` records.
        target_name: Desired residue name in outputs (e.g. ``"E;T"``).
            If None, read from the reference CIF.
        core_atoms: Reference atom names of rigid-core atoms for the RMSD
            tiebreak. When given, the RMSD used for the "good enough" cutoff
            is computed over only these atoms, ignoring flexible linker tails
            whose rotamer space would otherwise inflate the RMSD of correct
            mappings (and prevent the scan from ever clearing ``rmsd_ok``,
            forcing a full-chunk fall-through). Must name >=3 atoms in the
            reference to take effect.
        buried_atoms: Backward-compatible alias for ``core_atoms``.
        sample_size: Start by scanning a seeded random sample of up to this
            many structures. If none yields ``rmsd < rmsd_ok`` in the
            sample, continue streaming through the remaining rows.
            Set to 0 or None to scan in the handler's own order.
        rmsd_ok: Stop on the first mapping with RMSD below this threshold.
        num_workers: Parallelism for the scan. Defaults to
            ``min(os.cpu_count(), 8)``. 0 forces serial (reproducibility).

    Raises:
        ValueError: If no matching ligand is found in any output.

    Example::

        handler = engine.run(af3_block, data=input_handler)
        remap_predicted_ligand(
            handler, "ligand.cif", target_name="E;T",
            core_atoms=["C0", "C1", "N1", "O3"],  # rigid core
        )
    """
    from chaperone.data.components import apply_atom_name_remap

    # ── 1. Build reference bundle (SMILES, ref-mol, core index) ─────
    reference = build_remap_reference(
        reference_cif,
        target_name=target_name,
        core_atoms=core_atoms,
        buried_atoms=buried_atoms,
    )
    target_name = reference["target_name"]

    # ── 2. Collect all paths, discover the mapping globally ──────────
    paths = [loaded.path for loaded in handler.get_data(load=False, which="last")]
    if not paths:
        logger.info("remap_predicted_ligand: empty handler, skipping.")
        return

    result = discover_ligand_remap(
        paths,
        reference,
        sample_size=sample_size,
        rmsd_ok=rmsd_ok,
        num_workers=num_workers,
    )
    if result is None:
        logger.info(
            "remap_predicted_ligand: no un-remapped structures found for %s "
            "(all may already be remapped). Skipping.",
            target_name,
        )
        return
    best_rmsd, best_map, best_pred_res = result

    # ── 3. Apply mapping to every output CIF in place ────────────────
    residue_remap = None
    if best_pred_res and best_pred_res != target_name:
        residue_remap = {best_pred_res: target_name}

    apply_atom_name_remap(handler, best_map, residue_remap=residue_remap)
    rmsd_label = "core" if reference["ref_buried_idx"] is not None else "full"
    logger.info(
        "Remapped ligand: %s → %s (%d atoms, %s-atom RMSD %.3f Å)",
        best_pred_res,
        target_name,
        len(best_map),
        rmsd_label,
        best_rmsd,
    )
