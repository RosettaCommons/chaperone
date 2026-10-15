import glob
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.enums import ChainType
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import to_cif_string
from biotite import structure as struc
from biotite.structure.io.pdbx.convert import COMP_BOND_TYPE_TO_ORDER
from biotite.structure.residues import residue_iter
from pydantic import Field, field_validator, model_validator

from blocks.constants import CUDA_INSTALL
from blocks.networks.prediction.monomer_reference import (
    default_msa_dirs,
    load_monomer_references,
    normalize_msa_dirs,
    residue_sequence,
    resolve_msa_path,
    stage_monomer_manifest,
    stage_query_a3m,
    standard_residue_sequence,
    write_monomer_manifest,
)
from blocks.networks.prediction.helpers import (
    create_af3_json_from_design,
    fill_af3_json_from_template,
)
from chaperone import JobArgs, PipelineDataHandler, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, ScriptSpec
from chaperone.block_config.prediction import (
    FULL_MEAN_PAE_KEY,
    INTERFACE_MEAN_PAE_KEY,
    MIN_PAE_INTERACTION_KEY,
)
from chaperone.common import glob_designs_in_dir
from chaperone.containers import Image
from chaperone.data import make_row
from chaperone.io import parse_to_atom_array
from typing import Any

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
AF3_RUNTIME_SCRIPT_DIR = Path("/opt/chaperone-runtime-scripts")

###########################################################################
# Container
###########################################################################

ALPHAFOLD3_CONTAINER = (
    Image("alphafold3")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "gnupg",
        "cmake",
        "ninja-build",
        "patch",
        "zlib1g-dev",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
        "python3-pip",
        "libx11-6",
        "libxau6",
        "libxext6",
        "libxrender1",
    )
    .run(*CUDA_INSTALL)
    .git_clone(
        "https://github.com/google-deepmind/alphafold3",
        "/opt/alphafold3",
        branch="main",
    )
    .run("cd /opt/alphafold3 && git checkout --detach 71e671112ab5f287fdabd3ce86b5d05805dd0d81")
    # Build HMMER 3.4 from source
    .run(
        "mkdir -p /opt/hmmer_build /usr/local/hmmer",
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /opt/hmmer_build",
        "cd /opt/hmmer_build && echo "
        "'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz' "
        "| sha256sum --check",
        "cd /opt/hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz",
        "cd /opt/hmmer_build && patch -p0 < /opt/alphafold3/docker/jackhmmer_seq_limit.patch",
        "cd /opt/hmmer_build/hmmer-3.4 && ./configure --prefix /usr/local/hmmer && make -j16 && make install",
        "cd /opt/hmmer_build/hmmer-3.4/easel && make install",
        "rm -rf /opt/hmmer_build",
    )
    # Pin both the source and its tested dependency lock. Resolving transitive
    # Flax/Qwix versions independently can import APIs missing from JAX even
    # when pip accepts their declared dependency ranges.
    .run(
        "python3.12 -m venv /opt/uv-env && /opt/uv-env/bin/pip install uv==0.9.24",
        "cd /opt/alphafold3 && UV_PROJECT_ENVIRONMENT=/opt/af3-env "
        "/opt/uv-env/bin/uv sync --frozen --no-dev --no-editable",
        "/opt/uv-env/bin/uv pip check --python /opt/af3-env/bin/python",
        "/opt/af3-env/bin/build_data",
    )
    # The lock includes CUDA 12.9/cuDNN 9.17 wheels with Blackwell support.
    # Expose their libraries to the linker as well as the Python packages.
    .run(
        "for d in /opt/af3-env/lib/python3.12/site-packages/nvidia/*/lib; do "
        'for f in "$d"/*.so*; do '
        '[ -f "$f" ] && ln -sf "$f" /usr/local/cuda/lib64/; '
        "done; done",
        # Register nvidia pip package lib dirs with ldconfig so the dynamic
        # linker finds them even if LD_LIBRARY_PATH is stripped.
        "echo /opt/af3-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib > /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/cublas/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/cusparse/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/cudnn/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/cufft/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/cusolver/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/nccl/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /opt/af3-env/lib/python3.12/site-packages/nvidia/nvjitlink/lib >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && echo /usr/local/cuda/lib64 >> /etc/ld.so.conf.d/nvidia-pip.conf"
        " && ldconfig",
    )
    .copy(str(SCRIPT_DIR / "run_alphafold_AF.py"), "/opt/chaperone-build-smoke/run_alphafold_AF.py")
    .run(
        "JAX_PLATFORMS=cpu /opt/af3-env/bin/python -c "
        "\"import runpy; runpy.run_path('/opt/chaperone-build-smoke/run_alphafold_AF.py', "
        "run_name='chaperone_build_smoke')\"",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/af3-env/bin:/usr/local/cuda/bin:/usr/local/hmmer/bin:$PATH",
        LD_LIBRARY_PATH="/opt/af3-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/af3-env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/af3-env",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false",
        XLA_PYTHON_CLIENT_PREALLOCATE="true",
        XLA_CLIENT_MEM_FRACTION="0.95",
        TF_ENABLE_ONEDNN_OPTS="0",
    )
    .label(Version="71e671112ab5f287fdabd3ce86b5d05805dd0d81", CUDA="12.8-toolkit/12.9-wheels")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[(str(SCRIPT_DIR), str(AF3_RUNTIME_SCRIPT_DIR))],
        env={
            "ALPHAFOLD3_WEIGHTS_DIR": lambda: get_setting_field(
                "ALPHAFOLD3_WEIGHTS_DIR", ""
            )
        },
    )
)


###########################################################################
# AlphaFold3
###########################################################################


###########################################################################
# mmCIF input payload (AtomArray -> AF3 mmCIF fold input)
###########################################################################

# Placeholder components already present in AF3's built-in CCD.  Observed
# ``UNK`` residues can contain different subsets of atoms, so attempting to
# reconstruct one user CCD from coordinates is both unnecessary and ambiguous.
_AF3_BUILTIN_PLACEHOLDER_CCDS = {"UNK"}


def _adapt_unsupported_polymers_for_af3(atom_array):
    """Represent D-polypeptides as bonded ligand components for AF3.

    AF3's native mmCIF reader silently omits ``polypeptide(D)`` entities.
    Keeping their atoms and bonds but declaring them non-polymer makes AF3
    ingest the residues as a multi-component ligand instead of dropping the
    chain. L-polypeptides and nucleic acids retain their polymer semantics.
    """
    annotations = atom_array.get_annotation_categories()
    if "chain_type" not in annotations:
        return atom_array

    unsupported = np.asarray(atom_array.chain_type) == ChainType.POLYPEPTIDE_D
    if not np.any(unsupported):
        return atom_array

    out = atom_array.copy()
    out.chain_type[unsupported] = ChainType.NON_POLYMER
    if "is_polymer" in annotations:
        out.is_polymer[unsupported] = False
    if "is_protein" in annotations:
        out.is_protein[unsupported] = False
    return out


def _disambiguate_chain_ids(atom_array) -> tuple[Any, dict[str, str] | None]:
    """Assign unique AF3 chain IDs when one ``chain_id`` has several iids."""
    if "chain_iid" not in atom_array.get_annotation_categories():
        return atom_array, None

    pairs = list(
        dict.fromkeys(
            (str(chain_id), str(chain_iid))
            for chain_id, chain_iid in zip(
                atom_array.chain_id, atom_array.chain_iid, strict=True
            )
        )
    )
    iids_by_chain: dict[str, list[str]] = {}
    for chain_id, chain_iid in pairs:
        iids_by_chain.setdefault(chain_id, []).append(chain_iid)

    ambiguous = {
        chain_id
        for chain_id, chain_iids in iids_by_chain.items()
        if len(chain_iids) > 1
    }
    af3_to_original_iid = {
        chain_id: chain_iids[0]
        for chain_id, chain_iids in iids_by_chain.items()
        if chain_id not in ambiguous
    }
    if not ambiguous:
        return atom_array, af3_to_original_iid

    generator = create_chain_id_generator(
        unavailable_chain_ids=set(
            str(chain) for chain in np.unique(atom_array.chain_id)
        )
    )
    renamed = atom_array.copy()
    for chain_id, chain_iid in pairs:
        if chain_id not in ambiguous:
            continue
        new_chain_id = next(generator)
        mask = (atom_array.chain_id == chain_id) & (atom_array.chain_iid == chain_iid)
        renamed.chain_id[mask] = new_chain_id
        af3_to_original_iid[new_chain_id] = chain_iid
    return renamed, af3_to_original_iid


def _renumber_res_ids_per_chain(atom_array):
    """Renumber residues to ``1..N`` within each AF3-visible chain."""
    out = atom_array.copy()
    new_res_id = out.res_id.copy()
    for chain_id in dict.fromkeys(str(chain) for chain in out.chain_id):
        atom_indices = np.where(out.chain_id == chain_id)[0]
        bounds = struc.get_residue_starts(out[atom_indices], add_exclusive_stop=True)
        for new_id, (start, stop) in enumerate(
            zip(bounds[:-1], bounds[1:], strict=True), start=1
        ):
            new_res_id[atom_indices[start:stop]] = new_id
    out.res_id = new_res_id
    if "auth_seq_id" in out.get_annotation_categories():
        out.set_annotation("auth_seq_id", new_res_id.copy())
    return out


def _normalize_entity_ids(atom_array):
    """Make entity IDs valid, positive mmCIF identifiers.

    AtomWorks' parsed assembly annotations are zero-based, while mmCIF entity
    IDs are one-based.  Biotite writes ``label_entity_id`` verbatim and AF3
    consequently drops entity ``0`` entirely.  Preserve the existing entity
    equivalence classes, but remap them to contiguous IDs starting at 1 for
    both the AtomWorks metadata writer and Biotite's ``atom_site`` writer.
    """
    annotations = atom_array.get_annotation_categories()
    if "chain_entity" in annotations:
        source = atom_array.chain_entity
    elif "label_entity_id" in annotations:
        source = atom_array.label_entity_id
    else:
        source = atom_array.chain_id

    keys = [str(value) for value in source]
    mapping = {
        key: entity_id for entity_id, key in enumerate(dict.fromkeys(keys), start=1)
    }
    entity_ids = np.asarray([mapping[key] for key in keys], dtype=int)
    out = atom_array.copy()
    out.set_annotation("chain_entity", entity_ids)
    out.set_annotation("label_entity_id", entity_ids.astype(str))
    return out


def _af3_entity_categories(atom_array) -> dict[str, dict[str, list]]:
    """Build the entity tables AF3 requires to retain polymer chains.

    Biotite can reconstruct ``entity`` and ``struct_asym`` from ``atom_site``,
    but it cannot reconstruct ``entity_poly_seq``. AF3 filters polymers before
    it synthesizes the later scheme tables, so omitting this category silently
    drops every protein while retaining ligands.
    """
    annotations = atom_array.get_annotation_categories()
    if "is_polymer" in annotations:
        polymer_mask = np.asarray(atom_array.is_polymer, dtype=bool)
    else:
        polymer_mask = np.isin(
            atom_array.chain_type,
            [
                ChainType.POLYPEPTIDE_L,
                ChainType.DNA,
                ChainType.RNA,
            ],
        )

    chain_starts = struc.get_chain_starts(atom_array)
    chain_ids = [str(atom_array.chain_id[index]) for index in chain_starts]
    chain_entities = [str(atom_array.chain_entity[index]) for index in chain_starts]
    entity_ids = list(dict.fromkeys(chain_entities))
    entity_types = []
    for entity_id in entity_ids:
        mask = np.asarray(
            [str(value) == entity_id for value in atom_array.chain_entity]
        )
        entity_types.append("polymer" if np.any(polymer_mask & mask) else "non-polymer")

    categories = {
        "entity": {
            "id": entity_ids,
            "type": entity_types,
            "pdbx_description": ["?"] * len(entity_ids),
        },
        "struct_asym": {
            "id": chain_ids,
            "entity_id": chain_entities,
        },
    }

    poly_type_by_chain_type = {
        ChainType.POLYPEPTIDE_L: "polypeptide(L)",
        ChainType.DNA: "polydeoxyribonucleotide",
        ChainType.RNA: "polyribonucleotide",
    }
    poly_types: list[str] = []
    poly_type_entity_ids: list[str] = []
    poly_entity_ids: list[str] = []
    poly_nums: list[int] = []
    poly_monomers: list[str] = []
    poly_hetero: list[str] = []
    poly_scheme: dict[str, list] = {
        "asym_id": [],
        "entity_id": [],
        "seq_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "pdb_seq_num": [],
        "auth_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
        "hetero": [],
    }
    nonpoly_scheme: dict[str, list] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "pdb_seq_num": [],
        "auth_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
    }
    for entity_id, entity_type in zip(entity_ids, entity_types, strict=True):
        entity_chain_ids = [
            chain_id
            for chain_id, chain_entity in zip(chain_ids, chain_entities, strict=True)
            if chain_entity == entity_id
        ]
        if entity_type != "polymer":
            for chain_id in entity_chain_ids:
                chain = atom_array[atom_array.chain_id == chain_id]
                for number, residue in enumerate(residue_iter(chain), start=1):
                    monomer = str(residue.res_name[0])
                    residue_id = int(residue.res_id[0])
                    ins_code = str(residue.ins_code[0])
                    if ins_code in {"", "?", "."}:
                        ins_code = "."
                    nonpoly_scheme["asym_id"].append(chain_id)
                    nonpoly_scheme["entity_id"].append(entity_id)
                    nonpoly_scheme["mon_id"].append(monomer)
                    nonpoly_scheme["ndb_seq_num"].append(number)
                    nonpoly_scheme["pdb_seq_num"].append(residue_id)
                    nonpoly_scheme["auth_seq_num"].append(residue_id)
                    nonpoly_scheme["pdb_mon_id"].append(monomer)
                    nonpoly_scheme["auth_mon_id"].append(monomer)
                    nonpoly_scheme["pdb_strand_id"].append(chain_id)
                    nonpoly_scheme["pdb_ins_code"].append(ins_code)
            continue
        entity_mask = np.asarray(
            [str(value) == entity_id for value in atom_array.chain_entity]
        )
        chain_type = ChainType.as_enum(atom_array.chain_type[entity_mask][0])
        if chain_type not in poly_type_by_chain_type:
            raise ValueError(
                f"AF3 mmCIF input does not support polymer chain type {chain_type}"
            )
        poly_type_entity_ids.append(entity_id)
        poly_types.append(poly_type_by_chain_type[chain_type])
        sequences = []
        insertion_codes = []
        for chain_id in entity_chain_ids:
            chain = atom_array[atom_array.chain_id == chain_id]
            residues = list(residue_iter(chain))
            sequences.append([str(residue.res_name[0]) for residue in residues])
            insertion_codes.append(
                [
                    "."
                    if str(residue.ins_code[0]) in {"", "?", "."}
                    else str(residue.ins_code[0])
                    for residue in residues
                ]
            )
        if any(sequence != sequences[0] for sequence in sequences[1:]):
            raise ValueError(
                f"Entity {entity_id} groups polymer chains with different sequences: "
                f"{entity_chain_ids}"
            )
        for number, monomer in enumerate(sequences[0], start=1):
            poly_entity_ids.append(entity_id)
            poly_nums.append(number)
            poly_monomers.append(monomer)
            poly_hetero.append("n")
        for chain_id, sequence, chain_ins_codes in zip(
            entity_chain_ids, sequences, insertion_codes, strict=True
        ):
            for number, (monomer, ins_code) in enumerate(
                zip(sequence, chain_ins_codes, strict=True), start=1
            ):
                poly_scheme["asym_id"].append(chain_id)
                poly_scheme["entity_id"].append(entity_id)
                poly_scheme["seq_id"].append(number)
                poly_scheme["mon_id"].append(monomer)
                poly_scheme["ndb_seq_num"].append(number)
                poly_scheme["pdb_seq_num"].append(number)
                poly_scheme["auth_seq_num"].append(number)
                poly_scheme["pdb_mon_id"].append(monomer)
                poly_scheme["auth_mon_id"].append(monomer)
                poly_scheme["pdb_strand_id"].append(chain_id)
                poly_scheme["pdb_ins_code"].append(ins_code)
                poly_scheme["hetero"].append("n")

    if poly_entity_ids:
        # AtomWorks currently lowercases ``polypeptide(L)`` to
        # ``polypeptide(l)``. AF3 matches this controlled vocabulary
        # case-sensitively, so write the standards-compliant value explicitly.
        categories["entity_poly"] = {
            "entity_id": poly_type_entity_ids,
            "type": poly_types,
        }
        categories["entity_poly_seq"] = {
            "entity_id": poly_entity_ids,
            "num": poly_nums,
            "mon_id": poly_monomers,
            "hetero": poly_hetero,
        }
        categories["pdbx_poly_seq_scheme"] = poly_scheme
    if nonpoly_scheme["asym_id"]:
        categories["pdbx_nonpoly_scheme"] = nonpoly_scheme
    return categories


def _residue_signature(residue) -> tuple:
    """Return an order-independent atom/bond signature for one residue."""
    annotations = residue.get_annotation_categories()
    charges = residue.charge if "charge" in annotations else np.zeros(len(residue), int)
    atoms = tuple(
        sorted(
            (str(name), str(element), int(charge))
            for name, element, charge in zip(
                residue.atom_name, residue.element, charges, strict=True
            )
        )
    )
    bonds = ()
    if residue.bonds is not None:
        bonds = tuple(
            sorted(
                (
                    *sorted((str(residue.atom_name[i]), str(residue.atom_name[j]))),
                    int(bond_type),
                )
                for i, j, bond_type in residue.bonds.as_array()
            )
        )
    return atoms, bonds


def _cif_token(value) -> str:
    """Quote a scalar for the small user-CCD tables below."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    value = str(value)
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    raise ValueError(
        f"Cannot serialize CIF value containing both quote styles: {value!r}"
    )


def _user_ccd_block(res_name: str, residue) -> str:
    """Build one AF3-compatible user-CCD data block from a residue."""
    annotations = residue.get_annotation_categories()
    charge = residue.charge if "charge" in annotations else np.zeros(len(residue), int)
    aromatic = (
        residue.is_aromatic
        if "is_aromatic" in annotations
        else np.zeros(len(residue), bool)
    )
    leaving = (
        residue.is_leaving_atom
        if "is_leaving_atom" in annotations
        else np.zeros(len(residue), bool)
    )
    chem_comp_type = (
        str(residue.chem_comp_type[0])
        if "chem_comp_type" in annotations
        else "NON-POLYMER"
    )

    lines = [
        f"data_{res_name}",
        "#",
        f"_chem_comp.id {_cif_token(res_name)}",
        f"_chem_comp.name {_cif_token(res_name)}",
        f"_chem_comp.type {_cif_token(chem_comp_type)}",
        "_chem_comp.formula ?",
        "_chem_comp.mon_nstd_parent_comp_id ?",
        "_chem_comp.pdbx_synonyms ?",
        "_chem_comp.formula_weight ?",
        "#",
        "loop_",
        "_chem_comp_atom.comp_id",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_aromatic_flag",
        "_chem_comp_atom.pdbx_leaving_atom_flag",
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    ]
    for index in range(len(residue)):
        lines.append(
            " ".join(
                _cif_token(value)
                for value in (
                    res_name,
                    residue.atom_name[index],
                    residue.element[index],
                    int(charge[index]),
                    "Y" if aromatic[index] else "N",
                    "Y" if leaving[index] else "N",
                    *residue.coord[index],
                )
            )
        )

    lines.extend(
        [
            "#",
            "loop_",
            "_chem_comp_bond.comp_id",
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
            "_chem_comp_bond.pdbx_aromatic_flag",
        ]
    )
    for i, j, bond_type_int in residue.bonds.as_array():
        order, aromatic_flag = COMP_BOND_TYPE_TO_ORDER.get(
            struc.BondType(bond_type_int), ("SING", "N")
        )
        lines.append(
            " ".join(
                _cif_token(value)
                for value in (
                    res_name,
                    residue.atom_name[i],
                    residue.atom_name[j],
                    order,
                    aromatic_flag,
                )
            )
        )
    return "\n".join([*lines, "#", ""])


def build_af3_user_ccd(atom_array) -> str | None:
    """Serialize nonstandard residues as an inline AF3 user CCD."""
    standard_residues = np.asarray(
        list(STANDARD_AA) + list(STANDARD_DNA) + list(STANDARD_RNA)
    )
    nonstandard = ~np.isin(
        atom_array.res_name,
        np.concatenate(
            [standard_residues, np.asarray(sorted(_AF3_BUILTIN_PLACEHOLDER_CCDS))]
        ),
    )
    if not np.any(nonstandard):
        return None

    residues_by_name: dict[str, Any] = {}
    signatures: dict[str, tuple] = {}
    for residue in residue_iter(atom_array[nonstandard]):
        res_name = str(residue.res_name[0])
        # A bondless observed component (normally a monoatomic ion) cannot be
        # represented as an AF3 user CCD: AF3 requires non-empty bond columns.
        # These components come from the PDB CCD and are already available in
        # AF3's built-in dictionary, so leave them there instead of overriding
        # them with an invalid partial definition.
        if residue.bonds is None or len(residue.bonds.as_array()) == 0:
            continue
        signature = _residue_signature(residue)
        if res_name in signatures and signatures[res_name] != signature:
            raise ValueError(
                f"Residue name {res_name!r} refers to different atom/bond topologies; "
                "rename the components before preparing AF3 input."
            )
        signatures[res_name] = signature
        residues_by_name.setdefault(res_name, residue)

    if not residues_by_name:
        return None

    return "".join(
        _user_ccd_block(res_name, residue)
        for res_name, residue in residues_by_name.items()
    )


def _write_structure_cif(atom_array, name: str, *, entity_source) -> str:
    """Write AF3's structure mmCIF with AtomWorks, including covalent bonds."""
    from chaperone.io import ensure_bonds

    atom_array = ensure_bonds(atom_array)
    extra_categories = _af3_entity_categories(entity_source)
    from atomworks.io.utils.io_utils import CIFWriteConfig

    return to_cif_string(
        atom_array,
        config=CIFWriteConfig(
            id=name,
            include_entity_categories=True,
            include_chem_comp=False,
            include_nan_coords=False,
            extra_categories=extra_categories,
        ),
    )


def create_af3_mmcif_payload(
    atom_array,
    name: str,
    *,
    remove_ligands: bool = False,
    seed: int = 1,
) -> dict[str, Any]:
    """Create a manifest payload that AF3 can parse through ``Input.from_mmcif``."""
    if np.any(atom_array.res_name == "UNL"):
        raise ValueError(
            f"Input structure {name!r} contains reserved AF3 residue name 'UNL'; "
            "rename it before prediction."
        )

    prepared = atom_array
    if remove_ligands:
        if "is_polymer" in prepared.get_annotation_categories():
            polymer_mask = prepared.is_polymer
        else:
            polymer_mask = np.isin(
                prepared.chain_type,
                [ChainType.POLYPEPTIDE_L, ChainType.DNA, ChainType.RNA],
            )
        prepared = prepared[polymer_mask]
    if len(prepared) == 0:
        raise ValueError(f"Input structure {name!r} contains no atoms for AF3.")

    prepared, af3_chain_id_to_original_iid = _disambiguate_chain_ids(prepared)
    prepared = _adapt_unsupported_polymers_for_af3(prepared)
    # Preserve the full polymer sequence before dropping unresolved atoms.
    # AF3 uses entity_poly_seq to represent those residues as missing.
    entity_source = _normalize_entity_ids(_renumber_res_ids_per_chain(prepared))
    finite = np.all(np.isfinite(entity_source.coord), axis=-1)
    prepared = entity_source[finite]
    if len(prepared) == 0:
        raise ValueError(f"Input structure {name!r} contains no finite coordinates.")

    prepared = prepared.copy()
    prepared.set_annotation("occupancy", np.ones(len(prepared), dtype=float))
    prepared.set_annotation("b_factor", np.zeros(len(prepared), dtype=float))

    payload = {
        "name": name,
        "cif": _write_structure_cif(
            prepared,
            name,
            entity_source=entity_source,
        ),
        "seed": int(seed),
    }
    if user_ccd := build_af3_user_ccd(prepared):
        payload["user_ccd"] = user_ccd
    if af3_chain_id_to_original_iid is not None:
        payload["af3_chain_id_to_original_iid"] = af3_chain_id_to_original_iid
    return payload


class AlphaFold3Block(AF3LikePredictionBlock):
    """AlphaFold3 structure prediction.

    Basic usage serializes the upstream design's atom array as mmCIF and lets
    AF3 recover polymers, modifications, ligands, and covalent bonds from that
    structure. For hand-crafted inputs, ``template_json`` retains the legacy
    JSON path and the block fills in the designed sequence at runtime.

    PTM / covalent modification workflow (enzyme design):
        1. Create an AF3 template JSON with a ``userCCD`` block defining the
           custom ligand and an empty ``modifications`` list.
        2. Use ``add_ptm_at_mapped_residues`` to attach the ligand at a
           catalytic residue from the theozyme motif (position resolved via
           RFD3's diffused_index_map), or ``add_ptm_at_residues`` to specify
           positions directly.
    """

    _name = "af3"
    _display_name = "AlphaFold3"
    _description = "AlphaFold3 structure prediction"
    # ALPHAFOLD3_WEIGHTS_DIR overrides the container-internal model path.
    _script_spec = ScriptSpec(
        command=[
            "python",
            str(AF3_RUNTIME_SCRIPT_DIR / "run_alphafold_AF.py"),
            "--run_data_pipeline=False",
            "--run_inference=True",
            "--db_dir=/databases",
            "--model_dir=/databases/alphafold",
        ],
        image=ALPHAFOLD3_CONTAINER,
        source_url="https://github.com/google-deepmind/alphafold3",
    )

    def get_command_template(self) -> list[str]:
        """Build the AF3 command, resolving weight paths for the target env.

        ``ALPHAFOLD3_WEIGHTS_DIR`` overrides ``--model_dir`` and uses its
        parent for ``--db_dir``. Otherwise the container defaults are kept.
        """
        weights_dir = get_setting_field("ALPHAFOLD3_WEIGHTS_DIR", default="")
        binds = []
        if self.jax_cache_dir:
            self.jax_cache_dir = str(Path(self.jax_cache_dir).expanduser().resolve())
            binds.append((self.jax_cache_dir, self.jax_cache_dir))
        weights_path = None
        if weights_dir:
            weights_path = Path(weights_dir).expanduser().resolve()
            weights_dir = str(weights_path)
            binds.append((weights_dir, weights_dir + ":ro"))
        base = self._script_spec.image.build_command(
            *self._script_spec.command, binds=binds,
        )
        if not weights_dir:
            return base
        db_dir = str(weights_path.parent)
        overrides = [
            arg
            for arg in base
            if not arg.startswith("--db_dir=") and not arg.startswith("--model_dir=")
        ]
        overrides.extend(
            [
                f"--db_dir={db_dir}",
                f"--model_dir={weights_dir}",
            ]
        )
        return overrides

    jax_cache_dir: Optional[str] = Field(
        default_factory=lambda: get_setting_field("AF3_JAX_CACHE_DIR", default=None),
        description="Writable compilation cache mounted directly into the container",
    )

    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=200,
    )

    # JSON creation from template
    input_mode: Literal["auto", "mmcif", "json"] = Field(
        "mmcif",
        description=(
            "AF3 input preparation mode. The default 'mmcif' serializes the complete "
            "AtomArray and uses AF3's native Input.from_mmcif converter. Use 'json' "
            "for legacy JSON-only options, or 'auto' to select JSON when one is set. "
            "Native sequence artifacts always use JSON, including those carrying "
            "a pre-design backbone as context, so the designed sequence is authoritative."
        ),
    )
    input_preparation_version: Literal[2] = Field(
        2,
        frozen=True,
        repr=False,
        json_schema_extra={"cli_type": None, "internal": True},
        description="Internal cache revision for sequence-authoritative inputs and multi-seed manifests; not a user option.",
    )
    # ── Benchmarking against the input structures ───────────────────────
    #
    # AF3 here runs with --run_data_pipeline=False, so it never builds its own
    # MSAs: whatever is in the JSON's unpairedMsa field is all the model gets,
    # and create_af3_json_from_design leaves that empty. Without use_local_msas
    # an af3 run on natural targets is therefore single-sequence, which is a
    # very different experiment from the one people assume they are running.
    use_local_msas: bool = Field(
        False,
        description="Inject each protein chain's precomputed a3m from the on-disk "
        "MSA stores into the AF3 input. Works with both native mmCIF and legacy "
        "JSON input modes. Offline.",
    )
    msa_dirs: Optional[list] = Field(
        None,
        description="Sharded MSA roots searched in order, as paths or as "
        '{"dir", "extension", "directory_depth"} dicts. None uses PROTEIN_MSA_DIRS.',
    )
    max_msa_sequences: Optional[int] = Field(
        10_000,
        ge=1,
        description="Truncate injected MSAs to this depth.",
    )
    raise_if_missing_msa_for_protein_of_length_n: Optional[int] = Field(
        None,
        ge=0,
        description="Fail when a protein at least this many residues long has no "
        "on-disk MSA, instead of silently falling back to single-sequence.",
    )
    template_json: Optional[str] = Field(
        None,
        description="Path to template JSON file for AF3 input. If not provided, create input from scratch.",
    )
    add_ptm_at_mapped_residues: Optional[dict] = Field(
        None,
        description=(
            "Map of {ptmType: original_residue_key} for covalent modifications. "
            "The residue key (e.g. 'A10') is looked up in the RFD3 diffused_index_map "
            "to find the final residue position after diffusion. This is a niche option "
            "for enzyme design where the substrate is covalently attached at a catalytic "
            "residue that was part of the theozyme motif. Requires template_json with a "
            "userCCD block defining the custom ligand. "
            "Example: {'L:1': 'A10'} attaches custom CCD 'L:1' at the mapped position "
            "of original residue A10."
        ),
    )
    add_ptm_at_residues: Optional[list[dict]] = Field(
        None,
        description=(
            "Direct list of AF3 modification dicts to add to the protein sequence. "
            "Each entry should have 'ptmType' and 'ptmPosition' keys. Unlike "
            "add_ptm_at_mapped_residues, positions are specified directly (1-indexed) "
            "without going through the diffused_index_map. Requires template_json. "
            "Example: [{'ptmType': 'CCD_HY3', 'ptmPosition': 42}]"
        ),
    )

    consolidate_intermediates: bool = Field(
        True,
        description=(
            "After compiling, pack each design's redundant per-sample subdirs "
            "(outputs/<design>/seed-*_sample-*/) into a single uncompressed "
            "outputs/<design>/_intermediates.tar and delete the originals. "
            "Collapses ~15 unreferenced files per design (the bulk of a run's "
            "file count) into one archive. Only runs when read_most_confident_model "
            "is True (those subdirs are never read in that mode); a no-op otherwise. "
            "Idempotent and --recompile safe."
        ),
    )

    # `which` is inherited from AF3LikePredictionBlock.

    # ── Throughput knobs (plumbed to run_alphafold_AF.py) ────────────────────
    buckets: Optional[list[int]] = Field(
        None,
        description=(
            "Token-count buckets for JIT compilation. None uses the inner "
            "script's smart default (fine-grained at small token counts, "
            "coarser above). Strictly increasing list."
        ),
    )
    flash_attention_implementation: str = Field(
        "triton",
        description="'triton' (fastest, Ampere+), 'cudnn', or 'xla' (portable).",
    )
    conformer_max_iterations: Optional[int] = Field(
        None,
        description="RDKit conformer search max iterations. None uses RDKit default.",
    )
    mean_pae_contact_distance: float = Field(
        5.0,
        description=(
            "Contact cutoff (Å) for the `interface_mean_pae` metric: only protein "
            "residues with any atom within this distance of the ligand contribute. "
            "Lower focuses on the tightest pocket contacts."
        ),
    )
    mean_pae_core_atoms: Optional[list[str]] = Field(
        None,
        description=(
            "Ligand atom names defining the buried/core substructure for the "
            "`interface_mean_pae` and `full_mean_pae` metrics. When set, only "
            "these atoms are used — both to select pocket residues and as the "
            "ligand side of the PAE average — so a flexible linker or "
            "solvent-exposed tail does not dilute the score. None (default) uses "
            "all ligand heavy atoms."
        ),
    )
    mean_pae_sidechain_angle: Optional[float] = Field(
        90.0,
        description=(
            "Max angle (degrees) between a pocket residue's CA→sidechain-centroid "
            "vector and its CA→nearest-ligand-atom vector for that residue to "
            "count toward `interface_mean_pae`. Excludes residues whose sidechain "
            "is flipped away from the ligand despite being within the contact "
            "cutoff. 90° keeps sidechains on the ligand-facing hemisphere. "
            "Glycine (no sidechain) is always kept. Set to None to disable the "
            "orientation gate and use distance alone."
        ),
    )
    model_seeds: list[int] = Field(
        default_factory=lambda: [1],
        description=(
            "Independent AF3 model seeds written into each input JSON. Multiple "
            "seeds rerun trunk and diffusion inference and are useful for true "
            "replication; diffusion_batch_size samples are produced per seed."
        ),
    )
    # sort_by_length / sort_length_key / pre_chunk_sort are inherited from
    # AF3LikePredictionBlock (shared across all prediction oracles).

    @field_validator("model_seeds")
    @classmethod
    def _validate_model_seeds(cls, value):
        if not value or any(seed < 0 for seed in value):
            raise ValueError("model_seeds must contain nonnegative integers")
        if len(set(value)) != len(value):
            raise ValueError("model_seeds must be unique")
        return value

    @model_validator(mode="before")
    @classmethod
    def _remap_num_diffusion_samples(cls, data):
        return cls._remap_legacy_samples_field(data, "num_diffusion_samples")

    @model_validator(mode="after")
    def _check_unsupported(self):
        self._assert_unsupported_fields_default(
            n_recycles=None,
            num_diffusion_steps=200,
            ckpt_path=None,
        )
        return self

    @model_validator(mode="after")
    def validate_template_json_exists(self):
        if self.template_json is not None and not os.path.exists(self.template_json):
            raise ValueError(f"template_json file does not exist: {self.template_json}")
        if self.input_mode == "mmcif":
            incompatible = []
            if self.template_json is not None:
                incompatible.append("template_json")
            if self.specified_smiles:
                incompatible.append("specified_smiles")
            if self.specified_ccd:
                incompatible.append("specified_ccd")
            if incompatible:
                raise ValueError(
                    "input_mode='mmcif' is incompatible with legacy JSON options: "
                    + ", ".join(incompatible)
                )
        return self

    def _use_mmcif_input(self) -> bool:
        if self.input_mode == "mmcif":
            return True
        if self.input_mode == "json":
            return False
        return not (
            self.template_json is not None
            or self.specified_smiles
            or self.specified_ccd
        )

    def _manifest_dir(self, sandbox_dir) -> Path:
        return Path(sandbox_dir) / f"{self.name}_manifest"

    def _resolved_msa_dirs(self) -> Optional[list]:
        if self.msa_dirs is not None:
            return normalize_msa_dirs(self.msa_dirs)
        return default_msa_dirs()

    def _inject_msas(self, json_data: dict, manifest: dict, msa_scratch: Path) -> int:
        """Put the stored a3m into every protein chain's unpairedMsa field.

        AF3 reads the a3m content inline, so the staged file is only a
        convenient place to normalise and verify it first. Returns the depth
        actually injected (1 when no MSA was found).
        """
        sequence = manifest["sequence"]
        msa_path = resolve_msa_path(
            sequence,
            self._resolved_msa_dirs(),
            alternates=(manifest.get("msa_lookup_sequence") or "",),
        )
        if msa_path is None:
            threshold = self.raise_if_missing_msa_for_protein_of_length_n
            if threshold is not None and len(sequence) >= threshold:
                raise ValueError(
                    f"No on-disk MSA for {manifest['target']} "
                    f"({len(sequence)} residues)"
                )
            logger.warning(
                "No on-disk MSA for %s (%d residues); AF3 will run single-sequence",
                manifest["target"],
                len(sequence),
            )
            return 1

        msa_scratch.mkdir(parents=True, exist_ok=True)
        staged = msa_scratch / f"{manifest['target']}.a3m"
        depth = stage_query_a3m(
            msa_path, staged, sequence=sequence, max_sequences=self.max_msa_sequences
        )
        a3m_text = staged.read_text()
        injected = 0
        for entry in json_data.get("sequences", []):
            protein = entry.get("protein")
            if protein is None:
                continue
            if protein.get("sequence") != sequence:
                raise ValueError(
                    f"{manifest['target']}: AF3 chain sequence does not match the "
                    "reference crop, refusing to attach an MSA to the wrong chain"
                )
            protein["unpairedMsa"] = a3m_text
            protein["unpairedMsaPath"] = None
            injected += 1
        if injected == 0:
            raise ValueError(
                f"{manifest['target']}: no protein chain to attach an MSA to"
            )
        return depth

    def _mmcif_chain_msas(
        self,
        atom_array,
        payload: dict,
        *,
        target: str,
        msa_scratch: Path,
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Stage and return AF3-chain keyed A3Ms for a native mmCIF input."""
        annotations = atom_array.get_annotation_categories()
        if "chain_type" in annotations:
            # AF3's mmCIF reader does not accept polypeptide(D) as a protein
            # chain. The converter preserves it as a bonded ligand instead,
            # so do not stage an MSA that AF3 cannot consume.
            protein_mask = np.asarray(atom_array.chain_type) == ChainType.POLYPEPTIDE_L
        elif "is_protein" in annotations:
            protein_mask = np.asarray(atom_array.is_protein, dtype=bool)
        else:
            from biotite.structure import filter_amino_acids

            protein_mask = filter_amino_acids(atom_array)

        proteins = atom_array[protein_mask]
        if len(proteins) == 0:
            return {}, {}

        chain_category = "chain_iid" if "chain_iid" in annotations else "chain_id"
        af3_to_original = payload.get("af3_chain_id_to_original_iid") or {}
        original_to_af3 = {value: key for key, value in af3_to_original.items()}
        msa_dirs = self._resolved_msa_dirs()
        if not msa_dirs:
            raise ValueError(
                "use_local_msas=True but no MSA roots are configured; set msa_dirs "
                "or PROTEIN_MSA_DIRS"
            )

        unpaired_msas: dict[str, str] = {}
        depths: dict[str, int] = {}
        chain_values = proteins.get_annotation(chain_category)
        for chain_index, original_chain in enumerate(
            dict.fromkeys(str(value) for value in chain_values)
        ):
            chain = proteins[chain_values == original_chain]
            sequence = residue_sequence(chain)
            alternate = standard_residue_sequence(chain)
            af3_chain = original_to_af3.get(original_chain, str(chain.chain_id[0]))
            msa_path = resolve_msa_path(
                sequence,
                msa_dirs,
                alternates=(alternate,),
            )
            if msa_path is None:
                threshold = self.raise_if_missing_msa_for_protein_of_length_n
                if threshold is not None and len(sequence) >= threshold:
                    raise ValueError(
                        f"No on-disk MSA for {target} chain {af3_chain} "
                        f"({len(sequence)} residues)"
                    )
                logger.warning(
                    "No on-disk MSA for %s chain %s (%d residues); AF3 will run "
                    "that chain single-sequence",
                    target,
                    af3_chain,
                    len(sequence),
                )
                depths[af3_chain] = 1
                continue

            msa_scratch.mkdir(parents=True, exist_ok=True)
            staged = msa_scratch / f"{target}.chain_{chain_index}.a3m"
            depth = stage_query_a3m(
                msa_path,
                staged,
                sequence=sequence,
                max_sequences=self.max_msa_sequences,
            )
            unpaired_msas[af3_chain] = staged.read_text()
            depths[af3_chain] = depth

        return unpaired_msas, depths

    def create_command(self, handler, sandbox_dir, chunk_id) -> list:
        """Create a batched AF3 command in the directory."""
        _, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        input_manifest = self._prepare_af3_inputs(handler, inputdir)
        return self._build_af3_command(
            inputdir, outputdir, input_manifest=input_manifest
        )

    def _prepare_af3_inputs(self, handler, inputdir):
        """Write JSON input files for each design. Override for custom input prep."""
        if hasattr(handler, "get_prediction_inputs"):
            inputs = handler.get_prediction_inputs(which=self.which, load_context=True)
        else:  # Backward-compatible duck-typed handlers used by integrations.
            inputs = handler.get_data(which=self.which, load=True)
        inputs = list(inputs)

        # A native sequence artifact may carry the pre-design backbone as
        # context. Serializing that structure would silently predict its old
        # sequence instead of the authoritative MPNN sequence.
        use_mmcif = self._use_mmcif_input()
        if use_mmcif and (
            getattr(handler, "has_sequence_data", False)
            or any(loaded.atom_array is None for loaded in inputs)
        ):
            logger.info("Native sequence artifacts require AF3 JSON sequence input")
            use_mmcif = False

        if use_mmcif:
            sandbox_dir = Path(inputdir).parent
            manifest_path = sandbox_dir / "af3_input_manifest.json"
            metadata_dir = self._manifest_dir(sandbox_dir)
            payloads = []
            for loaded in inputs:
                payload = create_af3_mmcif_payload(
                    loaded.atom_array,
                    loaded.basename,
                    remove_ligands=self.remove_ligands,
                    seed=self.model_seeds[0],
                )
                payload["seeds"] = list(self.model_seeds)
                if self.use_local_msas:
                    unpaired_msas, msa_depths = self._mmcif_chain_msas(
                        loaded.atom_array,
                        payload,
                        target=loaded.basename,
                        msa_scratch=sandbox_dir / "msa_staging",
                    )
                    payload["unpaired_msa"] = unpaired_msas
                    write_monomer_manifest(
                        metadata_dir,
                        {
                            "target": loaded.basename,
                            "source_path": loaded.path,
                            "msa_depth": min(msa_depths.values(), default=1),
                            "msa_depths": msa_depths,
                        },
                    )
                payloads.append(payload)
            with open(manifest_path, "w") as f:
                json.dump(payloads, f)
            return manifest_path

        sandbox_dir = Path(inputdir).parent
        manifest_dir = self._manifest_dir(sandbox_dir)
        uses_manifest = self.use_local_msas
        for loaded in inputs:
            protein_sequences = getattr(loaded, "sequences", None)
            json_file = os.path.join(inputdir, f"{loaded.basename}.json")
            # With a manifest the manifest is the completion marker, written
            # last, so a target half-staged by a dead chunk is redone whole.
            done = (
                (manifest_dir / f"{loaded.basename}.json").exists()
                if uses_manifest
                else os.path.exists(json_file)
            )
            if done:
                logger.info(f"Skipping existing input: {json_file}")
                continue

            manifest = None
            if uses_manifest:
                manifest = stage_monomer_manifest(
                    loaded,
                    manifest_dir=manifest_dir,
                    write=False,
                )

            if self.template_json:
                json_data = fill_af3_json_from_template(
                    template_json=self.template_json,
                    add_ptm_at_mapped_residues=self.add_ptm_at_mapped_residues,
                    add_ptm_at_residues=self.add_ptm_at_residues,
                    atom_array=loaded.atom_array,
                    name=loaded.basename,
                    diffused_index_map=loaded.metadata.get(
                        "diffused_index_map",
                    ),
                    protein_sequences=protein_sequences,
                )
            else:
                json_data = create_af3_json_from_design(
                    atom_array=loaded.atom_array,
                    name=loaded.basename,
                    remove_ligands=self.remove_ligands,
                    specified_smiles=self.specified_smiles,
                    specified_ccd=self.specified_ccd,
                    disable_ccd_detection=self.disable_ccd_detection,
                    protein_sequences=protein_sequences,
                )

            json_data["modelSeeds"] = self.model_seeds
            if manifest is not None:
                manifest["msa_depth"] = (
                    self._inject_msas(json_data, manifest, sandbox_dir / "msa_staging")
                    if self.use_local_msas
                    else 1
                )

            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=4)

            if manifest is not None:
                write_monomer_manifest(manifest_dir, manifest)
        return None

    def _build_af3_command(
        self, inputdir, outputdir, input_manifest: Path | None = None
    ) -> list:
        """Build the AF3 CLI command. Shared across AF3 variants."""
        cmd = self.command_template + [
            (
                f"--input_manifest={input_manifest}"
                if input_manifest is not None
                else f"--input_dir={inputdir}"
            ),
            f"--output_dir={outputdir}",
            f"--num_diffusion_samples={self.diffusion_batch_size}",
            f"--flash_attention_implementation={self.flash_attention_implementation}",
            f"--write_all_structs={'false' if self.read_most_confident_model else 'true'}",
        ]
        if self.buckets is not None:
            cmd.append(f"--buckets={','.join(str(b) for b in self.buckets)}")
        if self.conformer_max_iterations is not None:
            cmd.append(f"--conformer_max_iterations={self.conformer_max_iterations}")
        if self.jax_cache_dir:
            Path(self.jax_cache_dir).mkdir(parents=True, exist_ok=True)
            cmd.append(f"--jax_compilation_cache_dir={self.jax_cache_dir}")
        return cmd

    def _compute_min_pae_interaction(self, pae_matrix):
        if not pae_matrix or len(pae_matrix) <= 1:
            return None
        off_diag = [
            pae_matrix[i][j]
            for i in range(len(pae_matrix))
            for j in range(len(pae_matrix[i]))
            if i != j and pae_matrix[i][j] is not None
        ]
        if not off_diag:
            return None
        return min(off_diag)

    @staticmethod
    def _map_ligand_tokens_to_atoms(lig_tok, tchain, tres, aa_chain, aa_res, heavy):
        """Map each ligand token index to its heavy-atom index in the AtomArray.

        AF3 atomises non-polymer entities: every ligand heavy atom becomes its
        own token, emitted in the same order as the output CIF. So within a
        given ``(chain_id, res_id)`` group the k-th ligand token is the k-th
        heavy atom. Grouping (rather than one global index) keeps the mapping
        correct for multi-residue ligands and for several ligand chains.

        Returns ``None`` if that 1:1 correspondence does not hold — e.g. a
        polymer chain got classified as ligand (one token per *residue*, not
        per atom), or the CIF and the confidences JSON disagree — so callers
        can refuse rather than silently score the wrong atoms.
        """
        atoms_by_group = {}
        for idx in np.flatnonzero(heavy):
            atoms_by_group.setdefault((aa_chain[idx], aa_res[idx]), []).append(idx)

        counts, mapping = {}, []
        for t in lig_tok:
            key = (tchain[t], tres[t])
            k = counts.get(key, 0)
            counts[key] = k + 1
            atoms = atoms_by_group.get(key)
            if atoms is None or k >= len(atoms):
                return None
            mapping.append(atoms[k])
        # Every heavy atom of a touched group must be accounted for, otherwise
        # we matched only a prefix and the correspondence is not 1:1.
        if any(len(atoms_by_group[key]) != n for key, n in counts.items()):
            return None
        return np.array(mapping, dtype=int)

    # Backbone atoms excluded when computing a residue's sidechain centroid.
    _BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

    def _resolve_ligand_selection(
        self, pae, token_chain_ids, token_res_ids, atom_array, core_atoms, metric_key
    ):
        """Shared setup for the mean-PAE metrics.

        Validates shapes, splits protein/ligand tokens, and applies the optional
        ``core_atoms`` restriction. Returns
        ``(pae, tchain, tres, aa_chain, aa_res, protein_chain, prot_tok,
        lig_tok, lig_coord)`` or ``None`` if the inputs cannot support a metric.

        The first chain in ``token_chain_ids`` is treated as the protein/binder;
        every other chain is ligand (matching the per-chain pLDDT convention).
        """
        if (
            pae is None
            or token_chain_ids is None
            or token_res_ids is None
            or atom_array is None
        ):
            return None
        pae = np.asarray(pae, dtype=float)
        n = len(token_chain_ids)
        if pae.shape != (n, n) or n <= 1 or len(token_res_ids) != n:
            return None

        tchain = [str(c) for c in token_chain_ids]
        tres = [str(r) for r in token_res_ids]
        unique_chains = list(dict.fromkeys(tchain))
        if len(unique_chains) < 2:
            return None
        protein_chain = unique_chains[0]
        prot_tok = [i for i, c in enumerate(tchain) if c == protein_chain]
        lig_tok = np.array([i for i, c in enumerate(tchain) if c != protein_chain])
        if not prot_tok or lig_tok.size == 0:
            return None

        # Ligand heavy-atom coordinates (all non-protein chains) from the model.
        aa = atom_array
        aa_chain = np.asarray([str(c) for c in aa.chain_id])
        aa_res = np.asarray([str(r) for r in aa.res_id])
        heavy = (np.asarray(aa.element) != "H") & (aa_chain != protein_chain)
        if not heavy.any():
            return None

        if core_atoms:
            # Restrict to the named core atoms. Resolve tokens to atoms first so
            # lig_tok and lig_coord stay index-aligned by construction.
            tok2atom = self._map_ligand_tokens_to_atoms(
                lig_tok, tchain, tres, aa_chain, aa_res, heavy
            )
            if tok2atom is None:
                logger.error(
                    f"{metric_key}: cannot map ligand tokens to atoms 1:1 "
                    f"({lig_tok.size} tokens vs {int(heavy.sum())} heavy atoms); "
                    "mean_pae_core_atoms cannot be applied, skipping metric."
                )
                return None
            aa_name = np.asarray([str(a) for a in aa.atom_name])
            wanted = set(core_atoms)
            keep = np.isin(aa_name[tok2atom], list(wanted))
            missing = wanted - set(aa_name[tok2atom].tolist())
            if missing:
                logger.warning(
                    f"{metric_key}: mean_pae_core_atoms not found in ligand: "
                    f"{sorted(missing)}"
                )
            if not keep.any():
                logger.error(
                    f"{metric_key}: no ligand atom matched mean_pae_core_atoms "
                    f"{sorted(wanted)}, skipping metric."
                )
                return None
            lig_tok = lig_tok[keep]
            lig_coord = aa.coord[tok2atom[keep]]
        else:
            lig_coord = aa.coord[heavy]
        if lig_coord.shape[0] == 0:
            return None

        return (
            pae,
            tchain,
            tres,
            aa_chain,
            aa_res,
            protein_chain,
            prot_tok,
            lig_tok,
            lig_coord,
        )

    def _sidechain_faces_ligand(self, res_coord, res_names, res_atom_names, lig_coord):
        """Whether a residue's sidechain projects toward the ligand.

        Compares the CA→sidechain-centroid vector with the CA→nearest-ligand-atom
        vector; the residue passes when the angle between them is below
        ``mean_pae_sidechain_angle``. This distinguishes a residue that actually
        presents its sidechain to the pocket from one that is merely nearby with
        the sidechain rotated away — both look identical to a distance cutoff.

        Glycine (and any residue with no resolved sidechain heavy atom) has no
        sidechain to orient and is always kept: its backbone contact is real.
        """
        max_angle = self.mean_pae_sidechain_angle
        if max_angle is None:
            return True
        if len(res_names) and str(res_names[0]).strip().upper() == "GLY":
            return True

        is_ca = res_atom_names == "CA"
        if not is_ca.any():
            return True  # No CA -> cannot orient; fall back to distance alone.
        ca = res_coord[is_ca][0]

        sidechain = res_coord[~np.isin(res_atom_names, list(self._BACKBONE_ATOMS))]
        if sidechain.shape[0] == 0:
            return True  # No sidechain heavy atoms resolved (e.g. GLY-like).

        v_sc = sidechain.mean(axis=0) - ca
        nearest = lig_coord[((lig_coord - ca) ** 2).sum(-1).argmin()]
        v_lig = nearest - ca

        n_sc = np.linalg.norm(v_sc)
        n_lig = np.linalg.norm(v_lig)
        if n_sc == 0 or n_lig == 0:
            return True  # Degenerate geometry; don't discard on numerical noise.

        cos = float(np.dot(v_sc, v_lig) / (n_sc * n_lig))
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return angle <= float(max_angle)

    def _compute_interface_mean_pae(
        self,
        pae,
        token_chain_ids,
        token_res_ids,
        atom_array,
        contact_distance,
        core_atoms=None,
    ):
        """Mean interface PAE between pocket residues and the ligand.

        Averages ``pae`` (both directions) over token pairs between the ligand
        and only the protein residues that have any atom within
        ``contact_distance`` Å of the ligand — i.e. the binding pocket, not the
        whole protein. Unlike ``_compute_min_pae_interaction`` (chain-pair min)
        or a naive all-cross-chain mean (dominated by distant residues and
        length-dependent), this reflects the confidence of the actual interface.

        Pocket residues are further gated on sidechain orientation (see
        ``_sidechain_faces_ligand``) unless ``mean_pae_sidechain_angle`` is None.

        ``core_atoms`` optionally restricts the ligand to named atoms (the
        buried core). The restriction applies to *both* sides of the metric:
        pocket residues are those contacting the core, and only core-atom
        tokens enter the average — so an exposed linker neither recruits its
        own contacting residues nor contributes its (typically poor) PAE.

        Returns ``None`` if inputs are missing, there is a single chain, no
        residue survives the contact/orientation gates, or ``core_atoms`` cannot
        be resolved.
        """
        resolved = self._resolve_ligand_selection(
            pae,
            token_chain_ids,
            token_res_ids,
            atom_array,
            core_atoms,
            INTERFACE_MEAN_PAE_KEY,
        )
        if resolved is None:
            return None
        (
            pae,
            _tchain,
            tres,
            aa_chain,
            aa_res,
            protein_chain,
            prot_tok,
            lig_tok,
            lig_coord,
        ) = resolved

        aa = atom_array
        aa_name = np.asarray([str(a) for a in aa.atom_name])
        aa_resname = np.asarray([str(r) for r in aa.res_name])

        # Group protein atoms by residue once, rather than re-slicing per token.
        is_protein = aa_chain == protein_chain
        res_index = {}
        for idx in np.flatnonzero(is_protein):
            res_index.setdefault(aa_res[idx], []).append(idx)

        # Keep protein tokens whose residue has any atom within contact_distance
        # of any ligand atom (the pocket) AND whose sidechain faces the ligand.
        c2 = float(contact_distance) ** 2
        iface = []
        for i in prot_tok:
            idxs = res_index.get(tres[i])
            if not idxs:
                continue
            idxs = np.asarray(idxs)
            res_coord = aa.coord[idxs]
            d2 = ((res_coord[:, None, :] - lig_coord[None, :, :]) ** 2).sum(-1).min()
            if d2 > c2:
                continue
            if not self._sidechain_faces_ligand(
                res_coord, aa_resname[idxs], aa_name[idxs], lig_coord
            ):
                continue
            iface.append(i)
        if not iface:
            return None

        iface = np.array(iface)
        vals = np.concatenate(
            [
                pae[np.ix_(iface, lig_tok)].ravel(),
                pae[np.ix_(lig_tok, iface)].ravel(),
            ]
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(vals.mean())

    def _compute_full_mean_pae(
        self,
        pae,
        token_chain_ids,
        token_res_ids,
        atom_array,
        core_atoms=None,
    ):
        """Mean PAE between the ligand and the whole binder — ungated.

        Averages ``pae`` (both directions) over every binder token × every
        (core) ligand token. Unlike ``interface_mean_pae`` there is NO pocket
        contact filter and NO sidechain-orientation gate on the protein side:
        all binder tokens contribute. This mirrors bcov's ``ligand_mean_pae``
        in ``run_alphafold_sm.py`` (binder-mask × ligand-mask, both directions),
        so it is protein-length-dependent by construction — that is intentional,
        not a bug.

        The ligand side still honours ``core_atoms`` (bcov's ``-rmsd_atoms``
        equivalent) so an exposed linker does not dilute the score. The
        distinction from the interface metric is the protein side only.

        Returns ``None`` if inputs are missing, there is a single chain, or
        ``core_atoms`` cannot be resolved.
        """
        resolved = self._resolve_ligand_selection(
            pae,
            token_chain_ids,
            token_res_ids,
            atom_array,
            core_atoms,
            FULL_MEAN_PAE_KEY,
        )
        if resolved is None:
            return None
        pae, _tchain, _tres, _aa_chain, _aa_res, _prot, prot_tok, lig_tok, _lc = (
            resolved
        )

        prot_tok = np.asarray(prot_tok)
        if prot_tok.size == 0 or lig_tok.size == 0:
            return None

        vals = np.concatenate(
            [
                pae[np.ix_(prot_tok, lig_tok)].ravel(),
                pae[np.ix_(lig_tok, prot_tok)].ravel(),
            ]
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(vals.mean())

    def compile(self, sandbox_dir):
        outputdir = os.path.join(sandbox_dir, "outputs")
        manifest_dir = self._manifest_dir(sandbox_dir)
        manifests = (
            load_monomer_references(manifest_dir) if manifest_dir.is_dir() else {}
        )
        if self.read_most_confident_model:
            # AF3 top-level: outputs/<design>/<design>_model.cif
            paths = []
            for design_dir in glob.glob(os.path.join(outputdir, "*")):
                if os.path.isdir(design_dir):
                    paths.extend(glob_designs_in_dir(design_dir))
        else:
            # All samples: outputs/<design>/seed-*_sample-*/<design>_*_model.cif
            paths = []
            for design_dir in glob.glob(os.path.join(outputdir, "*")):
                if os.path.isdir(design_dir):
                    for sample_dir in glob.glob(os.path.join(design_dir, "seed-*")):
                        if os.path.isdir(sample_dir):
                            paths.extend(glob_designs_in_dir(sample_dir))

        self._guard_outputs(paths, outputdir)

        # Build rows with new schema: design_metadata + step_metadata
        rows = []
        for path in paths:
            # Load metadata from confidence JSON
            meta_path = path.replace("_model.cif", "_summary_confidences.json")
            block_meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load AF3 metadata from {meta_path}: {e}")
                    data = {}
                block_meta = {
                    k: data.get(k)
                    for k in [
                        "fraction_disordered",
                        "has_clash",
                        "iptm",
                        "ptm",
                        "ranking_score",
                        "chain_pair_pae_min",
                    ]
                }

                pae_matrix = data.get("chain_pair_pae_min")
                min_pae_interaction = self._compute_min_pae_interaction(pae_matrix)
                if min_pae_interaction is not None:
                    block_meta[MIN_PAE_INTERACTION_KEY] = min_pae_interaction

                # Compute mean pLDDT from full confidences JSON
                full_conf_path = path.replace("_model.cif", "_confidences.json")
                if os.path.exists(full_conf_path):
                    try:
                        with open(full_conf_path, "r") as f:
                            full_data = json.load(f)
                        # Interface mean PAE: pocket residues (within
                        # mean_pae_contact_distance Å of the ligand, sidechain
                        # facing it) <-> ligand. Full mean PAE: ligand-internal.
                        try:
                            atom_array = parse_to_atom_array(path)
                        except Exception as e:
                            logger.error(f"Failed to parse {path} for mean PAE: {e}")
                            atom_array = None
                        interface_mean_pae = self._compute_interface_mean_pae(
                            pae=full_data.get("pae"),
                            token_chain_ids=full_data.get("token_chain_ids"),
                            token_res_ids=full_data.get("token_res_ids"),
                            atom_array=atom_array,
                            contact_distance=self.mean_pae_contact_distance,
                            core_atoms=self.mean_pae_core_atoms,
                        )
                        if interface_mean_pae is not None:
                            block_meta[INTERFACE_MEAN_PAE_KEY] = interface_mean_pae

                        full_mean_pae = self._compute_full_mean_pae(
                            pae=full_data.get("pae"),
                            token_chain_ids=full_data.get("token_chain_ids"),
                            token_res_ids=full_data.get("token_res_ids"),
                            atom_array=atom_array,
                            core_atoms=self.mean_pae_core_atoms,
                        )
                        if full_mean_pae is not None:
                            block_meta[FULL_MEAN_PAE_KEY] = full_mean_pae

                        atom_plddts = full_data.get("atom_plddts")
                        if atom_plddts:
                            plddts = np.array(atom_plddts)
                            block_meta["plddt"] = float(np.mean(plddts))

                            # Per-chain pLDDT: compute ligand_plddt from non-protein chains
                            atom_chain_ids = full_data.get("atom_chain_ids")
                            if atom_chain_ids and len(atom_chain_ids) == len(plddts):
                                unique_chains = list(dict.fromkeys(atom_chain_ids))
                                if len(unique_chains) > 1:
                                    protein_chain = unique_chains[0]
                                    ligand_mask = np.array(
                                        [c != protein_chain for c in atom_chain_ids]
                                    )
                                    if ligand_mask.any():
                                        block_meta["ligand_plddt"] = float(
                                            np.mean(plddts[ligand_mask])
                                        )
                    except Exception as e:
                        logger.error(f"Failed to load pLDDT from {full_conf_path}: {e}")

            # AF3 nests its outputs, and how deeply depends on the mode:
            #   read_most_confident_model=True  -> outputs/<design>/<design>_model.cif
            #   read_most_confident_model=False -> outputs/<design>/seed-N_sample-M/...
            # so walk up until a directory name matches a staged manifest rather
            # than assuming a fixed depth (which picks up "seed-1_sample-2").
            candidate = Path(str(path))
            manifest = None
            for parent in list(candidate.parents)[:3]:
                if parent.name in manifests:
                    manifest = manifests[parent.name]
                    break
            path_handle = None
            if manifest is not None:
                block_meta["msa_depth"] = manifest["msa_depth"]
                path_handle = manifest["source_path"]
            rows.append(
                make_row(
                    name=self.name,
                    path=str(path),
                    path_handle=path_handle,
                    metrics={self.name: block_meta},
                    as_metric=self.compile_as_metric,
                )
            )

        # Metrics are now extracted from the top-level files; collapse the
        # redundant per-sample subdirs into one archive per design. Safe only
        # when read_most_confident_model is True (the sample subdirs are never
        # read in that mode); a no-op otherwise.
        if self.consolidate_intermediates and self.read_most_confident_model:
            self._consolidate_intermediates(outputdir)

        return PipelineDataHandler(rows)

    def _consolidate_intermediates(self, outputdir: str) -> None:
        """Pack each design's redundant per-sample subdirs into a single tar.

        AF3 writes alternative diffusion samples under
        ``outputs/<design>/seed-*_sample-*/``. When ``read_most_confident_model``
        is True (the default) only the top-level ``<design>_model.cif`` and its
        ``*_confidences.json`` / ``*_summary_confidences.json`` are ever read —
        the per-sample subdirs (~15 files/design) are unreferenced dead weight
        and account for the bulk of a finished run's file count and size.
        Collapse each design's sample subdirs into a single uncompressed
        ``_intermediates.tar`` and delete the originals.

        Idempotent and ``--recompile`` safe: skips designs that already have a
        tar (writes to a ``.tmp`` then atomically renames so an interrupted run
        never leaves a partial archive that would be mistaken for complete),
        no-ops when there is nothing to archive, and never touches the
        parquet-referenced top-level structure.
        """
        n_designs = 0
        n_subdirs = 0
        for design_dir in glob.glob(os.path.join(outputdir, "*")):
            if not os.path.isdir(design_dir):
                continue
            sample_dirs = sorted(
                d
                for d in glob.glob(os.path.join(design_dir, "seed-*"))
                if os.path.isdir(d)
            )
            if not sample_dirs:
                continue  # nothing to archive (already consolidated or none)
            tar_path = os.path.join(design_dir, "_intermediates.tar")
            if os.path.exists(tar_path):
                continue  # already consolidated on a prior run
            tmp_path = tar_path + ".tmp"
            try:
                with tarfile.open(tmp_path, "w") as tar:
                    for sd in sample_dirs:
                        tar.add(sd, arcname=os.path.basename(sd))
                os.replace(tmp_path, tar_path)
            except Exception as e:
                logger.warning(
                    f"Failed to consolidate intermediates in {design_dir}: {e}"
                )
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                continue
            for sd in sample_dirs:
                shutil.rmtree(sd, ignore_errors=True)
                n_subdirs += 1
            n_designs += 1
        if n_designs:
            logger.info(
                f"Consolidated AF3 intermediates: {n_designs} designs, "
                f"{n_subdirs} sample subdirs collapsed into per-design "
                "_intermediates.tar"
            )


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "af3"
    SANDBOX = REPO_ROOT / "unittest_outs" / "af3"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = AlphaFold3Block(num_diffusion_samples=1, diffusion_batch_size=1)
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
