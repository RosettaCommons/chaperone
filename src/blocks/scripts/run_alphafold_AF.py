# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction runner.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

# ruff: noqa: E402 -- sys.path must be adjusted before importing AlphaFold 3.

from collections.abc import Callable, Iterable, Sequence
import csv
import dataclasses
import datetime
import functools
import json
import multiprocessing
import os
import pathlib
import sys

# This runner lives beside the Chaperone block module ``alphafold3.py``. When
# executed by file path, Python prepends that directory to ``sys.path`` and the
# block module would shadow the installed Google ``alphafold3`` package.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if pathlib.Path(entry or os.curdir).resolve() != _SCRIPT_DIR
]

# from Bio import SeqIO
# from Bio.PDB import PDBParser
import shutil
import string
import textwrap
import time
import typing
from typing import overload

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
import tokamax
from alphafold3.model import features
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
from alphafold3.model import model

import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import random

# Restore access to the adjacent MSA helper only after Google's package is
# resident in ``sys.modules``.
sys.path.insert(0, str(_SCRIPT_DIR))


def rewrite_a3m_query_for_af3(
    a3m: str,
    *,
    query_sequence: str,
    ptm_positions: Iterable[int],
) -> str:
    """Rewrite an A3M query to AF3's parsed sequence at PTM positions only.

    AF3 uses a frozen CCD-to-one-letter mapping that can differ from the
    current CCD used to locate the stored MSA.  Its exact query sequence is
    only known after ``Input.from_mmcif``.  Ordinary-residue mismatches remain
    fatal; substitutions are permitted solely at AF3-declared PTM positions.
    """
    if not a3m:
        return a3m

    lines = a3m.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith(">")
        )
    except StopIteration as error:
        raise ValueError("A3M contains no FASTA records") from error

    query_start = header_index + 1
    query_stop = next(
        (
            index
            for index in range(query_start, len(lines))
            if lines[index].startswith(">")
        ),
        len(lines),
    )
    stored_query = "".join(lines[query_start:query_stop])
    stored_query = "".join(char for char in stored_query if not char.islower())
    expected = query_sequence.upper()
    if len(stored_query) != len(expected):
        raise ValueError(
            "A3M query length does not match AF3's parsed sequence: "
            f"msa={len(stored_query)}, af3={len(expected)}"
        )

    mismatches = {
        index
        for index, (msa_residue, af3_residue) in enumerate(
            zip(stored_query.upper(), expected, strict=True), start=1
        )
        if msa_residue != af3_residue
    }
    disallowed = mismatches - {int(position) for position in ptm_positions}
    if disallowed:
        position = min(disallowed)
        raise ValueError(
            "A3M query differs from AF3 at a non-PTM residue: "
            f"position={position}, msa={stored_query[position - 1]!r}, "
            f"af3={expected[position - 1]!r}"
        )

    rewritten = [*lines[:query_start], expected, *lines[query_stop:]]
    return "\n".join(rewritten) + "\n"


_HOME_DIR = pathlib.Path(os.environ.get("HOME"))
_DEFAULT_MODEL_DIR = _HOME_DIR / "models"
_DEFAULT_DB_DIR = _HOME_DIR / "public_databases"

# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    "json_path",
    None,
    "Path to the input JSON file.",
)
_PDB_PATH = flags.DEFINE_string(
    "pdb_path",
    None,
    "Path to a PDB file to fold. If provided, the script will ignore JSON inputs.",
)
_INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Path to the directory containing input JSON files.",
)
_INPUT_MANIFEST = flags.DEFINE_string(
    "input_manifest",
    None,
    "Path to a Chaperone AF3 manifest containing inline mmCIF inputs.",
)

_PROTEIN_SEQS = flags.DEFINE_list(
    "protein_seqs",
    None,
    "Comma-separated list of protein sequences to fold. Example: --protein_seqs=ACDE,XYZ",
)
_RNA_SEQS = flags.DEFINE_list(
    "rna_seqs",
    None,
    "Comma-separated list of RNA sequences to fold. Example: --rna_seqs=AGCU,UUUUAA",
)
_DNA_SEQS = flags.DEFINE_list(
    "dna_seqs",
    None,
    "Comma-separated list of DNA sequences to fold. Example: --dna_seqs=ATGC,TTTTAA",
)
_ION_LIST = flags.DEFINE_list(
    "ion_list",
    None,
    "Comma-separated list of ions to include, and how many you want of each. Example: --ion_list=K:10,Mg:5,Ca:5",
)


_NAME = flags.DEFINE_string(
    "name",
    None,
    "job name, or string to use for output name when specifying seqs via commandline arguments (no pdb/cif/json/fasta file inputs to derive name from).",
)

_MODEL_SEEDS = flags.DEFINE_list(
    "model_seeds",
    None,
    "Comma-separated list of random seeds. Example: --model_seeds=123,456. "
    "If not provided, we will generate a single random seed automatically.",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Path to a directory where the results will be saved.",
)
MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    _DEFAULT_MODEL_DIR.as_posix(),
    "Path to the model to use for inference.",
)


# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    "run_data_pipeline",
    True,
    "Whether to run the data pipeline on the fold inputs.",
)
_RUN_INFERENCE = flags.DEFINE_bool(
    "run_inference",
    True,
    "Whether to run inference on the fold inputs.",
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    "jackhmmer_binary_path",
    shutil.which("jackhmmer"),
    "Path to the Jackhmmer binary.",
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    "nhmmer_binary_path",
    shutil.which("nhmmer"),
    "Path to the Nhmmer binary.",
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    "hmmalign_binary_path",
    shutil.which("hmmalign"),
    "Path to the Hmmalign binary.",
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    "hmmsearch_binary_path",
    shutil.which("hmmsearch"),
    "Path to the Hmmsearch binary.",
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    "hmmbuild_binary_path",
    shutil.which("hmmbuild"),
    "Path to the Hmmbuild binary.",
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    "db_dir",
    (_DEFAULT_DB_DIR.as_posix(),),
    "Path to the directory containing the databases. Can be specified multiple"
    " times to search multiple directories in order.",
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    "small_bfd_database_path",
    "${DB_DIR}/bfd-first_non_consensus_sequences.fasta",
    "Small BFD database path, used for protein MSA search.",
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    "mgnify_database_path",
    "${DB_DIR}/mgy_clusters_2022_05.fa",
    "Mgnify database path, used for protein MSA search.",
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    "uniprot_cluster_annot_database_path",
    "${DB_DIR}/uniprot_all_2021_04.fa",
    "UniProt database path, used for protein paired MSA search.",
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    "uniref90_database_path",
    "${DB_DIR}/uniref90_2022_05.fa",
    "UniRef90 database path, used for MSA search. The MSA obtained by "
    "searching it is used to construct the profile for template search.",
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    "ntrna_database_path",
    "${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    "NT-RNA database path, used for RNA MSA search.",
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    "rfam_database_path",
    "${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
    "Rfam database path, used for RNA MSA search.",
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    "rna_central_database_path",
    "${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta",
    "RNAcentral database path, used for RNA MSA search.",
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    "pdb_database_path",
    "${DB_DIR}/mmcif_files",
    "PDB database directory with mmCIF files path, used for template search.",
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    "seqres_database_path",
    "${DB_DIR}/pdb_seqres_2022_09_28.fasta",
    "PDB sequence database path, used for template search.",
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    "jackhmmer_n_cpu",
    min(multiprocessing.cpu_count(), 8),
    "Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going"
    " beyond 8 CPUs provides very little additional speedup.",
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    "nhmmer_n_cpu",
    min(multiprocessing.cpu_count(), 8),
    "Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going"
    " beyond 8 CPUs provides very little additional speedup.",
)

# Template search configuration.
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    "max_template_date",
    "2021-09-30",  # By default, use the date from the AlphaFold 3 paper.
    "Maximum template release date to consider. Format: YYYY-MM-DD. All "
    "templates released after this date will be ignored.",
)

_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    "conformer_max_iterations",
    None,  # Default to RDKit default parameters value.
    "Optional override for maximum number of iterations to run for RDKit "
    "conformer search.",
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "jax_compilation_cache_dir",
    None,
    "Path to a directory for the JAX compilation cache.",
)
_BUCKETS = flags.DEFINE_list(
    "buckets",
    # pyformat: disable
    [
        # Fine-grained at the small-token end (binder design + small enzymes
        # typically 100-300 tokens), coarser as inputs get larger. Cuts
        # attention-padding waste from up to 60% (132→256) to ~6% (132→144).
        # Extra compiles fully amortise once AF3_JAX_CACHE_DIR is set and
        # the input handler is sorted by length (sort_by_length=True on the
        # AlphaFold3Block).
        "128",
        "144",
        "160",
        "176",
        "192",
        "208",
        "224",
        "240",
        "256",
        "288",
        "320",
        "352",
        "384",
        "448",
        "512",
        "640",
        "768",
        "896",
        "1024",
        "1280",
        "1536",
        "2048",
        "2560",
        "3072",
        "3584",
        "4096",
        "4608",
        "5120",
    ],
    # pyformat: enable
    "Strictly increasing order of token sizes for which to cache compilations."
    " For any input with more tokens than the largest bucket size, a new bucket"
    " is created for exactly that number of tokens.",
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    "flash_attention_implementation",
    default="triton",
    enum_values=["triton", "cudnn", "xla"],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        " Triton and cuDNN flash attention implementation, respectively. The"
        " Triton kernel is fastest and has been tested more thoroughly. The"
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        " XLA attention implementation (no flash attention) and is portable"
        " across GPU devices."
    ),
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    "num_diffusion_samples",
    5,
    "Number of diffusion samples to generate.",
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    "save_embeddings",
    False,
    "Whether to save the final trunk single and pair embeddings in the output.",
)

_WRITE_ALL_STRUCTS = flags.DEFINE_bool(
    "write_all_structs",
    True,
    "If False, only write the top-ranked diffusion sample to disk; "
    "otherwise also write every sample under seed-*_sample-* subdirectories. "
    "The top-level <name>_model.cif (highest ranking_score) is always written.",
)

_MAKE_MSAS = flags.DEFINE_bool(
    "make_msas",
    False,
    "do we want to make msas?.",
)

_SKIP_EXISTING = flags.DEFINE_bool(
    "skip_existing",
    True,
    "Whether to skip jobs that already have completed output files in the output directory.",
)


def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = "triton",
    num_diffusion_samples: int = 5,
    return_embeddings: bool = False,
) -> model.Model.Config:
    """Returns a model config with some defaults overridden."""
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.return_embeddings = return_embeddings
    return config


# Use the residue map above
_RESIDUE_MAP = {
    # Protein
    "ALA": ("protein", "A"),
    "ARG": ("protein", "R"),
    "ASN": ("protein", "N"),
    "ASP": ("protein", "D"),
    "CYS": ("protein", "C"),
    "GLN": ("protein", "Q"),
    "GLU": ("protein", "E"),
    "GLY": ("protein", "G"),
    "HIS": ("protein", "H"),
    "ILE": ("protein", "I"),
    "LEU": ("protein", "L"),
    "LYS": ("protein", "K"),
    "MET": ("protein", "M"),
    "PHE": ("protein", "F"),
    "PRO": ("protein", "P"),
    "SER": ("protein", "S"),
    "THR": ("protein", "T"),
    "TRP": ("protein", "W"),
    "TYR": ("protein", "Y"),
    "VAL": ("protein", "V"),
    "MSE": ("protein", "M"),  # Selenomethionine
    # RNA
    "A": ("rna", "A"),
    "U": ("rna", "U"),
    "G": ("rna", "G"),
    "C": ("rna", "C"),
    # DNA
    "DA": ("dna", "A"),
    "DT": ("dna", "T"),
    "DG": ("dna", "G"),
    "DC": ("dna", "C"),
}


def _resolve_path_or_string(value: str) -> str:
    """Read a path-valued payload, otherwise return its inline contents."""
    if "\n" not in value and os.path.exists(value):
        return pathlib.Path(value).read_text()
    return value


def build_fold_input_from_cif(
    *,
    cif_string: str,
    cif_name: str,
    paired_msa: dict[str, str] | None = None,
    unpaired_msa: dict[str, str] | None = None,
    template: dict[str, typing.Any] | None = None,
    user_ccd: str | None = None,
    rng_seeds: Sequence[int] = (1,),
) -> folding_input.Input:
    """Create an AF3 input from mmCIF plus optional per-chain overrides."""
    user_ccd_string = (
        _resolve_path_or_string(user_ccd) if user_ccd is not None else None
    )
    ccd = chemical_components.Ccd(user_ccd=user_ccd_string)
    fold_input = folding_input.Input.from_mmcif(cif_string, ccd=ccd)
    fold_input = dataclasses.replace(
        fold_input,
        name=cif_name,
        user_ccd=user_ccd_string,
        rng_seeds=tuple(int(seed) for seed in rng_seeds),
    )

    paired_by_chain = {
        chain_id: _resolve_path_or_string(value)
        for chain_id, value in (paired_msa or {}).items()
    }
    unpaired_by_chain = {
        chain_id: _resolve_path_or_string(value)
        for chain_id, value in (unpaired_msa or {}).items()
    }
    template_by_chain: dict[str, dict[str, typing.Any]] = {}
    for chain_id, value in (template or {}).items():
        if isinstance(value, str):
            template_by_chain[chain_id] = {
                "mmcif": _resolve_path_or_string(value),
                "query_indices": None,
                "template_indices": None,
            }
        else:
            template_by_chain[chain_id] = {
                "mmcif": _resolve_path_or_string(value["mmcif"]),
                "query_indices": value.get("query_indices"),
                "template_indices": value.get("template_indices"),
            }

    protein_chain_ids = {chain.id for chain in fold_input.protein_chains}
    rna_chain_ids = {chain.id for chain in fold_input.rna_chains}
    invalid_overrides = {
        "paired_msa": set(paired_by_chain) - protein_chain_ids,
        "unpaired_msa": set(unpaired_by_chain) - (protein_chain_ids | rna_chain_ids),
        "template": set(template_by_chain) - protein_chain_ids,
    }
    invalid_overrides = {
        key: sorted(value) for key, value in invalid_overrides.items() if value
    }
    if invalid_overrides:
        raise ValueError(
            "AF3 mmCIF chain overrides do not match parsed polymer chains: "
            f"{invalid_overrides}. The mmCIF parser may have dropped or "
            "reclassified an input chain."
        )

    updated_chains = []
    for chain in fold_input.chains:
        if not isinstance(chain, folding_input.ProteinChain):
            updated_chains.append(chain)
            continue

        templates = ()
        if chain.id in template_by_chain:
            template_value = template_by_chain[chain.id]
            query_indices = template_value["query_indices"]
            template_indices = template_value["template_indices"]
            if query_indices is None:
                query_to_template = dict(enumerate(range(len(chain.sequence))))
            else:
                if template_indices is None:
                    raise ValueError(
                        f"Template for chain {chain.id!r} has query_indices but no "
                        "template_indices"
                    )
                query_to_template = dict(
                    zip(query_indices, template_indices, strict=True)
                )
            templates = (
                folding_input.Template(
                    mmcif=template_value["mmcif"],
                    query_to_template_map=query_to_template,
                ),
            )
        chain._templates = templates
        ptm_positions = [position for _, position in chain.ptms]
        paired = paired_by_chain.get(chain.id, "")
        unpaired = unpaired_by_chain.get(chain.id, "")
        chain._paired_msa = rewrite_a3m_query_for_af3(
            paired,
            query_sequence=chain.sequence,
            ptm_positions=ptm_positions,
        )
        chain._unpaired_msa = rewrite_a3m_query_for_af3(
            unpaired,
            query_sequence=chain.sequence,
            ptm_positions=ptm_positions,
        )
        updated_chains.append(chain)

    fold_input = dataclasses.replace(fold_input, chains=updated_chains)
    for chain in fold_input.rna_chains:
        if chain.unpaired_msa is None:
            chain._unpaired_msa = unpaired_by_chain.get(chain.id, "")
    return fold_input


def load_fold_inputs_from_manifest(manifest_path: pathlib.Path):
    """Load Chaperone mmCIF manifest entries as native AF3 inputs."""
    payloads = json.loads(manifest_path.read_text())
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"AF3 input manifest is empty or invalid: {manifest_path}")

    fold_inputs = []
    for payload in payloads:
        cif_value = payload["cif"]
        cif_string = _resolve_path_or_string(cif_value)
        seed = payload.get("seed", 1)
        fold_inputs.append(
            build_fold_input_from_cif(
                cif_string=cif_string,
                cif_name=payload["name"],
                paired_msa=payload.get("paired_msa"),
                unpaired_msa=payload.get("unpaired_msa"),
                template=payload.get("template"),
                user_ccd=payload.get("user_ccd"),
                rng_seeds=(int(seed),),
            )
        )
    return fold_inputs


def load_fold_inputs_from_pdb(
    pdb_path: pathlib.Path,
    model_seeds: list[str] | None,
) -> list[folding_input.Input]:
    """
    Parse a PDB file line-by-line to extract protein, RNA, and DNA chain sequences
    without using BioPython. Returns a list containing one folding_input.Input
    object with zero or more Chain objects (ProteinChain, RnaChain, DnaChain).
    """
    pdb_path = str(pdb_path)
    if "," in pdb_path:
        pdb_paths = pdb_path.split(",")
    else:
        pdb_paths = [pdb_path]
    fold_inputs = []
    for pdb_path in pdb_paths:
        pdb_path = pathlib.Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # For each chain ID, we'll store a list of (res_num_str, biomol_type, one_letter_code).
        chain_residues = {}
        # Also track a single biomol_type per chain (protein vs rna vs dna)
        chain_types = {}

        with open(pdb_path, "r") as f:
            for line in f:
                record_type = line[0:6].strip()
                # We only read ATOM lines.
                # If your nucleotides might appear in HETATM, you could also allow record_type == 'HETATM'.
                if record_type != "ATOM":
                    continue

                res_name = line[17:20].strip()  # columns 17-20
                chain_id = line[21].strip()  # column 21
                res_num = line[22:26].strip()  # columns 22-26

                # See if we recognize the residue
                if res_name in _RESIDUE_MAP:
                    biomol_type, one_letter = _RESIDUE_MAP[res_name]

                    if chain_id not in chain_residues:
                        chain_residues[chain_id] = []
                        chain_types[chain_id] = biomol_type

                    # If we see a conflicting residue type in the same chain, raise an error
                    if chain_types[chain_id] != biomol_type:
                        raise ValueError(
                            f"Chain {chain_id} has both {chain_types[chain_id]} and {biomol_type} residues."
                        )

                    # Record the residue number + letter only if new
                    existing = chain_residues[chain_id]
                    if not existing or existing[-1][0] != res_num:
                        existing.append((res_num, biomol_type, one_letter))

        # Build sequences, one per chain
        sequences_by_chain = {}
        for chain_id, residues in chain_residues.items():
            # Sort them by residue number.
            # This is naive for insertion codes (e.g. "12A"), but you can customize if needed.
            def parse_resnum(rnum_str):
                try:
                    return int(rnum_str)
                except ValueError:
                    # insertion code or something else
                    return 999999

            sorted_residues = sorted(residues, key=lambda x: parse_resnum(x[0]))
            # Join just the one-letter codes for the final sequence
            seq_str = "".join(item[2] for item in sorted_residues)

            sequences_by_chain[chain_id] = (chain_types[chain_id], seq_str)

        # Create Chain objects (ProteinChain, RnaChain, or DnaChain)
        chain_objects = []
        for i, (chain_id, (chain_type, seq_str)) in enumerate(
            sequences_by_chain.items(), start=1
        ):
            # The "id" in these classes must be uppercase letters, per folding_input.py requirement
            # If chain_id is not guaranteed uppercase, we might force it:
            chain_label = chain_id.upper() or f"X{i}"  # fallback if chain_id is empty

            if chain_type == "protein":
                # We skip modifications or MSAs for simplicity:
                ch = folding_input.ProteinChain(
                    id=chain_label,
                    sequence=seq_str,
                    ptms=[],  # no post-translational modifications
                    paired_msa="",  # or empty string if you want
                    unpaired_msa="",  # or empty string
                    templates=[],  # or []
                    # paired_msa=None,    # or empty string if you want
                    # unpaired_msa=None,  # or empty string
                    # templates=None      # or []
                )
                chain_objects.append(ch)

            elif chain_type == "rna":
                ch = folding_input.RnaChain(
                    id=chain_label,
                    sequence=seq_str,
                    modifications=[],
                    unpaired_msa="",  # or ''
                    # unpaired_msa=None,  # or ''
                )
                chain_objects.append(ch)

            elif chain_type == "dna":
                ch = folding_input.DnaChain(
                    id=chain_label,
                    sequence=seq_str,
                    modifications=[],
                )
                chain_objects.append(ch)

            else:
                # Unexpected biomol_type
                raise ValueError(f"Unknown biomolecule type: {chain_type}")

        # Finally, create a single Input object containing all chain objects
        # If we have no chain objects, you can raise an error or return an empty list
        if not chain_objects:
            raise ValueError(
                f"No protein, DNA, or RNA chains recognized in {pdb_path}."
            )

        # # Provide at least one RNG seed (AlphaFold 3 requires seeds).
        # # You can randomize or just pick one:
        # rng_seeds = [42]
        rng_seeds = model_seeds or [42]

        fold_input = folding_input.Input(
            name=pdb_path.stem,
            chains=chain_objects,
            rng_seeds=rng_seeds,
            bonded_atom_pairs=None,
            user_ccd=None,
        )
        fold_inputs.append(fold_input)

    return fold_inputs


def load_fold_inputs_from_commandline_lists(
    protein_seqs: list[str] | None,
    rna_seqs: list[str] | None,
    dna_seqs: list[str] | None,
    ion_list: list[str] | None,
    job_name: str | None,
    model_seeds: list[str] | None,
    make_msas=False,
) -> list[folding_input.Input]:
    """
    Create a single folding_input.Input from user-provided protein/RNA/DNA seqs.
    Returns a list with one Input.
    """

    # Convert model_seeds from strings to int. If none, generate or fallback:
    if model_seeds:
        rng_seeds = [int(s) for s in model_seeds]
    else:
        # Provide at least one seed (AlphaFold 3 requires at least one).
        rng_seeds = [42]

    chains = []

    # We'll create a unique chain ID for each sequence.
    # E.g., 'A', 'B', 'C'... up to 26. If we have more, we can do something fancier.
    # For simplicity, let's do chain IDs like A, B, C ... Z, then AA, AB, ...
    # (like Excel columns).
    def int_to_chain_id(i: int) -> str:
        """
        Convert a zero-based integer into an uppercase chain ID.
        0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, etc.
        """
        letters = []
        while True:
            letters.append(string.ascii_uppercase[i % 26])
            i //= 26
            if i < 1:
                break
            i -= 1
        return "".join(reversed(letters))

    chain_index = 0

    # Protein sequences -> ProteinChain
    if protein_seqs:
        for seq in protein_seqs:
            chain_id = int_to_chain_id(chain_index)
            chain_index += 1
            if make_msas:
                chains.append(
                    folding_input.ProteinChain(
                        id=chain_id, sequence=seq.strip(), ptms=[]
                    )
                )
            else:
                chains.append(
                    folding_input.ProteinChain(
                        id=chain_id,
                        sequence=seq.strip(),
                        ptms=[],  # no modifications
                        paired_msa="",  # optional
                        unpaired_msa="",  # optional
                        templates=[],  # optional
                    )
                )

    # RNA sequences -> RnaChain
    if rna_seqs:
        for seq in rna_seqs:
            chain_id = int_to_chain_id(chain_index)
            chain_index += 1
            if make_msas:
                chains.append(
                    folding_input.RnaChain(
                        id=chain_id,
                        sequence=seq.strip(),
                        modifications=[],  # no modifications
                    )
                )
            else:
                chains.append(
                    folding_input.RnaChain(
                        id=chain_id,
                        sequence=seq.strip(),
                        modifications=[],  # no modifications
                        unpaired_msa="",  # optional
                    )
                )

    # DNA sequences -> DnaChain
    if dna_seqs:
        for seq in dna_seqs:
            chain_id = int_to_chain_id(chain_index)
            chain_index += 1
            chains.append(
                folding_input.DnaChain(
                    id=chain_id,
                    sequence=seq.strip(),
                    modifications=[],
                )
            )

    # ion lists -> Ligand
    if ion_list:
        chain_id = int_to_chain_id(chain_index)
        chain_index += 1
        ccd_ion_list = []
        for ion_spec in ion_list:
            ion_name, count = ion_spec.split(":")
            ccd_ion_list.extend([ion_name] * int(count))

        chains.append(
            folding_input.Ligand(
                id=chain_id,
                ccd_ids=ccd_ion_list,
            )
        )

    if not chains:
        raise ValueError(
            "No protein/RNA/DNA sequences provided. At least one sequence is required."
        )

    fold_name = job_name or f"af3_fold_{random.randint(0, 999999):06d}"

    rng_seeds = model_seeds or [42]

    # Create a single Input object
    fold_input = folding_input.Input(
        name=job_name,
        chains=chains,
        rng_seeds=rng_seeds,
        bonded_atom_pairs=None,
        user_ccd=None,
    )

    return [fold_input]


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        config: model.Model.Config,
        device: jax.Device,
        model_dir: pathlib.Path,
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """Loads model parameters from the model directory."""
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(
        self,
    ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""

        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), self.model_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> model.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self._device,
        )

        result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params["__meta__"]["__identifier__"].tobytes()
        result["__identifier__"] = identifier
        return result

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def extract_embeddings(
        self,
        result: model.ModelResult,
    ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if "single_embeddings" in result:
            embeddings["single_embeddings"] = result["single_embeddings"]
        if "pair_embeddings" in result:
            embeddings["pair_embeddings"] = result["pair_embeddings"]
        return embeddings or None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
      embeddings: The final trunk single and pair embeddings, if requested.
    """

    seed: int
    inference_results: Sequence[model.InferenceResult]
    full_fold_input: folding_input.Input
    embeddings: dict[str, np.ndarray] | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    print(f"Featurising data for seeds {fold_input.rng_seeds}...")
    featurisation_start_time = time.time()
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=True,
        conformer_max_iterations=conformer_max_iterations,
    )
    print(
        f"Featurising data for seeds {fold_input.rng_seeds} took "
        f" {time.time() - featurisation_start_time:.2f} seconds."
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f"Running model inference for seed {seed}...")
        inference_start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)
        result = model_runner.run_inference(example, rng_key)
        print(
            f"Running model inference for seed {seed} took "
            f" {time.time() - inference_start_time:.2f} seconds."
        )
        print(f"Extracting output structures (one per sample) for seed {seed}...")
        extract_structures = time.time()
        inference_results = model_runner.extract_structures(
            batch=example, result=result, target_name=fold_input.name
        )
        print(
            f"Extracting output structures (one per sample) for seed {seed} took "
            f" {time.time() - extract_structures:.2f} seconds."
        )

        embeddings = model_runner.extract_embeddings(result)

        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
                embeddings=embeddings,
            )
        )
        print(
            "Running model inference and extracting output structures for seed"
            f" {seed} took {time.time() - inference_start_time:.2f} seconds."
        )
    print(
        "Running model inference and extracting output structures for seeds"
        f" {fold_input.rng_seeds} took"
        f" {time.time() - all_inference_start_time:.2f} seconds."
    )
    return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"{fold_input.sanitised_name()}_data.json"), "wt"
    ) as f:
        f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / "OUTPUT_TERMS_OF_USE.md"
    ).read_text()

    os.makedirs(output_dir, exist_ok=True)
    write_all = _WRITE_ALL_STRUCTS.value
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            if write_all:
                sample_dir = os.path.join(
                    output_dir, f"seed-{seed}_sample-{sample_idx}"
                )
                os.makedirs(sample_dir, exist_ok=True)
                output_name = f"{job_name}_model_{sample_idx}"
                post_processing.write_output(
                    inference_result=result,
                    output_dir=sample_dir,
                    name=output_name,
                )
            ranking_score = float(result.metadata["ranking_score"])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

        if embeddings := results_for_seed.embeddings:
            embeddings_dir = os.path.join(output_dir, f"seed-{seed}_embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            post_processing.write_embeddings(
                embeddings=embeddings, output_dir=embeddings_dir
            )

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(os.path.join(output_dir, "ranking_scores.csv"), "wt") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "sample", "ranking_score"])
            writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    skip_existing: bool = True,
) -> folding_input.Input | list: ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    skip_existing: bool = True,
) -> Sequence[ResultsForSeed] | list: ...


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
    """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
    template = string.Template(path_with_db_dir)
    if "DB_DIR" in template.get_identifiers():
        for db_dir in db_dirs:
            path = template.substitute(DB_DIR=db_dir)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}."
        )
    if not os.path.exists(path_with_db_dir):
        raise FileNotFoundError(f"{path_with_db_dir} does not exist.")
    return path_with_db_dir


def _check_job_already_completed(
    output_dir: os.PathLike[str] | str,
    fold_input_name: str,
) -> bool:
    """Checks if a job has already been completed in the output directory.

    A job is considered complete if the output directory exists and contains:
      - ranking_scores.csv
      - The best-ranked CIF file (fold_input_name_model.cif)

    Args:
      output_dir: Output directory to check.
      fold_input_name: Sanitized name of the fold input.

    Returns:
      True if the job appears to be already completed, False otherwise.
    """
    if not os.path.exists(output_dir):
        return False
    # Check for expected output files
    ranking_scores_file = os.path.join(output_dir, "ranking_scores.csv")
    best_cif_file = os.path.join(output_dir, f"{fold_input_name}_model.cif")

    if os.path.isfile(ranking_scores_file) and os.path.isfile(best_cif_file):
        return True

    return False


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
    skip_existing: bool = True,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.
      conformer_max_iterations: Optional override for maximum number of iterations
        to run for RDKit conformer search.
      skip_existing: If True, skip processing if output files already exist.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    print(f"Processing fold input {fold_input.name}")

    if not fold_input.chains:
        raise ValueError("Fold input has no chains.")

    # Check if job is already completed
    if (
        skip_existing
        and _check_job_already_completed(output_dir, fold_input.sanitised_name())
        or os.path.exists(output_dir)
    ):
        print(
            f"Output directory {output_dir} already contains completed results. "
            "Skipping this job."
        )
        # Return empty results to indicate skipping
        return []

    if not skip_existing and os.path.exists(output_dir) and os.listdir(output_dir):
        new_output_dir = (
            f"{output_dir}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(
            f"Output directory {output_dir} exists and non-empty, using instead "
            f" {new_output_dir}."
        )
        output_dir = new_output_dir

    if model_runner is not None:
        # If we're running inference, check we can load the model parameters before
        # (possibly) launching the data pipeline.
        print("Checking we can load the model parameters...")
        _ = model_runner.model_params

    if data_pipeline_config is None:
        print("Skipping data pipeline...")
    else:
        print("Running data pipeline...")
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

    print(f"Output directory: {output_dir}")
    print(f"Writing model input JSON to {output_dir}")
    write_fold_input_json(fold_input, output_dir)
    if model_runner is None:
        print("Skipping inference...")
        output = fold_input
    else:
        print(
            f"Predicting 3D structure for {fold_input.name} for seed(s)"
            f" {fold_input.rng_seeds}..."
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            conformer_max_iterations=conformer_max_iterations,
        )
        print(
            f"Writing outputs for {fold_input.name} for seed(s)"
            f" {fold_input.rng_seeds}..."
        )
        write_outputs(
            all_inference_results=all_inference_results,
            output_dir=output_dir,
            job_name=fold_input.sanitised_name(),
        )
        output = all_inference_results

    print(f"Done processing fold input {fold_input.name}.")
    return output


def main(_):
    if _JAX_COMPILATION_CACHE_DIR.value is not None:
        jax.config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR.value)

    # Exactly one file-based or direct-sequence input mode is allowed.
    input_modes_used = 0
    if _JSON_PATH.value is not None:
        input_modes_used += 1
    if _INPUT_DIR.value is not None:
        input_modes_used += 1
    if _PDB_PATH.value is not None:
        input_modes_used += 1
    if _INPUT_MANIFEST.value is not None:
        input_modes_used += 1

    direct_seq_mode = False
    if (
        (_PROTEIN_SEQS.value is not None and len(_PROTEIN_SEQS.value) > 0)
        or (_RNA_SEQS.value is not None and len(_RNA_SEQS.value) > 0)
        or (_DNA_SEQS.value is not None and len(_DNA_SEQS.value) > 0)
        or (_ION_LIST.value is not None and len(_ION_LIST.value) > 0)
    ):
        direct_seq_mode = True
        input_modes_used += 1

    if input_modes_used != 1:
        raise ValueError(
            "Exactly one of the following must be specified:\n"
            "  --pdb_path OR --json_path OR --input_dir OR --input_manifest OR "
            "  (any of --protein_seqs/--rna_seqs/--dna_seqs)."
        )

    if _PDB_PATH.value is not None:
        fold_inputs = load_fold_inputs_from_pdb(
            pdb_path=pathlib.Path(_PDB_PATH.value),
            model_seeds=_MODEL_SEEDS.value,
        )
    elif _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    elif _INPUT_MANIFEST.value is not None:
        fold_inputs = load_fold_inputs_from_manifest(
            pathlib.Path(_INPUT_MANIFEST.value)
        )
    elif direct_seq_mode:
        fold_inputs = load_fold_inputs_from_commandline_lists(
            protein_seqs=_PROTEIN_SEQS.value,
            rna_seqs=_RNA_SEQS.value,
            dna_seqs=_DNA_SEQS.value,
            ion_list=_ION_LIST.value,
            model_seeds=_MODEL_SEEDS.value,
            job_name=_NAME.value,
            make_msas=_MAKE_MSAS.value,
        )
    else:
        raise ValueError(
            "No valid input mode found. Provide either --pdb_path, "
            "--json_path, --input_dir, --input_manifest, or sequences via --protein_seqs, "
            "--rna_seqs, --dna_seqs."
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            "At least one of --run_inference or --run_data_pipeline must be"
            " set to true."
        )

    # Make sure we can create the output directory before running anything.
    try:
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f"Failed to create output directory {_OUTPUT_DIR.value}: {e}")
        raise

    if _RUN_INFERENCE.value:
        # Fail early on incompatible devices, but only if we're running inference.
        gpu_devices = jax.local_devices(backend="gpu")
        if gpu_devices:
            compute_capability = float(gpu_devices[0].compute_capability)
            if compute_capability < 6.0:
                raise ValueError(
                    "AlphaFold 3 requires at least GPU compute capability 6.0 (see"
                    " https://developer.nvidia.com/cuda-gpus)."
                )
            elif 7.0 <= compute_capability < 8.0:
                xla_flags = os.environ.get("XLA_FLAGS")
                required_flag = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
                if not xla_flags or required_flag not in xla_flags:
                    raise ValueError(
                        "For devices with GPU compute capability 7.x (see"
                        " https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must"
                        f' include "{required_flag}".'
                    )

    notice = textwrap.wrap(
        "Running AlphaFold 3. Please note that standard AlphaFold 3 model"
        " parameters are only available under terms of use provided at"
        " https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md."
        " If you do not agree to these terms and are using AlphaFold 3 derived"
        " model parameters, cancel execution of AlphaFold 3 inference with"
        " CTRL-C, and do not use the model parameters.",
        break_long_words=False,
        break_on_hyphens=False,
        width=80,
    )
    print("\n".join(notice))

    if _RUN_DATA_PIPELINE.value:
        expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
        max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
        data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
            nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
            hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
            hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
            hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
            small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
            mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
            uniprot_cluster_annot_database_path=expand_path(
                _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
            ),
            uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
            ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
            rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
            rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
            pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
            seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
            jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
            nhmmer_n_cpu=_NHMMER_N_CPU.value,
            max_template_date=max_template_date,
        )
    else:
        print("Skipping running the data pipeline.")
        data_pipeline_config = None

    if _RUN_INFERENCE.value:
        devices = jax.local_devices(backend="gpu")
        print(f"Found local devices: {devices}")

        print("Building model from scratch...")
        model_runner = ModelRunner(
            config=make_model_config(
                flash_attention_implementation=typing.cast(
                    tokamax.DotProductAttentionImplementation,
                    _FLASH_ATTENTION_IMPLEMENTATION.value,
                ),
                num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
                return_embeddings=_SAVE_EMBEDDINGS.value,
            ),
            device=devices[0],
            model_dir=pathlib.Path(MODEL_DIR.value),
        )
    else:
        print("Skipping running model inference.")
        model_runner = None

    print("Processing fold inputs.")
    num_fold_inputs = 0
    num_skipped = 0
    for fold_input in fold_inputs:
        print(f"Processing fold input #{num_fold_inputs + 1}")
        result = process_fold_input(
            fold_input=fold_input,
            data_pipeline_config=data_pipeline_config,
            model_runner=model_runner,
            output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
            buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
            conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
            skip_existing=_SKIP_EXISTING.value,
        )
        num_fold_inputs += 1
        # Check if job was skipped
        if isinstance(result, list) and len(result) == 0:
            num_skipped += 1

    print(f"Done processing {num_fold_inputs} fold inputs ({num_skipped} skipped).")


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "output_dir",
        ]
    )
    app.run(main)
