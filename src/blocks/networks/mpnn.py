# MPNN block for inverse folding, backed by LigandMPNN
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from atomworks.io.utils.chain import create_chain_id_generator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chaperone import (
    CompiledBlockFunction,
    PipelineDataHandler,
    JobArgs,
)
from chaperone.block_config import TIER_PRIMARY
from chaperone.containers import Image
from chaperone.data import make_row

from blocks.networks.mpnn_io import (
    load_manifest,
    manifest_entry_for_fasta,
    parse_fasta,
    sequence_row,
    write_manifest,
)

from chaperone.common import glob_designs_in_dir

logger = logging.getLogger(__name__)

_PUBLIC_MPNN_REVISION = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
_PUBLIC_MPNN_ROOT = "/opt/ligandmpnn"
# Every official checkpoint of the pinned revision (get_model_params.sh), keyed by
# model_type, then by the backbone noise (Angstrom) the model was trained with:
# {filename: sha256}. All are baked into the image and verified at build time.
# The first group is also published in dauparas/ProteinMPNN; the rest come from
# files.ipd.uw.edu, the host get_model_params.sh downloads from.
_PUBLIC_MPNN_CHECKPOINTS = {
    "protein_mpnn": {
        0.02: ("proteinmpnn_v_48_002.pt", "925f2ca1007bf9b02e0e7f420ff00eb91f50fcc2722f64b42e644ae95adaa131"),
        0.10: ("proteinmpnn_v_48_010.pt", "db866fae956a28661f926053d630610c55e9fc4bc03922f2aeeb98a37435ccce"),
        0.20: ("proteinmpnn_v_48_020.pt", "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"),
        0.30: ("proteinmpnn_v_48_030.pt", "c34b7bfb38418ea30989fda3314f4781ac4e3920f9825731cf555f1fed44ac66"),
    },
    "ligand_mpnn": {
        0.05: ("ligandmpnn_v_32_005_25.pt", "ee07a7afb53bce98a0b1d33996bbe3b46b0831df5ce0684d3cbd55eed1aa9263"),
        0.10: ("ligandmpnn_v_32_010_25.pt", "161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970"),
        0.20: ("ligandmpnn_v_32_020_25.pt", "42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981"),
        0.30: ("ligandmpnn_v_32_030_25.pt", "ed13e130787f70b6c385efba56949220d16a1d1a98073ada3002b3701a6a8ecf"),
    },
    "soluble_mpnn": {
        0.02: ("solublempnn_v_48_002.pt", "0877f840978fe770be6fcec025784d8f50c438571db3260c05e41aa207a7c448"),
        0.10: ("solublempnn_v_48_010.pt", "79562f7444f72c84595a1c96010713864865a616f4f3967633493041e169fa6e"),
        0.20: ("solublempnn_v_48_020.pt", "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a"),
        0.30: ("solublempnn_v_48_030.pt", "1dd63f1e9fc68a133cc9ef859edf43b489e5ac581cb5624e0b9ec848ff062421"),
    },
    "per_residue_label_membrane_mpnn": {
        0.20: ("per_residue_label_membrane_mpnn_v_48_020.pt", "1e62e193bee12f1b64fff87ae3ca9da95b0f60b15839ae3cbe48e4983ef055ba"),
    },
    "global_label_membrane_mpnn": {
        0.20: ("global_label_membrane_mpnn_v_48_020.pt", "89ef0abddbfb956c4c7c02dcba6523c0b5152733055f7509292365eaecc38b21"),
    },
}
# The upstream run.py defaults, used when noise_level is not given.
_PUBLIC_MPNN_DEFAULT_NOISE = {
    "protein_mpnn": 0.20,
    "ligand_mpnn": 0.10,
    "soluble_mpnn": 0.20,
    "per_residue_label_membrane_mpnn": 0.20,
    "global_label_membrane_mpnn": 0.20,
}
_PUBLIC_MPNN_PACKER = (
    "ligandmpnn_sc_v_32_002_16.pt",
    "799a9de6c1c72bb0c7ceb37998391a6fd0e3e21cb42928ee31a28313a3b1b46a",
)
_PUBLIC_MPNN_ALL_CHECKPOINTS = [
    entry
    for versions in _PUBLIC_MPNN_CHECKPOINTS.values()
    for entry in versions.values()
] + [_PUBLIC_MPNN_PACKER]


def _available_versions() -> str:
    return "; ".join(
        f"{model_type}: {', '.join(f'{noise:g}' for noise in versions)}"
        for model_type, versions in _PUBLIC_MPNN_CHECKPOINTS.items()
    )


def _public_checkpoint(model_type: str, noise_level: float | None = None) -> str:
    """Image path of the checkpoint for ``model_type`` at ``noise_level``.

    Raises ``ValueError`` listing every valid combination if there is none.
    """
    versions = _PUBLIC_MPNN_CHECKPOINTS.get(model_type)
    if versions is None:
        raise ValueError(
            f"Unknown MPNN model_type {model_type!r}; "
            f"choose one of {sorted(_PUBLIC_MPNN_CHECKPOINTS)}."
        )
    noise = _PUBLIC_MPNN_DEFAULT_NOISE[model_type] if noise_level is None else round(noise_level, 2)
    if noise not in versions:
        raise ValueError(
            f"model_type={model_type!r} has no checkpoint with noise_level={noise_level}. "
            f"Available noise_level values by model_type: {_available_versions()}."
        )
    return f"{_PUBLIC_MPNN_ROOT}/model_params/{versions[noise][0]}"


_PUBLIC_MPNN_PACKER_PATH = f"{_PUBLIC_MPNN_ROOT}/model_params/{_PUBLIC_MPNN_PACKER[0]}"


# These removed NumPy aliases were aliases of the corresponding Python types.
# Pin expected replacements so an upstream source change fails the build gate.
_PUBLIC_MPNN_NUMPY_COMPATIBILITY_PATCH = f"""
from pathlib import Path
import re
root = Path({_PUBLIC_MPNN_ROOT!r}) / "openfold"
for filename, alias, expected in [
    ("np/residue_constants.py", "int", 3),
    ("data/templates.py", "object", 2),
    ("np/relax/utils.py", "bool", 1),
]:
    path = root / filename
    source = path.read_text()
    patched, count = re.subn(r"\\bnp\\." + alias + r"\\b", alias, source)
    if count != expected:
        raise RuntimeError(f"Unexpected NumPy alias count in {{path}}: {{count}}, expected {{expected}}")
    path.write_text(patched)
"""


@dataclass(frozen=True)
class ChainRename:
    """Single-letter stand-ins for the multi-character chain ids of one structure.

    RFD3 names unindexed residues with chain ids such as ``X1``, which the PDB format
    MPNN reads and writes cannot hold. The structure is handed to MPNN with those
    chains renamed (``rename``), fixed-position strings follow the rename (``apply``),
    and the outputs get the original ids back, so the block's outputs use the same
    chain ids as its inputs.
    """

    mapping: dict[str, str] = field(default_factory=dict)  # original -> stand-in
    # (chain, res_id) of kept chains whose id is a prefix of a renamed one, such as
    # chain "X" beside "X1". Only they make a position like "X15" ambiguous: it could
    # be chain X residue 15 or chain X1 residue 5.
    ambiguous: frozenset = frozenset()

    def __bool__(self) -> bool:
        return bool(self.mapping)

    @classmethod
    def for_atoms(cls, atoms) -> "ChainRename":
        """The rename a structure needs, empty if its chain ids are all one letter."""
        chains = list(dict.fromkeys(str(chain) for chain in atoms.chain_id))
        renamed = [chain for chain in chains if len(chain) > 1]
        if not renamed:
            return cls()
        kept = {chain for chain in chains if len(chain) == 1}
        free_ids = create_chain_id_generator(unavailable_chain_ids=kept)
        mapping = {}
        for old in renamed:
            new = next(free_ids)
            if len(new) > 1:
                raise ValueError(
                    f"Cannot rename chains {renamed}: not enough unused "
                    "single-character chain ids."
                )
            mapping[old] = new
        prefixes = {k for k in kept if any(old.startswith(k) for old in renamed)}
        ambiguous = frozenset(
            (str(chain), int(res_id))
            for chain, res_id in zip(atoms.chain_id, atoms.res_id)
            if str(chain) in prefixes
        )
        return cls(mapping, ambiguous)

    def rename(self, atoms):
        """A copy of ``atoms`` with the multi-character chains renamed."""
        renamed = atoms.copy()
        for old, new in self.mapping.items():
            renamed.chain_id[atoms.chain_id == old] = new
        return renamed

    def apply(self, position: str) -> str:
        """Follow the rename in a residue id such as ``X15`` (chain ``X1``, residue 5)."""
        for old in sorted(self.mapping, key=len, reverse=True):
            number = position[len(old) :]
            if not position.startswith(old) or not re.fullmatch(r"-?\d+", number):
                continue
            kept_chain, kept_number = position[0], position[1:]
            if (
                re.fullmatch(r"-?\d+", kept_number)
                and (kept_chain, int(kept_number)) in self.ambiguous
            ):
                return position
            return self.mapping[old] + number
        return position

    def restorable_in(self, chain_ids) -> bool:
        """True if a file looks like MPNN's output for this rename.

        Every stand-in is present and no original id is, so an output that merely
        shares a name prefix with a renamed input is never rewritten.
        """
        present = {str(chain) for chain in chain_ids}
        return set(self.mapping.values()) <= present and not set(self.mapping) & present


# ── Container ───────────────────────────────────────────────────────────
#
# LigandMPNN (dauparas/LigandMPNN, MIT) built from source with every official
# checkpoint baked in, so a model_type / noise_level choice never needs a weights
# directory on the host. Install it with ``chaperone install ligandmpnn_public``.

LIGANDMPNN_PUBLIC_CONTAINER = (
    Image("ligandmpnn_public")
    .base_image("ubuntu:24.04")
    .packages("build-essential", "git", "wget", "ca-certificates", "python3.12", "python3.12-dev", "python3.12-venv")
    .run(
        "python3.12 -m venv /opt/mpnn-env",
        "/opt/mpnn-env/bin/pip install --upgrade pip setuptools wheel",
        "/opt/mpnn-env/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128",
        "/opt/mpnn-env/bin/pip install numpy==1.26.4 prody==2.6.1 scipy biopython ml-collections dm-tree",
    )
    .git_clone("https://github.com/dauparas/LigandMPNN", _PUBLIC_MPNN_ROOT)
    .run(
        f"git -C {_PUBLIC_MPNN_ROOT} checkout {_PUBLIC_MPNN_REVISION}",
        "/opt/mpnn-env/bin/python -c " + shlex.quote(_PUBLIC_MPNN_NUMPY_COMPATIBILITY_PATCH),
        f"mkdir -p {_PUBLIC_MPNN_ROOT}/model_params",
        *[
            f"wget -q https://files.ipd.uw.edu/pub/ligandmpnn/{filename} "
            f"-O {_PUBLIC_MPNN_ROOT}/model_params/{filename} && "
            f"echo '{checksum}  {_PUBLIC_MPNN_ROOT}/model_params/{filename}' | sha256sum -c -"
            for filename, checksum in _PUBLIC_MPNN_ALL_CHECKPOINTS
        ],
        "/opt/mpnn-env/bin/pip check",
        f"cd {_PUBLIC_MPNN_ROOT} && /opt/mpnn-env/bin/python -c "
        "'import run, data_utils, model_utils, sc_utils; import pathlib, torch; "
        "[torch.load(p, map_location=\"cpu\", weights_only=True)[\"model_state_dict\"] "
        "for p in pathlib.Path(\"model_params\").glob(\"*.pt\")]'",
        f"/opt/mpnn-env/bin/python {_PUBLIC_MPNN_ROOT}/run.py --help >/dev/null",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .env(PATH="/opt/mpnn-env/bin:$PATH", VIRTUAL_ENV="/opt/mpnn-env", PYTHONPATH=_PUBLIC_MPNN_ROOT)
    .label(Version=f"LigandMPNN-{_PUBLIC_MPNN_REVISION}", License="MIT", CUDA="12.8")
    .runscript('"$@"')
    .template_command(exec_mode="exec")
)


class MPNNArgs(BaseModel):
    """Arguments passed to the LigandMPNN ``run.py`` CLI.

    User-supplied values override defaults — unset fields keep their safe defaults.
    These are serialized as ``--{field_name} {value}`` CLI flags.
    """

    # Sampling
    temperature: float = Field(
        0.1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Sampling temperature. Lower = more conservative.",
    )
    seed: int = Field(
        0,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Random seed for torch, numpy, and python random.",
    )
    batch_size: int = Field(
        1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of sequences to generate per forward pass.",
    )
    number_of_batches: int = Field(
        1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of batches to run.",
    )

    # Amino acid constraints
    omit_AA: str = Field(
        "X",
        description="Amino acids to omit from design (e.g. 'XC'). "
        "X is always omitted. Add 'C' to avoid unpaired disulfides.",
    )
    bias_AA: str = Field(
        "",
        description="Per-AA logit bias, e.g. 'A:-1.024,P:2.34,C:-12.34'.",
    )
    bias_AA_per_residue: str = Field(
        "",
        description="Path to JSON mapping residue-level AA biases, "
        "e.g. {'A12': {'G': -0.3, 'C': -2.0}}.",
    )

    # Residue selection
    fixed_residues: str = Field(
        "", description="Space-separated fixed residues, e.g. 'A12 A13 B2'."
    )
    redesigned_residues: str = Field(
        "",
        description="Space-separated residues to redesign (all others fixed), e.g. 'A12 A13 B2'.",
    )
    chains_to_design: str = Field(
        "",
        description="Comma-separated chain IDs to redesign; others stay fixed, e.g. 'A,B'.",
    )
    parse_these_chains_only: str = Field(
        "",
        description="Only parse these chains from the input structure, e.g. 'A,B'.",
    )

    # Symmetry
    symmetry_residues: str = Field(
        "",
        description="Symmetric residue groups, e.g. 'A12,A13,A14|C2,C3|A5,B6'.",
    )
    symmetry_weights: str = Field(
        "",
        description="Weights matching symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0'.",
    )
    homo_oligomer: int = Field(
        0,
        description="1 = auto-set symmetry for homo-oligomer design with equal weighting.",
    )

    # Ligand context
    ligand_mpnn_use_atom_context: int = Field(
        1,
        description="1 = use non-protein atom context, 0 = ignore it.",
    )
    ligand_mpnn_cutoff_for_score: float = Field(
        8.0,
        description="Angstrom cutoff between protein and context atoms for scoring.",
    )
    ligand_mpnn_use_side_chain_context: int = Field(
        1,
        description="1 = use side chain atoms of fixed residues as ligand context.",
    )

    # Side chain packing
    pack_side_chains: int = Field(
        1,
        description=(
            "Legacy structure-mode side-chain packing flag. output_mode='sequences' "
            "forces this to 0; output_mode='structures' requires 1."
        ),
    )
    checkpoint_path_sc: str = Field(
        _PUBLIC_MPNN_PACKER_PATH,
        description=(
            "Path to side chain packer weights. The default is the official "
            "checkpoint baked into the image; a custom host file is bound in "
            "read-only when output_mode='structures'."
        ),
    )
    number_of_packs_per_design: int = Field(
        4,
        description="Independent side chain packing samples per design.",
    )
    repack_everything: int = Field(
        0,
        description="1 = repack all residues including fixed; 0 = keep fixed side chains.",
    )
    pack_with_ligand_context: int = Field(
        1,
        description="1 = use ligand context during side chain packing.",
    )
    sc_num_denoising_steps: int = Field(
        3,
        description="Number of denoising/recycling steps for side chain packing.",
    )
    sc_num_samples: int = Field(
        16,
        description="Samples from mixture distribution; highest-likelihood sample is kept.",
    )

    # Output formatting
    packed_suffix: str = Field(
        "_", description="Suffix appended to packed PDB filenames."
    )
    zero_indexed: int = Field(1, description="1 = start output PDB numbering at 0.")
    force_hetatm: int = Field(
        1,
        description="1 = write ligand atoms as HETATM in output PDB.",
    )
    save_stats: int = Field(0, description="1 = save output statistics.")
    file_ending: str = Field("", description="String appended to output filenames.")
    verbose: int = Field(1, description="Verbosity level.")

    # Membrane model fields
    transmembrane_buried: str = Field(
        "",
        description="Buried residues for per_residue_label_membrane_mpnn, e.g. 'A12 A13 B2'.",
    )
    transmembrane_interface: str = Field(
        "",
        description="Interface residues for per_residue_label_membrane_mpnn, e.g. 'A12 A13 B2'.",
    )
    global_transmembrane_label: int = Field(
        0,
        description="Global label for global_label_membrane_mpnn. 1 = transmembrane, 0 = soluble.",
    )

    parse_atoms_with_zero_occupancy: int = Field(
        0,
        description="1 = parse atoms with zero occupancy from input PDB.",
    )

    def to_cli_dict(self) -> dict:
        """Return only non-default, non-empty fields for CLI serialization."""
        defaults = MPNNArgs()
        result = {}
        for name, value in self:
            default_val = getattr(defaults, name)
            # Always include these critical fields regardless of default
            if name in (
                "temperature",
                "omit_AA",
                "pack_side_chains",
                "packed_suffix",
                "force_hetatm",
                "zero_indexed",
                "ligand_mpnn_use_side_chain_context",
                "repack_everything",
                "checkpoint_path_sc",
            ):
                result[name] = value
            elif value != default_val and value != "":
                result[name] = value
        return result


class ClassicMPNNBlock(CompiledBlockFunction):
    """Inverse folding with LigandMPNN.

    ``model_type`` picks the model family and ``noise_level`` the checkpoint
    version within it (see ``_PUBLIC_MPNN_CHECKPOINTS``).
    """

    # An option this block no longer has (such as one for a removed model) must fail
    # loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    _name = "mpnn"
    _display_name = "ProteinMPNN"
    _description = "Inverse folding with LigandMPNN (protein, ligand, soluble and membrane models)"
    _image = LIGANDMPNN_PUBLIC_CONTAINER
    _resources = JobArgs(
        partition="cpu",
        mem_gb=8,
        rows_per_job=100,
    )
    _inline = False
    _chunked = True
    _emits_sequence_table = True

    def get_command_template(self) -> list[str]:
        """The ``run.py`` command with every checkpoint flag spelled out.

        All five model checkpoints are passed, each at its upstream default noise
        level; ``noise_level`` overrides the one for ``model_type``. Code and
        checkpoints are baked into the image, so nothing is mounted for them.
        """
        # Host environment prefixes are discarded by --cleanenv / Docker.
        runtime_env = {"CUDA_VISIBLE_DEVICES": ""} if self.device == "cpu" else None
        binds = []
        packer = self.args.checkpoint_path_sc
        if packer != _PUBLIC_MPNN_PACKER_PATH and self.output_mode == "structures":
            path = Path(packer).expanduser().absolute()
            if not path.is_file():
                raise FileNotFoundError(f"MPNN custom packer checkpoint does not exist: {path}")
            binds.append((str(path), f"{path}:ro"))
        cmd = LIGANDMPNN_PUBLIC_CONTAINER.build_command(
            "python", "-u", f"{_PUBLIC_MPNN_ROOT}/run.py", binds=binds, env=runtime_env
        )
        for model_type in _PUBLIC_MPNN_CHECKPOINTS:
            noise = self.noise_level if model_type == self.model_type else None
            cmd += [f"--checkpoint_{model_type}", _public_checkpoint(model_type, noise)]
        return cmd

    # Model family and version
    model_type: Literal[
        "protein_mpnn",
        "ligand_mpnn",
        "soluble_mpnn",
        "per_residue_label_membrane_mpnn",
        "global_label_membrane_mpnn",
    ] = Field(
        default="ligand_mpnn",
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "Model family: protein_mpnn, ligand_mpnn, soluble_mpnn, or a membrane "
            "model (per_residue_label_membrane_mpnn, global_label_membrane_mpnn)."
        ),
    )
    noise_level: Literal[0.02, 0.05, 0.1, 0.2, 0.3] | None = Field(
        default=None,
        json_schema_extra={"tier": TIER_PRIMARY, "cli_type": float},
        description=(
            "Checkpoint version: backbone noise (Angstrom) the model was trained "
            "with. None uses the upstream default. Valid values depend on "
            "model_type: protein_mpnn and soluble_mpnn 0.02/0.1/0.2/0.3, "
            "ligand_mpnn 0.05/0.1/0.2/0.3, membrane models 0.2."
        ),
    )

    # Sequence generation parameters
    num_seq_per_pdb: int = Field(
        default=8,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of sequences to generate per structure",
    )
    output_mode: Literal["sequences", "structures"] = Field(
        default="sequences",
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "Return compact sequence rows (default) or legacy side-chain-packed "
            "structures. Sequence mode disables packing."
        ),
    )

    skip_existing: bool = Field(
        default=True, description="Skip processing if outputs already exist"
    )
    device: str | None = Field(
        default=None,
        description="Force device for MPNN inference (e.g. 'cpu'). Default: auto-detect.",
    )

    # Design constraints for pipelining
    keep_motif_sequences_fixed: bool = Field(
        default=True,
        description="Keep sequences of motifs fixed (if read into the dataframe)",
    )
    fix_interface_residues: bool = Field(
        default=False, description="Fix the sequence at the interfaces"
    )
    hbond_fix: bool = Field(
        default=False, description="Fix residues involved in hydrogen bonds"
    )
    aromatic_fix: bool = Field(
        default=False, description="Fix aromatic residues near ligand"
    )
    component_remap: dict[str, str] | None = Field(
        default=None,
        description="Remap residue names in CIF files before MPNN. Maps old -> new (e.g., {'LIG_B': 'B;N'}).",
    )

    args: MPNNArgs = Field(
        default_factory=MPNNArgs,
        description="LigandMPNN/ProteinMPNN CLI arguments. Partial overrides keep defaults.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_args_dict(cls, values):
        """Allow passing args as a plain dict for backwards compatibility."""
        if isinstance(values, dict) and isinstance(values.get("args"), dict):
            values["args"] = MPNNArgs(**values["args"])
        return values

    @model_validator(mode="after")
    def validate_model_version(self):
        # Also raised by get_command_template when the template is built; this
        # keeps the check when a command_template is supplied explicitly.
        _public_checkpoint(self.model_type, self.noise_level)
        membrane_only = {
            "transmembrane_buried": "per_residue_label_membrane_mpnn",
            "transmembrane_interface": "per_residue_label_membrane_mpnn",
            "global_transmembrane_label": "global_label_membrane_mpnn",
        }
        for name, model_type in membrane_only.items():
            if getattr(self.args, name) not in ("", 0) and self.model_type != model_type:
                raise ValueError(
                    f"args.{name} only applies to model_type={model_type!r}, "
                    f"not {self.model_type!r}."
                )
        return self

    @model_validator(mode="after")
    def validate_output_mode(self):
        if self.output_mode == "structures" and self.args.pack_side_chains != 1:
            raise ValueError(
                "output_mode='structures' requires args.pack_side_chains=1."
            )
        if self.output_mode == "sequences" and self.store != "none":
            raise ValueError("Sequence-only MPNN output cannot be stored as catcif.")
        return self

    def _remap_inputs(self, handler, sandbox_dir):
        """Remap residue names in input CIF files if component_remap is set."""
        from concurrent.futures import ThreadPoolExecutor
        from atomworks.io.utils.io_utils import to_cif_file

        from chaperone.io import ensure_bonds
        from tqdm import tqdm

        remap_dir = Path(sandbox_dir) / "remapped_inputs"
        remap_dir.mkdir(parents=True, exist_ok=True)

        component_remap = self.component_remap

        def _remap_one(data):
            atom_array = data.atom_array
            for old_name, new_name in component_remap.items():
                mask = atom_array.res_name == old_name
                atom_array.res_name[mask] = new_name
            out_name = Path(data.path).name
            if out_name.endswith(".gz"):
                out_name = out_name[:-3]
            out_path = remap_dir / out_name
            to_cif_file(ensure_bonds(atom_array), str(out_path))
            return data.to_row_with_updates(path=str(out_path))

        items = list(handler.get_data(which="last", load=True, num_workers=8))
        with ThreadPoolExecutor(max_workers=8) as executor:
            updated_rows = list(
                tqdm(
                    executor.map(_remap_one, items),
                    total=len(items),
                    desc="remapping",
                )
            )

        logger.info(
            f"Remapped residue names {self.component_remap} in {len(updated_rows)} files -> {remap_dir}"
        )
        return PipelineDataHandler(updated_rows)

    def _has_packed_output(self, input_path: str, packed_dir: Path) -> bool:
        """Check if packed outputs already exist for an input structure."""
        if not packed_dir.exists():
            return False
        stem = Path(input_path).stem
        if stem.endswith(".cif"):
            stem = stem[:-4]
        return any(packed_dir.glob(f"{stem}*"))

    @staticmethod
    def _atoms_of(data):
        """The structure's atoms, loading the file only if the handler has not."""
        if data.atom_array is not None:
            return data.atom_array
        return data.load_array().atom_array

    @staticmethod
    def _write_portable_pdb(input_path, atoms, rename: ChainRename, out_dir: Path) -> str:
        """Stage a PDB for LigandMPNN, preserving the original CIF as parent."""
        from chaperone.io import write_pdb

        stem = Path(input_path).name
        for suffix in (".gz", ".cif", ".pdb"):
            stem = stem.removesuffix(suffix)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.pdb"
        try:
            write_pdb(rename.rename(atoms), out_path)
        except OSError:
            raise
        except Exception as error:
            raise ValueError(
                f"MPNN requires PDB-representable inputs: {input_path}. "
                "Check chain/residue identifiers and ligand component names."
            ) from error
        return str(out_path)

    @staticmethod
    def _restore_chain_ids(paths, sandbox_dir) -> list[Path]:
        """Give outputs the chain ids of the inputs they came from.

        ``create_command`` renames multi-character chains for MPNN and records each
        rename under ``inputs/``; this undoes it on the packed PDBs (written back as
        CIF), so the block's outputs use the same chain ids as its inputs and the
        metadata they inherit still matches.
        """
        renames = {}
        for sidecar in sorted(
            (Path(sandbox_dir) / "inputs").glob("chain_renames_chunk*.json")
        ):
            renames.update(json.loads(sidecar.read_text()))
        restored_paths = []
        for path in map(Path, paths):
            owners = [
                stem
                for stem in renames
                if re.match(re.escape(stem) + r"(?:_|\.|$)", path.name)
            ]
            if not owners:
                restored_paths.append(path)
                continue
            rename = ChainRename(renames[max(owners, key=len)])
            if path.suffix == ".pdb":
                from atomworks.io.utils.io_utils import to_cif_file
                from chaperone.io import ensure_bonds, parse_to_atom_array

                atoms = parse_to_atom_array(str(path))
                if rename.restorable_in(atoms.chain_id):
                    original_chains = atoms.chain_id.copy()
                    for original, stand_in in rename.mapping.items():
                        atoms.chain_id[original_chains == stand_in] = original
                    path = path.with_suffix(".cif")
                    to_cif_file(ensure_bonds(atoms), str(path))
            restored_paths.append(path)
        return list(dict.fromkeys(restored_paths))

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        if self.hbond_fix or self.aromatic_fix:
            from blocks.util_blocks.annotate import (
                find_aromatic_residues_near_ligand,
                find_hbonds_in_pdb,
            )

        if self.component_remap:
            handler = self._remap_inputs(handler, sandbox_dir)
        inputdir = sandbox_dir / "inputs"
        inputdir.mkdir(parents=True, exist_ok=True)
        fix_pos_json = os.path.join(
            inputdir, f"pdb_position_fixed_chunk{chunk_id}.json"
        )

        packed_dir = Path(sandbox_dir) / "packed"

        # Write JSON in single loop - iterate over rows directly
        fixed_position_dict = {}
        skipped = 0
        needs_load = (
            self.hbond_fix or self.aromatic_fix or self.output_mode == "sequences"
        )
        manifest = {}
        renames: dict[str, dict[str, str]] = {}  # input stem -> chain rename
        chain_options: dict[str, str] = {}
        for data in handler.get_data(which="last", load=needs_load):
            # get path and design_metadata - contains motif, interface, hbond, aromatics_near_ligand.
            # Materialize archived structures; ordinary local paths pass through.
            input_path = str(self.stage_file(data.path, sandbox_dir, subdir="inputs"))

            if self.output_mode == "sequences":
                chain_sequences = {
                    str(chain): str(sequence)
                    for chain, sequence in (data.sequence or {}).items()
                    if sequence
                }
                input_name = Path(input_path).stem
                if input_name.endswith((".cif", ".pdb")):
                    input_name = input_name[:-4]
                manifest[input_name] = {
                    "input_name": input_name,
                    "parent_design_id": data.design_id or data.basename,
                    "chain_ids": list(chain_sequences),
                    "chain_lengths": [len(seq) for seq in chain_sequences.values()],
                }

            # Skip if outputs already exist
            if self.skip_existing and self._has_packed_output(input_path, packed_dir):
                skipped += 1
                continue

            design_metadata = data.metadata

            # LigandMPNN reads PDB, so the structure is staged as one with
            # single-letter chain ids. The manifest maps the resulting FASTA back
            # to the original chain identifiers.
            atoms = self._atoms_of(data)
            rename = ChainRename.for_atoms(atoms)
            selected_chains = None
            original_chains = set(map(str, atoms.chain_id))
            options = {}
            for option in ("chains_to_design", "parse_these_chains_only"):
                value = getattr(self.args, option)
                if not value:
                    continue
                chains = value.split(",")
                if option == "parse_these_chains_only" and "," not in value and value not in original_chains:
                    chains = list(value)  # accept the old concatenated single-letter spelling
                if not set(chains) <= original_chains:
                    raise ValueError(f"{option} refers to missing chains: {set(chains) - original_chains}")
                options[option] = ",".join(rename.mapping.get(chain, chain) for chain in chains)
                if option == "parse_these_chains_only":
                    selected_chains = set(chains)
            for option in ("redesigned_residues", "symmetry_residues", "transmembrane_buried", "transmembrane_interface"):
                value = getattr(self.args, option)
                if value:
                    options[option] = "".join(
                        part if re.fullmatch(r"[\s,|]+", part) else rename.apply(part)
                        for part in re.split(r"([\s,|]+)", value)
                    )
            if rename and self.args.bias_AA_per_residue:
                raise ValueError("MPNN residue bias with renamed chains is unsupported; supply single-character input chain IDs for bias_AA_per_residue.")
            for option, value in options.items():
                if option in chain_options and chain_options[option] != value:
                    raise ValueError(f"MPNN {option} needs different chain mappings across inputs; run those inputs in separate chunks.")
                chain_options[option] = value
            input_path = self._write_portable_pdb(
                input_path, atoms, rename, inputdir / "portable_pdb"
            )
            if self.output_mode == "sequences":
                # LigandMPNN writes FASTA chains in sorted PDB-chain order.
                # Keep original identifiers in that same order for compile.
                chain_ids = sorted(
                    (chain for chain in chain_sequences if selected_chains is None or chain in selected_chains),
                    key=lambda chain: rename.mapping.get(chain, chain),
                )
                manifest[input_name]["chain_ids"] = chain_ids
                manifest[input_name]["chain_lengths"] = [
                    len(chain_sequences[chain]) for chain in chain_ids
                ]
            if rename:
                renames[Path(input_path).stem] = rename.mapping

            # Build fixed positions list from design_metadata
            fixed_positions = []
            if self.keep_motif_sequences_fixed:
                fixed_positions.extend(design_metadata.get("motif") or [])
            if self.fix_interface_residues:
                interface = design_metadata.get("interface")
                if not interface:
                    raise ValueError(
                        f"fix_interface_residues=True but no 'interface' metadata found for {input_path}. "
                        "Run the 'annotate' block first to compute interface residues."
                    )
                fixed_positions.extend(interface)
            if self.hbond_fix:
                hbond = design_metadata.get("hbond")
                if not hbond:
                    hbond = find_hbonds_in_pdb(data.atom_array)
                fixed_positions.extend(hbond)
            if self.aromatic_fix:
                aromatics = design_metadata.get("aromatics_near_ligand")
                if not aromatics:
                    aromatics = find_aromatic_residues_near_ligand(data.atom_array)
                fixed_positions.extend(aromatics)
            # ``run.py`` ignores the scalar ``--fixed_residues`` option whenever
            # ``--fixed_residues_multi`` is present. Chaperone always supplies
            # the latter, so merge the MPNNArgs constraint into each
            # per-input entry rather than silently dropping it.
            fixed_positions.extend(self.args.fixed_residues.split())
            fixed_position_dict[input_path] = sorted(
                set(rename.apply(p) for p in fixed_positions if p)
            )

        if renames:
            # compile() reads this to give the outputs their original chain ids back.
            with open(inputdir / f"chain_renames_chunk{chunk_id}.json", "w") as f:
                json.dump(renames, f)

        # run.py falls back to CIF `is_motif_atom_with_fixed_seq` inference when
        # ALL entries are empty (any([[],[],...])==False). rfd3_partial writes
        # that annotation True on every atom → full no-op despite the user
        # setting keep_motif_sequences_fixed=False. A single non-matching
        # sentinel makes any() pass, skips CIF inference, redesigns everything.
        if (
            not self.keep_motif_sequences_fixed
            and fixed_position_dict
            and not any(fixed_position_dict.values())
        ):
            for k in fixed_position_dict:
                fixed_position_dict[k] = ["_suppress_cif_motif_inference"]
            logger.info(
                "keep_motif_sequences_fixed=False with no other constraints → "
                "writing sentinel to suppress run.py CIF motif inference"
            )

        if skipped:
            logger.info(f"Skipped {skipped} structures with existing packed outputs")

        if not fixed_position_dict:
            logger.info("All structures already processed, skipping MPNN")
            return ["echo", "All structures already processed, skipping MPNN"]

        # Write JSON file
        with open(fix_pos_json, "w") as f:
            json.dump(
                {path: " ".join(positions) for path, positions in fixed_position_dict.items()},
                f,
            )
            f.write("\n")

        logger.info(f"Fixed position json file is saved at {fix_pos_json}")
        if self.output_mode == "sequences":
            write_manifest(sandbox_dir, manifest)

        # Create command
        cmd = self.command_template + [
            "--out_folder",
            sandbox_dir,
            "--model_type",
            self.model_type,
            "--pdb_path_multi",
            fix_pos_json,
            "--fixed_residues_multi",
            fix_pos_json,
            "--batch_size",
            self.num_seq_per_pdb,
        ]
        cli_args = self.args.to_cli_dict()
        cli_args.update(chain_options)
        cli_args["fasta_seq_separation"] = "/"
        cli_args["number_of_packs_per_design"] = self.args.number_of_packs_per_design
        cli_args["pack_with_ligand_context"] = self.args.pack_with_ligand_context
        if self.args.checkpoint_path_sc != _PUBLIC_MPNN_PACKER_PATH:
            cli_args["checkpoint_path_sc"] = str(Path(self.args.checkpoint_path_sc).expanduser().absolute())
        cli_args["pack_side_chains"] = 0 if self.output_mode == "sequences" else 1
        bias_path = cli_args.get("bias_AA_per_residue")
        if bias_path:
            source = Path(bias_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"bias_AA_per_residue JSON not found: {source}")
            staged_bias = inputdir / source.name
            self.stage_or_copy(source, staged_bias)
            cli_args["bias_AA_per_residue"] = str(staged_bias)
        for name, argv in cli_args.items():
            cmd += [f"--{name}", str(argv)]
        return cmd

    def compile(self, sandbox_dir):
        """Compile MPNN outputs into a PipelineDataHandler.

        Sequence mode parses generated FASTA records without materializing
        coordinates. Metadata and structure context are inherited from the
        parent during the engine's compile phase via deep merge.
        """

        if self.output_mode == "sequences":
            manifest = load_manifest(sandbox_dir)
            fasta_paths = sorted((Path(sandbox_dir) / "seqs").glob("*.fa"))
            self._guard_outputs(fasta_paths, str(Path(sandbox_dir) / "seqs"))
            rows = []
            for fasta_path in fasta_paths:
                parent = manifest_entry_for_fasta(fasta_path, manifest)
                for index, (header, sequence) in enumerate(parse_fasta(fasta_path)):
                    # run.py writes the native/input sequence first. Only
                    # sampled records carry the generated-design id field.
                    if "id=" not in header:
                        continue
                    rows.append(
                        sequence_row(
                            block_name=self.name,
                            parent=parent,
                            header=header,
                            sequence=sequence,
                            delimiter="/",
                            fallback_index=index,
                        )
                    )
            self._guard_outputs(rows, "generated records in MPNN FASTA files")
            return PipelineDataHandler(rows)

        packed_dir = Path(sandbox_dir) / "packed"
        paths = glob_designs_in_dir(packed_dir)
        self._guard_outputs(paths, str(packed_dir))

        # Rename files with .cif.gz embedded in stem (MPNN inherits input ext)
        cleaned_paths = []
        for path in paths:
            path = Path(path)
            if ".cif.gz" in path.name:
                new_name = path.name.replace(".cif.gz", "")
                new_path = path.parent / new_name
                if not new_path.exists():
                    path.rename(new_path)
                path = new_path
            cleaned_paths.append(path)

        cleaned_paths = self._restore_chain_ids(cleaned_paths, sandbox_dir)

        # Single path per row - parent metadata inherited during compile_chunk_job
        rows = []
        for path in cleaned_paths:
            rows.append(
                make_row(
                    name=self.name,
                    path=str(path),
                )
            )
        return PipelineDataHandler(rows)


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[3]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "mpnn"
    SANDBOX = REPO_ROOT / "unittest_outs" / "mpnn"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = ClassicMPNNBlock(num_seq_per_pdb=1)
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="rfd3"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
