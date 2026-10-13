# Classic MPNN block for inverse folding
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from atomworks.io.utils.chain import create_chain_id_generator
from pydantic import BaseModel, Field, model_validator

from chaperone import (
    CompiledBlockFunction,
    PipelineDataHandler,
    JobArgs,
    get_setting_field,
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


@dataclass(frozen=True)
class ChainRename:
    """Single-letter stand-ins for the multi-character chain ids of one structure.

    RFD3 names unindexed residues with chain ids such as ``X1``, which fused_mpnn's
    PDB writer rejects. The structure is handed to MPNN with those chains renamed
    (``rename``), fixed-position strings follow the rename (``apply``), and the
    outputs get the original ids back (``restore_cif``), so the block's outputs use
    the same chain ids as its inputs.
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

    def restore_cif(self, path: Path) -> None:
        """Rewrite every chain-id column of a CIF in place, stand-ins back to originals."""
        import gzip

        import numpy as np
        from biotite.structure.io.pdbx import CIFColumn, CIFFile

        original_of = {new: old for old, new in self.mapping.items()}
        compressed = str(path).endswith(".gz")
        if compressed:
            with gzip.open(path, "rt") as handle:
                cif = CIFFile.read(handle)
        else:
            cif = CIFFile.read(path)
        changed = False
        for block in cif.values():
            for category in block.values():
                for name in list(category.keys()):
                    # atom_site.auth_asym_id / label_asym_id, entity_poly.pdbx_strand_id, ...
                    if not re.search(r"(asym_id|strand_id)$", name):
                        continue
                    values = category[name].as_array(str)
                    restored = np.array(
                        [
                            ",".join(
                                original_of.get(part, part) for part in v.split(",")
                            )
                            for v in values
                        ]
                    )
                    if not np.array_equal(values, restored):
                        category[name] = CIFColumn(restored)
                        changed = True
        if changed:
            if compressed:
                with gzip.open(path, "wt") as handle:
                    cif.write(handle)
            else:
                cif.write(path)


def _chain_ids_in_cif(path: Path) -> set[str]:
    """Chain ids of a CIF, read from its atom_site table without building atoms."""
    import gzip

    from biotite.structure.io.pdbx import CIFFile

    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            cif = CIFFile.read(handle)
    else:
        cif = CIFFile.read(path)
    return {str(c) for c in cif.block["atom_site"]["auth_asym_id"].as_array(str)}


# ── Container ───────────────────────────────────────────────────────────
#
# Two variants of the "modelhub" container exist:
#
#   1. The legacy institutional SIF at /projects/ml/modelhub/apptainer/...
#      This is what IPD SLURM jobs use. It bakes fused_mpnn's run.py at
#      /net/software/lab/fused_mpnn/v1.1.1/run.py via a bind mount.
#
#   2. A portable buildable variant that git-clones
#      baker-laboratory/fused_mpnn at build time and installs a minimal
#      python venv with torch, numpy, and prody. The run.py lives at
#      /opt/fused_mpnn/run.py inside this image.
#
# We auto-select at import time: if the on-prem SIF exists on disk, prefer
# it (so IPD runs stay byte-identical to the past); otherwise fall back to
# the buildable variant for macOS and fresh clones.

_ONPREM_MODELHUB_CANDIDATES = (
    "/projects/ml/modelhub/apptainer/frozen_modelhub_datahub_cifutils_2025-02-05.sif",
    "/projects/ml/modelhub/apptainer/test_modelhub_2025-05-01.sif",
)
_ONPREM_MODELHUB_SIF = next(
    (path for path in _ONPREM_MODELHUB_CANDIDATES if Path(path).is_file()),
    _ONPREM_MODELHUB_CANDIDATES[0],
)

_MODELHUB_SIF_CONTAINER = Image.from_existing(
    "modelhub", _ONPREM_MODELHUB_SIF
).template_command(
    exec_mode="exec",
    # The on-prem SIF deliberately keeps code and weights outside the image.
    # These sources are first allowlisted into the isolated worker by the
    # controller, then explicitly rebound across this nested container layer.
    binds=[
        ("/net/software/lab/fused_mpnn", "/net/software/lab/fused_mpnn"),
        ("/net/databases/mpnn", "/databases/mpnn"),
    ],
)

# fused_mpnn is a private Baker-lab fork of LigandMPNN with added model
# types (pssm, antibody, lddt-enhanced). The container is a minimal
# Ubuntu + python3.12 venv with torch/numpy/prody/openfold plus the repo cloned
# into /opt/fused_mpnn. Host-side `git clone` happens during
# `chaperone install modelhub_portable` via `prepare_git_clones()`; the
# builder copies the result in via `COPY` / `%files`.
_MODELHUB_PORTABLE_CONTAINER = (
    Image("modelhub_portable")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
        "python3-pip",
    )
    .run(
        "python3.12 -m venv /opt/mpnn-env",
        ". /opt/mpnn-env/bin/activate && pip install --upgrade pip setuptools wheel",
        ". /opt/mpnn-env/bin/activate && pip install --no-cache-dir torch numpy prody ml-collections dm-tree biopython modelcif",
    )
    .git_clone(
        "https://github.com/baker-laboratory/fused_mpnn",
        "/opt/fused_mpnn",
    )
    # openfold is required by fused_mpnn's sc_utils.py for side-chain packing.
    # Only pure-Python modules are used (residue_constants, rigid_utils, feats,
    # data_transforms, loss), so we clone the repo and add it to PYTHONPATH
    # instead of building the full package (which needs CUDA toolkit).
    .git_clone(
        "https://github.com/aqlaboratory/openfold",
        "/opt/openfold",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        PATH="/opt/mpnn-env/bin:$PATH",
        VIRTUAL_ENV="/opt/mpnn-env",
        # fused_mpnn's run.py imports data_utils, model_utils, sc_utils
        # as top-level modules, not as a package — the repo root itself
        # must be on PYTHONPATH.
        PYTHONPATH="/opt/fused_mpnn:/opt/openfold",
    )
    .label(
        Version="fused_mpnn-main", Platform="cpu-or-gpu",
        BuildAccess="Requires authorized access to private baker-laboratory/fused_mpnn; "
        "foundry_mpnn is a separate public implementation, not a validated drop-in replacement",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        env={
            "MPNN_WEIGHTS_DIR": lambda: get_setting_field("MPNN_WEIGHTS_DIR", ""),
        },
    )
)


def _pick_modelhub_container() -> Image:
    """On-prem: use the frozen SIF. Elsewhere: build from fused_mpnn."""
    if Path(_ONPREM_MODELHUB_SIF).is_file():
        return _MODELHUB_SIF_CONTAINER
    return _MODELHUB_PORTABLE_CONTAINER


MODELHUB_CONTAINER = _pick_modelhub_container()
MODELHUB_PORTABLE_CONTAINER = _MODELHUB_PORTABLE_CONTAINER


class MPNNArgs(BaseModel):
    """Arguments passed to LigandMPNN/ProteinMPNN run.py CLI.

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
        default_factory=lambda: (
            f"{wd.rstrip('/')}/packer_weights/s_300756.pt"
            if (wd := get_setting_field("MPNN_WEIGHTS_DIR", ""))
            else "/databases/mpnn/packer_weights/s_300756.pt"
        ),
        description=(
            "Path to side chain packer model weights. "
            "Defaults to MPNN_WEIGHTS_DIR/packer_weights/s_300756.pt when "
            "MPNN_WEIGHTS_DIR is set; otherwise to "
            "the on-prem path /databases/mpnn/packer_weights/s_300756.pt."
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
    """Classic MPNN block for inverse folding."""

    _name = "mpnn"
    _display_name = "ProteinMPNN"
    _description = "Classic MPNN block for inverse folding"
    _image = MODELHUB_CONTAINER
    _resources = JobArgs(
        partition="cpu",
        mem_gb=8,
        rows_per_job=100,
    )
    _inline = False
    _chunked = True
    _emits_sequence_table = True

    def _get_image(self=None):
        return MODELHUB_CONTAINER

    def get_command_template(self) -> list[str]:
        """Build the MPNN command template, routing around the on-prem SIF
        vs the portable buildable container.

        On-prem: the SIF exposes run.py via a bind mount at
        ``/net/software/lab/fused_mpnn/v1.1.1/run.py`` and weights at
        ``/databases/mpnn/...``. Host database roots vary by machine
        (``/databases`` vs ``/net/databases``), so we bind whichever exists.

        Portable: run.py is baked into the image at ``/opt/fused_mpnn/run.py``
        and weight paths come from ``MPNN_WEIGHTS_DIR``.
        """
        is_portable_container = self._get_image() is _MODELHUB_PORTABLE_CONTAINER
        weights_dir = None
        if is_portable_container:
            run_py = "/opt/fused_mpnn/run.py"
            binds: list[tuple[str, str]] = []
            weights_dir = get_setting_field("MPNN_WEIGHTS_DIR", default=None)
            if not weights_dir:
                weights_dir = "/weights/mpnn"
            # The image carries no checkpoints and --containall hides the host
            # filesystem, so expose the weights directory at the same path.
            if Path(weights_dir).is_dir():
                host_weights = os.path.abspath(weights_dir)
                binds.append((host_weights, host_weights))
        else:
            run_py = "/net/software/lab/fused_mpnn/v1.1.1/run.py"
            host_databases = (
                "/databases"
                if Path("/databases").is_dir()
                else "/net/databases"
                if Path("/net/databases").is_dir()
                else "/databases"
            )
            binds = [
                (host_databases, "/databases"),
                ("/net/software", "/net/software"),
            ]

        cmd = self._get_image().build_command("python", "-u", run_py, binds=binds)

        if not is_portable_container:
            return cmd

        # Spell out the common checkpoint paths. run.py looks up a
        # single path based on --model_type so unused entries are harmless.
        wd = weights_dir.rstrip("/")
        cmd += [
            "--checkpoint_protein_mpnn",
            f"{wd}/vanilla_model_weights/v_48_020.pt",
            "--checkpoint_ligand_mpnn",
            f"{wd}/ligand_mpnn_model_weights/s25_r010_t300_p.pt",
            "--checkpoint_per_residue_label_membrane_mpnn",
            f"{wd}/tmd_per_residue_weights/tmd_v_48_020.pt",
            "--checkpoint_global_label_membrane_mpnn",
            f"{wd}/tmd_weights/v_48_020.pt",
            "--checkpoint_soluble_mpnn",
            f"{wd}/no_transmembrane/v_48_020.pt",
        ]
        return cmd

    # Checkpoint type
    model_type: Literal[
        "protein_mpnn",
        "ligand_mpnn",
        "soluble_mpnn",
    ] = Field(
        default="ligand_mpnn",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Model type to use for inverse folding",
    )
    lddt_enhanced: bool = Field(
        default=False,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Use lddt-enhanced model (Fanglei's masterpiece)",
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
    def _write_renamed(input_path, atoms, rename: ChainRename, out_dir: Path) -> str:
        """Write ``atoms`` with the renamed chains as a new file; the input is untouched."""
        from atomworks.io.utils.io_utils import to_cif_file

        from chaperone.io import ensure_bonds

        stem = Path(input_path).name
        for suffix in (".gz", ".cif", ".pdb"):
            stem = stem.removesuffix(suffix)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.cif"
        to_cif_file(ensure_bonds(rename.rename(atoms)), str(out_path))
        logger.info("Renamed chains %s in %s for MPNN", rename.mapping, input_path)
        return str(out_path)

    @staticmethod
    def _restore_chain_ids(paths, sandbox_dir) -> None:
        """Give outputs the chain ids of the inputs they came from.

        ``create_command`` renames multi-character chains for MPNN and records each
        rename under ``inputs/``; this undoes it, so the block's outputs use the same
        chain ids as its inputs and the metadata they inherit still matches.
        """
        renames = {}
        for sidecar in sorted(
            (Path(sandbox_dir) / "inputs").glob("chain_renames_chunk*.json")
        ):
            renames.update(json.loads(sidecar.read_text()))
        if not renames:
            return
        for path in map(Path, paths):
            owners = [
                stem
                for stem in renames
                if re.match(re.escape(stem) + r"(?:_|\.|$)", path.name)
            ]
            if not owners:
                continue
            rename = ChainRename(renames[max(owners, key=len)])
            if path.suffix == ".pdb":
                logger.warning(
                    "Cannot restore chain ids %s in %s: PDB files hold one-letter ids",
                    list(rename.mapping),
                    path,
                )
            elif rename.restorable_in(_chain_ids_in_cif(path)):
                rename.restore_cif(path)

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
                if input_name.endswith(".cif"):
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

            # Sequence mode reports per-chain sequences under the original chain
            # ids, so only structure mode needs the single-letter stand-ins.
            rename = ChainRename()
            if self.output_mode != "sequences":
                atoms = self._atoms_of(data)
                rename = ChainRename.for_atoms(atoms)
                if rename:
                    input_path = self._write_renamed(
                        input_path, atoms, rename, inputdir / "chain_ids_renamed"
                    )
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
            # the latter, so merge the public MPNNArgs constraint into each
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
            json.dump(fixed_position_dict, f)
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
        cmd += ["--enhance", "plddt_3_20240930-f9c9ea0f"] if self.lddt_enhanced else []
        if self.device == "cpu":
            cmd = ["env", "CUDA_VISIBLE_DEVICES="] + cmd
        cli_args = self.args.to_cli_dict()
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
                    # fused_mpnn writes the native/input sequence first. Only
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

        self._restore_chain_ids(cleaned_paths, sandbox_dir)

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
