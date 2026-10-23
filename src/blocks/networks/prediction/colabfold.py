"""ColabFold (AlphaFold2) structure prediction block."""

import glob
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from biotite.structure import filter_amino_acids
from pydantic import Field, model_validator

from blocks.networks.prediction.monomer_reference import (
    default_msa_dirs,
    normalize_msa_dirs,
    read_query_a3m,
    residue_sequence,
    resolve_msa_path,
    standard_residue_sequence,
)
from chaperone import JobArgs, PipelineDataHandler, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, ScriptSpec
from chaperone.containers import Image
from chaperone.data import make_row

logger = logging.getLogger(__name__)

###########################################################################
# Input helpers
###########################################################################


def protein_chain_sequences(atom_array) -> list[tuple[str, str, str]]:
    """Return ``(chain, folded_sequence, msa_lookup_sequence)`` in input order."""
    if "is_protein" in atom_array.get_annotation_categories():
        protein_mask = np.asarray(atom_array.is_protein, dtype=bool)
    else:
        protein_mask = filter_amino_acids(atom_array)
    protein = atom_array[protein_mask]
    if len(protein) == 0:
        raise ValueError("AlphaFold2 requires at least one protein chain")

    chain_category = (
        "chain_iid"
        if "chain_iid" in protein.get_annotation_categories()
        else "chain_id"
    )
    chain_ids = np.asarray(protein.get_annotation(chain_category)).astype(str)
    ordered_chain_ids = list(dict.fromkeys(chain_ids.tolist()))
    return [
        (
            chain_id,
            residue_sequence(protein[chain_ids == chain_id]),
            standard_residue_sequence(protein[chain_ids == chain_id]),
        )
        for chain_id in ordered_chain_ids
    ]


def unique_chain_sequences(
    chains: Sequence[tuple[str, str, str]],
) -> list[dict]:
    """Collapse identical folded sequences into ColabFold cardinalities."""
    unique: list[dict] = []
    by_sequence: dict[str, dict] = {}
    for chain_id, sequence, msa_lookup_sequence in chains:
        entry = by_sequence.get(sequence)
        if entry is None:
            entry = {
                "sequence": sequence,
                "cardinality": 0,
                "chain_ids": [],
                "msa_lookup_sequences": [],
            }
            by_sequence[sequence] = entry
            unique.append(entry)
        entry["cardinality"] += 1
        entry["chain_ids"].append(chain_id)
        if msa_lookup_sequence != sequence:
            entry["msa_lookup_sequences"].append(msa_lookup_sequence)
    return unique


def colabfold_fasta(target: str, chains: Sequence[tuple[str, str, str]]) -> str:
    """Serialize every protein chain using ColabFold's colon-separated FASTA."""
    return f">{target}\n{':'.join(chain[1] for chain in chains)}\n"


def colabfold_multimer_a3m(
    unique_chains: Sequence[dict],
    alignments: Sequence[tuple[Sequence[str], Sequence[str]]],
) -> str:
    """Serialize per-chain unpaired alignments in ColabFold's complex A3M format."""
    if len(unique_chains) != len(alignments):
        raise ValueError("One alignment is required for each unique protein sequence")

    sequences = [entry["sequence"] for entry in unique_chains]
    cardinalities = [entry["cardinality"] for entry in unique_chains]
    lines = [
        f"#{','.join(str(len(sequence)) for sequence in sequences)}"
        f"\t{','.join(str(cardinality) for cardinality in cardinalities)}",
        ">query",
        "".join(sequences),
    ]

    for chain_index, (headers, records) in enumerate(alignments):
        if len(headers) != len(records):
            raise ValueError("A3M headers and records must have the same length")
        left_padding = "-" * sum(len(sequence) for sequence in sequences[:chain_index])
        right_padding = "-" * sum(
            len(sequence) for sequence in sequences[chain_index + 1 :]
        )
        for header, record in zip(headers, records, strict=True):
            lines.extend((header, f"{left_padding}{record}{right_padding}"))

    return "\n".join(lines) + "\n"


###########################################################################
# Container
###########################################################################

# A public build recipe is used consistently on every host. The configured
# container search directories select an existing build with this interface.

COLABFOLD_VERSION = "v1.6.1"

# Ubuntu 24.04 is required by this cluster's root-mapped Apptainer fakeroot
# helper (built against glibc >=2.38). The CUDA runtime base lets conda pick
# the CUDA build of OpenMM; JAX brings its own CUDA libraries via pip wheels
# and --nv injects the host driver at run time.
_COLABFOLD_PORTABLE_CONTAINER = (
    Image("colabfold_portable")
    .base_image("nvidia/cuda:12.8.0-runtime-ubuntu24.04")
    .packages(
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "hhsuite",
        "kalign",
        "wget",
    )
    .run(
        "wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh",
        "bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh",
        # Pin the interpreter first so conda and pip share one Python.
        "/opt/conda/bin/conda install -y python=3.13",
        "/opt/conda/bin/pip install --no-cache-dir --upgrade 'pip>=26.2.2' setuptools wheel",
    )
    .run(
        # [alphafold] only: the [openmm] extra would replace conda's CUDA
        # OpenMM below with the CPU-only PyPI build.
        f'/opt/conda/bin/pip install --no-cache-dir "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold@{COLABFOLD_VERSION}"',
        # After ColabFold, to override the JAX it pins. jaxlib >= 0.5.2 carries
        # the sm_120 (Blackwell) kernels.
        '/opt/conda/bin/pip install --no-cache-dir "jax[cuda12]>=0.5.2"',
        "/opt/conda/bin/conda install -y -c conda-forge openmm pdbfixer",
        # Let the loader find conda's libstdc++ and libOpenMM.
        'echo "/opt/conda/lib" > /etc/ld.so.conf.d/conda.conf && ldconfig',
        "/opt/conda/bin/conda clean -a -y",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .env(
        PYTHONNOUSERSITE="1",
        PATH="/opt/conda/bin:$PATH",
        LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH",
    )
    .label(Version=f"colabfold-{COLABFOLD_VERSION}", CUDA="12.8", Python="3.13")
    .runscript('"$@"')
    .template_command(exec_mode="exec")
)


COLABFOLD_CONTAINER = _COLABFOLD_PORTABLE_CONTAINER
COLABFOLD_PORTABLE_CONTAINER = _COLABFOLD_PORTABLE_CONTAINER


###########################################################################
# Block
###########################################################################


class ColabFoldBlock(AF3LikePredictionBlock):
    """ColabFold (AlphaFold2) monomer and multimer structure prediction.

    Refolds all protein chains in a design with AlphaFold2 via
    ``colabfold_batch``. ``model_type=auto`` selects AF2-PTM for monomers and
    AF2-Multimer v3 when a chunk contains a complex.

    Three MSA sources are available, in order of preference for natural
    proteins:

    * ``use_local_msas=True`` — read precomputed a3m files from the shared
      sequence-hash-sharded stores (``msa_dirs``, defaulting to the site's
      ``PROTEIN_MSA_DIRS``). This is the same MSA store RF3 validation reads,
      needs no internet, and is the right choice for benchmarking against
      natural targets.
    * ``use_msa_server=True`` (or ``msa_mode``) — query the ColabFold MMseqs2
      server; requires outbound internet from the compute node.
    * neither — single-sequence AlphaFold2.

    Note: ``diffusion_batch_size`` is inherited but unused — AF2 is not a
    diffusion model. Use ``num_models`` / ``num_seeds`` instead.
    """

    _name = "colabfold"
    _display_name = "ColabFold"
    _description = "ColabFold (AlphaFold2) structure prediction"
    _script_spec = ScriptSpec(
        command=["colabfold_batch"],
        image=COLABFOLD_CONTAINER,
        source_url="https://github.com/sokrypton/ColabFold",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=200,
    )

    # ── ColabFold-specific fields ────────────────────────────────────────
    num_models: int = Field(5, ge=1, le=5, description="Number of AF2 models (1-5).")
    num_seeds: int = Field(1, ge=1, description="Number of random seeds per model.")
    random_seed: Optional[int] = Field(
        None, description="Base random seed. None = ColabFold default (0)."
    )
    msa_mode: str = Field(
        "single_sequence",
        description="MSA databases: single_sequence (no MSA, offline), "
        "mmseqs2_uniref_env, mmseqs2_uniref, mmseqs2_uniref_env_envpair.",
    )
    use_msa_server: bool = Field(
        False,
        description="Query the ColabFold MMseqs2 server for MSAs. When True and "
        "msa_mode is still single_sequence, switches to mmseqs2_uniref_env. "
        "Requires outbound internet from the compute node.",
    )
    use_local_msas: bool = Field(
        False,
        description="Look each folded sequence up in the precomputed on-disk MSA "
        "stores and stage it as an a3m input, which overrides msa_mode. Offline.",
    )
    msa_dirs: Optional[list] = Field(
        None,
        description="Sharded MSA roots searched in order, as paths or as "
        '{"dir", "extension", "directory_depth"} dicts. None uses the site\'s '
        "PROTEIN_MSA_DIRS setting.",
    )
    max_msa_sequences: Optional[int] = Field(
        10_000,
        ge=1,
        description="Truncate staged MSAs to this depth. None keeps the full store entry.",
    )
    raise_if_missing_msa_for_protein_of_length_n: Optional[int] = Field(
        None,
        ge=0,
        description="Fail when a protein at least this many residues long has no "
        "on-disk MSA. None (default) uses a query-only alignment. Set to 0 to "
        "require an MSA for every input.",
    )
    model_type: str = Field(
        "auto",
        description="AF2 model set (auto, alphafold2_ptm, alphafold2_multimer_v3, ...).",
    )
    use_templates: bool = Field(
        False, description="Query PDB templates from the MSA server (needs internet)."
    )
    amber_relax: bool = Field(
        False,
        description="Run AMBER relaxation (CPU only; --use-gpu-relax is never emitted).",
    )
    rank: str = Field(
        "auto", description="Model ranking metric (auto, plddt, ptm, iptm, multimer)."
    )
    data_dir: Optional[str] = Field(
        default_factory=lambda: get_setting_field("COLABFOLD_DATA_DIR", None),
        description="Writable dir for AF2 params (--data). ~3-4GB downloaded on first run. "
        "Bound into the container automatically when the portable image is used.",
    )

    @model_validator(mode="after")
    def _check_unsupported(self):
        # AF2 has no diffusion trunk / external checkpoint.
        self._assert_unsupported_fields_default(ckpt_path=None, num_diffusion_steps=200)
        if self.use_local_msas:
            if self.use_msa_server:
                raise ValueError(
                    "use_local_msas and use_msa_server are mutually exclusive: an "
                    "a3m input overrides --msa-mode, so the server would be queried "
                    "and then ignored"
                )
            if self._resolved_msa_dirs() is None:
                raise ValueError(
                    "use_local_msas requires msa_dirs or the PROTEIN_MSA_DIRS setting"
                )
        elif self.msa_dirs is not None:
            raise ValueError("msa_dirs has no effect unless use_local_msas is True")
        if (
            self.raise_if_missing_msa_for_protein_of_length_n is not None
            and not self.use_local_msas
        ):
            raise ValueError(
                "raise_if_missing_msa_for_protein_of_length_n requires use_local_msas"
            )
        return self

    def _resolved_msa_dirs(self) -> Optional[list]:
        if self.msa_dirs is not None:
            return normalize_msa_dirs(self.msa_dirs)
        return default_msa_dirs()

    def _effective_msa_mode(self) -> str:
        if self.use_msa_server and self.msa_mode == "single_sequence":
            return "mmseqs2_uniref_env"
        return self.msa_mode

    def _local_alignment(self, target: str, chain: dict):
        sequence = chain["sequence"]
        msa_path = resolve_msa_path(
            sequence,
            self._resolved_msa_dirs(),
            alternates=chain["msa_lookup_sequences"],
        )
        if msa_path is not None:
            return read_query_a3m(
                msa_path,
                sequence=sequence,
                max_sequences=self.max_msa_sequences,
            )

        threshold = self.raise_if_missing_msa_for_protein_of_length_n
        if threshold is not None and len(sequence) >= threshold:
            raise ValueError(
                f"No on-disk MSA for {target} chain(s) {chain['chain_ids']} "
                f"({len(sequence)} residues) in "
                f"{[entry['dir'] for entry in self._resolved_msa_dirs()]}"
            )
        logger.warning(
            "No on-disk MSA for %s chain(s) %s (%d residues); using query only",
            target,
            chain["chain_ids"],
            len(sequence),
        )
        return [">query"], [sequence]

    def _stage_input(
        self, inputdir: str, target: str, atom_array, sequences=None
    ) -> bool:
        """Write one ColabFold input and return whether it is multichain."""
        if atom_array is not None:
            chains = protein_chain_sequences(atom_array)
        else:  # sequence-only input: no structure to read modifications from
            chains = [(c, s, s) for c, s in (sequences or {}).items()]
        is_multimer = len(chains) > 1
        fasta_path = Path(inputdir) / f"{target}.fasta"
        a3m_path = Path(inputdir) / f"{target}.a3m"

        if not self.use_local_msas:
            a3m_path.unlink(missing_ok=True)
            fasta_path.write_text(colabfold_fasta(target, chains))
            return is_multimer

        fasta_path.unlink(missing_ok=True)
        unique_chains = unique_chain_sequences(chains)
        if not is_multimer:
            chain = unique_chains[0]
            headers, records = self._local_alignment(target, chain)
            a3m_path.write_text(
                "".join(
                    f"{header}\n{record}\n"
                    for header, record in zip(headers, records, strict=True)
                )
            )
            return False

        alignments = [self._local_alignment(target, chain) for chain in unique_chains]
        a3m_path.write_text(colabfold_multimer_a3m(unique_chains, alignments))
        return True

    @classmethod
    def _get_image(cls) -> Image:
        return COLABFOLD_CONTAINER

    def get_command_template(self) -> list[str]:
        # Force a writable temp path so a stale SLURM scratch TMPDIR from the
        # host doesn't break JIT/relax when running --local (same as Chai1).
        env = self._jit_safe_env()
        image = self._get_image()
        binds = None
        if self.data_dir:
            # Expose the explicitly configured AF2 parameters directory.
            data_dir = os.path.abspath(str(self.data_dir))
            binds = [(data_dir, data_dir)]
        return image.build_command(*self._script_spec.command, env=env, binds=binds)

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        # ColabFold accepts the whole input directory in one invocation and
        # uses each FASTA/A3M stem as the job name.
        has_multimer = False
        for loaded in handler.get_prediction_inputs(
            which=self.which, load_context=True
        ):
            has_multimer |= self._stage_input(
                inputdir, loaded.basename, loaded.atom_array, loaded.sequences
            )

        if has_multimer and self.model_type in {
            "alphafold2",
            "alphafold2_ptm",
            "deepfold_v1",
        }:
            raise ValueError(
                f"model_type={self.model_type!r} cannot predict a multimer; "
                "use model_type='auto' or an alphafold2_multimer model"
            )

        # A staged A3M overrides --msa-mode. Keeping the flag explicit makes
        # non-local-MSA FASTA operation deterministic as well.
        cmd = self.command_template + [
            inputdir,
            outputdir,
            "--msa-mode",
            self._effective_msa_mode(),
            "--num-models",
            str(self.num_models),
            "--num-seeds",
            str(self.num_seeds),
            "--model-type",
            self.model_type,
            "--rank",
            self.rank,
            "--overwrite-existing-results",
        ]
        if self.n_recycles is not None:
            cmd += ["--num-recycle", str(self.n_recycles)]
        if self.random_seed is not None:
            cmd += ["--random-seed", str(self.random_seed)]
        if self.use_templates:
            cmd += ["--templates"]
        if self.amber_relax:
            # CPU relax only; --use-gpu-relax is intentionally never emitted.
            cmd += ["--amber", "--num-relax", str(self.num_models)]

        data_dir = self.data_dir
        if data_dir:
            if self._get_image() is _COLABFOLD_PORTABLE_CONTAINER:
                # Must exist on the host to be bound into the container.
                os.makedirs(data_dir, exist_ok=True)
            cmd += ["--data", str(data_dir)]
        elif self._get_image() is _COLABFOLD_PORTABLE_CONTAINER:
            logger.warning(
                "COLABFOLD_DATA_DIR is not set; the portable ColabFold image "
                "carries no AF2 params (~3-4GB) and may be unable to download "
                "them at run time."
            )

        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = os.path.abspath(sandbox_dir)
        outputdir = os.path.join(sandbox_dir, "outputs")

        # ColabFold writes all jobs flat into the output dir:
        #   <design>_{unrelaxed,relaxed}_rank_001_<model>_seed_000.pdb
        #   <design>_scores_rank_001_<model>_seed_000.json
        tag = "_relaxed_rank_" if self.amber_relax else "_unrelaxed_rank_"
        rank_glob = "001" if self.read_most_confident_model else "[0-9][0-9][0-9]"
        paths = glob.glob(os.path.join(outputdir, f"*{tag}{rank_glob}_*.pdb"))

        self._guard_outputs(paths, outputdir)

        rows = []
        for path in sorted(paths):
            fname = os.path.basename(path)
            design_name, _, rest = fname.partition(tag)
            rank_str = rest.split("_")[0]  # e.g. "001"

            if self.read_most_confident_model:
                row_name = design_name
                suffix = "_model"
            else:
                row_name = f"{design_name}_rank_{rank_str}"
                suffix = f"_rank_{rank_str}"

            # Symlink with the design name so merge_rows can match lineage.
            linked_path = os.path.join(outputdir, f"{design_name}{suffix}.pdb")
            if not os.path.exists(linked_path):
                os.symlink(os.path.basename(path), linked_path)

            # rank is ColabFold's own pLDDT ordering (001 = its best model) and
            # model_name identifies which of the five AF2 parameter sets it was.
            # Both are needed to aggregate best-of-1 / best-of-N downstream
            # without re-parsing row names.
            metrics = {"rank": int(rank_str)}
            model_match = re.search(rf"{re.escape(tag)}\d+_(.+)_seed_\d+\.pdb$", fname)
            if model_match:
                metrics["model_name"] = model_match.group(1)
            score_matches = glob.glob(
                os.path.join(outputdir, f"{design_name}_scores_rank_{rank_str}_*.json")
            )
            if score_matches:
                try:
                    with open(score_matches[0]) as f:
                        scores = json.load(f)
                    if "plddt" in scores:
                        metrics["plddt"] = float(np.mean(scores["plddt"]))
                    for key in ("ptm", "iptm", "max_pae"):
                        if key in scores and scores[key] is not None:
                            metrics[key] = float(scores[key])
                except Exception as e:
                    logger.warning(
                        f"Failed to load scores from {score_matches[0]}: {e}"
                    )

            row_metrics = {self.name: metrics}
            rows.append(
                make_row(
                    name=row_name,
                    path=str(linked_path),
                    metrics=row_metrics,
                    as_metric=self.compile_as_metric,
                )
            )

        return PipelineDataHandler(rows)


if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "colabfold"
    SANDBOX = REPO_ROOT / "unittest_outs" / "colabfold"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = ColabFoldBlock(num_models=1)
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
