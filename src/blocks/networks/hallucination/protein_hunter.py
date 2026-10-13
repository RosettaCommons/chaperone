"""Protein Hunter block for de novo protein binder design using Boltz-based hallucination.

Protein Hunter uses a **forked copy of Boltz2** with two custom parameters injected
into the model's ``forward()`` method:

  - ``randomly_kill_helix_feature`` (bool, default False): When enabled, scales the
    pair representation z at positions (i, i+4) — the alpha-helix H-bond spacing — by
    a random negative factor for residues on the binder chain only. This disrupts helix
    formation during diffusion, encouraging more diverse secondary structure.

  - ``negative_helix_constant`` (float, default 0.2): Controls the magnitude of the
    scaling factor applied above (``random_factor = -constant * rand()``).

The patch lives in ``boltz_ph/src/boltz/model/models/boltz2.py`` (PH's fork) and
modifies the pair embeddings before the diffusion sampling loop::

    if randomly_kill_helix_feature:
        binder_mask = feats["entity_id"] == chain_to_number[binder_chain]
        for i in range(seq_len - 4):
            if binder_mask[0, i]:
                random_factor = -negative_helix_constant * torch.rand(1).item()
                z[:, i, i+4, :] = random_factor * z[:, i, i+4, :]
                z[:, i+4, i, :] = random_factor * z[:, i+4, i, :]

These parameters are **not available** in upstream Boltz2's ``boltz predict`` CLI.
When ``randomly_kill_helix_feature=False`` (the default), the code path is never
executed and standard Boltz2 behaviour is identical. To replicate the ``True`` case
outside Protein Hunter, one would need to monkey-patch or fork the Boltz2 model's
forward method.
"""

import csv
import glob
import logging
import os
import re

from pydantic import Field

from blocks.constants import CUDA_INSTALL, cuda_python_base, cuda_venv_env
from pipelines import (
    CompiledBlockFunction,
    JobArgs,
    PipelineDataHandler,
    get_setting_field,
)
from pipelines.block_config import TIER_PRIMARY
from pipelines.common import try_float
from pipelines.containers import Image
from pipelines.data import make_row
from pipelines.data.schema import Column, DesignMetadata, merge_rows

logger = logging.getLogger(__name__)

_PROTEIN_HUNTER_COMMIT = "a3e048e96c93953f022d15a4d35cd744b7e39e41"
_TORCH_VERSION = "2.11.0"

###########################################################################
# Container
###########################################################################

PROTEIN_HUNTER_CONTAINER = (
    # Shared base + apt packages; CUDA/venv handled inline because Python 3.10
    # must be installed from the deadsnakes PPA before the venv is created.
    cuda_python_base(
        Image("protein_hunter"),
        "protein-hunter",
        create_venv=False,
        cuda_install=[],
        packages=(
            "build-essential",
            "git",
            "wget",
            "curl",
            "ca-certificates",
            "gnupg",
            "python3-pip",
            "software-properties-common",
        ),
    )
    .run(
        # Add deadsnakes PPA and install Python 3.10 (not in Ubuntu 24.04 default repos)
        "add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && "
        "apt-get install -y python3.10 python3.10-dev python3.10-venv",
    )
    .run(*CUDA_INSTALL)
    .run(
        "python3.10 -m venv /opt/protein-hunter-env",
        ". /opt/protein-hunter-env/bin/activate && "
        "pip install --no-cache-dir --upgrade pip setuptools wheel",
    )
    .run(
        # CUDA 12.8 is required for Blackwell (sm_120). Keep the wheel and
        # toolkit on the same CUDA minor version; the former cu126/latest-torch
        # combination could not run across this cluster's GPU generations.
        ". /opt/protein-hunter-env/bin/activate && "
        f"pip install --no-cache-dir torch=={_TORCH_VERSION} "
        "--index-url https://download.pytorch.org/whl/cu128",
    )
    .git_clone("https://github.com/yehlincho/Protein-Hunter.git", "/opt/protein_hunter")
    .run(
        f"git -C /opt/protein_hunter checkout {_PROTEIN_HUNTER_COMMIT} && "
        f'test "$(git -C /opt/protein_hunter rev-parse HEAD)" = "{_PROTEIN_HUNTER_COMMIT}"',
        ". /opt/protein-hunter-env/bin/activate && "
        "cd /opt/protein_hunter/boltz_ph && pip install --no-cache-dir -e .",
    )
    .run(
        # Install additional dependencies
        ". /opt/protein-hunter-env/bin/activate && "
        "pip install --no-cache-dir "
        "matplotlib seaborn prody tqdm PyYAML requests pypdb py3Dmol",
    )
    .run(
        # Setup LigandMPNN model weights
        "cd /opt/protein_hunter/LigandMPNN && bash get_model_params.sh ./model_params",
    )
    .run(
        # Boltz 2.2.1 requires numba 0.61.0. Pin its matching llvmlite and a
        # NumPy version accepted by both Boltz and LigandMPNN.
        ". /opt/protein-hunter-env/bin/activate && "
        "pip install --no-cache-dir --upgrade "
        '"numpy==1.26.4" "numba==0.61.0" "llvmlite==0.44.0" '
        '"ml-collections==1.0.0"',
    )
    .run(
        "chmod 0755 /opt/protein_hunter/utils/DAlphaBall.gcc",
        "test -s /opt/protein_hunter/LigandMPNN/model_params/proteinmpnn_v_48_020.pt",
        "test -s /opt/protein_hunter/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        # Upstream uses script-relative imports (for example, ``model_utils``)
        # rather than package-relative imports. Validate from the same working
        # directory used by design.py so both boltz_ph and the repository root
        # are on sys.path.
        ". /opt/protein-hunter-env/bin/activate && "
        "cd /opt/protein_hunter/boltz_ph && "
        'python -c "import ml_collections, numba, torch; '
        f"assert torch.__version__.startswith('{_TORCH_VERSION}'); "
        "assert torch.version.cuda == '12.8'; "
        "from pipeline import ProteinHunter_Boltz; "
        'from LigandMPNN.wrapper import LigandMPNNWrapper"',
        ". /opt/protein-hunter-env/bin/activate && "
        "cd /opt/protein_hunter/boltz_ph && python design.py --help >/dev/null",
        ". /opt/protein-hunter-env/bin/activate && pip check",
        "mkdir -p /opt/protein_hunter/manifest",
        ". /opt/protein-hunter-env/bin/activate && "
        "pip freeze > /opt/protein_hunter/manifest/pip-freeze.txt",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
)
PROTEIN_HUNTER_CONTAINER = (
    cuda_venv_env(PROTEIN_HUNTER_CONTAINER, "/opt/protein-hunter-env")
    .env(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", TMPDIR="/tmp")
    .label(
        Version=f"main-{_PROTEIN_HUNTER_COMMIT[:12]}",
        Revision=_PROTEIN_HUNTER_COMMIT,
        CUDA="12.8",
        Source="https://github.com/yehlincho/Protein-Hunter",
        License="MIT",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[("/mnt/home", "/mnt/home"), ("/projects", "/projects")],
        env={},
    )
)


###########################################################################
# Block
###########################################################################


class ProteinHunterBlock(CompiledBlockFunction):
    """Protein Hunter: de novo protein binder design via structure hallucination with Boltz.

    Iteratively predicts structure with Boltz2, redesigns sequence with LigandMPNN,
    and selects high-confidence binders. Supports protein, ligand, DNA/RNA, and
    multimer targets.

    Reference: Cho et al., "Protein Hunter: exploiting structure hallucination
    within diffusion for protein design" (2025).
    """

    _name = "protein_hunter"
    _display_name = "Protein Hunter"
    _description = "De novo binder design via Boltz structure hallucination"
    _image = PROTEIN_HUNTER_CONTAINER
    _resources = JobArgs(
        partition="gpu",
        gres="gpu:small:1",
        mem_gb=32,
        rows_per_job=10,
    )
    _requires_input = False
    _inline = False
    _chunked = True

    # Target specification
    protein_seqs: str = Field(
        "",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Target protein sequence(s). Separate multiple chains with ':'.",
    )
    ligand_ccd: str = Field(
        "", description="CCD code for small molecule target (e.g., 'SAM')."
    )
    ligand_smiles: str = Field(
        "", description="SMILES string for small molecule target."
    )
    nucleic_seq: str = Field("", description="DNA/RNA target sequence.")
    nucleic_type: str = Field("dna", description="Nucleic acid type: 'dna' or 'rna'.")

    # Design parameters
    name: str = Field("protein_hunter_design", description="Job name for output files.")
    num_designs: int = Field(
        10,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of independent design trajectories.",
    )
    num_cycles: int = Field(
        5,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of predict-redesign cycles per trajectory.",
    )
    emit_all_cycles: bool = Field(
        False,
        description=(
            "Emit one handler row per predicted cycle instead of one multi-structure "
            "lineage row per trajectory. Use this when every intermediate cycle must "
            "be scored independently."
        ),
    )
    min_protein_length: int = Field(
        100,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Minimum binder length (residues).",
    )
    max_protein_length: int = Field(
        150,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Maximum binder length (residues).",
    )
    percent_X: int = Field(
        90, description="Percentage of X (unknown) residues in initial sequence."
    )
    seq: str = Field(
        "", description="Starting binder sequence (for refiner mode). Empty = random."
    )
    refiner_mode: bool = Field(
        False,
        description=(
            "Refine the sequence supplied by seq instead of initializing a de novo "
            "binder trajectory."
        ),
    )

    # MSA and model
    msa_mode: str = Field("mmseqs", description="MSA mode: 'mmseqs' or 'single'.")
    boltz_model_version: str = Field("boltz2", description="Boltz model version.")
    diffuse_steps: int = Field(
        200,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of diffusion steps.",
    )
    recycling_steps: int = Field(
        3,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of recycling steps.",
    )

    # Filtering thresholds
    high_iptm_threshold: float = Field(
        0.8, description="ipTM threshold for saving designs."
    )
    high_plddt_threshold: float = Field(
        0.8, description="pLDDT threshold for saving designs."
    )

    # Design options
    cyclic: bool = Field(False, description="Enable cyclic peptide design.")
    alanine_bias: bool = Field(
        False, description="Apply negative alanine bias during design."
    )
    randomly_kill_helix_feature: bool = Field(
        False,
        description=(
            "Disrupt binder i/i+4 pair features during diffusion to reduce "
            "alpha-helical collapse and explore alternative topologies."
        ),
    )
    negative_helix_constant: float = Field(
        0.2,
        ge=0.0,
        description=(
            "Magnitude of the random negative i/i+4 pair-feature scaling used "
            "when randomly_kill_helix_feature is enabled."
        ),
    )
    omit_AA: str = Field(
        "C", description="Amino acids to omit from design (e.g., 'C')."
    )
    temperature: float = Field(
        0.1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="LigandMPNN sampling temperature.",
    )
    contact_residues: str = Field(
        "",
        description="Target chain residue positions for contact specification. "
        "Comma-separated, pipe-separated for multiple chains (e.g., '1,2,5|3,10').",
    )

    # Template
    template_path: str = Field(
        "", description="PDB code or path(s) for template conditioning."
    )
    template_cif_chain_id: str = Field(
        "", description="Chain ID(s) for mmCIF templates."
    )

    # Boltz data paths (default from BOLTZ_CACHE_DIR setting, fallback ~/.boltz)
    ccd_path: str = Field(
        default_factory=lambda: os.path.join(
            get_setting_field("BOLTZ_CACHE_DIR", os.path.expanduser("~/.boltz")), "mols"
        ),
        description="Path to CCD molecule data directory (.pkl files) for Boltz.",
    )
    boltz_model_path: str = Field(
        default_factory=lambda: os.path.join(
            get_setting_field("BOLTZ_CACHE_DIR", os.path.expanduser("~/.boltz")),
            "boltz2_conf.ckpt",
        ),
        description="Path to Boltz model checkpoint.",
    )

    # GPU
    gpu_id: int = Field(0, description="GPU device ID.")

    def get_command_template(self) -> list[str]:
        cmd = self._image.build_command(
            "python",
            "/opt/protein_hunter/boltz_ph/design.py",
        )
        # Insert --pwd and --writable-tmpfs before the container image ref.
        # --pwd: LigandMPNN uses relative paths (./LigandMPNN/run.py)
        # --writable-tmpfs: Protein Hunter writes to /tmp at runtime
        try:
            img_idx = next(
                i
                for i, c in enumerate(cmd)
                if c.startswith("${CONTAINER:") or c.endswith(".sif")
            )
            cmd.insert(img_idx, "--writable-tmpfs")
            cmd.insert(img_idx, "/opt/protein_hunter")
            cmd.insert(img_idx, "--pwd")
        except StopIteration:
            pass
        return cmd

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        """Build the Protein Hunter CLI command for one design trajectory."""
        sandbox_dir = os.path.abspath(sandbox_dir)

        cmd = self.command_template + [
            "--name",
            f"{self.name}_chunk{chunk_id}",
            "--save_dir",
            sandbox_dir,
            "--gpu_id",
            str(self.gpu_id),
            "--num_designs",
            str(len(handler)),
            "--num_cycles",
            str(self.num_cycles),
            "--min_protein_length",
            str(self.min_protein_length),
            "--max_protein_length",
            str(self.max_protein_length),
            "--percent_X",
            str(self.percent_X),
            "--msa_mode",
            self.msa_mode,
            "--boltz_model_version",
            self.boltz_model_version,
            "--diffuse_steps",
            str(self.diffuse_steps),
            "--recycling_steps",
            str(self.recycling_steps),
            "--high_iptm_threshold",
            str(self.high_iptm_threshold),
            "--high_plddt_threshold",
            str(self.high_plddt_threshold),
            "--omit_AA",
            self.omit_AA,
            "--temperature",
            str(self.temperature),
        ]

        # Target specification
        if self.protein_seqs:
            cmd.extend(["--protein_seqs", self.protein_seqs])
        if self.ligand_ccd:
            cmd.extend(["--ligand_ccd", self.ligand_ccd])
        if self.ligand_smiles:
            cmd.extend(["--ligand_smiles", self.ligand_smiles])
        if self.nucleic_seq:
            cmd.extend(["--nucleic_seq", self.nucleic_seq])
            cmd.extend(["--nucleic_type", self.nucleic_type])
        if self.seq:
            cmd.extend(["--seq", self.seq])
        if self.refiner_mode:
            cmd.append("--refiner_mode")

        # Template
        if self.template_path:
            tp = str(self.stage_file(self.template_path, sandbox_dir))
            cmd.extend(["--template_path", tp])
        if self.template_cif_chain_id:
            cmd.extend(["--template_cif_chain_id", self.template_cif_chain_id])

        # Contact specification
        if self.contact_residues:
            cmd.extend(["--contact_residues", self.contact_residues])

        # Boltz data paths
        cmd.extend(["--ccd_path", self.ccd_path])
        cmd.extend(["--boltz_model_path", self.boltz_model_path])

        # Boolean flags
        if self.cyclic:
            cmd.append("--cyclic")
        if self.alanine_bias:
            cmd.append("--alanine_bias")
        if self.randomly_kill_helix_feature:
            cmd.extend(
                [
                    "--randomly_kill_helix_feature",
                    "--negative_helix_constant",
                    str(self.negative_helix_constant),
                ]
            )

        return cmd

    def init_handler(self):
        """Create placeholder rows for de novo design trajectories."""
        rows = []
        for i in range(self.num_designs):
            rows.append({Column.STRUCTURES: [], Column.METRICS: "{}"})
        return PipelineDataHandler(rows)

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        """Collect Protein Hunter outputs into a PipelineDataHandler.

        Protein Hunter outputs:
        # Ignore the high iptm
        # - PDB files in high_iptm_pdb/ (high-confidence designs)
        # - CSV metrics in summary_high_iptm.csv and summary_all_runs.csv
        - Per-run PDB files in 0_protein_hunter_design/run_*/
        """
        sandbox_dir = os.path.abspath(sandbox_dir)
        rows = []

        # # First try high-confidence designs from high_iptm_pdb directories
        # high_iptm_pdbs = []
        # for dirpath, _, filenames in os.walk(sandbox_dir):
        #     if "high_iptm_pdb" in os.path.basename(dirpath):
        #         for f in filenames:
        #             if f.endswith(".pdb"):
        #                 high_iptm_pdbs.append(os.path.join(dirpath, f))

        # Load metrics from summary CSVs
        # 1) summary_high_iptm.csv: keyed by pdb_filename
        metrics_by_pdb = {}
        # 2) summary_all_runs.csv: keyed by run_id, with per-cycle metrics
        metrics_by_run = {}
        for dirpath, _, filenames in os.walk(sandbox_dir):
            for f in filenames:
                csv_path = os.path.join(dirpath, f)
                if f == "summary_high_iptm.csv":
                    try:
                        with open(csv_path) as fh:
                            reader = csv.DictReader(fh)
                            for row in reader:
                                pdb_fn = row.get("pdb_filename", "")
                                metrics_by_pdb[pdb_fn] = {
                                    k: try_float(v)
                                    for k, v in row.items()
                                    if k
                                    not in ("sequence", "pdb_filename", "yaml_filename")
                                }
                    except Exception as e:
                        logger.warning(f"Failed to parse {csv_path}: {e}")
                elif f == "summary_all_runs.csv":
                    try:
                        with open(csv_path) as fh:
                            reader = csv.DictReader(fh)
                            for row in reader:
                                run_id = row.get("run_id", "")
                                # Extract best cycle metrics and per-cycle data
                                run_metrics = {}
                                for k, v in row.items():
                                    if k in ("run_id",):
                                        continue
                                    # Skip sequence columns (long strings)
                                    if k.endswith("_seq") or k == "best_seq":
                                        continue
                                    run_metrics[k] = try_float(v)
                                metrics_by_run[str(run_id)] = run_metrics
                    except Exception as e:
                        logger.warning(f"Failed to parse {csv_path}: {e}")

        # if high_iptm_pdbs:
        #     for idx, pdb_path in enumerate(sorted(high_iptm_pdbs)):
        #         basename = os.path.basename(pdb_path)
        #         metrics = metrics_by_pdb.get(basename, {})
        #         rows.append(
        #             make_row(
        #                 name=f'{self.name}/cycle_{idx}',
        #                 path=pdb_path,
        #                 metadata={},
        #                 metrics={self.name: metrics},
        #             )
        #         )
        # else:
        # Collect all cycle PDBs and group by run directory
        all_pdbs = sorted(
            glob.glob(
                os.path.join(
                    sandbox_dir,
                    "**/0_protein_hunter_design/run_*/*predicted_cycle*.pdb",
                ),
                recursive=True,
            )
        )
        runs: dict[str, list[str]] = {}
        for p in all_pdbs:
            run_dir = os.path.dirname(p)
            runs.setdefault(run_dir, []).append(p)

        for run_dir, pdbs in runs.items():
            run_match = re.search(r"run_(\d+)", run_dir)
            run_id = run_match.group(1) if run_match else ""
            metrics = metrics_by_run.get(run_id, {})

            # Sort by cycle index (files end in cycle_{N}.pdb)
            def _cycle_key(p):
                m = re.search(r"cycle_(\d+)\.pdb$", p)
                return int(m.group(1)) if m else 0

            pdbs = sorted(pdbs, key=_cycle_key)

            # Annotate ligand metadata so downstream blocks can find ligand atoms.
            ligand_meta = {}
            if self.ligand_ccd or self.ligand_smiles:
                ligand_meta[DesignMetadata.LIGAND] = ["LIG"]

            if self.emit_all_cycles:
                for pdb_path in pdbs:
                    cycle = _cycle_key(pdb_path)
                    prefix = f"cycle_{cycle}_"
                    cycle_metrics = {
                        key.removeprefix(prefix): value
                        for key, value in metrics.items()
                        if key.startswith(prefix)
                    }
                    rows.append(
                        make_row(
                            name=self.name,
                            path=pdb_path,
                            metadata={
                                **ligand_meta,
                                "protein_hunter_run_id": run_id,
                                "protein_hunter_cycle": cycle,
                            },
                            metrics={self.name: cycle_metrics},
                        )
                    )
            else:
                cycle_rows = [
                    make_row(
                        name=f"{self.name}/cycle_{_cycle_key(pdb_path)}",
                        path=pdb_path,
                        metadata=ligand_meta,
                        metrics={self.name: metrics} if pdb_path == pdbs[-1] else {},
                    )
                    for pdb_path in pdbs
                ]
                # A zero-optimization-cycle diagnostic still produces the
                # initial cycle-0 prediction. ``merge_rows`` intentionally
                # requires at least two rows, so preserve that single row
                # directly while retaining normal lineage merging for full
                # trajectories.
                rows.append(
                    cycle_rows[0] if len(cycle_rows) == 1 else merge_rows(*cycle_rows)
                )

        if not rows:
            raise RuntimeError(
                f"Protein Hunter produced no output PDB files in {sandbox_dir}. "
                f"Check logs for errors."
            )

        return PipelineDataHandler(rows)
