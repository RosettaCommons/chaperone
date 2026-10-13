"""Settings for Chaperone.

All configuration flows through ChaperoneSettings. Add new env vars here as
fields rather than reading os.environ directly in blocks.
"""

from __future__ import annotations

import getpass
import os
import shutil
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_files() -> tuple[str, ...]:
    """Load portable defaults, an explicit site, then user overrides.

    CHAPERONE_SITE=digs opts into packaged defaults; a supplied .env.ipd is
    loaded when present. Shared cluster directories never select a site. CHAPERONE_ENV_FILE can point at a shared site/user file;
    .env.local and process environment values take precedence.
    """
    env_files: list[str] = []
    project_root = next(
        (p for p in Path(__file__).resolve().parents if (p / ".env").is_file()),
        Path.cwd(),
    )
    if (project_root / ".env").is_file():
        env_files.append(str(project_root / ".env"))
    site = os.environ.get("CHAPERONE_SITE", "portable")
    if site not in {"portable", "digs"}:
        raise ValueError("CHAPERONE_SITE must be portable or digs")
    if site == "digs":
        env_files.append(str(Path(__file__).parent / "site_profiles" / "digs.env"))
    # A supplied (untracked) IPD file is an explicit opt-in, even without a site
    # flag. Installed wheels also accept the file in the working directory.
    ipd_file = project_root / ".env.ipd"
    if ipd_file.is_file():
        env_files.append(str(ipd_file))
    home_env = Path("~/.env.chaperone").expanduser()
    if home_env.is_file():
        env_files.append(str(home_env))
    configured = os.environ.get("CHAPERONE_ENV_FILE")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"CHAPERONE_ENV_FILE does not exist: {path}")
        env_files.append(str(path))
    if (project_root / ".env.local").is_file():
        env_files.append(str(project_root / ".env.local"))
    return tuple(env_files)


_VALID_CONTAINER_BACKENDS = {"apptainer", "docker"}


def _resolve_container_backend(value: str) -> str:
    """Resolve container backend from a slash-separated priority string.

    Accepts 'apptainer', 'docker', 'apptainer/docker', or 'docker/apptainer'.
    Returns the first backend that is available on the system.
    Falls back to the first entry if none are found (will fail at runtime with clear error).
    """
    backends = [b.strip() for b in value.split("/")]
    for b in backends:
        if b not in _VALID_CONTAINER_BACKENDS:
            raise ValueError(
                f"Unknown container backend: {b!r}. Must be one of {sorted(_VALID_CONTAINER_BACKENDS)}."
            )
    for b in backends:
        if shutil.which(b):
            return b
    return backends[0]


def _get_default_chaperone_root() -> Path:
    """Get default CHAPERONE_ROOT, resolving user properly."""
    user = getpass.getuser()
    if os.environ.get("CHAPERONE_SITE") == "digs":
        return Path(f"/net/scratch/{user}/chaperone")
    return Path.cwd()


class ChaperoneSettings(BaseSettings):
    """Central configuration for Chaperone.

    All env vars go here. In .env files, use the field name directly
    (no prefix needed). Shell env vars also work without a prefix.

    Example: ``RUNDIR=/path/to/rundir`` in ``.env`` or ``export RUNDIR=...``.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        env_parse_none_str="None",
        extra="ignore",
        validate_assignment=True,
        env_file=_find_env_files(),
        env_file_encoding="utf-8",
    )

    # ---------------------------
    # Core Settings
    # ---------------------------
    RUNDIR: Path | None = Field(
        default=None,
        description="Default directory for pipeline execution outputs.",
    )
    CHAPERONE_ROOT: Path = Field(
        default_factory=_get_default_chaperone_root,
        description="Root directory for resolving relative rundirs. Defaults to the working directory (digs: /net/scratch/{user}/chaperone).",
    )
    ENFORCE_RELATIVE_RUNDIR: bool = Field(
        default=False,
        description="When True, raise an error if an absolute rundir is provided.",
    )
    ENGINE_RECOMPILE: bool = Field(
        default=False,
        description="Default value for recompile.",
    )
    ATOMWORKS_CACHE_OVERRIDE: Path | None = Field(
        default=None,
        alias="CACHE_DIR",
        description="Explicit cache directory. If None, defaults to RUNDIR/.cache when RUNDIR is set.",
    )
    ENABLE_ATOMWORKS_CACHE: bool = Field(
        default=False,
        description="Enable caching of parsed atom arrays.",
    )
    ENABLE_LOCAL_LOGS: bool = Field(
        default=True,
        description="Write log files for local block execution to {rundir}/logs/{block_name}/.",
    )
    DEBUG_CSV_MAX_ROWS: int = Field(
        default=100,
        description="Skip debug CSV for DataFrames larger than this. 0=disable, -1=always write.",
    )

    @computed_field
    @property
    def CACHE_DIR(self) -> Path | None:
        """Get cache directory, defaulting to RUNDIR/.cache if not explicitly set."""
        cache_dir = None
        if self.ATOMWORKS_CACHE_OVERRIDE is not None:
            cache_dir = self.ATOMWORKS_CACHE_OVERRIDE
        elif self.RUNDIR is not None:
            cache_dir = self.RUNDIR / ".cache"

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir

    # ---------------------------
    # Container / Compute Backend
    # ---------------------------
    CHAPERONE_CONTAINER_BACKEND: Literal["apptainer", "docker"] = Field(
        default="apptainer/docker",
        description="Container runtime. Supports slash-separated priority order.",
    )

    @field_validator("CHAPERONE_CONTAINER_BACKEND", mode="before")
    @classmethod
    def _resolve_container_backend_validator(cls, v: str) -> str:
        return _resolve_container_backend(str(v))

    CHAPERONE_CONTAINER_DIRS: str = Field(
        default="",
        description="Colon-separated list of directories to search for container images.",
    )
    CHAPERONE_CONTAINER_BIND_RUNDIR: bool = Field(
        default=True,
        description="Bind the exact active RUNDIR into tool containers, once the engine selects it.",
    )
    CHAPERONE_CONTAINER_BIND_WORKSPACE: bool = Field(
        default=True,
        description=(
            "Bind the exact work folder into tool containers. Evaluations always use "
            "their controller-provided HARBOR_HOST_WORKSPACE."
        ),
    )
    CHAPERONE_CONTAINER_WORKSPACE: Optional[Path] = Field(
        default=None,
        description=(
            "Ordinary work folder containing source scripts and inputs; defaults to the "
            "working directory. Ignored when HARBOR_HOST_WORKSPACE is supplied."
        ),
    )

    CHAPERONE_DISABLE_RICH: bool = Field(
        default=False,
        description="Disable rich formatting in console output.",
    )

    CHAPERONE_COMPUTE_BACKEND: Literal["auto", "slurm", "local"] = Field(
        default="auto",
        description="Default compute backend. 'auto' detects local GPU → SLURM → local fallback.",
    )
    CHAPERONE_SLURM_PYTHON: Optional[str] = Field(
        default=None,
        description=(
            "Python command Submitit should run inside Slurm allocations. "
            "Use a shared launcher when the submission Python lives in a container."
        ),
    )
    CHAPERONE_SLURM_ARRAY_PARALLELISM: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Default maximum number of simultaneously running Slurm array tasks. "
            "JobArgs.slurm_array_parallelism overrides this value."
        ),
    )
    # Site mapping for Slurm resources. Blocks declare an abstract need (a CPU or
    # GPU job and a GPU size class); these settings turn it into the concrete
    # partition and --gres string for this cluster. Unset means "portable":
    # no --partition (the cluster default applies) and --gres=gpu:1.
    CHAPERONE_DEFAULT_PARTITION: Optional[str] = Field(
        default=None,
        description=(
            "Slurm partition for jobs that do not name one: CPU blocks, bare "
            "JobArgs(), and GPU blocks when CHAPERONE_GPU_PARTITION is unset. "
            "Unset omits --partition so the cluster default applies."
        ),
    )
    CHAPERONE_GPU_PARTITION: Optional[str] = Field(
        default=None,
        description=(
            "Slurm partition for GPU blocks. Falls back to CHAPERONE_DEFAULT_PARTITION."
        ),
    )
    CHAPERONE_GRES_MAP: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "JSON map from a block's GPU size class (JobArgs.gpu_class, e.g. "
            '"small", "large") to the --gres string for this cluster, e.g. '
            '\'{"small": "gpu:a4000:1", "large": "gpu:a6000:1"}\'. The key '
            '"default" covers classes that are not listed. Anything not mapped '
            "requests one generic GPU (gpu:1)."
        ),
    )
    CHAPERONE_DEFAULT_ACCOUNT: Optional[str] = Field(
        default=None,
        description="Slurm --account for jobs that do not set JobArgs.account.",
    )
    CHAPERONE_DEFAULT_QOS: Optional[str] = Field(
        default=None,
        description="Slurm --qos for jobs that do not set JobArgs.qos.",
    )
    CHAPERONE_DEFAULT_EXTRA_SBATCH: Optional[str] = Field(
        default=None,
        description=(
            "Extra sbatch long options for jobs that do not set "
            "JobArgs.extra_sbatch, shell-split, e.g. "
            "'--reservation=gpu_mine --mail-type=END'."
        ),
    )

    # ---------------------------
    # Tool / Weight Paths
    # ---------------------------
    HF_HUB_CACHE: Optional[str] = Field(
        default=None,
        description="HuggingFace hub cache directory for model weights.",
    )
    BOLTZ_CACHE_DIR: str = Field(
        default_factory=lambda: os.path.expanduser("~/.boltz"),
        description="Directory containing Boltz model weights and CCD data.",
    )
    CHAI_DOWNLOADS_DIR: Optional[str] = Field(
        default=None,
        description="Directory for Chai-1 model weights.",
    )
    X3DNA_BIN_DIR: Optional[str] = Field(
        default=None,
        description="Directory containing x3dna-dssr and x3dna-snap.",
    )
    X3DNA_DIR: Optional[str] = Field(
        default=None,
        description="3DNA install root exported as $X3DNA for its binaries.",
    )
    DALPHABALL_BIN: Optional[str] = Field(
        default=None,
        description="DAlphaBall executable required by the ligand_ddg Holes metric.",
    )
    ESM_WEIGHTS_DIR: str = Field(
        default_factory=lambda: str(Path("~/.cache/chaperone/esm").expanduser()),
        description="HuggingFace cache directory for ESMFold2 / ESMC model weights "
        "(used as HF_HUB_CACHE by the esmfold2 block).",
    )
    SIMPLEFOLD_WEIGHTS_DIR: str = Field(
        default_factory=lambda: str(Path("~/.cache/chaperone/simplefold").expanduser()),
        description=(
            "Shared cache for SimpleFold checkpoints, its fair-esm ESM2 weights, "
            "and the CCD dictionary."
        ),
    )
    COLABFOLD_DATA_DIR: Optional[str] = Field(
        default=None,
        description="Writable directory for ColabFold/AlphaFold2 params (--data). "
        "~3-4GB downloaded on first run.",
    )
    PROTEIN_MSA_DIRS: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Sharded on-disk protein MSA roots, searched in order and keyed by "
            "sequence hash. JSON list of "
            '{"dir": ..., "extension": ".a3m.gz", "directory_depth": 2}, matching '
            "the paths.data.protein_msa_dirs entries RF3 validation uses."
        ),
    )
    FOUNDRY_CHECKPOINT_DIRS: Optional[str] = Field(
        default=None,
        description="Colon-separated list of directories for Foundry model checkpoints.",
    )
    FOUNDRY_DEV_PATH: Optional[str] = Field(
        default=None,
        description="Path to local foundry-dev repository for RF3DevBlock.",
    )
    FOUNDRY_DEV_SIF: Optional[str] = Field(
        default=None,
        description="Path to the foundry-dev (deps-only) apptainer SIF for RF3DevBlock.",
    )
    AF3_JAX_CACHE_DIR: Optional[str] = Field(
        default=None,
        description="Persistent directory for JAX compilation cache (speeds up repeated AF3 runs).",
    )
    OPENFOLD3_CACHE_DIR: str = Field(
        default_factory=lambda: os.path.expanduser("~/.openfold3"),
        description="Directory containing OpenFold3 model weights (downloaded via setup_openfold).",
    )
    PROTENIX_CACHE_DIR: str = Field(
        default_factory=lambda: os.path.expanduser("~/.protenix"),
        description="Root directory for Protenix data (CCD, checkpoints). Maps to PROTENIX_ROOT_DIR inside container.",
    )
    ALPHAFOLD3_WEIGHTS_DIR: Optional[str] = Field(
        default=None,
        description=(
            "Directory containing AlphaFold3 model weights (license-gated; "
            "request access from DeepMind)."
        ),
    )
    CHAPERONE_MPNN_CONTAINER: Literal["auto", "public", "portable", "legacy"] = Field(
        default="auto",
        description=(
            "MPNN runtime selection. Auto uses the public MIT LigandMPNN image "
            "unless MPNN_WEIGHTS_DIR supplies a legacy site profile, in which case "
            "it prefers modelhub_portable.sif, then the existing IPD image. "
            "Public forces public source and bundled public checkpoints; "
            "portable forces the rebuildable private fused_mpnn image; "
            "legacy explicitly selects the existing IPD Apptainer image. "
            "Set before importing blocks."
        ),
    )
    MPNN_WEIGHTS_DIR: Optional[str] = Field(
        default=None,
        description=(
            "Directory containing LigandMPNN/fused_mpnn checkpoints "
            "(e.g. vanilla_model_weights/, ligand_mpnn_model_weights/, "
            "packer_weights/). When unset, runs fall back to the paths "
            "provided by the existing Apptainer image."
        ),
    )
    CCD_MIRROR_PATH: Optional[str] = Field(
        default=None,
        description="Path to local CCD mirror directory.",
    )
    PDB_MIRROR_PATH: Optional[str] = Field(
        default=None,
        description="Path to local PDB mirror directory.",
    )
    DUNBRACK_LIBRARY_COPY: Optional[str] = Field(
        default=None,
        description="Path to Dunbrack rotamer library.",
    )
    HBPLUS_COPY: str = Field(
        default="hbplus",
        description="Path to HBPLUS binary.",
    )
    ROSETTA_BIN: str = Field(
        default="rosetta_scripts",
        description="Path to Rosetta rosetta_scripts binary.",
    )
    PYROSETTA_PATH: Optional[str] = Field(
        default=None,
        description="Path to the PyRosetta 'setup' directory. When set, "
        "blocks.utils.pyrosetta adds it to sys.path before importing PyRosetta.",
    )
    FOLDSEEK_EXEC: str = Field(
        default="foldseek",
        description="Path to foldseek executable.",
    )
    USALIGN_BIN: str = Field(
        default="USalign",
        description="Path to the US-align executable used by tm_score_rna. "
        "US-align is a single C++ file rather than a package; build it with "
        "src/blocks/scripts/install_usalign.sh, which drops it in the venv's bin.",
    )
    LIGAND_PARAMS_DIR: Optional[str] = Field(
        default=None,
        description="Directory for Rosetta ligand .params files. Falls back to bundled params.",
    )
    MOLFILE_TO_PARAMS_SCRIPT: Optional[str] = Field(
        default=None,
        description="Path to Rosetta's molfile_to_params.py. Required for "
        "dynamic .params generation (generate_params / ligand_ddg_soft).",
    )
    RPXDOCK_CONTAINER: Optional[str] = Field(
        default=None,
        description="Path to RPXdock container image (.sif). Used by the rpxdock block for icosahedral docking.",
    )
    RFDIFFUSION_WEIGHTS_DIR: Optional[str] = Field(
        default=None,
        description="Path to RFdiffusion v1 model weights directory. "
        "Mounted into the container at /opt/RFdiffusion/models/.",
    )

    # ---------------------------
    # MCP / Campaign
    # ---------------------------
    CHAPERONE_EVAL_RUNDIR: Optional[str] = Field(
        default=None,
        description="Working directory for the MCP campaign tools.",
    )

    # ---------------------------
    # Third-party numeric runtime controls
    # ---------------------------
    OPENBLAS_NUM_THREADS: int = Field(
        default=1,
        description="Number of OpenBLAS threads (keep at 1 to avoid oversubscription).",
    )

    def infer_backend(
        self,
        env: dict[str, str] | None = None,
    ) -> Literal["slurm", "gpu_local", "local"]:
        """Infer the execution backend for accounting and logs."""
        env = env or os.environ
        if env.get("SLURM_JOB_ID"):
            return "slurm"
        from chaperone.execution_tracking import infer_allocated_gpus

        allocated_gpus = infer_allocated_gpus(env=env)

        if allocated_gpus > 0:
            return "gpu_local"
        return "local"


def get_setting_field(name: str, default=None):
    """Lazily read a single field from ChaperoneSettings.

    .. deprecated:: Moved to ``chaperone.__init__``. Import from ``chaperone`` directly.
    """
    from chaperone import get_settings

    return getattr(get_settings(), name, default)
