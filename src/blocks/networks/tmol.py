# tmol container + thin runner block.
#
# tmol (https://github.com/uw-ipd/tmol) is a GPU-accelerated PyTorch
# reimplementation of the Rosetta energy function (beta_nov2016_cart) with
# custom C++/CUDA kernels, from the Institute for Protein Design. It computes
# energies and analytic derivatives and supports gradient-based minimisation.
#
# Packaging notes (why the install line looks the way it does):
#   * On PyPI tmol ships ONLY an sdist. Its custom PEP 517 backend
#     (`tmol_build_backend`) auto-downloads a prebuilt AOT wheel from GitHub
#     Releases, picking a `cuXXXtorchY.Y` lane from the *installed* torch. In
#     pip's isolated build env that detection is unreliable, so we sidestep the
#     backend entirely and install the exact prebuilt wheel by URL.
#   * The AOT wheel carries precompiled `.so` kernels, so NO nvcc / CUDA toolkit
#     is needed to install or run it -- only a matching torch and (for GPU) an
#     NVIDIA driver supplied at runtime via apptainer `--nv`.
#   * Wheel lane: cu132 / torch 2.12, built for sm_80;86;89;90;120
#     (Ampere->Hopper->Blackwell: A4000/A6000/A100/H100/B100/b4000).
#     Upgraded from cu128/torch2.8 (no sm_120) to add Blackwell support.
import json
import logging
from pathlib import Path

from pydantic import Field

from chaperone import CompiledBlockFunction, JobArgs, PipelineDataHandler
from chaperone.block_config import ScriptSpec
from chaperone.containers import Image
from chaperone.data.schema import make_row
from blocks.constants import cuda_python_base

logger = logging.getLogger(__name__)

RUN_TMOL_PATH = Path(__file__).parent.parent / "scripts" / "run_tmol.py"

TMOL_VERSION = "0.1.35"
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/"
    f"v{TMOL_VERSION}/tmol-{TMOL_VERSION}+cu132torch2.12-cp312-cp312-linux_x86_64.whl"
)

TMOL_CONTAINER = (
    # cuda_install=[] skips the ~5-6GB system CUDA toolkit: the AOT wheel needs
    # no nvcc, the torch cu132 wheel bundles its own CUDA runtime libs, and
    # libcuda is injected from the host driver via apptainer `--nv`.
    cuda_python_base(Image("tmol"), "tmol", cuda_install=[])
    # X11 runtime libs required by openbabel (the `[ligand]` extra) -- its format
    # plugins link against libX11/libXext/libXrender even for headless use.
    .run(
        "apt-get update && apt-get install -y --no-install-recommends "
        "libx11-6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*"
    )
    .run(
        ". /opt/tmol-env/bin/activate && "
        "pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132 && "
        # Direct-URL install of the prebuilt AOT wheel (skips the sdist build
        # backend); `[ligand]` pulls openbabel-wheel for ligand support. torch is
        # already satisfied, so pip will not pull a CPU torch over the cu132 one.
        f"pip install 'tmol[ligand] @ {TMOL_WHEEL}'"
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(PYTORCH_ALLOC_CONF="expandable_segments:True")
    .label(
        Version=TMOL_VERSION,
        CUDA="13.2",
        Torch="2.12.0",
        Source="https://github.com/uw-ipd/tmol",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[],
        env={},
    )
)


class _TmolBaseBlock(CompiledBlockFunction):
    """Shared plumbing for the tmol container blocks.

    Both blocks stage the upstream structures into ``inputs.txt`` (one path per
    line) and invoke ``run_tmol.py --mode {interface,minimize}`` inside the tmol
    container. The run script writes ``<basename>.metrics.json`` per structure
    (and, for ``minimize``, ``<basename>.min.pdb``), which ``compile`` reads back.

    LIGAND TYPING — non-canonical ligands need a chemical residue-type. tmol's
    auto-perception is broken in this container (missing ``dimorphite_dl``), so
    supply ONE of:

    * ``ligand_smiles`` — a SMILES; the runner generates a ``.tmol`` params file
      on the fly (OpenBabel MMFF94, ``protonate=False``) and applies it to the
      structure's OWN coordinates (verified bit-exact, 0.0000 Å drift).
    * ``ligand_params_files`` — pre-made ``.tmol`` params file(s) (most
      deterministic; reproduces tmol's golden ddGs).

    The params file only provides typing (atom types / bonds / charges); the
    coordinates always come from the input structure. A ligand that cannot be
    typed is skipped per-row with a logged warning (the chunk keeps going).

    Runs on CPU by default (per-structure scoring/min is fast on CPU). For large
    batches pass ``job_args={"partition": <your GPU partition>, "gres": <your gres>}``
    with your site’s resource names. The cu132 wheel
    covers Ampere through Blackwell (sm_120); see the build notes at the top of
    this file.
    """

    _mode: str = "interface"  # overridden by subclasses
    _inline = False
    _chunked = True

    _resources = JobArgs(
        partition="cpu",
        mem_gb=16,
        # tmol min/scoring is not trivial; keep chunks moderate so a job isn't huge.
        rows_per_job=100,
    )

    ligand_smiles: str | None = Field(
        default=None,
        description=(
            "SMILES of the ligand. Used to generate a tmol .tmol params file "
            "(typing only — input ligand coordinates are scored as-is). Required "
            "for non-canonical ligands unless ligand_params_files is given."
        ),
    )
    ligand_params_files: list[str] | None = Field(
        default=None,
        description=(
            "Pre-made tmol .tmol params file(s) for the ligand(s). Most "
            "deterministic typing route; bypasses SMILES generation."
        ),
    )

    def _write_inputs(self, handler: PipelineDataHandler, sandbox_dir: Path) -> Path:
        inputs_file = sandbox_dir / "inputs.txt"
        with open(inputs_file, "w") as f:
            for loaded in handler.get_data(which="last", load=False):
                f.write(f"{loaded.path}\n")
        return inputs_file

    def _ligand_args(self, sandbox_dir: Path) -> list[str]:
        args: list[str] = []
        if self.ligand_smiles:
            args += ["--ligand-smiles", self.ligand_smiles]
        for params in self.ligand_params_files or []:
            staged = str(self.stage_file(params, sandbox_dir))
            args += ["--ligand-params", staged]
        return args

    def create_command(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int,
    ) -> list[str]:
        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        inputs_file = self._write_inputs(handler, sandbox_dir)
        # Shared across all chunks of this step (sandbox_dir is <step>/chunk_N) so a
        # SMILES ligand is typed once per batch, not once per chunk.
        cache_dir = sandbox_dir.parent / "ligand_params"
        return self.command_template + [
            "--inputs",
            str(inputs_file),
            "--output-dir",
            str(sandbox_dir),
            "--cache-dir",
            str(cache_dir),
            "--mode",
            self._mode,
            *self._ligand_args(sandbox_dir),
        ]

    def _iter_metric_jsons(self, sandbox_dir: Path):
        for jf in sorted(Path(sandbox_dir).glob("*.metrics.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception as e:
                logger.warning("tmol: failed to read %s (%s)", jf, e)
                continue
            yield jf, data


class TmolInterfaceBlock(_TmolBaseBlock):
    """Protein<->ligand interface energetics (single-point, beta_nov2016).

    Emits the weighted block-pair ddG (``ddg``) plus a per-term breakdown
    (``fa_atr``, ``fa_rep``, ``fa_sol``, ``fa_elec``, ``hbond``, ``lk_ball*``,
    ...) as metrics under ``<name>.*``. The ligand is auto-detected as the
    non-polymer / HETATM residue(s); all such residues are treated as ligand.
    Structures with no ligand are skipped (``<name>.skipped`` is set).

    Attaches metrics only (does not create a new structure).
    """

    _name = "tmol_interface"
    _display_name = "tmol interface ddG"
    _description = "tmol protein-ligand interface ddG + per-term breakdown"
    _mode = "interface"
    _creates_new_step = False
    _script_spec = ScriptSpec(
        command=["python", str(RUN_TMOL_PATH)],
        image=TMOL_CONTAINER,
        source_url="https://github.com/uw-ipd/tmol",
    )

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = Path(sandbox_dir)
        rows = []
        for _, data in self._iter_metric_jsons(sandbox_dir):
            input_path = data.pop("input_path", None)
            metrics = {k: v for k, v in data.items()}
            rows.append(
                make_row(
                    name=self.name,
                    path=None,
                    path_handle=input_path,
                    metrics={self.name: metrics},
                )
            )
        if not rows:
            raise ValueError("tmol_interface: no metric outputs found in compile.")
        return PipelineDataHandler(rows)


class TmolMinimizeBlock(_TmolBaseBlock):
    """Cartesian energy minimisation (beta_nov2016_cart), all atoms free.

    Produces a minimised PDB per input (preserving the input basename) and emits
    ``energy_start`` / ``energy_final`` / ``energy_delta`` (total weighted energy)
    as metrics under ``<name>.*``. Set ``fix_backbone=True`` to freeze the
    backbone and move only sidechains (default: all atoms free).

    Creates a new structure step (the minimised structure).
    """

    _name = "tmol_minimize"
    _display_name = "tmol minimize"
    _description = "tmol Cartesian all-atom energy minimisation (beta_nov2016_cart)"
    _mode = "minimize"
    _creates_new_step = True
    _script_spec = ScriptSpec(
        command=["python", str(RUN_TMOL_PATH)],
        image=TMOL_CONTAINER,
        source_url="https://github.com/uw-ipd/tmol",
    )

    fix_backbone: bool = Field(
        default=False,
        description="Freeze backbone and minimise sidechains only (default: all free).",
    )

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        cmd = super().create_command(handler, sandbox_dir, chunk_id)
        if self.fix_backbone:
            cmd.append("--fix-backbone")
        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = Path(sandbox_dir)
        rows = []
        for _, data in self._iter_metric_jsons(sandbox_dir):
            input_path = data.pop("input_path", None)
            structure_path = data.pop("structure_path", None)
            if structure_path is None:
                # skipped row (e.g. pose build failed) — attach metrics only
                rows.append(
                    make_row(
                        name=self.name,
                        path=None,
                        path_handle=input_path,
                        metrics={self.name: data},
                    )
                )
                continue
            rows.append(
                make_row(
                    name=self.name,
                    path=structure_path,
                    path_handle=input_path,
                    metrics={self.name: data},
                )
            )
        if not rows:
            raise ValueError("tmol_minimize: no outputs found in compile.")
        return PipelineDataHandler(rows)
