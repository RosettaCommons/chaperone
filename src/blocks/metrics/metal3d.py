"""Metal3D block — predict transition-metal binding sites with a 3D-CNN.

Wraps Metal3D (Dürr et al. 2022, https://doi.org/10.1101/2022.08.22.504853)
as a metric block: each row's input structure is voxelized around its
metal-binding residues (HIS/CYS/ASP/GLU/ASN/GLN/MET), scored by the
pretrained 3D-CNN, and the predicted probe positions are clustered into
unique sites. Per-row metrics (``max_p``, ``n_sites``, top-site
coordinates) and the probefile path are attached to each output row.
"""

import json
import logging
from pathlib import Path

from pydantic import Field

from pipelines import CompiledBlockFunction, JobArgs, PipelineDataHandler
from pipelines.block_config import TIER_PRIMARY, ScriptSpec
from pipelines.containers import Image
from pipelines.data import make_row

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_metal3d.py"


# ── Container ───────────────────────────────────────────────────────────
#
# Metal3D's reference environment (environment.yml + ColabMetal.ipynb) pins
# moleculekit==1.6.9 from the acellera channel together with pdb2pqr and
# openbabel from conda-forge. The voxel-descriptor extension shipped with
# moleculekit is a compiled C++ module, so we use a miniforge base image
# and install via mamba to match the validated stack rather than risking a
# pip wheel mismatch.
#
# The repo (with weights and helper utils) is cloned at build time into
# /opt/metal-site-prediction; the wrapper script reads weights from
# ``$METAL3D_REPO/Metal3D/weights/...`` so the container is self-contained.

METAL3D_CONTAINER = (
    Image("metal3d")
    .base_image("condaforge/miniforge3:latest")
    .packages("git", "wget", "curl", "ca-certificates", "build-essential")
    .run(
        # Pin numpy<2 across the env: moleculekit 1.6.9 (Aug 2022) predates
        # numpy 2 and its compiled voxel-descriptor extension is built
        # against the numpy 1.x ABI. Without an explicit pin the conda
        # solver picks scipy/sklearn/biopython wheels compiled against
        # numpy 2, producing a runtime ABI mismatch ("Expected 96 from C
        # header, got 88 from PyObject").
        "mamba install -n base -y -c acellera -c conda-forge "
        "python=3.10 moleculekit=1.6.9 pdb2pqr openbabel rdkit "
        "'numpy<2' 'scipy<1.13' 'scikit-learn<1.4' biopython tqdm "
        "&& mamba clean -afy",
        # Torch from PyPI — conda-forge's pytorch builds rarely match the
        # CUDA stack we need, and Metal3D works fine with CPU torch
        # (inference is small: 32^3 voxels per residue). Use a torch
        # version still compiled against numpy 1.x (torch 2.2.x); newer
        # torch wheels bundle numpy-2-compiled extensions that re-trip
        # the ABI mismatch above.
        "pip install --no-cache-dir 'torch==2.2.*' 'numpy<2' "
        "--index-url https://download.pytorch.org/whl/cu121 "
        "--extra-index-url https://pypi.org/simple",
    )
    .git_clone(
        "https://github.com/lcbc-epfl/metal-site-prediction.git",
        "/opt/metal-site-prediction",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache")
    .env(
        METAL3D_REPO="/opt/metal-site-prediction",
        PYTHONPATH="/opt/metal-site-prediction/Metal3D",
    )
    .label(Version="metal3d-2022.08", Source="lcbc-epfl/metal-site-prediction")
    .runscript('"$@"')
    # Bind both /home and /mnt/home: the wrapper script lives outside the
    # image and __file__ on IPD machines often resolves via /mnt/home,
    # which is not bound by default.
    .template_command(
        exec_mode="exec",
        binds=[("/home", "/home"), ("/mnt/home", "/mnt/home")],
    )
)


class Metal3DBlock(CompiledBlockFunction):
    """Score each row's structure for transition-metal binding sites.

    Adds the following metrics under ``metal3d.*`` to every row:

    * ``max_p``: peak metal-occupancy probability across the search grid
    * ``max_xyz``: ``[x, y, z]`` of the peak-probability voxel (Å)
    * ``n_sites``: number of distinct predicted sites after agglomerative
      clustering at ``cluster_threshold`` Å
    * ``top_p``: probability of the highest-scoring clustered site
    * ``top_xyz``: ``[x, y, z]`` of the highest-scoring clustered site
    * ``sites``: full list of ``{x, y, z, p}`` per clustered site
    * ``probefile``: relative path (within the sandbox) to the per-input
      probe PDB written by Metal3D

    Inputs are taken from the most recent step's structures (``which="last"``).

    Performance: a single ~150 aa protein takes ~4 min wall on one CPU
    (most of it spent in scipy's KDTree query over the ~130k voxel
    centres in the protein voxelisation). ``rows_per_job=20`` keeps a
    SLURM job comfortably under its 24h limit. GPU is not required —
    inference runs on a handful of 32³ voxel boxes.
    """

    _name = "metal3d"
    _display_name = "Metal3D"
    _description = "Predict transition-metal binding sites with the Metal3D 3D-CNN"
    _script_spec = ScriptSpec(
        command=["python", str(SCRIPT_PATH)],
        image=METAL3D_CONTAINER,
        source_url="https://github.com/lcbc-epfl/metal-site-prediction",
    )
    _resources = JobArgs(
        partition="cpu",
        mem_gb=16,
        rows_per_job=20,
    )
    _creates_new_step = False
    _inline = False
    _chunked = True

    metalbinding: bool = Field(
        default=True,
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "If True, only voxelize residues whose sidechains can coordinate "
            "metals (HIS/CYS/ASP/GLU/ASN/GLN/MET). If False, scan every CA "
            "in the protein (much slower)."
        ),
    )
    cluster_threshold: float = Field(
        default=7.0,
        description=(
            "Maximum distance (Å) between probe voxels in the same cluster. "
            "Larger values merge nearby sites into one."
        ),
    )
    pthreshold: float = Field(
        default=0.10,
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "Minimum probability for a voxel to seed a clustered site. "
            "Lower values produce more (noisier) sites."
        ),
    )
    batch_size: int = Field(
        default=128, description="Inference batch size (number of voxel boxes)."
    )
    write_cube: bool = Field(
        default=False,
        description=(
            "If True, also write the volumetric probability map as a "
            ".cube file per input. Off by default — files can be large."
        ),
    )
    compile_as_metric: bool = Field(
        default=True,
        description=(
            "If True (default), attach Metal3D metrics to the parent row "
            "without adding the probefile to the lineage — same convention "
            "as the prediction blocks' compile_as_metric. Set to False to "
            "promote the probefile to a new structure step instead, e.g. "
            "if a downstream block consumes the predicted probe positions."
        ),
    )

    def create_command(
        self, handler: PipelineDataHandler, sandbox_dir: Path, chunk_id: int
    ) -> list[str]:
        from atomworks.io.utils.io_utils import to_pdb_string

        sandbox_dir = Path(sandbox_dir)
        inputs_dir = sandbox_dir / "inputs"
        outputs_dir = sandbox_dir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        pdb_list = inputs_dir / f"inputs_chunk{chunk_id}.txt"

        # Metal3D's moleculekit-based loader is fragile on modern CIFs
        # (multi-chain, alt-locs, embedded ligands etc.). Parse each row's
        # structure into a biotite AtomArray, drop non-protein atoms +
        # hydrogens, and re-emit a plain PDB so processStructures sees a
        # clean, well-formed input regardless of the upstream block.
        with open(pdb_list, "w") as f:
            for i, data in enumerate(handler.get_data(which="last", load=True)):
                arr = data.atom_array
                if arr is None:
                    raise RuntimeError(
                        f"metal3d: failed to load atom array for {data.path}"
                    )
                # Filter: protein heavy atoms only — Metal3D's
                # voxelizer ignores everything else and chokes on some
                # ligands during atomtyping.
                if "hetero" in arr.get_annotation_categories():
                    arr = arr[~arr.hetero]
                arr = arr[arr.element != "H"]

                stem = Path(data.path).name
                for suffix in (".gz", ".cif", ".pdb"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                # Disambiguate basename collisions (e.g. multiple chunks
                # producing design_0.pdb) by prefixing the row index.
                cleaned_pdb = inputs_dir / f"{i:05d}_{stem}.pdb"
                cleaned_pdb.write_text(to_pdb_string(arr))
                f.write(f"{cleaned_pdb}\n")

        cmd = self.command_template + [
            "--pdb-list",
            str(pdb_list),
            "--output-dir",
            str(outputs_dir),
            "--metalbinding",
            "1" if self.metalbinding else "0",
            "--cluster-threshold",
            str(self.cluster_threshold),
            "--pthreshold",
            str(self.pthreshold),
            "--batch-size",
            str(self.batch_size),
            "--write-cube",
            "1" if self.write_cube else "0",
        ]
        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = Path(sandbox_dir)
        results_path = sandbox_dir / "outputs" / "metal3d_results.json"
        if not results_path.exists():
            raise RuntimeError(
                f"{self._name} produced no results file at {results_path}. "
                f"Check the chunk's manifest stderr for failures."
            )

        with open(results_path) as f:
            results = json.load(f)

        rows = []
        for input_name, res in results.items():
            metrics: dict = {}
            if "error" in res:
                metrics["error"] = res["error"]
            else:
                sites = res.get("sites") or []
                metrics["max_p"] = res.get("max_p")
                metrics["max_xyz"] = res.get("max_xyz")
                metrics["n_sites"] = res.get("n_sites", len(sites))
                if sites:
                    top = sites[0]
                    metrics["top_p"] = top["p"]
                    metrics["top_xyz"] = [top["x"], top["y"], top["z"]]
                else:
                    metrics["top_p"] = None
                    metrics["top_xyz"] = None
                metrics["sites"] = sites
                if res.get("probefile"):
                    metrics["probefile"] = res["probefile"]
                if res.get("cubefile"):
                    metrics["cubefile"] = res["cubefile"]

            # The artifact this block produces is the probefile. When no
            # sites cleared the probability threshold the probefile is
            # absent; fall back to the cleaned input path so make_row
            # still has something to record.
            artifact_path = res.get("probefile") or res.get("input") or input_name
            rows.append(
                make_row(
                    name=self.name,
                    path=artifact_path,
                    metrics={self.name: metrics},
                    as_metric=self.compile_as_metric,
                )
            )

        return PipelineDataHandler(rows)


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[3]
    TEST_DATA = REPO_ROOT / "tests" / "test_data"
    SANDBOX = REPO_ROOT / "unittest_outs" / "metal3d"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    pdbs = list((TEST_DATA / "unittests" / "mpnn").glob("*.pdb"))[:1]
    block = Metal3DBlock()
    handler = PipelineDataHandler.from_paths([str(p) for p in pdbs], name="input")
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
