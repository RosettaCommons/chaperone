# Basic blocks for design pipelines
import dataclasses
import glob
import hashlib
import os
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, model_validator
import numpy as np

from chaperone import PipelineDataHandler, JobArgs, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, ScriptSpec
from chaperone.block_config.prediction import MIN_PAE_INTERACTION_KEY
from chaperone.common import glob_designs_in_dir, tail_block_log
from chaperone.data import make_row
from blocks.networks.foundry.container import FOUNDRY_CONTAINER
from blocks.networks.prediction.helpers import (
    create_rf3_json_from_design,
    fill_rf3_json_from_template,
    is_pdb_file,
    pdb_assign_ligand_bonds,
)

logger = logging.getLogger(__name__)

_RF3_INPUT_MANIFEST = "rf3_input_manifest.json"


def _rf3_input_records(items) -> list[tuple[object, str]]:
    """Assign globally distinctive, deterministic RF3 names to input rows.

    Different campaign branches routinely emit the same basename and logical
    design ID. RF3 uses that name as an output-directory key, so forwarding it
    verbatim can overwrite or mis-backlink predictions. The source artifact,
    sequence, and row position make the staged name unique without relying on
    campaign naming discipline.
    """
    records = []
    used: set[str] = set()
    for index, loaded in enumerate(items):
        basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(loaded.basename))
        basename = basename[:160] or "design"
        sequences = getattr(loaded, "sequences", None)
        if sequences is None:
            sequences = getattr(loaded, "sequence", None)
        identity = json.dumps(
            {
                "row": index,
                "design_id": getattr(loaded, "design_id", None),
                "path": getattr(loaded, "path", None),
                "sequences": sequences,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
        name = f"{basename}__rf3_{suffix}"
        if name in used:
            raise ValueError(
                f"RF3 could not construct a unique input name for row {index}: {name}"
            )
        used.add(name)
        records.append((loaded, name))
    return records


def _write_rf3_input_manifest(sandbox_dir, records) -> None:
    payload = {
        name: {
            "path_handle": getattr(loaded, "path", None),
            "parent_design_id": getattr(loaded, "design_id", None),
        }
        for loaded, name in records
    }
    path = Path(sandbox_dir) / _RF3_INPUT_MANIFEST
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rf3_input_record_for_output(path: str, manifest: dict) -> dict:
    """Resolve an RF3 output back to its staged input manifest record."""
    output = Path(path)
    for parent in output.parents:
        if parent.name in manifest:
            return manifest[parent.name]
    stem = output.name
    for extension in (".cif.gz", ".cif", ".pdb"):
        if stem.endswith(extension):
            stem = stem[: -len(extension)]
            break
    matches = [name for name in manifest if stem == name or stem.startswith(name + "_")]
    if len(matches) == 1:
        return manifest[matches[0]]
    return {}


###########################################################################
# Design oracle blocks
###########################################################################


class RosettaFold3Block(AF3LikePredictionBlock):
    """Vanilla RosettaFold3 block for structure prediction (container-based).

    Recommended settings:
        - Default (balanced): n_recycles=10, num_diffusion_steps=50
        - Full accuracy:      n_recycles=10, num_diffusion_steps=200
        - Fast:               n_recycles=5,  num_diffusion_steps=50

    For CFG, template selection, and auto-templating, use
    RF3DevBlock (``rf3_dev``) instead.
    """

    _name = "rf3"
    _display_name = "RosettaFold3"
    _description = "RosettaFold3 structure prediction"
    _script_spec = ScriptSpec(
        command=["rf3", "fold"],
        image=FOUNDRY_CONTAINER,
        source_url="https://github.com/RosettaCommons/foundry",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        # RF3 preprocessing and compilation retain substantial per-complex
        # state. A 77-row production chunk exceeded 32 GB of worker RAM on an
        # A4000 node; keep the default failure domain safely below that.
        rows_per_job=32,
    )
    n_recycles: Optional[int] = 10
    num_diffusion_steps: int = 50
    early_stopping_plddt_threshold: float = Field(
        0.5,
        description=(
            "Stop diffusion early if pLDDT falls below this threshold after "
            "the initial recycle. "
            "Set to 0.5 for 10-20x throughput improvement, or 0 to disable. "
            "Early-stopped inputs do not produce a structure or trajectory file."
        ),
    )
    dump_trajectories: bool = Field(
        False,
        description=(
            "Write the noisy and denoised diffusion trajectories for each sample. "
            "Trajectory files are large, place the final prediction last, and "
            "remain in the chunk output sandbox."
        ),
    )
    seed_offset: int = Field(
        42,
        description="Base random seed; each chunk increments by 1. "
        "Ignored when model_seeds is set.",
    )
    model_seeds: Optional[list[int]] = Field(
        None,
        description=(
            "List of random seeds for multi-seed inference per input. Each "
            "seed runs a completely independent trunk pass, giving more "
            "structural diversity than diffusion_batch_size (which shares "
            "trunk features across samples). Total predictions per input = "
            "len(model_seeds) x diffusion_batch_size."
        ),
    )
    msa_dirs: Optional[list[str]] = Field(
        None,
        description=(
            "Absolute on-disk protein MSA roots. Chaperone exports these to both "
            "LOCAL_MSA_DIRS (installed Foundry) and PROTEIN_MSA_DIRS (current "
            "foundry-dev), so RF3 MSA use no longer depends on ambient worker "
            "environment variables."
        ),
    )
    raise_if_missing_msa_for_protein_of_length_n: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Fail when a protein at least this many residues long has no MSA. "
            "Set to 0 to require an MSA for every protein chain."
        ),
    )

    # Unlike most AF3LikePredictionBlock subclasses, which resolve weights via
    # cache_dir, this block only appends `ckpt_path=...` when the field is
    # truthy. Inheriting the base class's None therefore sends `rf3 fold` no
    # checkpoint argument at all, and foundry falls back to a per-user path
    # under ~/.foundry that does not exist on a worker — surfacing as
    # "Invalid checkpoint: rf3" long after the real cause. Default to the
    # Shared Foundry checkpoint.
    ckpt_path: Optional[str] = (
        "/projects/ml/pipelines/weights/foundry/checkpoints/"
        "rf3_foundry_01_24_latest_remapped.ckpt"
    )

    # Ligand bond graph assignment
    create_bond_graph: bool = Field(
        False,
        description=(
            "Assign ligand bonds from specified_smiles and write inputs as CIF "
            "with bond topology. Required when input structures have ligands "
            "with non-standard residue names (not in CCD) and lack CONECT "
            "records. Without this, such ligands are atomized into disconnected "
            "tokens and fragment during diffusion. Also auto-triggers for PDB "
            "inputs when specified_smiles is set."
        ),
    )

    # Sequence-based input mode
    input_mode: Literal["file", "sequence"] = Field(
        "file",
        description=(
            "'file' passes CIF/PDB directly (current default). "
            "'sequence' builds RF3 component JSON from sequences + CCD/SMILES."
        ),
    )
    template_json: Optional[str] = Field(
        None,
        description=(
            "Path to RF3 JSON template. Protein sequences are filled "
            "from upstream designs; everything else (ligands, bonds, "
            "templates) is preserved from the template."
        ),
    )

    # Templating options (standard rf3 fold CLI args, independent of CFG)
    template_selection: Optional[list[str]] = Field(
        None,
        description=(
            "Token-level templates for polymers and small molecules. "
            "Uses AtomSelection format, e.g. ['A/*/30', 'A/*/53', 'X']."
        ),
    )
    ground_truth_conformer_selection: Optional[list[str]] = Field(
        None,
        description=(
            "Lock selected residues/chains to ground truth conformers. "
            "Useful for keeping known ligand conformations while allowing "
            "the model to predict protein structure around them. "
            "E.g. ['C', 'D'] to fix two ligand chains."
        ),
    )
    cyclic_chains: Optional[list[str]] = Field(
        None,
        description=(
            "Chain IDs that should be cyclized. If input is CIF, "
            "cyclization is auto-detected."
        ),
    )

    @model_validator(mode="after")
    def _validate_create_bond_graph_requires_smiles(self):
        if self.create_bond_graph and not self.specified_smiles:
            raise ValueError(
                "create_bond_graph=True requires specified_smiles to map each "
                "ligand residue name to its SMILES string, e.g. "
                "specified_smiles={'LIG': 'c1ccccc1'}"
            )
        return self

    @model_validator(mode="after")
    def _validate_msa_dirs(self):
        if self.msa_dirs is None:
            if (
                self.raise_if_missing_msa_for_protein_of_length_n is not None
                and not os.environ.get("LOCAL_MSA_DIRS")
                and not os.environ.get("PROTEIN_MSA_DIRS")
            ):
                raise ValueError(
                    "raise_if_missing_msa_for_protein_of_length_n requires "
                    "msa_dirs or LOCAL_MSA_DIRS/PROTEIN_MSA_DIRS"
                )
            return self
        if not self.msa_dirs:
            raise ValueError("msa_dirs must contain at least one MSA root")
        relative = [path for path in self.msa_dirs if not os.path.isabs(path)]
        if relative:
            raise ValueError(f"msa_dirs paths must be absolute: {relative}")
        return self

    def get_resources(self) -> JobArgs | None:
        """Raise the memory floor when building a ligand bond graph.

        A production-shaped ligand run OOM-killed 400-row chunks at 64 GB.
        Keep the 64 GB memory request but halve the failure domain: bond-graph
        jobs default to at most 200 rows. `_resources` is a ClassVar, so return
        a copy rather than mutating the shared default.
        """
        resources = super().get_resources()
        if self.create_bond_graph and resources is not None:
            return dataclasses.replace(
                resources,
                mem_gb=max(resources.mem_gb, 64),
                rows_per_job=min(resources.rows_per_job, 200),
            )
        return resources

    @model_validator(mode="after")
    def _validate_template_json_exists(self):
        if self.template_json is not None and not os.path.exists(self.template_json):
            raise ValueError(f"template_json file does not exist: {self.template_json}")
        return self

    @model_validator(mode="after")
    def _validate_create_bond_graph_incompatible_with_json_modes(self):
        if self.create_bond_graph and (
            self.input_mode == "sequence" or self.template_json
        ):
            mode = "template_json" if self.template_json else "input_mode='sequence'"
            raise ValueError(
                f"create_bond_graph=True is incompatible with {mode}. "
                "Bond graph assignment requires structure files, but "
                "JSON input modes don't write structure files."
            )
        return self

    # -- shared helpers (used by both RosettaFold3Block and RF3DevBlock) --

    def _needs_bonds(self, path) -> bool:
        """Whether this input file requires ligand bond assignment."""
        return (self.create_bond_graph or is_pdb_file(path)) and bool(
            self.specified_smiles
        )

    @property
    def _needs_processing(self) -> bool:
        """Whether inputs need loading and processing before passing to RF3."""
        return (
            self.remove_ligands
            or self.specified_smiles
            or self.specified_ccd
            or self.disable_ccd_detection
            or self.create_bond_graph
        )

    def _build_base_cmd(self, input_path, outputdir):
        """Build the common command prefix shared by all input modes."""
        command_template = self.command_template
        if self.dump_trajectories or self.msa_dirs is not None:
            compat_script = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "scripts",
                    "rf3_trajectory_compat.py",
                )
            )
            if command_template[-2:] == ["rf3", "fold"]:
                command_template = command_template[:-2]
            elif command_template[-1] == "fold" and command_template[-3] == "python":
                command_template = command_template[:-3]
            else:
                raise ValueError(
                    "RF3 compatibility handling requires a command ending in "
                    "'rf3 fold' or 'python <compat-script> fold'"
                )
            command_template = command_template + ["python", compat_script, "fold"]

        if self.msa_dirs is not None:
            msa_dirs = ",".join(self.msa_dirs)
            msa_environment = [
                f"CHAPERONE_LOCAL_MSA_DIRS={msa_dirs}",
                f"LOCAL_MSA_DIRS={msa_dirs}",
                f"PROTEIN_MSA_DIRS={msa_dirs}",
            ]
            if "apptainer" in command_template:
                msa_environment.extend(
                    [
                        f"APPTAINERENV_LOCAL_MSA_DIRS={msa_dirs}",
                        f"APPTAINERENV_PROTEIN_MSA_DIRS={msa_dirs}",
                        f"APPTAINERENV_CHAPERONE_LOCAL_MSA_DIRS={msa_dirs}",
                    ]
                )
            elif "singularity" in command_template:
                msa_environment.extend(
                    [
                        f"SINGULARITYENV_LOCAL_MSA_DIRS={msa_dirs}",
                        f"SINGULARITYENV_PROTEIN_MSA_DIRS={msa_dirs}",
                        f"SINGULARITYENV_CHAPERONE_LOCAL_MSA_DIRS={msa_dirs}",
                    ]
                )
            command_template = [
                "env",
                *msa_environment,
                *command_template,
            ]

        return command_template + [
            f"inputs={input_path}",
            f"out_dir={outputdir}",
            f"diffusion_batch_size={self.diffusion_batch_size}",
            f"num_steps={self.num_diffusion_steps}",
        ]

    def _write_json_inputs(self, entries, sandbox_dir):
        """Write JSON entries to file and return the path."""
        path = os.path.join(sandbox_dir, "rf3_inputs.json")
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)
        return path

    def _append_common_cli_args(self, cmd, chunk_id=0):
        """Append common CLI args shared across all input modes."""
        if self.model_seeds is None:
            cmd.append(self._seed_override(self.seed_offset + chunk_id))

        if self.n_recycles is not None:
            cmd.append(f"n_recycles={self.n_recycles}")

        if self.early_stopping_plddt_threshold > 0:
            cmd.append(
                f"early_stopping_plddt_threshold={self.early_stopping_plddt_threshold}"
            )

        if self.dump_trajectories:
            cmd.append("dump_trajectories=true")

        if self.raise_if_missing_msa_for_protein_of_length_n is not None:
            cmd.append(
                "raise_if_missing_msa_for_protein_of_length_n="
                f"{self.raise_if_missing_msa_for_protein_of_length_n}"
            )

        if self.ckpt_path:
            cmd.append(f"ckpt_path={self.ckpt_path}")

        return cmd

    def _seed_override(self, seed: int) -> str:
        """Return the Hydra override accepted by this RF3 distribution."""
        return f"seed={seed}"

    def _expand_model_seed_commands(self, cmd, outputdir):
        """Run one RF3 invocation per model seed.

        Foundry RF3 exposes a singular ``seed`` setting, not ``model_seeds``.
        Separate invocations are required for independent trunk passes.  Each
        invocation gets its own output subtree and the normal recursive
        compiler combines all samples afterward.
        """
        if self.model_seeds is None:
            return cmd

        commands = []
        for seed in self.model_seeds:
            seed_output = os.path.join(outputdir, f"seed_{seed}")
            seeded = [
                f"out_dir={seed_output}" if arg.startswith("out_dir=") else arg
                for arg in cmd
            ]
            seeded.append(self._seed_override(seed))
            commands.append(shlex.join(seeded))
        return ["bash", "-lc", "set -euo pipefail; " + " && ".join(commands)]

    def _append_file_mode_cli_args(self, cmd):
        """Append CLI args only applicable to file-based input mode."""
        if self.template_selection is not None:
            items = ",".join(self.template_selection)
            cmd.append(f"template_selection=[{items}]")

        if self.ground_truth_conformer_selection is not None:
            items = ",".join(self.ground_truth_conformer_selection)
            cmd.append(f"ground_truth_conformer_selection=[{items}]")

        if self.cyclic_chains is not None:
            items = ",".join(self.cyclic_chains)
            cmd.append(f"cyclic_chains=[{items}]")

        return cmd

    def _process_inputs(self, records, inputdir):
        """Load atom arrays and write CIF inputs.

        When create_bond_graph is set (or input is PDB with specified_smiles),
        assigns ligand bonds from SMILES and writes CIF with bond topology.
        Otherwise, strips non-protein atoms.
        """
        from atomworks.io.utils.io_utils import to_cif_file

        from chaperone.io import ensure_bonds

        for loaded, input_name in records:
            dst = os.path.join(inputdir, f"{input_name}.cif")

            if self._needs_bonds(loaded.path):
                atom_array = pdb_assign_ligand_bonds(
                    loaded.atom_array, self.specified_smiles
                )
                to_cif_file(ensure_bonds(atom_array), dst)
            else:
                to_cif_file(ensure_bonds(loaded.atom_array), dst)

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        # JSON input modes (template_json or sequence) — differ only in
        # which helper builds the entries
        if (
            self.template_json
            or self.input_mode == "sequence"
            or handler.has_sequence_artifacts
        ):
            entries = []
            loaded_inputs = list(
                handler.get_prediction_inputs(which=self.which, load_context=True)
            )
            records = _rf3_input_records(loaded_inputs)
            _write_rf3_input_manifest(sandbox_dir, records)
            for loaded, input_name in records:
                if self.template_json:
                    entry = fill_rf3_json_from_template(
                        template_json=self.template_json,
                        atom_array=loaded.atom_array,
                        name=input_name,
                        protein_sequences=loaded.sequences,
                    )
                else:
                    entry = create_rf3_json_from_design(
                        atom_array=loaded.atom_array,
                        name=input_name,
                        remove_ligands=self.remove_ligands,
                        specified_smiles=self.specified_smiles,
                        specified_ccd=self.specified_ccd,
                        disable_ccd_detection=self.disable_ccd_detection,
                        protein_sequences=loaded.sequences,
                    )
                entries.append(entry)

            json_path = self._write_json_inputs(entries, sandbox_dir)
            cmd = self._build_base_cmd(json_path, outputdir)
            self._append_common_cli_args(cmd, chunk_id)
            return self._expand_model_seed_commands(cmd, outputdir)

        # File mode (default): pass structure files directly
        loaded_inputs = list(
            handler.get_data(
                which=self.which,
                load=self._needs_processing,
            )
        )
        records = _rf3_input_records(loaded_inputs)
        _write_rf3_input_manifest(sandbox_dir, records)
        if self._needs_processing:
            self._process_inputs(records, inputdir)
        else:
            for loaded, input_name in records:
                suffix = (
                    ".cif.gz"
                    if str(loaded.path).endswith(".cif.gz")
                    else Path(str(loaded.path)).suffix
                )
                dst = os.path.join(inputdir, f"{input_name}{suffix or '.cif'}")
                os.symlink(loaded.path, dst)

        cmd = self._build_base_cmd(inputdir, outputdir)
        self._append_common_cli_args(cmd, chunk_id)
        self._append_file_mode_cli_args(cmd)
        return self._expand_model_seed_commands(cmd, outputdir)

    def compile(self, sandbox_dir):
        return compile_rf3(
            sandbox_dir,
            name=self.name,
            compile_as_metric=self.compile_as_metric,
            read_most_confident_model=self.read_most_confident_model,
        )


def compile_rf3(
    sandbox_dir, name="rf3", compile_as_metric=False, read_most_confident_model=True
):
    """Compile RF3 outputs into a PipelineDataHandler."""
    outputdir = os.path.join(sandbox_dir, "outputs")

    # Fall back to sandbox_dir itself if no outputs/ subdir (e.g. direct run)
    search_dir = outputdir if os.path.isdir(outputdir) else sandbox_dir

    # RF3 outputs are nested: outputs/<name>/seed-N_sample-N/<name>.cif.gz
    # The top-level <name>_model.cif is the best model (selected by RF3).
    # Per-sample models are in seed-N_sample-N/ subdirectories.
    if read_most_confident_model:
        # Only collect top-level models (direct children of design dirs).
        # Multi-seed inference adds one seed_<N>/ level above each design.
        paths = []
        design_dirs = []
        for child in glob.glob(os.path.join(search_dir, "*")):
            if not os.path.isdir(child):
                continue
            if re.fullmatch(r"seed_-?\d+", os.path.basename(child)):
                design_dirs.extend(
                    nested
                    for nested in glob.glob(os.path.join(child, "*"))
                    if os.path.isdir(nested)
                )
            else:
                design_dirs.append(child)
        for design_dir in design_dirs:
            paths.extend(glob_designs_in_dir(design_dir))
    else:
        paths = glob_designs_in_dir(search_dir, recursive=True)
        # Current RF3 writes a top-ranked alias in each design directory in
        # addition to the genuine seed/sample predictions below it.  When all
        # samples are requested, prefer the explicit sample directories so the
        # alias is not counted and rescored as an extra prediction.
        sample_paths = [
            path
            for path in paths
            if re.fullmatch(
                r"seed-\d+_sample-\d+", os.path.basename(os.path.dirname(path))
            )
        ]
        if sample_paths:
            paths = sample_paths

    # Trajectory CIFs are multi-model diagnostics, not independent predictions.
    # Keep them on disk for the caller without adding them to handler lineage.
    paths = [
        path
        for path in paths
        if not os.path.basename(path).startswith(("denoised_model_", "noisy_model_"))
    ]

    if len(paths) == 0:
        raise RuntimeError(
            f"RF3 execution completed but no output files were generated in {search_dir}. "
            f"This may indicate:\n"
            f"  - Invalid input specifications\n"
            f"  - Execution errors (check sandbox directory for logs)\n"
            + tail_block_log(search_dir)
        )

    try:
        manifest = json.loads((Path(sandbox_dir) / _RF3_INPUT_MANIFEST).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        manifest = {}

    rows = []
    for path in paths:
        path_dir = os.path.dirname(path)
        stem = os.path.basename(path)
        for ext in (".cif.gz", ".cif", ".pdb"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        # RF3 names the best model <name>_model.cif but confidences
        # are <name>_summary_confidences.json — strip the _model suffix.
        if stem.endswith("_model"):
            stem = stem[: -len("_model")]

        # RF3 writes <name>_summary_confidences.json next to the structure
        metrics = {}
        conf_path = os.path.join(path_dir, f"{stem}_summary_confidences.json")
        if os.path.exists(conf_path):
            try:
                with open(conf_path) as f:
                    data = json.load(f)
                metrics = {
                    k: data[k]
                    for k in [
                        "ptm",
                        "iptm",
                        "ranking_score",
                        "fraction_disordered",
                        "has_clash",
                        "overall_plddt",
                        "overall_pde",
                    ]
                    if k in data
                }
                if "overall_plddt" in metrics:
                    # Match the canonical mean-pLDDT key used by the other
                    # prediction blocks while preserving RF3's raw key.
                    metrics["plddt"] = metrics["overall_plddt"]

                min_pae = data.get("chain_pair_pae_min", None)
                if min_pae is not None:
                    arr = np.array(min_pae, dtype=float)  # None → nan
                    diag = np.fliplr(arr).diagonal()
                    finite = diag[np.isfinite(diag)]
                    # Canonical PAE-derived interface key (shared with AF3).
                    metrics[MIN_PAE_INTERACTION_KEY] = (
                        float(finite.min()) if len(finite) > 0 else None
                    )
            except Exception as e:
                logger.error(f"Failed to load RF3 confidences from {conf_path}: {e}")

        source = _rf3_input_record_for_output(path, manifest)
        path_handle = source.get("path_handle")
        rows.append(
            make_row(
                name=name,
                path=path,
                metadata={},
                metrics={name: metrics},
                path_handle=path_handle,
                parent_design_id=(
                    source.get("parent_design_id") if path_handle is None else None
                ),
                as_metric=compile_as_metric,
            )
        )

    return PipelineDataHandler(rows)


###########################################################################
# RF3 via local foundry-dev installation
###########################################################################


class RF3DevBlock(RosettaFold3Block):
    """RF3 via local foundry-dev installation with CFG and auto-templating.

    Runs RF3 using a local foundry-dev checkout instead of a container.
    Supports both manual CFG template selection and automatic pocket-based
    template discovery (auto-templating).

    Classifier-free guidance (CFG):
        CFG amplifies template-biased signal during diffusion sampling. At each
        step, RF3 runs two trunk passes — one with templates (conditioned) and
        one without (unconditional reference). The difference is scaled by
        (cfg_scale - 1) and added back, controlling how strongly the output
        conforms to the template structure.

        template_selection:
            Marks which atoms to use as spatial coordinate templates. Primary
            driver of CFG. Typically derived from MPNN's --fixed_residues.

        relative_template_selection:
            Controls inter-component relative positioning. Organizes templates
            into named groups for multi-component spatial relationships.

        ground_truth_conformer_selection:
            Locks selected atoms to their input geometry during conformer
            sampling. Typically the ligand/cofactor chain(s).

    Auto-templating (``auto_templating=True``):
        Automatically discovers protein residues near the ligand and builds
        per-structure JSON inputs with template_selection,
        ground_truth_conformer_selection, and relative_template_selection.

    Examples::

        # Basic prediction (no CFG)
        engine.run("rf3_dev", data=handler)

        # Manual CFG template selection
        engine.run(
            RF3DevBlock(
                inference_classifier_free_guidance=True,
                inference_cfg_scale=0.5,
                template_selection=["A/*/30", "A/*/53", "X"],
                relative_template_selection={"group1": ["A/*/30", "A/*/53", "X"]},
                ground_truth_conformer_selection=["X"],
            ),
            data=handler,
        )

        # Auto-templating for binder pocket conditioning
        engine.run(
            RF3DevBlock(
                auto_templating=True,
                auto_templating_chain="B",
                auto_templating_radius=6.0,
                inference_classifier_free_guidance=True,
                inference_cfg_scale=1.5,
            ),
            data=handler,
            job_args={"partition": "gpu", "gres": "gpu:small:1"},
        )

        # Multi-seed hallucination with CFG
        engine.run(
            RF3DevBlock(
                model_seeds=[0, 1, 2, 3, 4],
                diffusion_batch_size=1,
                inference_classifier_free_guidance=True,
                inference_cfg_scale=0.5,
                template_selection=["A/*/30", "X"],
            ),
            data=handler,
        )
    """

    _name = "rf3_dev"
    _display_name = "RosettaFold3"
    _description = "RF3 via local foundry-dev with CFG and auto-templating"

    foundry_dev_path: Optional[str] = Field(
        None,
        description=(
            "Path to local foundry-dev repository. If not set, falls back "
            "to FOUNDRY_DEV_PATH from settings. Required for this block."
        ),
    )

    # Classifier-free guidance (CFG)
    inference_classifier_free_guidance: bool = Field(
        False,
        description=(
            "Enable classifier-free guidance for template-based inference. "
            "Requires at least one of template_selection, relative_template_selection, "
            "or ground_truth_conformer_selection to be set (or auto_templating=True). "
            "Without templates, CFG has no effect."
        ),
    )
    inference_cfg_scale: float = Field(
        1.0,
        description=(
            "CFG guidance scale. Controls how strongly the template signal is "
            "amplified during diffusion. <1.0 = less guidance (more diversity), "
            "1.0 = no effect, >1.0 = stronger template guidance."
        ),
    )

    # CFG-specific template field (template_selection and
    # ground_truth_conformer_selection are inherited from parent)
    relative_template_selection: Optional[dict] = Field(
        None,
        description=(
            "Grouped template specification for relative positioning between "
            "components. Dict mapping group names to selection lists. group1 is "
            "the primary template set. Example: {'group1': ['A/*/30', 'X']}."
        ),
    )

    # Auto-templating fields
    auto_templating: bool = Field(
        False,
        description=(
            "Automatically discover protein residues near the ligand and "
            "build per-structure template selections for CFG. When enabled, "
            "template_selection/relative_template_selection/"
            "ground_truth_conformer_selection are auto-generated per input."
        ),
    )
    auto_templating_radius: float = Field(
        5.0,
        description="Radius in angstroms for finding nearby protein residues.",
    )
    auto_templating_chain: str = Field(
        "B",
        description="Chain ID of the ligand in input structures.",
    )
    protein_chain: str = Field(
        "A",
        description="Chain ID of the protein for auto-templating.",
    )
    ligand_atoms: Optional[list[str]] = Field(
        None,
        description=(
            "Ligand atom names to use for radius search in auto-templating. "
            "If None, all ligand atoms are used. Use to restrict to "
            "buried/core atoms, e.g. ['N1', 'N2', 'S1', 'C2']."
        ),
    )

    def _seed_override(self, seed: int) -> str:
        """foundry-dev's inference config does not declare ``seed`` yet."""
        return f"+seed={seed}"

    def _resolve_foundry_dev_path(self) -> str:
        """Resolve foundry-dev path: field > settings (.env.ipd) > env var."""
        path = self.foundry_dev_path
        if not path:
            path = get_setting_field("FOUNDRY_DEV_PATH", default="") or None
        if not path:
            path = os.getenv("FOUNDRY_DEV_PATH", "") or None
        if not path:
            raise ValueError(
                "FOUNDRY_DEV_PATH is not configured. Set it via foundry_dev_path field, "
                ".env.ipd, .env.local, or as an environment variable."
            )
        if not os.path.exists(path):
            raise ValueError(f"FOUNDRY_DEV_PATH '{path}' does not exist.")
        return path

    @model_validator(mode="after")
    def _validate_cfg_requires_templates(self):
        """CFG requires templates — unless auto_templating generates them."""
        if not self.inference_classifier_free_guidance:
            return self
        if self.auto_templating:
            return self
        has_templates = (
            self.template_selection is not None
            or self.relative_template_selection is not None
            or self.ground_truth_conformer_selection is not None
        )
        if not has_templates:
            raise ValueError(
                "inference_classifier_free_guidance=True requires at least one of "
                "template_selection, relative_template_selection, or "
                "ground_truth_conformer_selection to be set (or auto_templating=True). "
                "CFG amplifies template signal — without templates it has no effect."
            )
        return self

    @model_validator(mode="after")
    def _validate_auto_templating_with_template_json(self):
        """Auto-templating generates its own JSON, so a user template conflicts."""
        if self.auto_templating and self.template_json:
            raise ValueError(
                "auto_templating=True is incompatible with template_json. "
                "Auto-templating generates per-structure JSON inputs with "
                "template selections; a user-supplied template would conflict."
            )
        return self

    def get_command_template(self):
        """Build containerized foundry-dev command using the deps-only dev SIF."""
        from chaperone.containers import Image

        foundry_dev = self._resolve_foundry_dev_path()
        pythonpath = ":".join(
            [
                os.path.join(foundry_dev, "models", "rf3", "src"),
                os.path.join(foundry_dev, "src"),
                os.path.join(foundry_dev, "lib", "atomworks", "src"),
            ]
        )
        cli_script = os.path.join(foundry_dev, "models", "rf3", "src", "rf3", "cli.py")
        sif = get_setting_field("FOUNDRY_DEV_SIF")
        dev_container = Image.from_existing("foundry_dev", sif).template_command(
            exec_mode="exec",
            binds=[(foundry_dev, foundry_dev)],
            env={"PYTHONPATH": pythonpath},
        )
        return dev_container.build_command("python", cli_script, "fold")

    def _append_cfg_args(self, cmd):
        """Append CFG-specific args to command."""
        if self.inference_classifier_free_guidance:
            cmd.append("inference_classifier_free_guidance=true")
            cmd.append(f"inference_cfg_scale={self.inference_cfg_scale}")

        if self.relative_template_selection is not None:
            parts = []
            for group, selections in self.relative_template_selection.items():
                if isinstance(selections, list):
                    parts.append(f"{group}:[{','.join(selections)}]")
                else:
                    parts.append(f"{group}:{selections}")
            cmd.append(f"relative_template_selection={{{','.join(parts)}}}")

        return cmd

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        if self.auto_templating:
            return self._create_autotemplate_command(handler, sandbox_dir, chunk_id)
        # Normal path: parent builds base cmd (incl. templates), then add CFG args
        cmd = super().create_command(handler, sandbox_dir, chunk_id)
        return self._append_cfg_args(cmd)

    def _create_autotemplate_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        """Build per-structure JSON inputs with auto-discovered templates."""
        import numpy as np
        from blocks.utils.geometry import _find_nearest_neighbors

        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        ligand_chain = self.auto_templating_chain

        json_entries = []
        for loaded in handler.get_data(which=self.which, load=True):
            atom_array = loaded.atom_array

            # Assign ligand bonds if needed (PDB inputs or explicit create_bond_graph)
            if self._needs_bonds(loaded.path):
                from atomworks.io.utils.io_utils import to_cif_file

                from chaperone.io import ensure_bonds

                atom_array = pdb_assign_ligand_bonds(atom_array, self.specified_smiles)
                cif_path = os.path.join(inputdir, f"{loaded.basename}.cif")
                to_cif_file(ensure_bonds(atom_array), cif_path)
                input_path = cif_path
            else:
                staged = self.stage_file(loaded.path, sandbox_dir, subdir="inputs")
                input_path = os.path.abspath(str(staged))

            # Find protein residues near the ligand chain
            ligand_mask = atom_array.chain_id == ligand_chain
            if self.ligand_atoms is not None:
                ligand_mask = ligand_mask & np.isin(
                    atom_array.atom_name, self.ligand_atoms
                )
            ligand_indices = np.where(ligand_mask)[0]

            if len(ligand_indices) == 0:
                logger.warning(
                    f"Skipping {loaded.basename}: no atoms found for "
                    f"chain '{ligand_chain}'"
                )
                continue

            cb_indices = _find_nearest_neighbors(
                atom_array, ligand_indices.tolist(), self.auto_templating_radius
            )

            # Extract unique residue IDs on the protein chain
            res_ids = set()
            for idx in cb_indices:
                if atom_array.chain_id[idx] == self.protein_chain:
                    res_ids.add(int(atom_array.res_id[idx]))

            motif_residues = [
                f"{self.protein_chain}/*/{rid}" for rid in sorted(res_ids)
            ]

            logger.info(
                f"{loaded.basename}: {len(motif_residues)} motif residues within "
                f"{self.auto_templating_radius}A of chain {ligand_chain}"
            )

            entry = {
                "name": loaded.basename,
                "components": [{"path": input_path}],
                "template_selection": motif_residues,
                "ground_truth_conformer_selection": [ligand_chain],
                "relative_template_selection": {
                    "group1": motif_residues + [ligand_chain],
                },
            }
            json_entries.append(entry)

        if not json_entries:
            raise RuntimeError(
                f"No valid entries produced for auto-templating. Check that "
                f"auto_templating_chain='{ligand_chain}' exists in input structures."
            )

        json_path = self._write_json_inputs(json_entries, sandbox_dir)
        cmd = self._build_base_cmd(json_path, outputdir)
        self._append_common_cli_args(cmd, chunk_id)

        if self.cyclic_chains is not None:
            items = ",".join(self.cyclic_chains)
            cmd.append(f"cyclic_chains=[{items}]")

        self._append_cfg_args(cmd)
        return self._expand_model_seed_commands(cmd, outputdir)


if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "rf3"
    SANDBOX = REPO_ROOT / "unittest_outs" / "rf3"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = RosettaFold3Block(
        num_diffusion_samples=1,
        diffusion_batch_size=1,
        ckpt_path="/projects/ml/pipelines/weights/foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt",
    )
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
