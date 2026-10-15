# Basic blocks for design pipelines
import csv
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
import polars as pl

from chaperone import PipelineDataHandler, JobArgs
from chaperone.block_config import AF3LikePredictionBlock, ScriptSpec
from chaperone.block_config.prediction import MIN_PAE_INTERACTION_KEY
from chaperone.common import glob_designs_in_dir, tail_block_log
from chaperone.data import make_row
from chaperone.data.schema import SCHEMA
from blocks.networks.foundry.container import FOUNDRY_CONTAINER, default_checkpoint
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


def _rf3_input_name_for_output(path: str, manifest: dict) -> str | None:
    """Resolve an RF3 output back to its staged input manifest key."""
    output = Path(path)
    for parent in output.parents:
        if parent.name in manifest:
            return parent.name
    stem = output.name
    for extension in (".cif.gz", ".cif", ".pdb"):
        if stem.endswith(extension):
            stem = stem[: -len(extension)]
            break
    matches = [name for name in manifest if stem == name or stem.startswith(name + "_")]
    if len(matches) == 1:
        return matches[0]
    return None


def _rf3_input_record_for_output(path: str, manifest: dict) -> dict:
    return manifest.get(_rf3_input_name_for_output(path, manifest), {})


###########################################################################
# Design oracle blocks
###########################################################################


class RosettaFold3Block(AF3LikePredictionBlock):
    """Vanilla RosettaFold3 block for structure prediction (container-based).

    Recommended settings:
        - Default (balanced): n_recycles=10, num_diffusion_steps=50
        - Full accuracy:      n_recycles=10, num_diffusion_steps=200
        - Fast:               n_recycles=5,  num_diffusion_steps=50
    """

    _name = "rf3"
    _display_name = "RosettaFold3"
    _description = "RosettaFold3 structure prediction"
    output_lineage_version: Literal[2] = Field(
        2, frozen=True, repr=False, json_schema_extra={"cli_type": None, "internal": True},
        description="Internal cache revision for complete outputs and explicit sequence lineage.",
    )

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
            "LOCAL_MSA_DIRS and PROTEIN_MSA_DIRS (supported Foundry versions), "
            "so RF3 MSA use no longer depends on ambient worker "
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

    ckpt_path: Optional[str] = Field(
        default_factory=lambda: default_checkpoint(
            "rf3", "rf3_foundry_01_24_latest_remapped.ckpt"
        ),
        description="Checkpoint path or Foundry registry name (default: rf3).",
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
        if self.msa_dirs is not None:
            msa_dirs = ",".join(self.msa_dirs)
            msa_environment = [
                f"LOCAL_MSA_DIRS={msa_dirs}",
                f"PROTEIN_MSA_DIRS={msa_dirs}",
            ]
            if "apptainer" in command_template:
                msa_environment.extend(
                    [
                        f"APPTAINERENV_LOCAL_MSA_DIRS={msa_dirs}",
                        f"APPTAINERENV_PROTEIN_MSA_DIRS={msa_dirs}",
                    ]
                )
            elif "singularity" in command_template:
                msa_environment.extend(
                    [
                        f"SINGULARITYENV_LOCAL_MSA_DIRS={msa_dirs}",
                        f"SINGULARITYENV_PROTEIN_MSA_DIRS={msa_dirs}",
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
            expected_seeds=self.model_seeds,
            early_stopping_plddt_threshold=self.early_stopping_plddt_threshold,
        )


def _rf3_was_early_stopped(output_root, input_name, threshold):
    """Recognize Foundry's per-input outcome, never infer stopping from no CIF.

    Pinned Foundry records an intentional stop as one ranking CSV row. Require
    its explicit flag and finite pLDDT below the configured threshold so an
    unrelated, truncated, or malformed output cannot hide a missing prediction.
    """
    if threshold is None or threshold <= 0:
        return False
    ranking = Path(output_root) / input_name / f"{input_name}_ranking_scores.csv"
    try:
        with ranking.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0].get("early_stopped") != "True":
            return False
        plddt = float(rows[0]["mean_plddt"])
        return bool(np.isfinite(plddt) and 0 <= plddt < threshold)
    except (OSError, csv.Error, KeyError, ValueError, TypeError):
        return False


def compile_rf3(
    sandbox_dir, name="rf3", compile_as_metric=False, read_most_confident_model=True,
    expected_seeds=None, early_stopping_plddt_threshold=0.5,
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

    try:
        manifest = json.loads((Path(sandbox_dir) / _RF3_INPUT_MANIFEST).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        manifest = {}

    recognized_stops = set()
    if manifest:
        for seed in expected_seeds or [None]:
            seed_root = Path(search_dir) / f"seed_{seed}"
            selected = paths if seed is None else [
                path for path in paths if Path(path).is_relative_to(seed_root)
            ]
            completed = {_rf3_input_name_for_output(path, manifest) for path in selected}
            stopped = {
                input_name for input_name in set(manifest) - completed
                if _rf3_was_early_stopped(
                    search_dir if seed is None else seed_root,
                    input_name, early_stopping_plddt_threshold,
                )
            }
            recognized_stops.update((seed, input_name) for input_name in stopped)
            missing = set(manifest) - completed - stopped
            if missing:
                raise RuntimeError(
                    "RF3 did not produce outputs for required inputs: "
                    + ", ".join(sorted(missing))
                    + (f" (seed {seed})" if seed is not None else "")
                )

    if not paths and not recognized_stops:
        raise RuntimeError(
            f"RF3 execution completed but no output files were generated in {search_dir}. "
            f"This may indicate:\n"
            f"  - Invalid input specifications\n"
            f"  - Execution errors (check sandbox directory for logs)\n"
            + tail_block_log(search_dir)
        )

    if recognized_stops:
        logger.info("RF3 intentionally early-stopped %d input/seed prediction(s)", len(recognized_stops))

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
                # Native MPNN siblings share a backbone path; keep their exact
                # logical parent as well as the legacy structure backlink.
                parent_design_id=source.get("parent_design_id"),
                as_metric=compile_as_metric,
            )
        )

    return PipelineDataHandler(rows if rows else pl.DataFrame(schema=SCHEMA))


if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "rf3"
    SANDBOX = REPO_ROOT / "unittest_outs" / "rf3"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = RosettaFold3Block(
        num_diffusion_samples=1,
        diffusion_batch_size=1,
    )
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
