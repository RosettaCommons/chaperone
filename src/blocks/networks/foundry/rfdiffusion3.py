# Basic blocks for design pipelines
import os
import json
import numpy as np
import yaml
import re
from pathlib import Path
from pydantic import Field, field_validator
from typing import Optional
from chaperone import CompiledBlockFunction, PipelineDataHandler, JobArgs
from chaperone.block_config import TIER_PRIMARY, ScriptSpec
from chaperone.data import make_row

from chaperone.common import glob_designs_in_dir, tail_block_log
from blocks.networks.foundry.container import FOUNDRY_CONTAINER
import logging

logger = logging.getLogger(__name__)

###########################################################################
# Design oracle blocks
###########################################################################


# Configuration, the `inputs=` JSON schema, and troubleshooting live in
# rfdiffusion3.md beside this file. Read that rather than this source when
# writing an input spec.
class RFD3Block(CompiledBlockFunction):
    """RFdiffusion3 block for backbone generation."""

    _name = "rfd3"
    _display_name = "RFdiffusion3"
    _description = "RFdiffusion3 block"
    _script_spec = ScriptSpec(
        command=["rfd3", "design"],
        image=FOUNDRY_CONTAINER,
        source_url="https://github.com/RosettaCommons/foundry",
    )
    _resources = JobArgs(
        partition="gpu",
        gres="gpu:small:1",
        mem_gb=16,
        rows_per_job=200,
    )
    _inline = False
    _chunked = True

    # Model and input configuration
    inputs: Optional[str | dict] = Field(
        None,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="""Path to input json/yaml file, or dict of design specifications.
JSON inputs take the following nested top-level structure;
```json
{
    "spec-1": {  // First design configuration
      "input": "<path/to/pdb>",
      "contig": "50-80,/0,A1-100",  // Diffuses length 50-80 monomer in chain A & selects indices A1 -> A100 in input pdb to have fixed coordinates and sequences
      "select_unfixed_sequence": "A20-35", // Converts selected indices in input to have unfixed sequence (inputs become atom14).
      "ligand": "HAX,OAA",  // Selects ligands HAX and OAA based on res name in the input
    },
    "spec-2": {
      // ... args for the second (independent) configuration for design.
    }
}
```
Note that "spec-1", "spec-2", etc must obey the regex [A-Za-z0-9_-], specifically must not contain '.' characters
which can cause downstream processing to mess up.
See the RFdiffusion3 skill for more information
""",
    )
    ckpt_path: Optional[str] = Field(
        "/projects/ml/pipelines/weights/foundry/checkpoints/rfd3_latest.ckpt",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Optional checkpoint path (foundry will manage checkpoints itself)",
    )
    json_keys_subset: list = Field(
        default_factory=list, description="Keys to fetch from json"
    )
    jitter_ori_sigma: float = Field(
        0, description="Jitter origin of the noise cloud by this standard deviation"
    )

    # Diffusion parameters
    # number of forward diffusion samples will be num designs // batch_size
    # NOTE: These are values for each json key (if using json inputs)
    batch_size: int = Field(
        8,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of designs to diffuse per batch",
    )
    n_batches: int = Field(
        2,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of diffusion batches to run (in total)",
    )

    convert_config_to_test_config: bool = Field(
        False,
        description="When True, shorten generated protein length to 50 residues "
        "and use fast inference settings for testing. Only shortens tasks "
        "without 'contig' or 'partial_t' fields.",
    )

    inference_sampler: dict = Field(
        # Defaults made explicit such that they can be easily changed
        default_factory=lambda: {
            "num_timesteps": 200,
            "use_classifier_free_guidance": False,
            "cfg_scale": 1.0,
            "step_scale": 1.5,
        },
        description="Sampler parameters to override (not configurable via CLI)",
    )
    seed_offset: int = Field(
        42,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Increments seed by 1 from offset for every chunk",
    )
    _requires_input = False

    @field_validator("inputs")
    def ensure_input_exists(cls, v):
        if v is None:
            return v  # None allowed for partial diffusion
        if isinstance(v, str):
            if not os.path.exists(v):
                raise ValueError(f"Input file {v} does not exist")
            v = os.path.abspath(v)
        return v

    def _load_inputs_to_dict(self):
        """Load inputs from dict or file path (JSON/YAML) and return as dict.

        Returns:
            tuple: (data_dict, input_path, basename)
                - data_dict: The loaded configuration dictionary
                - input_path: Original file path (None if inputs was a dict)
                - basename: Base name for output files
        """
        # Dictionary input - use directly
        if isinstance(self.inputs, dict):
            return self.inputs.copy()

        # File path input - load from file
        if self.inputs.endswith(".json"):
            with open(self.inputs, "r") as f:
                data = json.load(f)
        elif self.inputs.endswith(".yaml"):
            with open(self.inputs, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Input file {self.inputs} must be a .json or .yaml file")

        # Process data
        data = {
            k: v
            for k, v in data.items()
            if not self.json_keys_subset or k in self.json_keys_subset
        }
        return data

    def _validate_inputs(self, inputs):
        """Validates the input dict to prevent downstream errors which are difficult to catch.

        Returns:
            None if valid, raises ValueError if error is found.
        """
        regex_str = r"^[A-Za-z0-9_-]*$"
        pattern = re.compile(regex_str)
        for key in inputs.keys():
            if not isinstance(key, str) or not pattern.match(key):
                raise ValueError(
                    f"Invalid spec found in RFD3 json: '{key}'. "
                    f"All keys must be strings matching the pattern: {regex_str}"
                )

    def _basename_for_input(self):
        if isinstance(self.inputs, dict):
            return "input"
        return os.path.basename(self.inputs).replace(".json", "").replace(".yaml", "")

    def init_handler(self):
        assert self.inputs is not None, (
            "No input provided (use field inputs={'design': {}} to create inputs inline)."
        )
        from chaperone.data.schema import Column

        rows = []
        for batch_idx in range(self.n_batches):
            rows.append({Column.STRUCTURES: [], Column.METRICS: "{}"})
        return PipelineDataHandler(rows)

    def create_command(self, handler, sandbox_dir, chunk_id):
        """Prepare the full RFDiffusion3 command based on the chunked dataframe."""
        sandbox_dir = os.path.abspath(sandbox_dir)
        inputdir = os.path.join(sandbox_dir, "inputs")
        os.makedirs(inputdir, exist_ok=True)

        n_batches_in_chunk = len(handler)
        data = self._load_inputs_to_dict()
        self._validate_inputs(data)
        basename = self._basename_for_input()
        data = {k: ensure_input_is_abspath(d, self.inputs) for k, d in data.items()}
        if self.convert_config_to_test_config:
            data = _apply_test_config(data)
            self.inference_sampler = {
                **{"num_timesteps": 50, "use_classifier_free_guidance": False},
                **self.inference_sampler,
            }
        # Materialize archived structures before writing the Foundry input JSON.
        for spec_name, spec in data.items():
            if isinstance(spec, dict) and spec.get("input"):
                staged = Path(
                    self.stage_file(spec["input"], sandbox_dir, subdir="inputs")
                )
                spec["input"] = str(staged)

        if self.jitter_ori_sigma > 0:
            data = jitter_ori_token_in_input(
                data, self.jitter_ori_sigma, self.json_keys_subset
            )

        chunk_input_src = f"{inputdir}/{basename}_chunk{chunk_id}.json"
        json.dump(data, open(chunk_input_src, "w"), indent=4)

        return self._build_command(
            sandbox_dir,
            chunk_input_src,
            n_batches_in_chunk,
            chunk_id,
            global_prefix=f"{basename}_chunk{chunk_id}_",
        )

    def _build_command(
        self, sandbox_dir, chunk_input_src, n_batches_in_chunk, chunk_id, **extra_args
    ):
        """Build the RFD3 CLI command.

        Args:
            sandbox_dir: Output directory for this chunk.
            chunk_input_src: Path to the chunked input JSON.
            n_batches_in_chunk: Number of batches in this chunk.
            chunk_id: Chunk index (used for seed offset).
            **extra_args: Additional CLI args as key=value pairs. None values are skipped.
        """
        subset_arg = (
            "json_keys_subset={}".format(",".join(self.json_keys_subset))
            if self.json_keys_subset
            else None
        )
        sampler_args = [
            f"inference_sampler.{k}={v}" for k, v in self.inference_sampler.items()
        ]
        seed_arg = f"seed={self.seed_offset + chunk_id}" if self.seed_offset else None
        ckpt = self.ckpt_path
        ckpt_arg = f"ckpt_path={ckpt}" if ckpt else None
        cmd = self.command_template + [
            f"out_dir={sandbox_dir}",
            ckpt_arg,
            f"inputs={chunk_input_src}",
            subset_arg,
            f"diffusion_batch_size={self.batch_size}",
            f"n_batches={n_batches_in_chunk}",
            *sampler_args,
            seed_arg,
        ]
        for k, v in extra_args.items():
            if v is not None:
                cmd.append(f"{k}={v}")
        return list(filter(None, cmd))

    def _create_partial_diffusion_inputs(self, handler, sandbox_dir=None) -> dict:
        """Create JSON input specifications from handler for partial diffusion.

        For each design in the handler, creates an RFD3 spec with:
        - input: path to the design PDB
        - contig/ligand: from diffused_index_map if available, otherwise omitted
          (RFD3 will diffuse the entire input structure)
        - partial_t: noise level for partial diffusion
        """
        data = {}
        for loaded in handler.get_data(which="last"):
            # Materialize archived structures; ordinary paths pass through.
            input_path = str(self.stage_file(loaded.path, sandbox_dir, subdir="inputs"))

            spec = {
                "input": input_path,
                "partial_t": self.partial_t,
                # dialect-2 defaults select_fixed_atoms=True → all atoms fixed → zero noise.
                # False = diffuse all protein; ligand stays fixed via the "ligand" key.
                "select_fixed_atoms": False,
            }

            # Extract diffused_index_map from metadata (if available)
            diffused_index_map_str = loaded.metadata.get("diffused_index_map")
            if diffused_index_map_str:
                if isinstance(diffused_index_map_str, str):
                    diffused_index_map = json.loads(diffused_index_map_str)
                else:
                    diffused_index_map = diffused_index_map_str
            else:
                diffused_index_map = {}

            if diffused_index_map:
                spec["contig"] = ",".join(str(v) for v in diffused_index_map.values())
                ligand_keys = [k for k in diffused_index_map if k.startswith("L:")]
                if ligand_keys:
                    spec["ligand"] = ",".join(ligand_keys)

            # Resolve ligand: explicit field > diffused_index_map > upstream metadata
            if self.ligand_name:
                spec["ligand"] = self.ligand_name
            elif "ligand" not in spec:
                ligand_meta = loaded.metadata.get("ligand")
                if ligand_meta:
                    if isinstance(ligand_meta, list):
                        spec["ligand"] = ",".join(ligand_meta)
                    elif isinstance(ligand_meta, str):
                        spec["ligand"] = ligand_meta

            data[loaded.basename] = spec

        return data

    def compile(self, sandbox_dir):
        return compile_rfd3(
            sandbox_dir,
            name=self.name,
        )


class RFD3PartialBlock(RFD3Block):
    """RFdiffusion3 partial diffusion. Useful for diversifying structures."""

    _name = "rfd3_partial"
    _display_name = "RFdiffusion3"
    _description = "RFdiffusion3 partial diffusion"
    _requires_input = True

    partial_t: int = Field(
        5,
        description="Timestep for partial diffusion (e.g., 5 = ~80% diffused).",
    )
    partial_diffusion_mode: str = Field(
        "indexed", description="Mode for partial diffusion"
    )
    ligand_name: Optional[str] = Field(
        None,
        description="Ligand name for partial diffusion specs (e.g., 'LIG_B'). "
        "Overrides ligand from metadata when set.",
    )

    def create_command(self, handler, sandbox_dir, chunk_id):
        sandbox_dir = os.path.abspath(sandbox_dir)
        inputdir = os.path.join(sandbox_dir, "inputs")
        os.makedirs(inputdir, exist_ok=True)

        data = self._create_partial_diffusion_inputs(handler, sandbox_dir=sandbox_dir)
        basename = "partial_diffusion"
        chunk_input_src = f"{inputdir}/{basename}_chunk{chunk_id}.json"
        json.dump(data, open(chunk_input_src, "w"), indent=4)

        # Empty global_prefix prevents RFD3 from using the JSON filename as a
        # prefix (which breaks merge).
        return self._build_command(
            sandbox_dir,
            chunk_input_src,
            self.n_batches,
            chunk_id,
            global_prefix="",
        )


def extract_rfd3_design_metadata(data: dict) -> dict:
    specification = data.get("specification") or {}
    extra = specification.get("extra") or {}
    diffused_index_map = data.get("diffused_index_map") or {}
    ligand = specification.get("ligand") or ""
    return {
        "task": (
            data.get("task")
            or extra.get("example")
            or extra.get("task_name")
            or extra.get("example_id")
        ),
        "original_input": specification.get("input"),
        "diffused_index_map": json.dumps(diffused_index_map),
        # Also parse to motif pdb idx and ligands (exclude ligand entries)
        "motif": [
            v
            for k, v in diffused_index_map.items()
            if re.match(r"[A-Z]\d+", k)
        ]
        or None,
        "ligand": [x.strip() for x in ligand.split(",") if x.strip()] or None,
    }


def extract_rfd3_metadata(data: dict) -> dict:
    metrics = data.get("metrics", {})
    if metrics and "join_point_rmsd_by_token" in metrics:
        metrics = {k: v for k, v in metrics.items() if k != "join_point_rmsd_by_token"}

    return metrics


def compile_rfd3(sandbox_dir, name="rfd3"):
    """Compile RFD3 outputs into a PipelineDataHandler."""

    paths = glob_designs_in_dir(dir=sandbox_dir)
    paths = [p for p in paths if not ("_denoised_model_" in p or "_noisy_model_" in p)]

    if len(paths) == 0:
        raise RuntimeError(
            f"RFD3 execution completed but no output files were generated in {sandbox_dir}. "
            f"This may indicate:\n"
            f"  - Invalid input specifications\n"
            f"  - Execution errors (check sandbox directory for logs)\n"
            f"  - Filtering removed all outputs\n"
            + tail_block_log(sandbox_dir)
        )

    # Single path per row with separated design_metadata and step_metadata
    rows = []
    for path in paths:
        if path.endswith(".cif.gz"):
            meta_path = path.replace(".cif.gz", ".json")
        elif path.endswith(".cif"):
            meta_path = path.replace(".cif", ".json")
        else:
            meta_path = path.replace(".pdb", ".json")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
            metadata = extract_rfd3_design_metadata(data)
            metrics = extract_rfd3_metadata(data)
        else:
            logger.error(f"Metadata JSON file {meta_path} not found for design {path}")
            metadata = {}
            metrics = {}

        rows.append(
            make_row(
                name=name,
                path=path,
                metadata=metadata,
                metrics={name: metrics},
            )
        )

    return PipelineDataHandler(rows)


###########################################################################
# Helper functions
###########################################################################


# TODO: This is duplicate code with modelhub projects.
def ensure_input_is_abspath(args: dict, path: str | None):
    """
    Ensures the input source is an absolute path if exists, if not it will convert

    args:
        spec: Inference specification for atom array
        path: None or file to which the input is relative to.

    Note: Relative paths in the input JSON are resolved against the JSON file's
    parent directory. For example, if the JSON is at ``files/config.json`` and
    contains ``"input": "7c7m.pdb"``, the path resolves to ``files/7c7m.pdb``.
    A common mistake is to include the parent directory name in the input path
    (e.g. ``"input": "files/7c7m.pdb"``), which would resolve to
    ``files/files/7c7m.pdb``.
    """
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise TypeError(
            f"Invalid RFD3 input specification: expected a dict, got {type(args).__name__}: {args!r}.\n"
            f"Each key in the inputs dict should map to a specification dict, e.g.:\n"
            f'  {{"spec-1": {{"contig": "50-80", "input": "path/to/pdb"}}}}\n'
            f"See the rfdiffusion3-block skill for detailed input format guidance."
        )
    if "input" not in args or args["input"] is None:
        return args
    input = args["input"]
    if not os.path.isabs(input):
        input = os.path.abspath(os.path.join(os.path.dirname(str(path)), input))
        args["input"] = input
    assert os.path.exists(input), f"Input file {input} does not exist"
    return args


def _apply_test_config(data: dict, test_length: str = "50-50") -> dict:
    """Shorten protein length for fast testing.

    For each task that has no 'contig' and no 'partial_t' (i.e. de novo design
    driven by 'length'), override the length to a short value.
    """
    for key, spec in data.items():
        if not isinstance(spec, dict):
            continue
        has_contig = spec.get("contig") is not None
        has_partial_t = spec.get("partial_t") is not None
        if not has_contig and not has_partial_t:
            original = spec.get("length", "unknown")
            spec["length"] = test_length
            logger.info(
                f"convert_config_to_test_config: {key} length {original} -> {test_length}"
            )
    return data


def jitter_ori_token(ori_token, sigma):
    """Adds Gaussian noise to the origin placement in the input."""
    ori_token = np.array(ori_token, dtype=np.float32)
    noise = np.random.normal(0, sigma, ori_token.shape)
    return ori_token + noise


def jitter_ori_token_in_input(input_dict, sigma, json_keys_subset=None):
    """
    Adds a jittered origin token to the input dictionary.

    Args:
        input_dict (dict): The input dictionary to modify.
        sigma (float): Standard deviation for Gaussian noise.

    Returns:
        dict: The modified input dictionary with the new origin token.
    """

    for design_case, design_specifications in input_dict.items():
        if json_keys_subset is not None and design_case not in json_keys_subset:
            continue
        if "ori_token" not in design_specifications:
            raise ValueError(
                f"ori_token not found in design_case {design_case}, You cannot jitter the origin token if you don't provide it"
            )
        # Jitter the origin of the coordinate system
        jittered_token = jitter_ori_token(design_specifications["ori_token"], sigma)
        logger.info(
            f"Provided ori_token: {design_specifications['ori_token']}, Jittered ori_token: {jittered_token.tolist()}"
        )
        design_specifications["ori_token"] = jittered_token.tolist()

    return input_dict


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "rfd3"
    SANDBOX = REPO_ROOT / "unittest_outs" / "rfd3"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = RFD3Block(
        inputs=str(TEST_DATA / "rfd3_config.json"),
        n_batches=1,
        batch_size=1,
    )
    handler = block.init_handler()
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
