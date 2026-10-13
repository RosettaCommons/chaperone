# ESM-2 block for protein sequence embeddings
import logging
from pathlib import Path
from typing import Literal

from pydantic import Field

from chaperone import (
    CompiledBlockFunction,
    PipelineDataHandler,
    JobArgs,
    get_setting_field,
)
from chaperone.block_config import TIER_PRIMARY, ScriptSpec
from chaperone.containers import Image
from blocks.constants import cuda_python_base

logger = logging.getLogger(__name__)

ESM2_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "run_esm2.py"

ESM2_CONTAINER = (
    cuda_python_base(Image("esm2"), "esm2")
    .run(
        ". /opt/esm2-env/bin/activate && "
        "pip install torch --index-url https://download.pytorch.org/whl/cu126 && "
        "pip install transformers safetensors click"
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[
            ("/mnt/home", "/mnt/home"),
            ("/projects/ml/pipelines", "/projects/ml/pipelines"),
        ],
        env={"HF_HUB_CACHE": lambda: get_setting_field("HF_HUB_CACHE", "")},
    )
)


class ESM2Block(CompiledBlockFunction):
    """ESM-2 block for protein sequence embeddings.

    Runs ESM-2 (Evolutionary Scale Modeling 2) to generate per-residue and
    sequence-level embeddings for protein sequences. Uses HuggingFace transformers
    for inference.

    Model sizes:
    - 8m:   320 hidden dim, 6 layers   (fastest, lowest quality)
    - 35m:  480 hidden dim, 12 layers
    - 150m: 640 hidden dim, 30 layers
    - 650m: 1280 hidden dim, 33 layers (default, good balance)
    - 3b:   2560 hidden dim, 36 layers (highest quality, most memory)

    Outputs are saved as safetensors files with per-residue (token_embeddings)
    and mean-pooled sequence embeddings (mean_token_embeddings).
    """

    _name = "esm2"
    _display_name = "ESM-2"
    _description = "ESM-2 protein language model for sequence embeddings"
    _script_spec = ScriptSpec(
        command=["python", str(ESM2_SCRIPT_PATH)],
        image=ESM2_CONTAINER,
        source_url="https://github.com/facebookresearch/esm",
    )
    _resources = JobArgs(
        partition="gpu",
        gres="gpu:small:1",
        mem_gb=16,
        rows_per_job=200,
    )
    _creates_new_step = False
    _inline = False
    _chunked = True

    model_name: Literal["8m", "35m", "150m", "650m", "3b"] | str = Field(
        default="650m",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Model size (8m, 35m, 150m, 650m, 3b) or full HuggingFace path",
    )
    max_batch_tokens: int = Field(
        default=4096,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Maximum number of tokens to batch in a single forward pass",
    )
    fields_to_save: list[str] = Field(
        default=["token_embeddings", "mean_token_embeddings"],
        description="Fields to save in output safetensors",
    )

    def _resolve_model_name(self) -> str:
        """Resolve model shortcut to full HuggingFace path."""
        shortcuts = {
            "8m": "facebook/esm2_t6_8M_UR50D",
            "35m": "facebook/esm2_t12_35M_UR50D",
            "150m": "facebook/esm2_t30_150M_UR50D",
            "650m": "facebook/esm2_t33_650M_UR50D",
            "3b": "facebook/esm2_t36_3B_UR50D",
        }
        return shortcuts.get(self.model_name, self.model_name)

    def create_command(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int,
    ) -> list[str]:
        """Create command to run ESM-2 on input sequences.

        Extracts sequences from the handler, writes them to a FASTA file,
        and builds the apptainer command to run ESM-2 inference.

        Args:
            handler: PipelineDataHandler containing input structures/sequences
            sandbox_dir: Directory for input/output files
            chunk_id: Chunk identifier for parallel execution

        Returns:
            Command list for subprocess execution
        """
        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        input_file = sandbox_dir / "inputs.fasta"
        with open(input_file, "w") as f:
            for loaded in handler.get_data(which="last", load=True):
                seq = loaded.sequence.get("A")
                if seq is None:
                    raise ValueError(f"No sequence for chain A in input data: {loaded}")
                f.write(f">{loaded.basename}\n{seq}\n")

        cmd = self.command_template + [
            "--model-name",
            self._resolve_model_name(),
            "--input-path",
            str(input_file),
            "--output-dir",
            str(sandbox_dir),
            "--max-batch-tokens",
            str(self.max_batch_tokens),
        ]

        # Add fields to save
        for field in self.fields_to_save:
            cmd.extend(["--fields-to-save", field])

        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        """Compile ESM-2 safetensors outputs into a PipelineDataHandler.

        Args:
            sandbox_dir: Directory containing output safetensors files

        Returns:
            PipelineDataHandler with paths to embedding files
        """
        sandbox_dir = Path(sandbox_dir)
        files = list(sandbox_dir.glob("*.safetensors"))
        return PipelineDataHandler.from_paths(files, name=self.name)

    @classmethod
    def load_embeddings(cls, handler, map_location: str = "cpu"):
        """Load embeddings from ESM-2 block output handler.

        Convenience method to load embeddings from the safetensors files
        produced by this block.

        Args:
            handler: PipelineDataHandler from ESM2Block.compile()
            map_location: Device to load tensors to ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            Tuple of (ids, mean_embeddings, token_embeddings)
            - ids: List of sequence identifiers
            - mean_embeddings: numpy array of shape (n_sequences, hidden_dim)
            - token_embeddings: List of numpy arrays or None if not saved
        """
        import numpy as np
        from safetensors import safe_open

        def _load_esm2_tensors(safetensor_path: str | Path) -> dict:
            tensors = {}
            with safe_open(safetensor_path, framework="pt", device=map_location) as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            return tensors

        ids = []
        mean_embeddings = []
        token_embeddings = []

        for data in handler.get_data(which="last"):
            path = data.path
            seq_id = data.metadata.get("sequence_id", Path(path).stem)

            tensors = _load_esm2_tensors(path)
            ids.append(seq_id)

            if "mean_token_embeddings" in tensors:
                mean_embeddings.append(tensors["mean_token_embeddings"].cpu().numpy())
            elif "token_embeddings" in tensors:
                mean_embeddings.append(
                    tensors["token_embeddings"].mean(dim=0).cpu().numpy()
                )

            if "token_embeddings" in tensors:
                token_embeddings.append(tensors["token_embeddings"].cpu().numpy())

        mean_embeddings = np.array(mean_embeddings)
        token_embeddings = token_embeddings if token_embeddings else None

        return ids, mean_embeddings, token_embeddings
