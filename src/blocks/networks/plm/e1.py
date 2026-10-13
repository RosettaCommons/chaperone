# E1 block for masked language modeling on protein sequences
import logging
from pathlib import Path
from typing import Literal

from pydantic import Field

from pipelines import CompiledBlockFunction, PipelineDataHandler, JobArgs
from pipelines.block_config import TIER_PRIMARY, ScriptSpec
from pipelines.containers import Image
from blocks.constants import cuda_python_base

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "run_e1.py"

E1_CONTAINER = (
    cuda_python_base(Image("e1"), "e1")
    .run(
        "cd /opt && git clone https://github.com/Profluent-AI/E1.git",
        ". /opt/e1-env/bin/activate && cd /opt/E1 && pip install . && pip install accelerate",
    )
    .run(
        ". /opt/e1-env/bin/activate && "
        "TMPDIR=/tmp pip cache purge && pip install flash-attn --no-build-isolation --no-cache-dir"
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .runscript('"$@"')
)


class E1Block(CompiledBlockFunction):
    """E1 block for masked language modeling on protein sequences.

    Runs E1 model to get logits and embeddings for sequences extracted from structures.
    Outputs are saved as safetensors files.
    """

    _name = "e1"
    _display_name = "E1"
    _description = "E1 masked language model for sequence prediction"
    _script_spec = ScriptSpec(
        command=["python", str(SCRIPT_PATH)],
        image=E1_CONTAINER,
        source_url="https://github.com/Profluent-AI/E1",
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

    model_name: Literal["150m", "300m", "600m"] | str = Field(
        default="150m",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Model size (150m, 300m, 600m) or full HuggingFace path",
    )
    max_batch_tokens: int = Field(
        default=65536,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Maximum number of tokens to batch in a single forward pass",
    )
    save_masked_positions_only: bool = Field(
        default=False,
        description="Whether to save logits/embeddings for only the masked positions",
    )
    fields_to_save: list[str] = Field(
        default=["logits", "token_embeddings", "mean_token_embeddings"],
        description="Fields to save in output safetensors",
    )
    context_fasta: str | None = Field(
        default=None,
        description="Optional path to context FASTA for retrieval-augmented prediction",
    )

    def _resolve_model_name(self) -> str:
        """Resolve the configured model name to a full HuggingFace repository path.

        Known size shortcuts (e.g. "150m") are expanded; other values are returned unchanged.
        """
        shortcuts = {
            "150m": "Profluent-Bio/E1-150m",
            "300m": "Profluent-Bio/E1-300m",
            "600m": "Profluent-Bio/E1-600m",
        }
        return shortcuts.get(self.model_name, self.model_name)

    def create_command(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int,
    ) -> list[str]:
        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        input_file = sandbox_dir / "inputs.fasta"

        if not (
            input_file.exists() and sum(1 for _ in open(input_file)) >= len(handler) * 2
        ):
            with open(input_file, "w") as f:
                for loaded in handler.get_data(which="last", load=True):
                    seq = loaded.sequence.get("A")
                    if seq is None:
                        raise ValueError(
                            f"No sequence for chain A in input data: {loaded}"
                        )
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
            "--save-masked-positions-only",
            str(self.save_masked_positions_only),
        ]

        for field in self.fields_to_save:
            cmd.extend(["--fields-to-save", field])

        if self.context_fasta:
            ctx = str(self.stage_file(self.context_fasta, sandbox_dir))
            cmd.extend(["--context-path", ctx])

        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = Path(sandbox_dir)
        files = list(sandbox_dir.glob("*.safetensors"))
        return PipelineDataHandler.from_paths(files, name=self.name)

    @classmethod
    def load_embeddings(cls, handler, map_location: str = "cpu"):
        """Load embeddings from E1 block output handler."""
        import numpy as np
        from safetensors import safe_open

        def _load_tensors(path):
            tensors = {}
            with safe_open(path, framework="pt", device=map_location) as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            return tensors

        ids, mean_emb, token_emb = [], [], []
        for data in handler.get_data(which="last"):
            tensors = _load_tensors(data.path)
            ids.append(data.metadata.get("sequence_id", Path(data.path).stem))
            if "mean_token_embeddings" in tensors:
                mean_emb.append(tensors["mean_token_embeddings"].cpu().numpy())
            elif "token_embeddings" in tensors:
                mean_emb.append(tensors["token_embeddings"].mean(dim=0).cpu().numpy())
            if "token_embeddings" in tensors:
                token_emb.append(tensors["token_embeddings"].cpu().numpy())

        return ids, np.array(mean_emb), token_emb or None
