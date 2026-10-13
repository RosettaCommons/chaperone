"""Foundry MPNN block for inverse folding (Protein/LigandMPNN)."""

import json
import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator

from pipelines import CompiledBlockFunction, PipelineDataHandler, JobArgs
from pipelines.block_config import TIER_PRIMARY, ScriptSpec
from pipelines.common import glob_designs_in_dir, is_apple_silicon
from pipelines.data import make_row

from blocks.networks.mpnn_io import (
    load_manifest,
    manifest_entry_for_fasta,
    parse_fasta,
    sequence_row,
    write_manifest,
)

from blocks.networks.foundry.container import FOUNDRY_CONTAINER

logger = logging.getLogger(__name__)


class FoundryMPNNBlock(CompiledBlockFunction):
    """Foundry Protein/LigandMPNN for inverse folding.

    Uses the foundry ``mpnn`` CLI with JSON-driven config. Supports
    protein_mpnn and ligand_mpnn model types. Sequence-native FASTA output is
    the default; generated CIF structures remain available as a legacy mode.

    Note: MPS has a GatherND crash on Apple Silicon. On ARM64 Macs the
    block auto-generates a wrapper that forces CPU execution.
    """

    _name = "foundry_mpnn"
    _display_name = "ProteinMPNN"
    _description = "Foundry MPNN for inverse folding"
    _script_spec = ScriptSpec(
        command=["mpnn"],
        image=FOUNDRY_CONTAINER,
        source_url="https://github.com/RosettaCommons/foundry",
    )
    _resources = JobArgs(
        partition="cpu",
        mem_gb=8,
        rows_per_job=100,
    )
    _inline = False
    _chunked = True
    _emits_sequence_table = True

    # Model configuration
    model_type: Literal["protein_mpnn", "ligand_mpnn"] = Field(
        default="ligand_mpnn",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Model type: protein_mpnn or ligand_mpnn.",
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to model checkpoint. None uses foundry default.",
    )

    # Sampling
    num_seq_per_pdb: int = Field(
        default=8,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of sequences to generate per structure.",
    )
    output_mode: Literal["sequences", "structures"] = Field(
        default="sequences",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Return compact sequence rows or legacy generated CIF structures.",
    )
    temperature: float = Field(
        default=0.1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Sampling temperature. Lower = more conservative.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility.",
    )

    # Design constraints
    fixed_residues: Optional[list[str]] = Field(
        default=None,
        description="Residues to keep fixed, e.g. ['A12', 'A13'].",
    )
    designed_residues: Optional[list[str]] = Field(
        default=None,
        description="Residues to redesign (all others fixed).",
    )
    fixed_chains: Optional[list[str]] = Field(
        default=None,
        description="Chain IDs to keep fixed.",
    )
    designed_chains: Optional[list[str]] = Field(
        default=None,
        description="Chain IDs to redesign.",
    )

    # Pipeline integration
    keep_motif_sequences_fixed: bool = Field(
        default=True,
        description="Keep sequences of motifs fixed (from RFD3 metadata).",
    )

    @model_validator(mode="after")
    def validate_output_mode(self):
        if self.output_mode == "sequences" and self.store != "none":
            raise ValueError("Sequence-only MPNN output cannot be stored as catcif.")
        return self

    def _build_config(self, handler, sandbox_dir):
        """Build a JSON config dict for foundry MPNN from the handler."""
        inputs = []
        manifest = {}
        for data in handler.get_data(
            which="last", load=self.output_mode == "sequences"
        ):
            # Route through stage_file so catcif-tagged paths get materialized
            # to a real .cif in the sandbox before the inner tool sees them.
            input_path = os.path.abspath(
                str(self.stage_file(data.path, sandbox_dir, subdir="inputs"))
            )
            basename = Path(input_path).stem
            if basename.endswith(".cif"):
                basename = basename[:-4]

            if self.output_mode == "sequences":
                chain_sequences = {
                    str(chain): str(sequence)
                    for chain, sequence in (data.sequence or {}).items()
                    if sequence
                }
                manifest[basename] = {
                    "input_name": basename,
                    "parent_design_id": data.design_id or data.basename,
                    "chain_ids": list(chain_sequences),
                    "chain_lengths": [len(seq) for seq in chain_sequences.values()],
                }

            input_spec = {
                "structure_path": input_path,
                "name": basename,
                "batch_size": self.num_seq_per_pdb,
                "number_of_batches": 1,
                "temperature": self.temperature,
            }

            if self.seed is not None:
                input_spec["seed"] = self.seed

            # Resolve fixed residues from metadata or explicit config
            fixed = list(self.fixed_residues or [])
            if self.keep_motif_sequences_fixed:
                motif = data.metadata.get("motif") or []
                fixed.extend(motif)
            if fixed:
                input_spec["fixed_residues"] = list(set(fixed))

            if self.designed_residues is not None:
                input_spec["designed_residues"] = self.designed_residues
            if self.fixed_chains is not None:
                input_spec["fixed_chains"] = self.fixed_chains
            if self.designed_chains is not None:
                input_spec["designed_chains"] = self.designed_chains

            inputs.append(input_spec)

        config = {
            "model_type": self.model_type,
            "is_legacy_weights": True,
            "out_directory": os.path.abspath(sandbox_dir),
            "write_fasta": True,
            "write_structures": self.output_mode == "structures",
            "inputs": inputs,
        }

        # Leave the checkpoint unset so the Foundry process resolves its own
        # registry default inside FOUNDRY_CONTAINER. Importing that registry in
        # this outer Pipelines worker fails because rc-foundry intentionally
        # lives only in the nested scientific image. An explicit override is
        # still passed through unchanged.
        if self.checkpoint_path is not None:
            config["checkpoint_path"] = self.checkpoint_path

        if self.output_mode == "sequences":
            write_manifest(sandbox_dir, manifest)

        return config

    def _build_cpu_wrapper(self, config_dir, chunk_id):
        """Write a Python wrapper that disables MPS before running MPNN.

        MPS has a GatherND assertion crash in the MPNN inference engine.
        The wrapper patches ``torch.backends.mps`` and uses ``fork`` start
        method so subprocess workers inherit the patch.
        """
        wrapper_path = config_dir / f"_mpnn_cpu_wrapper_{chunk_id}.py"
        wrapper_path.write_text(
            '"""Auto-generated: force CPU on Apple Silicon (MPS GatherND crash)."""\n'
            "import torch, multiprocessing\n"
            "torch.backends.mps.is_available = lambda: False\n"
            "torch.backends.mps.is_built = lambda: False\n"
            "multiprocessing.set_start_method('fork', force=True)\n"
            "from mpnn.inference import main\n"
            "main()\n"
        )
        return str(wrapper_path)

    def create_command(self, handler, sandbox_dir, chunk_id):
        sandbox_dir = Path(sandbox_dir)
        config_dir = sandbox_dir / "inputs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_config(handler, sandbox_dir)
        config_path = config_dir / f"mpnn_config_chunk{chunk_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        if is_apple_silicon():
            # MPS crashes with GatherND assertion — force CPU via wrapper
            wrapper = self._build_cpu_wrapper(config_dir, chunk_id)
            logger.info("Apple Silicon: wrapping MPNN to force CPU (MPS GatherND bug)")
            return ["python", wrapper, "--config_json", str(config_path)]

        return self.command_template + [
            "--config_json",
            str(config_path),
        ]

    def compile(self, sandbox_dir):
        """Compile foundry MPNN outputs into a PipelineDataHandler.

        Sequence mode parses ``{name}.fa`` records. Structure mode compiles
        legacy ``{name}_b{batch}_d{design}.cif`` files.
        """
        if self.output_mode == "sequences":
            manifest = load_manifest(sandbox_dir)
            fasta_paths = sorted(Path(sandbox_dir).glob("*.fa"))
            self._guard_outputs(fasta_paths, str(sandbox_dir))
            rows = []
            for fasta_path in fasta_paths:
                parent = manifest_entry_for_fasta(fasta_path, manifest)
                for index, (header, sequence) in enumerate(parse_fasta(fasta_path)):
                    rows.append(
                        sequence_row(
                            block_name=self.name,
                            parent=parent,
                            header=header,
                            sequence=sequence,
                            delimiter=None,
                            fallback_index=index,
                        )
                    )
            self._guard_outputs(rows, "generated records in Foundry MPNN FASTA files")
            return PipelineDataHandler(rows)

        paths = glob_designs_in_dir(dir=sandbox_dir)
        self._guard_outputs(paths, str(sandbox_dir))

        rows = []
        for path in paths:
            rows.append(
                make_row(
                    name=self.name,
                    path=str(path),
                )
            )
        return PipelineDataHandler(rows)
