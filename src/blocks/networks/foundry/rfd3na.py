"""RFdiffusion3NA - all-atom protein/DNA/RNA design.

Wraps the public ``rfd3na`` CLI shipped in the foundry container
(``/opt/foundry/models/rfd3na``). Differs from RFD3 only in the entry point, the
checkpoint, and two flags the upstream README recommends for NA design, so
everything else - input JSON handling, batch chunking, per-chunk seeding and
output compilation - is inherited unchanged.

Nucleic-acid design is expressed entirely through the input JSON: contig parts
take an ``R`` or ``D`` suffix to request RNA or DNA, e.g. ``"10-10,20-20R,30-30D"``
designs a 10-residue protein chain, a 20-nucleotide RNA chain and a 30-nucleotide
DNA chain. See ``rfd3na_input.py`` for building ligand-conditioned specs.
"""

from typing import Optional

from pydantic import Field

from chaperone import JobArgs
from chaperone.block_config import TIER_PRIMARY, ScriptSpec

from blocks.networks.foundry.container import FOUNDRY_CONTAINER, default_checkpoint
from blocks.networks.foundry.rfdiffusion3 import RFD3Block

# Install with: foundry install rfd3na --checkpoint-dir <dir>
# Source: https://files.ipd.uw.edu/pub/rfdiffusion3na/rfd3na-1190.ckpt
RFD3NA_CKPT = "rfd3na"


class RFD3NABlock(RFD3Block):
    """RFdiffusion3NA block for nucleic-acid and mixed-polymer design."""

    _name = "rfd3na"
    _display_name = "RFdiffusion3NA"
    _description = "RFdiffusion3NA block (all-atom protein/DNA/RNA design)"
    _script_spec = ScriptSpec(
        command=["rfd3na", "design"],
        image=FOUNDRY_CONTAINER,
        source_url="https://github.com/RosettaCommons/foundry/tree/production/models/rfd3na",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        cpus_per_task=2,
        time_limit="6:00:00",
        rows_per_job=100,
    )

    ckpt_path: Optional[str] = Field(
        default_factory=lambda: default_checkpoint("rfd3na", "rfd3na_1190.ckpt"),
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "RFD3NA checkpoint path. Set to 'rfd3na' or None to use Foundry's "
            "checkpoint registry/search path."
        ),
    )
    inference_sampler: dict = Field(
        default_factory=lambda: {
            "num_timesteps": 200,
            "use_classifier_free_guidance": False,
            # RFD3NA requires 1.5 even when classifier-free guidance is disabled.
            "cfg_scale": 1.5,
            "step_scale": 1.5,
        },
        description="RFD3NA inference-sampler overrides.",
    )
    read_sequence_from_sequence_head: bool = Field(
        False,
        description="Upstream README recommends False as the global setting for RFD3NA.",
    )
    prevalidate_inputs: bool = Field(
        True,
        description="Validate every RFD3NA input specification before loading the model.",
    )
    low_memory_mode: bool = Field(
        False,
        description="Use memory-efficient tokenization when GPU memory is constrained.",
    )
    dump_trajectories: bool = Field(
        False,
        description="Write large noisy and denoised diffusion trajectory files.",
    )
    cloud: Optional[bool] = Field(
        None,
        description="Cloud execution is unsupported until an RFD3NA checkpoint is staged.",
    )
    skip_existing: bool = Field(
        True, description="Skip designs whose output file already exists."
    )

    def _build_command(
        self, sandbox_dir, chunk_input_src, n_batches_in_chunk, chunk_id, **extra_args
    ):
        # RFD3Block rewrites ckpt_path to the protein-only rfd3_latest.ckpt when
        # running on cloud. There is no cloud RFD3NA checkpoint, so fail loudly
        # rather than silently produce protein designs.
        if self.cloud:
            raise NotImplementedError(
                "rfd3na has no cloud checkpoint; RFD3Block would substitute the "
                "protein-only rfd3_latest.ckpt. Run on-prem instead."
            )
        return super()._build_command(
            sandbox_dir,
            chunk_input_src,
            n_batches_in_chunk,
            chunk_id,
            read_sequence_from_sequence_head=self.read_sequence_from_sequence_head,
            prevalidate_inputs=self.prevalidate_inputs,
            low_memory_mode=self.low_memory_mode,
            dump_trajectories=self.dump_trajectories,
            skip_existing=self.skip_existing,
            **extra_args,
        )
