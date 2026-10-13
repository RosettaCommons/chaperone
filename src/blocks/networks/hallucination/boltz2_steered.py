"""Boltz-2 structure prediction with helix steering from Protein Hunter.

Uses the Protein Hunter fork of Boltz2 which adds pair-representation
steering to suppress alpha-helix formation on the binder chain during
diffusion. When ``randomly_kill_helix_feature`` is enabled, the pair
embeddings z at (i, i+4) positions are scaled by a random negative factor,
disrupting the helix H-bond pattern and encouraging diverse secondary
structure.

Reference: Cho et al., "Protein Hunter: exploiting structure hallucination
within diffusion for protein design" (2025).
"""

import os

from pydantic import Field

from pipelines import JobArgs
from pipelines.block_config import TIER_PRIMARY

from blocks.networks.prediction.boltz2 import Boltz2Block
from blocks.networks.hallucination.protein_hunter import PROTEIN_HUNTER_CONTAINER

import logging

logger = logging.getLogger(__name__)

# Path to wrapper script inside the container (copied at build time or bind-mounted)
_WRAPPER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "boltz2_steered_predict.py"
)


class Boltz2SteeredBlock(Boltz2Block):
    """Boltz-2 prediction with helix-steering from Protein Hunter.

    Extends the standard Boltz2 block with pair-representation steering
    to discourage alpha-helix formation on a specified binder chain.
    Uses the Protein Hunter container which includes the forked Boltz2
    model with steering support.
    """

    _name = "boltz2_steered"
    _display_name = "Boltz-2 Steered"
    _description = "Boltz-2 with helix steering (Protein Hunter)"
    _image = PROTEIN_HUNTER_CONTAINER
    _resources = JobArgs(
        partition="gpu",
        gres="gpu:small:1",
        mem_gb=32,
        rows_per_job=200,
    )

    # Steering parameters
    randomly_kill_helix_feature: bool = Field(
        False,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Scale pair embeddings at (i, i+4) positions by a random negative "
        "factor on the binder chain, disrupting helix formation.",
    )
    negative_helix_constant: float = Field(
        0.2,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Magnitude of the negative scaling factor for helix disruption. "
        "Higher = stronger suppression.",
    )
    binder_chain: str = Field(
        "A",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Chain ID of the binder to apply helix steering to.",
    )

    def get_command_template(self) -> list[str]:
        # Run the wrapper script via the PH container's python
        wrapper = os.path.abspath(_WRAPPER_SCRIPT)
        cmd = self._image.build_command("python", wrapper)
        # Add --writable-tmpfs for the PH container
        try:
            img_idx = next(
                i
                for i, c in enumerate(cmd)
                if c.startswith("${CONTAINER:") or c.endswith(".sif")
            )
            cmd.insert(img_idx, "--writable-tmpfs")
        except StopIteration:
            pass
        return cmd

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        """Build command for steered Boltz2 prediction.

        Reuses the parent's YAML generation but calls our wrapper script
        instead of `boltz predict`.
        """
        import yaml as _yaml

        sandbox_dir = os.path.abspath(sandbox_dir)
        inputdir = os.path.join(sandbox_dir, "inputs")
        os.makedirs(inputdir, exist_ok=True)

        # Write YAML inputs (reuse parent logic)
        for loaded in handler.get_data(which="last", load=True):
            yaml_file = os.path.join(inputdir, f"{loaded.basename}.yaml")
            if os.path.exists(yaml_file):
                logger.info(f"Skipping existing input: {yaml_file}")
                continue

            if self.template_yaml:
                from blocks.networks.prediction.helpers import (
                    fill_boltz_yaml_from_template,
                )

                yaml_data = fill_boltz_yaml_from_template(
                    template_yaml=self.template_yaml,
                    atom_array=loaded.atom_array,
                )
            else:
                from blocks.networks.prediction.helpers import (
                    create_boltz_yaml_from_design,
                )

                yaml_data = create_boltz_yaml_from_design(
                    atom_array=loaded.atom_array,
                    remove_ligands=self.remove_ligands,
                    specified_smiles=self.specified_smiles,
                    specified_ccd=self.specified_ccd,
                    disable_ccd_detection=self.disable_ccd_detection,
                    predict_affinity=self.predict_affinity,
                )

            with open(yaml_file, "w") as f:
                _yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        # Build command for the wrapper script
        cmd = self.command_template + [
            inputdir,
            "--out_dir",
            sandbox_dir,
            "--cache",
            self.cache_dir,
            "--sampling_steps",
            str(self.num_diffusion_steps),
            "--diffusion_samples",
            str(self.diffusion_batch_size),
            "--max_parallel_samples",
            str(self.max_parallel_samples),
            "--step_scale",
            str(self.step_scale),
            "--output_format",
            self.output_format,
        ]

        if self.n_recycles is not None:
            cmd += ["--recycling_steps", str(self.n_recycles)]

        # Steering flags
        if self.randomly_kill_helix_feature:
            cmd.append("--randomly_kill_helix_feature")
        cmd.extend(["--negative_helix_constant", str(self.negative_helix_constant)])
        cmd.extend(["--binder_chain", self.binder_chain])

        if self.use_potentials:
            cmd.append("--use_potentials")

        return cmd
