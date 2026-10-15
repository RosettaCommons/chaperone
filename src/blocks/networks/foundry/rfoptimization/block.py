"""RFOptimization of input complexes; see input.md for supported semantics."""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Literal

import yaml
import numpy as np
from biotite.structure.io.pdb import PDBFile
from pydantic import Field

from chaperone import (
    CompiledBlockFunction,
    JobArgs,
    PipelineDataHandler,
    get_setting_field,
)
from chaperone.block_config import ScriptSpec
from chaperone.data import make_row
from blocks.networks.foundry.rfoptimization.container import RFOPTIMIZATION_CONTAINER

# Execute each input independently and propagate a failed cycle as a failed job.
_RUN_CONFIGS = """import json, subprocess, sys
for config in json.load(open(sys.argv[1])):
    subprocess.run(['/opt/rfo-env/bin/rfo', '--config', config], check=True)
"""


class RFOptimizationBlock(CompiledBlockFunction):
    """Refine chain A of an existing complex with RF3/Boltz and MPNN.

    Outputs the final redesigned complex. Side chains at mutations are removed
    by upstream RFO, not repacked. Refold/pack before scoring the final sequence.
    """

    _name = "rfoptimization"
    _display_name = "RFOptimization"
    _description = "RF3 gradient / Boltz-2 cycling with external MPNN sequence redesign"
    _script_spec = ScriptSpec(
        command=["/opt/rfo-env/bin/python", "-c", _RUN_CONFIGS],
        image=RFOPTIMIZATION_CONTAINER,
        source_url="https://github.com/RosettaCommons/foundry/tree/94b766fa7c62d43d0795d24ff1353de0076065f5/models/rfo",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="large",
        mem_gb=64,
        rows_per_job=1,
        time_limit="04:00:00",
    )
    _pipeline_inputs = ["last"]
    _inline = False
    _chunked = True

    total_cycles: int = Field(
        10, ge=1, description="Number of RF3/Boltz + MPNN cycles per input."
    )
    backprop_fraction: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Probability of RF3 gradients per cycle; otherwise Boltz-2.",
    )
    mpnn_model_type: Literal["protein_mpnn", "ligand_mpnn"] = "protein_mpnn"
    mpnn_temperature: float = Field(0.1, gt=0)
    seed: int = Field(42, ge=0)
    rf3_checkpoint: str | None = Field(
        None,
        description="Public confidence-enabled RF3 checkpoint; required when RF3 is used. Defaults to configured Foundry checkpoint directory.",
    )
    boltz_cache: str = Field(
        default_factory=lambda: get_setting_field(
            "BOLTZ_CACHE_DIR", str(Path.home() / ".boltz")
        )
    )
    rf3_config: str | None = Field(
        None,
        description="Optional upstream optimizer YAML. Omit to use the pinned cycle_ppi preset.",
    )
    msa_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Optional chain to A3M path mapping; no remote MSA service is used.",
    )
    templates: dict[str, str] = Field(
        default_factory=dict,
        description="Optional chain to template path mapping for the RF3 branch.",
    )
    cyclic: bool = Field(
        False,
        description="Mark chain A cyclic in Boltz only; RF3 does not constrain its closure.",
    )

    def prepare_inputs(self, handler, sandbox_dir):
        if handler is None or not len(handler):
            raise ValueError(
                "rfoptimization requires an input complex with designed chain A"
            )
        return handler

    def _checkpoint(self):
        if self.backprop_fraction == 0:
            return None
        if self.rf3_checkpoint:
            path = Path(self.rf3_checkpoint).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"RF3 checkpoint does not exist: {path}")
            return str(path)
        dirs = get_setting_field("FOUNDRY_CHECKPOINT_DIRS", "") or ""
        for directory in [*dirs.split(":"), str(Path.home() / ".foundry/checkpoints")]:
            if directory:
                path = (
                    Path(directory).expanduser()
                    / "rf3_foundry_01_24_latest_remapped.ckpt"
                )
                if path.is_file():
                    return str(path.resolve())
        raise ValueError(
            "Install RF3 weights with 'foundry install rf3', then set rf3_checkpoint or FOUNDRY_CHECKPOINT_DIRS"
        )

    def create_command(self, handler, sandbox_dir, chunk_id):
        from chaperone.io import write_pdb

        root = Path(sandbox_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        def stage_auxiliary(path, subdir):
            source = Path(self.stage_file(path, root, subdir=subdir)).resolve()
            destination = root / subdir / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination:
                shutil.copy2(source, destination)
            return str(destination)

        checkpoint = self._checkpoint()
        configs, manifest = [], []
        for index, data in enumerate(handler.get_data(which="last", load=True)):
            atom_array = data.atom_array
            if "A" not in set(atom_array.chain_id):
                raise ValueError(
                    "RFOptimization designs chain A; rename the designed protein chain explicitly before running"
                )
            if self.backprop_fraction > 0 and "B" not in set(atom_array.chain_id):
                raise ValueError(
                    "The RF3 cycling preset requires interaction partner chain B"
                )
            if data.metadata.get("motif"):
                raise ValueError(
                    "RFO does not preserve Chaperone motif masks; remove the motif constraint only if redesign is intended"
                )
            # The RF3 optimizer identifies its binder by first-chain position,
            # while the cycling/MPNN code uses literal chain A. Reconcile both
            # contracts without mutating the input or renaming partners.
            atom_array = atom_array[
                np.concatenate(
                    (
                        np.flatnonzero(atom_array.chain_id == "A"),
                        np.flatnonzero(atom_array.chain_id != "A"),
                    )
                )
            ]
            job = root / f"design_{index}"
            job.mkdir(exist_ok=True)
            input_path = job / "start.pdb"
            write_pdb(atom_array, input_path)
            assets = {
                "mpnn_repo": "/opt/ligandmpnn",
                "mpnn_python": "/opt/mpnn-env/bin/python",
                "mpnn_model_type": self.mpnn_model_type,
                "mpnn_sif": "",
                "rf3_python": "/opt/rfo-env/bin/python",
                "rf3_sif": "",
                "boltz_executable": "/opt/boltz-env/bin/boltz",
                "boltz_sif": "",
                "boltz_cache": str(Path(self.boltz_cache).expanduser().resolve()),
            }
            if checkpoint:
                assets["rf3_ckpt"] = checkpoint
            config = {
                "input_structure": str(input_path),
                "out_dir": str(job / "output"),
                "total_cycles": self.total_cycles,
                "backprop_fraction": self.backprop_fraction,
                "mpnn_num_seqs": 1,
                "mpnn_temperature": self.mpnn_temperature,
                "seed": self.seed + int(chunk_id) * 1000003 + index,
                "cyclic": self.cyclic,
                "assets": assets,
            }
            for key in ("msa_paths", "templates"):
                config[key] = {
                    chain: stage_auxiliary(path, f"{key}/{index}")
                    for index, (chain, path) in enumerate(getattr(self, key).items())
                }
            if self.rf3_config:
                config["rf3_config"] = stage_auxiliary(self.rf3_config, "optimizer")
            config_path = job / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            configs.append(str(config_path))
            config_id = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:12]
            manifest.append(
                {
                    "job": str(job),
                    "path_handle": data.path,
                    "parent_design_id": data.design_id or data.basename,
                    "design_id": f"{data.design_id or data.basename}_{self.name}_{config['seed']}_{config_id}",
                }
            )
        if not manifest:
            raise ValueError("rfoptimization requires at least one input complex")
        (root / "rfo_inputs.json").write_text(json.dumps(manifest, indent=2))
        (root / "rfo_configs.json").write_text(json.dumps(configs))
        binds = []
        for path in [checkpoint, assets["boltz_cache"]]:
            if path:
                path = Path(path)
                if path == Path(assets["boltz_cache"]):
                    path.mkdir(parents=True, exist_ok=True)
                binds.append((str(path), str(path)))
        command = self._script_spec.image.build_command(
            *self._script_spec.command, binds=binds
        )
        return command + [str(root / "rfo_configs.json")]

    def compile(self, sandbox_dir):
        from chaperone.io import get_sequences
        from biotite.structure import get_residue_starts

        root = Path(sandbox_dir)
        rows = []
        for item in json.loads((root / "rfo_inputs.json").read_text()):
            output = Path(item["job"]) / "output"
            records_path = output / "start_records.json"
            if not records_path.is_file():
                raise RuntimeError(f"RFO records missing: {records_path}")
            records = json.loads(records_path.read_text())
            final = output / "inputs" / f"start_{self.total_cycles}.pdb"
            if (
                len(records) != self.total_cycles
                or not final.is_file()
                or not final.stat().st_size
            ):
                raise RuntimeError(
                    f"RFO did not complete {self.total_cycles} cycles in {output}"
                )
            if any(
                record.get("cycle") != index
                or any(
                    value == "Failed"
                    for key, value in record.items()
                    if key.endswith("_Status")
                )
                for index, record in enumerate(records)
            ):
                raise RuntimeError(
                    f"RFO records contain incomplete or failed cycles in {output}"
                )
            atoms = PDBFile.read(final).get_structure(model=1)
            if (
                not len(atoms)
                or not np.isfinite(atoms.coord).all()
                or "A" not in atoms.chain_id
            ):
                raise RuntimeError(
                    f"RFO final complex has missing or invalid coordinates: {final}"
                )
            original = PDBFile.read(Path(item["job"]) / "start.pdb").get_structure(
                model=1
            )
            original_sequences = get_sequences(original)
            final_sequences = get_sequences(atoms)
            if set(final_sequences) != set(original_sequences) or any(
                final_sequences.get(chain) != sequence
                for chain, sequence in original_sequences.items()
                if chain != "A"
            ):
                raise RuntimeError(
                    f"RFO changed or lost an interaction partner in {final}"
                )
            original_residues = original[get_residue_starts(original)]
            final_residues = atoms[get_residue_starts(atoms)]
            for chain in original_sequences:
                if chain != "A" and list(
                    original_residues.res_name[original_residues.chain_id == chain]
                ) != list(final_residues.res_name[final_residues.chain_id == chain]):
                    raise RuntimeError(
                        f"RFO changed partner residue identities in chain {chain}: {final}"
                    )
            if len(final_sequences.get("A") or "") != len(
                original_sequences.get("A") or ""
            ):
                raise RuntimeError(f"RFO changed the designed chain length in {final}")
            # Upstream always writes start_N.pdb. Unique published basenames
            # are required by downstream tools that key jobs by path stem.
            published = (
                root
                / "outputs"
                / (
                    "rfo_"
                    + hashlib.sha256(item["design_id"].encode()).hexdigest()[:16]
                    + ".pdb"
                )
            )
            published.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final, published)
            rows.append(
                make_row(
                    name=self.name,
                    path=published,
                    path_handle=item["path_handle"],
                    design_id=item["design_id"],
                    parent_design_id=item["parent_design_id"],
                    metrics={
                        self.name: {
                            "completed_cycles": len(records),
                            "records_path": str(records_path),
                        }
                    },
                )
            )
        if not rows:
            raise RuntimeError("RFO produced no completed designs")
        return PipelineDataHandler(rows)
