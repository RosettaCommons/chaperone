"""Public, pinned RPXdock I3 docking with explicit model/score provenance."""

import json
from pathlib import Path
import shlex
from typing import ClassVar

import biotite.structure as struc
import numpy as np
from pydantic import Field

from chaperone import CompiledBlockFunction, JobArgs, PipelineDataHandler
from chaperone.containers import Image
from chaperone.data import make_row
from chaperone.io import parse_to_atom_array

RPX_REVISION = "61264c6cbaae882bf52d6547cd2d40b6ed4de0d9"
WILLUTIL_REVISION = "ded19f6ca8048e33383b87a3517527550b9503cc"
RUNNER = Path(__file__).resolve().parents[1] / "scripts/rpxdock_run.py"
# Keep the model dimension for one pose and accept explicit NumPy symmetry frames.
_RPX_COMPATIBILITY_PATCH = """from pathlib import Path
p = Path('/opt/rpxdock/rpxdock/search/onecomp.py')
s = p.read_text()
old = "np.sum(xforms[:, :3, 3] * spec.axis[None, :3], axis=-1).squeeze()"
new = "np.sum(xforms[:, :3, 3] * spec.axis[None, :3], axis=-1).reshape(-1)"
if s.count(old) != 1:
    raise RuntimeError('Unexpected RPXdock onecomp source; recheck the single-pose patch')
p.write_text(s.replace(old, new))
p = Path('/opt/rpxdock/rpxdock/search/result.py')
s = p.read_text()
old = "elif symframes == 'primary':"
new = "elif isinstance(symframes, str) and symframes == 'primary':"
if s.count(old) != 1:
    raise RuntimeError('Unexpected RPXdock result source; recheck the symmetry-frame patch')
p.write_text(s.replace(old, new))
"""
RPXDOCK_CONTAINER = (
    Image("rpxdock")
    .base_image("ubuntu:24.04")
    .packages(
        "python3",
        "python3-dev",
        "python3-venv",
        "git",
        "build-essential",
        "cmake",
        "ca-certificates",
    )
    .run(
        "python3 -m venv /opt/rpx-env",
        "/opt/rpx-env/bin/pip install --upgrade 'pip>=26.2.2'",
        f"git clone https://github.com/willsheffler/willutil.git /opt/willutil && cd /opt/willutil && git checkout {WILLUTIL_REVISION}",
        f"git clone https://github.com/willsheffler/rpxdock.git /opt/rpxdock && cd /opt/rpxdock && git checkout {RPX_REVISION}",
        "/opt/rpx-env/bin/pip install 'numpy==1.26.4' 'mdtraj==1.11.0' -e /opt/willutil",
        "/opt/rpx-env/bin/pip install cffi tabulate sphinx breathe exhale sphinx-rtd-theme",
        "/opt/rpx-env/bin/pip install --no-deps -e /opt/rpxdock",
        "/opt/rpx-env/bin/python -c " + shlex.quote(_RPX_COMPATIBILITY_PATCH),
        "PYTHONPATH=/opt/rpxdock:/opt/willutil /opt/rpx-env/bin/python -c 'import rpxdock'",
    )
    .copy(str(RUNNER), "/opt/run_rpxdock.py")
    .env(PATH="/opt/rpx-env/bin:$PATH", PYTHONPATH="/opt/rpxdock:/opt/willutil")
)


class RPXDockBlock(CompiledBlockFunction):
    _name = "rpxdock"
    _description = (
        "Dock a z-oriented C3 asymmetric unit into an I3 cage and emit C3 trimers"
    )
    _image: ClassVar = RPXDOCK_CONTAINER
    _resources = JobArgs(
        partition="cpu",
        mem_gb=8,
        cpus_per_task=1,
        rows_per_job=1,
        time_limit="02:00:00",
    )
    nout_top: int = Field(
        1, ge=1, description="Number of highest scoring C3 trimers per input"
    )
    beam_size: int = Field(
        10000, ge=1, description="RPXdock hierarchical search beam size"
    )

    ignored_aas: str = Field(
        "CGP",
        description="Residue types excluded from RPX scoring; use an empty string for glycine-placeholder backbones.",
    )
    cache_dir: str = Field(
        default_factory=lambda: str(Path.home() / ".cache" / "chaperone" / "rpxdock"),
        description="Writable home for decompressed RPXdock scoring and symmetry caches.",
    )

    def get_command_template(self):
        self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())
        return self._image.build_command(
            # Apptainer reserves HOME during startup; set it for the tool
            # process after entering the container, before importing willutil.
            "env",
            f"HOME={self.cache_dir}",
            "python",
            "/opt/run_rpxdock.py",
            exec_mode="exec",
            binds=[(self.cache_dir, self.cache_dir)],
            env={"TMPDIR": str(Path(self.cache_dir) / "tmp")},
        )

    def create_command(self, handler, sandbox_dir, chunk_id):
        (Path(self.cache_dir) / "tmp").mkdir(parents=True, exist_ok=True)
        root = Path(sandbox_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        commands, inputs = [], []
        for index, item in enumerate(handler.get_data(which="last", load=True)):
            staged = root / "inputs" / f"input_{index}.pdb"
            staged.parent.mkdir(exist_ok=True)
            from chaperone.io import write_pdb

            write_pdb(item.atom_array, staged)
            ca = item.atom_array[item.atom_array.atom_name == "CA"]
            motif = set(item.metadata.get("motif") or [])
            ordinals = [
                i
                for i, (chain, res) in enumerate(zip(ca.chain_id, ca.res_id))
                if f"{chain}{res}" in motif
            ]
            if len(struc.get_chains(item.atom_array)) != 1 or not ordinals:
                raise ValueError(
                    "RPXdock needs a prepared monomer with fixed motif metadata"
                )
            output = root / f"dock_{index}"
            commands.append(
                shlex.join(
                    self.command_template
                    + [
                        "--input",
                        str(staged),
                        "--output-dir",
                        str(output),
                        "--nout",
                        str(self.nout_top),
                        "--beam-size",
                        str(self.beam_size),
                        "--ignored-aas",
                        self.ignored_aas,
                    ]
                )
            )
            inputs.append(
                {
                    "path": item.path,
                    "design_id": item.design_id,
                    "output": str(output),
                    "motif_ordinals": ordinals,
                    "residue_count": len(ca),
                }
            )
        if not inputs:
            raise ValueError("RPXdock received no inputs")
        (root / "inputs.json").write_text(json.dumps(inputs, indent=2))
        return ["bash", "-c", " && ".join(commands)]

    def compile(self, sandbox_dir):
        rows = []
        for source_index, source in enumerate(
            json.loads((Path(sandbox_dir) / "inputs.json").read_text())
        ):
            poses = json.loads((Path(source["output"]) / "poses.json").read_text())
            if not poses:
                raise ValueError("RPXdock returned no scored poses")
            for pose in poses:
                if not np.isfinite(pose["score"]):
                    raise ValueError("RPXdock returned a non-finite score")
                atoms = parse_to_atom_array(pose["path"])
                chains = list(struc.get_chains(atoms))
                if len(chains) != 3:
                    raise ValueError(
                        "RPXdock must return a C3 trimer, not a full cage or monomer"
                    )
                motif = []
                residue_ids = None
                for chain in chains:
                    ca = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
                    if len(ca) != source["residue_count"]:
                        raise ValueError("Docked subunit residue count changed")
                    if residue_ids is not None and not np.array_equal(
                        ca.res_id, residue_ids
                    ):
                        raise ValueError(
                            "C3 chains must share residue numbering for tied MPNN design"
                        )
                    residue_ids = ca.res_id
                    motif.extend(
                        f"{chain}{ca.res_id[i]}" for i in source["motif_ordinals"]
                    )
                rows.append(
                    make_row(
                        name=self.name,
                        path=pose["path"],
                        path_handle=source["path"],
                        design_id=f"{source['design_id']}_dock{source_index}_model{pose['model_index']}",
                        parent_design_id=source["design_id"],
                        metadata={
                            "motif": motif,
                            "diffused_index_map": "{}",
                            "motif_atom_contigs": {},
                        },
                        metrics={
                            self.name: {
                                "rpx_score": pose["score"],
                                "model_index": pose["model_index"],
                            }
                        },
                    )
                )
        return PipelineDataHandler(rows)
