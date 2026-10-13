"""Run RF3 with compatibility handling for its runtime and trajectories.

Some Foundry RF3 releases route ``X_*_L_traj`` tensors through the ordinary
AtomArray writer, which fails after prediction with ``Tensor has no copy``.
RF3 already ships the correct trajectory converter; this entrypoint redirects
only tensor trajectory writes through it and otherwise delegates to ``rf3``.

The installed RF3 entrypoint also loads a baked ``.env`` with override enabled.
Restore pipeline-configured MSA roots after importing that entrypoint.
"""

from __future__ import annotations

from typing import Any
import os

import torch
from rf3.inference_engines import rf3 as rf3_engine
from rf3.utils import io as rf3_io


_ORIGINAL_BUILD_STACK = rf3_engine.build_stack_from_atom_array_and_batched_coords
_ORIGINAL_DUMP_STRUCTURES = rf3_engine.dump_structures
_LAST_ATOM_ARRAY: Any = None


def _capture_atom_array(*args, **kwargs):
    """Remember the input AtomArray used to build the prediction stack."""
    global _LAST_ATOM_ARRAY
    if "atom_array" in kwargs:
        _LAST_ATOM_ARRAY = kwargs["atom_array"]
    elif len(args) >= 2:
        _LAST_ATOM_ARRAY = args[1]
    return _ORIGINAL_BUILD_STACK(*args, **kwargs)


def _dump_structures_or_trajectory(*args, **kwargs):
    """Use RF3's converter and leave the final prediction as the last frame."""
    atom_arrays = kwargs.get("atom_arrays", args[0] if args else None)
    is_tensor_trajectory = (
        isinstance(atom_arrays, (list, tuple))
        and bool(atom_arrays)
        and torch.is_tensor(atom_arrays[0])
    )
    if not is_tensor_trajectory:
        return _ORIGINAL_DUMP_STRUCTURES(*args, **kwargs)

    if _LAST_ATOM_ARRAY is None:
        raise RuntimeError("RF3 trajectory write was requested before input capture")

    base_path = kwargs.get("base_path", args[1] if len(args) >= 2 else None)
    trajectory_list = list(atom_arrays)

    # RF3's helper aligns every frame to the final prediction, then reverses
    # the stack so that prediction is model 1. Preserve the useful alignment,
    # but pre-reverse the aligned list so the helper emits diffusion order:
    # noisy/start first and the final prediction last.
    w_L = torch.ones(*trajectory_list[0].shape[:2]).to(trajectory_list[0].device)
    X_exists_L = torch.ones(
        trajectory_list[0].shape[1],
        dtype=torch.bool,
        device=trajectory_list[0].device,
    )
    for step in range(len(trajectory_list) - 1):
        trajectory_list[step] = rf3_io.weighted_rigid_align(
            X_L=trajectory_list[-1],
            X_gt_L=trajectory_list[step],
            X_exists_L=X_exists_L,
            w_L=w_L,
        )

    return rf3_io.dump_trajectories(
        trajectory_list=trajectory_list[::-1],
        atom_array=_LAST_ATOM_ARRAY,
        base_path=base_path,
        align_structures=False,
        file_type=kwargs.get("file_type", "cif.gz"),
    )


def main() -> None:
    """Patch the two RF3 module symbols and invoke the standard Typer app."""
    rf3_engine.build_stack_from_atom_array_and_batched_coords = _capture_atom_array
    rf3_engine.dump_structures = _dump_structures_or_trajectory

    import dotenv

    original_load_dotenv = dotenv.load_dotenv

    def load_dotenv_without_override(*args, **kwargs):
        kwargs["override"] = False
        return original_load_dotenv(*args, **kwargs)

    dotenv.load_dotenv = load_dotenv_without_override

    from rf3.cli import app

    msa_dirs = os.environ.get("CHAPERONE_LOCAL_MSA_DIRS")
    if msa_dirs:
        os.environ["LOCAL_MSA_DIRS"] = msa_dirs
        os.environ["PROTEIN_MSA_DIRS"] = msa_dirs

    app()


if __name__ == "__main__":
    main()
