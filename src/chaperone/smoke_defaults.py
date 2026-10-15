"""Small sampling counts for CLI smoke runs, shipped with installed Chaperone.

These overrides bound samples per input, not the number or size of inputs.
They do not relax scientific acceptance criteria or replace model checkpoints.
Explicit --config entries and block flags can override these values.
"""

from copy import deepcopy


_SMOKE_DEFAULTS = {
    "rfd1": {"num_designs": 1},
    "rpxdock": {"nout_top": 1},
    "rfd2": {"num_designs": 1},
    "rfd3": {"n_batches": 1, "batch_size": 1},
    "rfd3_partial": {"n_batches": 1, "batch_size": 1},
    "rfd3na": {"n_batches": 1, "batch_size": 1},
    "mpnn": {"num_seq_per_pdb": 1},
    "foundry_mpnn": {"num_seq_per_pdb": 1},
    "disco": {"num_seeds": 1, "n_batches": 1, "effort": "fast"},
    "boltzgen": {"num_designs": 1, "n_batches": 1, "diffusion_batch_size": 1},
    "bindcraft": {"number_of_final_designs": 1, "max_trajectories": 1},
    "protein_hunter": {"num_designs": 1, "num_cycles": 1},
    "af3": {"diffusion_batch_size": 1, "model_seeds": [42]},
    "af3_with_target_template": {"diffusion_batch_size": 1, "model_seeds": [42]},
    "rf3": {"diffusion_batch_size": 1, "model_seeds": [42]},
    "boltz2": {"diffusion_batch_size": 1},
    "boltz2_steered": {"diffusion_batch_size": 1},
    "chai1": {"diffusion_batch_size": 1},
    "protenix": {"diffusion_batch_size": 1, "seeds": "42"},
    "openfold3": {"diffusion_batch_size": 1, "num_model_seeds": 1},
    "esmfold2": {"diffusion_batch_size": 1},
    "simplefold": {"diffusion_batch_size": 1},
    "colabfold": {"num_models": 1, "num_seeds": 1},
}


def smoke_defaults() -> dict[str, dict]:
    """Return a fresh copy so config merging cannot mutate future runs."""
    return deepcopy(_SMOKE_DEFAULTS)
