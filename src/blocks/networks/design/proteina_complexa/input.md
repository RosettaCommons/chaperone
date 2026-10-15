# Proteina-Complexa protein binder generation

`proteina_complexa` jointly generates a binder sequence and all-atom structure,
conditioned on a protein target PDB and interface hotspots. Returned PDBs contain
both target and binder. The target is centered and its chains/residues are
renumbered by upstream; target residues are recorded as `motif` metadata so a
subsequent MPNN step can keep them fixed.

This block runs the upstream **single-pass generator**. Refolding and binder
quality evaluation are separate Chaperone steps. It does not run the upstream
AF2 reward search, hallucination refinement, ligand model, or AME model.

## Installation and checkpoints

```bash
chaperone install proteina_complexa
mkdir -p "$HOME/.cache/proteina-complexa"
```

Download the two public protein-target checkpoints from NVIDIA NGC into that
directory. No NVIDIA-internal paths or credentials are needed:

```bash
curl --fail --location --retry 3 \
  'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa/1.0/files?redirect=true&path=complexa.ckpt' \
  -o "$HOME/.cache/proteina-complexa/complexa.ckpt"
curl --fail --location --retry 3 \
  'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa/1.0/files?redirect=true&path=complexa_ae.ckpt' \
  -o "$HOME/.cache/proteina-complexa/complexa_ae.ckpt"
```

The files total approximately 6.6 GiB. Expected SHA256 checksums:

| File | SHA256 |
| --- | --- |
| `complexa.ckpt` | `589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9` |
| `complexa_ae.ckpt` | `35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816` |

Model weights retain the [NVIDIA Open Model License](https://huggingface.co/nvidia/NV-Proteina-Complexa-Protein-Target-160M-v1).
The container pins [upstream source](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa/tree/acf55a89a9a7206d5ce9abc0bf2246c3b3829f92),
PyTorch 2.7/CUDA 12.6, AtomWorks 2.2.1 and Biotite 1.4.0. This **separate
protein-generation environment** preserves upstream's `AtomSelectionStack`
API, which AtomWorks 3 removed; it does not alter the Foundry environment.
The recipe corrects upstream's conflicting SciPy package constraints (SciPy 1.12 to 1.13.1; Einops 0.6 to 0.8) and checks installed dependencies. It lazily imports optional AF2/JAX hallucination code so single-pass generation
does not require those unrelated runtimes. Sampling code is unchanged.

## Inputs

| Field | Default | Meaning |
| --- | --- | --- |
| `target_pdb` | required | Existing protein PDB, copied into the job sandbox |
| `target_selection` | required | Inclusive, comma-separated ranges such as `A1-115` or `A1-40,B10-30` |
| `hotspots` | required | Nonempty list of original PDB residue IDs, e.g. `["A37", "A39"]` |
| `checkpoint_dir` | required | Host directory containing both checkpoints, mounted explicitly |
| `binder_length` | `(80, 120)` | Inclusive minimum/maximum generated binder length |
| `num_designs` | `1` | Number of structures per engine batch |
| `n_batches` | `1` | Independent batches; seed increments by chunk ID |
| `batch_size` | `1` | Samples evaluated together on one GPU |
| `num_steps` | `400` | Upstream flow integration step count |
| `seed` | `0` | Base random seed |

Hotspots must exist inside the selected target crop. Empty crops and absent
hotspots fail before GPU submission. Only ordinary protein targets with
single-letter PDB chain IDs and integer residue ranges are supported here.
For a single residue, use an explicit range such as `A37-37`.

## PDL1 example

Use the official
[PD-L1 target PDB](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa/blob/acf55a89a9a7206d5ce9abc0bf2246c3b3829f92/assets/target_data/bindcraft_targets/PD-L1.pdb).
The crop and hotspots below match upstream's `02_PDL1` target:

```python
from pathlib import Path
from chaperone import ExecutionEngine
from blocks import Blocks

engine = ExecutionEngine(rundir="runs/complexa-pdl1")
designs = engine.run(
    Blocks["proteina_complexa"](
        target_pdb="inputs/PD-L1.pdb",
        target_selection="A1-115",
        hotspots=["A37", "A39", "A49", "A98"],
        checkpoint_dir=str(Path.home() / ".cache/proteina-complexa"),
        binder_length=(80, 80),
        num_designs=1,
        num_steps=400,
    ),
)
```

Inspect the generated structure and run a downstream complex refolding block
before interpreting any design as a promising binder. Shortening `num_steps`
can test execution, but is not a quality-equivalent production setting.

For more diverse lengths use `binder_length=(64, 155)`, the upstream PDL1 range.
For independent replicas use `n_batches=4`; this generates four designs with
seeds 0–3 at the defaults. Increase `batch_size` only within GPU memory limits.

## Output and limitations

- The generated PDB retains all-atom target and binder coordinates. The last
  output chain is the binder; preceding chains are the selected target.
- Compilation rejects missing/extra structures, non-finite coordinates, missing
  chains, unexpected binder lengths, and changes to the target sequence/crop or CA geometry.
- Published structure basenames and design IDs include each chunk seed, so
  independent batches remain distinct in downstream tools. Native outputs and
  their CSV references are retained.
- Output metadata includes `binder_chain`, `target_chains`, fixed-target `motif`,
  source selection/hotspots, seed, and upstream revision.
- Original target coordinates are translated by upstream target centering.
  Compare after aligning target atoms; do not assume the original coordinate frame.
- Keep `rows_per_job=1`; combining independent batches into one engine chunk
  is rejected rather than silently generating fewer designs.
- Checkpoint files must be present before submission. Container installation
  installs software; it does not download these large model weights.
- This block requires an NVIDIA CUDA GPU. MPS/CPU inference is unsupported.
- A generated complex is not evidence of affinity, experimental binding,
  refolding agreement, or superiority over another design method.
