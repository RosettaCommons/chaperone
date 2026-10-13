# Installing block containers

Chaperone-authored code is MIT licensed. See the [third-party notices](https://github.com/RosettaCommons/chaperone/blob/main/THIRD_PARTY_NOTICES.md) for bundled AF3 code and example outputs. Each scientific tool, model weight, database,
and external executable retains its own license and access requirements.
A container recipe being present is not proof that its weights are public or
that the model has been scientifically validated on your data.

## Choose your site

Without a site file, Chaperone uses portable defaults and searches your
`~/.chaperone/containers` directory. Set `CHAPERONE_CONTAINER_DIRS` to a
colon-separated list to use another location. Supply an untracked `.env.ipd`
in the checkout root for existing IPD installations, or export `CHAPERONE_SITE=digs`
before running Python. A wheel installation also reads `.env.ipd` and `.env.local`
from the working directory. `CHAPERONE_ENV_FILE` can select an external dotenv file.

```bash
chaperone list-blocks
chaperone install                 # list available recipes and restrictions
chaperone install rfd3 --dry_run   # inspect; does not clone, build, or write files
chaperone install rfd3            # builds the selected block's image
```

Choose `CHAPERONE_CONTAINER_BACKEND=apptainer` or `docker` in `.env.local`.
Build on a Linux machine compatible with your compute nodes, with enough local
scratch, internet access to the sources, and a suitable container builder.
Configure Slurm queues, account and GPU mapping separately in the
[syntax guide](syntax.md#slurm-site-configuration).

Cloud (Modal) execution is not supported; run with `local` or `slurm` using Docker or Apptainer.

Portable variants remain available even when a local IPD SIF is selected:
`colabfold_portable`, `rfdiffusion2_portable`, `foundry`, and `foundry_public`. For a deliberate rebuild, name that variant and use
`--force`. A plain block name can select an already supplied site image.

## Recipe audit

The release checks render all **22** source recipes for Docker and Apptainer,
check shell syntax and local copy inputs, and exercise installer staging and
site selection. That audit is static; the separate source builds and GPU runs
are recorded in [release validation](release-validation.md). Not all images have
been built or exercised. Validate one small workload on your cluster before scaling up.

| Recipes | Installation notes |
|---|---|
| `foundry`, `foundry_public` | Public source/registry paths for RF3, RFD3, RFD3NA and FoundryMPNN. Configure `FOUNDRY_CHECKPOINT_DIRS` for predownloaded weights; otherwise registry names are used. FoundryMPNN equivalence to the `mpnn` block is not established here. |
| `ligandmpnn_public` | Public MIT-licensed `dauparas/LigandMPNN` source pinned to `26ec57ac976ade5379920dbd43c7f97a91cf82de`, with all fifteen official checkpoints (every `model_type` and noise level, plus the side-chain packer) baked in and verified by SHA-256 at build time. The backend of the `mpnn` block; see the [validation record](release-validation.md) for completed build/GPU checks. |
| `alphafold3` | Source recipe available; model parameters require the upstream access process. |
| `esmfold2` | Uses released `esm==3.4.1.post1` and Torch 2.11/CUDA 13. The removed Transformers fork was unavailable. Check GPU/driver support and upstream model terms. Existing older IPD images keep their legacy import path. |
| `chai1` | Blackwell recipe explicitly overrides `chai_lab==0.6.1`'s `torch<2.7` requirement with Torch 2.7.1/CUDA 12.8. Other dependencies resolve normally; the build's dependency check allows only this exact mismatch. This is not a clean `pip check` stack. See the [validation record](release-validation.md) for actual inference status. |
| `rfdiffusion1` | HTTPS source recipe; install the checkpoints and provide the configured weight path. |
| `rfdiffusion2_portable` | HTTPS source recipe. The `RFD_173.pt` checkpoint is downloaded during the build and checked against a pinned SHA-256; pass `ckpt_path` to use another file. |
| `colabfold_portable`, `simplefold` | Public source recipes with model downloads/caches configured separately. SimpleFold's CUDA apt repository matches its Ubuntu base. |
| `bindcraft`, `boltz2`, `boltzgen`, `disco`, `e1`, `esm2`, `metal3d`, `openfold3`, `protein_hunter`, `protenix`, `tmol` | Source recipes available. Consult the tool links in the [block reference](blocks-reference.md) for checkpoints, databases, model licenses, hardware requirements, and task-specific settings. |

Docker builds use an isolated build context containing the recipe's declared
scripts and source checkouts. Apptainer builds use the same declared inputs.
Missing institutional default mounts are omitted for portable recipes; explicit
input mounts and mounts on supplied existing images remain required.

Build on a host that permits Apptainer's build namespaces. A cluster login node
can permit Slurm submission while rejecting `%post` with `Failed to set mount
propagation: Permission denied`. On the tested CoreWeave cluster, the same
`chaperone install` command succeeded past this stage inside a compute allocation.
Use your site's approved build host or compute allocation and a shared
`CHAPERONE_CONTAINER_DIRS` directory. Changing GPU model names or adding CPU
resources does not repair a login-node namespace restriction.

Runs mount their active run directory and the working directory where the engine
was created. Launch from the project root, or set
`CHAPERONE_CONTAINER_WORKSPACE=/absolute/project/root` when source files and
inputs live elsewhere. Place referenced inputs beneath that root or configure
an explicit image bind for their location. Evaluation workers use their assigned
workspace instead of inferring a host project directory.

The `mpnn` block runs public LigandMPNN from the `ligandmpnn_public` image;
install it with `chaperone install ligandmpnn_public`. The image bakes every
official checkpoint, so no weights directory is configured. `model_type` selects
the model and `noise_level` the checkpoint version (the backbone noise, in
Angstrom, the model was trained with); leaving `noise_level` unset uses the
upstream default for the model.

| `model_type` | `noise_level` | Default | Checkpoints |
|---|---|---|---|
| `protein_mpnn` | 0.02, 0.1, 0.2, 0.3 | 0.2 | `proteinmpnn_v_48_{002,010,020,030}.pt` |
| `ligand_mpnn` (default) | 0.05, 0.1, 0.2, 0.3 | 0.1 | `ligandmpnn_v_32_{005,010,020,030}_25.pt` |
| `soluble_mpnn` | 0.02, 0.1, 0.2, 0.3 | 0.2 | `solublempnn_v_48_{002,010,020,030}.pt` |
| `per_residue_label_membrane_mpnn` | 0.2 | 0.2 | `per_residue_label_membrane_mpnn_v_48_020.pt` |
| `global_label_membrane_mpnn` | 0.2 | 0.2 | `global_label_membrane_mpnn_v_48_020.pt` |

Requesting a combination that is not in this table is an error that lists the
valid ones. The membrane models take `args.transmembrane_buried` and
`args.transmembrane_interface` (per-residue) or `args.global_transmembrane_label`
(global); those options are rejected for any other model. Side-chain packing
(`output_mode="structures"`) uses `ligandmpnn_sc_v_32_002_16.pt`, or a host file
given as `args.checkpoint_path_sc`.

The backend stages PDB inputs for its pinned runner. It keeps the original CIF and parent links, remaps
fixed residues with chain names, and restores original chain identifiers in
compiled sequences or structures. Inputs must fit PDB's residue and component
identifier limits; coordinates use PDB precision. Residue-level bias requires
single-character input chain IDs; different per-input chain constraint mappings
require separate chunks. Integration checks do not establish scientific
equivalence between public LigandMPNN and FoundryMPNN. See the
[release validation record](release-validation.md) for actual build/GPU status.

Some CPU blocks use host tools rather than a container. For example,
`ligand_ddg` needs Rosetta and, for its bundled Holes metric, DAlphaBall: set
`ROSETTA_BIN` and `DALPHABALL_BIN`. Its Rosetta protocol, flags, and bundled ligand
parameter files are included in the wheel and source distribution.
