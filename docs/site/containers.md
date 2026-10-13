# Installing block containers

Chaperone itself is MIT licensed. Each scientific tool, model weight, database,
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

Portable variants remain available even when a local IPD SIF is selected:
`colabfold_portable`, `rfdiffusion2_portable`, `modelhub_portable`, `foundry`,
and `foundry_public`. For a deliberate rebuild, name that variant and use
`--force`. A plain block name can select an already supplied site image.

## Recipe audit

The release checks render all **21** source recipes for Docker and Apptainer,
check shell syntax and local copy inputs, and exercise installer staging and
site selection. They do **not** build all images or run GPU inference. Validate
one small real workload on your cluster before scaling up.

| Recipes | Installation notes |
|---|---|
| `foundry`, `foundry_public` | Public source/registry paths for RF3, RFD3, RFD3NA and FoundryMPNN. Configure `FOUNDRY_CHECKPOINT_DIRS` for predownloaded weights; otherwise registry names are used. FoundryMPNN equivalence to legacy MPNN is not established here. |
| `modelhub_portable` | Legacy `mpnn` depends on private `baker-laboratory/fused_mpnn`. Authorized repository access is required. The installer reports this restriction; it does not substitute a different model. |
| `alphafold3` | Source recipe available; model parameters require the upstream access process. |
| `esmfold2` | Uses released `esm==3.4.1.post1` and Torch 2.11/CUDA 13. The removed Transformers fork was unavailable. Check GPU/driver support and upstream model terms. Existing older IPD images keep their legacy import path. |
| `rfdiffusion1`, `rfdiffusion2_portable` | HTTPS source recipes; install the corresponding checkpoints and provide their configured weight paths. |
| `colabfold_portable`, `simplefold` | Public source recipes with model downloads/caches configured separately. SimpleFold's CUDA apt repository matches its Ubuntu base. |
| `bindcraft`, `boltz2`, `boltzgen`, `chai1`, `disco`, `e1`, `esm2`, `metal3d`, `openfold3`, `protein_hunter`, `protenix`, `tmol` | Source recipes available. Consult the tool links in the [block reference](blocks-reference.md) for checkpoints, databases, model licenses, hardware requirements, and task-specific settings. |

Docker builds use an isolated build context containing the recipe's declared
scripts and source checkouts. Apptainer builds use the same declared inputs.
Missing institutional default mounts are omitted for portable recipes; explicit
input mounts and mounts on supplied existing images remain required.

Some CPU blocks use host tools rather than a container. For example,
`ligand_ddg` needs Rosetta and, for its bundled Holes metric, DAlphaBall: set
`ROSETTA_BIN` and `DALPHABALL_BIN`. Its Rosetta protocol, flags, and bundled ligand
parameter files are included in the wheel and source distribution.
