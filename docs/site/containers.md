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

Cloud (Modal) execution is available for compiled blocks; see [cloud setup](cloud.md). Otherwise run with `local` or `slurm` using Docker or Apptainer.

Blocks default to buildable public-source recipes, including
`colabfold_portable`, `rfdiffusion2_portable`, `foundry`, and `foundry_public`.
They do not switch models or source checkouts when institutional files happen
to exist on the host. To use a prebuilt image, place it under its registered
recipe name in a directory listed in `CHAPERONE_CONTAINER_DIRS`. Use `--force`
for a deliberate rebuild.

For existing Digs installations, this also makes RFdiffusion2 default to the
public `RFD_173.pt` checkpoint rather than the formerly auto-selected private
`RFD_140_ORI_tipatom.pt`. Treat that as a model change when comparing runs;
configure `ckpt_path` explicitly when using another compatible checkpoint.

## Download Apptainer definitions

All registered, buildable scientific images have checked-in `.def` files in the
[Apptainer recipe directory](https://github.com/RosettaCommons/chaperone/tree/main/containers/apptainer).
The [download and build guide](https://github.com/RosettaCommons/chaperone/blob/main/containers/README.md)
links each definition and its preparation script; the
[catalog](https://github.com/RosettaCommons/chaperone/blob/main/containers/apptainer/catalog.json)
maps image names to block names and lists required source inputs. Several blocks
share one image: RFD3, RFD3NA, RF3 and FoundryMPNN use `foundry`; ProteinMPNN,
LigandMPNN and SolubleMPNN use `ligandmpnn`.

For a standalone build, clone or download the repository so the definitions,
preparation scripts and bundled `sources/` directory stay together. Run builds
from `containers/apptainer`, because `%files` paths are relative to that directory:

```bash
git clone https://github.com/RosettaCommons/chaperone.git
cd chaperone/containers/apptainer
apptainer build foundry.sif foundry.def
bash alphafold3.prepare.sh
apptainer build alphafold3.sif alphafold3.def
```

Run the matching `<image>.prepare.sh` first whenever the catalog lists one; it
fetches the declared upstream source checkouts into the build context. Bundled
local scripts are already under `sources/`. Prepare each source context once;
the scripts refuse existing checkout paths. To retry an image build, reuse its
prepared context without rerunning preparation. Downloading a single `.def` alone
does not supply its `%files` inputs. Builds need outbound network access, disk
space and a host that permits Apptainer build namespaces; use `--fakeroot` if
required by your site's supported build configuration. These downloads are
recipes and source inputs, not prebuilt `.sif` images or model weights.

To use these images in Chaperone, put the `.sif` files in a directory listed in
`CHAPERONE_CONTAINER_DIRS`, keeping their recipe names, for example `foundry.sif`
and `alphafold3.sif`. Alternatively, `chaperone install rfd3 af3` stages the same
declared inputs and builds the images for you. Installing Chaperone from a wheel
does not require downloading this source-tree recipe directory to use the installer.

## RFD3 and AF3 runtime setup

Set your paths in `.env.local`; use absolute shared paths when running on Slurm:

```dotenv
CHAPERONE_CONTAINER_BACKEND=apptainer
CHAPERONE_CONTAINER_DIRS=/absolute/shared/path/containers
ALPHAFOLD3_WEIGHTS_DIR=/absolute/shared/path/af3_weights
# Explicitly install Foundry checkpoints here before running its blocks
FOUNDRY_CHECKPOINT_DIRS=/absolute/shared/path/foundry_weights
```

AF3 model parameters require the upstream access process; the definition does
not grant access or include those parameters. Install Foundry checkpoints
explicitly before inference or warmup. Its pinned 1.0 runtime resolves registry
names to local checkpoints and fails if they are missing; it does not download
them on the first block call. For example, on a network-enabled staging host:

```bash
mkdir -p /absolute/shared/path/foundry_weights
apptainer exec --cleanenv \
  --bind /absolute/shared/path/foundry_weights:/checkpoints \
  /absolute/shared/path/containers/foundry.sif \
  foundry install rfd3 --checkpoint-dir /checkpoints
```

Use `foundry install base-models --checkpoint-dir /checkpoints` in the same
container command to stage the latest RFD3, RF3 and MPNN checkpoints together.
The [pinned Foundry installation guide](https://github.com/RosettaCommons/foundry/blob/54d188b3c0c27654a01c875b173148a508fcd858/README.md)
describes other model choices and access requirements. Reuse verified existing
checkpoints and make their host directory visible to the compute nodes through
`FOUNDRY_CHECKPOINT_DIRS`.

The AF3 block runs inference with its database-search pipeline
disabled: provide prepared MSAs when your workflow needs them. Building an AF3
image does not prepare an MSA database or run a search.

Image installation and execution are separate. Choose
`CHAPERONE_COMPUTE_BACKEND=local` on a GPU machine or `slurm` for a cluster, and
configure the cluster resources separately. Input files, images, weights and
caches must be visible on compute nodes. Check `chaperone list-blocks`, then run
one real small RFD3 workload to verify checkpoint loading, GPU support and mounts
before scaling. Container imports and CLI help alone do not verify inference.
RFD3's warmup command also exercises the real model; run it inside a GPU
allocation after checkpoint staging:

```bash
mkdir -p /absolute/shared/path/rfd3_compile_cache
chaperone warmup-rfd3 --local \
  --cache-dir /absolute/shared/path/rfd3_compile_cache \
  --rundir /absolute/shared/path/rfd3_smoke
```

## Recipe audit

The release checks render the source recipes for Docker and Apptainer,
check shell syntax and local copy inputs, and exercise installer staging and
site selection. These automated checks are static and CPU based.
Execution validation on 2026-10-18 built all 25 scientific images on Linux x86-64
with an NVIDIA H200 and exercised representative workloads: 23 passed
inference checks, while BindCraft and BindCraft2 completed trajectories with zero
accepted designs. This does not cover every host, modality or optional mode.
Validate one small workload on your cluster before scaling up.

| Recipes | Installation notes |
|---|---|
| `bindcraft2` | Pinned PacesaLab implementation with JAX 0.11/CUDA 12 and bundled ProteinMPNN. AlphaFold2 weights use a writable cache or explicit `af2_params_dir`. See the [binder guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/bindcraft2/input.md) for installation, supported inputs, and the upstream hosting-restricted license. |
| `foundry`, `foundry_public` | Foundry 1.0.0 at `54d188b3`, AtomWorks 3.0.1 and Biotite 1.6.0 for RF3, RFD3, RFD3NA and FoundryMPNN. Install the required checkpoints explicitly with the Foundry CLI before inference, then configure `FOUNDRY_CHECKPOINT_DIRS`; registry names resolve installed weights. FoundryMPNN equivalence to the `mpnn` block is not established here. |
| `rfoptimization` | Separate Foundry/Boltz2/LigandMPNN environments for iterative complex refinement, with the same Foundry 1.0.0 and AtomWorks 3.0.1 pins. See the [input and checkpoint guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/rfoptimization/input.md). |
| `proteina_complexa` | NVIDIA's protein binder generator with an isolated AtomWorks 2 environment. Download the two public model checkpoints and configure `checkpoint_dir`; see the [binder guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/proteina_complexa/input.md). |
| `ligandmpnn` | `dauparas/LigandMPNN` source pinned to `26ec57ac976ade5379920dbd43c7f97a91cf82de`, with all fifteen official checkpoints (every `model_type` and noise level, plus the side-chain packer) baked in and verified by SHA-256 at build time. The backend of the `mpnn` block. |
| `alphafold3` | Source recipe available; model parameters require the upstream access process. |
| `esmfold2` | Uses released `esm==3.4.1.post1` and Torch 2.11/CUDA 13. The removed Transformers fork was unavailable. Check GPU/driver support and upstream model terms. Existing older IPD images keep their legacy import path. |
| `chai1` | Blackwell recipe explicitly overrides `chai_lab==0.6.1`'s `torch<2.7` requirement with Torch 2.7.1/CUDA 12.8. Other dependencies resolve normally; the build's dependency check allows only this exact mismatch. This is not a clean `pip check` stack. |
| `rfdiffusion1` | HTTPS source recipe; install the checkpoints and provide the configured weight path. |
| `rfdiffusion2_portable` | HTTPS source recipe. The `RFD_173.pt` checkpoint is downloaded during the build and checked against a pinned SHA-256; pass `ckpt_path` to use another file. |
| `colabfold_portable`, `simplefold` | Public source recipes with model downloads/caches configured separately. SimpleFold's CUDA apt repository matches its Ubuntu base. |
| `bindcraft`, `boltz2`, `boltzgen`, `disco`, `e1`, `esm2`, `metal3d`, `openfold3`, `protein_hunter`, `protenix`, `tmol` | Source recipes available. Consult the tool links in the [block reference](blocks-reference.md) for checkpoints, databases, model licenses, hardware requirements, and task-specific settings. |

Docker builds use an isolated build context containing the recipe's declared
scripts and source checkouts. Apptainer builds use the same declared inputs.
Blocks mount configured model caches, inputs, and working directories. Explicit
input mounts and mounts on supplied existing images remain required.

### Upgrading an existing Foundry image

Updating Chaperone does not replace an already installed container. Rebuild it
to move the four shared Foundry blocks to AtomWorks 3:

```bash
chaperone install foundry --force
# CPU-only installations:
# chaperone install foundry_public --force
```

The build checks the installed AtomWorks and Biotite versions, imports all four
model engines, and exercises RNA/DNA input parsing. RFOptimization and
Proteina-Complexa have separate recipes and dependency pins.
Native upstream RF3 inference disables the private residue-feature cache inherited
from upstream training configuration, making that behavior independent of host
filesystem contents. The pinned source also fixes RFD3NA generated contigs and
RF3 trajectory serialization with AtomWorks 3. Chaperone no longer rewrites
Foundry source or substitutes a trajectory entrypoint. These native fixes are
tracked in [Foundry PR #406](https://github.com/RosettaCommons/foundry/pull/406). Rebuild existing images
with `chaperone install foundry --force` (or `foundry_public`); for RFO use
`chaperone install rfoptimization --force`.

AtomWorks 3 validates ligand atom names against the Chemical Component
Dictionary (CCD). Ligands with custom atom names need their corresponding
`chem_comp_atom` and `chem_comp_bond` definitions in the input CIF. An atom-site-only
file can be usable by classic LigandMPNN yet fail this validation in Foundry.
Supply the component definitions or use a structure with canonical CCD names;
renaming atoms without a verified chemical mapping can change the model input.

For RFD3-to-FoundryMPNN chaining, Chaperone removes RFD3's virtual `V0`–`V8`
sidechain slots from `UNK` peptide residues in a staged CIF copy. Backbone and
real ligand atoms, custom component definitions, and the original input remain
intact; this does not disable AtomWorks' chemical-name validation.

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

`mpnn` is the single LigandMPNN block. Install its image once with
`chaperone install ligandmpnn`; no settings are needed, because the image bakes
every official checkpoint. Within the block, `model_type` selects the model and
`noise_level` the checkpoint version (the backbone noise, in Angstrom, the model
was trained with); leaving `noise_level` unset uses the upstream default for the
model.

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

`mpnn` stages PDB inputs for its pinned runner. It keeps the original CIF and parent links, remaps
fixed residues with chain names, and restores original chain identifiers in
compiled sequences or structures. Inputs must fit PDB's residue and component
identifier limits; coordinates use PDB precision. Residue-level bias requires
single-character input chain IDs; different per-input chain constraint mappings
require separate chunks. Integration checks do not establish scientific
equivalence between LigandMPNN and FoundryMPNN.

### Host tools and external assets

The definitions cover scientific images, while other blocks execute in the host
Python environment or call separately installed tools. A complete recipe catalog
does not make every block runnable without these task-specific dependencies:

| Dependency | Host configuration |
|---|---|
| PyRosetta | Install separately and set `PYROSETTA_PATH` when needed. |
| Rosetta and DAlphaBall | Set `ROSETTA_BIN` and `DALPHABALL_BIN` for `ligand_ddg`, including its Holes metric. The Rosetta protocol, flags and bundled ligand parameters ship with Chaperone. |
| HBPLUS and Dunbrack data | Set `HBPLUS_COPY` and `DUNBRACK_LIBRARY_COPY` for blocks using those resources. |
| Foldseek and US-align | Install the binaries; configure `FOLDSEEK_EXEC` and `USALIGN_BIN` when they are not found on `PATH`. |
| DSSR and SNAP | Configure `X3DNA_BIN_DIR` and `X3DNA_DIR` for your separately installed 3DNA tools. |
| ViennaRNA | Install its Python `RNA` module for the blocks that use it. |
| PLIP and OpenBabel | Install the relevant Python/native dependencies for interaction profiling; Chaperone offers a `plip` extra. |
| CCD, model weights and databases | Supply the assets required by your selected blocks; image builds do not replace download, cache or access setup. |

Use the [block reference](blocks-reference.md) for each block's prerequisites.
Existing site images and specialized runtime/evaluation images are separate from
the generated scientific recipe catalog.

### Chaperone host and agent runtime

`containers/chaperone.def` installs Chaperone with AtomWorks 3.0.1 and Biotite
1.6.0. The evaluation `agent.def` links to this same recipe. From the repository
root, build it with `apptainer build --fakeroot chaperone.sif containers/chaperone.def`.
Model containers and checkpoints are installed separately with `chaperone install`.

The Foundry 1.0 recipes fetch PR #404 and check out the exact revision above.
Inference compatibility fixes and their regression tests live in Foundry.
Chaperone installs the native source and checks its CLI contracts during the build.
