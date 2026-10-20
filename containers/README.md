# Download and build scientific tool images

Chaperone ships Apptainer definitions for every registered buildable scientific
image. Several blocks share an image. These are source recipes and build inputs;
model weights, databases, separately licensed tools and GPU drivers still need
their own setup. See the [installation guide](../docs/site/containers.md) for
runtime settings, model access and host dependencies.

## Get the files

- Clone this repository, or download its [source ZIP](https://github.com/RosettaCommons/chaperone/archive/refs/heads/main.zip).
- Download individual definitions using the links below. Keep their preparation
  scripts and all listed `sources/` files too; a `.def` alone may be insufficient.

The [machine-readable catalog](apptainer/catalog.json) maps images to blocks,
source paths and build-access notes. Download links follow `main`; use a fixed
commit or release when you need a reproducible snapshot.

## Build from the downloaded context

For a repository checkout:

```bash
cd containers/apptainer
apptainer build foundry.sif foundry.def
bash alphafold3.prepare.sh
apptainer build alphafold3.sif alphafold3.def
```

Run every build from the directory containing `catalog.json` and the `.def` files:
`%files` paths
are relative to this build context. Use the matching `<image>.prepare.sh` when
listed below, then `apptainer build <image>.sif <image>.def`. Preparation fetches
upstream source repositories; the bundled `sources/` files need no extra staging.
Preparation refuses to overwrite existing checkout paths. If retrying after an
interrupted preparation, inspect and move aside those paths before rerunning it.
You can rebuild an image from an already prepared context without rerunning prep.

Build on a Linux host with Apptainer build support, outbound network access and
enough disk, RAM and CPUs. Add `--fakeroot` only as required by your site's build
configuration. A site may require building inside a compute allocation. Building
does not require loading models on a GPU; execution still needs the hardware and
driver supported by each tool.

To use an image with Chaperone, keep its registered filename (for example,
`foundry.sif`) in a directory listed in `CHAPERONE_CONTAINER_DIRS`, and select
`CHAPERONE_CONTAINER_BACKEND=apptainer`. For automatic staging and building from
an installed Chaperone environment, use `chaperone install rfd3 af3` or another
block/image name. Configure weights and `CHAPERONE_COMPUTE_BACKEND` separately.
For Foundry, run `foundry install rfd3 --checkpoint-dir <directory>` (or
`base-models` for RFD3, RF3 and MPNN) inside its image before inference, then set
`FOUNDRY_CHECKPOINT_DIRS` to the corresponding host directory. Missing Foundry
weights are not downloaded by the first block call; see the
[complete staging command](../docs/site/containers.md#rfd3-and-af3-runtime-setup).

## Definitions

The table covers every buildable registered image. The preparation column links
scripts only for recipes that clone host-side sources. Image names without blocks
are explicit recipe variants. Static checks verify rendering, staging inputs and
shell syntax; they do not establish that every image builds and runs on every host.

| Image / definition | Blocks | Source preparation |
|---|---|---|
| [`alphafold3.def`](apptainer/alphafold3.def) | `af3`, `af3_with_target_template` | [Prepare](apptainer/alphafold3.prepare.sh) |
| [`bindcraft.def`](apptainer/bindcraft.def) | `bindcraft` | [Prepare](apptainer/bindcraft.prepare.sh) |
| [`bindcraft2.def`](apptainer/bindcraft2.def) | `bindcraft2` | Not required |
| [`boltz2.def`](apptainer/boltz2.def) | `boltz2` | Not required |
| [`boltzgen.def`](apptainer/boltzgen.def) | `boltzgen` | Not required |
| [`chai1.def`](apptainer/chai1.def) | `chai1` | Not required |
| [`colabfold_portable.def`](apptainer/colabfold_portable.def) | `colabfold` | Not required |
| [`disco.def`](apptainer/disco.def) | `disco` | [Prepare](apptainer/disco.prepare.sh) |
| [`e1.def`](apptainer/e1.def) | `e1` | Not required |
| [`esm2.def`](apptainer/esm2.def) | `esm2` | Not required |
| [`esmfold2.def`](apptainer/esmfold2.def) | `esmfold2` | Not required |
| [`foundry.def`](apptainer/foundry.def) | `foundry_mpnn`, `rf3`, `rfd3`, `rfd3_partial`, `rfd3na` | Not required |
| [`foundry_public.def`](apptainer/foundry_public.def) | Explicit variant | Not required |
| [`ligandmpnn.def`](apptainer/ligandmpnn.def) | `mpnn` | [Prepare](apptainer/ligandmpnn.prepare.sh) |
| [`metal3d.def`](apptainer/metal3d.def) | `metal3d` | [Prepare](apptainer/metal3d.prepare.sh) |
| [`openfold3.def`](apptainer/openfold3.def) | `openfold3` | [Prepare](apptainer/openfold3.prepare.sh) |
| [`protein_hunter.def`](apptainer/protein_hunter.def) | `boltz2_steered`, `protein_hunter` | [Prepare](apptainer/protein_hunter.prepare.sh) |
| [`proteina_complexa.def`](apptainer/proteina_complexa.def) | `proteina_complexa` | Not required |
| [`protenix.def`](apptainer/protenix.def) | `protenix` | Not required |
| [`rfdiffusion1.def`](apptainer/rfdiffusion1.def) | `rfd1` | [Prepare](apptainer/rfdiffusion1.prepare.sh) |
| [`rfdiffusion2_portable.def`](apptainer/rfdiffusion2_portable.def) | `rfd2` | [Prepare](apptainer/rfdiffusion2_portable.prepare.sh) |
| [`rfoptimization.def`](apptainer/rfoptimization.def) | `rfoptimization` | Not required |
| [`rpxdock.def`](apptainer/rpxdock.def) | `rpxdock` | Not required |
| [`simplefold.def`](apptainer/simplefold.def) | `simplefold` | [Prepare](apptainer/simplefold.prepare.sh) |
| [`tmol.def`](apptainer/tmol.def) | `tmol_interface`, `tmol_minimize` | Not required |

## Maintain the exported definitions

The Python `Image` declarations in `src/blocks` are the source of truth. After
changing them, regenerate the downloadable contexts using the Chaperone Python
environment from the repository root:

```bash
python containers/export_definitions.py
python containers/export_definitions.py --check
```

Export performs no network downloads or image builds. Use `--check` to verify
that exports match the current source declarations. Edit the source declarations and
regenerate; do not edit generated definitions or bundled copies directly.

The separate [Chaperone host runtime](chaperone.def) builds from the repository
root with `apptainer build --fakeroot chaperone.sif containers/chaperone.def`.
Evaluation runtime images have their own [setup guide](../docs/site/evaluations.md).
