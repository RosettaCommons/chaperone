# Release validation — 2026-10-13

This records engineering checks for the public-release changes. It is **not a
scientific benchmark or certification of every block**. The results below record completed checks at PR closeout. Unfinished coverage
is explicitly deferred; no additional cluster tests are running.

## Local checks

| Check | Result |
|---|---|
| Framework regression suite | **631 passed, 14 skipped** after the targeted naming-check rerun following the main-branch docs merge. Includes PR #356 regressions, smoke configuration, container selection/binds, site settings and Slurm resources. |
| Eval controller suite | **259 passed, 2 skipped**, plus **27 passing subtests**. Includes snapshots, isolation, site profiles, scoring dispatch and guarded Slurm submission. |
| Submission/scoring helpers | **13 passed**. |
| Wheel and source distribution | Built successfully. Wheel: 184 members; source archive: 243. Project MIT metadata, packaged third-party notices, the MPNN recipe and required runtime assets present; no eval harness or `.env.ipd` included. |
| Documentation | Block reference regenerated; strict MkDocs build passed. |
| Container recipes | All 22 scientific-tool recipes rendered and checked statically. This does not mean all 22 were rebuilt or exercised. Eval image shell sections also passed syntax checks. |

Local tests used the existing Python environment with an isolated dependency
overlay containing AtomWorks 3.0.1, Biotite 1.6.0, Numba, llvmlite and
`catcif-tools`. The GPU
driver used an isolated virtual environment with shared packages and updated
its declared AtomWorks/Biotite dependencies. These are not clean-host installation
results. The initial old-dependency failures were resolved before the counts
above were recorded.

To repeat the local checks with the declared framework and eval test dependencies
installed:

```bash
PYTHONPATH=src python -m pytest tests -q
python -m build
python docs/site/gen_block_docs.py
python -m mkdocs build --strict
cd src/chaperone-evals
PYTHONPATH=harbor-controller python -m pytest harbor-controller/tests -q
PYTHONPATH=harbor-controller python -m pytest scoring -q
```

Controller socket tests need local Unix-socket permission. The final controller
run used the sandbox for non-socket coverage and reran the complete socket test
file with that permission; the counts above include the passing rerun. On macOS, use a
canonical temporary directory such as `TMPDIR=/private/tmp` to avoid `/var` versus
`/private/var` path differences in test fixtures.

## CoreWeave clean-install check — 2026-10-13

A separate checkout and new Python 3.12.10 virtual environment were installed
with `uv pip install -e '.[all]' --override overrides.txt`, without inheriting
system packages. All 197 dependencies installed and `uv pip check` passed.
The driver resolved AtomWorks 3.0.1, Biotite 1.6.0, Torch 2.14.1 and Submitit
1.5.4. The scientific container has its own dependency set. After the two repairs
below, the full framework suite passed: **647 passed, 6 skipped**. Ruff checks
on changed Python files and the strict MkDocs build also passed.

Two clean-install regressions were repaired: MPNN PDB staging now supplies
PDB defaults for missing optional CIF B-factor/occupancy values on a copy, and
partial runtime dictionaries preserve the engine's configured execution mode
and retry/concurrency policy. Explicit `RuntimeArgs` objects continue to replace
the full runtime configuration. A cache-only override previously triggered an
unintended Slurm submission when running on a host with Slurm installed.

The CPU functional smoke passed locally and through Slurm, including real
copy/RMSD execution, strict missing-metric rejection, required-structure
selection, CatCIF round trips, native sequences and cache replay. Requests used
one CPU, 4 GB RAM, five minutes and no GPU; Slurm selected its default `hpc-mid`
partition. Copy and archive-copy records report zero allocated GPUs. No IPD site
file, `cpu` partition, GPU feature constraint or GRES size alias was needed.

The tested cluster exposes `gpu:h200:8` with `select/cons_tres`; a generic
`--gres=gpu:1` request allocated one H200 (driver 595.71.05). These observations
are not defaults for every CoreWeave installation. Inspect a new site's
partitions/GRES and choose its shared paths:

```bash
sinfo -o '%P %G %l'
export CHAPERONE_SITE=portable
export CHAPERONE_CONTAINER_BACKEND=apptainer
export CHAPERONE_CONTAINER_DIRS=/shared/chaperone/containers
# Optional when the Slurm default is unsuitable:
# export CHAPERONE_GPU_PARTITION=your-gpu-partition
# export CHAPERONE_GRES_MAP='{"default":"gpu:h200:1"}'
```

Portable GPU jobs omit the partition and request `gpu:1` unless configured
otherwise. Explicit site mappings remain available for heterogeneous hardware;
we did not introduce a CoreWeave preset or infer resource policy from GPU names.

The login node rejected Apptainer `%post` with a mount-propagation permission
error. Run container installation inside an approved compute allocation on this
cluster, using the same shared image directory. The fresh LigandMPNN
build there completed with Torch 2.11.0/CUDA 12.8 and passed package
compatibility, six checksum/load gates and imports.

To test packing without Foundry or external checkpoint directories, first
run the CPU smoke above, then use its two copied structures:

```bash
# Omit --local: one engine-managed Slurm GPU job, capped at 15 minutes.
python tests/mpnn_packing_smoke.py \
  --inputs=/absolute/path/to/cpu-smoke/pipeline/copy/output.parquet \
  --test --rundir=/absolute/path/to/new-packing-smoke
```

This helper requests the configured GPU partition and large-class GRES explicitly
because MPNN otherwise defaults to CPU. It checks eight packed structures,
fixed residues, restored multi-character chains, parent lineage, finite coordinates,
one allocated GPU and cache replay. `--local` instead uses an existing allocation.

The fresh-image Slurm packing check **passed**: two 24-residue inputs × two
sequences × two packs produced all eight structures, with fixed residues and
`X1` chain IDs preserved, valid lineage and finite coordinates. Execution
recorded **11.2 GPU seconds** on one H200; reopening the engine replayed disk
cache without another submission or changed output timestamps. The CPU and GPU
checks used normal block algorithms and retained ordinary pipeline artifacts.
This is execution/data-integrity evidence, not a design-quality benchmark.
Foundry remains on its prior tested container revision and AtomWorks 2.2.1 pin;
the separate upstream AtomWorks 3 port is not claimed as validated by this run.

## GPU and build matrix

The results below cover a subset of the scientific runtimes and two eval
images. **The 22 scientific recipes have not all been built or GPU-tested.**
A successful build checks installation/imports; a GPU smoke
checks execution and output integration. Neither establishes prediction quality.

| Runtime | Source build | Runtime evidence / outstanding work |
|---|---|---|
| SimpleFold | Passed | Fresh image GPU smoke passed on a 48-residue input, using `simplefold_100M` without the optional pLDDT model. |
| ESMFold2 | Passed | Fresh-image GPU smoke passed on the same 48-residue input using the upstream default three loops and no local MSA: 394 atoms, finite coordinates, and the coordinate-extent check passed. The earlier one-loop diagnostic is superseded by this run. |
| MPNN (`ligandmpnn`) | Repaired full rebuild v3 passed | Fresh-image GPU sequence generation passed: two RFD3 backbones, two sequences each, with exact lineage and sequence checks. The six official checkpoints baked at the time passed checksums and load checks. The image now bakes all fifteen official checkpoints (see the [container guide](containers.md)); a rebuild passed all fifteen checksums, and a CPU run of `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`, both membrane models and side-chain packing on a bundled 150-residue structure returned sequences and structures. The GPU checks above have not been repeated on the rebuilt image. Fresh-image side-chain packing subsequently passed on a CoreWeave H200: eight outputs, fixed residues and X1 chains preserved, verified lineage and disk-cache replay. No equivalence claim versus other backends. |
| AlphaFold3 | Passed | Fresh-image 48-residue GPU smoke produced 394 atoms with finite coordinates. The repaired FGFR2 workflow also executed through target-template AF3 and downstream metrics/selection; see below. |
| Foundry | Full repaired source rebuild v3 passed | Initial runtime exposed an incompatible AtomWorks polymer-bond API. An isolated derivative pinned to AtomWorks 2.2.1 passed real GPU RFD3 → FoundryMPNN in two Slurm array jobs. The complete rebuilt image also passed the two-backbone RFD3 core check with MPNN. Fresh-image RF3 batch coverage is deferred; previous RF3 example checks used the site image. |
| DISCO | Transformers-repaired full build passed; final architecture repair not rebuilt | The first GPU failure was a Transformers 5 checkpoint-layout mismatch. The corrected rebuild reached a second upstream failure: a comma-delimited CUDA architecture list. The final recipe repairs that assignment and passes parser/regression checks; final GPU validation is deferred. |
| Boltz2 | Passed | Fresh-image GPU smoke passed for 48 residues, one sample and the normal 512-step default, returning finite coordinates. |
| Chai1 | Passed, with explicit Torch compatibility override | Fresh built-image GPU smoke passed with one sample and the normal 200 diffusion steps; compiled artifact step naming is correct after its repair. |
| Protenix | Full source build passed | Fresh-image GPU smoke passed on 48 residues with the normal 200 diffusion steps and ten cycles: finite coordinates, pLDDT 78.271866, no reported clash. |
| E1 | Passed | Fresh-image GPU inference passed after the DNS fix, including public Hugging Face checkpoint download and finite logits, token embeddings, and mean embeddings. |
| Eval agent | Passed | Clean imports/block discovery and CPU functional smoke passed. Nested Apptainer execution passed with `/dev/fuse` bound, matching the controller runtime binds. Full LLM agent trial remains untested. |
| Eval baseline | Passed | Image build checks passed; full baseline trial pending. |

The final core batch on one RTX A6000 passed fresh-image RFD3 → MPNN,
then independent AF3 and ESMFold2 predictions of all four designed sequences.
The helper verified exact sequences, parent lineage, finite bounded coordinates,
confidence metrics and disk-cache replay without new execution or changed output
tables: **296.0 GPU seconds**. AF3 used its normal sampling settings; ESMFold2
used its normal three loops. RF3 was not part of this final batch. Its separate
early-stop regression tests preserve the default threshold and distinguish
intentional stops from unexplained missing outputs.

Foundry remains pinned to **AtomWorks 2.2.1** as the tested compatibility
workaround. Fixing upstream compatibility with AtomWorks 3 is deferred until
after merge; this PR does not claim that compatibility.

A prior PPI Foundry example executed using an existing site image, with **0 of 1**
designs passing its scientific filters. This supplies execution evidence only;
it does not establish a successful design campaign or validate the fresh image.
The repaired FGFR2 example executed, in an earlier run, RFD3 → MPNN → AF3 with target
template → RMSD → ipSAE → CMS → PEPPR → final filter. All these stages executed
successfully, but **0 of 1** designs passed the unchanged scientific gates. The
completed PPI and FGFR2 workflows therefore demonstrate runtime integration,
not successful binder design. The bounded AME workflow completed both inputs
after the AF3 sequence-authority fix, supplying execution evidence without a
claim of design quality.

The earlier CatCIF example completed archive writing/materialization and
RFD3 → MPNN → AF3 → RMSD, but that run preceded the AF3 fix: default
mmCIF preparation could predict the backbone context sequence instead of the
designed MPNN sequence. Its RMSD is therefore not validation of the intended
designed sequence. The fix routes authoritative sequence data, including folded
descendants, through JSON input and invalidates the affected AF3 caches. A
corrected CatCIF inference rerun remains pending. Other example reruns are deferred at the requested scope cutoff. No full scientific benchmark, FoundryMPNN equivalence study, or
experimental validation was performed.

Chai1's recipe deliberately uses Torch 2.7.1/CUDA 12.8 for Blackwell although
`chai_lab==0.6.1` declares `torch<2.7`. The build resolves its other dependencies
normally and permits only that exact mismatch in its dependency check. This is
an explicit compatibility override, not a clean `pip check` result; the repaired image passed the bounded GPU smoke described above.

Public recipes do not grant access to private repositories, licensed software or
model weights. The `mpnn` block installs LigandMPNN from source with its official checkpoints. Third-party
software and weights retain their own terms despite Chaperone's MIT license.
The [third-party notices](https://github.com/RosettaCommons/chaperone/blob/main/THIRD_PARTY_NOTICES.md)
identify the bundled AF3 runner (CC BY-NC-SA 4.0) and nine eval example structures
with embedded AF3 output terms. Their original notices remain intact; MIT applies
to Chaperone-authored code, not to relicensing these upstream materials.
Held-out scorers/reference material in public repository history remain accessible
to network-enabled agents; snapshot and wheel isolation cannot make that history
private. Separate private material/history or describe these as public tasks.

Older prebuilt images failed on Blackwell with unsupported GPU kernels. Rebuild
the image as described in the [container guide](containers.md).
Do not interpret a CPU substitution as successful GPU validation.

## Repeat the CPU functional smoke

This check uses actual installed framework blocks, one CPU worker, 24 residues
from a bundled structure and three short sequences. By default it runs locally;
`--mode=slurm` runs compiled blocks as engine-managed CPU jobs. It performs no
model inference. Run with the installed framework Python and a
new, empty output directory (the repository is needed only for the script/input):

```bash
python tests/cpu_functional_smoke.py --repo . --rundir /absolute/path/to/cpu-smoke
# Shared Python installation; CPU partition comes from site settings or Slurm.
python tests/cpu_functional_smoke.py --mode=slurm --repo . --rundir /absolute/path/to/cpu-slurm-smoke
```

The local run passed: real copy execution, disk resume with unchanged cache
artifacts and execution accounting, lineage RMSD, passing/failing/missing metric
gates, selection requiring a measured structure step, CatCIF and parquet
roundtrips, materialization, archived-input command staging, native sequence CSV
roundtrip and sequence deduplication. Its ordinary manifests, metrics and files
remain in the output directory. The same assertions also passed inside the
freshly built agent image. The local dependency overlay included `catcif-tools`
package for this check.

## Repeat the bounded Slurm array smoke

With Foundry and site Slurm settings available, run this foreground driver using
the installed framework Python. It may run inside an interactive allocation;
the engine submits its own array and does not reuse the driver's GPU:

```bash
python tests/slurm_batch_smoke.py --test --check-config
python tests/slurm_batch_smoke.py --test --rundir=/absolute/path/to/new-slurm-smoke
```

The script explicitly selects two RFD3 batches despite `--test` normally choosing
one. Each chunk generates one 40-residue backbone and one FoundryMPNN sequence in
the same job. It requests 24 GB RAM, two CPUs, a 15-minute limit and the site's
small-class GPU mapping, with at most two array tasks running concurrently and
no retries. `SMOKE_GPU_CONSTRAINT` can supply a node constraint. It checks both
outputs and their lineage, actual inner Slurm job IDs and positive GPU accounting,
then reopens the engine and requires unchanged artifacts and execution records
on cache replay. The real two-job array passed using the repaired Foundry
derivative: **264.5 GPU seconds**, two finite 40-residue backbones and two matching
sequence outputs. Reopening the engine and replaying the chain submitted no
additional jobs and changed neither cache artifacts nor execution accounting.
No separate polling of engine-managed inner jobs is needed.

## Repeat the seven example smoke runs

Run inside an allocated GPU shell with the required containers, checkpoints and
licensed assets available. See the [example prerequisites](https://github.com/RosettaCommons/chaperone/blob/main/examples/README.md#bounded-example-smoke-runs).
From the repository root, choose a new output root for each validation attempt:

```bash
export CHAPERONE_REPO="$(pwd)"
export SMOKE_ROOT=/absolute/path/to/release-smokes

python examples/benchmarks/ppi/pipeline_foundry.py --test --local --rundir="$SMOKE_ROOT/ppi-foundry"
python examples/benchmarks/ppi/pipeline_mpnn.py --test --local --rundir="$SMOKE_ROOT/ppi-mpnn"
python examples/benchmarks/disco/pipeline.py --test --local --rundir="$SMOKE_ROOT/disco"
python examples/benchmarks/ame/pipeline.py --test --local --rundir="$SMOKE_ROOT/ame"
python examples/benchmarks/ame/pipeline_full.py --test --local --rundir="$SMOKE_ROOT/ame-full"
python examples/fgfr2-binder/pipeline.py --test --local --rundir="$SMOKE_ROOT/fgfr2"
python examples/tests/catcif-storage/pipeline.py --test --local --auto_store_format=catcif --auto_store_cleanup --rundir="$SMOKE_ROOT/catcif"
```

`--test` reduces sampling; it does not relax scientific filters. AME explicitly
selects one input from each included task set. Other examples retain their target
sizes. A zero-survivor filter can leave subsequent stages untested: report the
last executed stage and validate an unexercised block with a separate diagnostic.
Do not weaken the campaign's acceptance criteria to obtain a passing smoke run.

For the independent prediction-block diagnostic:

```bash
SMOKE_MODEL=simplefold SMOKE_RUNDIR="$SMOKE_ROOT/simplefold" \
  python tests/gpu_prediction_smoke.py --test --local
SMOKE_MODEL=esmfold2 SMOKE_RUNDIR="$SMOKE_ROOT/esmfold2" \
  python tests/gpu_prediction_smoke.py --test --local
SMOKE_MODEL=af3 SMOKE_RUNDIR="$SMOKE_ROOT/af3" \
  python tests/gpu_prediction_smoke.py --test --local
```

This script uses the first 48 residues of the bundled insulin-receptor input and
requires exactly one output with finite coordinates. Select the newly built
images through `CHAPERONE_CONTAINER_DIRS`; an optional
`SMOKE_ESMFOLD2_SNAPSHOT` selects a local model snapshot. Retain the pipeline
manifests, logs, output structures, image/source revisions and checkpoint versions
with each result. The assertions verify execution, not structural accuracy.

### Latest runtime findings

Chai now passes fresh built-image GPU inference and artifact integration after correcting its compiled step name. Protenix now has an exact external-cache mount and a coherent pinned upstream dependency stack; its fresh full source build and bounded GPU smoke passed. Failed subprocesses now stop both compiled execution paths before stale outputs can be compiled; real subprocess regressions passed. The final RFD3/MPNN/AF3/ESMFold2 batch passed. Additional RF3, B4000, MPNN packing and nested scientific eval runs are deferred; their opt-in helpers remain available.
