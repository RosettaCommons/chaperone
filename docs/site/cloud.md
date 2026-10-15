# Modal cloud execution

Install the optional client and authenticate with your Modal workspace:

```bash
uv pip install -e '.[cloud]'
modal token set --profile=my-workspace
modal profile activate my-workspace
export CHAPERONE_COMPUTE_BACKEND=modal
```

The token command prompts for credentials. Keep credentials in Modal's profile
file or its supported environment variables, never in pipeline code. GPU
Sandboxes require a workspace with billing enabled.

## Running a pipeline

Existing scripts can select Modal with the environment setting above.
Alternatively, select it for one engine:

```python
from chaperone import ExecutionEngine

engine = ExecutionEngine(
    rundir="runs/modal",
    run_args_default={"mode": "modal", "max_workers": 2, "max_retries": 0},
)
result = engine.run("copy", data="input.pdb", n_multiply=2)
```

`cloud` is an alias for the `modal` backend/mode. The legacy compiled-block
option `cloud=True` also requests Modal. An explicit `mode="local"` overrides
the environment backend. Local GPU detection does not override Modal.

Chaperone stages and compiles on the client, runs the external command in a
fresh Modal Sandbox, and downloads outputs before the normal lineage, storage,
and cache steps. Inline Python blocks run on the client. Chunking and retries
use the existing engine. Each attempt has an isolated filesystem and its
Sandbox is terminated on success or failure. Input and output paths keep their
absolute names inside the Sandbox, including paths embedded in JSON files.

## Resources

| Setting | Default | Meaning |
| --- | --- | --- |
| `CHAPERONE_MODAL_GPU` | `A10G` | Modal GPU type for GPU jobs; choose a larger GPU when the model needs it |
| `CHAPERONE_MODAL_GPU_MAP` | `{}` | Optional per-class override, e.g. `{"large":"A100-80GB"}` |
| `CHAPERONE_MODAL_TIMEOUT` | `1800` | Maximum Sandbox lifetime in seconds, further limited by `JobArgs.time_limit` |
| `CHAPERONE_MODAL_MAX_WORKERS` | `2` | Maximum concurrent chunks, unless `RuntimeArgs.max_workers` is supplied |
| `CHAPERONE_MODAL_APP` | `chaperone` | Modal app name |
| `CHAPERONE_MODAL_VOLUMES` | `{}` | JSON mapping of absolute mount paths to existing Modal volume names |

CPU-only jobs request no GPU. `JobArgs.cpus_per_task` and `mem_gb` become Modal
CPU and memory requests. Slurm partitions and node constraints do not select
Modal hardware. `JobArgs.gres="gpu:2"` requests two GPUs of the configured type.
Use `max_retries=0` for a first smoke test to avoid repeated
failed cloud attempts.

## Images, inputs and checkpoints

Buildable Chaperone image recipes become Modal images. A local SIF is not
required. The first run builds the image; Modal caches subsequent builds.
Only the generated Dockerfile and explicitly declared recipe `COPY` sources
are in the image build context. Credentials are
not inserted into Dockerfiles, URLs or image layers.

Compiled blocks should use `stage_file()` or `stage_or_copy()` for inputs.
In cloud mode these make real copies. Duplicate basenames are staged under
distinct directories. External symlinks and unstaged host file arguments are
rejected before starting a Sandbox.

The `mpnn` block (LigandMPNN) has its checkpoints in its image. For tools with external checkpoints, upload the permitted model data
to a Modal volume and mount it at the path the tool expects. For example:

```bash
modal volume create chaperone-foundry-checkpoints
modal volume put chaperone-foundry-checkpoints /local/checkpoints/rfd3_latest.ckpt /rfd3_latest.ckpt
export CHAPERONE_MODAL_VOLUMES='{"/weights/foundry":"chaperone-foundry-checkpoints"}'
export FOUNDRY_CHECKPOINT_DIRS=/weights/foundry
```

Use `ckpt_path="/weights/foundry/rfd3_latest.ckpt"` when explicitly selecting
that checkpoint. Volume contents are managed by the user; Chaperone does not
automatically upload host directories or infer which files may be shared.
The mounted volumes persist between Sandbox attempts; chunk workspaces do not.

Local-only SIF images, editable host checkouts, private-repository image builds,
and arbitrary `command_template` overrides are not automatically portable.
Explicit host binds require a corresponding configured Modal volume.
RFD3NA retains its explicit cloud checkpoint guard. Run a small smoke test for each scientific block and checkpoint combination.

## Smoke tests

The current Foundry recipes use AtomWorks 3. Rebuild cached images after changing
recipe revisions and verify the installed versions before comparing results.
The commands below exercise execution, output retrieval and caching; passing
smoke tests does not establish scientific quality or equivalence between model
versions.

The ordinary regression suite performs no cloud submissions. Live tests are
opt-in and may incur Modal charges:

```bash
CHAPERONE_RUN_MODAL_TESTS=1 pytest tests/test_modal_backend.py -m cloud -k copy
CHAPERONE_RUN_MODAL_MPNN=1 pytest tests/test_modal_backend.py -m cloud -k live_modal_mpnn
# After uploading both public checkpoints to chaperone-foundry-checkpoints:
CHAPERONE_RUN_MODAL_FOUNDRY=1 pytest tests/test_modal_foundry.py -q -s
```

The first command tests CPU and GPU copies; the second builds the MPNN
image and checks CPU and GPU sequence generation and cache replay. Synthetic inputs
exercise pipeline functionality, not protein design quality.
The Foundry test uses the volume named by `CHAPERONE_MODAL_FOUNDRY_VOLUME`,
defaulting to `chaperone-foundry-checkpoints`. It expects `rfd3_latest.ckpt` and
`rf3_foundry_01_24_latest_remapped.ckpt` at the volume root.
