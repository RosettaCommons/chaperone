# Containers & Environment Configuration

## Environment Variables & Settings

All env vars are managed through `ChaperoneSettings` in `settings.py`. **Never** read env vars directly with `os.environ.get()` / `os.getenv()` in blocks or Chaperone code. Use `get_setting_field()` from `chaperone` instead.

**Block field defaults:** Never hardcode institution-specific paths (e.g. `/projects/ml/...`) in Pydantic field defaults. Those belong in `.env.ipd`. Use generic tool names (e.g. `"foldseek"`, `"rosetta_scripts"`) and `get_setting_field("FIELD_NAME", check_bin=True)` in `default_factory` lambdas.

**Helper function:** `get_setting_field(name, default=None, *, check_bin=False)` — reads from `ChaperoneSettings`. If `default` is `None` and the field is unset, raises `ValueError`. With `check_bin=True`, falls back to `shutil.which()` for executable lookup.

**Adding new env vars:** Any new env var that should be read from `.env`/`.env.ipd`/`.env.local` **must** have a corresponding field in `ChaperoneSettings` (`settings.py`). The pydantic-settings loader only picks up env vars that match declared fields — undeclared vars in `.env.ipd` are silently ignored.

Key `CHAPERONE_` settings:
- `CHAPERONE_COMPUTE_BACKEND` — `auto` (default: local GPU → SLURM → local fallback), `slurm`, `local`, `modal`/`cloud`, `sge`/`pbs` (not yet implemented)
- `CHAPERONE_CONTAINER_BACKEND` — `apptainer/docker` (auto-resolves to first available)
- `CHAPERONE_CONTAINER_DIRS` — colon-separated container image search paths
- `CHAPERONE_DISABLE_RICH` — disable rich console formatting
- `CHAPERONE_CLOUD_MODE` — when True (auto-enabled for modal/cloud backend), blocks stage all external files into hermetic chunk directories

## Configuration

All settings are managed via `ChaperoneSettings` (pydantic-settings). There is **no** automatic env prefix (`env_prefix=""`); env var names match field names exactly (e.g. `RUNDIR`, `CHAPERONE_COMPUTE_BACKEND`). Values are resolved in this order (highest priority first):
1. Environment variables matching the field name (e.g. `RUNDIR`, `CHAPERONE_COMPUTE_BACKEND`)
2. `.env.local` — user-specific overrides (gitignored)
3. `.env.ipd` — IPD-specific paths/mirrors (checked in, auto-loaded only when at IPD)
4. `.env` — generic defaults (checked in)
5. Field defaults in `settings.py`

**Env file structure:**
- `.env` — generic, portable defaults. Checked into git. No machine-specific paths.
- `.env.ipd` — IPD environment overrides (mirrors, weights, tools). Checked into git. Only loaded when `/projects/ml/pipelines` exists.
- `.env.local` — user-specific overrides. Gitignored. Created by `/setup` or manually.

## Container Rebuilds

Full container rebuilds (`chaperone install foundry --force`) take 30-60 minutes due to CUDA toolkit and apex compilation. To **patch only the upstream tool** (e.g. update foundry to latest production) without rebuilding CUDA/apex:

```bash
# 1. Convert SIF to writable sandbox (~1 min)
apptainer build --fakeroot --sandbox /tmp/foundry_sandbox /path/to/foundry.sif

# 2. Update just the tool inside (~1 min)
apptainer exec --fakeroot --writable /tmp/foundry_sandbox bash -c '
  . /opt/foundry-env/bin/activate
  cd /opt/foundry && git pull origin production
  pip install -e ".[all]"
'

# 3. Repack to SIF (~1 min)
apptainer build --fakeroot --force /path/to/foundry.sif /tmp/foundry_sandbox
rm -rf /tmp/foundry_sandbox
```

This works for any container where the tool is installed via `pip install -e` from a git clone inside the image. Use `CHAPERONE_CONTAINER_DIRS` to point at your local rebuild:
```bash
export CHAPERONE_CONTAINER_DIRS="$HOME/.chaperone/containers:/projects/ml/pipelines/containers"
```

## BindCraft image

Build the integrated BindCraft image with:

```bash
chaperone install bindcraft
```

The image embeds BindCraft, its CUDA 12 JAX environment, and the public
AlphaFold2 parameters. It is large and requires substantial temporary build
space. At runtime, use an NVIDIA GPU with at least 32 GB VRAM; the block
requests `gpu:large` by default. PyRosetta in the image is licensed for
academic/non-commercial use by default, and AlphaFold2 parameters are CC BY
4.0. Keep the SIF access-controlled and verify dependency licensing before
redistributing it or using it commercially.
