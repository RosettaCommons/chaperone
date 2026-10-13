import logging
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

from chaperone.block_config import (
    BlockFunction,
    CompiledBlockFunction,
    ScriptSpec,
    InlineBlockFunction,
    InlineMetricBlock,
    loaded_block,
    metric_block,
)
from chaperone.data import PipelineDataHandler
from chaperone.engine import (
    ChunkContext,
    ExecutionEngine,
    RuntimeArgs,
)
from chaperone.manifest import PipelineManifest
from chaperone.settings import ChaperoneSettings
from chaperone.orchestration import JobArgs

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global shared settings (all threads see the same settings)
# ─────────────────────────────────────────────────────────────────────────────
_settings_lock = threading.Lock()
_global_settings: ChaperoneSettings | None = None


def _internally_set_settings(
    changes: dict[str, Any] | None = None, reset: bool = False
) -> None:
    """Update global settings. Thread-safe."""
    global _global_settings
    changes = changes or {}
    with _settings_lock:
        if _global_settings is None or reset:
            _global_settings = ChaperoneSettings()
        for key, value in changes.items():
            setattr(_global_settings, key, value)


def get_settings() -> ChaperoneSettings:
    """Get global settings shared across all threads."""
    global _global_settings
    if _global_settings is None:
        _internally_set_settings(reset=True)
    return _global_settings


_UNSET = object()


def get_setting_field(name: str, default=_UNSET, *, check_bin: bool = False):
    """Lazily read a single field from ChaperoneSettings.

    Designed for use in ``default_factory`` lambdas on block Pydantic fields,
    where importing ``get_settings`` directly would cause circular imports::

        cache_dir: str = Field(
            default_factory=lambda: get_setting_field("BOLTZ_CACHE_DIR", "~/.boltz"),
        )

    Args:
        name: Settings field name.
        default: Fallback if the field is not set. If not provided and the
            field is not found, raises ``ValueError``. Pass ``None`` explicitly
            to return ``None`` when the field is unset.
        check_bin: If ``True`` and the resolved value is not an absolute path
            that exists, try ``shutil.which()`` as a fallback before erroring.
    """
    import shutil as _shutil
    from pathlib import Path as _Path

    value = getattr(get_settings(), name, None)

    # If the settings field returned None, try default / error
    if value is None:
        if default is not _UNSET:
            value = default
        else:
            raise ValueError(
                f"Setting {name!r} is not configured. "
                f"Set it in .env, .env.ipd, .env.local, or as an environment variable."
            )

    # Optional executable lookup via PATH
    if check_bin and value and not _Path(value).is_file():
        found = _shutil.which(value)
        if found:
            value = found

    return value


_settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# CLI init state — stores overrides parsed by init()
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _InitState:
    initialized: bool = False
    block_overrides: dict = field(default_factory=dict)  # block_name -> {field: value}
    cli_config_defaults: dict = field(
        default_factory=dict
    )  # block_name -> {field: value} from --config CLI flag
    script_yaml_defaults: dict = field(
        default_factory=dict
    )  # block_name -> {field: value} from init_cli(defaults_yaml=...)
    runtime_overrides: dict = field(
        default_factory=dict
    )  # recompile, local, rundir, etc.


_init_state = _InitState()


def get_block_overrides(block_name: str) -> dict:
    """Return CLI flag overrides for a specific block, or {} if none."""
    return _init_state.block_overrides.get(block_name, {})


def get_cli_config_defaults(block_name: str) -> dict:
    """Return --config YAML defaults for a specific block, or {} if none."""
    return _init_state.cli_config_defaults.get(block_name, {})


def get_yaml_defaults(block_name: str) -> dict:
    """Return script-level YAML defaults (init_cli(defaults_yaml=...)), or {} if none."""
    return _init_state.script_yaml_defaults.get(block_name, {})


def get_runtime_overrides() -> dict:
    """Return CLI runtime overrides (recompile, local, rundir, etc.), or {} if none."""
    return _init_state.runtime_overrides


# Global flags that mirror cli.py lines 29-56
_GLOBAL_FLAG_DEFS = [
    ("rundir", "string", None, "Directory to store pipeline run outputs."),
    ("force_exec", "boolean", False, "Force execution even if cached results exist"),
    (
        "skip_cache",
        "boolean",
        False,
        "Skip step-level cache (resubmit jobs) but preserve chunk-level caches.",
    ),
    (
        "recompile",
        "boolean",
        False,
        "Recompile outputs from existing sandbox directories",
    ),
    (
        "local",
        "boolean",
        False,
        "Use local ThreadPool execution instead of SLURM (faster for small jobs)",
    ),
    (
        "pickup",
        "boolean",
        False,
        "If provided, pick up parquet from existing run directory, overwrite otherwise.",
    ),
    (
        "use_local_gpus",
        "boolean",
        True,
        "Use locally detected GPUs instead of submitting to SLURM. "
        "Set --nouse_local_gpus to force SLURM submission even when local GPUs are available.",
    ),
    (
        "partition",
        "string",
        None,
        "SLURM partition for all GPU blocks, or 'local' (e.g. --partition=gpu). "
        "If unset, GPU blocks use CHAPERONE_GPU_PARTITION, then "
        "CHAPERONE_DEFAULT_PARTITION, then the cluster default. CPU blocks are "
        "not affected. A partition override drops the GPU type from the block's "
        "gres (gpu:a100:1 becomes gpu:1) unless --gres is also given.",
    ),
    (
        "gres",
        "string",
        None,
        "SLURM --gres for all GPU blocks, passed to sbatch as-is (e.g. gpu:1, "
        "gpu:a100:1). Valid GPU types depend on the cluster. If unset, uses "
        "each block's size class mapped through CHAPERONE_GRES_MAP, else gpu:1.",
    ),
    (
        "account",
        "string",
        None,
        "SLURM --account for all blocks. If unset, uses CHAPERONE_DEFAULT_ACCOUNT.",
    ),
    (
        "qos",
        "string",
        None,
        "SLURM --qos for all blocks. If unset, uses CHAPERONE_DEFAULT_QOS.",
    ),
    (
        "extra_sbatch",
        "string",
        None,
        "Extra sbatch long options for all blocks, space-separated and quoted, "
        "e.g. --extra_sbatch='--reservation=mine --mail-type=END'. Replaces "
        "CHAPERONE_DEFAULT_EXTRA_SBATCH.",
    ),
    (
        "config",
        "string",
        None,
        "Name of the Hydra config to use for global block configuration.",
    ),
    (
        "test",
        "boolean",
        False,
        "Shorthand for --config=tests/test_overrides.yaml. "
        "Runs a minimal single-design test.",
    ),
    (
        "auto_store_format",
        "string",
        None,
        "Engine-level auto-store format: 'catcif' or 'none'. "
        "Overrides ExecutionEngine(auto_store_format=...). "
        "Automatically forced to 'none' under --test.",
    ),
    (
        "auto_store_cleanup",
        "boolean",
        None,
        "Delete chunk sandboxes after archival. "
        "Overrides ExecutionEngine(auto_store_cleanup=...). "
        "Automatically forced False under --test.",
    ),
    (
        "cleanup_logs",
        "boolean",
        None,
        "Consolidate per-chunk logs into logs/<step>.log and remove "
        "logs/<step>/ (including slurm/) after each step completes. "
        "Overrides ExecutionEngine(cleanup_logs=...). "
        "Automatically forced False under --test.",
    ),
]


def _load_yaml_defaults(path: str) -> dict[str, dict]:
    """Load block defaults from a YAML config file.

    Expected format::

        rfd3:
          n_batches: 10
          input: "input.json"
        mpnn:
          temperature: 0.1

    Returns:
        Mapping of block_name -> {field: value}.
    """
    from pathlib import Path

    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config YAML must be a mapping of block names to defaults, got {type(raw).__name__}"
        )

    defaults: dict[str, dict] = {}
    for block_name, fields in raw.items():
        if not isinstance(fields, dict):
            raise ValueError(
                f"Config YAML entry for '{block_name}' must be a mapping, got {type(fields).__name__}"
            )
        defaults[str(block_name)] = fields
    return defaults


def init_cli(
    argv: list[str] | None = None,
    defaults_yaml: str | None = None,
    blocks: list[type[BlockFunction]] | None = None,
) -> None:
    """Initialize CLI flag parsing for standalone pipeline scripts.

    Call this at the top of your script (after imports) to enable CLI
    flag overrides for block parameters and runtime settings.

    This replaces the need for a separate ``setup_logging()`` call.

    Priority order (highest wins):
        1. CLI flags (``--rfd3.n_batches=1``)
        2. CLI config (``--config=overrides.yaml``)
        3. Script kwargs (``engine.run("rfd3", n_batches=10)``)
        4. Script YAML defaults (``init_cli(defaults_yaml="defaults.yaml")``)
        5. Pydantic field defaults

    Args:
        argv: Command-line arguments to parse. Defaults to ``sys.argv``.
        defaults_yaml: Path to a YAML file that sets default values for block
            parameters. These are the lowest-priority overrides (above
            Pydantic defaults only). The YAML format is::

                rfd3:
                  n_batches: 10
                mpnn:
                  temperature: 0.1

            The ``--config`` CLI flag provides a separate, higher-priority
            YAML layer that sits above script kwargs.
        blocks: Block classes used by this script. Supplying this keeps startup
            selective and avoids importing every registered scientific tool.

    Example::

        from chaperone import init_cli, ExecutionEngine
        init()
        engine = ExecutionEngine(rundir="my_run")
        engine.run("rfd3", n_batches=10)  # --rfd3.n_batches=50 overrides
    """
    if _init_state.initialized:
        warnings.warn(
            "chaperone.init_cli() called more than once — ignoring", stacklevel=2
        )
        return

    # Lazy imports — ABSL is only pulled in when init() is actually called
    from absl import flags
    from chaperone.common import setup_logging
    from chaperone_cli.registry import register_flags

    # Setup logging (replaces manual setup_logging() call)
    setup_logging()

    # Register all block flags + global flags
    registered_blocks = register_flags(blocks)

    # Register global runtime flags
    FLAGS = flags.FLAGS
    for name, ftype, default, help_text in _GLOBAL_FLAG_DEFS:
        if hasattr(FLAGS, name):
            continue  # already registered (e.g. by cli.py)
        if ftype == "string":
            flags.DEFINE_string(name, default, help_text)
        elif ftype == "boolean":
            flags.DEFINE_boolean(name, default, help_text)

    # ``absl.app.run`` normally installs and handles the conventional help
    # flags. Standalone Chaperone scripts deliberately call FlagValues
    # directly, so provide the same read-only interface before parsing;
    # otherwise every generated campaign's first ``--help`` probe fails with
    # UnrecognizedFlagError even though flag registration succeeded.
    if argv is None:
        argv = sys.argv
    help_flags = {"-h", "--help", "--helpshort", "--helpfull"}
    if any(argument in help_flags for argument in argv[1:]):
        print(FLAGS.get_help())
        raise SystemExit(0)

    # Parse argv via FLAGS() — ABSL's direct parse API (no app.run() needed)
    FLAGS(argv)

    # --test expands to --config=tests/test_overrides.yaml + sensible test defaults
    if hasattr(FLAGS, "test") and FLAGS["test"].present and FLAGS["test"].value:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent.parent
        test_config = repo_root / "tests" / "test_overrides.yaml"
        if not test_config.is_file():
            raise FileNotFoundError(f"--test: cannot find {test_config}")
        # Only set config if not explicitly provided
        if not (hasattr(FLAGS, "config") and FLAGS["config"].present):
            FLAGS["config"].value = str(test_config)
            FLAGS["config"].present = True  # type: ignore[attr-defined]
        logger.info(f"--test mode: config={test_config}")

        # --test disables engine-level auto-store unless user explicitly set them
        if not FLAGS["auto_store_format"].present:
            FLAGS["auto_store_format"].value = "none"
            FLAGS["auto_store_format"].present = True  # type: ignore[attr-defined]
        if not FLAGS["auto_store_cleanup"].present:
            FLAGS["auto_store_cleanup"].value = False
            FLAGS["auto_store_cleanup"].present = True  # type: ignore[attr-defined]
        if not FLAGS["cleanup_logs"].present:
            FLAGS["cleanup_logs"].value = False
            FLAGS["cleanup_logs"].present = True  # type: ignore[attr-defined]

    # Collect explicitly-set overrides
    runtime_overrides: dict = {}
    for name, _, _, _ in _GLOBAL_FLAG_DEFS:
        if hasattr(FLAGS, name) and FLAGS[name].present:
            runtime_overrides[name] = FLAGS[name].value

    block_overrides: dict[str, dict] = {}
    for op_name, op_cls in registered_blocks:
        block_fields: dict = {}
        for field_name in op_cls.model_fields:
            flag_name = f"{op_name}.{field_name}"
            if hasattr(FLAGS, flag_name) and FLAGS[flag_name].present:
                block_fields[field_name] = FLAGS[flag_name].value
        if block_fields:
            block_overrides[op_name] = block_fields

    # Load YAML defaults from two independent sources:
    #   --config flag  → cli_config_defaults  (above script kwargs)
    #   defaults_yaml  → script_yaml_defaults (below script kwargs)
    cli_config_defaults: dict[str, dict] = {}
    script_yaml_defaults: dict[str, dict] = {}

    if hasattr(FLAGS, "config") and FLAGS["config"].present:
        cli_config_defaults = _load_yaml_defaults(FLAGS["config"].value)
        logger.info(f"init_cli(): loaded CLI config from {FLAGS['config'].value}")

    if defaults_yaml is not None:
        script_yaml_defaults = _load_yaml_defaults(defaults_yaml)
        logger.info(f"init_cli(): loaded script YAML defaults from {defaults_yaml}")

    _init_state.block_overrides = block_overrides
    _init_state.cli_config_defaults = cli_config_defaults
    _init_state.script_yaml_defaults = script_yaml_defaults
    _init_state.runtime_overrides = runtime_overrides
    _init_state.initialized = True

    logger.debug(f"init_cli(): cli_config_defaults={cli_config_defaults}")
    logger.debug(f"init_cli(): script_yaml_defaults={script_yaml_defaults}")
    logger.debug(f"init_cli(): block_overrides={block_overrides}")
    logger.debug(f"init_cli(): runtime_overrides={runtime_overrides}")


__all__ = [
    "ChunkContext",
    "ExecutionEngine",
    "PipelineDataHandler",
    "PipelineManifest",
    "BlockFunction",
    "CompiledBlockFunction",
    "InlineBlockFunction",
    "InlineMetricBlock",
    "JobArgs",
    "RuntimeArgs",
    "get_settings",
    "_internally_set_settings",
    "loaded_block",
    "metric_block",
    "init_cli",
    "get_block_overrides",
    "get_cli_config_defaults",
    "get_yaml_defaults",
    "get_runtime_overrides",
    "get_setting_field",
    "ScriptSpec",
]
