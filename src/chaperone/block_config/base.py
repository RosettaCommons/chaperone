from typing import Any, ClassVar, Literal, TYPE_CHECKING
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from pathlib import Path
import logging

from chaperone.orchestration.job_args import JobArgs

if TYPE_CHECKING:
    from chaperone import PipelineDataHandler


logger = logging.getLogger(__name__)

# Parameter tier constants for json_schema_extra
TIER_PRIMARY = "primary"  # Directly affects output count, quality, or runtime
TIER_SECONDARY = "secondary"  # Fine-tuning knobs, format options, paths


def get_tier(field: FieldInfo) -> str:
    """Extract the tier from a Pydantic field's json_schema_extra, defaulting to secondary."""
    extra = field.json_schema_extra
    if extra and isinstance(extra, dict):
        return extra.get("tier", TIER_SECONDARY)
    return TIER_SECONDARY


def is_primary(field: FieldInfo) -> bool:
    """Check if a field is marked as primary tier."""
    return get_tier(field) == TIER_PRIMARY


# Names reserved by engine.run() kwargs and RuntimeArgs fields.
# Blocks must not define Pydantic fields with these names, and engine.run(**kwargs)
# will reject them to prevent silent mis-routing.
RESERVED_FIELDS = frozenset(
    {
        # engine.run() explicit params
        "run_args",
        "data",
        "job_args",
        # RuntimeArgs fields
        "mode",
        "recompile",
        "allow_errors",
        "force_exec",
        "skip_cache",
        "skip_chunk_cache",
        "use_local_gpus",
        "max_retries",
    }
)


class BlockFunction(BaseModel, ABC):
    """Base class for blocks with CLI-registered parameters.

    Resource requirements are defined via _resources ClassVar on each block.
    """

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)
    _name: ClassVar[str]
    _display_name: ClassVar[str] = ""
    _description: ClassVar[str] = "N/A"
    _resources: ClassVar[JobArgs | None] = None  # Override per-block with JobArgs

    # Runtime arguments for class
    _requires_input: ClassVar[bool] = True
    """Whether ``engine.run()`` requires explicit ``data=...`` for this block."""
    _accepts_sequence_input: ClassVar[bool] = False
    """Whether raw sequence CSV/parquet input may be loaded for this block."""
    _emits_sequence_table: ClassVar[bool] = False
    """Whether the engine should publish ``sequences.csv`` for this step."""
    _creates_new_step: ClassVar[bool] = True  # False = metric/update block
    _chunked: ClassVar[bool] = False  # Whether block uses chunked execution
    _inline: ClassVar[bool] = True

    name: str = Field(
        default="",
        description="Name of the block function.",
    )
    store: Literal["none", "catcif"] = Field(
        default="none",
        description=(
            "Consolidate outputs after compilation. ``'none'`` disables storage; "
            "``'catcif'`` writes a concatenated-CIF archive."
        ),
    )
    store_cleanup: bool = Field(
        default=False,
        description=(
            "When ``store='catcif'``, delete sandbox output files after they "
            "have been consolidated into the archive. Ignored otherwise."
        ),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        conflicts = RESERVED_FIELDS & set(cls.model_fields)
        if conflicts:
            raise TypeError(
                f"Block {cls.__name__} defines fields that conflict with "
                f"reserved engine/runtime names: {conflicts}. "
                f"Rename these fields or pass them via run_args/job_args."
            )

    def __setstate__(self, state: dict[str, Any]) -> None:
        # After a cross-version unpickle (e.g. a mid-run `git pull` adds a new field
        # with a default to a base class), default-valued fields not present in the
        # pickle's __dict__ stay absent — pydantic's attribute lookup doesn't fall
        # back to field defaults, so the worker raises AttributeError on access.
        # Fill them from the current class's defaults.
        super().__setstate__(state)
        for field_name, field_info in self.__class__.model_fields.items():
            if field_name not in self.__dict__:
                self.__dict__[field_name] = field_info.get_default(
                    call_default_factory=True
                )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to handle callable fields that Pydantic can't serialize.

        Callable fields (e.g. ``selector``, ``function``) are serialized as
        their qualified name string so ``_save_step_config`` doesn't crash.
        This never mutates ``self`` — callables are patched in the output dict.
        """
        # Identify callable fields and exclude them from the default dump
        callable_labels: dict[str, str] = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            if callable(value) and not isinstance(value, type):
                qualname = getattr(value, "__qualname__", None) or getattr(
                    value, "__name__", repr(value)
                )
                module = getattr(value, "__module__", "")
                callable_labels[field_name] = (
                    f"{module}.{qualname}" if module else qualname
                )

        if not callable_labels:
            return super().model_dump(**kwargs)

        exclude = kwargs.pop("exclude", None) or set()
        if isinstance(exclude, dict):
            exclude = set(exclude)
        exclude |= set(callable_labels)

        data = super().model_dump(exclude=exclude, **kwargs)
        data.update(callable_labels)
        return data

    @property
    def _rows_per_job(self) -> int:
        """Get rows_per_job from block's _resources ClassVar.

        GPU resources default to exactly 200 rows per job. CPU resources retain
        the central 1000-row floor. Blocks should set `rows_per_job` explicitly
        only when their algorithm requires a different chunk size (for example,
        one-target design methods or expensive per-row metrics).
        """
        if self._resources is not None and self._resources.rows_per_job is not None:
            return self._resources.rows_per_job
        if self._resources is not None and (
            self._resources.partition == "gpu"
            or self._resources.gpu_class is not None
            or (self._resources.gres or "").startswith("gpu:")
        ):
            return 200
        return 1000

    def get_resources(self) -> JobArgs | None:
        """Get full resource requirements for this block.

        Returns:
            JobArgs with resource requirements, or None if no resources defined.
        """
        return self._resources

    def __init__(self, **data):
        super().__init__(**data)
        # Set name from _name ClassVar if not provided
        if not self.name:
            self.name = self._name

    def __repr__(self):
        """
        Readable format where all instantiated parameters are listed
        Used as a unique fingerprint for the execution
        """
        cls_name = self.__class__.__name__
        arg_str = " | ".join([f"{x}={y}" for x, y in self.model_dump().items()])
        return f"{cls_name}({arg_str})"

    @classmethod
    def from_flags(cls, **default_params):
        from absl import flags

        f = flags.FLAGS
        values = dict(**default_params)  # Set params as the default
        for name, field in cls.model_fields.items():
            flag_name = flag_from_names(cls._name, name)
            if hasattr(f, flag_name):
                flag_obj = f[flag_name]
                flag_value = getattr(f, flag_name)

                if flag_value == flag_obj.default:
                    # If the flag value equals its default, let Pydantic use the model default/default_factory
                    # This allows default_factory to be used when the user doesn't explicitly set the flag
                    logger.debug(
                        f"Skipping {flag_name}, found to be default: {flag_value} (default {flag_obj.default})"
                    )
                    values.pop(name, None)  # if provided by default_params
                else:
                    # User explicitly set this flag, use the flag value
                    values[name] = flag_value
            else:
                logger.debug(f"Skipping {flag_name}, not found in flags")
        return cls(**values)

    def init_handler(self) -> "PipelineDataHandler | None":
        """Create an initial handler for source blocks that generate their own inputs.

        Override in source blocks (e.g. rfd3, rfd2) to create placeholder rows.
        Returns None by default (block expects input from upstream).
        """
        return None

    @abstractmethod
    def __call__(
        self,
        handler: "PipelineDataHandler",
        sandbox_dir: Path,
    ) -> "PipelineDataHandler":
        """Execute the block on the given dataframe.

        Args:
            df: Input dataframe
            sandbox_dir: Directory for this partition's execution
            chunk_id: Partition index for logging (optional)
            recompile: If True, skip command execution and just recompile (optional)
            force_exec: If True, force execution even if cached (optional)

        Returns:
            pd.DataFrame: Output dataframe
        """


def flag_from_names(op_name, field_name):
    return f"{op_name}.{field_name}"
