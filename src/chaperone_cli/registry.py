"""
Block discovery + flag registration.
This file must ONLY run after the chaperone.blocks package is imported.
"""

from absl import flags
from pydantic_core._pydantic_core import PydanticUndefinedType
from chaperone.block_config import flag_from_names
from blocks import Blocks
from typing import get_args, get_origin, Union
import types

ALLOWED_FLAGTYPES = {int, float, bool, str, list}
SKIPPED_FLAGTYPES = {
    dict,
    set,
    tuple,
}  # Types we can't register as flags via absl cleanly


_flags_registered = False


def register_flags(block_classes=None):
    """Register flags for selected classes, or discover all blocks by default."""
    global _flags_registered
    if _flags_registered:
        return ()
    _flags_registered = True
    registered = (
        tuple((op._name, op) for op in block_classes)
        if block_classes is not None
        else tuple((name, Blocks[name]) for name in Blocks.keys())
    )
    for op_name, op in registered:
        flags.DEFINE_boolean(
            f"{op_name}.force_exec",
            default=False,
            help=f"Force execution of {op_name} block even if cached results exist",
        )
        flags.DEFINE_boolean(
            f"{op_name}.recompile",
            default=False,
            help=f"Recompile outputs for {op_name} block from existing sandbox directories",
        )

        for field_name, model_field in op.model_fields.items():
            flag_name = flag_from_names(op_name, field_name)
            field_type = _extract_field_type(model_field)
            if field_type is None:
                continue
            default = _get_default(model_field)
            description = model_field.description or ""

            _define_flag(field_type, flag_name, default, description)
    return registered


def _extract_field_type(model_field) -> type | None:
    """
    Extract the actual type from a Pydantic field, unwrapping Optional/Union.
    Returns None if the field should be skipped (contains only skipped types).
    """
    # Skip fields marked as exclude=True (non-serializable fields like Callable)
    if model_field.exclude:
        return None
    if model_field.json_schema_extra and "cli_type" in model_field.json_schema_extra:
        return model_field.json_schema_extra["cli_type"]
    field_type = model_field.annotation

    # Handle Optional[T] / Union[T, ...]
    # Support both typing.Union (Union[str, dict]) and types.UnionType (str | dict)
    origin = get_origin(field_type)
    is_union = origin is Union or isinstance(field_type, types.UnionType)

    if is_union:
        args = get_args(field_type)
        non_none_types = [arg for arg in args if arg is not type(None)]
        # Filter out skipped types (check both raw type and generic origin)
        allowed_types = [
            t
            for t in non_none_types
            if t not in SKIPPED_FLAGTYPES and get_origin(t) not in SKIPPED_FLAGTYPES
        ]
        # If we have str | dict, prefer str (first allowed type)
        if not allowed_types:
            return None  # Only skipped types, ignore this field
        field_type = allowed_types[0]

    # If the field itself is a skipped type, ignore it
    if field_type in SKIPPED_FLAGTYPES or get_origin(field_type) in SKIPPED_FLAGTYPES:
        return None

    # Skip nested Pydantic models (e.g. MPNNArgs) — not representable as flat CLI flags
    try:
        from pydantic import BaseModel

        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return None
    except ImportError:
        pass

    if get_origin(field_type) is list:
        field_type = list

    if field_type not in ALLOWED_FLAGTYPES:
        # Fallback: infer from default value
        if not isinstance(model_field.default, PydanticUndefinedType):
            inferred_type = type(model_field.default)
            if inferred_type not in SKIPPED_FLAGTYPES:
                field_type = inferred_type

        if field_type not in ALLOWED_FLAGTYPES:
            raise TypeError(
                f"Unsupported field type {model_field.annotation!r} "
                f"in {model_field} (resolved to {field_type})"
            )

    return field_type


def _get_default(model_field):
    """Extract default value, returning None for undefined fields."""
    return (
        None
        if isinstance(model_field.default, PydanticUndefinedType)
        else model_field.default
    )


def _define_flag(field_type: type, flag_name: str, default, description: str) -> None:
    """Define an ABSL flag based on field type."""
    flag_definitions = {
        int: lambda: flags.DEFINE_integer(flag_name, default, description),
        float: lambda: flags.DEFINE_float(flag_name, default, description),
        bool: lambda: flags.DEFINE_boolean(flag_name, default, description),
        str: lambda: flags.DEFINE_string(flag_name, default, description),
        list: lambda: flags.DEFINE_list(flag_name, default or [], description),
    }

    definer = flag_definitions.get(field_type)
    if not definer:
        raise TypeError(f"Unsupported field type {field_type!r} for flag {flag_name}")

    definer()


def print_registered_blocks(blocks=Blocks.items()):
    """
    Pretty-print only the CLI flags belonging to discovered blocks.
    Each block namespace gets its own color. Primary-tier params are marked with ★.
    """
    from chaperone.block_config.base import get_tier, TIER_PRIMARY

    FLAGS = flags.FLAGS
    COLORS = [
        "\033[95m",  # magenta
        "\033[94m",  # blue
        "\033[96m",  # cyan
        "\033[92m",  # green
        "\033[93m",  # yellow
        "\033[91m",  # red
    ]
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{BOLD}=== Registered block Flags ==={RESET}\n")

    for idx, (op_name, op_cls) in enumerate(blocks):
        color = COLORS[idx % len(COLORS)]
        desc = getattr(op_cls, "_description", "")
        print(f"{color}{BOLD}{op_name}{RESET}{color} — {desc}{RESET}")
        for field_name, model_field in op_cls.model_fields.items():
            if field_name == "name":
                continue
            flag_name = f"{op_name}.{field_name}"
            tier = get_tier(model_field)
            tier_marker = f" {BOLD}★{RESET}" if tier == TIER_PRIMARY else ""
            if flag_name not in FLAGS:
                print(f"  {color}--{flag_name}{tier_marker}{RESET}")
                print("      (not available as CLI flag)")
                continue
            flag_obj = FLAGS[flag_name]
            default = flag_obj.default
            help_text = flag_obj.help
            print(f"  {color}--{flag_name}{tier_marker}{RESET}")
            print(f"      default = {default}")
            if help_text and help_text != "(no help available)":
                print(f"      help    = {help_text}")
        print()

    print(f"{BOLD}==================================={RESET}")
    print("  ★ = primary parameter (affects output count, quality, or runtime)\n")


__all__ = ["register_flags", "print_registered_blocks"]
