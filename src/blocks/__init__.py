"""
Lazy-loading block registry.

Stage 1: AST scan identifies block names without importing modules (eager, at import time)
Stage 2: Blocks are imported on-demand when accessed via get() or __getitem__
"""

import ast
import functools
import importlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaperone import BlockFunction
    from chaperone.containers import Image


class LazyBlockRegistry:
    """Dict-like registry that lazily loads blocks on access.

    AST scan runs at construction (eager), module imports are deferred (lazy).
    """

    def __init__(self):
        self._block_locations: dict[
            str, tuple[str, str]
        ] = {}  # name -> (module, class)
        self._scanned = False

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            self._scan_all_blocks()
            self._scanned = True

    def _scan_all_blocks(self) -> None:
        """AST-scan block files to find names without importing."""
        blocks_dir = Path(__file__).parent

        for filepath in blocks_dir.rglob("*.py"):
            if filepath.name == "__init__.py":
                continue
            rel_parts = filepath.relative_to(blocks_dir).parts
            if any(skip in rel_parts for skip in ("utils",)):
                continue

            # Convert path to module name
            module_parts = ["blocks"] + list(rel_parts)
            module_parts[-1] = module_parts[-1].replace(".py", "")
            module_path = ".".join(module_parts)

            for block_name, class_name in _extract_block_names(filepath):
                if block_name in self._block_locations:
                    existing_module, _ = self._block_locations[block_name]
                    raise ValueError(
                        f"Duplicate block '{block_name}' found in modules "
                        f"'{existing_module}' and '{module_path}'"
                    )
                self._block_locations[block_name] = (module_path, class_name)

    @functools.lru_cache(maxsize=None)
    def _load_block(self, name: str) -> type["BlockFunction"]:
        """Import and cache a single block (thread-safe via lru_cache)."""
        self._ensure_scanned()
        if name not in self._block_locations:
            raise KeyError(f"Block '{name}' not found. Available: {list(self.keys())}")

        module_path, class_name = self._block_locations[name]
        module = importlib.import_module(module_path)
        block_cls = getattr(module, class_name)

        # Validate that the loaded object is a BlockFunction subclass before registering.
        from chaperone import BlockFunction

        if not isinstance(block_cls, type) or not issubclass(block_cls, BlockFunction):
            raise TypeError(
                f"Loaded object '{module_path}.{class_name}' is not a BlockFunction subclass"
            )
        return block_cls

    # Dict interface
    def __getitem__(self, name: str) -> type["BlockFunction"]:
        return self._load_block(name)

    def get(self, name: str, default=None):
        try:
            return self._load_block(name)
        except KeyError:
            return default

    def getall(self, *names: str) -> tuple[type["BlockFunction"], ...]:
        """Load multiple blocks by name, returning a tuple of classes.

        Usage:
            RFD3, MPNN, AF3 = Blocks.getall("rfd3", "mpnn", "af3")
        """
        return tuple(self._load_block(name) for name in names)

    def __contains__(self, name: str) -> bool:
        self._ensure_scanned()
        return name in self._block_locations

    def keys(self):
        self._ensure_scanned()
        return self._block_locations.keys()

    def items(self):
        for name in self._block_locations:
            yield name, self._load_block(name)

    def values(self):
        for _, cls in self.items():
            yield cls

    def __iter__(self):
        self._ensure_scanned()
        return iter(self._block_locations)

    def __len__(self) -> int:
        self._ensure_scanned()
        return len(self._block_locations)


def _extract_block_names(filepath: Path) -> list[tuple[str, str]]:
    """AST-parse file to find _name = "value" in class definitions."""
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError as e:
        warnings.warn(f"Syntax error parsing {filepath}: {e}")
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                # Handle: _name = "value"
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "_name":
                            if isinstance(item.value, ast.Constant) and isinstance(
                                item.value.value, str
                            ):
                                results.append((item.value.value, node.name))
                # Handle: _name: str = "value" or _name: ClassVar[str] = "value"
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name) and item.target.id == "_name":
                        if isinstance(item.value, ast.Constant) and isinstance(
                            item.value.value, str
                        ):
                            results.append((item.value.value, node.name))
    return results


@functools.lru_cache(maxsize=1)
def discover_all_blocks() -> dict[str, type["BlockFunction"]]:
    """Eagerly load ALL blocks (backwards compat). Triggers full import."""
    return dict(Blocks.items())


# Build recipes remain discoverable even when a site selects a prebuilt SIF.
_CONTAINER_VARIANTS = {
    "rfdiffusion2_portable": ("blocks.networks.rfdiffusion2", "RFD2_PORTABLE_CONTAINER"),
    "colabfold_portable": ("blocks.networks.prediction.colabfold", "COLABFOLD_PORTABLE_CONTAINER"),
    "foundry": ("blocks.networks.foundry.container", "_FOUNDRY_CUDA_CONTAINER"),
    "foundry_public": ("blocks.networks.foundry.container", "FOUNDRY_PUBLIC_CONTAINER"),
}


def get_all_containers() -> dict[str, "Image"]:
    """Get all containers by collecting _image ClassVars from discovered blocks.

    Each block module defines its container Image at module level and
    references it via the _image ClassVar on the block class. Imports blocks
    one at a time so a single broken block doesn't prevent discovering the rest.
    """
    containers = {}
    for name in Blocks.keys():
        try:
            block_cls = Blocks[name]
        except Exception as e:
            warnings.warn(f"Failed to import block '{name}': {e}")
            continue
        _get = getattr(block_cls, "_get_image", None)
        image = _get() if _get else getattr(block_cls, "_image", None)
        if image is not None and image.name not in containers:
            containers[image.name] = image
    for name, (module, attribute) in _CONTAINER_VARIANTS.items():
        if name not in containers:
            containers[name] = getattr(importlib.import_module(module), attribute)
    return containers


def get_container(container_name: str) -> "Image | None":
    """Get a single container by name without importing all blocks.

    Imports blocks one at a time, stopping as soon as the requested container
    is found. Much faster than get_all_containers() for targeted lookups.
    """
    if container_name in _CONTAINER_VARIANTS:
        module, attribute = _CONTAINER_VARIANTS[container_name]
        return getattr(importlib.import_module(module), attribute)
    if container_name in Blocks.keys():
        block = Blocks[container_name]
        get_image = getattr(block, "_get_image", None)
        image = get_image() if get_image else getattr(block, "_image", None)
        if image is not None:
            return image
    for name in Blocks.keys():
        try:
            block_cls = Blocks[name]
        except Exception:
            continue
        _get = getattr(block_cls, "_get_image", None)
        image = _get() if _get else getattr(block_cls, "_image", None)
        if image is not None and image.name == container_name:
            return image
    return None


Blocks = LazyBlockRegistry()
__all__ = ["Blocks", "discover_all_blocks", "get_all_containers", "get_container"]
