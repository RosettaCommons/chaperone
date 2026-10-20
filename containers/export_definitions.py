#!/usr/bin/env python3
"""Export relocatable Apptainer build contexts without fetching or building images.

Run with the Chaperone environment: python containers/export_definitions.py.
The checked-in outputs deliberately use the installer's host-side git cloning:
Apptainer applies %files before %post, so cloning into a copied directory during
%post would fail for recipes such as BindCraft and SimpleFold.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shlex
import sys
import warnings
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "apptainer"


def discover_images():
    """Include registered runtime images and every declared recipe variant."""
    import blocks
    from chaperone.containers import Image

    images = {}

    def add(image):
        previous = images.get(image.name)
        # Some accessors return equivalent copies instead of the declaration.
        if previous is not None and previous is not image and previous != image:
            raise ValueError(f"Conflicting recipes named {image.name!r}")
        images[image.name] = image

    # The registry warns and continues after a broken block import. Export must
    # fail loudly instead of publishing an apparently complete partial catalog.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Failed to import block.*")
        for image in blocks.get_all_containers().values():
            add(image)
    root = Path(blocks.__file__).parent
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py" or "Image" not in path.read_text():
            continue
        module = importlib.import_module(
            ".".join(("blocks", *path.relative_to(root).with_suffix("").parts))
        )
        for value in vars(module).values():
            if isinstance(value, Image):
                add(value)
    return dict(sorted(images.items()))


def render_bundle(images=None):
    """Return deterministic relative filename -> bytes, with no external I/O."""
    from blocks import Blocks

    images = discover_images() if images is None else images
    block_names = {}
    for name in sorted(Blocks.keys()):
        block = Blocks[name]  # Never suppress unavailable/broken registrations.
        getter = getattr(block, "_get_image", None)
        image = getter() if getter else getattr(block, "_image", None)
        if image is not None:
            block_names.setdefault(image.name, []).append(name)
    files = {}
    recipes = []
    existing = []
    for name, image in sorted(images.items()):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Unsafe image name: {name!r}")
        if image.is_existing:
            existing.append(
                {
                    "name": name,
                    "blocks": block_names.get(name, []),
                    "reason": "References an existing image; no source recipe is defined.",
                }
            )
            continue
        prepared = image._clone()
        copies = []
        source_info = []
        commands = [
            "#!/bin/sh",
            "set -eu",
            'cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"',
        ]
        for index, (url, destination, branch) in enumerate(image._git_clones):
            repo = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
                raise ValueError(f"Unsafe repository basename: {repo!r}")
            relative = f"sources/{name}/repos/{index}/{repo}"
            quote = shlex.quote
            clone = ["git", "clone"]
            if branch:
                clone += ["--branch", branch]
            clone += ["--", url, relative]
            commands += [
                f"mkdir -p {quote(str(Path(relative).parent))}",
                # Refuse reuse rather than silently accepting a partial
                # clone or another revision after a failed preparation.
                f"if [ -e {quote(relative)} ]; then",
                f"    echo {quote('Source already exists: ' + relative + '; remove or move it before preparing again.')} >&2",
                "    exit 1",
                "fi",
                shlex.join(clone),
            ]
            copies.append((relative, destination))
            source_info.append(
                {"kind": "git", "path": relative, "url": url, "branch": branch}
            )
        for index, (source, destination) in enumerate(image._copies):
            source = Path(source).expanduser().resolve(strict=True)
            relative = f"sources/{name}/files/{index}/{source.name}"
            paths = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in paths:
                if path.is_file():
                    target = (
                        relative
                        if source.is_file()
                        else f"{relative}/{path.relative_to(source).as_posix()}"
                    )
                    files[target] = path.read_bytes()
            copies.append((relative, destination))
            source_info.append({"kind": "bundled", "path": relative})
        prepared._git_clones = []
        prepared._copies = copies
        definition = f"{name}.def"
        files[definition] = (prepared.to_singularity().rstrip() + "\n").encode()
        prepare_script = f"{name}.prepare.sh" if image._git_clones else None
        if prepare_script:
            files[prepare_script] = ("\n".join(commands) + "\n").encode()
        recipes.append(
            {
                "name": name,
                "definition": definition,
                "prepare_script": prepare_script,
                "blocks": block_names.get(name, []),
                "build_access": image._labels.get("BuildAccess"),
                "sources": source_info,
            }
        )
    catalog = {
        "schema_version": 1,
        "recipes": recipes,
        "existing_images": existing,
        "generated_files": sorted([*files, "catalog.json"]),
    }
    files["catalog.json"] = (
        json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    ).encode()
    return files


def export(output: Path, *, check=False):
    files = render_bundle()
    previous = output / "catalog.json"
    prior_files = set()
    if previous.exists():
        prior_files.update(json.loads(previous.read_text()).get("generated_files", []))
    # Include definitions unknown to the old manifest (e.g. accidentally added).
    if output.exists():
        prior_files.update(
            p.relative_to(output).as_posix() for p in output.glob("*.def")
        )
        prior_files.update(
            p.relative_to(output).as_posix() for p in output.glob("*.prepare.sh")
        )
        prior_files.update(
            p.relative_to(output).as_posix()
            for p in output.glob("sources/*/files/**/*")
            if p.is_file()
        )
    stale = sorted(prior_files - files.keys())
    different = sorted(
        name
        for name, content in files.items()
        if not (output / name).is_file() or (output / name).read_bytes() != content
    )
    if check:
        if stale or different:
            for name in different:
                print(f"Missing or outdated: {name}", file=sys.stderr)
            for name in stale:
                print(f"Unexpected generated file: {name}", file=sys.stderr)
            return 1
        print(f"Apptainer definitions are current ({len(files)} files).")
        return 0
    for name in stale:
        path = output / name
        # Only delete files previously recorded inside this managed output.
        if path.resolve().is_relative_to(output.resolve()) and path.is_file():
            path.unlink()
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if name.endswith(".prepare.sh"):
            path.chmod(0o755)
    print(
        f"Exported {sum(name.endswith('.def') for name in files)} definitions to {output}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="Fail if checked-in exports differ"
    )
    args = parser.parse_args(argv)
    return export(args.output, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
