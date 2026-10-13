"""Every container definition in ``src/blocks`` renders to a buildable recipe.

The tests discover each module-level ``Image`` the same way the block registry
finds blocks (a walk over ``src/blocks``) and check that the definition can be
rendered without apptainer, docker, a network or a GPU. ``Image.from_existing``
points at a pre-built file and has no recipe; those are the only images allowed
to lack one, and each needs a documented reason below.
"""

import importlib
import re
from pathlib import Path

import pytest

import blocks
from chaperone.containers import Image

BLOCKS_DIR = Path(blocks.__file__).parent

# Existing-file images that deliberately have no recipe. ``colabfold`` and
# ``rfdiffusion2`` are on-prem SIF variants with a buildable ``*_portable``
# counterpart; ``modelhub`` (classic ProteinMPNN) has none and needs a site SIF.
EXISTING_ALLOWLIST = {
    "modelhub": "on-prem SIF only; no buildable counterpart (use public LigandMPNN)",
    "colabfold": "on-prem SIF variant; buildable counterpart is colabfold_portable",
    "rfdiffusion2": "on-prem SIF variant; buildable counterpart is rfdiffusion2_portable",
}
PORTABLE_COUNTERPARTS = {
    "colabfold": "colabfold_portable",
    "rfdiffusion2": "rfdiffusion2_portable",
}

# Images the audit asked to be covered explicitly, plus the buildable variants.
EXPECTED_BUILDABLE = {
    "alphafold3",
    "bindcraft",
    "boltz2",
    "boltzgen",
    "chai1",
    "colabfold_portable",
    "disco",
    "e1",
    "esm2",
    "esmfold2",
    "foundry",
    "foundry_public",
    "metal3d",
    "openfold3",
    "protein_hunter",
    "protenix",
    "rfdiffusion1",
    "rfdiffusion2_portable",
    "simplefold",
    "tmol",
}

# Paths that exist only on particular hosts. Binds in ``template_command`` may
# name them; build steps must not.
HOST_ONLY_PATH = re.compile(
    r"(?<![\w.-])(?:/net|/projects|/mnt/home|/home/[A-Za-z0-9_.-]+)(?:/|\b)"
)


def _discover_images() -> dict[str, Image]:
    """Return every distinct module-level ``Image`` under ``src/blocks``."""
    found: dict[int, tuple[str, Image]] = {}
    for path in sorted(BLOCKS_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        if "Image" not in path.read_text():
            continue
        parts = path.relative_to(BLOCKS_DIR).with_suffix("").parts
        module = importlib.import_module(".".join(("blocks", *parts)))
        for value in vars(module).values():
            if isinstance(value, Image):
                found.setdefault(id(value), (value.name, value))
    images: dict[str, Image] = {}
    for name, image in found.values():
        if name in images:
            raise AssertionError(f"two different module-level images named {name!r}")
        images[name] = image
    return images


IMAGES = _discover_images()
BUILDABLE = {name: image for name, image in IMAGES.items() if not image.is_existing}
EXISTING = {name: image for name, image in IMAGES.items() if image.is_existing}


def _build_step_text(image: Image) -> list[str]:
    """Every string that ends up in the build, excluding ``%files`` sources."""
    texts = [image._base_image or ""]
    texts += [part for group in image._commands for part in group]
    texts += list(image._apt_packages)
    texts += [f"{key}={value}" for key, value in image._env_vars.items()]
    texts += [f"{key}={value}" for key, value in image._labels.items()]
    texts += [dest for _, dest in image._copies]
    texts += [part for url, dest, branch in image._git_clones for part in (url, dest)]
    return texts


def test_expected_images_are_discovered():
    assert EXPECTED_BUILDABLE <= set(BUILDABLE)


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_definition_renders(name):
    image = BUILDABLE[name]
    assert image._base_image, f"{name} has no base image"

    singularity = image.to_singularity()
    assert f"From: {image._base_image}" in singularity
    assert "%post" in singularity or "%files" in singularity

    dockerfile = image.to_dockerfile()
    assert f"FROM {image._base_image}" in dockerfile


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_build_steps_have_no_host_only_paths(name):
    image = BUILDABLE[name]
    offenders = [t for t in _build_step_text(image) if HOST_ONLY_PATH.search(t)]
    assert not offenders, f"{name} build steps name host-only paths: {offenders}"

    # The rendered text carries the same steps; git clones stay as RUN lines
    # until the installer resolves them, so no host checkout path appears.
    rendered = image.to_singularity()
    body = "\n".join(
        line for line in rendered.splitlines() if not line.startswith("    /")
    )
    assert not HOST_ONLY_PATH.search(body)


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_copied_sources_exist_in_the_checkout(name):
    for src, _ in BUILDABLE[name]._copies:
        assert Path(src).is_file(), f"{name} copies a missing file: {src}"


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_template_command_builds(name):
    command = BUILDABLE[name].build_command("true")
    assert command[-1] == "true"
    assert "${CONTAINER:%s}" % name in command


def test_only_allowlisted_images_lack_a_recipe():
    assert set(EXISTING) <= set(EXISTING_ALLOWLIST)
    for name in EXISTING:
        assert EXISTING_ALLOWLIST[name]


@pytest.mark.parametrize("name", sorted(EXISTING))
def test_existing_images_cannot_render(name):
    with pytest.raises(ValueError):
        EXISTING[name].to_singularity()
    with pytest.raises(ValueError):
        EXISTING[name].to_dockerfile()


@pytest.mark.parametrize("onprem,portable", sorted(PORTABLE_COUNTERPARTS.items()))
def test_each_onprem_image_has_a_buildable_counterpart(onprem, portable):
    assert onprem in EXISTING
    assert portable in BUILDABLE
