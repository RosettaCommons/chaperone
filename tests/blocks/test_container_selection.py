"""Public container selection and explicit host model mounts on any cluster."""

import getpass
from types import SimpleNamespace

import pytest

import chaperone
from blocks import get_all_containers, get_container
from blocks.networks import rfdiffusion2
from blocks.networks.prediction import colabfold
from chaperone import containers


APPTAINER_PREFIX = [
    "unshare",
    "--mount",
    "--propagation",
    "private",
    "--fork",
    "--",
    "apptainer",
]
CONTAIN_FLAGS = [
    "--cleanenv",
    "--containall",
    "--no-mount",
    "home,tmp,bind-paths",
    "--writable-tmpfs",
    "--pid",
]


@pytest.fixture
def settings(monkeypatch):
    """Pin the container backend and host probes; return the settings object."""
    values = SimpleNamespace(CHAPERONE_CONTAINER_BACKEND="apptainer")
    monkeypatch.setattr(chaperone, "get_settings", lambda: values)
    monkeypatch.setattr(containers, "_supports_private_mount_namespace", lambda: True)
    # Exact image-selection commands should not depend on the test host's DNS;
    # resolver mounting and policy are covered by test_container_dns.py.
    monkeypatch.setattr(containers, "_RESOLVER_PATHS", ())
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    return values






# ── colabfold ───────────────────────────────────────────────────────────




def test_colabfold_falls_back_to_a_buildable_image_when_absent(monkeypatch):
    picked = colabfold.COLABFOLD_CONTAINER

    assert picked is colabfold.COLABFOLD_PORTABLE_CONTAINER
    assert not picked.is_existing
    assert picked.name == "colabfold_portable"
    assert "colabfold" in picked.to_singularity().lower()






def test_colabfold_portable_binds_only_the_params_dir(monkeypatch, settings, tmp_path):
    data_dir = tmp_path / "af2"

    command = colabfold.ColabFoldBlock(data_dir=str(data_dir)).command_template

    assert command == [
        *APPTAINER_PREFIX,
        "exec",
        *CONTAIN_FLAGS,
        "--bind",
        f"{data_dir}:{data_dir}",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_tester",
        "--nv",
        "${CONTAINER:colabfold_portable}",
        "colabfold_batch",
    ]


def test_colabfold_portable_without_a_params_dir_binds_nothing(monkeypatch, settings):

    command = colabfold.ColabFoldBlock(data_dir=None).command_template

    assert "--bind" not in command


def test_colabfold_portable_creates_the_params_dir_before_binding(
    monkeypatch, settings, tmp_path
):
    data_dir = tmp_path / "af2"
    block = colabfold.ColabFoldBlock(data_dir=str(data_dir))
    handler = SimpleNamespace(get_prediction_inputs=lambda **kwargs: iter(()))

    command = block.create_command(handler, tmp_path / "sandbox", chunk_id=0)

    assert data_dir.is_dir()
    assert command[-2:] == ["--data", str(data_dir)]


def test_colabfold_portable_is_registered_when_selected(monkeypatch):
    portable = colabfold.COLABFOLD_CONTAINER

    assert get_container("colabfold_portable") is portable
    assert get_container("colabfold") is portable
    assert get_all_containers()["colabfold_portable"] is portable


def test_colabfold_portable_installs_from_a_definition(monkeypatch, settings, tmp_path):
    from chaperone_cli import install

    portable = colabfold.COLABFOLD_CONTAINER

    install.build_container(portable, portable.name, tmp_path, dry_run=True)

    # A dry run renders nothing to disk but must not raise or fall through to
    # the "references existing file" branch.
    assert not list(tmp_path.iterdir())


# ── rfdiffusion2 ────────────────────────────────────────────────────────




def test_rfd2_falls_back_to_a_buildable_image_when_absent(monkeypatch):
    picked = rfdiffusion2.RFD2_CONTAINER

    assert picked is rfdiffusion2.RFD2_PORTABLE_CONTAINER
    assert not picked.is_existing
    assert picked.name == "rfdiffusion2_portable"






def test_rfd2_portable_runs_the_baked_in_code_and_checkpoint(monkeypatch, settings):

    block = rfdiffusion2.RFD2Block(inputs="cfg")

    assert block.ckpt_path == "/opt/rfd2-weights/RFD_173.pt"
    # Nothing is bound from the host: code and checkpoint are in the image.
    assert block.command_template == [
        *APPTAINER_PREFIX,
        "run",
        *CONTAIN_FLAGS,
        "--env",
        "PYTHONPATH=/opt/RFdiffusion2",
        "--env",
        "USER=tester",
        "--nv",
        "${CONTAINER:rfdiffusion2_portable}",
        "/opt/RFdiffusion2/rf_diffusion/run_inference.py",
    ]


def test_rfd2_portable_names_no_site_paths(monkeypatch, settings):

    block = rfdiffusion2.RFD2Block(inputs="cfg")

    assert not any("/net/" in part for part in block.command_template)
    assert "/net/" not in block.ckpt_path


def test_rfd2_portable_passes_the_baked_checkpoint(monkeypatch, settings, tmp_path):
    cfg = tmp_path / "design.yaml"
    cfg.write_text("{}\n")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    block = rfdiffusion2.RFD2Block(inputs=str(cfg))

    command = block.create_command(block.init_handler(), sandbox, chunk_id=0)

    assert "inference.ckpt_path=/opt/rfd2-weights/RFD_173.pt" in command


def test_rfd2_portable_binds_an_explicit_checkpoint(monkeypatch, settings, tmp_path):
    ckpt = tmp_path / "other" / "model.pt"

    block = rfdiffusion2.RFD2Block(inputs="cfg", ckpt_path=str(ckpt))

    assert f"{ckpt.parent}:{ckpt.parent}" in block.command_template


def test_rfd2_portable_reports_a_missing_explicit_checkpoint(
    monkeypatch, settings, tmp_path
):
    block = rfdiffusion2.RFD2Block(inputs="cfg", ckpt_path=str(tmp_path / "no.pt"))

    with pytest.raises(FileNotFoundError, match="no.pt"):
        block.create_command(block.init_handler(), tmp_path / "sandbox", chunk_id=0)


def test_rfd2_portable_is_registered_when_selected(monkeypatch):
    portable = rfdiffusion2.RFD2_CONTAINER

    assert get_container("rfdiffusion2_portable") is portable
    assert get_container("rfdiffusion2") is None


def test_rfd2_portable_clones_the_public_repo_at_build_time():
    image = rfdiffusion2.RFD2_PORTABLE_CONTAINER

    urls = [url for url, _, _ in image._git_clones]
    assert urls == ["https://github.com/RosettaCommons/RFdiffusion2"]
    definition = image.to_singularity()
    assert f"git checkout {rfdiffusion2.RFD2_COMMIT}" in definition
    assert "exec python" in definition


def test_rfd2_portable_bakes_a_hash_checked_checkpoint():
    definition = rfdiffusion2.RFD2_PORTABLE_CONTAINER.to_singularity()

    assert (
        "https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_173.pt"
        in definition
    )
    assert (
        "590e126057f780afc1249d29545f0f90635562b1a3df5aff013afcfc39c3d3c3"
        "  /opt/rfd2-weights/RFD_173.pt' | sha256sum -c -"
    ) in definition


def test_default_block_images_never_require_site_files(monkeypatch):
    """Discovery must produce buildable public images even on a site host."""
    from blocks import Blocks
    from chaperone.containers import Image

    def reject_existing_image(*args, **kwargs):
        pytest.fail("Default block selected a host-only prebuilt image")

    monkeypatch.setattr(Image, "from_existing", reject_existing_image)
    for name in Blocks.keys():
        block = Blocks[name]
        image = block._get_image() if hasattr(block, "_get_image") else getattr(block, "_image", None)
        if image is not None:
            assert not image.is_existing, name
            assert image._base_image, name
            config = image._command_config or {}
            binds = config.get("binds", [])
            if callable(binds):
                binds = binds()
            forbidden = {"/net", "/mnt/net", "/projects", "/mnt/projects", "/home", "/mnt/home"}
            assert all(source not in forbidden for source, _ in binds), name


def test_explicit_model_caches_are_mounted_outside_workspace(monkeypatch, settings, tmp_path):
    from blocks.networks.prediction.openfold3 import OpenFold3Block
    from blocks.networks.hallucination.protein_hunter import ProteinHunterBlock
    from blocks.networks.plm.esm2 import ESM2Block
    from blocks.networks.design.boltzgen import BoltzGenBlock

    cache = str(tmp_path / "external-cache")
    settings.HF_HUB_CACHE = cache
    for block in (ESM2Block(), OpenFold3Block(cache_dir=cache)):
        assert f"{cache}:{cache}" in block.command_template
    config = tmp_path / "design.yaml"
    config.write_text("{}\n")
    assert f"{cache}:{cache}" in BoltzGenBlock(config=str(config)).command_template
    hunter = ProteinHunterBlock(ccd_path=cache, boltz_model_path=str(tmp_path / "weights" / "model.ckpt"))
    assert f"{cache}:{cache}" in hunter.command_template
    assert f"{tmp_path / 'weights'}:{tmp_path / 'weights'}" in hunter.command_template


@pytest.mark.parametrize("name", ["esm2", "boltzgen", "openfold3"])
def test_new_model_cache_is_created_when_staging_job(name, settings, tmp_path):
    from blocks import Blocks

    cache = tmp_path / "external-cache" / name
    settings.HF_HUB_CACHE = str(cache)
    options = {}
    if name == "openfold3":
        options["cache_dir"] = str(cache)
    if name == "boltzgen":
        config = tmp_path / "design.yaml"
        config.write_text("{}\n")
        options["config"] = str(config)
    block = Blocks[name](**options)
    assert f"{cache}:{cache}" in block.command_template
    assert not cache.exists()
    handler = SimpleNamespace(
        get_data=lambda **kwargs: iter(()),
        get_prediction_inputs=lambda **kwargs: iter(()),
    )
    command = block.create_command(handler, tmp_path / "job", 0)
    assert cache.is_dir()
    assert f"{cache}:{cache}" in command
