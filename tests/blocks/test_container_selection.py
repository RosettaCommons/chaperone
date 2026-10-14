"""On-prem vs portable container selection for colabfold and rfdiffusion2.

Site images remain available alongside portable recipes. Existence checks
are monkeypatched so these tests behave the same on a site host and a fresh clone.
"""

import getpass
from types import SimpleNamespace

import pytest

import chaperone
from blocks import get_all_containers, get_container
from blocks.networks import rfdiffusion2
from blocks.networks.prediction import colabfold
from chaperone import containers

COLABFOLD_SIF = "/net/software/containers/colabfold.sif"
RFD2_SIF = (
    "/net/software/containers/versions/rf_diffusion_aa/24-05-21/rf_diffusion_aa.sif"
)
RFD2_CODE = "/net/tukwila/rohith/RFdiffusion2"
RFD2_CKPT = "/net/databases/lab/aa-diffusion/models/RFD_140_ORI_tipatom.pt"

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
    "--nv",
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


def _select_colabfold(monkeypatch, onprem: bool):
    monkeypatch.setattr(colabfold, "_onprem_colabfold_sif_exists", lambda: onprem)
    picked = colabfold._pick_colabfold_container()
    monkeypatch.setattr(colabfold, "COLABFOLD_CONTAINER", picked)
    return picked


def _select_rfd2(monkeypatch, onprem: bool):
    monkeypatch.setattr(rfdiffusion2, "_onprem_rfd2_sif_exists", lambda: onprem)
    picked = rfdiffusion2._pick_rfd2_container()
    monkeypatch.setattr(rfdiffusion2, "RFD2_CONTAINER", picked)
    return picked


# ── colabfold ───────────────────────────────────────────────────────────


def test_colabfold_prefers_the_onprem_sif_when_present(monkeypatch):
    picked = _select_colabfold(monkeypatch, onprem=True)

    assert picked is colabfold._COLABFOLD_SIF_CONTAINER
    assert picked.is_existing
    assert picked.name == "colabfold"


def test_colabfold_falls_back_to_a_buildable_image_when_absent(monkeypatch):
    picked = _select_colabfold(monkeypatch, onprem=False)

    assert picked is colabfold.COLABFOLD_PORTABLE_CONTAINER
    assert not picked.is_existing
    assert picked.name == "colabfold_portable"
    assert "colabfold" in picked.to_singularity().lower()


def test_colabfold_onprem_command_is_unchanged(monkeypatch, settings):
    _select_colabfold(monkeypatch, onprem=True)

    command = colabfold.ColabFoldBlock().command_template

    assert command == [
        *APPTAINER_PREFIX,
        "exec",
        *CONTAIN_FLAGS,
        "--bind",
        "/net:/net",
        "--bind",
        "/mnt/net:/mnt/net",
        "--bind",
        "/projects:/projects",
        "--bind",
        "/mnt/home:/mnt/home",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_tester",
        COLABFOLD_SIF,
        "colabfold_batch",
    ]


def test_colabfold_onprem_does_not_bind_the_data_dir(monkeypatch, settings, tmp_path):
    _select_colabfold(monkeypatch, onprem=True)

    command = colabfold.ColabFoldBlock(data_dir=str(tmp_path)).command_template

    assert f"{tmp_path}:{tmp_path}" not in command


def test_colabfold_portable_binds_only_the_params_dir(monkeypatch, settings, tmp_path):
    _select_colabfold(monkeypatch, onprem=False)
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
        "${CONTAINER:colabfold_portable}",
        "colabfold_batch",
    ]


def test_colabfold_portable_without_a_params_dir_binds_nothing(monkeypatch, settings):
    _select_colabfold(monkeypatch, onprem=False)

    command = colabfold.ColabFoldBlock(data_dir=None).command_template

    assert "--bind" not in command


def test_colabfold_portable_creates_the_params_dir_before_binding(
    monkeypatch, settings, tmp_path
):
    _select_colabfold(monkeypatch, onprem=False)
    data_dir = tmp_path / "af2"
    block = colabfold.ColabFoldBlock(data_dir=str(data_dir))
    handler = SimpleNamespace(get_prediction_inputs=lambda **kwargs: iter(()))

    command = block.create_command(handler, tmp_path / "sandbox", chunk_id=0)

    assert data_dir.is_dir()
    assert command[-2:] == ["--data", str(data_dir)]


def test_colabfold_portable_is_registered_when_selected(monkeypatch):
    portable = _select_colabfold(monkeypatch, onprem=False)

    assert get_container("colabfold_portable") is portable
    assert get_container("colabfold") is portable
    assert get_all_containers()["colabfold_portable"] is portable


def test_colabfold_portable_installs_from_a_definition(monkeypatch, settings, tmp_path):
    from chaperone_cli import install

    portable = _select_colabfold(monkeypatch, onprem=False)

    install.build_container(portable, portable.name, tmp_path, dry_run=True)

    # A dry run renders nothing to disk but must not raise or fall through to
    # the "references existing file" branch.
    assert not list(tmp_path.iterdir())


# ── rfdiffusion2 ────────────────────────────────────────────────────────


def test_rfd2_prefers_the_onprem_sif_when_present(monkeypatch):
    picked = _select_rfd2(monkeypatch, onprem=True)

    assert picked is rfdiffusion2._RFD2_SIF_CONTAINER
    assert picked.is_existing
    assert picked.name == "rfdiffusion2"


def test_rfd2_falls_back_to_a_buildable_image_when_absent(monkeypatch):
    picked = _select_rfd2(monkeypatch, onprem=False)

    assert picked is rfdiffusion2.RFD2_PORTABLE_CONTAINER
    assert not picked.is_existing
    assert picked.name == "rfdiffusion2_portable"


def test_rfd2_onprem_command_and_checkpoint_are_unchanged(monkeypatch, settings):
    _select_rfd2(monkeypatch, onprem=True)

    block = rfdiffusion2.RFD2Block(inputs="cfg")

    assert block.command_template == [
        *APPTAINER_PREFIX,
        "run",
        *CONTAIN_FLAGS,
        "--bind",
        f"{RFD2_CODE}:{RFD2_CODE}",
        "--bind",
        "/net/databases/lab/aa-diffusion:/net/databases/lab/aa-diffusion",
        "--env",
        f"PYTHONPATH={RFD2_CODE}/",
        RFD2_SIF,
        f"{RFD2_CODE}/rf_diffusion/run_inference.py",
    ]
    assert block.ckpt_path == RFD2_CKPT


def test_rfd2_onprem_full_command_is_unchanged(monkeypatch, settings, tmp_path):
    _select_rfd2(monkeypatch, onprem=True)
    cfg = tmp_path / "design.yaml"
    cfg.write_text("{}\n")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    block = rfdiffusion2.RFD2Block(inputs=str(cfg))

    command = block.create_command(block.init_handler(), sandbox, chunk_id=0)

    assert command[: len(block.command_template)] == block.command_template
    assert command[len(block.command_template) :] == [
        f"--config-dir={cfg.parent}",
        "--config-name=design",
        f"inference.output_prefix={sandbox}/design_chunk0",
        "inference.num_designs=1",
        f"inference.ckpt_path={RFD2_CKPT}",
        "inference.seed_offset=0",
    ]


def test_rfd2_portable_runs_the_baked_in_code_and_checkpoint(monkeypatch, settings):
    _select_rfd2(monkeypatch, onprem=False)

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
        "${CONTAINER:rfdiffusion2_portable}",
        "/opt/RFdiffusion2/rf_diffusion/run_inference.py",
    ]


def test_rfd2_portable_names_no_site_paths(monkeypatch, settings):
    _select_rfd2(monkeypatch, onprem=False)

    block = rfdiffusion2.RFD2Block(inputs="cfg")

    assert not any("/net/" in part for part in block.command_template)
    assert "/net/" not in block.ckpt_path


def test_rfd2_portable_passes_the_baked_checkpoint(monkeypatch, settings, tmp_path):
    _select_rfd2(monkeypatch, onprem=False)
    cfg = tmp_path / "design.yaml"
    cfg.write_text("{}\n")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    block = rfdiffusion2.RFD2Block(inputs=str(cfg))

    command = block.create_command(block.init_handler(), sandbox, chunk_id=0)

    assert "inference.ckpt_path=/opt/rfd2-weights/RFD_173.pt" in command


def test_rfd2_portable_binds_an_explicit_checkpoint(monkeypatch, settings, tmp_path):
    _select_rfd2(monkeypatch, onprem=False)
    ckpt = tmp_path / "other" / "model.pt"

    block = rfdiffusion2.RFD2Block(inputs="cfg", ckpt_path=str(ckpt))

    assert f"{ckpt.parent}:{ckpt.parent}" in block.command_template


def test_rfd2_portable_reports_a_missing_explicit_checkpoint(
    monkeypatch, settings, tmp_path
):
    _select_rfd2(monkeypatch, onprem=False)
    block = rfdiffusion2.RFD2Block(inputs="cfg", ckpt_path=str(tmp_path / "no.pt"))

    with pytest.raises(FileNotFoundError, match="no.pt"):
        block.create_command(block.init_handler(), tmp_path / "sandbox", chunk_id=0)


def test_rfd2_portable_is_registered_when_selected(monkeypatch):
    portable = _select_rfd2(monkeypatch, onprem=False)

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
